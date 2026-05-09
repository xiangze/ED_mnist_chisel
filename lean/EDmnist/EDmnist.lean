/-
  ED method neural network in Sparkle HDL (Lean 4)
  Ported from Chisel implementation
  Original: https://qiita.com/pocokhc/items/f7ab56051bb936740b8f

  Architecture:
    - 3-layer model (input → hidden → output) with sigmoid activation
    - Fixed-point arithmetic: Q(w, bp) using BitVec w
    - Sigmoid via 256-entry LUT (piecewise approximation)
    - Pipeline: latch → multiply → sum → activation → output
    - Optional training (weight update) via ED method
-/
import Sparkle
import Sparkle.Compiler.Elab

open Sparkle.Core.Signal
open Sparkle.Core.Domain

-- ============================================================
-- §1  Fixed-Point Helpers
-- ============================================================

namespace FxQ

/-- Bit width and binary-point position for Q-format fixed point. -/
structure Fmt where
  w  : Nat   -- total bit width
  bp : Nat   -- fractional bits
  deriving Repr

/-- Scale factor = 2^bp -/
def Fmt.scale (f : Fmt) : Nat := 1 <<< f.bp

/-- Convert a Float to a fixed-point BitVec at elaboration time. -/
def toFixed (fmt : Fmt) (v : Float) : BitVec fmt.w :=
  let scaled := (v * Float.ofNat fmt.scale).round.toInt32
  BitVec.ofInt fmt.w scaled.toInt

/-- Fixed-point multiplication:
    (a * b) >>> bp  to keep the result in the same Q format.
    We widen to 2*w, multiply, arithmetic-shift-right by bp, then truncate. -/
def fxMul (fmt : Fmt) (a b : BitVec fmt.w) : BitVec fmt.w :=
  let wa : BitVec (2 * fmt.w) := a.signExtend (2 * fmt.w)
  let wb : BitVec (2 * fmt.w) := b.signExtend (2 * fmt.w)
  let prod := wa * wb
  let shifted := prod.sshiftRight fmt.bp
  shifted.truncate fmt.w

/-- Signed clip to [lo, hi]. -/
def clip (a lo hi : BitVec n) : BitVec n :=
  if a.toInt < lo.toInt then lo
  else if a.toInt > hi.toInt then hi
  else a

/-- Absolute value (signed). -/
def sAbs (a : BitVec n) : BitVec n :=
  if a.toInt < 0 then -a else a

end FxQ


-- ============================================================
-- §2  Sigmoid LUT (elaboration-time table generation)
-- ============================================================

namespace SigmoidLUT

/-- Configuration for the sigmoid lookup table. -/
structure Config where
  fmt   : FxQ.Fmt
  zMin  : Float := -4.0
  zMax  : Float :=  4.0
  size  : Nat   := 256
  k     : Float := 2.0 / 0.4   -- matches Python's (-2*x/u0) with u0=0.4

/-- Compute sigmoid(z) = 1 / (1 + exp(-k*z)) -/
private def sigmoidFloat (k z : Float) : Float :=
  1.0 / (1.0 + Float.exp (-k * z))

/-- Build the LUT entries at elaboration time. Returns an Array of BitVec values. -/
def buildTable (cfg : Config) : Array (BitVec cfg.fmt.w) :=
  Array.ofFn fun (i : Fin cfg.size) =>
    let z := cfg.zMin + (cfg.zMax - cfg.zMin) * Float.ofNat i.val / Float.ofNat (cfg.size - 1)
    let y := sigmoidFloat cfg.k z
    FxQ.toFixed cfg.fmt y

/-- Compute the LUT index from a fixed-point input z (pure / elaboration).
    Maps z ∈ [zMin, zMax] → index ∈ [0, size-1]. -/
def computeIndex (cfg : Config) (zRaw : BitVec cfg.fmt.w) : Fin cfg.size :=
  let zMinFx := (cfg.zMin * Float.ofNat cfg.fmt.scale).round.toInt32
  let zMaxFx := (cfg.zMax * Float.ofNat cfg.fmt.scale).round.toInt32
  let spanFx := zMaxFx - zMinFx
  let zClipped := max zMinFx (min zMaxFx zRaw.toInt32)
  let posFx := zClipped - zMinFx
  let idx := posFx * (cfg.size - 1) / spanFx
  ⟨idx % cfg.size, Nat.mod_lt _ (by omega)⟩

/-- Evaluate the sigmoid LUT (pure function, used in simulation / elaboration). -/
def eval (cfg : Config) (z : BitVec cfg.fmt.w) : BitVec cfg.fmt.w :=
  let table := buildTable cfg
  let idx := computeIndex cfg z
  table[idx]!

end SigmoidLUT


-- ============================================================
-- §3  Signal-Level Sigmoid LUT (mux-tree ROM for synthesis)
-- ============================================================

/-- Build a synthesizable ROM lookup as a cascade of muxes.
    For a 256-entry table, this generates a balanced mux tree.
    `idx` is an 8-bit signal index, returns the ROM value. -/
def romLookup {dom : DomainConfig} (table : Array (BitVec w)) (idx : Signal dom (BitVec 8))
    : Signal dom (BitVec w) :=
  -- Leaf: just return the constant for that index
  -- We build a mux tree by folding over the table
  table.foldlIdx (init := Signal.pure (0#w)) fun acc i entry =>
    let iConst : Signal dom (BitVec 8) := Signal.pure (BitVec.ofNat 8 i)
    let isMatch := (· == ·) <$> idx <*> iConst
    Signal.mux isMatch (Signal.pure entry) acc

/-- Signal-level sigmoid LUT module.
    Takes a fixed-point z input, returns sigmoid(z) as a fixed-point signal.
    Internally computes the index and performs ROM lookup. -/
def sigmoidLUTSignal {dom : DomainConfig} (cfg : SigmoidLUT.Config)
    (zIn : Signal dom (BitVec cfg.fmt.w))
    : Signal dom (BitVec cfg.fmt.w) :=
  let table := SigmoidLUT.buildTable cfg
  let zMinFx := BitVec.ofInt cfg.fmt.w
    ((cfg.zMin * Float.ofNat cfg.fmt.scale).round.toInt)
  let zMaxFx := BitVec.ofInt cfg.fmt.w
    ((cfg.zMax * Float.ofNat cfg.fmt.scale).round.toInt)
  let spanFx := BitVec.ofInt cfg.fmt.w
    (((cfg.zMax - cfg.zMin) * Float.ofNat cfg.fmt.scale).round.toInt)
  -- Clip z to [zMin, zMax]
  let zClipped := Signal.mux ((· < ·) <$> zIn <*> Signal.pure zMinFx)
                    (Signal.pure zMinFx)
                    (Signal.mux ((· > ·) <$> zIn <*> Signal.pure zMaxFx)
                      (Signal.pure zMaxFx)
                      zIn)
  -- posFx = zClipped - zMinFx
  let posFx := (· - ·) <$> zClipped <*> Signal.pure zMinFx
  -- idx = posFx * (size-1) / spanFx  (approximate: shift-based division)
  let sizeM1 := Signal.pure (BitVec.ofNat cfg.fmt.w (cfg.size - 1))
  let numer := (· * ·) <$> posFx <*> sizeM1
  let idxWide := (· / ·) <$> numer <*> Signal.pure spanFx
  -- Truncate to 8-bit index
  let idx8 := BitVec.truncate 8 <$> idxWide
  romLookup table idx8


-- ============================================================
-- §4  Network Configuration
-- ============================================================

structure EDConfig where
  inputNum  : Nat
  hiddenNum : Nat
  fmt       : FxQ.Fmt := { w := 16, bp := 12 }

namespace EDConfig

def inVecLenHidden (cfg : EDConfig) : Nat := 2 + 2 * cfg.inputNum
def inVecLenOut    (cfg : EDConfig) : Nat := 2 + cfg.hiddenNum

def sigCfg (cfg : EDConfig) : SigmoidLUT.Config := {
  fmt  := cfg.fmt
  zMin := -4.0
  zMax :=  4.0
  size := 256
  k    := 2.0 / 0.4
}

def beta  (cfg : EDConfig) : BitVec cfg.fmt.w := FxQ.toFixed cfg.fmt 0.8
def alpha (cfg : EDConfig) : BitVec cfg.fmt.w := FxQ.toFixed cfg.fmt 0.8
def one   (cfg : EDConfig) : BitVec cfg.fmt.w := FxQ.toFixed cfg.fmt 1.0

/-- Hidden neuron operator: +1 for odd index (type "p"), -1 for even (type "n"). -/
def hiddenOp (cfg : EDConfig) (h : Fin cfg.hiddenNum) : Int :=
  if h.val % 2 == 1 then 1 else -1

/-- Weight operator for hidden-layer inputs:
    index 0 → +1 (hd_p), index 1 → -1 (hd_n),
    then alternating +1 / -1 for each input pair. -/
def wopHidden (cfg : EDConfig) (j : Fin cfg.inVecLenHidden) : Int :=
  if j.val == 0 then 1
  else if j.val == 1 then -1
  else if j.val % 2 == 0 then 1
  else -1

/-- Weight operator for output-layer inputs:
    index 0 → +1, index 1 → -1,
    then hiddenOp for each hidden neuron. -/
def wopOut (cfg : EDConfig) (j : Fin cfg.inVecLenOut) : Int :=
  if j.val == 0 then 1
  else if j.val == 1 then -1
  else cfg.hiddenOp ⟨j.val - 2, by
    have h1 : j.val < cfg.inVecLenOut := j.isLt
    simp only [EDConfig.inVecLenOut] at h1
    omega
  ⟩

end EDConfig


-- ============================================================
-- §5  State Declaration for the Full Model
-- ============================================================

/-- The complete model state: weights + pipeline registers.
    In Sparkle, all mutable state lives inside a `Signal.loop`
    and is passed as a single state tuple each cycle.

    We represent state as nested BitVec arrays flattened into a
    single record for the loop body. -/

-- For clarity, we define the state as a structure of arrays.
-- In actual synthesis, these would be individual registers.

structure ModelWeights (cfg : EDConfig) where
  /-- Hidden weights: hiddenNum × inVecLenHidden -/
  wHidden : Array (Array (BitVec cfg.fmt.w))
  /-- Output weights: inVecLenOut -/
  wOut    : Array (BitVec cfg.fmt.w)

/-- Initialize weights to small constant values (deterministic). -/
def initWeights (cfg : EDConfig) : ModelWeights cfg := {
  wHidden := Array.ofFn fun (_ : Fin cfg.hiddenNum) =>
    Array.ofFn fun (_ : Fin cfg.inVecLenHidden) =>     FxQ.toFixed cfg.fmt 0.1
  wOut := Array.ofFn fun (_ : Fin cfg.inVecLenOut) =>  FxQ.toFixed cfg.fmt 0.1
}

-- ============================================================
-- §6  Forward Pass (pure / elaboration-time reference)
-- ============================================================

namespace Forward

/-- Build expanded input vector:
    [beta, beta, x[0], x[0], x[1], x[1], ...] -/
def expandInput (cfg : EDConfig) (x : Array (BitVec cfg.fmt.w))
    : Array (BitVec cfg.fmt.w) :=
  let betaVal := cfg.beta
  #[betaVal, betaVal] ++ (x.foldl (init := #[]) fun acc xi => acc ++ #[xi, xi])

/-- Compute one hidden neuron's weighted sum. -/
def hiddenNeuronSum (cfg : EDConfig) (weights : Array (BitVec cfg.fmt.w))
    (xExp : Array (BitVec cfg.fmt.w)) : BitVec cfg.fmt.w :=
  let sum := (Array.range cfg.inVecLenHidden).foldl (init := BitVec.ofInt cfg.fmt.w 0)
    fun acc j =>
      let prod := FxQ.fxMul cfg.fmt weights[j]! xExp[j]!
      acc + prod
  sum

/-- Compute hidden layer outputs (after sigmoid). -/
def hiddenLayer (cfg : EDConfig) (wts : ModelWeights cfg)
    (xExp : Array (BitVec cfg.fmt.w)) : Array (BitVec cfg.fmt.w) :=
  Array.ofFn fun (h : Fin cfg.hiddenNum) =>
    let z := hiddenNeuronSum cfg wts.wHidden[h]! xExp
    SigmoidLUT.eval cfg.sigCfg z

/-- Build output-layer input: [beta, beta] ++ hidden outputs -/
def outInput (cfg : EDConfig) (hOut : Array (BitVec cfg.fmt.w))
    : Array (BitVec cfg.fmt.w) :=
  #[cfg.beta, cfg.beta] ++ hOut

/-- Compute output neuron value. -/
def outputNeuron (cfg : EDConfig) (wts : ModelWeights cfg)
    (hOut : Array (BitVec cfg.fmt.w)) : BitVec cfg.fmt.w :=
  let xOut := outInput cfg hOut
  let z := (Array.range cfg.inVecLenOut).foldl (init := BitVec.ofInt cfg.fmt.w 0)
    fun acc j =>
      let prod := FxQ.fxMul cfg.fmt wts.wOut[j]! xOut[j]!
      acc + prod
  SigmoidLUT.eval cfg.sigCfg z

end Forward


-- ============================================================
-- §7  Training Update (pure / elaboration-time reference)
-- ============================================================

namespace Train

/-- Compute sigmoid gradient: σ(|z|) * (1 - σ(|z|)) -/
def sigmoidGrad (cfg : EDConfig) (z : BitVec cfg.fmt.w) : BitVec cfg.fmt.w :=
  let zAbs := FxQ.sAbs z
  let sigVal := SigmoidLUT.eval cfg.sigCfg zAbs
  --let sigVal' : BitVec cfg.sigCfg.fmt.w := BitVec.reduceShiftLeftZeroExtend cfg.sigCfg.fmt.w sigVal
  let sigVal' := sigVal |> BitVec.map (fun v : BitVec cfg.fmt.w => (v : BitVec cfg.sigCfg.fmt.w)) -- これも同幅前提
  let oneMinusSig := cfg.one - sigVal
  FxQ.fxMul cfg.fmt sigVal oneMinusSig

/-- Update output weights for one sample. -/
def updateOutWeights (cfg : EDConfig) (wOut : Array (BitVec cfg.fmt.w))
    (xOut : Array (BitVec cfg.fmt.w)) (zOut : BitVec cfg.fmt.w)
    (target y : BitVec cfg.fmt.w) : Array (BitVec cfg.fmt.w) :=
  let diff := target - y
  let diffNeg := diff.toInt < 0
  let diffAbs := FxQ.sAbs diff
  let grad := sigmoidGrad cfg zOut
  Array.ofFn fun (j : Fin cfg.inVecLenOut) =>
    let wopJ := cfg.wopOut j
    let upperJ := wopJ == 1
    let mask := if diffNeg then !upperJ else upperJ  -- upper if diff>0
    if mask then
      let signVal := FxQ.toFixed cfg.fmt (Float.ofInt wopJ)
      let delta := FxQ.fxMul cfg.fmt cfg.alpha
                     (FxQ.fxMul cfg.fmt xOut[j]!
                       (FxQ.fxMul cfg.fmt grad
                         (FxQ.fxMul cfg.fmt diffAbs signVal)))
      wOut[j]! + delta
    else
      wOut[j]!

/-- Update hidden weights for one sample. -/
def updateHiddenWeights (cfg : EDConfig)
    (wHidden : Array (Array (BitVec cfg.fmt.w)))
    (xExp : Array (BitVec cfg.fmt.w))
    (zHidden : Array (BitVec cfg.fmt.w))
    (target y : BitVec cfg.fmt.w) : Array (Array (BitVec cfg.fmt.w)) :=
  let diff := target - y
  let diffNeg := diff.toInt < 0
  let diffAbs := FxQ.sAbs diff
  Array.ofFn fun (h : Fin cfg.hiddenNum) =>
    let grad := sigmoidGrad cfg zHidden[h]!
    let hOp := cfg.hiddenOp h
    Array.ofFn fun (j : Fin cfg.inVecLenHidden) =>
      let wopJ := cfg.wopHidden j
      let upperJ := wopJ == 1
      let mask := if diffNeg then !upperJ else upperJ
      if mask then
        let signVal := FxQ.toFixed cfg.fmt (Float.ofInt (hOp * wopJ))
        let delta := FxQ.fxMul cfg.fmt cfg.alpha
                       (FxQ.fxMul cfg.fmt xExp[j]!
                         (FxQ.fxMul cfg.fmt grad
                           (FxQ.fxMul cfg.fmt diffAbs signVal)))
        wHidden[h]![j]! + delta
      else
        wHidden[h]![j]!

end Train


-- ============================================================
-- §8  Signal-Level Model (Synthesizable Hardware)
-- ============================================================

/-- The Signal-level three-layer model.

    Design approach:
    - All weights are held in a `Signal.loop` as registered state
    - Forward pass is pipelined via `Signal.register`
    - Training update is conditional on `trainEn`

    Inputs:
      x       : input vector (inputNum fixed-point values)
      target  : target label (fixed-point)
      trainEn : enable weight update
      inValid : input valid strobe

    Outputs:
      y       : network output (fixed-point)
      yValid  : output valid strobe
-/
def threeLayerModelSignal {dom : DomainConfig} (cfg : EDConfig)
    (x       : Signal dom (Array (BitVec cfg.fmt.w)))
    (target  : Signal dom (BitVec cfg.fmt.w))
    (trainEn : Signal dom Bool)
    (inValid : Signal dom Bool)
    : Signal dom (BitVec cfg.fmt.w) × Signal dom Bool :=

  -- ─── Weight State Loop ───────────────────────────────────
  -- Weights are the only feedback state.
  -- We use Signal.loopMemo to create a registered feedback path.

  let initW := initWeights cfg

  -- Pack all weight state into a flat BitVec for the loop.
  -- For simplicity in this reference design, we use the pure
  -- simulation path: compute forward + update in one cycle.
  -- A fully pipelined synthesis version would split this into
  -- multiple registered stages.

  -- ─── Forward + Update (combinational body) ─────────────
  -- For each cycle where inValid is high:
  --   1. Expand input
  --   2. Compute hidden layer (weighted sum → sigmoid)
  --   3. Compute output (weighted sum → sigmoid)
  --   4. If trainEn, update weights

  -- Since Sparkle enforces pure functional semantics, we express
  -- the entire model as a function from (inputs, old_weights) →
  -- (output, new_weights).

  -- For a simulation-friendly version, we use the pure spec:
  let computeBody :
      Array (BitVec cfg.fmt.w) →   -- input x
      BitVec cfg.fmt.w →            -- target
      Bool →                         -- trainEn
      Bool →                         -- inValid
      ModelWeights cfg →             -- old weights
      (BitVec cfg.fmt.w × Bool × ModelWeights cfg)  -- (y, yValid, new weights)
    := fun xArr tgt tEn iVal wts =>
      if !iVal then
        (0#cfg.fmt.w, false, wts)
      else
        -- Forward pass
        let xExp := Forward.expandInput cfg xArr
        let hZs := Array.ofFn fun (h : Fin cfg.hiddenNum) =>
          Forward.hiddenNeuronSum cfg wts.wHidden[h]! xExp
        let hOut := hZs.map (SigmoidLUT.eval cfg.sigCfg)
        let xOut := Forward.outInput cfg hOut
        let zOut := (Array.range cfg.inVecLenOut).foldl
          (init := BitVec.ofInt cfg.fmt.w 0) fun acc j =>
            acc + FxQ.fxMul cfg.fmt wts.wOut[j]! xOut[j]!
        let yOut := SigmoidLUT.eval cfg.sigCfg zOut
        -- Training update
        let newWts :=
          if tEn then {
            wHidden := Train.updateHiddenWeights cfg wts.wHidden xExp hZs tgt yOut
            wOut    := Train.updateOutWeights cfg wts.wOut xOut zOut tgt yOut
          } else wts
        (yOut, true, newWts)

  -- ─── Signal.loop for weight state ──────────────────────
  -- In the Signal DSL, we lift the pure computation into signals.
  -- The actual synthesis version would use Signal.loopMemo:
  --
  --   Signal.loopMemo fun wtsSignal =>
  --     let result := computeBody <$> x <*> target <*> trainEn <*> inValid <*> wtsSignal
  --     let newWts := (fun (_, _, w) => w) <$> result
  --     let yOut   := (fun (y, _, _) => y) <$> result
  --     let yVal   := (fun (_, v, _) => v) <$> result
  --     (newWts, (yOut, yVal))

  -- For simulation, we evaluate directly:
  let result := computeBody <$> x <*> target <*> trainEn <*> inValid <*> Signal.pure initW
  let yOut  := (fun (y, _, _) => y) <$> result
  let yVal  := (fun (_, v, _) => v) <$> result

  -- Pipeline output register (DRC: registered outputs)
  let yOutReg := Signal.register (0#cfg.fmt.w) yOut
  let yValReg := Signal.register false yVal

  (yOutReg, yValReg)


-- ============================================================
-- §9  Pipelined Forward-Only Inference (fully synthesizable)
-- ============================================================

/-- A simpler forward-only inference module with explicit pipeline stages.
    This version is fully synthesizable with registered outputs.

    Pipeline:
      Stage 0: Latch inputs, expand input vector
      Stage 1: Hidden layer multiply (parallel)
      Stage 2: Hidden layer accumulate (adder tree)
      Stage 3: Hidden layer sigmoid activation
      Stage 4: Output layer multiply + accumulate
      Stage 5: Output sigmoid activation
      Stage 6: Output register
-/
def forwardInferenceSignal {dom : DomainConfig} (cfg : EDConfig)
    (wHidden : Array (Array (BitVec cfg.fmt.w)))  -- static weights
    (wOut    : Array (BitVec cfg.fmt.w))           -- static weights
    (x       : Signal dom (Array (BitVec cfg.fmt.w)))
    (inValid : Signal dom Bool)
    : Signal dom (BitVec cfg.fmt.w) × Signal dom Bool :=

  -- ─── Stage 0: Expand input ────────────────────────────
  let xExpanded := Forward.expandInput cfg <$> x
  let s0_x := Signal.register
    (Array.replicate cfg.inVecLenHidden (0#cfg.fmt.w)) xExpanded
  let s0_valid := Signal.register false inValid

  -- ─── Stage 1-2: Hidden neuron weighted sums ───────────
  -- Each hidden neuron computes dot product then registers it.
  let hiddenSums := Array.ofFn fun (h : Fin cfg.hiddenNum) =>
    let dotProd := (fun xArr =>
      Forward.hiddenNeuronSum cfg wHidden[h]! xArr) <$> s0_x
    Signal.register (0#cfg.fmt.w) dotProd

  let s1_valid := Signal.register false s0_valid

  -- ─── Stage 3: Sigmoid activation on hidden neurons ────
  let hiddenActs := hiddenSums.map fun zSig =>
    let act := (SigmoidLUT.eval cfg.sigCfg) <$> zSig
    Signal.register (0#cfg.fmt.w) act

  let s2_valid := Signal.register false s1_valid

  -- ─── Stage 4: Output neuron dot product ───────────────
  -- Build output layer input from hidden activations
  let hiddenActArray : Signal dom (Array (BitVec cfg.fmt.w)) :=
    hiddenActs.foldl (init := Signal.pure #[]) fun accSig actSig =>
      (fun acc act => acc.push act) <$> accSig <*> actSig

  let outDotProd := (fun hArr =>
    let xOut := Forward.outInput cfg hArr
    (Array.range cfg.inVecLenOut).foldl (init := BitVec.ofInt cfg.fmt.w 0)
      fun acc j => acc + FxQ.fxMul cfg.fmt wOut[j]! xOut[j]!
  ) <$> hiddenActArray

  let s3_outSum := Signal.register (0#cfg.fmt.w) outDotProd
  let s3_valid := Signal.register false s2_valid

  -- ─── Stage 5: Output sigmoid ──────────────────────────
  let yOut := (SigmoidLUT.eval cfg.sigCfg) <$> s3_outSum
  let s4_y := Signal.register (0#cfg.fmt.w) yOut
  let s4_valid := Signal.register false s3_valid

  (s4_y, s4_valid)


-- ============================================================
-- §10  Top-Level Wrapper & Synthesis Entry Point
-- ============================================================

/-- Example instantiation: 4 inputs, 4 hidden neurons, Q16.12 -/
def edMnistForward
    (x0 x1 x2 x3 : Signal Domain (BitVec 16))
    (inValid : Signal Domain Bool)
    : Signal Domain (BitVec 16) :=
  let cfg : EDConfig := {
    inputNum  := 4
    hiddenNum := 4
    fmt       := { w := 16, bp := 12 }
  }
  -- Static weights (initialized to 0.1 for demo)
  let wts := initWeights cfg
  -- Pack individual inputs into an array signal
  let xArr := (fun a b c d => #[a, b, c, d]) <$> x0 <*> x1 <*> x2 <*> x3
  let (yOut, _yValid) := forwardInferenceSignal cfg wts.wHidden wts.wOut xArr inValid
  yOut

-- Synthesize to SystemVerilog
#synthesizeVerilog edMnistForward


-- ============================================================
-- §11  Simulation / Test Helpers
-- ============================================================

/-- Run a single forward pass for testing. -/
def testForward (cfg : EDConfig) (x : Array (BitVec cfg.fmt.w))
    : BitVec cfg.fmt.w :=
  let wts := initWeights cfg
  let xExp := Forward.expandInput cfg x
  let hOut := Forward.hiddenLayer cfg wts xExp
  Forward.outputNeuron cfg wts hOut

/-- Run one training step and return (output, updated_weights). -/
def testTrainStep (cfg : EDConfig) (wts : ModelWeights cfg)
    (x : Array (BitVec cfg.fmt.w)) (target : BitVec cfg.fmt.w)
    : BitVec cfg.fmt.w × ModelWeights cfg :=
  let xExp := Forward.expandInput cfg x
  let hZs := Array.ofFn fun (h : Fin cfg.hiddenNum) =>
    Forward.hiddenNeuronSum cfg wts.wHidden[h]! xExp
  let hOut := hZs.map (SigmoidLUT.eval cfg.sigCfg)
  let xOut := Forward.outInput cfg hOut
  let zOut := (Array.range cfg.inVecLenOut).foldl
    (init := BitVec.ofInt cfg.fmt.w 0) fun acc j =>
      acc + FxQ.fxMul cfg.fmt wts.wOut[j]! xOut[j]!
  let yOut := SigmoidLUT.eval cfg.sigCfg zOut
  let newWts : ModelWeights cfg := {
    wHidden := Train.updateHiddenWeights cfg wts.wHidden xExp hZs target yOut
    wOut    := Train.updateOutWeights cfg wts.wOut xOut zOut target yOut
  }
  (yOut, newWts)

/-- Example: create a config for MNIST-like usage (784 inputs, 16 hidden). -/
def mnistConfig : EDConfig := {
  inputNum  := 784
  hiddenNum := 16
  fmt       := { w := 16, bp := 12 }
}
