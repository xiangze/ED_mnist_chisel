//ED method in chisel translate from https://qiita.com/pocokhc/items/f7ab56051bb936740b8f
// Sequencial Traning Version
package EDmnist

import chisel3._
import chisel3.util._
import chisel3.experimental.FixedPoint

// ============================================================
// Fixed-point + sigmoid LUT (SyncReadMem) for 1-cycle latency
// ============================================================
class SigmoidLUT_FF(val w: Int, val bp: Int,
                 val zMin: Double = -4.0, val zMax: Double = 4.0,
                 val size: Int = 256) extends Module {
  val io = IO(new Bundle {
    val zIn  = Input(FixedPoint(w.W, bp.BP))
    val yOut = Output(FixedPoint(w.W, bp.BP))
  })

  val k = 2.0 / 0.4 // Python: exp(-2*x/u0) with u0=0.4
  def sigmoid(z: Double): Double = 1.0 / (1.0 + math.exp(-k*z))

  val init = (0 until size).map { i =>
    val z = zMin + (zMax - zMin) * i.toDouble / (size - 1).toDouble
    val y = sigmoid(z)
    val fx = (y * (1 << bp)).round.toInt
    fx.S(w.W).asFixedPoint(bp.BP)
  }

  val mem = SyncReadMem(size, FixedPoint(w.W, bp.BP))
  // init memory (works in FIRRTL/Chisel elaboration contexts that support mem init)
  // If your flow doesn't support init, replace with ROM (VecInit) at cost of area.
  for (i <- 0 until size) { mem.write(i.U, init(i)) }

  // index mapping
  val zRaw = io.zIn.asSInt // scaled by 2^bp
  val zMinFx = (zMin * (1 << bp)).round.toInt
  val zMaxFx = (zMax * (1 << bp)).round.toInt
  val zClip = Wire(SInt((w+2).W))
  zClip := Mux(zRaw < zMinFx.S, zMinFx.S, Mux(zRaw > zMaxFx.S, zMaxFx.S, zRaw))

  val spanFx = (zMaxFx - zMinFx).max(1)
  val posFx  = zClip - zMinFx.S
  val idx    = (posFx * (size-1).S) / spanFx.S
  val idxU   = idx.asUInt

  io.yOut := mem.read(idxU, true.B) // 1-cycle latency
}

// ============================================================
// Three-layer model HW: online learning (sequential update)
// ============================================================
class ThreeLayerOnlineHW(
  val inputNum:  Int,
  val hiddenNum: Int,
  val w:         Int = 16,
  val bp:        Int = 12
) extends Module {

  // ---- IO
  val io = IO(new Bundle {
    val inValid = Input(Bool())
    val inReady = Output(Bool())
    val x       = Input(Vec(inputNum, FixedPoint(w.W, bp.BP)))
    val target  = Input(FixedPoint(w.W, bp.BP))
    val trainEn = Input(Bool())

    val outValid = Output(Bool())
    val y        = Output(FixedPoint(w.W, bp.BP))
  })

  // ---- constants
  val beta  = (0.8 * (1 << bp)).round.toInt.S(w.W).asFixedPoint(bp.BP)
  val alpha = (0.8 * (1 << bp)).round.toInt.S(w.W).asFixedPoint(bp.BP)
  val one   = (1.0 * (1 << bp)).round.toInt.S(w.W).asFixedPoint(bp.BP)

  // ---- dimensions
  val inLenHidden = 2 + 2*inputNum        // [beta,beta] + [x_i(p),x_i(n)]
  val inLenOut    = 2 + hiddenNum         // [beta,beta] + hidden outs

  // ---- operators: hidden neuron type alternating (Python: i%2==1 => p else n)
  val hiddenOp = VecInit((0 until hiddenNum).map(i => if (i%2==1) 1.S else (-1).S))

  // input weights_operator for hidden: hd_p=+1, hd_n=-1, each input duplicated: p=+1, n=-1
  val wopHidden = Wire(Vec(inLenHidden, SInt(2.W)))
  wopHidden(0) := 1.S
  wopHidden(1) := (-1).S
  for (i <- 0 until inputNum) {
    wopHidden(2 + 2*i)     :=  1.S
    wopHidden(2 + 2*i + 1) := (-1).S
  }

  // output weights_operator: hd_p=+1, hd_n=-1, then each hidden neuron operator
  val wopOut = Wire(Vec(inLenOut, SInt(2.W)))
  wopOut(0) := 1.S
  wopOut(1) := (-1).S
  for (h <- 0 until hiddenNum) { wopOut(2+h) := hiddenOp(h) }

  // ---- masks: upper if source op +1, lower if -1
  val upperMaskHidden = VecInit((0 until inLenHidden).map(j => wopHidden(j) === 1.S))
  val lowerMaskHidden = VecInit((0 until inLenHidden).map(j => wopHidden(j) === (-1).S))
  val upperMaskOut    = VecInit((0 until inLenOut).map(j => wopOut(j) === 1.S))
  val lowerMaskOut    = VecInit((0 until inLenOut).map(j => wopOut(j) === (-1).S))

  // ---- weights (small-scale): regs
  // init small deterministic values; replace with load path if needed
  def fx(d: Double): FixedPoint = (d * (1 << bp)).round.toInt.S(w.W).asFixedPoint(bp.BP)

  val wHidden = RegInit(VecInit(Seq.fill(hiddenNum)(
    VecInit(Seq.fill(inLenHidden)(fx(0.1)))
  )))
  val wOut = RegInit(VecInit(Seq.fill(inLenOut)(fx(0.1))))

  // ---- working regs for one sample (latched inputs)
  val xHiddenReg = Reg(Vec(inLenHidden, FixedPoint(w.W, bp.BP)))
  val targetReg  = Reg(FixedPoint(w.W, bp.BP))
  val trainReg   = Reg(Bool())

  // hidden outputs & pre-activations (store for grad)
  val hiddenY = Reg(Vec(hiddenNum, FixedPoint(w.W, bp.BP)))
  val hiddenZ = Reg(Vec(hiddenNum, FixedPoint(w.W, bp.BP))) // keep truncated z for grad LUT

  // output y and z
  val outY = Reg(FixedPoint(w.W, bp.BP))
  val outZ = Reg(FixedPoint(w.W, bp.BP))

  // ---- sigmoid LUT instances (shared)
  // We share one LUT and time-multiplex it for hidden/out and for grad.
  val lut = Module(new SigmoidLUT_FF(w, bp))

  // ---- FSM
  val sIdle :: sHMac :: sHSig :: sOMac :: sOSig :: sUpdPrep :: sUpdOut :: sUpdHidden :: sDone :: Nil =
    Enum(9)
  val state = RegInit(sIdle)

  // indices for sequential operations
  val hIdx = RegInit(0.U(log2Ceil(hiddenNum max 1).W))
  val jIdx = RegInit(0.U(log2Ceil((inLenHidden max inLenOut) max 1).W))

  // MAC accumulator (widen)
  val acc = RegInit(0.S((w+8).W).asFixedPoint(bp.BP))

  // sigmoid pipeline: LUT is 1-cycle latency, so we need a "pending" flag
  val sigPending = RegInit(false.B)
  val sigDestHidden = RegInit(false.B) // true => write hiddenY(hIdx), false => write outY
  val sigWriteZ     = RegInit(false.B) // for capturing Z before activation

  // ---- output controls
  io.inReady := (state === sIdle)
  io.outValid := (state === sDone)
  io.y := outY

  // ---- helper: absolute value for FixedPoint
  def absFx(x: FixedPoint): FixedPoint = {
    val s = x.asSInt
    Mux(s < 0.S, (-s).asFixedPoint(bp.BP), x)
  }

  // ---- latch input on fire
  val fire = io.inValid && io.inReady
  when(fire) {
    xHiddenReg(0) := beta
    xHiddenReg(1) := beta
    for (i <- 0 until inputNum) {
      xHiddenReg(2 + 2*i)     := io.x(i)
      xHiddenReg(2 + 2*i + 1) := io.x(i)
    }
    targetReg := io.target
    trainReg  := io.trainEn
  }

  // ---- default LUT input
  lut.io.zIn := 0.S(w.W).asFixedPoint(bp.BP)

  // ============================================================
  // FSM behavior
  // ============================================================
  switch(state) {

    is(sIdle) {
      when(fire) {
        // start hidden MAC for h=0, j=0
        hIdx := 0.U
        jIdx := 0.U
        acc  := 0.S.asFixedPoint(bp.BP)
        state := sHMac
      }
    }

    // ---- Hidden MAC: accumulate dot(xHiddenReg, wHidden[hIdx])
    is(sHMac) {
      val j = jIdx
      val prod = (xHiddenReg(j) * wHidden(hIdx)(j)).asFixedPoint(bp.BP)
      acc := (acc + prod).asFixedPoint(bp.BP)

      when(j === (inLenHidden-1).U) {
        // done MAC for this hidden neuron, capture Z and request sigmoid
        hiddenZ(hIdx) := acc.asFixedPoint(bp.BP)
        // request sigmoid(z)
        lut.io.zIn := acc.asFixedPoint(bp.BP)
        sigPending := true.B
        sigDestHidden := true.B
        sigWriteZ := false.B
        state := sHSig
      }.otherwise {
        jIdx := j + 1.U
      }
    }

    // ---- Hidden Sigmoid: wait 1 cycle then write hiddenY
    is(sHSig) {
      when(sigPending) {
        // LUT output available this cycle (1-cycle read); capture
        hiddenY(hIdx) := lut.io.yOut
        sigPending := false.B

        when(hIdx === (hiddenNum-1).U) {
          // proceed to output MAC
          jIdx := 0.U
          acc  := 0.S.asFixedPoint(bp.BP)
          state := sOMac
        }.otherwise {
          // next hidden neuron
          hIdx := hIdx + 1.U
          jIdx := 0.U
          acc  := 0.S.asFixedPoint(bp.BP)
          state := sHMac
        }
      }.otherwise {
        // assert LUT input from previous state (kept combinationally via reg not possible),
        // so we re-drive LUT with the stored Z for this hidden neuron.
        lut.io.zIn := hiddenZ(hIdx)
        sigPending := true.B
      }
    }

    // ---- Output MAC: dot([beta,beta,hiddenY...], wOut)
    is(sOMac) {
      val j = jIdx
      val xj = Wire(FixedPoint(w.W, bp.BP))
      xj := Mux(j === 0.U, beta,
           Mux(j === 1.U, beta,
             hiddenY(j - 2.U)
           ))
      val prod = (xj * wOut(j)).asFixedPoint(bp.BP)
      acc := (acc + prod).asFixedPoint(bp.BP)

      when(j === (inLenOut-1).U) {
        outZ := acc.asFixedPoint(bp.BP)
        lut.io.zIn := acc.asFixedPoint(bp.BP)
        sigPending := true.B
        sigDestHidden := false.B
        state := sOSig
      }.otherwise {
        jIdx := j + 1.U
      }
    }

    // ---- Output Sigmoid
    is(sOSig) {
      when(sigPending) {
        outY := lut.io.yOut
        sigPending := false.B
        state := sUpdPrep
      }.otherwise {
        lut.io.zIn := outZ
        sigPending := true.B
      }
    }

    // ---- Prepare update scalars: diffAbs, direction, grads
    is(sUpdPrep) {
      // If train disabled => finish
      when(!trainReg) {
        state := sDone
      }.otherwise {
        // compute diff & direction
        // store for update loops in regs
        state := sUpdOut
        jIdx := 0.U
      }
    }

    // ---- Update output weights sequentially
    is(sUpdOut) {
      val diff = (targetReg - outY).asFixedPoint(bp.BP)
      val directUpper = diff.asSInt >= 0.S
      val diffAbs = Mux(directUpper, diff, (-diff.asSInt).asFixedPoint(bp.BP))

      // gradOut = sigmoid(abs(outZ))*(1-sigmoid(abs(outZ)))
      // time-mux LUT: for simplicity, compute gradOut "on the fly" each cycle via LUT read.
      // We'll approximate by using sigmoid(abs(outZ)) already accessible by LUT if we drive it now.
      lut.io.zIn := absFx(outZ)

      // because LUT is 1-cycle, we introduce a small 2-phase inside this state:
      // phase=0 drive LUT, phase=1 use yAbs to update one weight.
      val phase = RegInit(0.U(1.W))
      val yAbsReg = Reg(FixedPoint(w.W, bp.BP))

      when(phase === 0.U) {
        yAbsReg := lut.io.yOut // from previous drive; first iteration is garbage but will settle after 1 cycle
        phase := 1.U
      }.otherwise {
        val yAbs = yAbsReg
        val gradOut = (yAbs * (one - yAbs)).asFixedPoint(bp.BP)

        val j = jIdx
        val mask = Mux(directUpper, upperMaskOut(j), lowerMaskOut(j))

        val xj = Wire(FixedPoint(w.W, bp.BP))
        xj := Mux(j === 0.U, beta,
             Mux(j === 1.U, beta,
               hiddenY(j - 2.U)
             ))

        // sign = operator(out)=+1 * wopOut(j)
        val sign = wopOut(j) // +/-1
        when(mask) {
          val signFx = sign.asFixedPoint(0.BP)
          val delta = (alpha * xj * gradOut * (diffAbs * signFx)).asFixedPoint(bp.BP)
          wOut(j) := (wOut(j) + delta).asFixedPoint(bp.BP)
        }

        when(j === (inLenOut-1).U) {
          // move to hidden update
          hIdx := 0.U
          jIdx := 0.U
          phase := 0.U
          state := sUpdHidden
        }.otherwise {
          jIdx := j + 1.U
          phase := 0.U
        }
      }
    }

    // ---- Update hidden weights sequentially (nested loops: h then j)
    is(sUpdHidden) {
      val diff = (targetReg - outY).asFixedPoint(bp.BP)
      val directUpper = diff.asSInt >= 0.S
      val diffAbs = Mux(directUpper, diff, (-diff.asSInt).asFixedPoint(bp.BP))

      // gradHidden(h) = sigmoid(abs(hiddenZ(h)))*(1-sigmoid(abs(hiddenZ(h))))
      lut.io.zIn := absFx(hiddenZ(hIdx))

      val phase = RegInit(0.U(1.W))
      val yAbsReg = Reg(FixedPoint(w.W, bp.BP))

      when(phase === 0.U) {
        yAbsReg := lut.io.yOut
        phase := 1.U
      }.otherwise {
        val yAbs = yAbsReg
        val gradH = (yAbs * (one - yAbs)).asFixedPoint(bp.BP)

        val j = jIdx
        val mask = Mux(directUpper, upperMaskHidden(j), lowerMaskHidden(j))

        // sign = operator(hidden) * wopHidden(j)
        val sign = (hiddenOp(hIdx) * wopHidden(j)).asSInt // +/-1
        when(mask) {
          val signFx = sign.asFixedPoint(0.BP)
          val delta = (alpha * xHiddenReg(j) * gradH * (diffAbs * signFx)).asFixedPoint(bp.BP)
          wHidden(hIdx)(j) := (wHidden(hIdx)(j) + delta).asFixedPoint(bp.BP)
        }

        when(j === (inLenHidden-1).U) {
          when(hIdx === (hiddenNum-1).U) {
            phase := 0.U
            state := sDone
          }.otherwise {
            hIdx := hIdx + 1.U
            jIdx := 0.U
            phase := 0.U
          }
        }.otherwise {
          jIdx := j + 1.U
          phase := 0.U
        }
      }
    }

    // ---- Done: single-cycle valid pulse
    is(sDone) {
      // After presenting outValid for a cycle, return to idle
      state := sIdle
    }
  }
}
