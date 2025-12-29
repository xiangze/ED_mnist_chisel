//ED method in chisel translate from https://qiita.com/pocokhc/items/f7ab56051bb936740b8f
package EDmnist

import chisel3._
import chisel3.util._
import chisel3.experimental.FixedPoint

/** 簡易 fixed-point (signed): raw represents value * 2^bp */
final case class Q(bp: Int, w: Int) {
  require(bp >= 0 && w > 0)
  def dtype: SInt = SInt(w.W)
  def scale: BigInt = BigInt(1) << bp
}
// -----------------------------
// Fixed-point helpers
// -----------------------------
object Fx {
  def clip(x: SInt, lo: Int, hi: Int): SInt = {
    val loS = lo.S
    val hiS = hi.S
    Mux(x < loS, loS, Mux(x > hiS, hiS, x))
  }
}

// -----------------------------
// Sigmoid LUT (piecewise via ROM)
// z is fixed-point, but we index by clipped z range
// -----------------------------
class SigmoidLUT(val w: Int, val bp: Int,
                 val zMin: Double = -4.0, val zMax: Double = 4.0,
                 val size: Int = 256) extends Module {
  val io = IO(new Bundle {
    val zIn  = Input(FixedPoint(w.W, bp.BP))
    val yOut = Output(FixedPoint(w.W, bp.BP))
  })

  // Precompute LUT in elaboration time
  // sigmoid approx: 1/(1+exp(-k*z))
  val k = 2.0 / 0.4  // matches Python's (-2*x/u0) with u0=0.4
  def sigmoid(z: Double): Double = 1.0 / (1.0 + math.exp(-k*z))

  val table = (0 until size).map { i =>
    val z = zMin + (zMax - zMin) * i.toDouble / (size - 1).toDouble
    val y = sigmoid(z)
    val fx = (y * (1 << bp)).round.toLong
    fx.S(w.W).asFixedPoint(bp.BP)
  }
  val rom = VecInit(table)

  // index: map zIn to [0, size-1]
  val zScaled = Wire(SInt((w+4).W))
  // Convert FixedPoint to SInt raw
  val zRaw = io.zIn.asSInt
  // zRaw represents z*(2^bp)
  // We map z in [zMin,zMax] to index range
  val zMinFx = (zMin * (1 << bp)).round.toLong
  val zMaxFx = (zMax * (1 << bp)).round.toLong
  val zClip  = Fx.clip(zRaw, zMinFx.toInt, zMaxFx.toInt)

  val spanFx = (zMaxFx - zMinFx).toInt
  val posFx  = zClip - zMinFx.S
  // idx = posFx * (size-1) / spanFx
  val idx = (posFx.asSInt * (size-1).S) / spanFx.S
  val idxU = idx.asUInt

  io.yOut := rom(idxU)
}

// -----------------------------
// Main 3-layer model (forward + optional train update)
// -----------------------------
class ThreeLayerModelHW(
  val inputNum:  Int,
  val hiddenNum: Int,
  val w:         Int = 16,
  val bp:        Int = 12
) extends Module {

  // Inputs are scalars 0/1 in Python; we accept fixed-point
  val io = IO(new Bundle {
    val inValid   = Input(Bool())
    val inReady   = Output(Bool())
    val x         = Input(Vec(inputNum, FixedPoint(w.W, bp.BP)))
    val target    = Input(FixedPoint(w.W, bp.BP))
    val trainEn   = Input(Bool()) // when high, update weights using the current sample
    val yValid    = Output(Bool())
    val y         = Output(FixedPoint(w.W, bp.BP))
  })

  // -----------------------------
  // Constants (beta, alpha)
  // -----------------------------
  // beta, alpha are fixed constants here (could be IO/config regs)
  val beta  = (0.8 * (1 << bp)).round.toInt.S(w.W).asFixedPoint(bp.BP)
  val alpha = (0.8 * (1 << bp)).round.toInt.S(w.W).asFixedPoint(bp.BP)
  val one   = (1.0 * (1 << bp)).round.toInt.S(w.W).asFixedPoint(bp.BP)

  // -----------------------------
  // Build expanded input vector: for each input n => [p=n, n=n] like Python
  // plus two "hd" bias-like inputs = [beta, beta]
  // Total inputs to hidden neuron = 2 (hd) + 2*inputNum
  // -----------------------------
  val inVecLenHidden = 2 + 2*inputNum
  val xExpanded = Wire(Vec(inVecLenHidden, FixedPoint(w.W, bp.BP)))
  xExpanded(0) := beta
  xExpanded(1) := beta
  for (i <- 0 until inputNum) {
    xExpanded(2 + 2*i)     := io.x(i) // p
    xExpanded(2 + 2*i + 1) := io.x(i) // n
  }

  // Hidden neuron types alternate: Python: ("p" if i%2==1 else "n")
  // operator = +1 for p, -1 for n
  val hiddenOperator = VecInit((0 until hiddenNum).map { i => if (i % 2 == 1) 1.S else (-1).S })

  // Input weights_operator: hd_p=+1, hd_n=-1, inputs are [p=+1,n=-1] repeated
  // total len = inVecLenHidden
  val wopHidden = Wire(Vec(inVecLenHidden, SInt(2.W)))
  wopHidden(0) := 1.S  // hd_p
  wopHidden(1) := (-1).S // hd_n
  for (i <- 0 until inputNum) {
    wopHidden(2 + 2*i)     :=  1.S
    wopHidden(2 + 2*i + 1) := (-1).S
  }

  // upper/lower index masks for each neuron:
  // Python: if input source is "p" => upper else lower
  // Here: upper indices are those with wop == +1, lower indices those with wop == -1
  val upperMaskHidden = VecInit((0 until inVecLenHidden).map(j => (wopHidden(j) === 1.S)))
  val lowerMaskHidden = VecInit((0 until inVecLenHidden).map(j => (wopHidden(j) === (-1).S)))

  // -----------------------------
  // Weights storage
  // Hidden weights: hiddenNum x inVecLenHidden
  // Output neuron input length = 2 (hd) + hiddenNum
  // -----------------------------
  val inVecLenOut = 2 + hiddenNum
  val wopOut = Wire(Vec(inVecLenOut, SInt(2.W)))
  wopOut(0) := 1.S
  wopOut(1) := (-1).S
  for (i <- 0 until hiddenNum) { wopOut(2 + i) := hiddenOperator(i) } // source neuron operator

  val upperMaskOut = VecInit((0 until inVecLenOut).map(j => (wopOut(j) === 1.S)))
  val lowerMaskOut = VecInit((0 until inVecLenOut).map(j => (wopOut(j) === (-1).S)))

  // Weight regs (initialized pseudo-random-like constants; in real HW you'd load them)
  // For a self-contained example, we init to small values depending on sign rules.
  def initWeight(sign: Int): FixedPoint = {
    val v = 0.1 * sign // deterministic small init (Python uses random)
    (v * (1 << bp)).round.toInt.S(w.W).asFixedPoint(bp.BP)
  }

  val wHidden = RegInit(VecInit(Seq.fill(hiddenNum)(
    VecInit((0 until inVecLenHidden).map { j =>
      // sign rule similar to Python's pp+, pn-, np-, nn+
      // hidden neuron type sign: hiddenOperator
      val sign = 1 // placeholder, we apply sign by neuron/operator below
      initWeight(sign)
    })
  )))

  val wOut = RegInit(VecInit((0 until inVecLenOut).map(_ => initWeight(1))))

  // -----------------------------
  // Pipeline control
  // -----------------------------
  // 4-stage pipeline:
  // S0 latch inputs
  // S1 parallel multiplies
  // S2 adder tree (sum)
  // S3 activation (sigmoid LUT)
  // S4 (optional) weight update
  val s0_fire = io.inValid
  io.inReady := true.B // always ready in this simple pipeline

  // Stage0 regs
  val s0_xHidden = Reg(Vec(inVecLenHidden, FixedPoint(w.W, bp.BP)))
  val s0_target  = Reg(FixedPoint(w.W, bp.BP))
  val s0_trainEn = Reg(Bool())

  when(s0_fire) {
    s0_xHidden := xExpanded
    s0_target  := io.target
    s0_trainEn := io.trainEn
  }

  // -----------------------------
  // Hidden forward: compute each hidden neuron output
  // -----------------------------
  // Stage1: muls
  val s1_mulHidden = Reg(Vec(hiddenNum, Vec(inVecLenHidden, FixedPoint((2*w).W, bp.BP))))
  when(s0_fire) {
    for (h <- 0 until hiddenNum) {
      for (j <- 0 until inVecLenHidden) {
        // widen for mul
        s1_mulHidden(h)(j) := (s0_xHidden(j) * wHidden(h)(j)).asFixedPoint(bp.BP)
      }
    }
  }

  // Stage2: sums (adder tree-ish; here linear fold)
  val s2_sumHidden = Reg(Vec(hiddenNum, FixedPoint((2*w+4).W, bp.BP)))
  when(RegNext(s0_fire, init=false.B)) {
    for (h <- 0 until hiddenNum) {
      var acc = 0.S((2*w+4).W).asFixedPoint(bp.BP)
      for (j <- 0 until inVecLenHidden) 
        acc = (acc + s1_mulHidden(h)(j)).asFixedPoint(bp.BP)
      s2_sumHidden(h) := acc
    }
  }

  // Stage3: sigmoid activation via LUT (one per neuron)
  val sigmoidHidden = Seq.fill(hiddenNum)(Module(new SigmoidLUT(w, bp)))
  val s3_yHidden = Reg(Vec(hiddenNum, FixedPoint(w.W, bp.BP)))
  val s3_zHidden = Reg(Vec(hiddenNum, FixedPoint((2*w+4).W, bp.BP))) // keep pre-activation for grad
  when(RegNext(RegNext(s0_fire, init=false.B), init=false.B)) {
    for (h <- 0 until hiddenNum) 
      s3_zHidden(h) := s2_sumHidden(h)
  }
  for (h <- 0 until hiddenNum) 
    sigmoidHidden(h).io.zIn := s3_zHidden(h).asFixedPoint(bp.BP) // truncate
    
  when(RegNext(RegNext(RegNext(s0_fire, init=false.B), init=false.B), init=false.B)) 
    for (h <- 0 until hiddenNum) 
      s3_yHidden(h) := sigmoidHidden(h).io.yOut
  
  // -----------------------------
  // Output forward
  // -----------------------------
  // Build out input vector: [beta,beta] + hidden outputs
  val s3_xOut = Wire(Vec(inVecLenOut, FixedPoint(w.W, bp.BP)))
  s3_xOut(0) := beta
  s3_xOut(1) := beta
  for (h <- 0 until hiddenNum) { s3_xOut(2+h) := s3_yHidden(h) }

  // Stage4: out mul + sum + activation
  val s4_mulOut = Reg(Vec(inVecLenOut, FixedPoint((2*w).W, bp.BP)))
  when(RegNext(RegNext(RegNext(s0_fire, init=false.B), init=false.B), init=false.B)) {
    for (j <- 0 until inVecLenOut) {
      s4_mulOut(j) := (s3_xOut(j) * wOut(j)).asFixedPoint(bp.BP)
    }
  }

  val s5_sumOut = Reg(FixedPoint((2*w+4).W, bp.BP))
  when(RegNext(RegNext(RegNext(RegNext(s0_fire, init=false.B), init=false.B), init=false.B), init=false.B)) {
    var acc = 0.S((2*w+4).W).asFixedPoint(bp.BP)
    for (j <- 0 until inVecLenOut) { acc = (acc + s4_mulOut(j)).asFixedPoint(bp.BP) }
    s5_sumOut := acc
  }

  val sigmoidOut = Module(new SigmoidLUT(w, bp))
  sigmoidOut.io.zIn := s5_sumOut.asFixedPoint(bp.BP)

  val s6_yOut = Reg(FixedPoint(w.W, bp.BP))
  val s6_zOut = Reg(FixedPoint((2*w+4).W, bp.BP))
  val s6_target = Reg(FixedPoint(w.W, bp.BP))
  val s6_trainEn = Reg(Bool())

  when(RegNext(RegNext(RegNext(RegNext(RegNext(s0_fire, init=false.B), init=false.B), init=false.B), init=false.B), init=false.B)) {
    s6_zOut    := s5_sumOut
    s6_yOut    := sigmoidOut.io.yOut
    s6_target  := s0_target
    s6_trainEn := s0_trainEn
  }

  // Output
  io.y := s6_yOut
  io.yValid := RegNext(RegNext(RegNext(RegNext(RegNext(RegNext(s0_fire, init=false.B), init=false.B), init=false.B), init=false.B), init=false.B), init=false.B)

  // -----------------------------
  // Training update stage (weight update pipeline)
  // Mimic Python:
  // diff = target - y
  // if diff>0 => upper else lower, diff=abs(diff)
  // update output weights and hidden weights with same diff
  // grad = sigmoid(abs(z))*(1-sigmoid(abs(z))) ; use LUT output for abs(z)
  // delta = alpha * x_i * grad * (diff * operator * wop_i)
  // -----------------------------

  // Determine direction and abs(diff)
  val diff = (s6_target - s6_yOut).asFixedPoint(bp.BP)
  val diffNeg = diff.asSInt < 0.S
  val diffAbs = Wire(FixedPoint(w.W, bp.BP))
  diffAbs := Mux(diffNeg, (-diff.asSInt).asFixedPoint(bp.BP), diff)
  
    val directUpper = !diffNeg

  // grad for output neuron: use abs(zOut) -> sigmoid -> y -> y*(1-y)

  // Hidden grads similarly (based on stored zHidden)
  val sigAbsHidden = Seq.fill(hiddenNum)(Module(new SigmoidLUT(w, bp)))
  val gradHidden = Wire(Vec(hiddenNum, FixedPoint(w.W, bp.BP)))
  for (h <- 0 until hiddenNum) {
    val zAbs = Mux(s3_zHidden(h).asSInt < 0.S, (-s3_zHidden(h).asSInt).asFixedPoint(bp.BP), s3_zHidden(h).asFixedPoint(bp.BP))
    sigAbsHidden(h).io.zIn := zAbs
    val yAbs = sigAbsHidden(h).io.yOut
    gradHidden(h) := (yAbs * (one - yAbs)).asFixedPoint(bp.BP)
  }

  val zOutAbs = WireDefault(Wire(FixedPoint(w.W, bp.BP)))
  zOutAbs := Mux(s6_zOut.asSInt < 0.S, (-s6_zOut.asSInt).asFixedPoint(bp.BP), s6_zOut.asFixedPoint(bp.BP))
  val sigAbsOut = Module(new SigmoidLUT(w, bp))
  sigAbsOut.io.zIn := zOutAbs
  val yAbsOut = sigAbsOut.io.yOut
  val gradOut = (yAbsOut * (one - yAbsOut)).asFixedPoint(bp.BP)
  // Update weights when trainEn at this stage
  when(io.yValid && s6_trainEn) {
    // Output operator is +1 (p)
    val outOp = 1.S
    for (j <- 0 until inVecLenOut) {
      val use = if (j < 2) true.B else true.B
      val mask = Mux(directUpper, upperMaskOut(j), lowerMaskOut(j))
      when(mask && use) {
        val sign = (outOp * wopOut(j)).asSInt // +/-1
        val delta = (alpha * s3_xOut(j) * gradOut * (diffAbs * sign.asFixedPoint(0.BP))).asFixedPoint(bp.BP)
        wOut(j) := (wOut(j) + delta).asFixedPoint(bp.BP)
      }
    }

    // Hidden weights update
    for (h <- 0 until hiddenNum) {
      val hOp = hiddenOperator(h) // +/-1
      for (j <- 0 until inVecLenHidden) {
        val mask = Mux(directUpper, upperMaskHidden(j), lowerMaskHidden(j))
        when(mask) {
          val sign = (hOp * wopHidden(j)).asSInt
          val delta = (alpha * s0_xHidden(j) * gradHidden(h) * (diffAbs * sign.asFixedPoint(0.BP))).asFixedPoint(bp.BP)
          wHidden(h)(j) := (wHidden(h)(j) + delta).asFixedPoint(bp.BP)
        }
      }
    }
  }
}
