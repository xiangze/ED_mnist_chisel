//ED method in chisel – integer-arithmetic version
// All fixed-point values are represented as SInt scaled by 2^bp.
// Multiplication of two Q(bp) values: (a * b) >> bp  (arithmetic right shift)
package EDmnist

import chisel3._
import chisel3.util._

/** Helper object describing the Q-format parameters */
final case class Q(bp: Int, w: Int) {
  require(bp >= 0 && w > 0)
  def scale: BigInt = BigInt(1) << bp
}

// -----------------------------
// Integer fixed-point helpers
// -----------------------------
object Fx {
  /** Saturate x to [lo, hi] */
  def clip(x: SInt, lo: Int, hi: Int): SInt = {
    val loS = lo.S
    val hiS = hi.S
    Mux(x < loS, loS, Mux(x > hiS, hiS, x))
  }

  /** Fixed-point multiply: (a * b) >>> bp, result truncated to rw bits */
  def mul(a: SInt, b: SInt, bp: Int, rw: Int): SInt = {
    val full = a * b            // width = a.width + b.width
    val shifted = (full >> bp)  // arithmetic right shift
    shifted(rw - 1, 0).asSInt
  }
}

// -----------------------------
// Sigmoid LUT (piecewise via ROM)
// All I/O and storage in SInt (Q-format integers)
// -----------------------------
class SigmoidLUT(val w: Int, val bp: Int,
                 val zMin: Double = -4.0, val zMax: Double = 4.0,
                 val size: Int = 256) extends Module {
  val io = IO(new Bundle {
    val zIn  = Input(SInt(w.W))
    val yOut = Output(SInt(w.W))
  })

  // Precompute LUT at elaboration time
  val k = 2.0 / 0.4  // matches Python's (-2*x/u0) with u0=0.4
  def sigmoid(z: Double): Double = 1.0 / (1.0 + math.exp(-k * z))

  val table = VecInit((0 until size).map { i =>
    val z  = zMin + (zMax - zMin) * i.toDouble / (size - 1).toDouble
    val y  = sigmoid(z)
    val fx = (y * (1 << bp)).round.toLong
    fx.S(w.W)
  })

  // Map zIn (Q-format) to table index [0, size-1]
  val zMinFx = (zMin * (1 << bp)).round.toLong.toInt
  val zMaxFx = (zMax * (1 << bp)).round.toLong.toInt
  val spanFx = (zMaxFx - zMinFx)

  val zClip = Fx.clip(io.zIn, zMinFx, zMaxFx)
  val posFx = zClip - zMinFx.S

  // idx = posFx * (size-1) / spanFx
  val idx  = (posFx * (size - 1).S) / spanFx.S
  val idxU = idx.asUInt

  io.yOut := table(idxU)
}

// -----------------------------
// Main 3-layer model (forward + optional train update)
// All values are SInt in Q(bp,w) format.
// -----------------------------
class ThreeLayerModelHW(
  val inputNum:  Int,
  val hiddenNum: Int,
  val w:         Int = 16,
  val bp:        Int = 12
) extends Module {

  val io = IO(new Bundle {
    val inValid = Input(Bool())
    val inReady = Output(Bool())
    val x       = Input(Vec(inputNum, SInt(w.W)))
    val target  = Input(SInt(w.W))
    val trainEn = Input(Bool())
    val yValid  = Output(Bool())
    val y       = Output(SInt(w.W))
  })

  // Wider types used inside the datapath
  val mw  = 2 * w      // after one multiplication
  val aw  = mw + 4     // after accumulation (adder tree)

  // -----------------------------
  // Constants (Q-format integers)
  // -----------------------------
  val beta:  SInt = (0.8 * (1 << bp)).round.toInt.S(w.W)
  val alpha: SInt = (0.8 * (1 << bp)).round.toInt.S(w.W)
  val one:   SInt = (1.0 * (1 << bp)).round.toInt.S(w.W)

  // -----------------------------
  // Expanded input vector for hidden layer
  // [hd_p=beta, hd_n=beta] ++ [p_i, n_i] for each input
  // -----------------------------
  val inVecLenHidden = 2 + 2 * inputNum

  val xExpanded = Wire(Vec(inVecLenHidden, SInt(w.W)))
  xExpanded(0) := beta
  xExpanded(1) := beta
  for (i <- 0 until inputNum) {
    xExpanded(2 + 2 * i)     := io.x(i)
    xExpanded(2 + 2 * i + 1) := io.x(i)
  }

  // Hidden neuron operator: +1 for p (odd index), -1 for n (even index)
  val hiddenOperator = VecInit((0 until hiddenNum).map { i =>
    if (i % 2 == 1) 1.S(2.W) else (-1).S(2.W)
  })

  // Weight operator for hidden-layer inputs
  val wopHidden = Wire(Vec(inVecLenHidden, SInt(2.W)))
  wopHidden(0) := 1.S
  wopHidden(1) := (-1).S
  for (i <- 0 until inputNum) {
    wopHidden(2 + 2 * i)     :=  1.S
    wopHidden(2 + 2 * i + 1) := (-1).S
  }

  val upperMaskHidden = VecInit((0 until inVecLenHidden).map(j => wopHidden(j) === 1.S))
  val lowerMaskHidden = VecInit((0 until inVecLenHidden).map(j => wopHidden(j) === (-1).S))

  // -----------------------------
  // Output layer sizing
  // -----------------------------
  val inVecLenOut = 2 + hiddenNum

  val wopOut = Wire(Vec(inVecLenOut, SInt(2.W)))
  wopOut(0) := 1.S
  wopOut(1) := (-1).S
  for (i <- 0 until hiddenNum) { wopOut(2 + i) := hiddenOperator(i) }

  val upperMaskOut = VecInit((0 until inVecLenOut).map(j => wopOut(j) === 1.S))
  val lowerMaskOut = VecInit((0 until inVecLenOut).map(j => wopOut(j) === (-1).S))

  // -----------------------------
  // Weight registers (SInt, Q-format)
  // -----------------------------
  def initWeight(sign: Int): SInt = {
    val v = 0.1 * sign
    (v * (1 << bp)).round.toInt.S(w.W)
  }

  val wHidden = RegInit(VecInit(Seq.fill(hiddenNum)(
    VecInit((0 until inVecLenHidden).map(_ => initWeight(1)))
  )))

  val wOut = RegInit(VecInit((0 until inVecLenOut).map(_ => initWeight(1))))

  // -----------------------------
  // Pipeline control
  // -----------------------------
  val s0_fire = io.inValid
  io.inReady := true.B

  // Stage 0: latch inputs
  val s0_xHidden = Reg(Vec(inVecLenHidden, SInt(w.W)))
  val s0_target  = Reg(SInt(w.W))
  val s0_trainEn = Reg(Bool())
  when(s0_fire) {
    s0_xHidden := xExpanded
    s0_target  := io.target
    s0_trainEn := io.trainEn
  }

  // ===========================
  //  HIDDEN LAYER FORWARD
  // ===========================

  // Stage 1: multiplications  (x_j * w_j) >> bp  → mw bits kept before accumulation
  val s1_mulHidden = Reg(Vec(hiddenNum, Vec(inVecLenHidden, SInt(mw.W))))
  when(s0_fire) {
    for (h <- 0 until hiddenNum; j <- 0 until inVecLenHidden) {
      s1_mulHidden(h)(j) := Fx.mul(s0_xHidden(j), wHidden(h)(j), bp, mw)
    }
  }

  // Stage 2: accumulation
  val s2_sumHidden = Reg(Vec(hiddenNum, SInt(aw.W)))
  when(RegNext(s0_fire, init = false.B)) {
    for (h <- 0 until hiddenNum) {
      val terms = (0 until inVecLenHidden).map(j =>
        Cat(s1_mulHidden(h)(j)(mw - 1), s1_mulHidden(h)(j)).asSInt  // sign-extend to aw
      )
      s2_sumHidden(h) := terms.reduce(_ +& _)
    }
  }

  // Stage 3: sigmoid activation
  val sigmoidHidden = Seq.fill(hiddenNum)(Module(new SigmoidLUT(w, bp)))
  val s3_yHidden = Reg(Vec(hiddenNum, SInt(w.W)))
  val s3_zHidden = Reg(Vec(hiddenNum, SInt(aw.W)))

  when(RegNext(RegNext(s0_fire, init = false.B), init = false.B)) {
    for (h <- 0 until hiddenNum)
      s3_zHidden(h) := s2_sumHidden(h)
  }

  for (h <- 0 until hiddenNum)
    sigmoidHidden(h).io.zIn := s3_zHidden(h)(w - 1, 0).asSInt  // truncate to w bits for LUT

  when(RegNext(RegNext(RegNext(s0_fire, init = false.B), init = false.B), init = false.B)) {
    for (h <- 0 until hiddenNum)
      s3_yHidden(h) := sigmoidHidden(h).io.yOut
  }

  // ===========================
  //  OUTPUT LAYER FORWARD
  // ===========================

  // Build output-layer input vector
  val s3_xOut = Wire(Vec(inVecLenOut, SInt(w.W)))
  s3_xOut(0) := beta
  s3_xOut(1) := beta
  for (h <- 0 until hiddenNum) { s3_xOut(2 + h) := s3_yHidden(h) }

  // Stage 4: output multiplications
  val s4_mulOut = Reg(Vec(inVecLenOut, SInt(mw.W)))
  when(RegNext(RegNext(RegNext(s0_fire, init = false.B), init = false.B), init = false.B)) {
    for (j <- 0 until inVecLenOut)
      s4_mulOut(j) := Fx.mul(s3_xOut(j), wOut(j), bp, mw)
  }

  // Stage 5: output accumulation
  val s5_sumOut = Reg(SInt(aw.W))
  when(RegNext(RegNext(RegNext(RegNext(s0_fire, init = false.B), init = false.B), init = false.B), init = false.B)) {
    val terms = (0 until inVecLenOut).map(j =>
      Cat(s4_mulOut(j)(mw - 1), s4_mulOut(j)).asSInt
    )
    s5_sumOut := terms.reduce(_ +& _)
  }

  // Stage 6: output sigmoid + latch results
  val sigmoidOut = Module(new SigmoidLUT(w, bp))
  sigmoidOut.io.zIn := s5_sumOut(w - 1, 0).asSInt

  val s6_yOut    = Reg(SInt(w.W))
  val s6_zOut    = Reg(SInt(aw.W))
  val s6_target  = Reg(SInt(w.W))
  val s6_trainEn = Reg(Bool())
  when(RegNext(RegNext(RegNext(RegNext(RegNext(s0_fire, init = false.B), init = false.B), init = false.B), init = false.B), init = false.B)) {
    s6_zOut    := s5_sumOut
    s6_yOut    := sigmoidOut.io.yOut
    s6_target  := s0_target
    s6_trainEn := s0_trainEn
  }

  // Final output
  io.y := s6_yOut
  io.yValid := RegNext(RegNext(RegNext(RegNext(RegNext(RegNext(
    s0_fire, init = false.B), init = false.B), init = false.B),
    init = false.B), init = false.B), init = false.B)

  // ===========================
  //  TRAINING UPDATE
  // ===========================

  // diff = target - y  (Q-format)
  val diff    = s6_target - s6_yOut
  val diffNeg = diff < 0.S
  val diffAbs = Mux(diffNeg, -diff, diff)

  val directUpper = !diffNeg

  // --- Hidden-layer gradients: grad = sig(|z|) * (1 - sig(|z|)) ---
  val sigAbsHidden = Seq.fill(hiddenNum)(Module(new SigmoidLUT(w, bp)))
  val gradHidden   = Wire(Vec(hiddenNum, SInt(w.W)))

  for (h <- 0 until hiddenNum) {
    val zTrunc = s3_zHidden(h)(w - 1, 0).asSInt
    val zAbs   = Mux(zTrunc < 0.S, -zTrunc, zTrunc)
    sigAbsHidden(h).io.zIn := zAbs
    val yAbs = sigAbsHidden(h).io.yOut          // Q(bp,w)
    // grad = yAbs * (one - yAbs) >> bp
    gradHidden(h) := Fx.mul(yAbs, one - yAbs, bp, w)
  }

  // --- Output-layer gradient ---
  val zOutTrunc = s6_zOut(w - 1, 0).asSInt
  val zOutAbs   = Mux(zOutTrunc < 0.S, -zOutTrunc, zOutTrunc)
  val sigAbsOut = Module(new SigmoidLUT(w, bp))
  sigAbsOut.io.zIn := zOutAbs
  val yAbsOut = sigAbsOut.io.yOut
  val gradOut = Fx.mul(yAbsOut, one - yAbsOut, bp, w)

  // --- Weight updates ---
  // delta = alpha * x_i * grad * diffAbs * sign
  // Computed as:  t1 = (alpha * x_i) >> bp
  //               t2 = (t1 * grad) >> bp
  //               t3 = (t2 * diffAbs) >> bp
  //               delta = t3 * sign            (sign is +/-1, no shift needed)

  when(io.yValid && s6_trainEn) {
    // -- Output weights --
    for (j <- 0 until inVecLenOut) {
      val mask = Mux(directUpper, upperMaskOut(j), lowerMaskOut(j))
      when(mask) {
        val sign = (1.S(2.W) * wopOut(j))   // outOp=+1, so sign = wopOut(j)
        val t1 = Fx.mul(alpha, s3_xOut(j), bp, mw)
        val t2 = Fx.mul(t1, gradOut, bp, mw)
        val t3 = Fx.mul(t2, diffAbs, bp, mw)
        val delta = (t3 * sign)(w - 1, 0).asSInt   // sign is +/-1, keep lower w bits
        wOut(j) := wOut(j) + delta
      }
    }

    // -- Hidden weights --
    for (h <- 0 until hiddenNum) {
      val hOp = hiddenOperator(h)
      for (j <- 0 until inVecLenHidden) {
        val mask = Mux(directUpper, upperMaskHidden(j), lowerMaskHidden(j))
        when(mask) {
          val sign = (hOp * wopHidden(j))
          val t1 = Fx.mul(alpha, s0_xHidden(j), bp, mw)
          val t2 = Fx.mul(t1, gradHidden(h), bp, mw)
          val t3 = Fx.mul(t2, diffAbs, bp, mw)
          val delta = (t3 * sign)(w - 1, 0).asSInt
          wHidden(h)(j) := wHidden(h)(j) + delta
        }
      }
    }
  }
}
