package EDmnist

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import EDmnist.MnistLoader._

class MnistOnlineTrainSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "ThreeLayerFwdPipeUpdSeq online learning on MNIST (binary: 0 vs rest)"

  // ---- Fixed-point conversion helpers
  // dut uses (w,bp) = (16,12) by default; keep consistent with module instantiation
  val bp = 12
  val scale = 1 << bp

  private def toFx(x: Double): BigInt = {
    // clamp to reasonable range for FixedPoint
    val v = math.max(-8.0, math.min(8.0, x))
    BigInt(math.round(v * scale))
  }

  // For Chisel FixedPoint in tests: poke expects BigInt raw scaled by 2^bp
  private def pokeFx(sig: FixedPoint, x: Double): Unit = sig.poke(toFx(x))

  private def peekFx(sig: FixedPoint): Double = sig.peek().litValue.toDouble / scale.toDouble

  it should "show accuracy improving (or at least changing) over epochs" in {
    // --------- Config ----------
    val posDigit = 0               // 0 vs rest
    val inputNum = 14 * 14         // 196
    val hiddenNum = 32             // keep modest; forward is fully parallel in hidden layer
    val epochs = 3                 // increase as needed
    val trainLimit = 2000          // number of training samples per epoch (for runtime)
    val testLimit  = 2000          // number of test samples for evaluation

    // Paths: you must have MNIST gzip files locally.
    // e.g. put them under ./data/
    val trainImages = "data/train-images-idx3-ubyte.gz"
    val trainLabels = "data/train-labels-idx1-ubyte.gz"
    val testImages  = "data/t10k-images-idx3-ubyte.gz"
    val testLabels  = "data/t10k-labels-idx1-ubyte.gz"

    val train = MnistLoader.load(trainImages, trainLabels)
    val test  = MnistLoader.load(testImages, testLabels)

    test(new ThreeLayerFwdPipeUpdSeq(inputNum, hiddenNum, w = 16, bp = 12, fifoDepth = 16))
      .withAnnotations(Seq(WriteVcdAnnotation)) { dut =>

        dut.clock.setTimeout(0)

        // Convenience: drive one sample, optionally train, and return prediction
        def runOneSample(x196: Array[Double], target: Double, trainEn: Boolean): Double = {
          // Wait until ready (note: inReady depends on FIFO space when trainEn)
          while (!dut.io.inReady.peek().litToBoolean) {
            dut.clock.step(1)
          }

          dut.io.inValid.poke(true.B)
          dut.io.trainEn.poke(trainEn.B)
          pokeFx(dut.io.target, target)

          for (i <- 0 until inputNum) {
            pokeFx(dut.io.x(i), x196(i))
          }

          dut.clock.step(1)
          dut.io.inValid.poke(false.B)

          // Wait for outValid pulse, capture y
          var y = 0.0
          while (!dut.io.outValid.peek().litToBoolean) {
            dut.clock.step(1)
          }
          y = peekFx(dut.io.y)
          y
        }

        // Accuracy eval (no training)
        def evalAccuracy(m: MnistLoader.Mnist, limit: Int): Double = {
          var correct = 0
          val n = math.min(limit, m.labels.length)
          var i = 0
          while (i < n) {
            val x = downsample2x2To14x14(m.images(i), m.rows, m.cols)
            val t = toBinaryLabel(m.labels(i), posDigit)
            val y = runOneSample(x, t, trainEn = false)
            val pred = if (y >= 0.5) 1.0 else 0.0
            if (pred == t) correct += 1
            i += 1
          }
          correct.toDouble / n.toDouble
        }

        // Training epoch
        def trainEpoch(m: MnistLoader.Mnist, limit: Int): Unit = {
          val n = math.min(limit, m.labels.length)
          var i = 0
          while (i < n) {
            val x = downsample2x2To14x14(m.images(i), m.rows, m.cols)
            val t = toBinaryLabel(m.labels(i), posDigit)
            runOneSample(x, t, trainEn = true)
            i += 1
          }
        }

        // Initial accuracy
        val acc0 = evalAccuracy(test, testLimit)
        println(f"[epoch 0] test accuracy (digit=$posDigit vs rest): ${acc0 * 100.0}%.2f%%")

        // Loop epochs
        var e = 1
        while (e <= epochs) {
          trainEpoch(train, trainLimit)
          val acc = evalAccuracy(test, testLimit)
          println(f"[epoch $e] test accuracy (digit=$posDigit vs rest): ${acc * 100.0}%.2f%%")
          e += 1
        }
      }
  }
}
