package EDmnist

import java.io.{DataInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

object MnistLoader {

  final case class Mnist(images: Array[Array[Byte]], labels: Array[Byte], rows: Int, cols: Int)

  private def readInt(dis: DataInputStream): Int = dis.readInt()

  def load(imagesGz: String, labelsGz: String): Mnist = {
    val imgDis = new DataInputStream(new GZIPInputStream(new FileInputStream(imagesGz)))
    val labDis = new DataInputStream(new GZIPInputStream(new FileInputStream(labelsGz)))

    val imgMagic = readInt(imgDis)
    val numImages = readInt(imgDis)
    val rows = readInt(imgDis)
    val cols = readInt(imgDis)

    val labMagic = readInt(labDis)
    val numLabels = readInt(labDis)

    require(imgMagic == 0x00000803, f"Bad image magic: 0x$imgMagic%08x")
    require(labMagic == 0x00000801, f"Bad label magic: 0x$labMagic%08x")
    require(numImages == numLabels, s"numImages($numImages) != numLabels($numLabels)")

    val images = Array.ofDim[Array[Byte]](numImages)
    val labels = Array.ofDim[Byte](numLabels)

    val imgSize = rows * cols
    for (i <- 0 until numImages) {
      val buf = Array.ofDim[Byte](imgSize)
      imgDis.readFully(buf)
      images(i) = buf
    }

    labDis.readFully(labels)

    imgDis.close()
    labDis.close()

    Mnist(images, labels, rows, cols)
  }

  /**
   * 28x28 -> 14x14 (2x2 average pooling), returns Double array length 196 in [0,1]
   */
  def downsample2x2To14x14(img: Array[Byte], rows: Int, cols: Int): Array[Double] = {
    require(rows == 28 && cols == 28, s"Expected 28x28, got ${rows}x${cols}")
    val out = Array.ofDim[Double](14 * 14)
    var oy = 0
    while (oy < 14) {
      var ox = 0
      while (ox < 14) {
        val y = oy * 2
        val x = ox * 2
        val p00 = img((y + 0) * 28 + (x + 0)) & 0xff
        val p01 = img((y + 0) * 28 + (x + 1)) & 0xff
        val p10 = img((y + 1) * 28 + (x + 0)) & 0xff
        val p11 = img((y + 1) * 28 + (x + 1)) & 0xff
        val avg = (p00 + p01 + p10 + p11).toDouble / 4.0
        out(oy * 14 + ox) = avg / 255.0
        ox += 1
      }
      oy += 1
    }
    out
  }

  /** binary target: digit==posDigit => 1.0 else 0.0 */
  def toBinaryLabel(digit: Byte, posDigit: Int): Double =
    if ((digit & 0xff) == posDigit) 1.0 else 0.0
}
