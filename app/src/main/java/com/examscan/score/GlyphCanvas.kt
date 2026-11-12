package com.examscan.score

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * 把掩膜里的单个字形抠出来，归一化成 MNIST 规范的 28x28。
 *
 * 识别（DigitRecognizer）和学习（Calibrator）都用它，保证模板和推理输入是同一套坐标。
 */
object GlyphCanvas {

    fun from(mask: Mat, box: Rect, pad: Int = 3): Mat? {
        val x0 = maxOf(0, box.x - pad)
        val y0 = maxOf(0, box.y - pad)
        val x1 = minOf(mask.cols(), box.x + box.width + pad)
        val y1 = minOf(mask.rows(), box.y + box.height + pad)
        if (x1 - x0 < 2 || y1 - y0 < 2) return null

        val sub = mask.submat(Rect(x0, y0, x1 - x0, y1 - y0)).clone()
        val pts = Mat()
        Core.findNonZero(sub, pts)
        if (pts.empty()) {
            pts.release()
            sub.release()
            return null
        }
        val bnd = Imgproc.boundingRect(pts)
        pts.release()
        val glyph = sub.submat(bnd).clone()
        sub.release()

        // 只在极端实心块（粗记号笔、涂改）时做一次细化，正常笔宽加腐蚀反而掉准确率
        val density = Core.countNonZero(glyph).toDouble() / (glyph.rows() * glyph.cols())
        if (density > 0.60) {
            val k = Mat.ones(2, 2, CvType.CV_8U)
            Imgproc.erode(glyph, glyph, k)
            k.release()
        }

        val hh = glyph.rows()
        val ww = glyph.cols()
        val scale = 20.0 / maxOf(hh, ww)
        val nw = maxOf(1, Math.round(ww * scale).toInt())
        val nh = maxOf(1, Math.round(hh * scale).toInt())
        val small = Mat()
        Imgproc.resize(glyph, small, Size(nw.toDouble(), nh.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        glyph.release()

        val canvas = Mat.zeros(28, 28, CvType.CV_8U)
        val ox = (28 - nw) / 2
        val oy = (28 - nh) / 2
        small.copyTo(canvas.submat(Rect(ox, oy, nw, nh)))
        small.release()

        val m = Imgproc.moments(canvas, true)
        if (m.m00 > 0) {
            val cx = m.m10 / m.m00
            val cy = m.m01 / m.m00
            val shift = Mat(2, 3, CvType.CV_32F)
            shift.put(0, 0, 1.0, 0.0, 13.5 - cx, 0.0, 1.0, 13.5 - cy)
            val dst = Mat()
            Imgproc.warpAffine(canvas, dst, shift, Size(28.0, 28.0),
                Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(0.0))
            canvas.release()
            shift.release()
            return dst
        }
        return canvas
    }

    /** 模板比对用：把 28x28 canvas 摊平成 0..255 的字节数组 */
    fun toBytes(canvas: Mat): ByteArray {
        val bytes = ByteArray(28 * 28)
        canvas.get(0, 0, bytes)
        return bytes
    }
}
