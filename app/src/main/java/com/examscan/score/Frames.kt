package com.examscan.score

import androidx.camera.core.ImageProxy
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer

/**
 * ImageProxy → OpenCV Mat 的统一转换，统计页和学习页共用一份，避免两套旋转逻辑跑偏。
 */
object Frames {

    /** RGBA_8888 转 BGR，不改朝向 */
    fun toBgr(image: ImageProxy): Mat? {
        val plane = image.planes.firstOrNull() ?: return null
        val w = image.width
        val h = image.height
        if (plane.pixelStride != 4) return null

        val rowStride = plane.rowStride
        val buffer: ByteBuffer = if (rowStride == w * 4) {
            plane.buffer
        } else {
            // 有些机型每行有对齐填充，先拷成紧凑的
            val packed = ByteBuffer.allocateDirect(w * h * 4)
            val src = plane.buffer.duplicate()
            for (row in 0 until h) {
                val start = row * rowStride
                src.position(start)
                src.limit(minOf(start + w * 4, src.capacity()))
                packed.put(src)
            }
            packed.rewind()
            packed
        }

        val rgba = Mat(h, w, CvType.CV_8UC4, buffer)
        val bgr = Mat()
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        rgba.release()
        return bgr
    }

    /** 转成「手机竖着看」的 BGR，取景框和框标注都用这个坐标系 */
    fun upright(image: ImageProxy): Mat? {
        val bgr = toBgr(image) ?: return null
        val dst = Mat()
        when (image.imageInfo.rotationDegrees) {
            90 -> Core.rotate(bgr, dst, Core.ROTATE_90_CLOCKWISE)
            180 -> Core.rotate(bgr, dst, Core.ROTATE_180)
            270 -> Core.rotate(bgr, dst, Core.ROTATE_90_COUNTERCLOCKWISE)
            else -> bgr.copyTo(dst)
        }
        bgr.release()
        return dst
    }
}
