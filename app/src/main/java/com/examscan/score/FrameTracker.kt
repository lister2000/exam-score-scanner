package com.examscan.score

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * 用相位相关估计画面位移，把帧坐标换算成稳定的"页面坐标"。
 *
 * 这是去重能成立的前提：手机在移动，同一数字在不同帧里的坐标会漂移，
 * 只有映射到页面坐标系，才能判断"这是不是同一个位置"。
 */
class FrameTracker {

    private var prev: Mat? = null
    private var cumX = 0.0
    private var cumY = 0.0

    /** 返回归一化的累计偏移（相对当前画面宽高）与响应值 */
    data class Offset(val x: Float, val y: Float, val response: Double)

    fun update(gray: Mat): Offset {
        val small = Mat()
        Imgproc.resize(gray, small, Size(320.0, 240.0), 0.0, 0.0, Imgproc.INTER_AREA)
        val f = Mat()
        small.convertTo(f, CvType.CV_32F)
        small.release()

        val p = prev
        if (p == null) {
            prev = f
            return Offset(0f, 0f, 1.0)
        }
        var response = 0.0
        try {
            // OpenCV Java 的 response 是「出参数组」，不是 Mat
            val out = doubleArrayOf(0.0)
            val pt = Imgproc.phaseCorrelate(p, f, Mat(), out)
            response = out[0]
            if (response > 0.2) {
                cumX += pt.x * (gray.cols() / 320.0)
                cumY += pt.y * (gray.rows() / 240.0)
            }
        } catch (e: Throwable) {
            // 某些机型上相位相关不可用，退化为"不做位移估计"，不影响单帧识别
            response = 0.0
        }
        p.release()
        prev = f
        return Offset((cumX / gray.cols()).toFloat(), (cumY / gray.rows()).toFloat(), response)
    }

    fun reset() {
        prev?.release()
        prev = null
        cumX = 0.0
        cumY = 0.0
    }
}
