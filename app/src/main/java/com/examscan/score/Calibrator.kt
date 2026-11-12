package com.examscan.score

import android.util.Log
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File

/**
 * 标定：从一张白纸上「随手写的若干数字」里，量出「这支笔的真实颜色」和「这个人的字形模板」。
 *
 * 不假设个数、顺序和排列：拍到几个字形就是几个，每个字形先用 CNN 猜一个值，
 * 再由用户在界面上校正（顺序不对、个数不齐都能用）。
 *
 * 为什么必须标定：真机相机的白平衡会把品红压成 BGR(150,120,165)，
 * 只按 R 通道判断红色的阈值（redness>26）会几乎全滤掉笔画；这里改用实测的颜色分布。
 */
object Calibrator {

    private const val TAG = "Calibrator"

    /** 一个字形：位置 + CNN 初值 + 归一化模板（模板带着标号一起存，才不依赖书写顺序） */
    data class Item(
        val box: Rect,
        val digit: Int,
        val conf: Float,
        val template: ByteArray
    )

    data class Result(
        /** 颜色统计出来的档案（templates 由界面按用户标号组装） */
        val calib: Calib,
        /** 检测到并成功识别的字形，按「从上到下、从左到右」排序 */
        val items: List<Item>,
        /** 检测到的连通块总数（可能多于 items，识别失败的不在其中） */
        val detected: Int,
        val strokePixels: Int,
        /** 标定所用图像的尺寸，界面用它把框坐标归一化 */
        val sizeW: Int,
        val sizeH: Int
    )

    /**
     * 实时预览用：只切框、不跑 CNN，一帧几十毫秒，能跟着镜头连着出框。
     * 返回的框已经过滤掉纸面噪点，尺寸就是笔迹本身的大小。
     */
    data class Live(val boxes: List<Rect>, val w: Int, val h: Int)

    /** inkMask 的产物：掩膜 + 两张「比周围更饱和 / 更暗」的差值图，供逐框做颜色一致性校验 */
    class Ink(val mask: Mat, val satDiff: Mat, val darkDiff: Mat) {
        fun release() {
            mask.release()
            satDiff.release()
            darkDiff.release()
        }
    }

    fun detect(img: Mat, dumpDir: File? = null): Live {
        val prepared = ScorePipeline.prepare(img)
        try {
            val w = prepared.cols()
            val h = prepared.rows()
            val ink = inkMask(prepared)
            val mask = ink.mask
            try {
                // 实时现场落盘：出问题时可以直接 pull 这两张图复盘，不必再拍一次
                if (dumpDir != null) {
                    dump(dumpDir, "live_frame.jpg", img)
                    dump(dumpDir, "live_mask.png", mask)
                }
                val merged = ScorePipeline.mergeGlyphs(ScorePipeline.components(mask, relax = true))
                val minArea = 0.00025 * w * h
                val margin = 0.02 * minOf(w, h)
                val dist = Mat()
                Imgproc.distanceTransform(mask, dist, Imgproc.DIST_L2, 5)
                val boxes = merged
                    .filter { b ->
                        val geo = b.height >= 0.015 * h && b.width >= 0.01 * w &&
                            b.width * b.height >= minArea
                        if (!geo) return@filter false
                        // 贴着画面外缘的多半是纸边、桌面交界
                        if (b.x < margin || b.y < margin ||
                            b.x + b.width > w - margin || b.y + b.height > h - margin
                        ) return@filter false
                        // 笔画是细结构：框内最大内切圆半径不该超过字高的 30%。
                        // 大片阴影、涂块、纸边是宽结构，半径会远超这个比例，靠它筛掉。
                        val sub = dist.submat(b)
                        val maxR = Core.minMaxLoc(sub).maxVal
                        sub.release()
                        if (maxR > 0.30 * b.height) return@filter false
                        // 颜色一致性：框里的像素必须真的「比周围更艳」或「更暗」。
                        // 中性灰的纸边/桌面交界两条都不满足（实测 9.8 / 5.9，而笔迹 22~50 / 30+）
                        val mk = mask.submat(b)
                        val sd = ink.satDiff.submat(b)
                        val dd = ink.darkDiff.submat(b)
                        val sat = Core.mean(sd, mk).`val`[0]
                        val dark = Core.mean(dd, mk).`val`[0]
                        mk.release()
                        sd.release()
                        dd.release()
                        sat >= 15.0 || dark >= 30.0
                    }
                    .sortedWith(compareBy({ it.y }, { it.x }))
                dist.release()
                return Live(boxes, w, h)
            } finally {
                ink.release()
            }
        } finally {
            prepared.release()
        }
    }

    /**
     * 输入一张 BGR 图（白纸上随意写了 N 个数字），返回颜色档案 + N 个字形。
     * 模板的最终标号由调用方确认后组装成 [Calib.templates]。
     */
    fun calibrate(
        img: Mat,
        rec: DigitRecognizer,
        dumpDir: File? = null,
        templates: Map<Int, List<ByteArray>> = emptyMap()
    ): Result {
        val prepared = ScorePipeline.prepare(img)
        val sizeW = prepared.cols()
        val sizeH = prepared.rows()

        val diag = StringBuilder()
        val ink = inkMask(prepared)
        val mask = ink.mask
        val merged = ScorePipeline.mergeGlyphs(ScorePipeline.components(mask, relax = true, diag = diag))
        // 只保留「像字」的块：太小的是纸面纹理/阴影边缘，空白处不该出框
        val minArea = 0.00025 * sizeW * sizeH
        val boxes = merged
            .filter { it.height >= 0.015 * sizeH && it.width >= 0.01 * sizeW && it.width * it.height >= minArea }
            .sortedWith(compareBy({ it.y }, { it.x }))
        diag.appendLine("掩膜像素=${Core.countNonZero(mask)} 合并后=${merged.size} 保留=${boxes.size}")
        dump(dumpDir, "mask_ink.png", mask)
        dumpText(dumpDir, "diag.txt", diag.toString())
        if (boxes.isEmpty()) {
            Log.w(TAG, "没切出像字的块，尺寸 ${sizeW}x$sizeH")
            ink.release()
            prepared.release()
            return Result(Calib.DEFAULT, emptyList(), 0, 0, sizeW, sizeH)
        }

        try {
            val (base, strokePixels) = colorStats(prepared, mask)

            val items = boxes.mapNotNull { b ->
                val canvas = GlyphCanvas.from(mask, b) ?: return@mapNotNull null
                val bytes = GlyphCanvas.toBytes(canvas)
                canvas.release()
                val (digit, conf) = rec.recognize(mask, b, templates)
                if (digit == null) null else Item(b, digit, conf, bytes)
            }
            Log.i(TAG, "标定结果：${sizeW}x$sizeH 框 ${boxes.size} 个，识别 ${items.size} 个，笔画采样 $strokePixels")
            return Result(base, items, boxes.size, strokePixels, sizeW, sizeH)
        } finally {
            ink.release()
            prepared.release()
        }
    }

    /**
     * 学习阶段的笔迹掩膜：不比绝对颜色，只比「这里和周围纸面差多少」。
     *
     * 空白纸上的阴影、反光是低频的，用大半径高斯抹平后相减就抵消了；
     * 而笔迹一定比周围更饱和（彩色笔）、更暗（黑/蓝笔）或更红（红/品红笔）。
     * 这样既不会把空白处当选区，也不挑笔的颜色。
     */
    fun inkMask(img: Mat): Ink {
        val sigma = Math.max(6.0, 0.03 * Math.min(img.cols(), img.rows()))
        val hsv = Mat()
        Imgproc.cvtColor(img, hsv, Imgproc.COLOR_BGR2HSV)
        val hsvCh = ArrayList<Mat>()
        Core.split(hsv, hsvCh)
        val sF = Mat()
        val vF = Mat()
        hsvCh[1].convertTo(sF, CvType.CV_32F)
        hsvCh[2].convertTo(vF, CvType.CV_32F)
        val lS = Mat()
        val lV = Mat()
        Imgproc.GaussianBlur(sF, lS, Size(0.0, 0.0), sigma)
        Imgproc.GaussianBlur(vF, lV, Size(0.0, 0.0), sigma)
        val satDiff = Mat()
        val darkDiff = Mat()
        Core.subtract(sF, lS, satDiff)   // 比周围更「有颜色」
        Core.subtract(lV, vF, darkDiff)  // 比周围更暗
        val satInk = Mat()
        val darkInk = Mat()
        Core.compare(satDiff, Scalar(10.0), satInk, Core.CMP_GT)
        Core.compare(darkDiff, Scalar(25.0), darkInk, Core.CMP_GT)

        // 红/品红笔：R 明显高于 G/B
        val bgrCh = ArrayList<Mat>()
        Core.split(img, bgrCh)
        val b16 = Mat()
        val g16 = Mat()
        val r16 = Mat()
        bgrCh[0].convertTo(b16, CvType.CV_16S)
        bgrCh[1].convertTo(g16, CvType.CV_16S)
        bgrCh[2].convertTo(r16, CvType.CV_16S)
        val gb = Mat()
        Core.max(g16, b16, gb)
        val redness = Mat()
        Core.subtract(r16, gb, redness)
        val r8 = Mat()
        redness.convertTo(r8, CvType.CV_8U)
        val redInk = Mat()
        Core.compare(r8, Scalar(12.0), redInk, Core.CMP_GT)

        val mask = Mat()
        Core.bitwise_or(satInk, darkInk, mask)
        Core.bitwise_or(mask, redInk, mask)
        val k = Mat.ones(3, 3, CvType.CV_8U)
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, k, Point(-1.0, -1.0), 1)
        k.release()

        for (m in hsvCh) m.release()
        for (m in bgrCh) m.release()
        listOf(hsv, sF, vF, lS, lV, satInk, darkInk,
            b16, g16, r16, gb, redness, r8, redInk).forEach { it.release() }
        return Ink(mask, satDiff, darkDiff)
    }

    /** 统计笔画像素的色相/饱和度/亮度/通道差，生成阈值；顺带返回采样点数 */
    /** 从 (img, mask) 采样算颜色分位数并组装成 Calib，供增量学习时合并 */
    fun colorStats(img: Mat, mask: Mat): Pair<Calib, Int> {
        val hsv = Mat()
        Imgproc.cvtColor(img, hsv, Imgproc.COLOR_BGR2HSV)
        val ch = ArrayList<Mat>()
        Core.split(img, ch)
        val pts = Mat()
        Core.findNonZero(mask, pts)

        val n = pts.rows()
        val hues = ArrayList<Int>()
        val sats = ArrayList<Int>()
        val vals = ArrayList<Int>()
        val reds = ArrayList<Int>()
        val step = maxOf(1, n / 4000)   // 最多采样 4000 个像素，够统计也够快
        var i = 0
        while (i < n) {
            val p = pts.get(i, 0)
            if (p != null && p.size >= 2) {
                val x = p[0].toInt()
                val y = p[1].toInt()
                if (x in 0 until img.cols() && y in 0 until img.rows()) {
                    val h = hsv.get(y, x)
                    val b = ch[0].get(y, x)
                    val g = ch[1].get(y, x)
                    val r = ch[2].get(y, x)
                    if (h != null && b != null && g != null && r != null) {
                        hues.add(h[0].toInt())
                        sats.add(h[1].toInt())
                        vals.add(h[2].toInt())
                        reds.add(r[0].toInt() - maxOf(g[0], b[0]).toInt())
                    }
                }
            }
            i += step
        }
        pts.release()
        for (m in ch) m.release()
        hsv.release()

        if (hues.isEmpty()) return Pair(Calib.DEFAULT, 0)

        // 色相是环形的（0 和 180 相邻），用「离中位数的环形距离」算容差
        val med = pct(hues, 50)
        val deltas = ArrayList<Int>(hues.size)
        for (h in hues) {
            val d = Math.abs(h - med)
            deltas.add(minOf(d, 180 - d))
        }
        val tol = pct(deltas, 95).coerceIn(6, 28)

        return Pair(Calib(
            hueRanges = hueRanges(med, tol),
            satMin = (pct(sats, 10) - 5).coerceIn(12, 70),
            valMin = (pct(vals, 5) - 10).coerceIn(20, 120),
            valMax = (pct(vals, 98) + 10).coerceIn(180, 255),
            minRedness = (pct(reds, 5) - 2).coerceIn(0, 25),
            templates = emptyMap()
        ), hues.size)
    }

    /** [center-tol, center+tol] 跨越 0/180 时拆成两段 */
    private fun hueRanges(center: Int, tol: Int): List<IntArray> {
        val lo = center - tol
        val hi = center + tol
        return when {
            lo < 0 && hi > 180 -> listOf(intArrayOf(0, 180))
            lo < 0 -> listOf(intArrayOf(0, hi), intArrayOf(180 + lo, 180))
            hi > 180 -> listOf(intArrayOf(lo, 180), intArrayOf(0, hi - 180))
            else -> listOf(intArrayOf(lo, hi))
        }
    }

    private fun pct(src: List<Int>, q: Int): Int {
        val s = src.sorted()
        return s[((s.size - 1) * q / 100).coerceIn(0, s.size - 1)]
    }

    private fun dump(dir: File?, name: String, mat: Mat) {
        if (dir == null) return
        try {
            Imgcodecs.imwrite(File(dir, name).absolutePath, mat)
        } catch (e: Throwable) {
            Log.w(TAG, "落盘失败 $name", e)
        }
    }

    private fun dumpText(dir: File?, name: String, text: String) {
        if (dir == null) return
        try {
            File(dir, name).writeText(text, Charsets.UTF_8)
        } catch (e: Throwable) {
            Log.w(TAG, "落盘失败 $name", e)
        }
    }
}
