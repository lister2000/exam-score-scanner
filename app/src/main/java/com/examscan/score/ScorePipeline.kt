package com.examscan.score

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * 红笔分数检测流水线（移植自 server/pipeline.py）
 *
 * 颜色分割 -> 连通域 -> 字形归并 -> 干扰过滤 -> 数字分组。
 * 只用红色像素参与识别，黑色印刷体、蓝色笔迹天然被排除；
 * √ / × / 圈画 / 下划线 / 印章靠几何规则剔除，不依赖模型。
 */
object ScorePipeline {

    const val MAX_SIDE = 1080
    const val MIN_CONF = 0.45f

    // 实时跟踪框的大小过滤（与 server/pipeline.py 的 track 保持一致）
    const val TRACK_MIN_H_RATIO = 0.012   // 字高不足画面 1.2% 的框太小，CNN 也认不出
    const val TRACK_TYPICAL_RATIO = 0.28  // 超过画面高度 28% 的框多半是圈画/印章，不参与统计
    const val TRACK_SMALL_RATIO = 0.5     // 多个框时，比典型字高小一半以上的当碎片丢掉
    const val TRACK_MIN_INK = 40          // 框里红像素少于这个数：污点、笔尖蹭的点

    /**
     * 识别规则参数（设置页可调）。
     *
     * · maxDigits：一个多位数最多几位（1 = 不合并，每个数字单独算）
     * · mergeGapRatio：相邻数字间距 <= 该倍数 × 数字宽度 就并成一个数（1.0 = 一个数字宽度）
     * · minConf：置信度下限，低于它直接丢弃
     * · rowOverlap：同一行的竖直重叠要求（占较矮框高度的比例）
     * · sizeSimilar：字号相似上限（两框高度差超过这个倍数就不合并）
     */
    data class Tuning(
        val maxDigits: Int = 4,
        val mergeGapRatio: Double = 0.8,
        val minConf: Float = MIN_CONF,
        val rowOverlap: Double = 0.5,
        val sizeSimilar: Double = 1.8,
        val negEnabled: Boolean = true,
        val negGapRatio: Double = 0.7
    ) {
        companion object {
            val DEFAULT = Tuning()
        }
    }

    /** 检测结果，坐标已归一化到当前画面（0~1），方便叠加层直接画 */
    data class Detected(val value: Int, val conf: Float, val nx: Float, val ny: Float, val nw: Float, val nh: Float)

    private class Glyph(val box: Rect, val digit: Int, val conf: Float, val neg: Boolean = false)

    /** 统一尺度：最长边不超过 MAX_SIDE，调用方负责 release */
    fun prepare(img: Mat): Mat {
        val longest = maxOf(img.cols(), img.rows())
        if (longest <= MAX_SIDE) return img.clone()
        val s = MAX_SIDE.toDouble() / longest
        val dst = Mat()
        Imgproc.resize(img, dst, Size(img.cols() * s, img.rows() * s), 0.0, 0.0, Imgproc.INTER_AREA)
        return dst
    }

    fun redMask(img: Mat, calib: Calib = Calib.DEFAULT): Mat {
        val hsv = Mat()
        Imgproc.cvtColor(img, hsv, Imgproc.COLOR_BGR2HSV)
        val mask = Mat.zeros(hsv.size(), CvType.CV_8U)
        val tmp = Mat()
        for (r in calib.hueRanges) {
            Core.inRange(
                hsv,
                Scalar(r[0].toDouble(), calib.satMin.toDouble(), calib.valMin.toDouble()),
                Scalar(r[1].toDouble(), 255.0, calib.valMax.toDouble()),
                tmp
            )
            Core.bitwise_or(mask, tmp, mask)
        }
        tmp.release()
        hsv.release()

        // 通道差兜底：R 明显高于 G/B 即为红色，能兜住浅红、暗红红笔，白纸黑字天然为 0
        val ch = ArrayList<Mat>()
        Core.split(img, ch)
        val b16 = Mat()
        val g16 = Mat()
        val r16 = Mat()
        ch[0].convertTo(b16, CvType.CV_16S)
        ch[1].convertTo(g16, CvType.CV_16S)
        ch[2].convertTo(r16, CvType.CV_16S)
        val gb = Mat()
        Core.max(g16, b16, gb)
        val redness = Mat()
        Core.subtract(r16, gb, redness)
        val r8 = Mat()
        redness.convertTo(r8, CvType.CV_8U)
        val m3 = Mat()
        Core.compare(r8, Scalar(calib.minRedness.toDouble()), m3, Core.CMP_GT)
        Core.bitwise_or(mask, m3, mask)

        for (m in ch) m.release()
        b16.release(); g16.release(); r16.release()
        gb.release(); redness.release(); r8.release(); m3.release()

        val k = Mat.ones(3, 3, CvType.CV_8U)
        // 只做一次闭运算：迭代多了会把 8/6 这类带环数字的封闭区域填死
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, k, Point(-1.0, -1.0), 1)
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, k, Point(-1.0, -1.0), 1)
        k.release()
        return mask
    }

    fun components(
        mask: Mat,
        relax: Boolean = false,
        diag: StringBuilder? = null,
        lines: MutableList<Rect>? = null
    ): MutableList<Rect> {
        val h = mask.rows()
        val w = mask.cols()
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        val n = Imgproc.connectedComponentsWithStats(mask, labels, stats, centroids, 8, CvType.CV_32S)
        val out = mutableListOf<Rect>()
        // 学习阶段不用太严：字写得随意也要能切出来
        val minH = if (relax) 0.006 else 0.008
        val minW = if (relax) 0.0015 else 0.002
        val minArea = if (relax) 8 else 10
        val maxFill = if (relax) 0.92 else 0.88
        val minFill = if (relax) 0.03 else 0.05
        val maxAr = if (relax) 6.0 else 4.0
        // 注意：不能用 stats.get(i, 0)（整数矩阵上会拿到 null），必须按 int[] 读一行
        val row = IntArray(5)
        diag?.appendLine("cc: ${w}x$h 连通块 $n 个")
        for (i in 1 until n) {
            stats.get(i, 0, row)
            val x = row[0]
            val y = row[1]
            val bw = row[2]
            val bh = row[3]
            val area = row[4]
            val fill = if (bw > 0 && bh > 0) area / (bw.toDouble() * bh) else 0.0
            val ar = if (bh > 0) bw.toDouble() / bh else 0.0
            // 横杠（负号）先单独收走：它们不参与字形识别，但要留给「负号 + 数字」配对
            if (lines != null && ar >= 2.5 && bh <= 0.10 * h && bw <= 0.12 * w &&
                fill >= 0.3 && area >= 10
            ) {
                lines.add(Rect(x, y, bw, bh))
                continue
            }
            val why = when {
                bh < minH * h -> "高度太小"
                bh > 0.45 * h -> "高度太大"
                bw < minW * w -> "宽度太小"
                bw > 0.45 * w -> "宽度太大"
                area < minArea -> "面积太小"
                x <= 1 || y <= 1 || x + bw >= w - 1 || y + bh >= h - 1 -> "贴边"
                fill < minFill -> "过稀"
                fill > maxFill -> "过实"
                ar > maxAr || ar < 0.08 -> "横竖线"
                else -> "OK"
            }
            diag?.appendLine(
                "cc#$i x=$x y=$y w=$bw h=$bh area=$area fill=${"%.2f".format(fill)} " +
                    "ar=${"%.2f".format(ar)} -> $why"
            )
            if (why != "OK") continue
            out.add(Rect(x, y, bw, bh))
        }
        labels.release()
        stats.release()
        centroids.release()
        return out
    }

    private fun union(a: Rect, b: Rect): Rect {
        val x1 = minOf(a.x, b.x)
        val y1 = minOf(a.y, b.y)
        val x2 = maxOf(a.x + a.width, b.x + b.width)
        val y2 = maxOf(a.y + a.height, b.y + b.height)
        return Rect(x1, y1, x2 - x1, y2 - y1)
    }

    /** 把同一个字的多个笔画片段合并成一个字形框（例如 '4' 的两笔） */
    fun mergeGlyphs(boxes: MutableList<Rect>, gapRatio: Double = 0.30, minVOverlap: Double = 0.5): MutableList<Rect> {
        var merged = true
        while (merged) {
            merged = false
            val n = boxes.size
            for (i in 0 until n) {
                for (j in i + 1 until n) {
                    val a = boxes[i]
                    val b = boxes[j]
                    val gap = maxOf(b.x - (a.x + a.width), a.x - (b.x + b.width), 0)
                    val vGap = maxOf(b.y - (a.y + a.height), a.y - (b.y + b.height), 0)
                    val ov = minOf(a.y + a.height, b.y + b.height) - maxOf(a.y, b.y)
                    val height = maxOf(a.height, b.height)
                    // 同一个字被切成的两块往往不是整齐的左右/上下，而是「斜着几乎贴住」：
                    // 例如手写的 3 断成上下两段，横竖两个方向都只差 1~3 像素。这种情况直接并。
                    val near = 0.06 * height
                    val touching = gap <= near && vGap <= near
                    val sideBySide = ov > 0 && gap <= gapRatio * height &&
                        ov >= minVOverlap * minOf(a.height, b.height)
                    // 单个数字的宽高比一般不超过 1.05；并完之后宽成这样，说明是两个挨着的数字，
                    // 不能再并（否则 CNN 会把「78」当成一个字形，置信度低直接被丢掉）
                    val u = union(a, b)
                    val keepDigitShape = u.width <= 1.05 * height
                    // 位于另一块腰部高度的细横杠是负号，不能并进字形（否则整个字形识别失败）
                    val aDash = isDash(a, b)
                    val bDash = isDash(b, a)
                    if ((touching || sideBySide) && keepDigitShape && !aDash && !bDash) {
                        boxes[i] = u
                        boxes.removeAt(j)
                        merged = true
                        break
                    }
                }
                if (merged) break
            }
        }
        return boxes
    }

    /** dash 是不是位于 other 腰部高度的一根细横杠（负号） */
    private fun isDash(dash: Rect, other: Rect): Boolean {
        if (other.height <= 0 || dash.height <= 0) return false
        if (dash.height > 0.28 * other.height) return false
        if (dash.width < 2.0 * dash.height) return false
        val cy = dash.y + dash.height / 2.0
        return cy >= other.y + 0.25 * other.height && cy <= other.y + 0.75 * other.height
    }

    /** 数字框 b 左侧是否有一根负号，有则这个数字取负 */
    private fun minusForDigit(lines: List<Rect>, b: Rect, tuning: Tuning): Boolean {
        if (!tuning.negEnabled || b.height <= 0) return false
        val h = b.height
        for (m in lines) {
            val gap = b.x - (m.x + m.width)
            val cy = m.y + m.height / 2.0
            val leftOk = gap >= -0.2 * h && gap <= tuning.negGapRatio * h
            val sizeOk = m.width >= 0.12 * h && m.width <= 0.9 * h && m.height <= 0.25 * h
            val bandOk = cy >= b.y + 0.2 * h && cy <= b.y + 0.8 * h
            if (leftOk && sizeOk && bandOk) return true
        }
        return false
    }

    /**
     * 宽框拆分：两位数字写得真正粘连时，连通域会合成一个宽框，
     * 直接送 CNN 只会识别失败。这里用竖直投影找笔画最少的列（数字间的缝），
     * 从中间向两边切；找不到明显低谷就原样返回。
     */
    private fun splitWide(mask: Mat, box: Rect, maxParts: Int): List<Rect> {
        val parts = mutableListOf(box)
        while (parts.size < maxParts) {
            var done = false
            for (i in parts.indices) {
                val r = parts[i]
                if (r.width <= 1.05 * r.height) continue
                val cut = valleyColumn(mask, r) ?: continue
                parts[i] = Rect(r.x, r.y, cut - r.x, r.height)
                parts.add(i + 1, Rect(cut, r.y, r.x + r.width - cut, r.height))
                done = true
                break
            }
            if (!done) break
        }
        return parts
    }

    /** 在框中间 30%~70% 里找笔画最少、且明显低于峰值的一列，返回绝对 x；没有缝返回 null */
    private fun valleyColumn(mask: Mat, r: Rect): Int? {
        if (r.width < 6 || r.height <= 0) return null
        val sub = mask.submat(r.y, r.y + r.height, r.x, r.x + r.width)
        try {
            val buf = ByteArray(r.width * r.height)
            sub.get(0, 0, buf)
            val row = IntArray(r.width)
            for (y in 0 until r.height) {
                val off = y * r.width
                for (x in 0 until r.width) {
                    if (buf[off + x].toInt() != 0) row[x]++
                }
            }
            val peak = row.maxOrNull() ?: return null
            if (peak <= 0) return null
            var bestIdx = -1
            var best = Int.MAX_VALUE
            for (i in (r.width * 0.3).toInt() until (r.width * 0.7).toInt()) {
                if (row[i] < best) {
                    best = row[i]
                    bestIdx = i
                }
            }
            if (bestIdx < 0) return null
            // 缝里的笔画量要比最密处稀 85% 以上，否则只是笔画内部，不是两个数字之间
            if (best > 0.15 * peak) return null
            return r.x + bestIdx
        } finally {
            sub.release()
        }
    }

    /**
     * 把同一行里靠得很近的字形拼成多位数（'3''2' -> 32，'3''4''5' -> 345）。
     *
     * 合并条件（两条同时满足）：
     * 1) 相邻两个框的左右间距 <= mergeGapRatio 个数字宽度（宽度取两框平均）；
     * 2) 竖直方向有明显重叠（同一行），且两框高度相差不超过 sizeSimilar 倍。
     * 最多拼 maxDigits 位。
     */
    private fun assemble(items: List<Glyph>, tuning: Tuning): List<Glyph> {
        val sorted = items.sortedBy { it.box.x }
        val used = BooleanArray(sorted.size)
        val out = mutableListOf<Glyph>()
        for (i in sorted.indices) {
            if (used[i]) continue
            used[i] = true
            val group = mutableListOf(sorted[i])
            var last = sorted[i].box
            for (j in i + 1 until sorted.size) {
                if (used[j]) continue
                if (group.size >= tuning.maxDigits) break
                val b = sorted[j].box
                val gap = b.x - (last.x + last.width)
                val ov = minOf(last.y + last.height, b.y + b.height) - maxOf(last.y, b.y)
                val wRef = (last.width + b.width) / 2.0
                val hMax = maxOf(last.height, b.height)
                val hMin = minOf(last.height, b.height)
                val sameRow = ov >= tuning.rowOverlap * hMin
                val similarSize = hMax <= tuning.sizeSimilar * hMin
                // 间距小于「一个数字宽度」→ 同一个多位数
                if (sameRow && similarSize && gap <= tuning.mergeGapRatio * wRef) {
                    group.add(sorted[j])
                    used[j] = true
                    last = b
                }
            }
            var value = 0
            var box = group[0].box
            for (g in group) {
                value = value * 10 + g.digit
                box = union(box, g.box)
            }
            out.add(Glyph(box, value, group.minOf { it.conf }, group[0].neg))
        }
        return out
    }

    /**
     * 输入 BGR 图像，输出检测到的分数（坐标归一化）。
     * 内部会缩放到 MAX_SIDE 以内，不改动入参。
     */
    fun scan(
        img: Mat,
        rec: DigitRecognizer,
        calib: Calib = Calib.DEFAULT,
        tuning: Tuning = Tuning.DEFAULT,
        /** 识别区（归一化 0~1：left,top,right,bottom），只识别中心点在该矩形内的字形，null 表示全屏 */
        roi: FloatArray? = null,
        /** true = 实时跟踪预览：先做「大小过滤」，把过小的碎片挡在 CNN 之前 */
        filterSmall: Boolean = false
    ): List<Detected> {
        val prepared = prepare(img)
        try {
            val w = prepared.cols()
            val h = prepared.rows()
            val mask = redMask(prepared, calib)
            try {
                val lines = mutableListOf<Rect>()
                val boxesRaw = components(mask, lines = lines)
                // 识别区过滤：只保留中心点在 roi 内的字形和横杠
                val boxes: List<Rect>
                val keptLines: List<Rect>
                val roiRect = if (roi != null) Rect(
                    (roi[0] * w).toInt(), (roi[1] * h).toInt(),
                    ((roi[2] - roi[0]) * w).toInt(), ((roi[3] - roi[1]) * h).toInt()
                ) else null
                if (roiRect != null) {
                    boxes = boxesRaw.filter { b -> centerIn(b, roiRect) }
                    keptLines = lines.filter { b -> centerIn(b, roiRect) }
                } else {
                    boxes = boxesRaw
                    keptLines = lines
                }
                val glyphs = mergeGlyphs(boxes.toMutableList())
                // 粘连的两位数先按投影缝拆开，再逐个识别
                val pieces = mutableListOf<Rect>()
                for (b in glyphs) {
                    if (tuning.maxDigits > 1 && b.width > 1.05 * b.height)
                        pieces.addAll(splitWide(mask, b, tuning.maxDigits))
                    else pieces.add(b)
                }
                // 实时跟踪：过小的框不进 CNN，避免画出一堆假框
                val candidates = if (filterSmall) filterSmallGlyphs(mask, pieces) else pieces
                val items = mutableListOf<Glyph>()
                for (b in candidates) {
                    // 数字应纵向略长或接近方形，宽扁的通常是 √、横杠、圈
                    if (b.width.toFloat() / b.height > 1.8f) continue
                    val (digit, conf) = rec.recognize(mask, b, calib.templates)
                    if (digit == null || conf < tuning.minConf) continue
                    items.add(Glyph(b, digit, conf, minusForDigit(keptLines, b, tuning)))
                }
                return assemble(items, tuning).map {
                    Detected(
                        if (it.neg) -it.digit else it.digit, it.conf,
                        it.box.x.toFloat() / w, it.box.y.toFloat() / h,
                        it.box.width.toFloat() / w, it.box.height.toFloat() / h
                    )
                }
            } finally {
                mask.release()
            }
        } finally {
            prepared.release()
        }
    }

    /**
     * 实时跟踪框的大小过滤（三层，和 server/pipeline.py 的 track 一致）：
     * 1) 绝对下限：字高不足画面 1.2%，太小连 28×28 都补不出来
     * 2) 相对下限：多个框时，比「典型字高（中位数）」小一半以上的当碎片丢掉
     * 3) 笔画像素数：框够大但里面几乎是空白（污点、蹭到的点）也丢掉
     */
    fun filterSmallGlyphs(mask: Mat, boxes: List<Rect>): List<Rect> {
        val h = mask.rows()
        val minH = maxOf(10.0, TRACK_MIN_H_RATIO * h)
        var kept = boxes.filter { it.height >= minH }
        if (kept.size >= 2) {
            // 用中位数而不是最大值：避免一个特别大的圈画把正常字挤掉
            var typical = kept.map { it.height }.filter { it <= TRACK_TYPICAL_RATIO * h }
            if (typical.isEmpty()) typical = kept.map { it.height }
            typical = typical.sorted()
            val ref = typical[typical.size / 2]
            kept = kept.filter { it.height >= TRACK_SMALL_RATIO * ref }
        }
        return kept.filter { inkCount(mask, it) >= TRACK_MIN_INK }
    }

    /** 框内的红色像素数 */
    private fun inkCount(mask: Mat, r: Rect): Int {
        val x = maxOf(0, r.x)
        val y = maxOf(0, r.y)
        val w = minOf(r.width, mask.cols() - x)
        val h = minOf(r.height, mask.rows() - y)
        if (w <= 0 || h <= 0) return 0
        val sub = mask.submat(y, y + h, x, x + w)
        val n = Core.countNonZero(sub)
        sub.release()
        return n
    }

    /** 矩形中心点是否落在 roi 内 */
    private fun centerIn(b: Rect, roi: Rect): Boolean {
        val cx = b.x + b.width / 2.0
        val cy = b.y + b.height / 2.0
        return cx >= roi.x && cx <= roi.x + roi.width &&
            cy >= roi.y && cy <= roi.y + roi.height
    }

    /** 清晰度评估，用于丢弃运动模糊的帧 */
    fun sharpness(img: Mat): Double {
        val gray = Mat()
        Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY)
        val lap = Mat()
        Imgproc.Laplacian(gray, lap, CvType.CV_64F)
        val mean = MatOfDouble()
        val std = MatOfDouble()
        Core.meanStdDev(lap, mean, std)
        val v = std.get(0, 0)[0]
        gray.release()
        lap.release()
        mean.release()
        std.release()
        return v * v
    }
}
