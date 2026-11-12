package com.examscan.score

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.os.Handler
import android.os.Looper
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View

enum class MarkStatus { NEW, DUP, PENDING, EXCLUDED }

/** 归一化坐标的标记框 */
data class Mark(
    val value: Int,
    val nx: Float,
    val ny: Float,
    val nw: Float,
    val nh: Float,
    val status: MarkStatus,
    /** 学习页用：框上显示的文字，如 "3. 7"（序号 + 识别值）；为空则显示 value */
    val label: String? = null
)

/**
 * 叠加层：在预览上画彩色框和分数。
 * 绿色=新计入，蓝色=重复跳过，黄色=存疑待确认。
 * PreviewView 用 FIT_CENTER，这里用同样的信箱式映射，保证框和画面对齐。
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3.5f
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        typeface = Typeface.DEFAULT_BOLD
        textAlign = Paint.Align.CENTER
    }
    private val labelBg = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(170, 0, 0, 0)
    }
    private val contentRect = RectF()
    private val roiPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#3b82f6")   // 蓝色
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }
    private val roiShade = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(60, 59, 130, 246) // 半透明蓝填充
    }
    private val roiDragPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#f59e0b")   // 划区时橙色
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }

    private var marks: List<Mark> = emptyList()
    private var imageAspect = 3f / 4f   // 画面 宽/高（竖屏下 960x1280）

    /** 已保存的识别区：归一化 [left, top, right, bottom]，未设置为 null */
    var roiRect: FloatArray? = null
        set(v) { field = v; invalidate() }

    // ---- ROI 调整模式：显示一个矩形，用户拖角/边/中间来调整 ----
    private var roiAdjusting = false
    private var roiAdjustRect = RectF()   // 画面坐标（已扣掉信箱黑边）
    private enum class DragMode { NONE, MOVE, TL, TR, BL, BR, L, R, T, B }
    private var dragMode = DragMode.NONE
    private var dragLastX = 0f
    private var dragLastY = 0f
    private val cornerSize = 36f   // 角点触摸半径
    /** 调整完成回调，参数为归一化 [l,t,r,b]，取消时传 null */
    private var onRoiAdjustDone: ((FloatArray?) -> Unit)? = null

    /** 点中某个框时回调它的下标（学习页用来改标号） */
    var onMarkTap: ((Int) -> Unit)? = null

    // ---- 浮动「+N」动画：拍照确认时数字向上跳一下 ----
    private data class FloatText(val text: String, val nx: Float, val ny: Float, val startMs: Long)
    private val floatTexts = mutableListOf<FloatText>()
    private val handler = Handler(Looper.getMainLooper())
    private val floatTick = object : Runnable {
        override fun run() {
            if (floatTexts.isNotEmpty()) {
                invalidate()
                handler.postDelayed(this, 16)
            }
        }
    }

    /** 在归一化坐标 (nx, ny) 处弹出一个向上飘 + 淡出的文字（如 "+5"） */
    fun addFloatText(text: String, nx: Float, ny: Float) {
        floatTexts.add(FloatText(text, nx, ny, System.currentTimeMillis()))
        handler.post(floatTick)
    }

    fun setFrame(aspect: Float) {
        if (aspect > 0f && aspect != imageAspect) {
            imageAspect = aspect
            invalidate()
        }
    }

    fun setMarks(list: List<Mark>) {
        marks = list
        invalidate()
    }

    /** 进入「调整识别区」模式：显示一个默认矩形，用户拖角/边/中间调整。
     *  [initial] 为已保存的归一化矩形，null 时用默认居中矩形。 */
    fun startRoiAdjust(initial: FloatArray?, onDone: (FloatArray?) -> Unit) {
        onRoiAdjustDone = onDone
        roiAdjusting = true
        val (ox, oy, dw, dh) = layout()
        if (initial != null && initial.size == 4) {
            roiAdjustRect.set(
                ox + initial[0] * dw, oy + initial[1] * dh,
                ox + initial[2] * dw, oy + initial[3] * dh
            )
        } else {
            // 默认：居中，宽 70%，高 50%
            roiAdjustRect.set(
                ox + dw * 0.15f, oy + dh * 0.25f,
                ox + dw * 0.85f, oy + dh * 0.75f
            )
        }
        invalidate()
    }

    /** 获取当前调整中的归一化矩形 */
    fun getRoiAdjustRect(): FloatArray {
        val (ox, oy, dw, dh) = layout()
        return floatArrayOf(
            ((roiAdjustRect.left - ox) / dw).coerceIn(0f, 1f),
            ((roiAdjustRect.top - oy) / dh).coerceIn(0f, 1f),
            ((roiAdjustRect.right - ox) / dw).coerceIn(0f, 1f),
            ((roiAdjustRect.bottom - oy) / dh).coerceIn(0f, 1f)
        )
    }

    /** 退出调整模式 */
    fun cancelRoiAdjust() {
        roiAdjusting = false
        onRoiAdjustDone = null
        dragMode = DragMode.NONE
        invalidate()
    }

    /** 画面在 View 里的信箱式摆放：ox, oy, dw, dh */
    private fun layout(): FloatArray {
        val vw = width.toFloat()
        val vh = height.toFloat()
        val dw = minOf(vw, vh * imageAspect)
        val dh = dw / imageAspect
        return floatArrayOf((vw - dw) / 2f, (vh - dh) / 2f, dw, dh)
    }

    override fun dispatchTouchEvent(event: MotionEvent): Boolean {
        return super.dispatchTouchEvent(event)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        // ---- ROI 调整模式：拖角/边/中间来改变矩形 ----
        if (roiAdjusting) {
            val r = roiAdjustRect
            val x = event.x
            val y = event.y
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    dragLastX = x
                    dragLastY = y
                    // 判断触摸点在哪个区域
                    val near = cornerSize
                    dragMode = when {
                        // 四个角
                        x in (r.left - near)..(r.left + near) && y in (r.top - near)..(r.top + near) -> DragMode.TL
                        x in (r.right - near)..(r.right + near) && y in (r.top - near)..(r.top + near) -> DragMode.TR
                        x in (r.left - near)..(r.left + near) && y in (r.bottom - near)..(r.bottom + near) -> DragMode.BL
                        x in (r.right - near)..(r.right + near) && y in (r.bottom - near)..(r.bottom + near) -> DragMode.BR
                        // 四条边
                        x in (r.left - near)..(r.left + near) -> DragMode.L
                        x in (r.right - near)..(r.right + near) -> DragMode.R
                        y in (r.top - near)..(r.top + near) -> DragMode.T
                        y in (r.bottom - near)..(r.bottom + near) -> DragMode.B
                        // 内部：移动
                        x in r.left..r.right && y in r.top..r.bottom -> DragMode.MOVE
                        else -> DragMode.NONE
                    }
                    return true
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = x - dragLastX
                    val dy = y - dragLastY
                    dragLastX = x
                    dragLastY = y
                    val (ox, oy, dw, dh) = layout()
                    val minW = 40f
                    val minH = 40f
                    when (dragMode) {
                        DragMode.MOVE -> {
                            r.offset(dx, dy)
                            // 限制在画面内
                            if (r.left < ox) r.offset(ox - r.left, 0f)
                            if (r.top < oy) r.offset(0f, oy - r.top)
                            if (r.right > ox + dw) r.offset(ox + dw - r.right, 0f)
                            if (r.bottom > oy + dh) r.offset(0f, oy + dh - r.bottom)
                        }
                        DragMode.TL -> {
                            r.left = (r.left + dx).coerceIn(ox, r.right - minW)
                            r.top = (r.top + dy).coerceIn(oy, r.bottom - minH)
                        }
                        DragMode.TR -> {
                            r.right = (r.right + dx).coerceIn(r.left + minW, ox + dw)
                            r.top = (r.top + dy).coerceIn(oy, r.bottom - minH)
                        }
                        DragMode.BL -> {
                            r.left = (r.left + dx).coerceIn(ox, r.right - minW)
                            r.bottom = (r.bottom + dy).coerceIn(r.top + minH, oy + dh)
                        }
                        DragMode.BR -> {
                            r.right = (r.right + dx).coerceIn(r.left + minW, ox + dw)
                            r.bottom = (r.bottom + dy).coerceIn(r.top + minH, oy + dh)
                        }
                        DragMode.L -> {
                            r.left = (r.left + dx).coerceIn(ox, r.right - minW)
                        }
                        DragMode.R -> {
                            r.right = (r.right + dx).coerceIn(r.left + minW, ox + dw)
                        }
                        DragMode.T -> {
                            r.top = (r.top + dy).coerceIn(oy, r.bottom - minH)
                        }
                        DragMode.B -> {
                            r.bottom = (r.bottom + dy).coerceIn(r.top + minH, oy + dh)
                        }
                        DragMode.NONE -> {}
                    }
                    invalidate()
                    return true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    dragMode = DragMode.NONE
                    return true
                }
            }
            return true
        }

        if (marks.isEmpty()) return super.onTouchEvent(event)
        when (event.actionMasked) {
            // 必须吃掉 DOWN，否则后续的 UP 根本不会送给这个 View，点击永远没反应
            MotionEvent.ACTION_DOWN -> return true
            MotionEvent.ACTION_UP -> {
                val l = layout()
                val x = event.x - l[0]
                val y = event.y - l[1]
                if (x in 0f..l[2] && y in 0f..l[3]) {
                    // 框在屏幕上很小，手指又粗，命中容差给 24dp
                    val pad = 24f * resources.displayMetrics.density
                    // 框可能互相重叠：取面积最小的那个，命中更符合直觉
                    var hit = -1
                    var bestArea = Float.MAX_VALUE
                    marks.forEachIndexed { i, m ->
                        val left = m.nx * l[2] - pad
                        val top = m.ny * l[3] - pad
                        val right = (m.nx + m.nw) * l[2] + pad
                        val bottom = (m.ny + m.nh) * l[3] + pad
                        if (x >= left && x <= right && y >= top && y <= bottom) {
                            val area = m.nw * m.nh
                            if (area < bestArea) {
                                bestArea = area
                                hit = i
                            }
                        }
                    }
                    if (hit >= 0) {
                        performClick()
                        onMarkTap?.invoke(hit)
                    }
                }
                return true
            }
        }
        return true
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val (ox, oy, dw, dh) = layout()
        contentRect.set(ox, oy, ox + dw, oy + dh)

        // ---- 已保存的识别区（蓝色框 + 半透明遮罩） ----
        roiRect?.let { r ->
            val l = ox + r[0] * dw
            val t = oy + r[1] * dh
            val right = ox + r[2] * dw
            val b = oy + r[3] * dh
            canvas.drawRect(l, t, right, b, roiShade)
            canvas.drawRect(l, t, right, b, roiPaint)
            // 四个角的小标记
            val c = 14f
            canvas.drawLine(l, t, l + c, t, roiPaint)
            canvas.drawLine(l, t, l, t + c, roiPaint)
            canvas.drawLine(right, t, right - c, t, roiPaint)
            canvas.drawLine(right, t, right, t + c, roiPaint)
            canvas.drawLine(l, b, l + c, b, roiPaint)
            canvas.drawLine(l, b, l, b - c, roiPaint)
            canvas.drawLine(right, b, right - c, b, roiPaint)
            canvas.drawLine(right, b, right, b - c, roiPaint)
        }

        // ---- 调整模式下的矩形（橙色 + 可拖角点） ----
        if (roiAdjusting) {
            val r = roiAdjustRect
            canvas.drawRect(r, roiShade)
            canvas.drawRect(r, roiDragPaint)
            // 四个角的小方块，提示可拖动
            val cs = 14f
            canvas.drawRect(r.left - cs, r.top - cs, r.left + cs, r.top + cs, roiDragPaint)
            canvas.drawRect(r.right - cs, r.top - cs, r.right + cs, r.top + cs, roiDragPaint)
            canvas.drawRect(r.left - cs, r.bottom - cs, r.left + cs, r.bottom + cs, roiDragPaint)
            canvas.drawRect(r.right - cs, r.bottom - cs, r.right + cs, r.bottom + cs, roiDragPaint)
        }

        if (marks.isEmpty() && floatTexts.isEmpty()) return

        val textSize = (dw * 0.042f).coerceIn(22f, 54f)
        textPaint.textSize = textSize

        for (m in marks) {
            val color = when (m.status) {
                MarkStatus.NEW -> Color.parseColor("#22c55e")
                MarkStatus.DUP -> Color.parseColor("#3b82f6")
                MarkStatus.PENDING -> Color.parseColor("#f59e0b")
                MarkStatus.EXCLUDED -> Color.parseColor("#dc2626")
            }
            val l = ox + m.nx * dw
            val t = oy + m.ny * dh
            val r = l + m.nw * dw
            val b = t + m.nh * dh

            boxPaint.color = color
            canvas.drawRect(l, t, r, b, boxPaint)

            // 分数标签画在框上方，出屏时挪到框内
            val label = m.label ?: m.value.toString()
            val tw = textPaint.measureText(label)
            val th = textSize
            var ty = t - th * 0.35f
            if (ty - th < oy) ty = t + th * 1.4f
            val cx = (l + r) / 2f

            labelBg.color = Color.argb(170, 0, 0, 0)
            canvas.drawRoundRect(cx - tw / 2f - 6, ty - th, cx + tw / 2f + 6, ty + th * 0.3f, 6f, 6f, labelBg)
            textPaint.color = color
            canvas.drawText(label, cx, ty, textPaint)
        }

        // 浮动「+N」动画：向上飘 + 淡出
        if (floatTexts.isNotEmpty()) {
            val now = System.currentTimeMillis()
            val iter = floatTexts.iterator()
            val floatSize = (dw * 0.09f).coerceIn(40f, 96f)
            textPaint.textSize = floatSize
            while (iter.hasNext()) {
                val ft = iter.next()
                val elapsed = now - ft.startMs
                if (elapsed > FLOAT_DURATION) {
                    iter.remove()
                    continue
                }
                val p = elapsed.toFloat() / FLOAT_DURATION        // 0~1
                val alpha = ((1f - p) * 255).toInt().coerceIn(0, 255)
                val offsetY = p * FLOAT_DISTANCE_DP * resources.displayMetrics.density
                val cx = ox + ft.nx * dw
                val cy = oy + ft.ny * dh - offsetY
                textPaint.color = Color.argb(alpha, 34, 197, 94)   // 绿色 #22c55e
                canvas.drawText(ft.text, cx, cy, textPaint)
            }
            textPaint.color = Color.WHITE
            textPaint.alpha = 255
        }
    }

    companion object {
        private const val FLOAT_DURATION = 900L         // 动画时长 ms
        private const val FLOAT_DISTANCE_DP = 90f       // 向上飘的距离 dp
    }
}
