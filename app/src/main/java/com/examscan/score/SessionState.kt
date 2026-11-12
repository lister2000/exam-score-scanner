package com.examscan.score

/**
 * 一份试卷的计分状态。去重按"页面坐标重叠度"判断，绝不用数值去重——
 * 同一个学生卷面上出现两个 10 分是很正常的事。
 */
class SessionState {

    /** pageBox = 归一化的页面坐标 [x, y, w, h] */
    class Item(val value: Int, val conf: Float, val pageBox: FloatArray)

    val digits = mutableListOf<Item>()
    val pending = mutableListOf<Item>()
    val students = mutableListOf<Pair<Int, List<Int>>>()
    val tracker = FrameTracker()

    val total: Int
        get() = digits.sumOf { it.value }

    @Synchronized
    fun nextStudent() {
        students.add(Pair(total, digits.map { it.value }))
        digits.clear()
        pending.clear()
        tracker.reset()
    }

    @Synchronized
    fun undo() {
        if (digits.isNotEmpty()) digits.removeAt(digits.size - 1)
    }

    /** 撤销末尾 n 个（改标号时用：先撤掉本次拍照的全部，再按新标号重加） */
    @Synchronized
    fun undoLast(n: Int) {
        repeat(n) { if (digits.isNotEmpty()) digits.removeAt(digits.size - 1) }
    }

    @Synchronized
    fun addManual(value: Int) {
        digits.add(Item(value, 1f, floatArrayOf(-99f, -99f, 0.001f, 0.001f)))
    }

    /** 把一批数值计入（无去重，调用方已经筛过） */
    @Synchronized
    fun addAll(values: List<Int>) {
        values.forEach { digits.add(Item(it, 1f, floatArrayOf(-99f, -99f, 0.001f, 0.001f))) }
    }

    /** 清空所有 */
    @Synchronized
    fun reset() {
        digits.clear()
        pending.clear()
        tracker.reset()
    }

    @Synchronized
    fun confirmPending(index: Int) {
        if (index in pending.indices) {
            val item = pending.removeAt(index)
            digits.add(Item(item.value, 1f, item.pageBox))
        }
    }

    @Synchronized
    fun dismissPending(index: Int) {
        if (index in pending.indices) pending.removeAt(index)
    }

    companion object {
        fun iou(a: FloatArray, b: FloatArray): Float {
            val ax2 = a[0] + a[2]
            val ay2 = a[1] + a[3]
            val bx2 = b[0] + b[2]
            val by2 = b[1] + b[3]
            val ix1 = maxOf(a[0], b[0])
            val iy1 = maxOf(a[1], b[1])
            val ix2 = minOf(ax2, bx2)
            val iy2 = minOf(ay2, by2)
            val iw = maxOf(0f, ix2 - ix1)
            val ih = maxOf(0f, iy2 - iy1)
            val inter = iw * ih
            if (inter <= 0f) return 0f
            val union = a[2] * a[3] + b[2] * b[3] - inter
            if (union <= 0f) return 0f
            return inter / union
        }
    }
}
