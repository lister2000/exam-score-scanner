package com.examscan.score

/**
 * 一支笔 + 一个人手写的标定档案。
 *
 * 默认值就是原来硬编码的「红色」阈值；学习阶段会用真实样本覆盖它。
 */
data class Calib(
    /** HSV 色相区间（OpenCV 的 H 是 0~180），可能拆成两段（跨越 0/180 时） */
    val hueRanges: List<IntArray> = listOf(intArrayOf(0, 16), intArrayOf(152, 180)),
    val satMin: Int = 25,
    val valMin: Int = 40,
    val valMax: Int = 255,
    /** R 比 max(G,B) 高出多少才算红笔（兜住浅色笔） */
    val minRedness: Int = 6,
    /**
     * 数字 -> 该数字的若干份 28x28 模板（784 字节，行优先）。
     * 一个数字可以写多遍，样本越多模板越稳；没写的数字就不在表里。
     */
    val templates: Map<Int, List<ByteArray>> = emptyMap(),
    /** 学习时实际生效的饱和度下限（0 表示没学习过），排查淡笔迹时有用 */
    val satUsed: Int = 0
) {

    /** 只要学到任意一个数字就算学过（缺的数字统计时退回纯 CNN） */
    val learned: Boolean
        get() = templates.values.any { it.isNotEmpty() }

    /** 已覆盖的数字，升序 */
    val covered: List<Int>
        get() = templates.keys.sorted()

    /** 还没学到的数字 */
    val missing: List<Int>
        get() = (0 until DIGITS).filter { templates[it].isNullOrEmpty() }

    /** 追加一个模板（digit 可以是 0~9 或多位数的某一位——但 10/11/12 之类不进档案） */
    fun plusTemplate(digit: Int, template: ByteArray): Calib {
        if (digit !in 0 until DIGITS) return this
        val list = (templates[digit] ?: emptyList()) + template
        return copy(templates = templates + (digit to list))
    }

    /** 合并两份 Calib：色相取并集、阈值取更严格侧、模板拼接 */
    fun mergeWith(other: Calib): Calib {
        // 色相：先把两边的 intArray 拼起来，每段 [a,b] 展开成 180 段布尔数组标记为真，
        // 然后从 0..180 重新找连续的 TRUE 段（自动合并相邻/重叠的）
        val mask = BooleanArray(181) // 0..180
        for (r in hueRanges) { val lo = r[0]; val hi = r[1]; for (h in lo..hi) mask[h] = true }
        for (r in other.hueRanges) { val lo = r[0]; val hi = r[1]; for (h in lo..hi) mask[h] = true }
        val mergedRanges = mutableListOf<IntArray>()
        var i = 0
        while (i <= 180) {
            if (mask[i]) {
                val lo = i
                while (i <= 180 && mask[i]) i++
                mergedRanges.add(intArrayOf(lo, i - 1))
            } else i++
        }
        // 模板拼接
        val mergedTpls = LinkedHashMap<Int, List<ByteArray>>()
        for ((d, list) in templates) mergedTpls[d] = list
        for ((d, list) in other.templates) {
            mergedTpls[d] = (mergedTpls[d] ?: emptyList()) + list
        }
        return copy(
            hueRanges = mergedRanges.ifEmpty { hueRanges },
            satMin = maxOf(satMin, other.satMin),
            valMin = minOf(valMin, other.valMin),
            valMax = 255,
            minRedness = maxOf(minRedness, other.minRedness),
            templates = mergedTpls,
            satUsed = maxOf(satUsed, other.satUsed)
        )
    }

    companion object {
        const val DIGITS = 10
        val DEFAULT = Calib()
    }
}
