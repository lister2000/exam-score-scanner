package com.examscan.score

import android.content.Context
import android.content.SharedPreferences

/**
 * 统计页可调规则，存 SharedPreferences，设置页改完立即生效。
 *
 * 默认值和之前在 ScorePipeline / MainActivity 里写死的常量保持一致，
 * 所以老用户升级后行为不变。
 */
class ScanSettings(context: Context) {

    private val p: SharedPreferences =
        context.getSharedPreferences("scan_settings", Context.MODE_PRIVATE)

    /** 一个多位数最多识别几位（1 = 不合并，每个数字单独算） */
    var maxDigits: Int
        get() = p.getInt("max_digits", DEF_MAX_DIGITS).coerceIn(1, 4)
        set(v) = p.edit().putInt("max_digits", v.coerceIn(1, 4)).apply()

    /** 数字间距合并阈值，单位是「数字宽度」：0.4 = 很紧才算，2.0 = 离得远也并 */
    var mergeGapRatio: Float
        get() = p.getFloat("merge_gap", DEF_MERGE_GAP).coerceIn(0.4f, 2.0f)
        set(v) = p.edit().putFloat("merge_gap", v.coerceIn(0.4f, 2.0f)).apply()

    /** 识别置信度下限：越高越严格（误识别少、漏检多） */
    var minConf: Float
        get() = p.getFloat("min_conf", DEF_MIN_CONF).coerceIn(0.30f, 0.90f)
        set(v) = p.edit().putFloat("min_conf", v.coerceIn(0.30f, 0.90f)).apply()

    /** 同一行判定：两框竖直重叠至少占较矮那个框高度的比例 */
    var rowOverlap: Float
        get() = p.getFloat("row_overlap", DEF_ROW_OVERLAP).coerceIn(0.3f, 0.9f)
        set(v) = p.edit().putFloat("row_overlap", v.coerceIn(0.3f, 0.9f)).apply()

    /** 字号相似上限：两框高度差超过这个倍数就不合并（防大小字误并） */
    var sizeSimilar: Float
        get() = p.getFloat("size_similar", DEF_SIZE_SIMILAR).coerceIn(1.2f, 2.5f)
        set(v) = p.edit().putFloat("size_similar", v.coerceIn(1.2f, 2.5f)).apply()

    /** 实时框刷新间隔（毫秒）：越小越跟手，但更耗电 */
    var liveIntervalMs: Int
        get() = p.getInt("live_interval", DEF_LIVE_INTERVAL).coerceIn(150, 600)
        set(v) = p.edit().putInt("live_interval", v.coerceIn(150, 600)).apply()

    /** 模糊帧阈值：低于它认为画面太糊，不识别 */
    var blurThreshold: Float
        get() = p.getFloat("blur", DEF_BLUR).coerceIn(2.0f, 20.0f)
        set(v) = p.edit().putFloat("blur", v.coerceIn(2.0f, 20.0f)).apply()

    /** 硬币音效开关 */
    var coinSound: Boolean
        get() = p.getBoolean("coin_sound", true)
        set(v) = p.edit().putBoolean("coin_sound", v).apply()

    /** 是否把数字左边的横杠识别成负号（关掉后横杠只被忽略，不会拉低识别） */
    var negEnabled: Boolean
        get() = p.getBoolean("neg_enabled", true)
        set(v) = p.edit().putBoolean("neg_enabled", v).apply()

    /** 负号与数字的最大水平间距，单位是「数字高度」 */
    var negGapRatio: Float
        get() = p.getFloat("neg_gap", DEF_NEG_GAP).coerceIn(0.1f, 1.5f)
        set(v) = p.edit().putFloat("neg_gap", v.coerceIn(0.1f, 1.5f)).apply()

    /** 是否启用「数字识别区域」：只在用户框出的矩形内识别，框外的字全部忽略 */
    var roiEnabled: Boolean
        get() = p.getBoolean("roi_enabled", false)
        set(v) = p.edit().putBoolean("roi_enabled", v).apply()

    /** 识别区矩形（归一化 0~1：left, top, right, bottom），未设置时为 null */
    val roiRect: FloatArray?
        get() {
            if (!roiEnabled) return null
            val l = p.getFloat("roi_l", -1f)
            val t = p.getFloat("roi_t", -1f)
            val r = p.getFloat("roi_r", -1f)
            val b = p.getFloat("roi_b", -1f)
            if (l < 0 || t < 0 || r < 0 || b < 0) return null
            return floatArrayOf(l, t, r, b)
        }

    /** 设置识别区：4 个值都在 0~1，且 left<right、top<bottom；传入 null 表示清除 */
    fun setRoi(rect: FloatArray?) {
        if (rect == null) {
            p.edit()
                .putBoolean("roi_enabled", false)
                .remove("roi_l").remove("roi_t").remove("roi_r").remove("roi_b")
                .apply()
            return
        }
        val l = rect[0].coerceIn(0f, 1f)
        val t = rect[1].coerceIn(0f, 1f)
        val r = rect[2].coerceIn(0f, 1f)
        val b = rect[3].coerceIn(0f, 1f)
        if (r <= l || b <= t) {
            setRoi(null)
            return
        }
        p.edit()
            .putBoolean("roi_enabled", true)
            .putFloat("roi_l", l).putFloat("roi_t", t)
            .putFloat("roi_r", r).putFloat("roi_b", b)
            .apply()
    }

    fun reset() = p.edit().clear().apply()

    /** 转成识别管道可直接用的参数 */
    fun tuning() = ScorePipeline.Tuning(
        maxDigits = maxDigits,
        mergeGapRatio = mergeGapRatio.toDouble(),
        minConf = minConf,
        rowOverlap = rowOverlap.toDouble(),
        sizeSimilar = sizeSimilar.toDouble(),
        negEnabled = negEnabled,
        negGapRatio = negGapRatio.toDouble()
    )

    companion object {
        const val DEF_MAX_DIGITS = 4
        const val DEF_MERGE_GAP = 0.8f
        const val DEF_MIN_CONF = 0.45f
        const val DEF_ROW_OVERLAP = 0.5f
        const val DEF_SIZE_SIMILAR = 1.8f
        const val DEF_LIVE_INTERVAL = 250
        const val DEF_BLUR = 6.0f
        const val DEF_NEG_GAP = 0.7f
    }
}
