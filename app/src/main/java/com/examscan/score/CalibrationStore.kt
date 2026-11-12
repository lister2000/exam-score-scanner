package com.examscan.score

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import org.opencv.core.Rect
import org.opencv.imgproc.Imgproc
import java.io.File

/**
 * 标定档案的本地存储：filesDir/calib.json。
 *
 * 只有一支笔/一个人时用它最省事；换笔或换人重新学习即可覆盖。
 */
class CalibrationStore(context: Context) {

    private val file = File(context.filesDir, "calib.json")

    var calib: Calib = Calib.DEFAULT
        private set

    init {
        calib = load() ?: Calib.DEFAULT
        if (calib.learned) Log.i(TAG, "读取标定：色相 ${calib.hueRanges.size} 段, S>=${calib.satMin}")
    }

    fun save(c: Calib) {
        val root = JSONObject()
        root.put("version", 2)
        val hues = JSONArray()
        for (r in c.hueRanges) hues.put(JSONArray().put(r[0]).put(r[1]))
        root.put("hues", hues)
        root.put("satMin", c.satMin)
        root.put("valMin", c.valMin)
        root.put("valMax", c.valMax)
        root.put("minRedness", c.minRedness)
        // 模板带标号存：学习时的书写顺序不影响，缺哪个数字也能看出来
        val tpls = JSONArray()
        for ((digit, samples) in c.templates) {
            val one = JSONObject()
            one.put("d", digit)
            val arr = JSONArray()
            for (t in samples) {
                val row = JSONArray()
                for (v in t) row.put(v.toInt() and 0xFF)
                arr.put(row)
            }
            one.put("s", arr)
            tpls.put(one)
        }
        root.put("templates", tpls)
        file.writeText(root.toString())
        calib = c
        Log.i(TAG, "标定已保存：${c.covered} 共 ${c.templates.values.sumOf { it.size }} 份模板")
    }

    fun clear() {
        file.delete()
        calib = Calib.DEFAULT
    }

    /** 从学习页返回时重新读取 */
    fun reload() {
        calib = load() ?: Calib.DEFAULT
    }

    /**
     * 增量学习：把统计页冻结态下用户确认过的字形追加入档案。
     *
     * frame = 冻结的 BGR 画面；items = 用户已确认的框（归一化坐标 + 标号，digit=-1 表示排除）。
     * 返回本次新增的模板份数。
     */
    fun addShot(frame: org.opencv.core.Mat, items: List<AddShotItem>): Int {
        val valid = items.filter { it.digit in 0 until Calib.DIGITS }
        if (valid.isEmpty()) return 0

        // 用当前档案的红色阈值算一次 mask，保证和统计页的识别一致
        val mask = ScorePipeline.redMask(frame, calib)
        val newCalib = Calibrator.colorStats(frame, mask).first

        var merged = calib.mergeWith(newCalib)
        var added = 0

        val fw = frame.cols(); val fh = frame.rows()
        for (it in valid) {
            val x = (it.nx * fw).toInt().coerceIn(0, fw - 1)
            val y = (it.ny * fh).toInt().coerceIn(0, fh - 1)
            val w = (it.nw * fw).toInt().coerceAtLeast(2)
            val h = (it.nh * fh).toInt().coerceAtLeast(2)
            val rect = Rect(x, y, minOf(w, fw - x), minOf(h, fh - y))
            val canvas = GlyphCanvas.from(mask, rect) ?: continue
            val bytes = GlyphCanvas.toBytes(canvas)
            canvas.release()
            merged = merged.plusTemplate(it.digit, bytes)
            added++
        }

        mask.release()
        save(merged)
        Log.i(TAG, "增量学习：追加 $added 份模板（${valid.map { it.digit }.distinct()}）")
        return added
    }

    private fun load(): Calib? {
        if (!file.exists()) return null
        return try {
            val root = JSONObject(file.readText())
            val hues = mutableListOf<IntArray>()
            val hj = root.optJSONArray("hues")
            if (hj != null) {
                for (i in 0 until hj.length()) {
                    val r = hj.getJSONArray(i)
                    hues.add(intArrayOf(r.getInt(0), r.getInt(1)))
                }
            }
            val tpls = LinkedHashMap<Int, MutableList<ByteArray>>()
            val tj = root.optJSONArray("templates")
            if (tj != null) {
                for (i in 0 until tj.length()) {
                    val e = tj.optJSONObject(i)
                    if (e != null) {
                        // 版本 2：{d: 数字, s: [模板...]}
                        val d = e.optInt("d", -1)
                        val sj = e.optJSONArray("s") ?: continue
                        if (d !in 0 until Calib.DIGITS) continue
                        val rows = tpls.getOrPut(d) { mutableListOf() }
                        for (k in 0 until sj.length()) {
                            readTemplate(sj.optJSONArray(k))?.let { rows.add(it) }
                        }
                    } else {
                        // 版本 1：数组的第 i 项就是数字 i
                        readTemplate(tj.optJSONArray(i))?.let { bytes ->
                            tpls.getOrPut(i) { mutableListOf() }.add(bytes)
                        }
                    }
                }
            }
            Calib(
                hueRanges = hues.ifEmpty { Calib.DEFAULT.hueRanges },
                satMin = root.optInt("satMin", Calib.DEFAULT.satMin),
                valMin = root.optInt("valMin", Calib.DEFAULT.valMin),
                valMax = root.optInt("valMax", Calib.DEFAULT.valMax),
                minRedness = root.optInt("minRedness", Calib.DEFAULT.minRedness),
                templates = tpls
            )
        } catch (e: Throwable) {
            Log.w(TAG, "标定文件损坏，回退默认值", e)
            null
        }
    }

    /** 读一个 784 字节模板；长度不足或为空返回 null */
    private fun readTemplate(a: JSONArray?): ByteArray? {
        if (a == null || a.length() < 28 * 28) return null
        val bytes = ByteArray(28 * 28)
        for (j in bytes.indices) bytes[j] = a.getInt(j).toByte()
        return bytes
    }

    companion object {
        private const val TAG = "CalibrationStore"
    }

    /** addShot 专用数据类：归一化坐标 + 标号（-1 = 排除） */
    data class AddShotItem(val nx: Float, val ny: Float, val nw: Float, val nh: Float, val digit: Int)
}
