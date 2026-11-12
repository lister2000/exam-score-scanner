package com.examscan.score

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.imgcodecs.Imgcodecs
import java.io.File
import java.util.concurrent.Executors

/**
 * 学习页：拍一张写着若干数字的白纸（个数、顺序、排列都不限），
 * 自动标出每个字形的位置和 CNN 初值，用户点框可改标号，最后生成学习报告并保存。
 */
class CalibrationActivity : AppCompatActivity() {

    private lateinit var preview: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var statusText: TextView

    private val store by lazy { CalibrationStore(this) }
    private val recognizer by lazy { DigitRecognizer(this) }
    private val work = Executors.newSingleThreadExecutor()

    @Volatile
    private var latest: Mat? = null
    private var result: Calibrator.Result? = null

    /** 与 result.items 一一对应的最终标号（用户校正后） */
    private var labels = mutableListOf<Int>()
    private var lastSampleMs = 0L
    private var lastDumpMs = 0L

    /** 最近几帧的框，用于多帧一致性过滤 */
    private val history = ArrayDeque<List<Rect>>()

    /** true = 已经拍好一张，停止实时出框；点「重拍」恢复 */
    @Volatile
    private var frozen = false

    /** 点「拍摄学习」后置位，由相机线程在下一帧上做落盘 + 标定，避免和 UI 线程抢 Mat */
    @Volatile
    private var captureRequested = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_calibration)

        if (!OpenCVLoader.initLocal()) {
            Toast.makeText(this, "OpenCV 初始化失败", Toast.LENGTH_LONG).show()
            finish()
            return
        }

        preview = findViewById(R.id.previewView)
        // 必须是 FIT_CENTER：默认的 FILL_CENTER 会把画面裁掉一块再放大，
        // 而叠加层是按「完整画面等比居中」算坐标的，两者不一致就会框偏、框小
        preview.scaleType = PreviewView.ScaleType.FIT_CENTER
        overlay = findViewById(R.id.overlay)
        statusText = findViewById(R.id.statusText)
        overlay.onMarkTap = ::onMarkTapped

        findViewById<TextView>(R.id.btnBack).setOnClickListener { finish() }
        findViewById<Button>(R.id.btnCapture).setOnClickListener {
            if (result != null) retake() else capture()
        }
        findViewById<Button>(R.id.btnSave).setOnClickListener { save() }
        findViewById<Button>(R.id.btnSave).isEnabled = false
        findViewById<Button>(R.id.btnResetLearn).setOnClickListener { resetLearn() }
        findViewById<Button>(R.id.btnGoScan).setOnClickListener {
            startActivity(android.content.Intent(this, MainActivity::class.java))
        }

        updateGoScanButton()

        statusText.text = getString(R.string.learn_hint)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), REQ_CAMERA)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_CAMERA &&
            grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            Toast.makeText(this, R.string.permission_need, Toast.LENGTH_LONG).show()
            finish()
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val prev = Preview.Builder().build().also { it.setSurfaceProvider(preview.surfaceProvider) }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(work, ::analyze)
            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, prev, analysis)
            } catch (e: Throwable) {
                Toast.makeText(this, "相机启动失败：${e.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * 相机帧回调（工作线程）：
     * · 没拍照时——实时切框，框就是笔迹本身的大小，跟着镜头动；
     * · 点过「拍摄学习」——把这一帧当拍摄帧，落盘 + 跑完整标定。
     */
    private fun analyze(image: ImageProxy) {
        try {
            val now = System.currentTimeMillis()
            if (now - lastSampleMs < 150) return
            lastSampleMs = now
            val fresh = Frames.upright(image) ?: return

            if (captureRequested) {
                captureRequested = false
                val snapshot = fresh.clone()
                val dir = getExternalFilesDir(null)
                val r = try {
                    if (dir != null) Imgcodecs.imwrite(File(dir, "learn_frame.jpg").absolutePath, snapshot)
                    Calibrator.calibrate(snapshot, recognizer, dir, store.calib.templates)
                } catch (e: Throwable) {
                    android.util.Log.e(TAG, "标定失败", e)
                    null
                } finally {
                    snapshot.release()
                }
                runOnUiThread {
                    if (r != null) {
                        render(r)
                    } else {
                        frozen = false
                        statusText.text = "分析出错，请重拍"
                    }
                }
            } else if (!frozen) {
                // 每 4 秒落一次现场图，出问题时我能 pull 回来复盘
                val now = System.currentTimeMillis()
                val dir = getExternalFilesDir(null)
                val wantDump = dir != null && now - lastDumpMs >= 4000
                if (wantDump) lastDumpMs = now
                val live = Calibrator.detect(fresh, if (wantDump) dir else null)
                // 多帧一致性：同一位置连续两帧都出现才显示，瞬态的光照假框会自己消失
                val stable = live.boxes.filter { b ->
                    history.any { prev -> prev.any { p -> iou(p, b) > 0.5f } }
                }
                history.addLast(live.boxes)
                while (history.size > 3) history.removeFirst()
                runOnUiThread {
                    if (frozen) return@runOnUiThread
                    overlay.setFrame(live.w.toFloat() / live.h)
                    overlay.setMarks(stable.mapIndexed { i, b ->
                        Mark(
                            value = i + 1,
                            nx = b.x.toFloat() / live.w,
                            ny = b.y.toFloat() / live.h,
                            nw = b.width.toFloat() / live.w,
                            nh = b.height.toFloat() / live.h,
                            status = MarkStatus.NEW,
                            label = "${i + 1}"
                        )
                    })
                }
            }

            val old = latest
            latest = fresh
            old?.release()
        } catch (e: Throwable) {
            android.util.Log.w(TAG, "取样失败", e)
        } finally {
            image.close()
        }
    }

    private fun capture() {
        if (result != null) {
            statusText.text = "已经拍好一张了，要重拍请点「重拍」"
            return
        }
        if (latest == null) {
            statusText.text = "还没取到画面，稍等一下再点"
            return
        }
        frozen = true
        captureRequested = true
        statusText.text = "正在分析…"
    }

    private fun render(r: Calibrator.Result) {
        frozen = true
        result = r
        labels = r.items.map { it.digit }.toMutableList()
        overlay.setFrame(r.sizeW.toFloat() / r.sizeH)
        paintMarks()
        renderReport()
        findViewById<Button>(R.id.btnCapture).text = getString(R.string.btn_retake)
    }

    private fun paintMarks() {
        val r = result ?: return
        val w = r.sizeW.toFloat()
        val h = r.sizeH.toFloat()
        val marks = r.items.mapIndexed { i, it ->
            val d = labels[i]
            Mark(
                value = if (d < 0) 0 else d,
                nx = it.box.x / w,
                ny = it.box.y / h,
                nw = it.box.width / w,
                nh = it.box.height / h,
                // 被标成「不是数字」的框用橙色，一眼能区分
                status = if (d < 0) MarkStatus.PENDING else MarkStatus.NEW,
                label = "${i + 1}. ${if (d < 0) "非数字" else d.toString()}"
            )
        }
        overlay.setMarks(marks)
    }

    /** 学习报告：位置、标号分布、颜色档案、缺口 */
    private fun renderReport() {
        val r = result ?: return
        val kept = labels.count { it >= 0 }
        val excluded = labels.count { it < 0 }
        val counts = labels.filter { it >= 0 }.groupingBy { it }.eachCount().toSortedMap()
        val covered = counts.keys.sorted()
        val missing = (0 until Calib.DIGITS).filter { it !in covered }
        val hue = r.calib.hueRanges.joinToString("/") { "${it[0]}~${it[1]}" }

        val lines = mutableListOf<String>()
        if (r.detected == 0) {
            lines.add("没切出字形（照片里没找到够大的笔迹）。请检查：")
            lines.add("· 纸上是不是还没写字 / 拍到了空白处")
            lines.add("· 光线是否够亮，别开闪光灯直射")
            lines.add("· 笔迹颜色是否太浅，换深一点的笔")
            lines.add("· 白纸是否占满取景框，不要拍到大片桌面/阴影")
            statusText.text = lines.joinToString("\n")
            findViewById<Button>(R.id.btnSave).isEnabled = false
            return
        }
        lines.add("检测到 ${r.detected} 处笔迹，可识别 ${r.items.size} 处；已标号 $kept 处，已排除 $excluded 处")
        if (r.detected > r.items.size) {
            lines.add("有 ${r.detected - r.items.size} 处没认出来，可能是写得太大/太扁或不是数字")
        }
        lines.add("标号分布：" + counts.entries.joinToString("  ") { "${it.key}×${it.value}" })
        lines.add("样本总数 $kept 份（同一数字写多遍样本更多）")
        lines.add(
            "颜色：色相 $hue · S≥${r.calib.satMin} · V≥${r.calib.valMin} · 通道差≥${r.calib.minRedness}\n" +
                "笔画像素 ${r.strokePixels}（掩膜按「比周围纸面更暗/更艳」取）"
        )
        lines.add(
            if (missing.isEmpty()) "0-9 已全部覆盖"
            else "未覆盖：${missing.joinToString(" ")}（统计时这些数字只用 CNN 识别）"
        )
        lines.add("点任意框：改标号，或标成「不是数字」排除掉（橙色）")
        statusText.text = lines.joinToString("\n")

        findViewById<Button>(R.id.btnSave).isEnabled = kept > 0
    }

    private fun onMarkTapped(index: Int) {
        val r = result
        if (r == null) {
            statusText.text = "这是实时预览的框，点「拍摄学习」之后才能改标号"
            return
        }
        if (index !in r.items.indices) return
        val options = ((0 until Calib.DIGITS).map { it.toString() } + "不是数字（排除）").toTypedArray()
        AlertDialog.Builder(this)
            .setTitle("第 ${index + 1} 个框里是哪个数字？")
            .setItems(options) { _, which ->
                labels[index] = if (which < Calib.DIGITS) which else -1
                paintMarks()
                renderReport()
            }
            .setNegativeButton("取消", null)
            .show()
    }

    /**
     * 保存学习：累加模式。把这张拍到的模板合并到已有档案里，不覆盖。
     * 保存后清空 result，允许用户继续拍下一张再保存（继续累加）。
     */
    private fun save() {
        val r = result ?: return
        if (r.items.isEmpty()) return
        val finalLabels = labels.toList()
        val newTpls = LinkedHashMap<Int, MutableList<ByteArray>>()
        r.items.forEachIndexed { i, item ->
            val d = finalLabels[i]
            if (d >= 0) newTpls.getOrPut(d) { mutableListOf() }.add(item.template)
        }
        if (newTpls.isEmpty()) {
            statusText.text = "所有框都被标成「不是数字」了，没有可保存的样本"
            return
        }
        // 追加模式：合并已有档案的色相 + 模板，再加上这张的新模板
        var merged = store.calib.mergeWith(r.calib)
        for ((d, list) in newTpls) {
            for (t in list) merged = merged.plusTemplate(d, t)
        }
        val toSave = merged
        work.execute {
            store.save(toSave)
            runOnUiThread {
                val added = newTpls.values.sumOf { it.size }
                val digits = newTpls.keys.sorted().joinToString(",")
                Toast.makeText(
                    this,
                    "已累加 $added 份模板（$digits），可继续拍或去统计",
                    Toast.LENGTH_SHORT
                ).show()
                // 保存后清空，允许继续拍下一张
                result = null
                labels = mutableListOf()
                history.clear()
                overlay.setMarks(emptyList())
                findViewById<Button>(R.id.btnSave).isEnabled = false
                findViewById<Button>(R.id.btnCapture).text = getString(R.string.btn_capture)
                statusText.text = "已保存，继续拍下一张或点「去统计分数」"
                frozen = false
                updateGoScanButton()
            }
        }
    }

    /** 落一份「原图 + 框 + 最终标号」，供离线调参/训练使用 */
    private fun dumpDataset(r: Calibrator.Result, finalLabels: List<Int>) {
        val dir = getExternalFilesDir(null) ?: return
        try {
            val ds = File(dir, "dataset")
            if (!ds.exists()) ds.mkdirs()
            val ts = System.currentTimeMillis()
            val frame = File(dir, "learn_frame.jpg")
            if (frame.exists()) frame.copyTo(File(ds, "cap_$ts.jpg"), overwrite = true)
            val sb = StringBuilder()
            sb.append("{\"w\":${r.sizeW},\"h\":${r.sizeH},\"boxes\":[")
            r.items.forEachIndexed { i, item ->
                if (i > 0) sb.append(',')
                sb.append(
                    "{\"x\":${item.box.x},\"y\":${item.box.y},\"w\":${item.box.width}," +
                        "\"h\":${item.box.height},\"label\":${finalLabels[i]}}"
                )
            }
            sb.append("]}")
            File(ds, "cap_$ts.json").writeText(sb.toString(), Charsets.UTF_8)
        } catch (e: Throwable) {
            android.util.Log.w(TAG, "样本落盘失败", e)
        }
    }

    private fun retake() {
        result = null
        labels = mutableListOf()
        history.clear()
        overlay.setMarks(emptyList())
        findViewById<Button>(R.id.btnSave).isEnabled = false
        findViewById<Button>(R.id.btnCapture).text = getString(R.string.btn_capture)
        statusText.text = getString(R.string.learn_hint)
        // 解冻，恢复实时出框
        frozen = false
    }

    /** 重新学习：清空已有档案（颜色 + 模板），从空白开始 */
    private fun resetLearn() {
        AlertDialog.Builder(this)
            .setTitle("重新学习？")
            .setMessage("会清除之前所有学习内容（颜色档案 + 模板），确定吗？")
            .setPositiveButton("清除") { _, _ ->
                store.clear()
                result = null
                labels = mutableListOf()
                history.clear()
                overlay.setMarks(emptyList())
                findViewById<Button>(R.id.btnSave).isEnabled = false
                findViewById<Button>(R.id.btnCapture).text = getString(R.string.btn_capture)
                statusText.text = "已清空学习档案，请重新拍照学习"
                frozen = false
                updateGoScanButton()
            }
            .setNegativeButton("取消", null)
            .show()
    }

    /** 已有学习档案时显示「去统计分数」，这样即使没保存就退出也能直接统计 */
    private fun updateGoScanButton() {
        val btn = findViewById<Button>(R.id.btnGoScan)
        if (store.calib.learned) {
            btn.visibility = android.view.View.VISIBLE
            val cov = store.calib.covered
            statusText.text = if (cov.isEmpty()) {
                "已加载上次学习数据，点「去统计分数」直接开始；或重新学习更新"
            } else {
                "已加载上次学习数据（覆盖 ${cov.size} 种：${cov.joinToString()}），点「去统计分数」直接开始；或重新学习更新"
            }
        } else {
            btn.visibility = android.view.View.GONE
        }
    }

    private fun iou(a: Rect, b: Rect): Float {
        val x1 = maxOf(a.x, b.x)
        val y1 = maxOf(a.y, b.y)
        val x2 = minOf(a.x + a.width, b.x + b.width)
        val y2 = minOf(a.y + a.height, b.y + b.height)
        if (x2 <= x1 || y2 <= y1) return 0f
        val inter = (x2 - x1) * (y2 - y1)
        val union = a.width * a.height + b.width * b.height - inter
        return if (union <= 0) 0f else inter.toFloat() / union
    }

    override fun onDestroy() {
        super.onDestroy()
        latest?.release()
        latest = null
        work.shutdown()
    }

    companion object {
        private const val TAG = "CalibrationActivity"
        private const val REQ_CAMERA = 1001
    }
}
