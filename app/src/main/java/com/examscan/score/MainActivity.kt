package com.examscan.score

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.TextPaint
import android.text.method.LinkMovementMethod
import android.text.style.ClickableSpan
import android.util.Log
import android.util.Size
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.Mat
import java.util.concurrent.Executors
import kotlin.math.abs
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var totalText: TextView
    private lateinit var subText: TextView
    private lateinit var shotFormula: TextView
    private lateinit var hintText: TextView
    private lateinit var btnShutter: Button
    private lateinit var btnRetake: Button
    private lateinit var btnAddLearn: Button

    private val session = SessionState()
    private lateinit var recognizer: DigitRecognizer
    private lateinit var calibration: CalibrationStore
    private lateinit var coinSound: CoinSound
    private lateinit var settings: ScanSettings

    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())
    private val busy = AtomicBoolean(false)

    /** 最新一帧的 BGR（竖屏已旋转），拍摄时用它来跑 scan */
    @Volatile private var latestFrame: Mat? = null
    /** 竖屏后画面的宽高比，给 overlay 用 */
    @Volatile private var latestAspect: Float = 0.75f
    /** 最后一次拍照的画面帧，供「添加学习」提取字形时用 */
    @Volatile private var lastShotFrame: Mat? = null

    /** 所有已确认的数字框（绿色，累积显示，可点击修改），直到重新开始才清空 */
    private val confirmedItems = mutableListOf<ConfirmedItem>()
    /** 最后一次拍照加入的 confirmedItems 数量，重拍时整批移除 */
    private var lastShotCount: Int = 0
    /** 实时跟踪检测到的当前帧框（橙色，每帧更新，不累计） */
    private var liveItems: List<ScorePipeline.Detected> = emptyList()
    /** 上一帧检测到的框，给小框做连续两帧确认 */
    private var prevLiveBoxes: List<ScorePipeline.Detected> = emptyList()

    /** 累计「已拍次数」，用于 subText */
    private var shotCount: Int = 0
    /** 实时预览扫描的节流时间戳，防止每帧都跑 CNN（太重） */
    @Volatile private var lastLiveScanTs: Long = 0L

    /** overlay 不再画绿色确认框，点框无需响应；数字改在上方加法式子里点 */
    private val onMarkTap: (Int) -> Unit = { }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        previewView.scaleType = PreviewView.ScaleType.FIT_CENTER
        overlay = findViewById(R.id.overlay)
        overlay.onMarkTap = onMarkTap
        totalText = findViewById(R.id.totalText)
        subText = findViewById(R.id.subText)
        shotFormula = findViewById(R.id.shotFormula)
        hintText = findViewById(R.id.hintText)
        btnShutter = findViewById(R.id.btnShutter)
        btnRetake = findViewById(R.id.btnRetake)
        btnAddLearn = findViewById(R.id.btnAddLearn)

        overlay.setFrame(latestAspect)

        if (!OpenCVLoader.initLocal()) {
            Toast.makeText(this, "OpenCV 初始化失败", Toast.LENGTH_LONG).show()
        }
        recognizer = DigitRecognizer(this)
        calibration = CalibrationStore(this)
        coinSound = CoinSound()
        settings = ScanSettings(this)
        overlay.roiRect = settings.roiRect
        Log.i(TAG, "CoinSound created")

        // 圆形拍摄按钮：点一下就把当前实时框确认为绿色（累计 + 加法式 + 动画 + 音效），继续跟踪
        btnShutter.setOnClickListener { onShutter() }
        btnRetake.setOnClickListener { onRetake() }
        btnAddLearn.setOnClickListener { onAddLearn() }
        findViewById<Button>(R.id.btnSettings).setOnClickListener { showSettingsDialog() }
        findViewById<Button>(R.id.btnGoLearn).setOnClickListener {
            startActivity(android.content.Intent(this, CalibrationActivity::class.java))
        }
        findViewById<Button>(R.id.btnReset).setOnClickListener { onReset() }

        updateSummary()
        if (hasCameraPermission()) startCamera() else requestCamera()
    }

    // ------------------------------------------------------------ 相机

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

    private fun requestCamera() {
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), REQ_CAMERA)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_CAMERA) {
            if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) startCamera()
            else Toast.makeText(this, R.string.permission_need, Toast.LENGTH_LONG).show()
        }
    }

    private var providerRef: ProcessCameraProvider? = null

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                providerRef = provider

                val preview = Preview.Builder()
                    .setTargetResolution(Size(1280, 960))
                    .build()
                preview.setSurfaceProvider(previewView.surfaceProvider)

                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(1280, 960))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                // 实时态：只存最新帧，不做 scan，避免重复统计
                analysis.setAnalyzer(analysisExecutor) { image -> storeFrame(image) }

                provider.unbindAll()
                provider.bindToLifecycle(
                    this@MainActivity, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis
                )
                showHint(getString(R.string.hint_default))
            } catch (e: Throwable) {
                Log.e(TAG, "相机启动失败", e)
                Toast.makeText(this@MainActivity, "相机启动失败：${e.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /** 存帧 + 实时节流 scan（橙色 PENDING 框），confirmedItems（绿色）始终保留 */
    private fun storeFrame(image: ImageProxy) {
        val bgr = Frames.toBgr(image)
        if (bgr != null) {
            val rot = Mat()
            when (image.imageInfo.rotationDegrees) {
                90 -> Core.rotate(bgr, rot, Core.ROTATE_90_CLOCKWISE)
                180 -> Core.rotate(bgr, rot, Core.ROTATE_180)
                270 -> Core.rotate(bgr, rot, Core.ROTATE_90_COUNTERCLOCKWISE)
                else -> bgr.copyTo(rot)
            }
            bgr.release()
            val old = latestFrame
            latestFrame = rot
            latestAspect = rot.cols().toFloat() / rot.rows()
            runOnUiThread { overlay.setFrame(latestAspect) }
            old?.release()

            // 节流实时扫描：橙色框给用户预览确认
            val now = System.currentTimeMillis()
            if (now - lastLiveScanTs >= settings.liveIntervalMs) {
                lastLiveScanTs = now
                val frameClone = rot.clone()
                analysisExecutor.execute {
                    try {
                        val sharp = ScorePipeline.sharpness(frameClone)
                        if (sharp >= settings.blurThreshold) {
                            val detected = ScorePipeline.scan(
                                frameClone, recognizer, calibration.calib, settings.tuning(),
                                roi = settings.roiRect, filterSmall = true
                            )
                            val aspect = frameClone.cols().toFloat() / frameClone.rows()
                            runOnUiThread {
                                overlay.setFrame(aspect)
                                liveItems = filterLiveSmall(detected)
                                renderOverlay()
                            }
                        }
                    } catch (e: Throwable) {
                        Log.w(TAG, "live scan failed", e)
                    } finally {
                        frameClone.release()
                    }
                }
            }
        }
        image.close()
    }

    // ------------------------------------------------------------ 拍摄 + 单次识别

    /**
     * 拍摄：把当前实时检测到的框确认为绿色（计入总分），然后继续实时跟踪。
     * confirmedItems 累积保留，直到「重新开始」才清空。
     */
    private fun onShutter() {
        Log.i(TAG, "onShutter clicked busy=${busy.get()}")
        if (busy.get()) return
        busy.set(true)
        val frame = latestFrame?.clone()
        if (frame == null || frame.empty()) {
            busy.set(false)
            showHint("画面还没准备好，等一下再拍")
            return
        }
        showHint("分析中…")

        analysisExecutor.execute {
            var detected = emptyList<ScorePipeline.Detected>()
            try {
                val sharp = ScorePipeline.sharpness(frame)
                if (sharp < settings.blurThreshold) {
                    runOnUiThread { showHint("画面太模糊，稳住手机重拍") }
                } else {
                    detected = ScorePipeline.scan(
                        frame, recognizer, calibration.calib, settings.tuning(),
                        roi = settings.roiRect
                    )
                    Log.i(TAG, "scan ${frame.cols()}x${frame.rows()} -> ${detected.size} hits, values=${detected.map { it.value }}")
                }
            } catch (e: Throwable) {
                Log.e(TAG, "scan failed", e)
                runOnUiThread { showHint("识别出错：${e.message}") }
            }

            val newConfirmed = detected
                .sortedWith(compareBy({ it.ny }, { it.nx }))
                .map { ConfirmedItem(value = it.value, nx = it.nx, ny = it.ny, nw = it.nw, nh = it.nh) }

            runOnUiThread {
                // 保存这帧供「添加学习」用
                lastShotFrame?.release()
                lastShotFrame = frame

                val valid = newConfirmed.filter { it.value != DELETED }
                if (valid.isNotEmpty()) {
                    val values = valid.map { it.value }
                    session.addAll(values)
                    confirmedItems.addAll(newConfirmed)
                    lastShotCount = newConfirmed.size
                    shotCount++
                    updateFormula()
                    for (item in valid) {
                        overlay.addFloatText(signedText(item.value), item.nx + item.nw / 2, item.ny + item.nh / 2)
                    }
                    // 几个数字就响几声硬币，一个接一个（间隔 280ms，够一声播完）
                    playCoinSequence(values.size)
                    showHint("本次 ${values.joinToString(" ") { signedText(it) }}，总分 ${session.total}")
                } else {
                    showHint("没识别到数字")
                }
                renderOverlay()
                updateSummary()
                busy.set(false)
            }
        }
    }

    /** 只画实时跟踪的橙色框；已确认数字不画框，改在上方加法式子里点击修改 */
    private fun renderOverlay() {
        val marks = liveItems.map { d ->
            Mark(
                value = d.value,
                nx = d.nx, ny = d.ny, nw = d.nw, nh = d.nh,
                status = MarkStatus.PENDING,
                label = d.value.toString()
            )
        }
        overlay.setMarks(marks)
    }

    /**
     * 小框要连续两帧都在同一位置才画：只闪一帧的多半是噪点/污点。
     * 明显是个字的框（高度 >= 画面 4%）当帧就画，不给正常跟踪加延迟。
     */
    private fun filterLiveSmall(list: List<ScorePipeline.Detected>): List<ScorePipeline.Detected> {
        val shown = list.filter { d ->
            d.nh >= 0.04f || prevLiveBoxes.any { p -> sameBox(d, p) }
        }
        prevLiveBoxes = list
        return shown
    }

    /** 两帧里的框是不是同一个字形：中心靠近、字高接近（手抖也能对上） */
    private fun sameBox(a: ScorePipeline.Detected, b: ScorePipeline.Detected): Boolean {
        val acx = a.nx + a.nw / 2f
        val acy = a.ny + a.nh / 2f
        val bcx = b.nx + b.nw / 2f
        val bcy = b.ny + b.nh / 2f
        val span = maxOf(maxOf(a.nw, a.nh), maxOf(b.nw, b.nh))
        if (abs(acx - bcx) > 0.45f * span || abs(acy - bcy) > 0.45f * span) return false
        val hi = maxOf(a.nh, b.nh)
        val lo = minOf(a.nh, b.nh)
        return lo > 0f && hi <= 1.8f * lo
    }

    /**
     * 顶部加法式子：显示所有已确认的数字，每个数字可点击修改或删除。
     * 用 ClickableSpan 给每个数字绑点击事件，点击时弹窗。
     */
    private fun updateFormula() {
        // 收集 value>=0 的项及其在 confirmedItems 中的原始索引
        val validList = confirmedItems.withIndex().filter { it.value.value != DELETED }
        if (validList.isEmpty()) {
            shotFormula.text = ""
            shotFormula.visibility = View.GONE
            return
        }
        val total = validList.sumOf { it.value.value }
        val sb = SpannableStringBuilder()
        validList.forEachIndexed { k, iv ->
            if (k > 0) sb.append(" + ")
            val start = sb.length
            sb.append(iv.value.value.toString())
            val end = sb.length
            val confirmedIdx = iv.index
            sb.setSpan(
                object : ClickableSpan() {
                    override fun onClick(widget: View) {
                        showConfirmedDigitDialog(confirmedIdx)
                    }
                    override fun updateDrawState(ds: TextPaint) {
                        super.updateDrawState(ds)
                        ds.color = Color.parseColor("#22c55e")
                        ds.isUnderlineText = false
                        ds.isFakeBoldText = true
                    }
                },
                start, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
            )
        }
        sb.append(" = $total")
        shotFormula.text = sb
        shotFormula.movementMethod = LinkMovementMethod.getInstance()
        shotFormula.highlightColor = Color.TRANSPARENT
        shotFormula.visibility = View.VISIBLE
    }

    /**
     * 连续播放 count 声硬币音效，一声接一声。
     * 用独立线程顺序播放，一声接一声。
     */
    private fun playCoinSequence(count: Int) {
        Log.i(TAG, "playCoinSequence count=$count")
        if (!settings.coinSound) return
        // 检查媒体音量，为 0 时提示用户
        val am = getSystemService(android.media.AudioManager::class.java)
        val vol = am?.getStreamVolume(android.media.AudioManager.STREAM_MUSIC) ?: 0
        if (vol == 0) {
            runOnUiThread { showHint("媒体音量为 0，请按音量+键开大声音") }
        }
        // 用独立线程顺序播放，一声接一声，不依赖 mainHandler.postDelayed
        Thread {
            try {
                repeat(count) {
                    coinSound.play()
                    Thread.sleep(COIN_INTERVAL_MS.toLong())
                }
            } catch (e: Throwable) {
                Log.e(TAG, "coin sequence interrupted", e)
            }
        }.start()
    }

    /**
     * 重拍：移除最后一次拍照加入的 confirmedItems，扣回对应的分数。
     */
    private fun onRetake() {
        if (lastShotCount <= 0 || confirmedItems.size < lastShotCount) {
            showHint("没有可重拍的")
            return
        }
        val removed = confirmedItems.takeLast(lastShotCount)
        val removedValues = removed.filter { it.value != DELETED }.map { it.value }
        confirmedItems.subList(confirmedItems.size - lastShotCount, confirmedItems.size).clear()
        session.undoLast(removedValues.size)
        lastShotCount = 0
        lastShotFrame?.release()
        lastShotFrame = null
        updateFormula()
        renderOverlay()
        updateSummary()
        showHint("已重拍，扣除 ${removedValues.joinToString("+")}，总分 ${session.total}")
    }

    // ------------------------------------------------------------ 加入学习档案

    /**
     * 把最后一次拍照确认的数字追加入学习档案。
     * 只会加 0-9（多位数自动按每一位拆开；标成「非数字」的跳过）。
     */
    private fun onAddLearn() {
        val frame = lastShotFrame
        if (frame == null || frame.empty()) {
            Toast.makeText(this, "先拍一张再添加学习", Toast.LENGTH_SHORT).show()
            return
        }
        val items = confirmedItems
            .filter { it.value != DELETED && kotlin.math.abs(it.value) in 0..9 }
            .map { CalibrationStore.AddShotItem(it.nx, it.ny, it.nw, it.nh, kotlin.math.abs(it.value)) }

        if (items.isEmpty()) {
            Toast.makeText(this, "当前没有 0-9 的数字可加入学习（点框可改标号）", Toast.LENGTH_SHORT).show()
            return
        }

        val cloned = frame.clone()
        busy.set(true)
        analysisExecutor.execute {
            var n = 0
            try {
                n = calibration.addShot(cloned, items)
            } catch (e: Throwable) {
                Log.e(TAG, "addShot failed", e)
            } finally {
                cloned.release()
            }
            runOnUiThread {
                busy.set(false)
                val digits = items.map { it.digit }.distinct().sorted().joinToString(",")
                if (n > 0)
                    Toast.makeText(
                        this,
                        "已追加 $n 份模板（覆盖 $digits），下次识别会用上",
                        Toast.LENGTH_SHORT
                    ).show()
                else
                    Toast.makeText(this, "这次没追加到新模板（框太小或识别失败）", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // ------------------------------------------------------------ 改框标号

    /**
     * 点击上方加法式子里的某个数字时弹出：用键盘输入新数字，或留空表示删除。
     * confirmedIdx 是该数字在 confirmedItems 中的原始索引。
     */
    private fun showConfirmedDigitDialog(confirmedIdx: Int) {
        val current = confirmedItems[confirmedIdx].value
        val input = EditText(this).apply {
            inputType = android.text.InputType.TYPE_CLASS_NUMBER or
                android.text.InputType.TYPE_NUMBER_FLAG_SIGNED
            setText(current.toString())
            setSelection(text.length)
            hint = "留空则删除这个数字；可输入负数，如 -5"
            setPadding(48, 24, 48, 24)
        }
        AlertDialog.Builder(this)
            .setTitle("修改数字（原为 $current）")
            .setView(input)
            .setPositiveButton("确定") { _, _ ->
                val text = input.text.toString().trim()
                val parsed = text.toIntOrNull()
                confirmedItems[confirmedIdx].value =
                    if (text.isEmpty() || parsed == null) DELETED
                    else parsed.coerceIn(-999, 999)
                applyLabelChange()
            }
            .setNegativeButton("取消", null)
            .show()
    }

    /**
     * 改标号后：清空 session 重新计入所有 confirmedItems，
     * 同步更新总分、加法式子和框显示。
     */
    private fun applyLabelChange() {
        session.reset()
        val all = confirmedItems.filter { it.value != DELETED }.map { it.value }
        session.addAll(all)
        updateFormula()
        renderOverlay()
        updateSummary()
        showHint("已修改，总分 ${session.total}")
    }

    // ------------------------------------------------------------ 预览态交互

    private fun onReset() {
        AlertDialog.Builder(this)
            .setTitle("重新开始？")
            .setMessage("当前已拍 $shotCount 次、共 ${session.digits.size} 处，总分 ${session.total}。确认后全部清零。")
            .setPositiveButton("重新开始") { _, _ ->
                session.reset()
                confirmedItems.clear()
                prevLiveBoxes = emptyList()
                lastShotCount = 0
                lastShotFrame?.release()
                lastShotFrame = null
                shotCount = 0
                shotFormula.visibility = View.GONE
                overlay.setMarks(emptyList())
                updateSummary()
                showHint("已清空，开始新的一份")
            }
            .setNegativeButton("取消", null)
            .show()
    }

    private fun updateSummary() {
        totalText.text = "总分 ${session.total}"
        subText.text = "已拍 $shotCount 次 · 共 ${session.digits.size} 处"
    }

    // ------------------------------------------------------------ 设置

    /**
     * 右上角齿轮：识别规则设置。
     * 每一项都是滑杆，改完立即写入 SharedPreferences，下一帧识别就用新参数。
     */
    private fun showSettingsDialog() {
        var dialog: AlertDialog? = null
        val scroll = ScrollView(this)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(4), dp(20), dp(4))
        }
        scroll.addView(root)

        fun addTitle(t: String) {
            root.addView(TextView(this).apply {
                text = t
                textSize = 13f
                setTextColor(Color.parseColor("#888888"))
                setPadding(0, dp(14), 0, 0)
            })
        }

        fun addSeek(
            title: String,
            min: Float,
            max: Float,
            step: Float,
            cur: Float,
            fmt: (Float) -> String,
            save: (Float) -> Unit
        ) {
            val label = TextView(this).apply {
                text = "$title：${fmt(cur)}"
                textSize = 15f
                setTextColor(Color.parseColor("#222222"))
                setPadding(0, dp(10), 0, 0)
            }
            val bar = SeekBar(this).apply {
                this.max = ((max - min) / step).toInt()
                progress = ((cur - min) / step).toInt().coerceIn(0, this.max)
            }
            bar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                    val v = min + progress * step
                    label.text = "$title：${fmt(v)}"
                    if (fromUser) save(v)
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {}
            })
            root.addView(label)
            root.addView(bar)
        }

        addTitle("数字识别区域")
        val roiStatus = TextView(this).apply {
            textSize = 14f
            setTextColor(Color.parseColor("#555555"))
            setPadding(0, dp(10), 0, 0)
        }
        fun refreshRoiStatus() {
            val r = settings.roiRect
            roiStatus.text = if (r == null) {
                "未设置：全屏都识别"
            } else {
                "已设置：左${(r[0]*100).toInt()}% 上${(r[1]*100).toInt()}%  " +
                    "宽${((r[2]-r[0])*100).toInt()}% 高${((r[3]-r[1])*100).toInt()}%"
            }
        }
        refreshRoiStatus()
        root.addView(roiStatus)

        val roiBtnRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, dp(6), 0, 0)
        }
        roiBtnRow.addView(Button(this).apply {
            text = "框选区域"
            textSize = 13f
            setOnClickListener {
                // 关掉设置面板，进入划区模式
                dialog?.dismiss()
                startRoiDraw()
            }
        })
        roiBtnRow.addView(Button(this).apply {
            text = "清除区域"
            textSize = 13f
            setOnClickListener {
                settings.setRoi(null)
                this@MainActivity.overlay.roiRect = null
                refreshRoiStatus()
                Toast.makeText(this@MainActivity, "已清除识别区域（改为全屏识别）", Toast.LENGTH_SHORT).show()
            }
        })
        root.addView(roiBtnRow)

        addTitle("多位数合并")
        addSeek(
            "最多识别位数", 1f, 4f, 1f, settings.maxDigits.toFloat(),
            { "${it.toInt()} 位" }, { settings.maxDigits = it.toInt() })
        addSeek(
            "数字间距（数字宽度的倍数）", 0.4f, 2.0f, 0.05f, settings.mergeGapRatio,
            { "%.2f 倍".format(it) }, { settings.mergeGapRatio = it })
        addSeek(
            "同一行竖直重叠要求", 0.3f, 0.9f, 0.05f, settings.rowOverlap,
            { "%.2f".format(it) }, { settings.rowOverlap = it })
        addSeek(
            "字号相似上限", 1.2f, 2.5f, 0.1f, settings.sizeSimilar,
            { "%.1f 倍".format(it) }, { settings.sizeSimilar = it })

        addTitle("识别严格度")
        addSeek(
            "置信度下限", 0.30f, 0.90f, 0.05f, settings.minConf,
            { "%.2f".format(it) }, { settings.minConf = it })

        addTitle("负数识别")
        val negSwitch = SwitchCompat(this).apply {
            text = "识别负数（数字左边的横杠当负号）"
            textSize = 15f
            isChecked = settings.negEnabled
            setPadding(0, dp(10), 0, dp(4))
        }
        negSwitch.setOnCheckedChangeListener { _, checked ->
            settings.negEnabled = checked
            lastLiveScanTs = 0
        }
        root.addView(negSwitch)
        addSeek(
            "负号与数字的最大间距（数字高度倍数）", 0.1f, 1.5f, 0.05f, settings.negGapRatio,
            { "%.2f 倍".format(it) }, { settings.negGapRatio = it })

        addTitle("性能与画面")
        addSeek(
            "实时框刷新间隔", 150f, 600f, 50f, settings.liveIntervalMs.toFloat(),
            { "${it.toInt()} ms" }, { settings.liveIntervalMs = it.toInt() })
        addSeek(
            "模糊帧阈值", 2f, 20f, 1f, settings.blurThreshold,
            { "%.0f".format(it) }, { settings.blurThreshold = it })

        addTitle("反馈")
        val switch = SwitchCompat(this).apply {
            text = "硬币音效"
            textSize = 15f
            isChecked = settings.coinSound
            setPadding(0, dp(10), 0, dp(10))
        }
        switch.setOnCheckedChangeListener { _, checked -> settings.coinSound = checked }
        root.addView(switch)

        dialog = AlertDialog.Builder(this)
            .setTitle("识别规则")
            .setView(scroll)
            .setPositiveButton("完成", null)
            .setNeutralButton("恢复默认") { _, _ ->
                settings.reset()
                overlay.roiRect = null
                lastLiveScanTs = 0
                Toast.makeText(this, "已恢复默认值，重新打开设置可查看", Toast.LENGTH_SHORT).show()
            }
            .setOnDismissListener { lastLiveScanTs = 0 }
            .create()
        dialog?.show()
    }

    /** 进入「调整识别区」模式：显示默认矩形，拖角/边调整，点确定保存 */
    private fun startRoiDraw() {
        hintText.visibility = View.GONE
        Toast.makeText(this, "拖动角点或边缘调整区域，点「确定」保存", Toast.LENGTH_LONG).show()

        // 进入调整模式（传入已保存的矩形，没有则用默认居中矩形）
        overlay.startRoiAdjust(settings.roiRect) { }

        // 临时把拍照按钮改成「确定」、重拍按钮改成「取消」
        val origShutterText = btnShutter.text
        val origRetakeText = btnRetake.text
        btnShutter.text = "确定"
        btnRetake.text = "取消"

        btnShutter.setOnClickListener {
            // 保存当前矩形
            val rect = overlay.getRoiAdjustRect()
            settings.setRoi(rect)
            overlay.roiRect = settings.roiRect
            overlay.cancelRoiAdjust()
            // 恢复按钮
            btnShutter.text = origShutterText
            btnRetake.text = origRetakeText
            btnShutter.setOnClickListener { onShutter() }
            btnRetake.setOnClickListener { onRetake() }
            hintText.visibility = View.VISIBLE
            lastLiveScanTs = 0
            showHint("识别区域已设置")
        }
        btnRetake.setOnClickListener {
            overlay.cancelRoiAdjust()
            btnShutter.text = origShutterText
            btnRetake.text = origRetakeText
            btnShutter.setOnClickListener { onShutter() }
            btnRetake.setOnClickListener { onRetake() }
            hintText.visibility = View.VISIBLE
            showHint("已取消")
        }
    }

    // ------------------------------------------------------------ 小工具

    private fun showHint(msg: String) {
        hintText.text = msg
        hintText.visibility = View.VISIBLE
        mainHandler.removeCallbacksAndMessages(null)
        val r = Runnable { hintText.visibility = View.GONE }
        mainHandler.postDelayed(r, 2600)
    }

    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()

    /** 上飘数字：正数带 +，负数原样（-5） */
    private fun signedText(v: Int) = if (v >= 0) "+$v" else v.toString()

    override fun onResume() {
        super.onResume()
        calibration.reload()
        startCamera()
    }

    override fun onPause() {
        super.onPause()
        latestFrame?.release()
        latestFrame = null
    }

    override fun onDestroy() {
        super.onDestroy()
        latestFrame?.release()
        lastShotFrame?.release()
        analysisExecutor.shutdown()
        coinSound.release()
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val REQ_CAMERA = 101
        /** 被用户删掉的数字用这个哨兵值（-1 现在是合法的负数，不能再用） */
        private const val DELETED = Int.MIN_VALUE
        /** 连续硬币音效之间的间隔（毫秒），280ms 够一声 220ms 播完 */
        private const val COIN_INTERVAL_MS = 280
    }
}

/** 已确认的数字框，可由用户改成别的数字或排除（value=-1），累积显示直到重新开始 */
private data class ConfirmedItem(
    var value: Int,
    val nx: Float, val ny: Float, val nw: Float, val nh: Float
)
