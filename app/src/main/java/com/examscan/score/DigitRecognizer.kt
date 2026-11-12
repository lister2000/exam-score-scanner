package com.examscan.score

import android.content.Context
import android.util.Log
import org.opencv.core.Mat
import org.opencv.core.Rect
import java.nio.FloatBuffer
import java.util.Collections
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

/**
 * 手写数字识别：ONNX Runtime + MNIST 训练的小 CNN。
 *
 * MNIST 是白字黑底、28x28、重心居中，预处理必须严格对齐这套规范，
 * 否则精度断崖式下降。toCanvas 与 server/recognizer.py 逐行等价。
 */
class DigitRecognizer(context: Context) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession?
    private val inputName: String

    var ready = false
        private set

    init {
        var s: OrtSession? = null
        var name = "input"
        try {
            val bytes = context.assets.open("digit_cnn.onnx").use { it.readBytes() }
            val opts = OrtSession.SessionOptions()
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            opts.setIntraOpNumThreads(2)
            s = env.createSession(bytes, opts)
            name = s.inputNames.iterator().next()
            ready = true
            Log.i(TAG, "模型就绪，输入=$name")
        } catch (e: Throwable) {
            Log.e(TAG, "模型加载失败", e)
        }
        session = s
        inputName = name
    }

    /**
     * 基础识别：仅走 ONNX CNN。返回 (digit, confidence)；识别失败返回 (null, 0)。
     * 这是兼容入口，内部会复用 runCnn 做 softmax + canvas 提取。
     */
    fun recognize(mask: Mat, box: Rect): Pair<Int?, Float> {
        val (probs, _) = runCnn(mask, box) ?: return Pair(null, 0f)
        return bestOf(probs)
    }

    /**
     * 带学习模板的识别：CNN softmax × 模板余弦相似度加权融合。
     * 模板来自 Calib.templates，是用户在统计页/学习页逐步追加的 28x28 字形样本。
     * 对每个有模板的数字 d：weight[d] = 0.5 * cnnProb[d] + 0.5 * max(sim(input, tpl) for tpl in templates[d])
     * 对没模板的数字：weight[d] = cnnProb[d]（纯 CNN）
     * 返回 argmax(weight), conf = max(weight)
     */
    fun recognize(mask: Mat, box: Rect, templates: Map<Int, List<ByteArray>>): Pair<Int?, Float> {
        if (templates.isEmpty()) return recognize(mask, box)
        val (probs, canvasBytes) = runCnn(mask, box) ?: return Pair(null, 0f)

        // 融合权重
        val weights = FloatArray(probs.size) { probs[it] }  // 先抄一份 CNN 概率
        for ((d, tpls) in templates) {
            if (d !in weights.indices || tpls.isEmpty()) continue
            var maxSim = -1f
            for (tpl in tpls) {
                val s = cosSim(canvasBytes, tpl)
                if (s > maxSim) maxSim = s
            }
            weights[d] = probs[d] * 0.5f + maxSim * 0.5f
            Log.d(TAG, "  tpl[$d]: cnn=${"%.2f".format(probs[d])} sim=${"%.2f".format(maxSim)} -> w=${"%.2f".format(weights[d])}")
        }

        var best = 0; var bestV = weights[0]
        for (i in weights.indices) if (weights[i] > bestV) { bestV = weights[i]; best = i }
        Log.i(TAG, "glyph ${box.x},${box.y},${box.width},${box.height} -> $best ${(bestV * 100).toInt()}% (with templates)")
        return Pair(best, bestV)
    }

    /** 跑 CNN + 提取 784 字节 canvas；返回 (softmax 概率数组, 归一化后的 28x28 字节)，失败返回 null */
    private fun runCnn(mask: Mat, box: Rect): Pair<FloatArray, ByteArray>? {
        val sess = session ?: return null
        val canvas = GlyphCanvas.from(mask, box) ?: return null
        try {
            val pixels = ByteArray(28 * 28)
            canvas.get(0, 0, pixels)
            // 拷贝一份给模板比对（pixels 是 canvas 原始字节，0~255）
            val canvasBytes = pixels.copyOf()

            val fb = FloatBuffer.allocate(28 * 28)
            for (p in pixels) {
                val v = (p.toInt() and 0xFF).toFloat()
                fb.put((v / 255.0f - MEAN) / STD)
            }
            fb.rewind()

            OnnxTensor.createTensor(env, fb, longArrayOf(1, 1, 28, 28)).use { tensor ->
                sess.run(Collections.singletonMap(inputName, tensor)).use { result ->
                    @Suppress("UNCHECKED_CAST")
                    val logits = (result.get(0).value as Array<FloatArray>)[0]
                    // softmax
                    var maxV = logits[0]
                    for (v in logits) if (v > maxV) maxV = v
                    val probs = FloatArray(logits.size)
                    var sum = 0.0
                    for (i in logits.indices) {
                        probs[i] = Math.exp((logits[i] - maxV).toDouble()).toFloat()
                        sum += probs[i]
                    }
                    for (i in probs.indices) probs[i] /= sum.toFloat()
                    return Pair(probs, canvasBytes)
                }
            }
        } catch (e: Throwable) {
            Log.e(TAG, "推理失败", e)
            return null
        } finally {
            canvas.release()
        }
    }

    private fun bestOf(probs: FloatArray): Pair<Int, Float> {
        var best = 0; var bestV = probs[0]
        for (i in probs.indices) if (probs[i] > bestV) { bestV = probs[i]; best = i }
        return Pair(best, bestV)
    }

    /** 两个 784 字节 28x28 灰度图的余弦相似度，范围 0~1（越高越像） */
    private fun cosSim(a: ByteArray, b: ByteArray): Float {
        var dot = 0.0; var normA = 0.0; var normB = 0.0
        var i = 0
        while (i < a.size) {
            val va = (a[i].toInt() and 0xFF).toDouble()
            val vb = (b[i].toInt() and 0xFF).toDouble()
            dot += va * vb
            normA += va * va
            normB += vb * vb
            i++
        }
        val denom = Math.sqrt(normA) * Math.sqrt(normB)
        return if (denom < 1e-6) 0f else (dot / denom).toFloat()
    }

    companion object {
        private const val TAG = "DigitRecognizer"
        private const val MEAN = 0.1307f
        private const val STD = 0.3081f
    }
}
