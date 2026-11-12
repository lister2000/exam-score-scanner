package com.examscan.score

import android.content.Context
import android.media.MediaPlayer
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.Locale

/**
 * 语音播报。
 *
 * 优先用系统 TTS（音质好、任意数字都能说）；部分国产 ROM 没装中文语音包，
 * 这时退回 assets/audio 里预生成的「N分」mp3（覆盖 0~100）。
 */
class Speaker(private val context: Context) {

    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private var chineseOk = false
    private var player: MediaPlayer? = null

    fun start() {
        try {
            tts = TextToSpeech(context) { status ->
                if (status == TextToSpeech.SUCCESS) {
                    val r = tts?.setLanguage(Locale.CHINA)
                    chineseOk = r != TextToSpeech.LANG_MISSING_DATA &&
                        r != TextToSpeech.LANG_NOT_SUPPORTED
                    ttsReady = true
                    Log.i(TAG, "TTS 就绪, 中文=$chineseOk")
                } else {
                    Log.w(TAG, "TTS 初始化失败，使用 mp3 兜底")
                }
            }
        } catch (e: Throwable) {
            Log.w(TAG, "TTS 不可用", e)
        }
    }

    /** 播报单个分数：「N分」 */
    fun speakScore(value: Int) {
        if (ttsReady && chineseOk) {
            tts?.speak("$value 分", TextToSpeech.QUEUE_ADD, null, "score$value")
            return
        }
        playAsset(value)
    }

    /** 播报当前总分 */
    fun speakTotal(total: Int) {
        if (ttsReady && chineseOk) {
            tts?.speak("当前总分 $total 分", TextToSpeech.QUEUE_ADD, null, "total$total")
            return
        }
        playAsset(total)
    }

    /** 一次性播报加法式，例如 [3,5,8] + 总分16 → "3 加 5 加 8 等于 16 分" */
    fun speakAddition(values: List<Int>, totalAfter: Int) {
        if (ttsReady && chineseOk) {
            val addStr = values.joinToString(" 加 ")
            tts?.speak("$addStr 等于 $totalAfter 分", TextToSpeech.QUEUE_ADD, null, "add$totalAfter")
            return
        }
        // 没 TTS 时逐个播数字，最后补个总分
        values.forEach { speakScore(it) }
        speakTotal(totalAfter)
    }

    private fun playAsset(value: Int) {
        if (value !in 0..100) return          // assets 只预生成了 0~100
        try {
            player?.release()
            val afd = context.assets.openFd("audio/$value.mp3")
            val mp = MediaPlayer()
            mp.setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)
            mp.setOnCompletionListener { p ->
                p.release()
                afd.close()
                if (player === p) player = null
            }
            mp.prepare()
            mp.start()
            player = mp
        } catch (e: Throwable) {
            Log.w(TAG, "mp3 播放失败: $value", e)
        }
    }

    fun release() {
        try {
            tts?.stop()
            tts?.shutdown()
        } catch (_: Throwable) {
        }
        tts = null
        try {
            player?.release()
        } catch (_: Throwable) {
        }
        player = null
    }

    companion object {
        private const val TAG = "Speaker"
    }
}
