package com.examscan.score

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import kotlin.math.exp
import kotlin.math.sin

/**
 * 硬币扔存钱罐音效：用 AudioTrack 程序化合成，不需要外部音频文件。
 *
 * 采用 MODE_STATIC：数据一次性写入，每次 play() 用 pause()+playbackHeadPosition=0+play() 从头播放。
 *
 * 声音分两段（频率调低，更容易从手机扬声器听到）：
 * · 前段「叮」：1800Hz 主音 + 3600Hz 谐波，快速衰减
 * · 后段「当」：1000Hz，延迟 28ms 起音，衰减更慢
 */
class CoinSound {

    private var track: AudioTrack? = null
    private val sampleRate = 44100
    private val pcmData: ShortArray = synthesize()

    init {
        val t = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(pcmData.size * 2)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()
        track = t
        try {
            val written = t.write(pcmData, 0, pcmData.size)
            t.setVolume(1.0f)
            Log.i("MainActivity", "CoinSound init written=$written/${pcmData.size} state=${t.state}")
        } catch (e: Throwable) {
            Log.e("MainActivity", "CoinSound init failed", e)
        }
    }

    /** 播放一次；线程安全。pause 后重置播放头到 0 再 play。 */
    @Synchronized
    fun play() {
        val t = track ?: return
        try {
            t.pause()
            t.playbackHeadPosition = 0
            t.play()
            Log.i("MainActivity", "CoinSound play() called playState=${t.playState}")
        } catch (e: Throwable) {
            Log.w("MainActivity", "CoinSound play failed", e)
        }
    }

    fun release() {
        try {
            track?.stop()
        } catch (_: Throwable) {
        }
        track?.release()
        track = null
    }

    /**
     * 合成 16bit PCM 单声道数据。
     * 叮(2600Hz) + 当(1400Hz)，各自指数衰减，叠加后归一化到 [-1,1]。
     */
    private fun synthesize(): ShortArray {
        val durMs = 260
        val n = sampleRate * durMs / 1000
        val out = FloatArray(n)

        val dingFreq = 1800.0
        val dingDecay = 42.0
        val dongFreq = 1000.0
        val dongDecay = 20.0
        val dongDelayMs = 28

        for (i in 0 until n) {
            val t = i.toDouble() / sampleRate
            var s = 0.0
            val envDing = exp(-dingDecay * t)
            s += sin(2 * Math.PI * dingFreq * t) * envDing * 0.7
            s += sin(2 * Math.PI * dingFreq * 2 * t) * envDing * 0.2
            val td = t - dongDelayMs / 1000.0
            if (td > 0) {
                val envDong = exp(-dongDecay * td)
                s += sin(2 * Math.PI * dongFreq * td) * envDong * 0.6
            }
            out[i] = s.toFloat()
        }

        var max = 0f
        for (v in out) if (Math.abs(v) > max) max = Math.abs(v)
        if (max < 1e-6f) max = 1f
        val scale = 0.95f / max

        val pcm = ShortArray(n)
        for (i in 0 until n) {
            val v = (out[i] * scale).coerceIn(-1f, 1f)
            pcm[i] = (v * Short.MAX_VALUE).toInt().toShort()
        }
        return pcm
    }
}
