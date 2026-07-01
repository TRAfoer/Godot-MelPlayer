using Godot;
using System;
using System.IO;
using System.Text;

/// <summary>
/// 音频编码器 — 将 PCM float 数据编码为音频文件 (WAV)。
///
/// === 功能 ===
/// - PCM float → WAV 文件 (8/16/24/32bit)
/// - PCM float → WAV byte[] (内存)
/// - PcmData → 直接存盘
///
/// === 使用示例 ===
/// <code>
/// // 解码 → 处理 → 编码
/// var pcm = AudioDecoder.DecodeFile("song.mp3");
/// // ... 处理 Samples ...
/// AudioEncoder.SaveWav("output.wav", pcm.Samples, pcm.SampleRate, pcm.Channels, 16);
///
/// // 或直接使用 PcmData
/// AudioEncoder.SaveWav("output.wav", pcm);
/// </code>
/// </summary>
public partial class AudioEncoder : GodotObject
{
    /// <summary>默认位深。</summary>
    public const int DEFAULT_BITS = 16;

    // ══════════════════════════════════════════════════════════════
    // 公共入口 — 存盘
    // ══════════════════════════════════════════════════════════════

    /// <summary>将 PcmData 编码为 WAV 文件。</summary>
    public static void SaveWav(string filePath, PcmData pcm, int bitsPerSample = DEFAULT_BITS)
    {
        byte[] wav = EncodeToWavBytes(pcm.Samples, pcm.SampleRate, pcm.Channels, bitsPerSample);
        File.WriteAllBytes(filePath, wav);
    }

    /// <summary>将 PCM float 数组编码为 WAV 文件。</summary>
    /// <param name="filePath">输出路径。</param>
    /// <param name="samples">交错存储的 float 采样 [-1, 1]。</param>
    /// <param name="sampleRate">采样率 (Hz)。</param>
    /// <param name="channels">声道数。</param>
    /// <param name="bitsPerSample">位深 (8/16/24/32)。</param>
    public static void SaveWav(string filePath, float[] samples, int sampleRate,
                                int channels, int bitsPerSample = DEFAULT_BITS)
    {
        byte[] wav = EncodeToWavBytes(samples, sampleRate, channels, bitsPerSample);
        File.WriteAllBytes(filePath, wav);
    }

    // ══════════════════════════════════════════════════════════════
    // 公共入口 — 内存编码
    // ══════════════════════════════════════════════════════════════

    /// <summary>将 PCM float 数组编码为 WAV 格式的 byte 数组。</summary>
    public static byte[] EncodeToWavBytes(float[] samples, int sampleRate,
                                           int channels, int bitsPerSample = DEFAULT_BITS)
    {
        if (samples == null || samples.Length == 0)
            throw new ArgumentException("采样数据不能为空");
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate));
        if (channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(channels));

        int validBits = bitsPerSample switch
        {
            8  => 8,
            16 => 16,
            24 => 24,
            32 => 32,
            _ => throw new ArgumentOutOfRangeException(nameof(bitsPerSample), $"不支持的位深: {bitsPerSample}")
        };

        int bytesPerSample = validBits / 8;
        int dataSize       = samples.Length * bytesPerSample;
        int sampleCount    = samples.Length;

        using var ms = new MemoryStream(44 + dataSize);
        using var bw = new BinaryWriter(ms);

        // ── RIFF 头 ─────────────────────────────────────────────
        bw.Write(Encoding.ASCII.GetBytes("RIFF"));
        bw.Write(36 + dataSize);   // file size - 8
        bw.Write(Encoding.ASCII.GetBytes("WAVE"));

        // ── fmt chunk ────────────────────────────────────────────
        bw.Write(Encoding.ASCII.GetBytes("fmt "));
        bw.Write(16);              // chunk size (PCM)
        bw.Write((short)1);        // audio format: PCM
        bw.Write((short)channels);
        bw.Write(sampleRate);
        bw.Write(sampleRate * channels * bytesPerSample); // byte rate
        bw.Write((short)(channels * bytesPerSample));     // block align
        bw.Write((short)validBits);

        // ── data chunk ───────────────────────────────────────────
        bw.Write(Encoding.ASCII.GetBytes("data"));
        bw.Write(dataSize);

        // ── 写入采样 ────────────────────────────────────────────
        switch (validBits)
        {
            case 8:
                // unsigned 8-bit
                for (int i = 0; i < sampleCount; i++)
                {
                    int v = (int)((samples[i] + 1f) * 127.5f); // [-1,1] → [0,255]
                    bw.Write((byte)Math.Clamp(v, 0, 255));
                }
                break;

            case 16:
                for (int i = 0; i < sampleCount; i++)
                {
                    short v = (short)Math.Clamp((int)(samples[i] * 32767), -32768, 32767);
                    bw.Write(v);
                }
                break;

            case 24:
                for (int i = 0; i < sampleCount; i++)
                {
                    int v = (int)Math.Clamp((int)(samples[i] * 8388607), -8388608, 8388607);
                    bw.Write((byte)(v & 0xFF));
                    bw.Write((byte)((v >> 8) & 0xFF));
                    bw.Write((byte)((v >> 16) & 0xFF));
                }
                break;

            case 32:
                for (int i = 0; i < sampleCount; i++)
                {
                    int v = (int)Math.Clamp((long)(samples[i] * 2147483647), -2147483648, 2147483647);
                    bw.Write(v);
                }
                break;
        }

        bw.Flush();
        return ms.ToArray();
    }

    // ══════════════════════════════════════════════════════════════
    // 便捷方法
    // ══════════════════════════════════════════════════════════════

    /// <summary>将 PcmData 编码为 WAV 字节数组。</summary>
    public static byte[] EncodeToWavBytes(PcmData pcm, int bitsPerSample = DEFAULT_BITS)
        => EncodeToWavBytes(pcm.Samples, pcm.SampleRate, pcm.Channels, bitsPerSample);

    /// <summary>将 PcmData 转换为 Godot AudioStreamWav，可直接赋给 AudioStreamPlayer。</summary>
    /// <param name="pcm">PCM 输入。</param>
    /// <param name="bitsPerSample">位深 (8 或 16)。</param>
    public static AudioStreamWav ToAudioStreamWav(PcmData pcm, int bitsPerSample = 16)
    {
        if (bitsPerSample != 8 && bitsPerSample != 16)
            throw new ArgumentOutOfRangeException(nameof(bitsPerSample), "仅支持 8/16 bit");

        int totalSamples = pcm.TotalSamples;
        int bytesPerSample = bitsPerSample / 8;
        byte[] data = new byte[totalSamples * bytesPerSample];

        if (bitsPerSample == 16)
        {
            for (int i = 0; i < totalSamples; i++)
            {
                short s = (short)Math.Clamp((int)(pcm.Samples[i] * 32767), -32768, 32767);
                data[i * 2]     = (byte)(s & 0xFF);
                data[i * 2 + 1] = (byte)((s >> 8) & 0xFF);
            }
        }
        else // 8-bit
        {
            for (int i = 0; i < totalSamples; i++)
            {
                byte b = (byte)Math.Clamp((int)((pcm.Samples[i] + 1f) * 127.5f), 0, 255);
                data[i] = b;
            }
        }

        var wav = new AudioStreamWav();
        wav.Data = data;
        wav.MixRate = pcm.SampleRate;
        wav.Format = bitsPerSample == 16
            ? AudioStreamWav.FormatEnum.Format16Bits
            : AudioStreamWav.FormatEnum.Format8Bits;
        // 注：AudioStreamWav.Format 的 setter 期望 int (0=8bit, 1=16bit)，enum 不直接兼容
        // 但 Godot 的 C# 绑定中 FormatEnum 的底层值恰好匹配 (0/1)
        wav.Stereo = pcm.Channels >= 2;
        return wav;
    }

    /// <summary>通过 ffmpeg 编码为其他格式 (MP3/OGG 等)。</summary>
    /// <param name="pcm">PCM 输入。</param>
    /// <param name="outputPath">输出路径 (扩展名决定格式)。</param>
    /// <returns>是否成功。</returns>
    public static bool EncodeViaFfmpeg(PcmData pcm, string outputPath)
    {
        // 先写临时 WAV
        string tmpWav = Path.GetTempFileName() + ".wav";
        try
        {
            SaveWav(tmpWav, pcm);

            string ext = Path.GetExtension(outputPath).ToLowerInvariant();
            string codec = ext switch
            {
                ".mp3" => "libmp3lame",
                ".ogg" => "libvorbis",
                ".flac"=> "flac",
                ".aac" => "aac",
                ".wma" => "wmav2",
                _      => "copy"  // 保持 WAV
            };

            var psi = new System.Diagnostics.ProcessStartInfo
            {
                FileName               = "ffmpeg",
                Arguments              = $"-i \"{tmpWav}\" -acodec {codec} -y \"{outputPath}\"",
                UseShellExecute        = false,
                CreateNoWindow         = true,
                RedirectStandardOutput = true,
                RedirectStandardError  = true,
            };

            using var proc = new System.Diagnostics.Process { StartInfo = psi };
            proc.Start();
            proc.WaitForExit(60000);

            if (proc.ExitCode != 0)
            {
                GD.PrintErr($"AudioEncoder: ffmpeg 编码失败 (exit={proc.ExitCode})");
                return false;
            }
            return File.Exists(outputPath);
        }
        finally
        {
            if (File.Exists(tmpWav))
                File.Delete(tmpWav);
        }
    }
}
