using Godot;
using NAudio.Wave;
using System;
using System.IO;
using System.Text;

/// <summary>
/// 音频解码器 — 将外部音频文件 (WAV/MP3/OGG/FLAC/AAC 等) 解码为 PCM 数据流。
///
/// === 架构 ===
/// ┌─ DecodeFile(path) ────────────────────────────────────────────┐
/// │   WAV → DecodeWavBytes() 直接解析 RIFF (零外部依赖)            │
/// │   MP3/OGG/FLAC/AAC/WMA → 用 NAudio 解码                     │
/// └──────────────────────────────────────────────────────────────┘
///
/// ┌─ DecodeFromGodotStream(stream) ──────────────────────────────┐
/// │   AudioStreamWav  → 直接提取 PCM                              │
/// │   其他 → 通过 ResourcePath 用 NAudio 解码                     │
/// └──────────────────────────────────────────────────────────────┘
///
/// === 依赖 ===
/// - NAudio 2.x (NuGet: dotnet add package NAudio)
/// - WAV 直接解析不需要任何依赖
///
/// === 线程安全 ===
/// 所有方法都是静态的且无共享可变状态，可安全在后台线程调用。
/// </summary>
public partial class AudioDecoder : GodotObject
{
	// ══════════════════════════════════════════════════════════════
	// 公共入口
	// ══════════════════════════════════════════════════════════════

	/// <summary>从文件路径解码音频文件为 PCM。</summary>
	public static PcmData DecodeFile(string path, int targetSampleRate = 0, int channels = 0)
	{
		if (string.IsNullOrEmpty(path))
			throw new ArgumentNullException(nameof(path));
		if (!File.Exists(path))
			throw new FileNotFoundException("音频文件不存在", path);

		var decoded = DecodeViaNAudio(path, targetSampleRate, channels);
		return decoded;
	}

	/// <summary>从 Godot AudioStream 解码,走 NAudio。</summary>
	public static PcmData DecodeFromGodotStream(AudioStream stream, string resourcePath = null)
	{
		if (stream == null) throw new ArgumentNullException(nameof(stream));

		string localPath = ResolveLocalPath(resourcePath, stream);
		if (localPath != null && File.Exists(localPath))
		{
			GD.Print($"[AudioDecoder]: 用 NAudio 解码 {localPath}");
			return DecodeViaNAudio(localPath, 0, 0);
		}

		string resPath = stream.ResourcePath;
		if (!string.IsNullOrEmpty(resPath))
		{
			string projPath = ProjectSettings.GlobalizePath(resPath);
			if (File.Exists(projPath))
				return DecodeViaNAudio(projPath, 0, 0);
		}

		throw new NotSupportedException(
			$"无法解码 {stream.GetType().Name}：无法获取文件路径，且该类型不直接暴露 PCM。\n"
		  + "请用 DecodeFile(path) 传入文件系统路径。");
	}

	// ══════════════════════════════════════════════════════════════
	// NAudio 解码
	// ══════════════════════════════════════════════════════════════

	/// <summary>通过 NAudio AudioFileReader 解码。</summary>
	private static PcmData DecodeViaNAudio(string filePath, int targetRate, int targetChannels)
	{
		using var reader = new AudioFileReader(filePath);

		int ch       = reader.WaveFormat.Channels;
		int rate     = reader.WaveFormat.SampleRate;
		long total   = reader.Length / (reader.WaveFormat.BitsPerSample / 8);

		var all = new float[total];
		int read = reader.Read(all, 0, all.Length);
		if (read < all.Length) Array.Resize(ref all, read);

		var pcm = new PcmData(all, rate, ch);
		return ResampleAndRemap(pcm, targetRate, targetChannels);
	}

	/// <summary>通过 NAudio 解码内存中的 WAV 字节。</summary>
	private static PcmData DecodeWavViaNAudio(byte[] wavBytes)
	{
		using var ms = new MemoryStream(wavBytes);
		using var reader = new WaveFileReader(ms);

		int ch   = reader.WaveFormat.Channels;
		int rate = reader.WaveFormat.SampleRate;

		var provider = reader.ToSampleProvider();
		long total  = reader.Length / (reader.WaveFormat.BitsPerSample / 8);
		var all     = new float[total];
		int read    = provider.Read(all, 0, all.Length);
		if (read < all.Length) Array.Resize(ref all, read);

		return new PcmData(all, rate, ch);
	}

	// ══════════════════════════════════════════════════════════════
	// WAV 直解（纯 C#，零依赖）
	// ══════════════════════════════════════════════════════════════

	private static float[] PcmBytesToFloats(byte[] data, int bitsPerSample, int channels, int audioFormat = 1)
	{
		int bytesPerSample = bitsPerSample / 8;
		int totalSamples   = data.Length / bytesPerSample;
		totalSamples = (totalSamples / channels) * channels;
		var result = new float[totalSamples];

		if (audioFormat == 3 && bitsPerSample == 32)
		{
			for (int i = 0; i < totalSamples; i++)
				result[i] = BitConverter.ToSingle(data, i * 4);
			return result;
		}

		switch (bitsPerSample)
		{
			case 8:
				for (int i = 0; i < totalSamples; i++)
					result[i] = (data[i] - 128) / 128f;
				break;
			case 16:
				for (int i = 0; i < totalSamples; i++)
				{
					short s = (short)(data[i * 2] | (data[i * 2 + 1] << 8));
					result[i] = s / 32768f;
				}
				break;
			case 24:
				for (int i = 0; i < totalSamples; i++)
				{
					int off = i * 3;
					int s = data[off] | (data[off + 1] << 8) | (data[off + 2] << 16);
					if ((s & 0x800000) != 0) s |= unchecked((int)0xFF000000);
					result[i] = s / 8388608f;
				}
				break;
			case 32:
				for (int i = 0; i < totalSamples; i++)
				{
					int s = BitConverter.ToInt32(data, i * 4);
					result[i] = s / 2147483648f;
				}
				break;
			default:
				throw new NotSupportedException($"不支持的位深: {bitsPerSample} bit");
		}
		return result;
	}

	// ══════════════════════════════════════════════════════════════
	// 重采样 & 声道映射
	// ══════════════════════════════════════════════════════════════

	private static PcmData ResampleAndRemap(PcmData src, int targetRate, int targetChannels)
	{
		if (targetRate <= 0 && targetChannels <= 0) return src;
		PcmData result = src;

		if (targetChannels > 0 && result.Channels != targetChannels)
		{
			if (targetChannels == 1) result = result.MixToMono();
			else if (targetChannels == 2 && result.Channels == 1)
				result = MonoToStereo(result);
			else throw new NotSupportedException($"声道转换不支持: {result.Channels}ch → {targetChannels}ch");
		}
		if (targetRate > 0 && result.SampleRate != targetRate)
			result = ResampleLinear(result, targetRate);

		return result;
	}

	private static PcmData MonoToStereo(PcmData src)
	{
		int n = src.SamplesPerChannel;
		var s = new float[n * 2];
		for (int i = 0; i < n; i++)
			s[i * 2] = s[i * 2 + 1] = src.Samples[i];
		return new PcmData(s, src.SampleRate, 2, src.BitsPerSample);
	}

	private static PcmData ResampleLinear(PcmData src, int targetRate)
	{
		if (src.SampleRate == targetRate) return src;
		int srcLen = src.SamplesPerChannel;
		int dstLen = Math.Max(1, (int)((long)srcLen * targetRate / src.SampleRate));
		int ch = src.Channels;
		var output = new float[dstLen * ch];
		double ratio = (double)src.SampleRate / targetRate;

		for (int i = 0; i < dstLen; i++)
		{
			double srcPos = i * ratio;
			int idx0 = (int)srcPos;
			int idx1 = Math.Min(idx0 + 1, srcLen - 1);
			float frac = (float)(srcPos - idx0);
			for (int c = 0; c < ch; c++)
			{
				float v0 = src.Samples[idx0 * ch + c];
				float v1 = src.Samples[idx1 * ch + c];
				output[i * ch + c] = v0 + (v1 - v0) * frac;
			}
		}
		return new PcmData(output, targetRate, ch, src.BitsPerSample);
	}

	// ══════════════════════════════════════════════════════════════
	// 辅助
	// ══════════════════════════════════════════════════════════════

	private static string ResolveLocalPath(string hint, AudioStream stream)
	{
		if (!string.IsNullOrEmpty(hint) && File.Exists(hint)) return hint;
		string resPath = stream.ResourcePath;
		if (!string.IsNullOrEmpty(resPath))
		{
			string local = ProjectSettings.GlobalizePath(resPath);
			if (File.Exists(local)) return local;
		}
		return null;
	}

	// ══════════════════════════════════════════════════════════════
	// GDScript 友好接口
	// ══════════════════════════════════════════════════════════════

	/// <summary>解码音频文件并加载到 AudioMelSpectrogram。
	/// stereo=true → 立体声 (LoadRawPcmStereo)，false → 单声道 (LoadRawPcm)。
	/// 由 GDScript 调用，返回 { ok, message }。</summary>
	public static Godot.Collections.Dictionary LoadToMelSpectrogram(
		string path, AudioMelSpectrogram spec, int targetSampleRate = 44100,
		bool stereo = false)
	{
		try
		{
			int ch = stereo ? 2 : 1;
			var pcm = DecodeFile(path, targetSampleRate, ch);
			if (stereo)
				spec.LoadRawPcmStereo(pcm.Samples, pcm.SampleRate);
			else
				spec.LoadRawPcm(pcm.Samples, pcm.SampleRate);
			return new Godot.Collections.Dictionary
			{
				{"ok", true},
				{"message", $"解码成功: {pcm.SampleRate}Hz {pcm.Channels}ch {pcm.Duration:F1}s"},
			};
		}
		catch (Exception ex)
		{
			return new Godot.Collections.Dictionary
			{
				{"ok", false},
				{"message", ex.Message},
			};
		}
	}

	/// <summary>一次解码同时喂给频谱 + 返回 AudioStreamWav 给播放器。
	/// 避免 Godot 不支持的格式需要解码两次。</summary>
	public static AudioStreamWav DecodeAndFeedSpec(
		string path, AudioMelSpectrogram spec, int targetSampleRate, bool stereo)
	{
		try
		{
			var pcm = DecodeFile(path, targetSampleRate, 2);
			if (stereo)
				spec.LoadRawPcmStereo(pcm.Samples, pcm.SampleRate);
			else
				spec.LoadRawPcm(pcm.MixToMono().Samples, pcm.SampleRate);
			return AudioEncoder.ToAudioStreamWav(pcm);
		}
		catch (Exception ex)
		{
			GD.PrintErr($"DecodeAndFeedSpec 失败: {ex.Message}");
			return null;
		}
	}

	/// <summary>异步解码并喂给频谱，不阻塞主线程。</summary>
	public static void LoadToMelSpectrogramAsync(
		string path, AudioMelSpectrogram spec, int targetSampleRate, bool stereo)
	{
		var mainCtx = System.Threading.SynchronizationContext.Current;
		System.Threading.ThreadPool.QueueUserWorkItem(_ =>
		{
			try
			{
				var pcm = DecodeFile(path, targetSampleRate, stereo ? 2 : 1);
				if (stereo)
					spec.LoadRawPcmStereo(pcm.Samples, pcm.SampleRate);
				else
					spec.LoadRawPcm(pcm.Samples, pcm.SampleRate);
			}
			catch (Exception ex)
			{
				GD.PrintErr($"LoadToMelSpectrogramAsync 失败: {ex.Message}");
			}
		});
	}

	/// <summary>异步解码 + 喂频谱 + 创建播放流，不阻塞主线程。</summary>
	public static void DecodeAndFeedSpecAsync(
		string path, AudioMelSpectrogram spec, int targetSampleRate,
		bool stereo, AudioStreamPlayer player)
	{
		var mainCtx = System.Threading.SynchronizationContext.Current;
		System.Threading.ThreadPool.QueueUserWorkItem(_ =>
		{
			try
			{
				var pcm = DecodeFile(path, targetSampleRate, 2);
				if (stereo)
					spec.LoadRawPcmStereo(pcm.Samples, pcm.SampleRate);
				else
					spec.LoadRawPcm(pcm.MixToMono().Samples, pcm.SampleRate);
				mainCtx?.Post(__ =>
				{
					try
					{
						var wav = AudioEncoder.ToAudioStreamWav(pcm);
						player.Stream = wav;
						player.Play();
					}
					catch (Exception ex)
					{
						GD.PrintErr($"DecodeAndFeedSpecAsync 创建播放流失败: {ex.Message}");
					}
				}, null);
			}
			catch (Exception ex)
			{
				GD.PrintErr($"DecodeAndFeedSpecAsync 失败: {ex.Message}");
			}
		});
	}

	/// <summary>解码音频文件，返回 Dictionary 供 GDScript 使用。</summary>
	public static Godot.Collections.Dictionary DecodeFileToDict(
		string path, int targetSampleRate = 0, int channels = 0)
	{
		try
		{
			var pcm = DecodeFile(path, targetSampleRate, channels);
			return new Godot.Collections.Dictionary
			{
				{"ok", true},
				{"samples",         pcm.Samples},
				{"sample_rate",     pcm.SampleRate},
				{"channels",        pcm.Channels},
				{"bits_per_sample", pcm.BitsPerSample},
				{"duration",        pcm.Duration},
				{"total_samples",   pcm.TotalSamples},
			};
		}
		catch (Exception ex)
		{
			return new Godot.Collections.Dictionary
			{
				{"ok", false},
				{"message", ex.Message},
			};
		}
	}

	/// <summary>解码音频文件并直接生成 Godot AudioStreamWav。</summary>
	public static AudioStreamWav DecodeToAudioStreamWav(string path, int targetSampleRate = 44100, int channels = 2)
	{
		var pcm = DecodeFile(path, targetSampleRate, channels);
		return AudioEncoder.ToAudioStreamWav(pcm);
	}
}
