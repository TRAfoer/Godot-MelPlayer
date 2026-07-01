using System;

/// <summary>
/// 承载解码后的 PCM 数据，不可变只读结构体。
///
/// === 字段 ===
///   Samples     — 归一化 float 采样，范围 [-1, 1]
///   SampleRate  — 采样率 (Hz)
///   Channels    — 声道数 (1=mono, 2=stereo)
///
/// === 辅助方法 ===
///   MixToMono()         — 混为单声道 (平均法)
///   GetChannel(ch)      — 提取指定声道
///   Duration             — 时长 (秒)
/// </summary>
public readonly struct PcmData
{
	// ────────────────────────────────────────────────────────────────
	// 字段
	// ────────────────────────────────────────────────────────────────

	/// <summary>归一化 float 采样，范围 [-1, 1]，交错存储 (interleaved)。</summary>
	public float[] Samples { get; }

	/// <summary>采样率 (Hz)。</summary>
	public int SampleRate { get; }

	/// <summary>声道数 (1=mono, 2=stereo)。</summary>
	public int Channels { get; }

	/// <summary>位深 (仅 WAV 来源有准确值，ffmpeg 解码默认 16bit)。</summary>
	public int BitsPerSample { get; }

	// ────────────────────────────────────────────────────────────────
	// 构造 & 工厂
	// ────────────────────────────────────────────────────────────────

	public PcmData(float[] samples, int sampleRate, int channels, int bitsPerSample = 16)
	{
		Samples       = samples ?? throw new ArgumentNullException(nameof(samples));
		SampleRate    = sampleRate > 0 ? sampleRate : throw new ArgumentOutOfRangeException(nameof(sampleRate));
		Channels      = channels > 0 ? channels : throw new ArgumentOutOfRangeException(nameof(channels));
		BitsPerSample = bitsPerSample;
	}

	/// <summary>总采样数 (所有声道合计)。</summary>
	public int TotalSamples => Samples.Length;

	/// <summary>每声道采样数。</summary>
	public int SamplesPerChannel => Samples.Length / Channels;

	/// <summary>时长 (秒)。</summary>
	public double Duration => (double)SamplesPerChannel / SampleRate;

	/// <summary>是否为空。</summary>
	public bool IsEmpty => Samples == null || Samples.Length == 0;

	// ────────────────────────────────────────────────────────────────
	// 声道操作
	// ────────────────────────────────────────────────────────────────

	/// <summary>混为单声道 (声道平均法)。返回新 PcmData (Channels=1)。</summary>
	public PcmData MixToMono()
	{
		if (Channels == 1) return this;

		int n   = SamplesPerChannel;
		var mon = new float[n];
		for (int i = 0; i < n; i++)
		{
			double sum = 0;
			for (int c = 0; c < Channels; c++)
				sum += Samples[i * Channels + c];
			mon[i] = (float)(sum / Channels);
		}
		return new PcmData(mon, SampleRate, 1, BitsPerSample);
	}

	/// <summary>提取指定声道 (0-indexed)。返回新 PcmData (Channels=1)。</summary>
	public PcmData GetChannel(int channelIndex)
	{
		if (channelIndex < 0 || channelIndex >= Channels)
			throw new ArgumentOutOfRangeException(nameof(channelIndex));

		int n   = SamplesPerChannel;
		var ch  = new float[n];
		for (int i = 0; i < n; i++)
			ch[i] = Samples[i * Channels + channelIndex];
		return new PcmData(ch, SampleRate, 1, BitsPerSample);
	}

	// ────────────────────────────────────────────────────────────────
	// 裁剪 / 重采样桩 (占位)
	// ────────────────────────────────────────────────────────────────

	/// <summary>裁剪时间段 (秒)。返回新 PcmData。</summary>
	public PcmData Slice(double startSeconds, double durationSeconds)
	{
		int start = (int)(startSeconds * SampleRate) * Channels;
		int len   = (int)(durationSeconds * SampleRate) * Channels;
		start = Math.Clamp(start, 0, Samples.Length);
		len   = Math.Clamp(len, 0, Samples.Length - start);

		var slice = new float[len];
		Array.Copy(Samples, start, slice, 0, len);
		return new PcmData(slice, SampleRate, Channels, BitsPerSample);
	}

	// ────────────────────────────────────────────────────────────────
	// 格式信息
	// ────────────────────────────────────────────────────────────────

	public override string ToString()
		=> $"PcmData {{ Rate={SampleRate}Hz Channels={Channels} Bits={BitsPerSample} "
		 + $"Samples={TotalSamples:N0} Duration={Duration:F2}s }}";
}
