using Godot;
using System;
using System.Threading;

/// <summary>
/// 非 Godot 节点 / 资源的 Beat（节拍）驱动型 Mel 频谱图渲染器。
///
/// === 与 AudioMelSpectrogram 的核心区别 ===
/// - 接收 curBeatF / maxBeatF（节拍浮点数）而非秒数
/// - 内置 BeatToTime() 工具函数，当前为占位实现（beat * 2）
/// - 逐列独立做 beat→time→frameIdx 换算，支持可变 BPM 表
///
/// === 架构 ===
/// 调用方节点._Process()
///   └─ AudioMelSpectrogramBeat.UpdateView(audioPlayer, curBeatF, maxBeatF, bpmList)
///        ├─ 首次检测 AudioStreamWav → 后台线程预计算全曲 FFT
///        ├─ 每帧逐列 beat→time→frameIdx → 颜色 LUT → byte[] → SetData → Update
///        └─ 视角（节拍范围）未变时整帧跳过
///
/// 后台线程：PCM 读取(主线程) → FFT + Mel 滤波 + Log(子线程) → 缓存 Log 能量
/// 主线程：  按 beat 范围 → 逐列调用 BeatToTime → 帧索引 → 查 LUT → 写像素 → GPU upload
/// </summary>
public partial class AudioMelSpectrogramBeat : RefCounted
{
	// ══════════════════════════════════════════════════════════════════════
	// [公共参数] 由调用方节点每帧通过字段赋值同步
	// ══════════════════════════════════════════════════════════════════════

	/// <summary>渲染目标的 TextureRect。尺寸从此读取 × resolutionScale 决定纹理大小。</summary>
	public TextureRect targetImage;

	/// <summary>纹理分辨率缩放。0.5 = 半分辨率（宽/高各砍一半，像素数 1/4）。</summary>
	public float resolutionScale = 0.5f;

	/// <summary>颜色渐变：归一化能量 [0,1] → Color。由调用方节点赋值。</summary>
	public Gradient colorGradient;


	// ══════════════════════════════════════════════════════════════════════
	// [FFT / Mel 常量] 改动需要重新预计算才会生效
	// ══════════════════════════════════════════════════════════════════════

	/// <summary>FFT 窗口大小（采样点数）。越大频率分辨率越高、时间分辨率越低。
	/// 4096 @ 44100Hz ≈ 93ms 窗口，频率分辨率 ≈ 10.8Hz。
	/// 配合 512 帧移（87.5% overlap）维持时间分辨率 ~11.6ms。</summary>
	private const int FFT_SIZE = 4096;

	/// <summary>帧移（相邻 FFT 窗口的起始采样偏移量）。
	/// 512 = 50% overlap（HOP_SIZE = FFT_SIZE / 2）。
	/// 每帧对应时间 = HOP_SIZE / sampleRate ≈ 11.6ms。</summary>
	private const int HOP_SIZE = 512;

	/// <summary>Mel 滤波器组数量。也是缓存和纹理的高度方向分辨率。
	/// 128 个三角滤波器在 Mel 尺度上等距分布，覆盖 0 ~ sampleRate/2 Hz。</summary>
	private const int MEL_BINS = 128;

	/// <summary>Mel 尺度常数：m = 2595 × log10(1 + f/700)。</summary>
	private const float MEL_A = 2595f;

	/// <summary>Mel 尺度常数：f = 700 × (10^(m/2595) - 1)。</summary>
	private const float MEL_B = 700f;


	// ══════════════════════════════════════════════════════════════════════
	// [节拍 → 时间] 占位函数
	// ══════════════════════════════════════════════════════════════════════

	/// <summary>节拍浮点数 → 时间（秒）。当前为占位实现，后续应接入 BPM 换算表。</summary>
	/// <param name="beat">节拍浮点数，如 4.0 = 第 4 拍。</param>
	/// <returns>秒数。占位：beat × 2（等效 120 BPM 匀速）。</returns>
	private static float BeatToTime(float beat)
	{
		return beat;
	}


	// ══════════════════════════════════════════════════════════════════════
	// [预计算缓存] 后台线程写入 → 主线程只读
	// ══════════════════════════════════════════════════════════════════════

	/// <summary>当前已缓存的 AudioStreamWav 引用。检测变化时触发重新预计算。</summary>
	private AudioStreamWav _cachedWav;

	/// <summary>预计算完成后的 Log-Mel 能量缓存。
	/// 布局：flat[ frameIndex × MEL_BINS + melBin ]，每个元素 = Log(能量)。
	/// volatile 保证主线程读到最新值。</summary>
	private volatile float[] _logMelData;

	/// <summary>后台线程正在写入的临时缓存。完成后通过 _cacheReady 交给主线程。</summary>
	private float[] _pendingLogData;

	/// <summary>后台线程完成标记。主线程检测到后切换 _pendingLogData → _logMelData。</summary>
	private volatile bool _cacheReady;

	/// <summary>总帧数。由 (totalSamples - FFT_SIZE) / HOP_SIZE + 1 计算。</summary>
	private int _totalFrames;

	/// <summary>每帧对应的时间（秒）= HOP_SIZE / sampleRate。</summary>
	private float _timePerFrame;

	/// <summary>稀疏 Mel 三角滤波器组（仅存储非零段）。</summary>
	private float[][] _melFilterWeights;
	private int[] _melFilterStart;
	private int[] _melFilterEnd;

	/// <summary>AudioStreamWav 的采样率（Hz）。</summary>
	private int _sampleRate;

	/// <summary>后台线程正在运行的标记。防止重复启动线程。</summary>
	private volatile bool _computing;

	/// <summary>后台线程已计算完成的帧数。每算完一批通过 volatile 更新。</summary>
	private volatile int _framesComputed;

	/// <summary>后台线程引用。用于 Stop() 时等待线程结束。</summary>
	private Thread _thread;


	// ══════════════════════════════════════════════════════════════════════
	// [运行时 Image / Texture]
	// ══════════════════════════════════════════════════════════════════════

	/// <summary>实际写入的 ImageTexture。尺寸 = TextureRect 像素尺寸 × resolutionScale。
	/// 每帧通过 Image.SetData + ImageTexture.Update 提交到 GPU。</summary>
	private ImageTexture _texture;

	/// <summary>CPU 侧 Image 对象。每帧将 _pixels 写入此 Image，然后 Upload。</summary>
	private Image _image;

	/// <summary>像素缓冲区（CPU 侧，byte[] RGBA8）。
	/// 长度为 texW × texH × 4。每帧直接写入后通过 SetData 推入 Image。</summary>
	private byte[] _pixels;

	/// <summary>当前纹理尺寸。用于检测尺寸变化并重建 Image/Texture。</summary>
	private int _texW, _texH;

	/// <summary>上一帧渲染的起始/结束 FFT 帧号。用于「帧号未变跳过」避免边界振荡。</summary>
	private int _prevStartF, _prevEndF;


	// ══════════════════════════════════════════════════════════════════════
	// [颜色 LUT] 256 级预计算查找表（byte[1024]：每项 4 字节 RGBA）
	// ══════════════════════════════════════════════════════════════════════

	/// <summary>256 级颜色查找表。norm[0,1] → LUT[(int)(norm × 255)]。
	/// 避免每像素调用 Gradient.Sample（函数调用 + 键插值开销）。
	/// 每项 4 字节 RGBA，总长度 1024。</summary>
	private byte[] _colorLut;

	/// <summary>上次构建 LUT 时使用的 Gradient 引用。</summary>
	private Gradient _appliedGradient;


	// ══════════════════════════════════════════════════════════════════════
	// [Bin 查找表] y → melBin 的预计算映射
	// ══════════════════════════════════════════════════════════════════════

	/// <summary>纹理行 y → Mel 滤波器索引。_binForY[y] = round(y / h × MEL_BINS)。</summary>
	private int[] _binForY;


	// ══════════════════════════════════════════════════════════════════════
	// 每帧入口
	// ══════════════════════════════════════════════════════════════════════

	/// <summary>每帧由调用方节点的 _Process() 调用。
	/// 参数为节拍（beat）浮点数，内部通过 BeatToTime() 转为秒后映射到 FFT 帧。</summary>
	/// <param name="audioPlayer">用于检测 AudioStreamWav 变化。</param>
	/// <param name="curBeatF">可见范围起点（节拍）。</param>
	/// <param name="maxBeatF">可见范围终点（节拍）。</param>
	/// <param name="forceUpdate">强制重绘（忽略帧号缓存）。</param>
	public void UpdateView(AudioStreamPlayer audioPlayer, float curBeatF, float maxBeatF, bool forceUpdate = false)
	{
		if (targetImage == null) return;

		// ── 检测新 AudioStream → 启动后台预计算 ────────────────────────
		if (audioPlayer != null && audioPlayer.Stream is AudioStreamWav wav && wav != _cachedWav)
		{
			if (!_computing)
				StartPreCompute(wav);
		}

		// ── 后台线程完成 → 切缓存 ──────────────────────────────────────
		if (_cacheReady)
		{
			if (_pendingLogData == null)
			{
				_cacheReady = false;
				_computing = false;
			}
			else
			{
				_logMelData = _pendingLogData;
				_totalFrames = _logMelData.Length / MEL_BINS;
				_cachedWav = audioPlayer?.Stream as AudioStreamWav;
				_cacheReady = false;
				_computing = false;
			}
			_pendingLogData = null;
		}

		// ── 完全态：全缓存就绪，正常渲染 ────────────────────────────────
		if (!_computing && _logMelData != null && _totalFrames > 0)
		{
			RenderCurrentView(curBeatF, maxBeatF, forceUpdate);
			return;
		}
		// ── 部分态：后台计算中，但已有部分帧可供显示 ────────────────────
		if (_computing && _pendingLogData != null && _framesComputed > 0)
		{
			RenderCurrentView(curBeatF, maxBeatF, true,
							  _pendingLogData, _framesComputed);
		}
	}


	// ══════════════════════════════════════════════════════════════════════
	// 渲染当前视图（被完全态和部分态共用）
	// ══════════════════════════════════════════════════════════════════════

	/// <summary>根据 curBeatF/maxBeatF 逐列 beat→time→frameIdx 换算，渲染到纹理并提交 GPU。</summary>
	/// <param name="data">数据源：完全态用 _logMelData，部分态用 _pendingLogData。</param>
	/// <param name="framesAvail">可用帧数。</param>
	private void RenderCurrentView(float curBeatF, float maxBeatF, bool forceUpdate,
								   float[] data = null, int framesAvail = 0)
	{
		if (data == null) { data = _logMelData; framesAvail = _totalFrames; }
		if (data == null || framesAvail == 0) return;

		// ── 纹理尺寸 ──
		Vector2 size = targetImage.Size;
		int w = Mathf.Max(1, Mathf.RoundToInt(size.X * resolutionScale));
		int h = Mathf.Max(1, Mathf.RoundToInt(size.Y * resolutionScale));
		bool sizeChanged = _texture == null || _texW != w || _texH != h;

		if (sizeChanged)
			InitTexture(w, h);

		float dur = maxBeatF - curBeatF;
		if (dur <= 0f) return;

		// ── 帧号未变 → 跳过 ──
		// 将节拍范围两端转为时间 → FFT 帧号，与节拍范围是否跨越 timePerFrame 边界有关。
		float curTime  = BeatToTime(curBeatF);
		float maxTime  = BeatToTime(maxBeatF);
		float frameOffset = FFT_SIZE / (2f * HOP_SIZE);
		int startF = Mathf.Max(0, Mathf.RoundToInt(curTime / _timePerFrame - frameOffset));
		int endF   = Mathf.Max(1, Mathf.RoundToInt(maxTime / _timePerFrame - frameOffset));
		if (!forceUpdate && startF == _prevStartF && endF == _prevEndF && _pixels != null)
			return;

		// ── 全量重绘 ──
		BuildColorLut();
		RenderColumns(curBeatF, maxBeatF, w, h, 0, w - 1, data, framesAvail);

		_image.SetData(w, h, false, Image.Format.Rgba8, _pixels);
		_texture.Update(_image);

		_prevStartF = startF;
		_prevEndF = endF;
		_texW = w;
		_texH = h;
	}


	// ══════════════════════════════════════════════════════════════════════
	// 列渲染（热路径）— 逐列独立 beat→time→frameIdx 换算
	// ══════════════════════════════════════════════════════════════════════

	/// <summary>逐列渲染。每列先通过节拍比例算出该列对应的 beat，
	/// 再调用 BeatToTime() 转为秒，最后映射到 FFT 帧索引。
	/// 用 fillMode 填充未计算帧以避免部分渲染全黑。</summary>
	private void RenderColumns(float curBeatF, float maxBeatF,
							   int w, int h, int colFrom, int colTo,
							   float[] data, int totalFrames)
	{
		float beatRange = maxBeatF - curBeatF;
		float frameOffset = FFT_SIZE / (2f * HOP_SIZE);

		// 每列 rows 个 float 临时缓冲（GC 分配一行，非每像素，可接受）。
		float[] colVals = new float[h];

		for (int x = colFrom; x <= colTo; x++)
		{
			// ── 当前列对应的节拍 → 时间 → FFT 帧索引 ──
			float t = (float)x / w;
			float beat = curBeatF + t * beatRange;
			float time = BeatToTime(beat);
			int frameIdx = (int)Mathf.Round(time / _timePerFrame - frameOffset);

			// fillMode：未计算帧用最近已计算帧填充，避免全黑
			if (frameIdx >= totalFrames || frameIdx < 0)
			{
				frameIdx = Mathf.Clamp(frameIdx, 0, totalFrames - 1);
			}
			int rowBase = frameIdx * MEL_BINS;

			// ── 一趟读入 colVals + 扫描 min/max ──
			float mn = float.MaxValue, mx = float.MinValue;
			for (int iy = 0; iy < h; iy++)
			{
				float v = data[rowBase + _binForY[iy]];
				colVals[iy] = v;
				if (v < mn) mn = v;
				if (v > mx) mx = v;
			}
			float invRange = 1f / Mathf.Max(mx - mn, 0.001f);

			// ── 从 colVals 查 LUT，逐字节写入 RGBA ──
			for (int iy = 0; iy < h; iy++)
			{
				float norm = (colVals[iy] - mn) * invRange;
				int idx = (int)(norm * 255);
				if (idx > 255) idx = 255;

				int pixelOff = (iy * w + x) * 4;
				int lutOff   = idx * 4;
				_pixels[pixelOff]     = _colorLut[lutOff];
				_pixels[pixelOff + 1] = _colorLut[lutOff + 1];
				_pixels[pixelOff + 2] = _colorLut[lutOff + 2];
				_pixels[pixelOff + 3] = _colorLut[lutOff + 3];
			}
		}
	}


	// ══════════════════════════════════════════════════════════════════════
	// 后台预计算
	// ══════════════════════════════════════════════════════════════════════

	/// <summary>启动后台线程预计算。
	/// 主线程：仅做快速准备（Mel 滤波器 ≈1ms + 数组分配）。
	/// 子线程：PCM 读取 → FFT → Mel 滤波全部在后台，首帧即出。</summary>
	private void StartPreCompute(AudioStreamWav wav)
	{
		// 立即初始化纹理（清空为黑色），让用户第一时间看到画面更新
		if (targetImage != null)
		{
			Vector2 size = targetImage.Size;
			int w = Mathf.Max(1, Mathf.RoundToInt(size.X * resolutionScale));
			int h = Mathf.Max(1, Mathf.RoundToInt(size.Y * resolutionScale));
			InitTexture(w, h);
		}

		_computing = true;
		_cacheReady = false;
		_framesComputed = 0;
		_sampleRate = (int)wav.MixRate;
		_timePerFrame = (float)HOP_SIZE / _sampleRate;

		// 构建 Mel 滤波器组（主线程）
		BuildMelFilterBank();

		// ── 获取 PCM 基本信息（主线程，仅算数不遍历） ──
		byte[] rawData = wav.Data;
		int channels = wav.Stereo ? 2 : 1;
		int formatVal = (int)wav.Format;
		GD.Print($"MelSpecBeat: format={formatVal} stereo={channels} rate={_sampleRate} dataLen={rawData.Length}");

		if (formatVal >= 2)
		{
			GD.PrintErr($"MelSpecBeat: 不支持的编码格式({formatVal})，仅支持 PCM 8/16bit");
			_computing = false;
			return;
		}
		int bytesPerSample = formatVal == 1 ? 2 : 1;
		int totalSamples = rawData.Length / bytesPerSample / channels;
		int totalFrames = Mathf.Max(1, (totalSamples - FFT_SIZE) / HOP_SIZE + 1);

		// ── 分配 pending 缓存 + 启动后台线程（PCM 读取融合在帧循环中） ──
		float[] pending = new float[totalFrames * MEL_BINS];
		_pendingLogData = pending;

		if (_thread != null && _thread.IsAlive)
			_thread.Join();
		_thread = new Thread(() =>
		{
			ComputeFftMelLogFromPcm(rawData, bytesPerSample, channels,
				totalSamples, totalFrames, pending);
		});
		_thread.IsBackground = true;
		_thread.Start();
	}

	/// <summary>从原始 PCM byte[] 直接计算 FFT + Mel + Log，无需前置 MixToMono。
	/// 每帧仅读取当前窗所需的 FFT_SIZE 个采样，第一帧即完成，大幅缩短首次显示延迟。</summary>
	private void ComputeFftMelLogFromPcm(byte[] rawData, int bytesPerSample, int channels,
										  int totalSamples, int totalFrames, float[] pending)
	{
		// Blackman-Harris 窗（所有帧共用）
		float[] window = new float[FFT_SIZE];
		float bhA0 = 0.35875f, bhA1 = 0.48829f, bhA2 = 0.14128f, bhA3 = 0.01168f;
		float invN1 = 1f / (FFT_SIZE - 1);
		for (int i = 0; i < FFT_SIZE; i++)
		{
			float p = Mathf.Pi * 2f * i * invN1;
			window[i] = bhA0 - bhA1 * Mathf.Cos(p) + bhA2 * Mathf.Cos(2f * p) - bhA3 * Mathf.Cos(3f * p);
		}

		float[] real  = new float[FFT_SIZE];
		float[] imag  = new float[FFT_SIZE];
		float[] power = new float[FFT_SIZE / 2];

		for (int f = 0; f < totalFrames; f++)
		{
			int offset = f * HOP_SIZE;

			// 读 PCM + 混单声道 + 加窗（一趟 pass）
			for (int i = 0; i < FFT_SIZE; i++)
			{
				int idx = offset + i;
				if (idx < totalSamples)
				{
					double sum = 0;
					for (int c = 0; c < channels; c++)
					{
						int off = (idx * channels + c) * bytesPerSample;
						if (bytesPerSample == 2)
						{
							short s = (short)(rawData[off] | (rawData[off + 1] << 8));
							sum += s / 32768.0;
						}
						else
						{
							sum += (rawData[off] - 128) / 128.0;
						}
					}
					real[i] = (float)(sum / channels) * window[i];
				}
				else
				{
					real[i] = 0f;
				}
				imag[i] = 0f;
			}

			// 原位 Cooley-Tukey FFT
			FFT(real, imag);

			// 功率谱 |X|²
			for (int i = 0; i < FFT_SIZE / 2; i++)
				power[i] = real[i] * real[i] + imag[i] * imag[i];

			// 稀疏 Mel 三角滤波 + Log
			int row = f * MEL_BINS;
			for (int m = 0; m < MEL_BINS; m++)
			{
				float e = 0f;
				int s = _melFilterStart[m], eIdx = _melFilterEnd[m];
				var w = _melFilterWeights[m];
				for (int i = s; i < eIdx; i++)
					e += power[i] * w[i - s];
				pending[row + m] = Mathf.Log(Mathf.Max(e, 1e-10f));
			}

			// 每 4 帧更新进度，主线程立即捕获并开始部分渲染
			if ((f & 3) == 0 || f == totalFrames - 1)
				_framesComputed = f + 1;
		}

		_cacheReady = true;
	}


	// ══════════════════════════════════════════════════════════════════════
	// Mel 三角滤波器组
	// ══════════════════════════════════════════════════════════════════════

	/// <summary>构建稀疏 Mel 三角滤波器组。
	/// 每个三角滤波器在频域上只覆盖窄范围，跳过零权重部分以加速计算。</summary>
	private void BuildMelFilterBank()
	{
		int half = FFT_SIZE / 2;
		float freqMax = _sampleRate / 2f;
		float melMin = FreqToMel(0f);
		float melMax = FreqToMel(freqMax);

		float[] binFreq = new float[half];
		for (int i = 0; i < half; i++)
			binFreq[i] = (float)i / FFT_SIZE * _sampleRate;

		_melFilterWeights = new float[MEL_BINS][];
		_melFilterStart   = new int[MEL_BINS];
		_melFilterEnd     = new int[MEL_BINS];
		for (int m = 0; m < MEL_BINS; m++)
		{
			float mL = melMin + (melMax - melMin) * (m) / (MEL_BINS + 1);
			float mC = melMin + (melMax - melMin) * (m + 1) / (MEL_BINS + 1);
			float mR = melMin + (melMax - melMin) * (m + 2) / (MEL_BINS + 1);
			float fL = MelToFreq(mL), fC = MelToFreq(mC), fR = MelToFreq(mR);

			// 扫描找到非零范围
			int s = -1, e = -1;
			for (int i = 0; i < half; i++)
			{
				float f = binFreq[i];
				if (f >= fL && f < fR)
				{
					if (s < 0) s = i;
					e = i + 1;
				}
			}
			if (s < 0) { s = 0; e = 1; }

			_melFilterStart[m] = s;
			_melFilterEnd[m]   = e;

			var w = new float[e - s];
			for (int i = s; i < e; i++)
			{
				float f = binFreq[i];
				if (f >= fL && f < fC)
					w[i - s] = (f - fL) / (fC - fL);
				else if (f >= fC && f < fR)
					w[i - s] = (fR - f) / (fR - fC);
			}
			_melFilterWeights[m] = w;
		}
	}


	// ══════════════════════════════════════════════════════════════════════
	// FFT — 实序列原位 Cooley-Tukey radix-2 DIT（时域抽取）
	// ══════════════════════════════════════════════════════════════════════

	/// <summary>一维实序列 FFT（正变换）。
	/// radix-2 = 输入长度必须为 2 的幂（FFT_SIZE = 4096 ✅）。</summary>
	private static void FFT(float[] real, float[] imag)
	{
		int n = real.Length;

		// 1. Bit-reversal 重排
		for (int i = 1, j = 0; i < n; i++)
		{
			int bit = n >> 1;
			for (; j >= bit; bit >>= 1) j -= bit;
			j += bit;
			if (i < j)
			{
				(real[i], real[j]) = (real[j], real[i]);
				(imag[i], imag[j]) = (imag[j], imag[i]);
			}
		}

		// 2. 蝶形运算
		for (int len = 2; len <= n; len <<= 1)
		{
			float ang = 2f * Mathf.Pi / len;
			float wlenR = Mathf.Cos(ang);
			float wlenI = -Mathf.Sin(ang);

			for (int i = 0; i < n; i += len)
			{
				float wr = 1f, wi = 0f;
				int half = len >> 1;
				for (int j = 0; j < half; j++)
				{
					int i1 = i + j;
					int i2 = i + j + half;
					float tr = wr * real[i2] - wi * imag[i2];
					float ti = wr * imag[i2] + wi * real[i2];
					real[i2] = real[i1] - tr;
					imag[i2] = imag[i1] - ti;
					real[i1] += tr;
					imag[i1] += ti;
					(wr, wi) = (wr * wlenR - wi * wlenI, wr * wlenI + wi * wlenR);
				}
			}
		}
	}


	// ══════════════════════════════════════════════════════════════════════
	// Mel 尺度转换
	// ══════════════════════════════════════════════════════════════════════

	/// <summary>频率（Hz）→ Mel 值。m = 2595 × log10(1 + f/700)。</summary>
	private static float FreqToMel(float f) => MEL_A * (float)Math.Log10(1f + f / MEL_B);

	/// <summary>Mel 值 → 频率（Hz）。f = 700 × (10^(m/2595) - 1)。</summary>
	private static float MelToFreq(float m) => MEL_B * (Mathf.Pow(10f, m / MEL_A) - 1f);


	// ══════════════════════════════════════════════════════════════════════
	// 纹理 / LUT / Bin 表
	// ══════════════════════════════════════════════════════════════════════

	/// <summary>创建或重建 Image/ImageTexture 与辅助表。</summary>
	private void InitTexture(int w, int h)
	{
		_pixels = new byte[w * h * 4];

		_image = Image.CreateEmpty(w, h, false, Image.Format.Rgba8);
		_image.SetData(w, h, false, Image.Format.Rgba8, _pixels);

		_texture = ImageTexture.CreateFromImage(_image);

		if (targetImage != null)
			targetImage.Texture = _texture;

		// 预计算 y → bin 索引
		_binForY = new int[h];
		for (int y = 0; y < h; y++)
			_binForY[y] = Mathf.Clamp(
				Mathf.RoundToInt((float)y / h * MEL_BINS),
				0, MEL_BINS - 1);

		_appliedGradient = null; // 强制下次重建 LUT
	}

	/// <summary>构建 256 级颜色 LUT（byte[1024]，每 4 字节 = RGBA）。</summary>
	private void BuildColorLut()
	{
		var grad = colorGradient ?? GetDefaultGradient();

		// 只在「使用默认 Gradient 且已缓存」时跳过重建
		if (colorGradient == null && _appliedGradient == grad && _colorLut != null)
			return;

		_colorLut = new byte[256 * 4];
		for (int i = 0; i < 256; i++)
		{
			Color c = grad.Sample(i / 255f);
			_colorLut[i * 4]     = (byte)(c.R * 255f);
			_colorLut[i * 4 + 1] = (byte)(c.G * 255f);
			_colorLut[i * 4 + 2] = (byte)(c.B * 255f);
			_colorLut[i * 4 + 3] = (byte)(c.A * 255f);
		}
		_appliedGradient = grad;
	}


	// ══════════════════════════════════════════════════════════════════════
	// 默认渐变
	// ══════════════════════════════════════════════════════════════════════

	private static Gradient _defaultGradient;

	/// <summary>默认半透渐变：低能透明 → 深紫 → 青蓝 → 青绿 → 金黄 → 白。</summary>
	private static Gradient GetDefaultGradient()
	{
		if (_defaultGradient != null)
			return _defaultGradient;

		_defaultGradient = new Gradient();
		_defaultGradient.Colors = new Color[]
		{
			new Color(0.4f, 0.1f, 0.6f, 0f),
			new Color(0.4f, 0.1f, 0.6f, 0.5f),
			new Color(0f,   0.6f, 0.8f, 0.75f),
			new Color(0.2f, 0.9f, 0.5f, 0.85f),
			new Color(1f,   0.85f, 0.3f, 0.9f),
			new Color(1f,   1f,   1f,   1f),
		};
		_defaultGradient.Offsets = new float[]
		{
			0f, 0.15f, 0.35f, 0.6f, 0.8f, 1f,
		};
		return _defaultGradient;
	}


	// ══════════════════════════════════════════════════════════════════════
	// 生命周期
	// ══════════════════════════════════════════════════════════════════════

	/// <summary>停止后台线程并等待结束。在不再需要此实例时调用。</summary>
	public void Stop()
	{
		if (_thread != null && _thread.IsAlive)
			_thread.Join();
		_thread = null;
	}
}
