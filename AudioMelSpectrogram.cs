using Godot;
using System;
using System.Threading;

/// <summary>
/// 非 Godot 节点 / 资源的 Mel 频谱图渲染器。
///
/// === 架构 ===
/// 调用方节点._Process()
///   └─ AudioMelSpectrogram.UpdateView(audioPlayer, curT, maxT)
///        ├─ 首次检测 AudioStreamWav → 后台线程预计算全曲 FFT
///        ├─ 每帧从缓存取出当前可见时间段 → 颜色 LUT → byte[] → SetData → Update
///        └─ 视角未变时整帧跳过
///
/// 后台线程：PCM 读取(主线程) → FFT + Mel 滤波 + Log(子线程) → 缓存 Log 能量
/// 主线程：  按 visibleStart/End 映射到帧范围 → 逐列查 LUT → 写像素 → GPU upload
/// </summary>
/// <remarks>
/// 纯安全代码，无需 AllowUnsafeBlocks。
/// 注：Gradient 独立于 AudioStreamWav 加载，如需 MP3/OGG 支持，
/// 可在调用方解码后调用 LoadRawPcm() 注入 float[] PCM 数据。
/// </remarks>
public partial class AudioMelSpectrogram : RefCounted
{
	// ══════════════════════════════════════════════════════════════════════
	// [公共参数] 由调用方节点通过属性赋值
	// ══════════════════════════════════════════════════════════════════════

	/// <summary>渲染目标的 TextureRect。尺寸从此读取 × resolutionScale 决定纹理大小。</summary>
	public TextureRect targetImage { get; set; }

	/// <summary>纹理分辨率缩放。0.5 = 半分辨率（宽/高各砍一半，像素数 1/4）。</summary>
	public float resolutionScale { get; set; } = 0.5f;

	/// <summary>颜色渐变：归一化能量 [0,1] → Color。由调用方节点赋值。</summary>
	public Gradient colorGradient { get; set; }

	/// <summary>是否启用双声道显示。
	/// 启用后 TextureRect 上下分半：上半=左声道，下半=右声道。
	/// 需要配合外部解码器传入立体声 PCM 数据。</summary>
	public bool UseStereo { get; set; } = false;


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

	/// <summary>Blackman-Harris 4 项系数：旁瓣 ≤ -92dB。</summary>
	private const float BH_A0 = 0.35875f;
	private const float BH_A1 = 0.48829f;
	private const float BH_A2 = 0.14128f;
	private const float BH_A3 = 0.01168f;

	/// <summary>进度汇报粒度（每 N 帧更新一次 volatile 进度）。</summary>
	private const int PROGRESS_INTERVAL = 4;

	/// <summary>线程 Join 超时（毫秒）。</summary>
	private const int JOIN_TIMEOUT_MS = 3000;

	/// <summary>预计算 Blackman-Harris 窗（所有 FFT 帧共用）。</summary>
	private static readonly float[] Window = BuildWindow();


	// ══════════════════════════════════════════════════════════════════════
	// [预计算缓存] 后台线程写入 → 主线程只读
	// ══════════════════════════════════════════════════════════════════════

	/// <summary>当前已缓存的 AudioStreamWav 引用。检测变化时触发重新预计算。</summary>
	private AudioStreamWav _cachedWav;

	/// <summary>预计算完成后的 Log-Mel 能量缓存。
	/// 布局：flat[ frameIndex × MEL_BINS + melBin ]，每个元素 = Log(能量)。
	/// volatile 保证主线程读到最新值。</summary>
	private volatile float[] _logMelData;

	/// <summary>右声道 Log-Mel 能量缓存（立体声模式）。</summary>
	private volatile float[] _logMelDataRight;

	/// <summary>后台线程正在写入的临时缓存。完成后通过 _cacheReady 交给主线程。</summary>
	private float[] _pendingLogData;

	/// <summary>后台线程正在写入的右声道临时缓存。</summary>
	private float[] _pendingLogDataRight;

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

	/// <summary>上次渲染时的 UseStereo 值。用于检测变化并重建纹理映射。</summary>
	private bool _lastUseStereo;


	// ══════════════════════════════════════════════════════════════════════
	// [线程控制]
	// ══════════════════════════════════════════════════════════════════════

	/// <summary>后台线程引用。用于 Stop() 时等待线程结束。</summary>
	private Thread _thread;

	/// <summary>后台线程正在运行的标记。防止重复启动线程。</summary>
	private volatile bool _computing;

	/// <summary>取消标记。设置后后台线程应尽快退出。</summary>
	private volatile bool _cancelRequested;

	/// <summary>后台线程已计算完成的帧数。每算完一批通过 volatile 更新。</summary>
	private volatile int _framesComputed;

	/// <summary>立体声模式下左声道已计算帧数。</summary>
	private volatile int _framesComputedLeft;

	/// <summary>立体声模式下右声道已计算帧数。</summary>
	private volatile int _framesComputedRight;


	// ══════════════════════════════════════════════════════════════════════
	// [运行时 Image / Texture]
	// ══════════════════════════════════════════════════════════════════════

	/// <summary>实际写入的 ImageTexture。</summary>
	private ImageTexture _texture;

	/// <summary>CPU 侧 Image 对象。</summary>
	private Image _image;

	/// <summary>像素缓冲区（CPU 侧，byte[] RGBA8）。</summary>
	private byte[] _pixels;

	/// <summary>当前纹理尺寸。</summary>
	private int _texW, _texH;

	/// <summary>上一帧渲染的 FFT 帧号。</summary>
	private int _prevStartF, _prevEndF;


	// ══════════════════════════════════════════════════════════════════════
	// [颜色 LUT / 查找表]
	// ══════════════════════════════════════════════════════════════════════

	private byte[] _colorLut;
	private Gradient _appliedGradient;
	private int[] _binForY;
	private float[] _colVals;


	// ══════════════════════════════════════════════════════════════════════
	// [默认渐变] 线程安全的 Lazy 初始化
	// ══════════════════════════════════════════════════════════════════════

	private static readonly Lazy<Gradient> _defaultGradient = new Lazy<Gradient>(() => new Gradient
	{
		Colors = new Color[]
		{
			new Color(0.4f, 0.1f, 0.6f, 0f),
			new Color(0.4f, 0.1f, 0.6f, 0.5f),
			new Color(0f,   0.6f, 0.8f, 0.75f),
			new Color(0.2f, 0.9f, 0.5f, 0.85f),
			new Color(1f,   0.85f, 0.3f, 0.9f),
			new Color(1f,   1f,   1f,   1f),
		},
		Offsets = new float[] { 0f, 0.15f, 0.35f, 0.6f, 0.8f, 1f },
	});


	// ══════════════════════════════════════════════════════════════════════
	// 每帧入口
	// ══════════════════════════════════════════════════════════════════════

	/// <summary>每帧由调用方节点 _Process() 调用。</summary>
	public void UpdateView(AudioStreamPlayer audioPlayer, float visibleStart, float visibleEnd, bool forceUpdate = false)
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
				_logMelDataRight = UseStereo ? _pendingLogDataRight : null;
				_totalFrames = _logMelData.Length / MEL_BINS;
				_cachedWav = audioPlayer?.Stream as AudioStreamWav;
				_cacheReady = false;
				_computing = false;
			}
			_framesComputedLeft = 0;
			_framesComputedRight = 0;
			_pendingLogData = null;
			_pendingLogDataRight = null;
		}

		// ── 完全态：全缓存就绪，正常渲染 ────────────────────────────────
		if (!_computing && _logMelData != null && _totalFrames > 0)
		{
			RenderCurrentView(visibleStart, visibleEnd, forceUpdate);
			return;
		}

		// ── 部分态：后台计算中，但已有部分帧可供显示 ────────────────────
		// 立体声只要左右任意一个有进度就渲染
		bool hasPartial = UseStereo
			? (_framesComputedLeft > 0 || _framesComputedRight > 0)
			: (_framesComputed > 0);

		if (_computing && _pendingLogData != null && hasPartial)
		{
			if (UseStereo)
			{
				ShowPartialStereo(visibleStart, visibleEnd,
					_framesComputedLeft, _framesComputedRight);
			}
			else
			{
				// fillMode=false：未计算区域留空，逐步刷出
				RenderCurrentView(visibleStart, visibleEnd, true,
					_pendingLogData, _framesComputed, false);
			}
		}
	}

	/// <summary>备选入口：直接注入原始 PCM float 数据（适用于非 WAV 格式音频）。</summary>
	public void LoadRawPcm(float[] monoData, int sampleRate)
	{
		if (_computing) return;

		_computing = true;
		_cacheReady = false;
		_framesComputed = 0;
		_sampleRate = sampleRate;
		_timePerFrame = (float)HOP_SIZE / _sampleRate;

		BuildMelFilterBank();

		int totalSamples = monoData.Length;
		int totalFrames = Mathf.Max(1, (totalSamples - FFT_SIZE) / HOP_SIZE + 1);
		float[] pending = new float[totalFrames * MEL_BINS];
		_pendingLogData = pending;
		_pendingLogDataRight = null;

		// ── 同步计算第 1 帧，让显示立刻出现 ──
		ComputeFftMelLog(monoData, totalSamples, Mathf.Min(1, totalFrames), pending,
			setCacheReady: false);
		_cacheReady = false;  // 双重保险
		_framesComputed = 1;

		_cancelRequested = true;      // 信号旧线程退出，不阻塞等待
		_cacheReady = false;           // 清除旧线程可能残留的完成信号
		_cancelRequested = false;
		_thread = StartBgThread(() =>
		{
			ComputeFftMelLog(monoData, totalSamples, totalFrames, pending, null, 1);
			_cacheReady = true;
		});
	}

	/// <summary>备选入口：注入立体声交错 PCM float 数据。</summary>
	public void LoadRawPcmStereo(float[] stereoData, int sampleRate)
	{
		if (_computing) return;

		int totalSamples = stereoData.Length / 2;

		_computing = true;
		_cacheReady = false;
		_framesComputed = 0;
		_sampleRate = sampleRate;
		_timePerFrame = (float)HOP_SIZE / _sampleRate;

		BuildMelFilterBank();

		int totalFrames = Mathf.Max(1, (totalSamples - FFT_SIZE) / HOP_SIZE + 1);
		float[] pending  = new float[totalFrames * MEL_BINS];
		float[] pendingR = new float[totalFrames * MEL_BINS];
		_pendingLogData      = pending;
		_pendingLogDataRight = pendingR;

		float[] left  = new float[totalSamples];
		float[] right = new float[totalSamples];
		for (int i = 0; i < totalSamples; i++)
		{
			left[i]  = stereoData[i * 2];
			right[i] = stereoData[i * 2 + 1];
		}

		_framesComputedLeft  = 0;
		_framesComputedRight = 0;

		// ── 同步计算左右声道第 1 帧 ──
		ComputeFftMelLog(left, totalSamples, Mathf.Min(1, totalFrames), pending,
			p => { _framesComputedLeft = p; }, setCacheReady: false);
		ComputeFftMelLog(right, totalSamples, Mathf.Min(1, totalFrames), pendingR,
			p => { _framesComputedRight = p; }, setCacheReady: false);
		_cacheReady = false;  // 双重保险
		_framesComputedLeft  = 1;
		_framesComputedRight = 1;

		_cancelRequested = true;      // 信号旧线程退出，不阻塞等待
		_cacheReady = false;           // 清除旧线程可能残留的完成信号
		_cancelRequested = false;
		_thread = StartBgThread(() =>
		{
			var tL = StartBgThread(() =>
				ComputeFftMelLog(left, totalSamples, totalFrames, pending,
					p => { _framesComputedLeft = p; }, 1));
			var tR = StartBgThread(() =>
				ComputeFftMelLog(right, totalSamples, totalFrames, pendingR,
					p => { _framesComputedRight = p; }, 1));
			tL.Join();
			tR.Join();
			_cacheReady = true;
		});
	}


	// ══════════════════════════════════════════════════════════════════════
	// 渲染当前视图
	// ══════════════════════════════════════════════════════════════════════

	private void RenderCurrentView(float visibleStart, float visibleEnd, bool forceUpdate,
		float[] data = null, int framesAvail = 0, bool fillMode = false)
	{
		if (data == null) { data = _logMelData; framesAvail = _totalFrames; }
		if (data == null || framesAvail == 0) return;

		Vector2 size = targetImage.Size;
		int w = Mathf.Max(1, Mathf.RoundToInt(size.X * resolutionScale));
		int h = Mathf.Max(1, Mathf.RoundToInt(size.Y * resolutionScale));
		bool sizeChanged = _texture == null || _texW != w || _texH != h;
		bool stereoChanged = UseStereo != _lastUseStereo;

		if (sizeChanged || stereoChanged)
		{
			InitTexture(w, h);
			_lastUseStereo = UseStereo;
		}

		float dur = visibleEnd - visibleStart;
		if (dur <= 0f) return;

		float frameOffset = FFT_SIZE / (2f * HOP_SIZE);
		int startF = Mathf.Max(0, Mathf.RoundToInt(visibleStart / _timePerFrame - frameOffset));
		int endF = Mathf.Max(1, Mathf.RoundToInt(visibleEnd / _timePerFrame - frameOffset));

		if (!forceUpdate && startF == _prevStartF && endF == _prevEndF && _pixels != null)
			return;

		int visFrames = endF - startF;
		BuildColorLut();

		if (UseStereo && _logMelDataRight != null && data == _logMelData)
		{
			int halfH = h / 2;
			RenderColumns(startF, visFrames, w, h, 0, w - 1, _logMelData,      framesAvail, 0,     halfH, fillMode);
			RenderColumns(startF, visFrames, w, h, 0, w - 1, _logMelDataRight, framesAvail, halfH, h,     fillMode);
		}
		else
		{
			RenderColumns(startF, visFrames, w, h, 0, w - 1, data, framesAvail, 0, h, fillMode);
		}

		_image.SetData(w, h, false, Image.Format.Rgba8, _pixels);
		_texture.Update(_image);

		_prevStartF = startF;
		_prevEndF = endF;
		_texW = w;
		_texH = h;
	}


	// ══════════════════════════════════════════════════════════════════════
	// 列渲染（热路径）
	// ══════════════════════════════════════════════════════════════════════

	private void RenderColumns(int startF, int visFrames, int w, int h,
		int colFrom, int colTo,
		float[] data, int totalFrames,
		int yStart = 0, int yEnd = -1,
		bool fillMode = false)
	{
		if (yEnd < 0) yEnd = h;
		int rows = yEnd - yStart;
		if (rows <= 0) return;

		float[] colVals = _colVals;

		for (int x = colFrom; x <= colTo; x++)
		{
			float t = (float)x / w;
			int frameIdx = startF + (int)Mathf.Round(t * visFrames);
			if (frameIdx >= totalFrames || frameIdx < 0)
			{
				if (fillMode)
				{
					frameIdx = Mathf.Clamp(frameIdx, 0, totalFrames - 1);
				}
				else
				{
					// 显式清除脏像素残留，防止 Godot 复用 _pixels 的旧帧拖影
					for (int iy = 0; iy < rows; iy++)
					{
						int absY = yStart + iy;
						int off = (absY * w + x) * 4;
						_pixels[off]     = 0;
						_pixels[off + 1] = 0;
						_pixels[off + 2] = 0;
						_pixels[off + 3] = 0;
					}
					continue;
				}
			}
			int rowBase = frameIdx * MEL_BINS;

			float mn = float.MaxValue, mx = float.MinValue;
			for (int iy = 0; iy < rows; iy++)
			{
				int absY = yStart + iy;
				float v = data[rowBase + _binForY[absY]];
				colVals[iy] = v;
				if (v < mn) mn = v;
				if (v > mx) mx = v;
			}
			float invRange = 1f / Mathf.Max(mx - mn, 0.001f);

			for (int iy = 0; iy < rows; iy++)
			{
				float norm = (colVals[iy] - mn) * invRange;
				int idx = (int)(norm * 255);
				if (idx > 255) idx = 255;

				int absY = yStart + iy;
				int pixelOff = (absY * w + x) * 4;
				int lutOff   = idx * 4;
				_pixels[pixelOff]     = _colorLut[lutOff];
				_pixels[pixelOff + 1] = _colorLut[lutOff + 1];
				_pixels[pixelOff + 2] = _colorLut[lutOff + 2];
				_pixels[pixelOff + 3] = _colorLut[lutOff + 3];
			}
		}
	}


	// ══════════════════════════════════════════════════════════════════════
	// 立体声部分渲染
	// ══════════════════════════════════════════════════════════════════════

	private void ShowPartialStereo(float visibleStart, float visibleEnd,
		int framesLeft, int framesRight)
	{
		if (_pendingLogData == null || targetImage == null) return;

		Vector2 size = targetImage.Size;
		int w = Mathf.Max(1, Mathf.RoundToInt(size.X * resolutionScale));
		int h = Mathf.Max(1, Mathf.RoundToInt(size.Y * resolutionScale));
		bool sizeChanged = _texture == null || _texW != w || _texH != h;
		bool stereoChanged = UseStereo != _lastUseStereo;

		if (sizeChanged || stereoChanged)
		{
			InitTexture(w, h);
			_lastUseStereo = UseStereo;
		}

		float dur = visibleEnd - visibleStart;
		if (dur <= 0f) return;

		float frameOffset = FFT_SIZE / (2f * HOP_SIZE);
		int startF = Mathf.Max(0, Mathf.RoundToInt(visibleStart / _timePerFrame - frameOffset));
		int endF = Mathf.Max(1, Mathf.RoundToInt(visibleEnd / _timePerFrame - frameOffset));
		int visFrames = endF - startF;

		BuildColorLut();

		int halfH = h / 2;

		if (framesLeft > 0)
			RenderColumns(startF, visFrames, w, h, 0, w - 1, _pendingLogData, framesLeft, 0, halfH, true);

		if (_pendingLogDataRight != null && framesRight > 0)
			RenderColumns(startF, visFrames, w, h, 0, w - 1, _pendingLogDataRight, framesRight, halfH, h, true);

		_image.SetData(w, h, false, Image.Format.Rgba8, _pixels);
		_texture.Update(_image);
	}


	// ══════════════════════════════════════════════════════════════════════
	// 后台预计算
	// ══════════════════════════════════════════════════════════════════════

	private void StartPreCompute(AudioStreamWav wav)
	{
		// 立即初始化纹理（清空为黑色）
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

		BuildMelFilterBank();

		int channels = wav.Stereo ? 2 : 1;
		int formatVal = (int)wav.Format;
		GD.Print($"MelSpec: format={formatVal} stereo={channels} rate={_sampleRate}");

		if (formatVal >= 2)
		{
			GD.PrintErr($"MelSpec: unsupported format({formatVal}), PCM 8/16bit only");
			_computing = false;
			return;
		}
		int bytesPerSample = formatVal == 1 ? 2 : 1;

		// 信号旧线程退出，不阻塞
		_cancelRequested = true;
		_cacheReady = false;
		_cancelRequested = false;

		// 全部放后台：wav.Data 读取 + PCM 解析 + FFT，主线程不碰大数组
		_thread = StartBgThread(() =>
		{
			byte[] rawData = wav.Data;
			int totalSamples = rawData.Length / bytesPerSample / channels;
			int totalFrames = Mathf.Max(1, (totalSamples - FFT_SIZE) / HOP_SIZE + 1);
			float[] pending = new float[totalFrames * MEL_BINS];

			if (UseStereo && channels >= 2)
			{
				float[] pendingR = new float[totalFrames * MEL_BINS];
				_pendingLogData = pending;
				_pendingLogDataRight = pendingR;
				_framesComputedLeft  = 0;
				_framesComputedRight = 0;

				float[] left  = new float[totalSamples];
				float[] right = new float[totalSamples];
				ReadStereoPcm(rawData, bytesPerSample, channels, totalSamples, left, right);

				if (_cancelRequested) return;

				var tL = StartBgThread(() =>
					ComputeFftMelLog(left, totalSamples, totalFrames, pending,
						p => { _framesComputedLeft = p; }));
				var tR = StartBgThread(() =>
					ComputeFftMelLog(right, totalSamples, totalFrames, pendingR,
						p => { _framesComputedRight = p; }));
				tL.Join();
				tR.Join();
			}
			else
			{
				_pendingLogData = pending;
				_pendingLogDataRight = null;
				ComputeFftMelLogFromPcm(rawData, bytesPerSample, channels,
					totalSamples, totalFrames, pending, 0);
			}

			_cacheReady = true;
		});
	}


	/// <summary>FFT + Mel 计算核心。</summary>
	private void ComputeFftMelLog(float[] mono, int totalSamples, int totalFrames, float[] pending,
		Action<int> reportProgress = null, int startFrame = 0,
		bool setCacheReady = true)
	{
		if (mono == null)
			throw new ArgumentNullException(nameof(mono));
		if (pending == null)
			throw new ArgumentNullException(nameof(pending));

		float[] real  = new float[FFT_SIZE];
		float[] imag  = new float[FFT_SIZE];
		float[] power = new float[FFT_SIZE / 2];

		for (int f = startFrame; f < totalFrames; f++)
		{
			if (_cancelRequested) return;

			int offset = f * HOP_SIZE;

			for (int i = 0; i < FFT_SIZE; i++)
			{
				int idx = offset + i;
				real[i] = idx < totalSamples ? mono[idx] * Window[i] : 0f;
				imag[i] = 0f;
			}

			FFT(real, imag);

			for (int i = 0; i < FFT_SIZE / 2; i++)
				power[i] = real[i] * real[i] + imag[i] * imag[i];

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

			if ((f & (PROGRESS_INTERVAL - 1)) == 0 || f == totalFrames - 1)
			{
				if (reportProgress != null)
					reportProgress(f + 1);
				else
					_framesComputed = f + 1;
			}
		}

		if (setCacheReady)
			_cacheReady = true;
	}


	/// <summary>从原始 PCM byte[] 直接计算 FFT + Mel + Log。</summary>
	private void ComputeFftMelLogFromPcm(byte[] rawData, int bytesPerSample, int channels,
		int totalSamples, int totalFrames, float[] pending,
		int startFrame = 0, bool setCacheReady = true)
	{
		if (rawData == null)
			throw new ArgumentNullException(nameof(rawData));
		if (pending == null)
			throw new ArgumentNullException(nameof(pending));

		float[] real  = new float[FFT_SIZE];
		float[] imag  = new float[FFT_SIZE];
		float[] power = new float[FFT_SIZE / 2];

		for (int f = startFrame; f < totalFrames; f++)
		{
			if (_cancelRequested) return;

			int offset = f * HOP_SIZE;

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
					real[i] = (float)(sum / channels) * Window[i];
				}
				else
				{
					real[i] = 0f;
				}
				imag[i] = 0f;
			}

			FFT(real, imag);

			for (int i = 0; i < FFT_SIZE / 2; i++)
				power[i] = real[i] * real[i] + imag[i] * imag[i];

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

			if ((f & (PROGRESS_INTERVAL - 1)) == 0 || f == totalFrames - 1)
				_framesComputed = f + 1;
		}

		if (setCacheReady)
			_cacheReady = true;
	}


	// ══════════════════════════════════════════════════════════════════════
	// PCM 读取辅助
	// ══════════════════════════════════════════════════════════════════════

	private static void ReadStereoPcm(byte[] rawData, int bytesPerSample, int channels,
		int totalSamples, float[] left, float[] right)
	{
		if (bytesPerSample == 2)
		{
			for (int i = 0; i < totalSamples; i++)
			{
				int offL = (i * channels) * 2;
				int offR = (i * channels + 1) * 2;
				left[i]  = (short)(rawData[offL] | (rawData[offL + 1] << 8)) / 32768f;
				right[i] = (short)(rawData[offR] | (rawData[offR + 1] << 8)) / 32768f;
			}
		}
		else
		{
			for (int i = 0; i < totalSamples; i++)
			{
				int offL = i * channels;
				int offR = i * channels + 1;
				left[i]  = (rawData[offL] - 128) / 128f;
				right[i] = (rawData[offR] - 128) / 128f;
			}
		}
	}


	// ══════════════════════════════════════════════════════════════════════
	// Mel 三角滤波器组
	// ══════════════════════════════════════════════════════════════════════

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
			float mL = melMin + (melMax - melMin) * m / (MEL_BINS + 1);
			float mC = melMin + (melMax - melMin) * (m + 1) / (MEL_BINS + 1);
			float mR = melMin + (melMax - melMin) * (m + 2) / (MEL_BINS + 1);
			float fL = MelToFreq(mL), fC = MelToFreq(mC), fR = MelToFreq(mR);

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
	// FFT — 实序列原位 Cooley-Tukey radix-2 DIT
	// ══════════════════════════════════════════════════════════════════════

	private static void FFT(float[] real, float[] imag)
	{
		int n = real.Length;

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

	private static float FreqToMel(float f) => MEL_A * (float)Math.Log10(1f + f / MEL_B);
	private static float MelToFreq(float m) => MEL_B * (Mathf.Pow(10f, m / MEL_A) - 1f);


	// ══════════════════════════════════════════════════════════════════════
	// 纹理 / LUT / Bin 表
	// ══════════════════════════════════════════════════════════════════════

	private void InitTexture(int w, int h)
	{
		_pixels = new byte[w * h * 4];

		_image = Image.CreateEmpty(w, h, false, Image.Format.Rgba8);
		_image.SetData(w, h, false, Image.Format.Rgba8, _pixels);

		_texture = ImageTexture.CreateFromImage(_image);

		if (targetImage is not null)
			targetImage.Texture = _texture;

		_binForY = new int[h];
		if (UseStereo && h >= 4)
		{
			int halfH = h / 2;
			for (int y = 0; y < h; y++)
			{
				int localY = y % halfH;
				_binForY[y] = Mathf.Clamp(
					Mathf.RoundToInt((float)localY / halfH * MEL_BINS),
					0, MEL_BINS - 1);
			}
		}
		else
		{
			for (int y = 0; y < h; y++)
				_binForY[y] = Mathf.Clamp(
					Mathf.RoundToInt((float)y / h * MEL_BINS),
					0, MEL_BINS - 1);
		}

		_colVals = new float[h];
		_appliedGradient = null;
	}

	private void BuildColorLut()
	{
		var grad = colorGradient ?? _defaultGradient.Value;

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
	// 窗函数
	// ══════════════════════════════════════════════════════════════════════

	private static float[] BuildWindow()
	{
		float[] w = new float[FFT_SIZE];
		double invN1 = 1.0 / (FFT_SIZE - 1);
		for (int i = 0; i < FFT_SIZE; i++)
		{
			double p = Math.PI * 2.0 * i * invN1;
			w[i] = (float)(BH_A0 - BH_A1 * Math.Cos(p) + BH_A2 * Math.Cos(2.0 * p) - BH_A3 * Math.Cos(3.0 * p));
		}
		return w;
	}


	// ══════════════════════════════════════════════════════════════════════
	// 线程辅助
	// ══════════════════════════════════════════════════════════════════════

	private static Thread StartBgThread(Action action)
	{
		var t = new Thread(() =>
		{
			try { action(); }
			catch (Exception ex) { GD.PrintErr($"MelSpec bg thread error: {ex.Message}"); }
		})
		{ IsBackground = true };
		t.Start();
		return t;
	}

	private void SafeJoin(ref Thread thread)
	{
		if (thread == null || !thread.IsAlive) { thread = null; return; }
		_cancelRequested = true;
		if (!thread.Join(JOIN_TIMEOUT_MS))
			GD.PrintErr("MelSpec: bg thread did not exit within timeout, abandoning");
		if (thread.IsAlive)
			GD.PrintErr("MelSpec: bg thread still alive after timeout, detaching");
		thread = null;
	}


	// ══════════════════════════════════════════════════════════════════════
	// 生命周期
	// ══════════════════════════════════════════════════════════════════════

	public void ResetView()
	{
		if (targetImage == null) return;
		var size = targetImage.Size;
		int w = Mathf.Max(1, Mathf.RoundToInt(size.X * resolutionScale));
		int h = Mathf.Max(1, Mathf.RoundToInt(size.Y * resolutionScale));
		InitTexture(w, h);
	}

	public void Stop()
	{
		SafeJoin(ref _thread);
	}
}
