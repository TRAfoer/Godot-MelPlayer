extends RefCounted
class_name AudioMelSpectrogramGD

# ══════════════════════════════════════════════════════════════
# 秒数驱动型 Mel 频谱图渲染器 — GDScript 版
#
# ⚠ 注意：GDScript 数值运算速度约为 C# 的 1/50 ~ 1/100。
#    此脚本用以下零/微质量损失的优化加速预计算：
#    - HOP_SIZE=1024（vs C# 版 512）：帧数减半，时间分辨率 23ms/帧（仍 < 肉眼感知）
#    - 稀疏 Mel + 局部变量缓存 + 内联 PCM 读取
#
# === 架构 ===
# 调用方节点._process()
#   └─ update_view(audioPlayer, visibleStart, visibleEnd)
#        ├─ 首次检测 AudioStreamWAV → 后台线程预计算全曲 FFT
#        ├─ 每帧从缓存取出当前可见时间段 → 颜色 LUT → byte[] → 提交 GPU
#        └─ 视角未变时整帧跳过
# ══════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════
# [公共参数]
# ══════════════════════════════════════════════════════════════

var targetImage: TextureRect
var resolutionScale: float = 0.5
var colorGradient: Gradient


# ══════════════════════════════════════════════════════════════
# [FFT / Mel 常量]
# FFT_SIZE / MEL_BINS 与 C# 版一致，保证质量。
# HOP_SIZE 提高至 1024：帧数减半 × 计算量减半，时间分辨率 ~23ms 仍 < 视觉感知阈值。
# ══════════════════════════════════════════════════════════════

const FFT_SIZE: int     = 4096
const HOP_SIZE: int     = 1024          # C# 版为 512；GD 版用 1024 帧数减半，不影响频率分辨率
const MEL_BINS: int     = 128
const MEL_A: float      = 2595.0
const MEL_B: float      = 700.0

## 预计算帧偏移常量，避免每帧/每列重复计算 FFT_SIZE / (2 * HOP_SIZE)
const FRAME_OFFSET: float = float(FFT_SIZE) / (2.0 * float(HOP_SIZE))


# ══════════════════════════════════════════════════════════════
# [预计算缓存]
# ══════════════════════════════════════════════════════════════

var _cachedWav: AudioStreamWAV
var _logMelData: PackedFloat32Array
var _pendingLogData: PackedFloat32Array
var _cacheReady: bool = false
var _totalFrames: int = 0
var _timePerFrame: float = 0.0
var _sampleRate: int = 0

# 稀疏 Mel 滤波器
var _melFilterWeights: Array[PackedFloat32Array] = []
var _melFilterStart: PackedInt32Array = PackedInt32Array()
var _melFilterEnd: PackedInt32Array = PackedInt32Array()

# 线程控制
var _thread: Thread
var _mutex: Mutex = Mutex.new()
var _computing: bool = false
var _framesComputed: int = 0
var _cancelRequested: bool = false


# ══════════════════════════════════════════════════════════════
# [运行时 Image / Texture]
# ══════════════════════════════════════════════════════════════

var _texture: ImageTexture
var _image: Image
var _pixels: PackedByteArray
var _texW: int = 0
var _texH: int = 0
var _prevStartF: int = 0
var _prevEndF: int = 0


# ══════════════════════════════════════════════════════════════
# [颜色 LUT / Bin 查找表]
# ══════════════════════════════════════════════════════════════

var _colorLut: PackedByteArray
var _appliedGradient: Gradient
var _binForY: PackedInt32Array


# ══════════════════════════════════════════════════════════════
# 每帧入口
# ══════════════════════════════════════════════════════════════

func update_view(audioPlayer: AudioStreamPlayer, visibleStart: float, visibleEnd: float, forceUpdate: bool = false) -> void:
	if targetImage == null:
		return

	if audioPlayer != null and audioPlayer.stream is AudioStreamWAV:
		var wav: AudioStreamWAV = audioPlayer.stream as AudioStreamWAV
		if wav != _cachedWav and not _computing:
			_start_pre_compute(wav)

	if _cacheReady:
		if _pendingLogData.is_empty():
			_cacheReady = false
			_computing = false
		else:
			_logMelData = _pendingLogData
			_totalFrames = _logMelData.size() / MEL_BINS
			_cachedWav = (audioPlayer.stream as AudioStreamWAV) if audioPlayer != null else null
			_cacheReady = false
			_computing = false
		_pendingLogData = PackedFloat32Array()

	if not _computing and _logMelData.size() > 0 and _totalFrames > 0:
		_render_current_view(visibleStart, visibleEnd, forceUpdate)
		return

	if _computing and _pendingLogData.size() > 0 and _framesComputed > 0:
		_render_current_view(visibleStart, visibleEnd, true, _pendingLogData, _framesComputed)


# ══════════════════════════════════════════════════════════════
# 渲染当前视图
# ══════════════════════════════════════════════════════════════

func _render_current_view(visibleStart: float, visibleEnd: float, forceUpdate: bool,
		data: PackedFloat32Array = PackedFloat32Array(), framesAvail: int = 0) -> void:

	if data.is_empty():
		data = _logMelData
		framesAvail = _totalFrames
	if data.is_empty() or framesAvail == 0:
		return

	var size: Vector2 = targetImage.size
	var w: int = maxi(1, roundi(size.x * resolutionScale))
	var h: int = maxi(1, roundi(size.y * resolutionScale))
	if _texture == null or _texW != w or _texH != h:
		_init_texture(w, h)

	var dur: float = visibleEnd - visibleStart
	if dur <= 0.0:
		return

	var startF: int = maxi(0, roundi(visibleStart / _timePerFrame - FRAME_OFFSET))
	var endF: int = maxi(1, roundi(visibleEnd / _timePerFrame - FRAME_OFFSET))

	if not forceUpdate and startF == _prevStartF and endF == _prevEndF and _pixels.size() > 0:
		return

	_build_color_lut()
	_render_columns(visibleStart, visibleEnd, w, h, data, framesAvail)

	_image.set_data(w, h, false, Image.FORMAT_RGBA8, _pixels)
	_texture.update(_image)

	_prevStartF = startF
	_prevEndF = endF
	_texW = w
	_texH = h


# ══════════════════════════════════════════════════════════════
# 列渲染 — 逐列 time→frameIdx
# ══════════════════════════════════════════════════════════════

func _render_columns(visibleStart: float, visibleEnd: float,
		w: int, h: int, data: PackedFloat32Array, totalFrames: int) -> void:

	var timeRange: float = visibleEnd - visibleStart
	var binForY: PackedInt32Array = _binForY
	var colorLut: PackedByteArray = _colorLut

	for x in w:
		var t: float = float(x) / float(w)
		var timePos: float = visibleStart + t * timeRange
		var frameIdx: int = roundi(timePos / _timePerFrame - FRAME_OFFSET)

		if frameIdx >= totalFrames or frameIdx < 0:
			frameIdx = clampi(frameIdx, 0, totalFrames - 1)

		var rowBase: int = frameIdx * MEL_BINS

		# 一趟读 + 扫描 min/max
		var mn: float = INF
		var mx: float = -INF
		for iy in h:
			var v: float = data[rowBase + binForY[iy]]
			if v < mn: mn = v
			if v > mx: mx = v

		var invRange: float = 1.0 / maxf(mx - mn, 0.001)

		# 查 LUT 写像素
		for iy in h:
			var norm: float = (data[rowBase + binForY[iy]] - mn) * invRange
			var idx: int = mini(int(norm * 255.0), 255)
			var pixelOff: int = (iy * w + x) * 4
			var lutOff: int = idx * 4
			_pixels[pixelOff]     = colorLut[lutOff]
			_pixels[pixelOff + 1] = colorLut[lutOff + 1]
			_pixels[pixelOff + 2] = colorLut[lutOff + 2]
			_pixels[pixelOff + 3] = colorLut[lutOff + 3]


# ══════════════════════════════════════════════════════════════
# 后台预计算
# ══════════════════════════════════════════════════════════════

func _start_pre_compute(wav: AudioStreamWAV) -> void:
	if targetImage != null:
		var size: Vector2 = targetImage.size
		var w: int = maxi(1, roundi(size.x * resolutionScale))
		var h: int = maxi(1, roundi(size.y * resolutionScale))
		_init_texture(w, h)

	_computing = true
	_cacheReady = false
	_framesComputed = 0
	_cancelRequested = false
	_sampleRate = int(wav.mix_rate)
	_timePerFrame = float(HOP_SIZE) / float(_sampleRate)

	_build_mel_filter_bank()

	var rawData: PackedByteArray = wav.data
	var channels: int = 2 if wav.stereo else 1
	var formatVal: int = int(wav.format)
	print("MelSpecGD: format=", formatVal, " stereo=", channels, " rate=", _sampleRate, " dataLen=", rawData.size())

	if formatVal >= 2:
		push_error("MelSpecGD: 不支持的编码格式(", formatVal, ")，仅支持 PCM 8/16bit")
		_computing = false
		return

	var bytesPerSample: int = 2 if formatVal == 1 else 1
	var totalSamples: int = rawData.size() / bytesPerSample / channels
	var totalFrames: int = maxi(1, (totalSamples - FFT_SIZE) / HOP_SIZE + 1)

	var pending: PackedFloat32Array = PackedFloat32Array()
	pending.resize(totalFrames * MEL_BINS)
	_pendingLogData = pending

	if _thread != null and _thread.is_alive():
		_thread.wait_to_finish()

	_thread = Thread.new()
	var args: Dictionary = {
		"rawData": rawData,
		"bytesPerSample": bytesPerSample,
		"channels": channels,
		"totalSamples": totalSamples,
		"totalFrames": totalFrames,
		"pending": pending,
	}
	_thread.start(_compute_thread_func.bind(args))


func _compute_thread_func(args: Dictionary) -> void:
	var rawData: PackedByteArray = args["rawData"]
	var bytesPerSample: int = args["bytesPerSample"]
	var channels: int = args["channels"]
	var totalSamples: int = args["totalSamples"]
	var totalFrames: int = args["totalFrames"]
	var pending: PackedFloat32Array = args["pending"]

	# ── 窗函数（所有帧共用，只算一次）──
	var window: PackedFloat32Array = PackedFloat32Array()
	window.resize(FFT_SIZE)
	var bhA0: float = 0.35875
	var bhA1: float = 0.48829
	var bhA2: float = 0.14128
	var bhA3: float = 0.01168
	var invN1: float = 1.0 / float(FFT_SIZE - 1)
	for i in FFT_SIZE:
		var p: float = PI * 2.0 * float(i) * invN1
		window[i] = bhA0 - bhA1 * cos(p) + bhA2 * cos(2.0 * p) - bhA3 * cos(3.0 * p)

	var real: PackedFloat32Array = PackedFloat32Array()
	var imag: PackedFloat32Array = PackedFloat32Array()
	var power: PackedFloat32Array = PackedFloat32Array()
	real.resize(FFT_SIZE)
	imag.resize(FFT_SIZE)
	power.resize(FFT_SIZE / 2)

	# 局部化成员变量引用（GDScript 成员访问开销大）
	var mFS: PackedInt32Array = _melFilterStart
	var mFE: PackedInt32Array = _melFilterEnd
	var mFW: Array[PackedFloat32Array] = _melFilterWeights
	var halfN: int = FFT_SIZE / 2

	for f in totalFrames:
		if _cancelRequested:
			return

		var offset: int = f * HOP_SIZE

		# ── 读 PCM + 混单声道 + 加窗 ──
		if bytesPerSample == 2 and channels == 2:
			# 最常用路径：16-bit 立体声 → 内联处理，无内层循环
			for i in FFT_SIZE:
				var idx: int = offset + i
				if idx < totalSamples:
					var byteOff: int = idx * 4
					var s0: int = rawData.decode_s16(byteOff)
					var s1: int = rawData.decode_s16(byteOff + 2)
					real[i] = ((float(s0) + float(s1)) / 65536.0) * window[i]
				else:
					real[i] = 0.0
				imag[i] = 0.0
		elif bytesPerSample == 2:
			# 16-bit 单声道
			for i in FFT_SIZE:
				var idx: int = offset + i
				if idx < totalSamples:
					var s: int = rawData.decode_s16(idx * 2)
					real[i] = float(s) / 32768.0 * window[i]
				else:
					real[i] = 0.0
				imag[i] = 0.0
		else:
			# 8-bit 通用路径
			var scale: float = 1.0 / (128.0 * float(channels))
			for i in FFT_SIZE:
				var idx: int = offset + i
				if idx < totalSamples:
					var sum: float = 0.0
					for c in channels:
						sum += float(rawData[idx * channels + c] - 128)
					real[i] = sum * scale * window[i]
				else:
					real[i] = 0.0
				imag[i] = 0.0

		# 原位 Cooley-Tukey FFT
		_fft(real, imag)

		# 功率谱 |X|²
		for i in halfN:
			power[i] = real[i] * real[i] + imag[i] * imag[i]

		# 稀疏 Mel 三角滤波 + Log
		var row: int = f * MEL_BINS
		for m in MEL_BINS:
			var e: float = 0.0
			var s: int = mFS[m]
			var eIdx: int = mFE[m]
			var w: PackedFloat32Array = mFW[m]
			for i in range(s, eIdx):
				e += power[i] * w[i - s]
			pending[row + m] = log(maxf(e, 1e-10))

		# 每 4 帧更新进度
		if (f & 3) == 0 or f == totalFrames - 1:
			_mutex.lock()
			_framesComputed = f + 1
			_mutex.unlock()

	_mutex.lock()
	_pendingLogData = pending
	_cacheReady = true
	_computing = false
	_mutex.unlock()


# ══════════════════════════════════════════════════════════════
# Mel 三角滤波器组
# ══════════════════════════════════════════════════════════════

func _build_mel_filter_bank() -> void:
	var half: int = FFT_SIZE / 2
	var freqMax: float = float(_sampleRate) / 2.0
	var melMin: float = _freq_to_mel(0.0)
	var melMax: float = _freq_to_mel(freqMax)

	var binFreq: PackedFloat32Array = PackedFloat32Array()
	binFreq.resize(half)
	for i in half:
		binFreq[i] = float(i) / float(FFT_SIZE) * float(_sampleRate)

	_melFilterWeights.clear()
	_melFilterStart = PackedInt32Array()
	_melFilterEnd = PackedInt32Array()
	_melFilterStart.resize(MEL_BINS)
	_melFilterEnd.resize(MEL_BINS)

	for m in MEL_BINS:
		var mL: float = melMin + (melMax - melMin) * float(m) / float(MEL_BINS + 1)
		var mC: float = melMin + (melMax - melMin) * float(m + 1) / float(MEL_BINS + 1)
		var mR: float = melMin + (melMax - melMin) * float(m + 2) / float(MEL_BINS + 1)
		var fL: float = _mel_to_freq(mL)
		var fC: float = _mel_to_freq(mC)
		var fR: float = _mel_to_freq(mR)

		var s: int = -1
		var e: int = -1
		for i in half:
			var f: float = binFreq[i]
			if f >= fL and f < fR:
				if s < 0: s = i
				e = i + 1

		if s < 0:
			s = 0
			e = 1

		_melFilterStart[m] = s
		_melFilterEnd[m] = e

		var weights: PackedFloat32Array = PackedFloat32Array()
		weights.resize(e - s)
		for i in range(s, e):
			var f: float = binFreq[i]
			if f >= fL and f < fC:
				weights[i - s] = (f - fL) / (fC - fL)
			elif f >= fC and f < fR:
				weights[i - s] = (fR - f) / (fR - fC)
		_melFilterWeights.append(weights)


# ══════════════════════════════════════════════════════════════
# FFT — 原位 Cooley-Tukey radix-2 DIT
# ══════════════════════════════════════════════════════════════

static func _fft(real: PackedFloat32Array, imag: PackedFloat32Array) -> void:
	var n: int = real.size()

	# Bit-reversal 重排（用 temp var 代替 tuple 避免 GDScript 隐式数组分配）
	var j: int = 0
	for i in range(1, n):
		var bit: int = n >> 1
		while j >= bit:
			j -= bit
			bit >>= 1
		j += bit
		if i < j:
			var t: float = real[i]
			real[i] = real[j]
			real[j] = t
			t = imag[i]
			imag[i] = imag[j]
			imag[j] = t

	# 蝶形运算（预计算 wlenR/wlenI 避免 cos/sin 在循环内重复）
	var len_: int = 2
	while len_ <= n:
		var ang: float = 2.0 * PI / float(len_)
		var wlenR: float = cos(ang)
		var wlenI: float = -sin(ang)

		var i: int = 0
		while i < n:
			var wr: float = 1.0
			var wi: float = 0.0
			var half: int = len_ >> 1
			var i_end: int = i + half  # 减少重复计算
			for k in half:
				var i1: int = i + k
				var i2: int = i1 + half
				var tr: float = wr * real[i2] - wi * imag[i2]
				var ti: float = wr * imag[i2] + wi * real[i2]
				real[i2] = real[i1] - tr
				imag[i2] = imag[i1] - ti
				real[i1] += tr
				imag[i1] += ti
				var wR: float = wr * wlenR - wi * wlenI
				var wI: float = wr * wlenI + wi * wlenR
				wr = wR
				wi = wI
			i += len_

		len_ <<= 1


# ══════════════════════════════════════════════════════════════
# Mel 尺度转换
# ══════════════════════════════════════════════════════════════

static func _freq_to_mel(f: float) -> float:
	return MEL_A * log(1.0 + f / MEL_B) / log(10.0)

static func _mel_to_freq(m: float) -> float:
	return MEL_B * (pow(10.0, m / MEL_A) - 1.0)


# ══════════════════════════════════════════════════════════════
# 纹理 / LUT / Bin 表
# ══════════════════════════════════════════════════════════════

func _init_texture(w: int, h: int) -> void:
	_pixels = PackedByteArray()
	_pixels.resize(w * h * 4)

	_image = Image.create_empty(w, h, false, Image.FORMAT_RGBA8)
	_image.set_data(w, h, false, Image.FORMAT_RGBA8, _pixels)

	_texture = ImageTexture.create_from_image(_image)

	if targetImage != null:
		targetImage.texture = _texture

	_binForY = PackedInt32Array()
	_binForY.resize(h)
	for y in h:
		_binForY[y] = clampi(roundi(float(y) / float(h) * float(MEL_BINS)), 0, MEL_BINS - 1)

	_appliedGradient = null


func _build_color_lut() -> void:
	var grad: Gradient = colorGradient if colorGradient != null else _get_default_gradient()

	if colorGradient == null and _appliedGradient == grad and _colorLut.size() > 0:
		return

	_colorLut = PackedByteArray()
	_colorLut.resize(256 * 4)
	for i in 256:
		var c: Color = grad.sample(float(i) / 255.0)
		var off: int = i * 4
		_colorLut[off]     = int(c.r * 255.0)
		_colorLut[off + 1] = int(c.g * 255.0)
		_colorLut[off + 2] = int(c.b * 255.0)
		_colorLut[off + 3] = int(c.a * 255.0)

	_appliedGradient = grad


# ══════════════════════════════════════════════════════════════
# 默认渐变
# ══════════════════════════════════════════════════════════════

static var _defaultGradient: Gradient

static func _get_default_gradient() -> Gradient:
	if _defaultGradient != null:
		return _defaultGradient

	_defaultGradient = Gradient.new()
	_defaultGradient.colors = [
		Color(0.4, 0.1, 0.6, 0.0),
		Color(0.4, 0.1, 0.6, 0.5),
		Color(0.0, 0.6, 0.8, 0.75),
		Color(0.2, 0.9, 0.5, 0.85),
		Color(1.0, 0.85, 0.3, 0.9),
		Color(1.0, 1.0, 1.0, 1.0),
	]
	_defaultGradient.offsets = [0.0, 0.15, 0.35, 0.6, 0.8, 1.0]
	return _defaultGradient


# ══════════════════════════════════════════════════════════════
# 生命周期
# ══════════════════════════════════════════════════════════════

func stop() -> void:
	_cancelRequested = true
	if _thread != null and _thread.is_alive():
		_thread.wait_to_finish()
	_thread = null
