extends Node

# 使用 C# 版 AudioMelSpectrogram
# 注意：必须先 Build → Build Solution（Ctrl+B）C# 类型才可用
var AudioMelSpectrogram=preload("res://AudioMelSpectrogram.cs")
var AudioMelSpectrogramBeat=preload("res://AudioMelSpectrogramBeat.cs")
var AudioDecoder    = preload("res://AudioCodec/AudioDecoder.cs")
var spec := AudioMelSpectrogram.new()
var spec1:=AudioMelSpectrogramGD.new()


@export var Render_rect: TextureRect
@export var Render_rectGD: TextureRect
@export var Player: AudioStreamPlayer
@export var GRD: Gradient
@export var res: float = 0.8
@export var UseGdEditon:bool=false

@export var use_stereo: bool = false:
	set(v):
		use_stereo = v
		if spec: spec.UseStereo = v

var current_time := 0.0


func _ready() -> void:
	spec1.targetImage = Render_rectGD
	spec1.resolutionScale = res
	spec1.colorGradient = GRD
	spec.targetImage = Render_rect
	spec.resolutionScale = res
	spec.colorGradient = GRD
	spec.UseStereo = use_stereo
	Player.play(0)


func _input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed:
		match event.keycode:
			KEY_SPACE:
				Player.stream_paused = !Player.stream_paused
			KEY_L:
				_open_file_dialog()

	if event is InputEventMouseButton:
		match event.button_index:
			MOUSE_BUTTON_WHEEL_UP:
				current_time += 1
				Player.play(clamp(current_time,0,Player.stream.get_length()))
				Player.stream_paused = true
			MOUSE_BUTTON_WHEEL_DOWN:
				current_time -= 1
				Player.play(clamp(current_time,0,Player.stream.get_length()))
				Player.stream_paused = true


func _process(_delta: float) -> void:
	current_time = Player.get_playback_position()
	spec.UpdateView(Player, current_time, current_time + 5.0, false)
	if(UseGdEditon):spec1.update_view(Player, current_time, current_time + 5.0, false)


# ════════════════════════════════════════════════════════════════
#  按 L 打开文件对话框 → 解码外部音频 → 喂给 Mel 频谱
# ════════════════════════════════════════════════════════════════

func _open_file_dialog() -> void:
	var dialog := FileDialog.new()
	Player.stream_paused=true
	dialog.title = "选择音频文件"
	dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	dialog.access = FileDialog.ACCESS_FILESYSTEM
	dialog.add_filter("*.wav,*.mp3,*.ogg,*.flac,*.aac,*.wma", "音频文件")
	dialog.add_filter("*.wav", "WAV 音频")
	dialog.add_filter("*.mp3", "MP3 音频")
	dialog.add_filter("*.ogg", "OGG / Vorbis 音频")
	dialog.add_filter("*.flac", "FLAC 音频")
	dialog.min_size = Vector2i(900, 600)

	dialog.file_selected.connect(_on_file_selected)
	dialog.canceled.connect(dialog.queue_free)
	add_child(dialog)
	dialog.popup_centered()


func _on_file_selected(path: String) -> void:
	print("加载音频: ", path)

	# ── 重置 Mel 频谱（停旧线程 → 新建实例 + 立即显示黑色纹理） ──
	_reset_spec()

	## ── 先试 Godot 原生加载（WAV/MP3/OGG/FLAC） ──
	#var new_stream = load(path)
	#if new_stream != null:
		## Godot 支持 → 直接播放，然后单独解码到频谱
		#Player.stop()
		#Player.stream = new_stream
		#current_time = 0.0
		#Player.play(0)
		#print("已切换播放: ", path.get_file())
#
		#var result = AudioDecoder.LoadToMelSpectrogram(path, spec, 44100,use_stereo)
		#if not result.ok:
			#push_error("频谱解码失败: ", result.message)
	#else:
		# Godot 不支持 → 一次 NAudio 解码同时服务频谱+播放
	print("用 NAudio 解码...")
	var wav_stream = AudioDecoder.DecodeAndFeedSpec(
		path, spec, 44100, use_stereo)
	if wav_stream != null:
		Player.stop()
		Player.stream = wav_stream
		current_time = 0.0
		Player.play(0)
		print("已切换播放（NAudio 解码）: ", path.get_file())
	else:
		push_error("解码失败")


func _reset_spec() -> void:
	spec.Stop()
	spec = AudioMelSpectrogram.new()
	spec.targetImage = Render_rect
	spec.resolutionScale = res
	spec.colorGradient = GRD
	spec.UseStereo = use_stereo
	spec.ResetView()
	print("频谱已重置")


func _exit_tree() -> void:
	spec.Stop()
