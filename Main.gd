extends Node

# 使用 C# 版 AudioMelSpectrogram
# 注意：必须先 Build → Build Solution（Ctrl+B）C# 类型才可用
var AudioMelSpectrogram=preload("res://AudioMelSpectrogram.cs")
var AudioDecoder = preload("res://AudioCodec/AudioDecoder.cs")
var spec := AudioMelSpectrogram.new()

@export var Render_rect: TextureRect
@export var Player: AudioStreamPlayer
@export var GRD: Gradient
@export var res: float = 0.8
@export var Debug_label: Label  # (可选) 显示当前音频信息

var current_time := 0.0

func _ready() -> void:
	spec.targetImage = Render_rect
	spec.resolutionScale = res
	spec.colorGradient = GRD
	Player.play(0)
	_update_debug_label()

func _process(_delta: float) -> void:
	current_time = Player.get_playback_position()
	spec.UpdateView(Player, current_time, current_time + 5.0, false)

func _input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and event.keycode == KEY_L:
		# 按 L 键打开文件选择对话框加载音频 (需要 FileDialog 节点)
		_load_audio_dialog()

func _load_audio_dialog() -> void:
	# 简单方法：让玩家将文件拖入窗口，或使用 FileDialog
	# 这里演示如何通过代码加载外部音频文件
	print("提示：将音频文件(WAV/MP3/OGG)拖入窗口来播放")
	print("或调用: AudioDecoder.DecodeFile(\"路径.mp3\") 解码到 PCM")

func load_external_audio(file_path: String) -> void:
	# 使用 C# AudioDecoder 解码外部音频文件
	# 解码为 44100Hz 单声道后喂给 Mel 频谱
	var result = AudioDecoder.DecodeFile(file_path, 44100, 1)
	if result != null and result.Samples.size() > 0:
		spec.LoadRawPcm(result.Samples, result.SampleRate)
		print("已加载外部音频到频谱: ", file_path)
		_update_debug_label()

func _update_debug_label() -> void:
	if Debug_label == null:
		return
	var stream_name = "未知"
	if Player.stream != null:
		stream_name = Player.stream.resource_path.get_file()
	Debug_label.text = "🎵 " + stream_name

func _exit_tree() -> void:
	spec.Stop()
