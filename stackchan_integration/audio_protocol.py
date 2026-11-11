"""栈酱音频协议的固定格式，以及两个入口共用的参数检查。"""

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
# 采样率：每秒16000个采样；采样宽度：每个16bit=2字节。固定为单声道 PCM16LE。
BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH
# 原始PCM每秒32000字节；压缩后ADPCM传输长度不同，不能用压缩长度计算播放时长。
MAX_CLIP_SECONDS = 12
MAX_PCM_BYTES = BYTES_PER_SECOND * MAX_CLIP_SECONDS


def pcm_seconds(byte_count: int) -> float:
    # FaceEvents/Bridge 给定原始PCM字节数，返回音频秒数，用于显示与计算等待预算。
    return byte_count / BYTES_PER_SECOND


def validate_pcm(pcm: bytes) -> None:
    # 播放入口共用检查。bytes 是二进制容器；不是文件路径、WAV头或文本 str。
    if not isinstance(pcm, bytes):
        raise TypeError("pcm must be bytes")
    if not 2 <= len(pcm) <= MAX_PCM_BYTES or len(pcm) % SAMPLE_WIDTH:
        # len(bytes) 得到字节数；% 是取余，余数非零说明最后一个16bit采样不完整。
        raise ValueError("PCM must be even-sized, 2..384000 bytes, PCM16LE/16kHz mono")


def validate_microphone(mode, wait_ms, max_ms, silence_ms, threshold) -> str:
    """先验证等待上限，再入队或发送命令；返回统一的小写模式。"""
    mode = str(mode).lower()
    # FaceEvents.record_pcm 入队前和 Bridge.record_pcm 发命令前都会调用。
    # mode 选择 touch/vad；三个 *_ms 是毫秒，threshold 是固件判定声音活动的数值参数。
    if mode not in ("touch", "vad"):
        raise ValueError("Microphone mode must be 'touch' or 'vad'")
    for name, value, low, high in (
        ("wait_ms", wait_ms, 100, 10000), ("max_ms", max_ms, 1000, 12000),
        ("silence_ms", silence_ms, 200, 2000), ("threshold", threshold, 30, 10000),
    ):
        # 每轮把一个四元组拆成名称、输入值、下界、上界；这一组已有协议参数约定不在此改动。
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    return mode
    # 成功返回标准化模式字符串；检查失败 raise 异常，由调用方决定怎样提示用户。
