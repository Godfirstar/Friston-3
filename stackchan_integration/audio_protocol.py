"""栈酱音频协议的固定格式，以及两个入口共用的参数检查。"""

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH
MAX_CLIP_SECONDS = 12
MAX_PCM_BYTES = BYTES_PER_SECOND * MAX_CLIP_SECONDS


def pcm_seconds(byte_count: int) -> float:
    return byte_count / BYTES_PER_SECOND


def validate_pcm(pcm: bytes) -> None:
    if not isinstance(pcm, bytes):
        raise TypeError("pcm must be bytes")
    if not 2 <= len(pcm) <= MAX_PCM_BYTES or len(pcm) % SAMPLE_WIDTH:
        raise ValueError("PCM must be even-sized, 2..384000 bytes, PCM16LE/16kHz mono")


def validate_microphone(mode, wait_ms, max_ms, silence_ms, threshold) -> str:
    """先验证等待上限，再入队或发送命令；返回统一的小写模式。"""
    mode = str(mode).lower()
    if mode not in ("touch", "vad"):
        raise ValueError("Microphone mode must be 'touch' or 'vad'")
    for name, value, low, high in (
        ("wait_ms", wait_ms, 100, 10000), ("max_ms", max_ms, 1000, 12000),
        ("silence_ms", silence_ms, 200, 2000), ("threshold", threshold, 30, 10000),
    ):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    return mode
