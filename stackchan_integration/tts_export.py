"""Natural phrase segmentation shared by the resident TTS producer."""

from __future__ import annotations
# 当前文件是“分句与合成调用”的小工具层；名字带export，但这里不负责导出神经网络权重。
# tts_daemon导入这两个函数；它们不会自己启动进程或连接设备。

import re

from .local_tts import OfflineCharacterVoice


def split_text(text: str, limit: int = 32, *, first_limit: int | None = None) -> list[str]:
    """Yield a short first phrase and natural later phrases without losing text.

    A complete short sentence need not fill half of a large chunk budget.
    English words/contractions, decimal numbers and dotted initials stay whole.
    """
    # 【调用方】tts_daemon.serve；输入整段文本，返回不丢正文的短句list，不执行模型推理。
    # *后的first_limit须按名传；int|None是类型联合，None表示首段也用普通limit。
    if limit < 1 or (first_limit is not None and first_limit < 1):
        raise ValueError("Segment limit must be positive")
    text = re.sub(r"\s+", " ", text).strip()
    # 标准库re做正则替换；多个空白合成一个空格，保留英文词之间的分隔。
    if not text:
        return []
    # Latin words (including don't / Friston-3) stay atomic; CJK characters can
    # form short phrases. Whitespace is retained between neighboring words.
    tokens = re.findall(
        r"\d+(?:[.:]\d+)+|(?:[A-Za-z]\.){2,}|[A-Za-z0-9]+(?:['’_-][A-Za-z0-9]+)*|\s+|.",
        text,
    )
    # 这里tokens只是正则匹配出的文字片段，不是神经网络词表ID。
    # 数字时间/小数、英文缩写和带连字符单词先整体匹配，其余字符按后面的规则接收。
    chunks, current = [], ""
    # chunks保存已定稿的短句list；current累积尚未提交的文本str，类型不同。
    for token in tokens:
        budget = first_limit if not chunks and first_limit is not None else limit
        # 条件表达式：还没有首段且指定first_limit时用较短预算，否则用普通预算。
        if len(current) + len(token) > budget and current.strip():
            chunks.append(current.strip())
            # list.append追加一个完整片段；current随后清空，开始累计下一段。
            current = ""
        # Extremely long identifiers may exceed the segment budget; keep words
        # intact up to the model's cap, and bound pathological input above it.
        while len(token) > 140:
            chunks.append(token[:140])
            # 字符串切片左闭右开；这里仅拆异常长标识符，避免超过单次模型文本容量。
            token = token[140:]
        current += token
        # +=拼接下一段文字；保留被正则匹配到的空白，使英文词不会粘连。
        if token in "。！？!?；;，,." and len(current.strip()) >= min(6, budget):
            chunks.append(current.strip())
            current = ""
    if current.strip():
        chunks.append(current.strip())
    return [p for p in chunks if p]
    # 列表推导式过滤空字符串，结果是纯文本短句，不含PCM或音频时间信息。


def synthesize_part(voice: OfflineCharacterVoice, text: str) -> list[bytes]:
    # 【调用方】daemon每个part；voice是已驻留引擎，text是一段短句；返回PCM片段list。
    # Long clips are split from the generated waveform near a quiet point.
    # Each phrase is synthesized exactly once; no expensive recursive reruns.
    return voice.pcm16_chunks(text)
    # 方法定义在local_tts；它做一次实际合成，再必要时按波形安静点切割，不递归重跑TTS。

