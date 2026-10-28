"""Ollama 的共同请求代码；调用方负责决定模型、提示词和等待预算。"""
import json
import math
from urllib.request import Request, urlopen

from friston3.config import OLLAMA_KEEP_ALIVE


def estimate_tokens(text):
    """粗估中英文文本长度；真实 token 数仍以 Ollama 的结果为准。"""
    wide = sum(ord(char) >= 0x2e80 for char in text)
    return math.ceil(wide * 1.5 + (len(text) - wide) / 3)


def performance_stats(result, wall_s):
    """把 Ollama 的纳秒计时转换成各调用方共用的秒数和计数。"""
    def seconds(key):
        value = result.get(key)
        return round(value / 1e9, 3) if isinstance(value, (int, float)) else None

    decode_s = seconds("eval_duration")
    output_tokens = result.get("eval_count")
    return {
        "wall_s": round(wall_s, 3),
        "load_s": seconds("load_duration"),
        "prompt_s": seconds("prompt_eval_duration"),
        "decode_s": decode_s,
        "prompt_tokens": result.get("prompt_eval_count"),
        "cached_prompt_tokens": result.get("prompt_eval_cached_count"),
        "output_tokens": output_tokens,
        "decode_tokens_per_s": round(output_tokens / decode_s, 2)
            if isinstance(output_tokens, int) and decode_s else None,
    }


def chat(messages, *, model, url, timeout, options, schema=None,
         diagnostics=None, reject_truncated=False):
    """只发送一次完整文本请求，返回原始结果，不负责重试或生成策略。

    diagnostics 是可选的本轮诊断记录。即使请求失败，也保留实际走到的
    阶段和模型原文，便于区分连接失败、响应读取失败与输出格式错误。
    """
    trace = diagnostics if diagnostics is not None else {}
    trace.update(request_attempted=False, response_headers_received=False,
                 response_received=False)
    stage = "build_request"
    try:
        payload = {"model": model, "messages": messages, "stream": False,
                   "think": False, "keep_alive": OLLAMA_KEEP_ALIVE,
                   "options": options}
        if schema is not None:
            payload["format"] = schema
        request = Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        stage = "request"
        trace["request_attempted"] = True
        with urlopen(request, timeout=timeout) as response:
            trace["response_headers_received"] = True
            stage = "read_response"
            result = json.load(response)
        trace["response_received"] = True
        stage = "validate_response"
        if isinstance(result, dict):
            trace["done_reason"] = result.get("done_reason")
            message = result.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                trace["raw_output"] = message["content"]
        if (not isinstance(result, dict) or result.get("error") or result.get("done") is not True
                or not isinstance(result.get("message"), dict)
                or not isinstance(result["message"].get("content"), str)):
            raise ValueError("模型没有返回完整的文本响应")
        if reject_truncated and result.get("done_reason") in (
                "length", "max_tokens", "max_output_tokens", "token_limit"):
            raise ValueError("模型输出被截断")
        return result
    except (OSError, ValueError, TypeError):
        trace["failure_stage"] = stage
        raise
