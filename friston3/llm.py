"""Ollama 的共同请求代码；调用方负责决定模型、提示词和等待预算。"""
import json
import math
import threading
import time
from urllib.request import Request, urlopen

from friston3.config import OLLAMA_KEEP_ALIVE


_POLICY_MEMORY_LOCK = threading.Lock()
_policy_memory = {"resident": False, "check_after": 0.0, "generation": 0,
                  "config_path": None, "error": ""}


def reserve_policy_memory(config_path=None):
    """Invalidate the service cache when this process submits a task."""
    with _POLICY_MEMORY_LOCK:
        _policy_memory.update(resident=True, check_after=0.0, config_path=config_path,
                              generation=_policy_memory["generation"] + 1, error="")


def _policy_memory_status():
    """Adapt this request to actual policy residency, retaining it on timeout."""
    now = time.monotonic()
    with _POLICY_MEMORY_LOCK:
        if now < _policy_memory["check_after"]:
            return _policy_memory["resident"], _policy_memory["error"]
        generation = _policy_memory["generation"]
        config_path = _policy_memory["config_path"]
        resident = _policy_memory["resident"]
    error = ""
    client = None
    try:
        from friston3.arms.client import ArmClient
        client = ArmClient(config_path, timeout=0.1)
        snapshot = client.status()
        resident = snapshot.get("policy_resident") is True
    except FileNotFoundError:
        pass
    except (OSError, ValueError, RuntimeError) as failure:
        cause = failure
        while cause is not None:
            if isinstance(cause, ConnectionRefusedError):
                resident = False
                break
            cause = cause.__cause__
        if resident and client is not None:
            try:
                if not client.is_listening(timeout=3.0):
                    resident = False
            except (OSError, RuntimeError):
                pass
        error = f"{type(failure).__name__}: {failure}"
    with _POLICY_MEMORY_LOCK:
        if generation == _policy_memory["generation"]:
            _policy_memory.update(resident=resident, check_after=time.monotonic() + 1.0, error=error)
        return _policy_memory["resident"], _policy_memory["error"]


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
         diagnostics=None, reject_truncated=False, policy_cpu_timeout=None):
    """只发送一次完整文本请求，返回原始结果，不负责重试或生成策略。

    diagnostics 是可选的本轮诊断记录。即使请求失败，也保留实际走到的
    阶段和模型原文，便于区分连接失败、响应读取失败与输出格式错误。
    """
    trace = diagnostics if diagnostics is not None else {}
    trace.update(request_attempted=False, response_headers_received=False,
                 response_received=False)
    stage = "build_request"
    try:
        resident, probe_error = _policy_memory_status()
        request_options = dict(options)
        if resident:
            request_options["num_gpu"] = 0
        effective_timeout = timeout
        if policy_cpu_timeout is not None:
            if (type(policy_cpu_timeout) not in (int, float)
                    or not math.isfinite(policy_cpu_timeout) or policy_cpu_timeout <= 0):
                raise ValueError("policy_cpu_timeout must be finite and positive")
            if resident and timeout is not None:
                effective_timeout = max(timeout, policy_cpu_timeout)
        trace.update(policy_resident=resident, requested_timeout_s=timeout,
                     policy_cpu_timeout_s=policy_cpu_timeout,
                     effective_timeout_s=effective_timeout, timeout_s=effective_timeout)
        if probe_error:
            trace["policy_probe_error"] = probe_error
        payload = {"model": model, "messages": messages, "stream": False,
                   "think": False, "keep_alive": 0 if resident else OLLAMA_KEEP_ALIVE,
                   "options": request_options}
        if schema is not None:
            payload["format"] = schema
        request = Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        stage = "request"
        trace["request_attempted"] = True
        with urlopen(request, timeout=effective_timeout) as response:
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
