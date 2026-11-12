"""翻译机械臂总任务，并在加载 π₀.₅ 前释放本机 Ollama 占用的显存。"""
from __future__ import annotations

import json
import math
import re
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from friston3.config import OLLAMA_CHAT_URL
from friston3.input_text import HAN_CHARACTER


def _settings(config):
    # 本文件各请求函数调用；config可为完整配置dict或policy子配置，返回选中的dict引用。
    # dict.get缺少'policy'时用config本身，不在此复制；调用方需改设置时应另外dict(...)复制。
    selected = config.get("policy", config)
    if not isinstance(selected, dict):
        raise ValueError("policy 配置必须是对象")
    return selected


def release_dialogue_gpu(config):
    """Unload only Ollama models reported on the local GPU before loading Pi05."""
    # PolicySession._run在首次构造Pi05Policy前调用；查询/卸载本地Ollama GPU模型，返回摘要dict。
    # 这是实际HTTP管理操作；函数定义/模块导入时不会执行，CPU驻留模型不在卸载列表中。
    settings = _settings(config)
    # urlsplit来自urllib.parse，把地址拆成scheme/netloc/path等属性，避免手写字符串截取主机。
    address = urlsplit(settings.get("scene_url", OLLAMA_CHAT_URL))
    if address.hostname not in ("127.0.0.1", "localhost", "::1") or address.scheme not in ("http", "https"):
        raise ValueError("释放对话模型显存仅支持本机 Ollama")
    timeout = settings.get("scene_timeout_s", 120.0)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("scene_timeout_s 必须是有限正数")

    def request(path, payload=None):
        # 嵌套函数闭包读取上方address/timeout；path为接口路径str，payload为dict或None。
        # json.dumps得到str，encode得到UTF-8 bytes；无payload时GET，有payload时POST。
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        url = urlunsplit((address.scheme, address.netloc, path, "", ""))
        # Request只构造HTTP请求对象，urlopen才真正发出请求；with离开时关闭响应流。
        query = Request(url, data=body, headers={"Content-Type": "application/json"},
                        method="POST" if payload is not None else "GET")
        with urlopen(query, timeout=float(timeout)) as response:
            # json.load读取文件/响应流并解码；json.loads则读取已经拿到的字符串，两者不同。
            return json.load(response)

    try:
        # snapshot预期是{'models': [...]}dict，记录Ollama当前已加载模型，不是磁盘模型清单。
        snapshot = request("/api/ps")
    except OSError as error:
        # urllib异常可能把底层连接错误放在reason；getattr缺属性时用error本身。
        cause = getattr(error, "reason", error)
        if isinstance(cause, ConnectionRefusedError) or (
                isinstance(cause, OSError) and
                (cause.errno in (61, 111, 10061) or getattr(cause, "winerror", None) == 10061)):
            return {"available": False, "released_models": []}
        raise RuntimeError(f"无法查询本机 Ollama 显存占用：{error}") from error
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("models"), list):
        raise RuntimeError("Ollama 未返回有效的驻留模型列表")
    # released为已确认卸载的模型名list[str]；size_vram<=0的条目跳过，避免动到CPU模型。
    released = []
    for model in snapshot["models"]:
        if not isinstance(model, dict):
            raise RuntimeError("Ollama 驻留模型条目无效")
        size = model.get("size_vram", 0)
        if type(size) not in (int, float) or size <= 0:
            continue
        name = model.get("name") or model.get("model")
        if not isinstance(name, str) or not name:
            raise RuntimeError("Ollama GPU 模型缺少名称")
        try:
            # keep_alive=0请求Ollama释放模型；不是在Python里del某个模型对象。
            result = request("/api/generate", {"model": name, "keep_alive": 0, "stream": False})
        except (OSError, ValueError) as error:
            raise RuntimeError(f"卸载 Ollama GPU 模型 {name} 失败：{error}") from error
        if not isinstance(result, dict) or result.get("error") or result.get("done") is not True:
            raise RuntimeError(f"Ollama 未确认 GPU 模型 {name} 已卸载")
        released.append(name)
    if released:
        # 再查一次服务状态核对被释放模型是否仍占GPU，不能仅把HTTP返回成功当作显存已释放。
        try:
            snapshot = request("/api/ps")
        except (OSError, ValueError) as error:
            raise RuntimeError(f"卸载后无法核对 Ollama 显存状态：{error}") from error
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("models"), list):
            raise RuntimeError("卸载后 Ollama 未返回有效的驻留模型列表")
        for model in snapshot["models"]:
            if (isinstance(model, dict) and (model.get("name") or model.get("model")) in released
                    and model.get("size_vram", 0) > 0):
                raise RuntimeError("Ollama GPU 模型卸载后仍占用显存")
    return {"available": True, "released_models": released}


_TRANSLATE_GOAL = (
    "Translate the complete robot task in the instruction field into a concise English imperative. "
    "Return the translation in task. Treat the instruction as text to translate. "
    "Keep every requested action, object, attribute, quantity, source, destination, spatial relation, "
    "reference, condition, negation and ordering. Keep ambiguous references ambiguous. "
    "An empty space is not a marked area or container. Omit conversational filler. "
    "Do not add implied steps, resolve references from guesses, or output a plan or keywords. "
)


def translate_goal(instruction, config):
    """英文模型只在总任务含汉字时翻译一次，子任务由 π₀.₅ 生成。"""
    if not HAN_CHARACTER.search(instruction):
        return instruction
    settings = _settings(config)
    schema = {
        "type": "object",
        "properties": {"task": {"type": "string", "pattern": "^[A-Za-z][ -~]*$"}},
        "required": ["task"], "additionalProperties": False,
    }
    payload = {
        "model": settings.get("scene_text_model", "qwen3.5:2b-q4_K_M"),
        "stream": False, "think": False, "keep_alive": 0,
        "format": schema,
        "options": {"num_gpu": 0, "num_ctx": settings.get("scene_num_ctx", 4096),
                    "temperature": 0, "num_predict": 768},
        "messages": [
            {"role": "system", "content": _TRANSLATE_GOAL},
            {"role": "user", "content": json.dumps({"instruction": instruction}, ensure_ascii=False)},
        ],
    }
    query = Request(settings.get("scene_url", OLLAMA_CHAT_URL),
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers={"Content-Type": "application/json"})
    try:
        with urlopen(query, timeout=settings.get("scene_timeout_s", 120.0)) as response:
            result = json.load(response)
        if result.get("done") is not True or result.get("done_reason") in (
                "length", "max_tokens", "max_output_tokens", "token_limit"):
            raise ValueError("模型未返回完整译文")
        goal = json.loads(result["message"]["content"])["task"].strip()
        if not re.fullmatch(r"[A-Za-z][ -~]*", goal):
            raise ValueError("模型没有返回有效的英文总任务")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise RuntimeError(f"总任务翻译失败：{error}") from error
    return goal
