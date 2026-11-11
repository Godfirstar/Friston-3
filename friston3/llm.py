"""Ollama 的共同请求代码；调用方负责决定模型、提示词和等待预算。"""
import json
import math
import threading
import time
from urllib.request import Request, urlopen

from friston3.config import OLLAMA_KEEP_ALIVE


_POLICY_MEMORY_LOCK = threading.Lock()
# Lock是标准库threading的互斥锁实例；保护下面同一进程内多个调用共享的缓存字典。
# resident表示机械臂模型是否驻留，不等于机械臂当前正在运动；check_after是下次允许查询的时刻。
_policy_memory = {"resident": False, "check_after": 0.0, "generation": 0,
                  "config_path": None, "error": ""}


def reserve_policy_memory(config_path=None):
    """Invalidate the service cache when this process submits a task."""
    # ArmIntegration提交策略任务时调用；先标记需要给策略留资源，并让下次状态查询立即发生。
    # with进入时获取锁，离开时自动释放；dict.update改原字典，函数没有显式return故返回None。
    with _POLICY_MEMORY_LOCK:
        _policy_memory.update(resident=True, check_after=0.0, config_path=config_path,
                              generation=_policy_memory["generation"] + 1, error="")


def _policy_memory_status():
    """Adapt this request to actual policy residency, retaining it on timeout."""
    # chat及规划器查询驻留状态时使用；返回(resident布尔值,探测错误文本)二元组。
    # monotonic是只增的计时钟，用来比较时间间隔，不是可以显示日期的日历时间。
    now = time.monotonic()
    with _POLICY_MEMORY_LOCK:
        if now < _policy_memory["check_after"]:
            return _policy_memory["resident"], _policy_memory["error"]
        generation = _policy_memory["generation"]
        # 取一份本次查询的版本号和参数，下面在锁外做HTTP，避免慢网络长时间占住缓存锁。
        config_path = _policy_memory["config_path"]
        resident = _policy_memory["resident"]
    error = ""
    client = None
    try:
        from friston3.arms.client import ArmClient
        client = ArmClient(config_path, timeout=0.1)
        # ArmClient在arms/client.py定义；构造客户端对象不打开机械臂串口。
        snapshot = client.status()
        # status才发送HTTP查询；snapshot是服务状态字典，只有真正的布尔True算驻留。
        resident = snapshot.get("policy_resident") is True
    except FileNotFoundError:
        # 配置文件不存在时保留此前resident值；pass表示该分支不执行额外操作。
        pass
    except (OSError, ValueError, RuntimeError) as failure:
        cause = failure
        while cause is not None:
            # 异常可以由raise ... from ...串联；__cause__追踪包装前的底层异常。
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
        # 查询期间可能又提交了新任务；版本号不同就不让旧查询覆盖新标记。
        if generation == _policy_memory["generation"]:
            _policy_memory.update(resident=resident, check_after=time.monotonic() + 1.0, error=error)
        return _policy_memory["resident"], _policy_memory["error"]


def estimate_tokens(text):
    """粗估中英文文本长度；真实 token 数仍以 Ollama 的结果为准。"""
    # PromptBuilder等调用方传入str，返回整数估计；不执行分词器，所以不能当精确token数。
    # ord取字符的Unicode码点，生成器逐字符比较；sum把True按1、False按0累加。
    wide = sum(ord(char) >= 0x2e80 for char in text)
    # ceil向上取整；这里是按宽字符/其他字符分别估计成本的启发式公式。
    return math.ceil(wide * 1.5 + (len(text) - wide) / 3)


def performance_stats(result, wall_s):
    """把 Ollama 的纳秒计时转换成各调用方共用的秒数和计数。"""
    def seconds(key):
        # 内部函数可读取外层result；key是计时字段名，.get缺字段返回None。
        value = result.get(key)
        # 1e9即10的9次方，纳秒除它变秒；round(...,3)仅保留展示精度，不改变服务原始结果。
        return round(value / 1e9, 3) if isinstance(value, (int, float)) else None

    decode_s = seconds("eval_duration")
    # 调用方是Planner/NPC等；result为Ollama完整响应，wall_s为客户端实际测到的总耗时。
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
    # 被Planner、NPC等调用；messages为消息字典列表，model/url/options由调用者显式提供。
    # *后的项必须按名字传入。返回完整Ollama响应dict，调用者再取message.content。
    # 若提供diagnostics，就直接更新调用方的原字典；不提供时创建本次独有的新dict。
    trace = diagnostics if diagnostics is not None else {}
    trace.update(request_attempted=False, response_headers_received=False,
                 response_received=False)
    stage = "build_request"
    try:
        resident, probe_error = _policy_memory_status()
        # 解包二元组。dict(options)做浅拷贝，避免下方改选项时污染config.py的公共字典。
        request_options = dict(options)
        if resident:
            request_options["num_gpu"] = 0
        effective_timeout = timeout
        # effective_timeout是这一次真正交给HTTP库的等待值；与响应实测耗时不同。
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
        # payload只是Python字典；stream=False要求完整响应，此刻尚未向Ollama发送本次生成请求。
        # 前面的状态探测可能已经访问机械臂服务；那是另一种HTTP请求。
        if schema is not None:
            # schema是JSON Schema描述数据，Ollama用它约束输出格式；它不是Python验证函数。
            payload["format"] = schema
        # Request/urlopen来自标准库urllib.request；先dumps将dict转JSON字符串，
        # encode('utf-8')再转网络发送的bytes。Request只创建请求描述，urlopen才实际发送。
        request = Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        stage = "request"
        trace["request_attempted"] = True
        with urlopen(request, timeout=effective_timeout) as response:
            # response是可读取的响应对象；with退出时关闭连接资源，json.load从流读取并解析JSON。
            trace["response_headers_received"] = True
            stage = "read_response"
            result = json.load(response)
        trace["response_received"] = True
        stage = "validate_response"
        if isinstance(result, dict):
            # 先尽量记录收到的内容，再检查完整性；失败时diagnostics仍能保留实际模型原文。
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
        # 此处只确认完整文本响应，尚未验证模型选择的技能/参数是否正确；那是Planner等的职责。
        return result
    except (OSError, ValueError, TypeError):
        trace["failure_stage"] = stage
        # 单独raise重新抛出当前异常，调用方决定如何呈现；本函数不会偷偷重试第二次模型请求。
        raise
