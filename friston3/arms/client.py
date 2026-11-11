"""机械臂 HTTP 客户端：把 Python 方法调用转换为本机服务请求。

调用方是 arms.__main__ 与 ArmIntegration；对端是 service._Handler。
这里只依赖标准库和配置模块，因此 NPC 环境无需安装机械臂训练依赖。
串口、控制循环和模型均由服务端拥有，客户端不复制这些实现。
"""
# 类型标注用于说明参数/返回值；此语句让标注延迟求值，不会自动检查或转换传入的数据。
from __future__ import annotations

import http.client
import json
import math
import re
import socket
import uuid
from pathlib import Path
# Any 表示这里允许不同类型的值，例如同一个结果字典可同时保存布尔值、文字和列表。
from typing import Any

# 开头的点表示从“本包 friston3.arms”导入；load_config 返回解析后的配置字典。
from .config import load_config


# 大写名字约定为模块常量；单位是字节，不是中文字符数。下方 _request 使用它们。
_MAX_RESPONSE_BYTES = 256 * 1024
_MAX_REQUEST_BYTES = 16 * 1024
# frozenset 是不可修改的集合；op in _OPERATIONS 用来检查操作名是否属于协议。
_OPERATIONS = frozenset(("arm", "stop", "auto", "play", "record", "list", "task", "shutdown"))
_JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")


def _reject_constant(value: str) -> None:
    # _request 把本函数交给 json.loads；若 JSON 出现 NaN/Infinity，解析器会调用它。
    # value 是解析到的文字；raise 抛异常中断解析，所以本函数不产生正常返回值。
    raise ValueError("non-finite JSON value: " + value)


class ArmClient:
    """One connection per request, without proxies, redirects, or retries.

    An accepted command is not a completed physical action. Call ``job`` to
    inspect its terminal result. A transport failure after POST is ambiguous;
    the client deliberately never resends it with a new request identifier.
    """

    def __init__(self, config_path: str | Path | None = None, *, timeout: float = 1.0):
        # ArmClient(...) 创建对象时自动调用 __init__；self 就是这个新客户端对象。
        # config_path 可为 str、Path 或 None；None 表示交给 load_config 选择默认配置。
        # 单独的 * 规定 timeout 只能按名字传，例如 ArmClient(path, timeout=1.0)。
        # CLI 与 ArmIntegration 创建本对象；这里只保存 HTTP 配置，不连接机械臂串口。
        config = load_config(config_path)
        # config 是 dict；.get("host") 读取键，缺失时得到 None，不会直接抛 KeyError。
        if config.get("host") != "127.0.0.1":
            raise ValueError("机械臂客户端仅允许连接 127.0.0.1。")
        port = config.get("port")
        # type(x) is int 是精确类型判断；bool 在 Python 中是 int 的子类，这里不接受它。
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("机械臂服务端口必须是 1–65535 的整数。")
        token = config.get("token")
        # isinstance 判断对象类型；any(...) 遇到任一不符合字符范围的字符就为 True。
        if (not isinstance(token, str) or not token
                or any(ord(char) < 33 or ord(char) > 126 for char in token)):
            raise ValueError("机械臂配置中的本机通信凭据无效，请检查配置。")
        if (type(timeout) not in (int, float) or not math.isfinite(timeout)
                or not 0.1 <= timeout <= 2.0):
            raise ValueError("客户端 timeout 必须在 0.1–2.0 秒之间。")
        # self.xxx 是保存在对象上的属性，后续每次 _request 都能读取；局部 config 不保留。
        self.port = port
        self.timeout = float(timeout)
        self.mode = config.get("mode")
        self._token = token

    def status(self) -> dict[str, Any]:
        # CLI、Integration 调用此方法查询状态。返回 str 键、任意类型值组成的 dict。
        # self._request 中的点号表示调用“这个客户端对象”的方法；两个字符串按位置传参。
        return self._request("GET", "/status")

    def is_listening(self, *, timeout: float = 3.0) -> bool:
        """Distinguish a closed port from HTTP timeout without sending a command.

        Windows can take just over two seconds to report a refused loopback
        connection. The short HTTP timeout is deliberately independent.
        """
        # 集成层启动前、CLI 等待退出时调用：只建立 TCP 连接，返回端口是否有人监听。
        # 能连上只说明端口存在，不能说明对方是已完成初始化的 ArmService。
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("端口探测 timeout 必须是有限正数")
        try:
            # (地址, 端口) 是二元素 tuple；timeout=... 是按名字传入的关键字参数。
            connection = socket.create_connection(("127.0.0.1", self.port), timeout=float(timeout))
        except OSError as error:
            # except 接住 try 中的网络异常，as error 将异常对象保存为局部变量。
            if isinstance(error, ConnectionRefusedError) or error.errno in (61, 111, 10061):
                return False
            raise RuntimeError("机械臂服务端口监听状态未确认，未将超时当作服务退出。") from error
        connection.close()
        # 探测用的 socket 到此释放；后续 HTTP 请求会自己创建连接。
        return True

    def job(self, job_id: str) -> dict[str, Any]:
        # CLI 的 _wait_for_job 反复调用：输入任务编号，返回该任务当前状态字典。
        # fullmatch 要求整个编号都符合正则格式，不是只找到其中一段匹配文字。
        if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
            raise ValueError("无效的机械臂任务编号。")
        return self._request("GET", "/jobs/" + job_id)

    def command(self, op: str, **kwargs: Any) -> dict[str, Any]:
        # Integration/CLI 调用：op 是操作名；**kwargs 把额外关键字参数收集成字典。
        # 例如 command("play", name="wave") 中，op="play"，kwargs={"name": "wave"}。
        if not isinstance(op, str) or op not in _OPERATIONS:
            raise ValueError("不支持的机械臂操作。")
        if "request_id" in kwargs:
            raise ValueError("request_id 由客户端生成，不能作为操作参数传入。")
        # request_id 标识一次提交；服务端可识别重复 ID。它与持续查询用的 job_id 不同。
        # 本客户端不会自动重发：回包丢失不等于命令没执行，重新提交可能重复动作。
        # uuid4() 生成随机请求标识，str(...) 转为 JSON 能表示的字符串。
        # 字典里的 **kwargs 是“展开键值对”，和函数定义中“收集参数”的 ** 用法相反。
        payload = {"op": op, "request_id": str(uuid.uuid4()), **kwargs}
        # payload 是第三个位置参数；_request 会将这个 Python 字典编码后发送。
        return self._request("POST", "/command", payload)

    def _request(self, method: str, path: str,
                 payload: dict[str, Any] | None = None) -> dict[str, Any]:
        # status/job/command 的共用实现：输入 HTTP 方法、地址路径、可选字典，返回结果字典。
        # method 是 GET/POST；path 是 /status 等路径；payload=None 表示本次没有请求正文。
        body = None
        if payload is not None:
            # dumps: dict -> str；encode: str -> bytes。网络发送字节，不直接发送 Python 对象。
            # ensure_ascii=False 保留中文；allow_nan=False 不允许写出非标准 JSON 数值。
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            if len(body) > _MAX_REQUEST_BYTES:
                raise ValueError("机械臂请求过大。")
        # headers 也是字典，但它承载 HTTP 头；令牌来自配置，不属于机器人任务参数。
        headers = {"Authorization": "Bearer " + self._token,
                   "Accept": "application/json", "Connection": "close"}
        if body is not None:
            # dict[键] = 值 在这里新增一项，告知服务端正文采用 UTF-8 编码的 JSON。
            headers["Content-Type"] = "application/json; charset=utf-8"
        # HTTPConnection 是标准库类；调用类创建 connection 对象，真正请求在下一行发送。
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
        try:
            # method/path 按位置传，body/headers 按名字传；对端是 service.py 的 _Handler。
            connection.request(method, path, body=body, headers=headers)
            # getresponse 返回 HTTPResponse 对象；其中 .status 是 HTTP 数字状态码。
            response = connection.getresponse()
            if 300 <= response.status < 400:
                raise RuntimeError("机械臂服务返回重定向；客户端已拒绝跟随。")
            # getheader 返回文本或 None；HTTP 头中的长度还不是 Python int。
            length = response.getheader("Content-Length")
            if length is not None:
                try:
                    declared_length = int(length)
                except ValueError as error:
                    # raise ... from error 把底层异常保留为原因，便于上层定位错误来源。
                    raise RuntimeError("机械臂服务返回无效的响应长度。") from error
                if declared_length < 0 or declared_length > _MAX_RESPONSE_BYTES:
                    raise RuntimeError("机械臂服务响应超过允许大小。")
            # raw 是响应正文 bytes；多读 1 字节用于识别实际正文是否超过已有协议上限。
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_RESPONSE_BYTES:
                raise RuntimeError("机械臂服务响应超过允许大小。")
            try:
                # 与发送方向相反：bytes.decode -> str，json.loads -> Python dict/list/标量。
                # parse_constant 传函数本身（没有括号），等解析器遇到特殊常量时再调用。
                result = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
            except (UnicodeDecodeError, ValueError) as error:
                raise RuntimeError("机械臂服务未返回有效 JSON；执行状态尚未确认。") from error
            # JSON 不保证一定是 dict，也不保证字段类型正确，所以先核对服务响应的约定。
            if (not isinstance(result, dict) or type(result.get("success")) is not bool
                    or not isinstance(result.get("status"), str)
                    or not isinstance(result.get("message"), str)):
                raise RuntimeError("机械臂服务响应字段不完整；执行状态尚未确认。")
            # HTTP 是否成功与业务 success 字段应一致；2xx 的链式比较等价于两个比较用 and。
            if not 200 <= response.status < 300 and result["success"]:
                raise RuntimeError("机械臂服务的 HTTP 状态与结果冲突；执行状态尚未确认。")
            # Preserve error dictionaries (including a precise HTTP 4xx reason)
            # so callers can distinguish unavailable, rejected and busy states.
            # return 将原字典交回 status/job/command，再传到 CLI 或 Integration，不改任务状态。
            return result
        except (ConnectionRefusedError, ConnectionResetError, ConnectionAbortedError,
                socket.timeout, OSError, http.client.HTTPException) as error:
            # 网络层异常在此转为带业务上下文的 RuntimeError；并没有自动补发请求。
            if method == "POST":
                detail = ("未收到机械臂命令的有效确认，命令可能已经被接收；不会自动重发。"
                          "请先查询 status，必要时在独立终端执行 stop。")
            else:
                detail = ("机械臂本机服务未连接或响应超时。"
                          "请在 lerobot 环境启动 python -m friston3.arms serve，"
                          "并确认两个终端使用同一配置。")
            raise RuntimeError(detail) from error
        finally:
            # finally 无论成功 return 还是中途抛异常都执行，避免连接一直占用系统资源。
            connection.close()
