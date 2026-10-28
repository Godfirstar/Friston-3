"""Small, bounded localhost client; safe to import outside the LeRobot environment."""
from __future__ import annotations

import http.client
import json
import math
import re
import socket
import uuid
from pathlib import Path
from typing import Any

from .config import load_config


_MAX_RESPONSE_BYTES = 256 * 1024
_MAX_REQUEST_BYTES = 16 * 1024
_OPERATIONS = frozenset(("arm", "stop", "auto", "play", "record", "list"))
_JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON value: " + value)


class ArmClient:
    """One connection per request, without proxies, redirects, or retries.

    An accepted command is not a completed physical action. Call ``job`` to
    inspect its terminal result. A transport failure after POST is ambiguous;
    the client deliberately never resends it with a new request identifier.
    """

    def __init__(self, config_path: str | Path | None = None, *, timeout: float = 1.0):
        config = load_config(config_path)
        if config.get("host") != "127.0.0.1":
            raise ValueError("机械臂客户端仅允许连接 127.0.0.1。")
        port = config.get("port")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("机械臂服务端口必须是 1–65535 的整数。")
        token = config.get("token")
        if (not isinstance(token, str) or not token
                or any(ord(char) < 33 or ord(char) > 126 for char in token)):
            raise ValueError("机械臂配置中的本机通信凭据无效，请检查配置。")
        if (type(timeout) not in (int, float) or not math.isfinite(timeout)
                or not 0.1 <= timeout <= 2.0):
            raise ValueError("客户端 timeout 必须在 0.1–2.0 秒之间。")
        self.port = port
        self.timeout = float(timeout)
        self.mode = config.get("mode")
        self._token = token

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/status")

    def job(self, job_id: str) -> dict[str, Any]:
        if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
            raise ValueError("无效的机械臂任务编号。")
        return self._request("GET", "/jobs/" + job_id)

    def command(self, op: str, **kwargs: Any) -> dict[str, Any]:
        if not isinstance(op, str) or op not in _OPERATIONS:
            raise ValueError("不支持的机械臂操作。")
        if "request_id" in kwargs:
            raise ValueError("request_id 由客户端生成，不能作为操作参数传入。")
        payload = {"op": op, "request_id": str(uuid.uuid4()), **kwargs}
        return self._request("POST", "/command", payload)

    def _request(self, method: str, path: str,
                 payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            if len(body) > _MAX_REQUEST_BYTES:
                raise ValueError("机械臂请求过大。")
        headers = {"Authorization": "Bearer " + self._token,
                   "Accept": "application/json", "Connection": "close"}
        if body is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            if 300 <= response.status < 400:
                raise RuntimeError("机械臂服务返回重定向；客户端已拒绝跟随。")
            length = response.getheader("Content-Length")
            if length is not None:
                try:
                    declared_length = int(length)
                except ValueError as error:
                    raise RuntimeError("机械臂服务返回无效的响应长度。") from error
                if declared_length < 0 or declared_length > _MAX_RESPONSE_BYTES:
                    raise RuntimeError("机械臂服务响应超过允许大小。")
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_RESPONSE_BYTES:
                raise RuntimeError("机械臂服务响应超过允许大小。")
            try:
                result = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
            except (UnicodeDecodeError, ValueError) as error:
                raise RuntimeError("机械臂服务未返回有效 JSON；执行状态尚未确认。") from error
            if (not isinstance(result, dict) or type(result.get("success")) is not bool
                    or not isinstance(result.get("status"), str)
                    or not isinstance(result.get("message"), str)):
                raise RuntimeError("机械臂服务响应字段不完整；执行状态尚未确认。")
            if not 200 <= response.status < 300 and result["success"]:
                raise RuntimeError("机械臂服务的 HTTP 状态与结果冲突；执行状态尚未确认。")
            # Preserve error dictionaries (including a precise HTTP 4xx reason)
            # so callers can distinguish unavailable, rejected and busy states.
            return result
        except (ConnectionRefusedError, ConnectionResetError, ConnectionAbortedError,
                socket.timeout, OSError, http.client.HTTPException) as error:
            if method == "POST":
                detail = ("未收到机械臂命令的有效确认，命令可能已经被接收；不会自动重发。"
                          "请先查询 status，必要时在独立终端执行 stop。")
            else:
                detail = ("机械臂本机服务未连接或响应超时。"
                          "请在 lerobot 环境启动 python -m friston3.arms serve，"
                          "并确认两个终端使用同一配置。")
            raise RuntimeError(detail) from error
        finally:
            connection.close()
