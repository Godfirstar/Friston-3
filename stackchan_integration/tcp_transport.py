"""Bounded byte-stream transport for StackChan's offline Wi-Fi access point.

This adapter intentionally exposes only the serial methods used by the bridge.
It neither reconnects nor retries a command: a failed connection ends the current
transaction so a gesture or audio clip cannot be replayed after reconnecting.
"""

from __future__ import annotations

import errno
import ipaddress
import math
import select
import socket
import time
from typing import Callable, Optional


class TcpByteStream:
    # 被 StackChanBridge._connect_endpoint 创建；模仿它用到的 pyserial 方法，
    # 使上层协议对 TCP/USB 共用 read/write/is_open/close，不另写一套相机或音频协议。
    def __init__(self, host: str, port: int, *, timeout: float = 0.15,
                 write_timeout: float = 1.0, connect_timeout: float = 3.0,
                 check_active: Optional[Callable[[], None]] = None):
        # self 是此连接对象；host/port 来自 Bridge 配置；* 后参数只能按名字传。
        # check_active 是回调函数（不接受参数、无业务返回值），由 Bridge 用来检查取消/总截止时间。
        # Numeric addresses avoid a blocking DNS lookup on the offline network.
        try:
            address = ipaddress.ip_address(str(host).strip())
        except ValueError as exc:
            raise ValueError("StackChan TCP host must be a numeric IP address, e.g. 192.168.4.1") from exc
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("StackChan TCP port must be an integer in 1..65535")
        self.timeout = float(timeout)
        self.write_timeout = float(write_timeout)
        connect_timeout = float(connect_timeout)
        if not all(math.isfinite(value) and value > 0
                   for value in (self.timeout, self.write_timeout, connect_timeout)):
            raise ValueError("TCP polling, write and connect timeouts must be finite and positive")
        self._check_active = check_active or (lambda: None)
        # 未传回调就用空操作函数；这里只保存函数，后面的 () 才真正调用。
        self._socket: Optional[socket.socket] = None
        self.endpoint = f"[{address}]:{port}" if address.version == 6 else f"{address}:{port}"
        family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
        peer = (str(address), port, 0, 0) if address.version == 6 else (str(address), port)
        try:
            self._check_active()
            sock = socket.socket(family, socket.SOCK_STREAM)
            # socket.socket 是标准库连接对象；SOCK_STREAM 表示 TCP 字节流，不保证消息边界。
            self._socket = sock
            sock.setblocking(False)
            # 非阻塞读写暂时做不了时会报 BlockingIOError，先用 select 等待就绪再尝试。
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # NODELAY 减少小命令被延迟合并；KEEPALIVE 帮助系统发现已经失联的对端。
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            # Windows can otherwise take minutes to notice a vanished AP while
            # the application waits for a microphone capture to finish.
            try:
                if hasattr(socket, "SIO_KEEPALIVE_VALS"):
                    sock.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 3000, 1000))
                else:
                    for name, value in (("TCP_KEEPIDLE", 3), ("TCP_KEEPINTVL", 1), ("TCP_KEEPCNT", 3)):
                        option = getattr(socket, name, None)
                        if option is not None:
                            sock.setsockopt(socket.IPPROTO_TCP, option, value)
            except OSError:
                # Protocol response deadlines still bound failures if the OS
                # does not expose configurable keepalive settings.
                pass
            pending = {
                errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY, errno.EINTR,
                getattr(errno, "WSAEWOULDBLOCK", 10035),
                getattr(errno, "WSAEINPROGRESS", 10036),
                getattr(errno, "WSAEALREADY", 10037),
            }
            result = sock.connect_ex(peer)
            # connect_ex 返回错误码而非仅抛异常；非阻塞连接可先返回“进行中”，随后再检查结果。
            if result not in (0, errno.EISCONN, getattr(errno, "WSAEISCONN", 10056)):
                if result not in pending:
                    raise OSError(result, f"Cannot connect to StackChan at {self.endpoint}")
                deadline = time.monotonic() + connect_timeout
                while True:
                    self._check_active()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"StackChan TCP connection timed out: {self.endpoint}")
                    _, writable, exceptional = select.select([], [sock], [sock], min(0.1, remaining))
                    # select 的三个列表分别关注可读/可写/异常；这里等待连接可写或出错。
                    # _ 表示不使用的返回值。短轮询让取消回调有机会再次运行。
                    if not writable and not exceptional:
                        continue
                    code = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                    # “可写”也可能代表连接失败；SO_ERROR 才给出实际连接结果。
                    if code or exceptional:
                        raise OSError(code or errno.ECONNABORTED,
                                      f"Cannot connect to StackChan at {self.endpoint}")
                    break
            self._check_active()
        except Exception:
            self.close()
            raise

    @property
    def is_open(self) -> bool:
        # @property 允许上层写 stream.is_open；fileno>=0 表示本地句柄仍有效，不证明对端一定在线。
        return self._socket is not None and self._socket.fileno() >= 0

    def _ready(self, deadline: float, *, writing: bool) -> bool:
        # read/write 内部调用，返回 bool；deadline 是 monotonic 时钟上的绝对截止时刻。
        self._check_active()
        sock = self._socket
        if sock is None:
            raise ConnectionError("StackChan TCP connection is closed")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        readable, writable, exceptional = select.select(
            [] if writing else [sock], [sock] if writing else [], [sock], min(0.1, remaining)
        )
        if exceptional:
            raise ConnectionError(f"StackChan TCP connection failed: {self.endpoint}")
        return bool(writable if writing else readable)

    def read(self, size: int = 1) -> bytes:
        # Bridge 的文本/二进制收包循环调用。一次返回最多size字节，可能比size短。
        # TCP没有“每次send对应一次recv”的保证，凑满消息由 Bridge 的协议循环负责。
        if type(size) is not int or size < 0:
            raise ValueError("Read size must be a nonnegative integer")
        if size == 0:
            return b""
        deadline = time.monotonic() + self.timeout
        try:
            while time.monotonic() < deadline:
                if not self._ready(deadline, writing=False):
                    continue
                try:
                    data = self._socket.recv(size)
                    # recv 是标准库socket方法；返回 bytes。真实对端关闭时返回 b""，下方转为异常。
                except (BlockingIOError, InterruptedError):
                    # select与recv之间状态仍可能变化；continue 重回循环，不把暂时不可读当掉线。
                    continue
                if not data:
                    raise ConnectionError(f"StackChan TCP disconnected: {self.endpoint}")
                return data
            self._check_active()
            return b""  # A short polling timeout, never confused with peer EOF.
            # 这里的空bytes仅表示本次短轮询没读到数据；Bridge 继续等待其更长的协议截止时间。
        except Exception:
            self.close()
            raise

    def write(self, data: bytes) -> int:
        """Complete partial socket writes within one bounded deadline."""
        view = memoryview(data)
        # memoryview 提供原数据的视图，分片发送时避免反复复制整个字节数组。
        deadline = time.monotonic() + self.write_timeout
        sent = 0
        try:
            while sent < len(view):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"StackChan TCP write timed out after {sent}/{len(view)} bytes")
                if not self._ready(deadline, writing=True):
                    continue
                try:
                    count = self._socket.send(view[sent:])
                    # send 返回本次接受的字节数；sent记录累计进度，[sent:]只发尚未发送部分。
                except (BlockingIOError, InterruptedError):
                    continue
                if count == 0:
                    raise ConnectionError(f"StackChan TCP disconnected during write: {self.endpoint}")
                sent += count
            self._check_active()
            return sent
            # 返回累计字节数给 Bridge；它随后仍要等固件协议 ACK，而非把TCP发送当任务完成。
        except Exception:
            self.close()
            raise

    def reset_input_buffer(self) -> None:
        """Compatibility no-op: valid TCP protocol bytes must never be flushed."""
        self._check_active()
        # 仅为兼容串口接口保留方法名；不会丢弃TCP中已收到但尚未解析的有效协议字节。

    def close(self) -> None:
        sock, self._socket = self._socket, None
        # 同时赋值：保留局部sock用于关闭，把对象字段先置None，重复close就无事可做。
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
                # shutdown 结束双向通信；finally 中 close 释放本地句柄，即使shutdown报错也执行。
            except OSError:
                pass
            finally:
                sock.close()
