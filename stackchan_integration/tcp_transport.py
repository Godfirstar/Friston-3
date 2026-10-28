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
    def __init__(self, host: str, port: int, *, timeout: float = 0.15,
                 write_timeout: float = 1.0, connect_timeout: float = 3.0,
                 check_active: Optional[Callable[[], None]] = None):
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
        self._socket: Optional[socket.socket] = None
        self.endpoint = f"[{address}]:{port}" if address.version == 6 else f"{address}:{port}"
        family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
        peer = (str(address), port, 0, 0) if address.version == 6 else (str(address), port)
        try:
            self._check_active()
            sock = socket.socket(family, socket.SOCK_STREAM)
            self._socket = sock
            sock.setblocking(False)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
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
                    if not writable and not exceptional:
                        continue
                    code = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
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
        return self._socket is not None and self._socket.fileno() >= 0

    def _ready(self, deadline: float, *, writing: bool) -> bool:
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
                except (BlockingIOError, InterruptedError):
                    continue
                if not data:
                    raise ConnectionError(f"StackChan TCP disconnected: {self.endpoint}")
                return data
            self._check_active()
            return b""  # A short polling timeout, never confused with peer EOF.
        except Exception:
            self.close()
            raise

    def write(self, data: bytes) -> int:
        """Complete partial socket writes within one bounded deadline."""
        view = memoryview(data)
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
                except (BlockingIOError, InterruptedError):
                    continue
                if count == 0:
                    raise ConnectionError(f"StackChan TCP disconnected during write: {self.endpoint}")
                sent += count
            self._check_active()
            return sent
        except Exception:
            self.close()
            raise

    def reset_input_buffer(self) -> None:
        """Compatibility no-op: valid TCP protocol bytes must never be flushed."""
        self._check_active()

    def close(self) -> None:
        sock, self._socket = self._socket, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            finally:
                sock.close()
