"""Friston-3 <-> StackChan K151 USB or offline Wi-Fi TCP bridge.

Use ADPCM4 for uploads and negotiate the firmware's acknowledged packet size.
Every clip is CRC-checked by the device before playback. Sensor downloads
use bounded, CRC-checked DATA frames on this same connection. This module
never starts a second reader; call it through StackChanFaceEvents.
Only one process/transport may own the device at a time.
"""

from __future__ import annotations

# 本文件是“设备协议”层。上游 face_events.py 的唯一工作线程调用它；
# 下游是 pyserial.Serial 或 tcp_transport.TcpByteStream，再由固件解析命令。
import argparse
import ipaddress
import sys
import threading
import time
import zlib
from array import array
from typing import Optional
from urllib.parse import urlsplit

try:
    import serial
    # serial 是第三方 pyserial 包的导入名；TCP 分支使用标准库，不依赖这个包成功导入。
except ImportError:
    serial = None  # TCP transport uses only the Python standard library.

from .tcp_transport import TcpByteStream
from .connection import WIFI_HOST, WIFI_PORT, preference_name, usb_candidates
from .audio_protocol import (
    MAX_PCM_BYTES, SAMPLE_RATE, pcm_seconds, validate_microphone, validate_pcm,
)


# Canonical IMA ADPCM step and index tables. Reset predictor/index each clip.
_IMA_STEPS = (
     7,     8,     9,    10,    11,    12,    13,    14,    16,
    17,    19,    21,    23,    25,    28,    31,    34,    37,
    41,    45,    50,    55,    60,    66,    73,    80,    88,
    97,   107,   118,   130,   143,   157,   173,   190,   209,
   230,   253,   279,   307,   337,   371,   408,   449,   494,
   544,   598,   658,   724,   796,   876,   963,  1060,  1166,
  1282,  1411,  1552,  1707,  1878,  2066,  2272,  2499,  2749,
  3024,  3327,  3660,  4026,  4428,  4871,  5358,  5894,  6484,
  7132,  7845,  8630,  9493, 10442, 11487, 12635, 13899, 15289,
 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
)
_IMA_INDEX_DELTAS = (-1, -1, -1, -1, 2, 4, 6, 8,
                     -1, -1, -1, -1, 2, 4, 6, 8)


def _encode_ima_adpcm(pcm16le: bytes) -> bytes:
    """Encode signed little-endian 16-bit mono PCM as 4-bit IMA ADPCM.

    Format: predictor=0, index=0 at clip start, two low-first nibbles per
    byte, final high nibble padded with zero for odd sample counts.
    """
    if len(pcm16le) % 2 != 0:
        raise ValueError("Expected PCM16 samples")
    samples = array("h")
    # array 是标准库的紧凑数值数组；"h" 表示有符号短整数，每个PCM采样占16bit。
    samples.frombytes(pcm16le)
    # frombytes 按数值解释已有字节，不做文本decode；little-endian指低位字节先存。
    if sys.byteorder != "little":
        samples.byteswap()
    encoded = bytearray((len(samples) + 1) // 2)
    # ADPCM用4bit表示一个采样的差值编码，两个采样装进一字节；// 是整数除法。
    predictor = 0
    step_index = 0
    for i, sample in enumerate(samples):
        # enumerate 给出采样序号i和采样值；predictor是编解码双方同步维护的预测值。
        step = _IMA_STEPS[step_index]
        delta = int(sample) - predictor
        code = 0
        if delta < 0:
            code = 8
            delta = -delta
        difference = step >> 3
        # >> 是按位右移，非负整数右移3位相当于除8取整；|= 为某个编码位设置标记。
        if delta >= step:
            code |= 4
            delta -= step
            difference += step
        half = step >> 1
        if delta >= half:
            code |= 2
            delta -= half
            difference += half
        quarter = step >> 2
        if delta >= quarter:
            code |= 1
            difference += quarter

        predictor += -difference if (code & 8) else difference
        # &8 取出符号位；每次按编码后的差值更新预测器，保证与固件解码过程一致。
        predictor = min(32767, max(-32768, predictor))
        step_index += _IMA_INDEX_DELTAS[code]
        step_index = min(88, max(0, step_index))
        if (i & 1) == 0:
            # i&1 判断奇偶：偶数采样写低4位，奇数采样左移4位后并入同一字节高4位。
            encoded[i // 2] = code
        else:
            encoded[i // 2] |= code << 4
    return bytes(encoded)
    # bytearray 可原位修改，bytes 不可变；最终返回可直接传输的压缩二进制。


class StackChanCommandRejected(RuntimeError):
    """A complete, framed GESTURE rejection; the link remains synchronized."""


class StackChanBridge:
    """Run the same bounded device protocol over USB CDC or a local TCP link."""

    VALID_FACES = frozenset({
        "idle", "thinking", "speaking", "happy", "neutral", "doubt",
        "sad", "angry", "sleepy", "listening", "proud", "surprised",
    })
    LEGACY_GESTURES = frozenset({
        "none", "nod", "shake", "tilt", "look_left", "look_right",
    })
    VALID_GESTURES = LEGACY_GESTURES | frozenset({
        "agree", "greet", "think", "proud", "chuckle", "surprised",
        "concerned", "bow", "scan", "attend", "settle", "beat",
    })

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200, *,
                 transport: Optional[str] = None, host: Optional[str] = None,
                 tcp_port: Optional[int] = None):
        # __init__ 只整理配置，不打开连接；真正的串口/socket构造发生在 connect 中。
        # Optional[str] 等价于 str|None；port可为COM名或tcp://地址，transport表示连接偏好。
        endpoint = str(port or "").strip()
        selected = preference_name(transport or ("usb" if endpoint else "auto"))
        if endpoint.lower().startswith("tcp://"):
            parsed = urlsplit(endpoint)
            if (not parsed.hostname or parsed.username is not None or parsed.password is not None
                    or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
                raise ValueError("Use tcp://IP:PORT without credentials, path or query")
            if transport is not None and preference_name(transport) not in ("auto", "wifi"):
                raise ValueError("A tcp:// endpoint requires Wi-Fi transport")
            host = parsed.hostname
            tcp_port = parsed.port if parsed.port is not None else WIFI_PORT
            endpoint, selected = "", "wifi"
        self.preference = selected
        self._explicit_usb = endpoint or None
        self.host = str(ipaddress.ip_address(host or WIFI_HOST))
        self.tcp_port = WIFI_PORT if tcp_port is None else int(tcp_port)
        if not 1 <= self.tcp_port <= 65535:
            raise ValueError("TCP port must be in 1..65535")
        self.transport = "auto"
        self.port = "自动发现"
        self._usb_attempted = set()
        self.baudrate = baudrate
        self._serial = None  # serial.Serial or TcpByteStream; one reader owns it.
        # self._serial 是统一字节流对象，虽叫serial也可能是TCP；两种实现都提供read/write。
        self._firmware_info: Optional[str] = None
        self.last_audio_timing: dict[str, float | str] = {}
        self.last_camera_timing: dict[str, float | int] = {}
        self.last_mic_timing: dict[str, float | int] = {}
        self._cancelled: Optional[threading.Event] = None
        self._deadline: Optional[float] = None
        # cancelled/deadline 由调度器给当前事务设置；分别表示取消标志和绝对截止时刻。

    def _set_operation(self, cancelled=None, deadline=None) -> None:
        """The serial owner's cancellation context; never call from another reader."""
        self._cancelled = cancelled
        self._deadline = deadline

    def _check_active(self) -> None:
        # 被连接/收发循环反复调用，或作为回调传入TcpByteStream；检查不读取另一份设备连接。
        if self._cancelled is not None and self._cancelled.is_set():
            raise RuntimeError("StackChan operation cancelled")
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise TimeoutError("StackChan operation deadline exceeded")

    def _write_bytes(self, data: bytes) -> None:
        # _send_line/音频上传共用。data 是已编码bytes，不会在这里自动补协议换行。
        try:
            self._check_active()
            if self._serial is None or not self._serial.is_open:
                raise RuntimeError("StackChan not connected")
            deadline = time.monotonic() + 1.0
            sent = 0
            while sent < len(data):
                self._check_active()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"StackChan write timed out after {sent}/{len(data)} bytes")
                count = self._serial.write(data[sent:])
                # write可能只接受部分字节；[sent:]只传剩余数据，count是实际本次写入长度。
                if not isinstance(count, int) or count <= 0:
                    raise ConnectionError(f"StackChan write stopped after {sent}/{len(data)} bytes")
                sent += count
            # Protocol ACKs confirm receipt; flush() may block on broken USB.
        except Exception:
            if self.transport == "tcp":
                self.close()
            raise

    def _send_line(self, command: str) -> None:
        # 文本协议入口：str是Python文字，encode('ascii')得到线上bytes，末尾\n划分一行。
        if (not command.isascii() or len(command) > 160
                or any(ch in command for ch in "\r\n")):
            raise ValueError("Commands must be one short ASCII line")
        self._write_bytes((command + "\n").encode("ascii"))

    def _read_line(self, deadline: float) -> str:
        """Read one whole line; retain fragments across short polling timeouts."""
        try:
            if self._serial is None or not self._serial.is_open:
                raise RuntimeError("StackChan not connected")
            line = bytearray()
            # 用可追加的bytearray累积一整行；短轮询超时不会丢掉已经收到的半行。
            while time.monotonic() < deadline:
                self._check_active()
                char = self._serial.read(1)
                # read来自pyserial或本项目TcpByteStream。b'\n'是一个字节，不是字符串'\n'。
                if not char:
                    continue
                if char == b"\n":
                    return bytes(line).decode("ascii", errors="replace").strip()
                    # 仅协议文本行进行decode；相机JPEG/音频内容不能用这套逐行解析。
                line.extend(char)
                if len(line) > 256:
                    raise RuntimeError("StackChan protocol line exceeds 256 bytes")
            raise TimeoutError(f"StackChan response timed out on {self.port}")
        except Exception:
            # TCP disconnect resets firmware framing. USB sensor errors still
            # need _receive_blob's best-effort CANCEL before its port closes.
            if self.transport == "tcp":
                self.close()
            raise

    @property
    def is_connected(self) -> bool:
        # 作为bridge.is_connected读取；除了句柄打开，还要求当前端点通过INFO握手。
        return self._serial is not None and self._serial.is_open and self._firmware_info is not None

    def set_preference(self, value: str) -> None:
        """Called by the device worker at a transaction boundary only."""
        self.preference = preference_name(value)
        self._usb_attempted.clear()

    def _usb_choices(self):
        choices = usb_candidates()
        present = {item.identity for item in choices}
        # An unplug followed by a later plug permits a new handshake even if
        # Windows reuses the same COM number and the device serial number.
        self._usb_attempted.intersection_update(present)
        return choices

    def handoff_needed(self) -> bool:
        """No port is opened here. Never call during a binary transaction."""
        if not self.is_connected:
            return False
        if self.preference == "wifi":
            return self.transport != "tcp"
        choices = self._usb_choices()
        if self.transport == "serial":
            # A disappearing enumerated USB device must not leave an apparently
            # open stale handle until the next long sensor command times out.
            if not self._explicit_usb and not any(item.port == self.port for item in choices):
                raise ConnectionError("栈酱 USB 已拔出")
            return False
        if self.preference == "usb":
            return True
        return any(item.identity not in self._usb_attempted for item in choices)

    def release_for_handoff(self) -> None:
        """Release the old owner before opening the new one; no queued action retries."""
        old_transport = self.transport
        # FaceEvents._run 在事务之间调用；先归还旧连接，再尝试新连接，避免两路抢同一固件。
        if self.is_connected and old_transport == "serial":
            # Firmware rejects a new TCP owner while camera/motion is active.
            # Stop only the current target; do not move to a fresh position.
            self.stop_head()
            self.close_camera()
        self.close()
        # Existing firmware reserves USB for 1200 ms after its last command.
        # TCP close already invokes firmware cleanup; USB opening has its own
        # bounded startup settling period below.
        if old_transport == "serial":
            deadline = time.monotonic() + 1.3
            while time.monotonic() < deadline:
                self._check_active()
                time.sleep(0.05)

    def _connect_endpoint(self, transport: str, endpoint: str) -> None:
        # connect选定候选后调用；transport是serial/tcp，endpoint是COM名/显示用TCP地址。
        self.transport, self.port = transport, endpoint
        self._firmware_info = None  # Identity and capabilities belong to this endpoint.
        if transport == "tcp":
            self._serial = TcpByteStream(
                self.host, self.tcp_port, timeout=0.15, write_timeout=1.0,
                connect_timeout=1.5, check_active=self._check_active,
            )
            # TcpByteStream 定义在tcp_transport.py；回调传self._check_active而非立即调用它。
        else:
            if serial is None:
                raise RuntimeError("USB transport requires: python -m pip install pyserial")
            self._serial = serial.Serial(endpoint, self.baudrate, timeout=0.15, write_timeout=1.0)
            # Serial是pyserial类；构造这里的实例才实际打开USB串口。
            # USB CDC may reboot on open on some board/driver combinations.
            deadline = time.monotonic() + 1.2
            while time.monotonic() < deadline:
                self._check_active()
                time.sleep(0.05)
            self._serial.reset_input_buffer()
        self._exchange("PING", "PONG", timeout=1.2)
        # 打开句柄不代表找对设备；PING与INFO的协议握手确认对端身份和固件能力。
        info = self.info()
        if "FRISTON" not in info.upper() or "STACKCHAN" not in info.upper():
            raise ConnectionError("该端口不是 Friston 栈酱，已停止探测")
        if "SERVO_MOTION_V1" in info:
            # A USB CDC reopen need not reboot the board. Clear the previous
            # owner's finite/pending sequence before this session can play
            # new audio. STOP holds the pose; it never commands a home move.
            self.stop_head()

    def connect(self) -> "StackChanBridge":
        """Discover and verify the device; commands are never replayed here."""
        self._check_active()
        if self.is_connected:
            return self
        choices = self._usb_choices() if self.preference != "wifi" else ()
        candidates = []
        # 候选项是(传输类型,端点,USB身份)三元组；只在当前线程按顺序尝试。
        if self.preference != "wifi":
            if self._explicit_usb:
                candidates.append(("serial", self._explicit_usb, None))
            else:
                # Try newly appeared devices first. Several unrelated ESP32
                # ports must not repeatedly consume the whole discovery slot.
                choices = sorted(choices, key=lambda item: item.identity in self._usb_attempted)
                candidates.extend(("serial", item.port, item.identity) for item in choices)
        if self.preference != "usb":
            host = f"[{self.host}]" if ":" in self.host else self.host
            candidates.append(("tcp", f"tcp://{host}:{self.tcp_port}", None))
        errors = []
        operation_deadline = self._deadline
        # 保存外层总截止时间；单次探测临时使用更短截止时间，finally恢复外层值。
        for transport, endpoint, usb_id in candidates:
            self._check_active()
            attempt_deadline = time.monotonic() + (4.5 if transport == "serial" else 3.0)
            if transport == "serial" and self.preference == "auto" and operation_deadline:
                # Leave a bounded Wi-Fi attempt in every auto-discovery cycle,
                # even when several plausible USB devices do not answer INFO.
                usb_deadline = operation_deadline - 3.0
                if usb_deadline - time.monotonic() < 2.5:
                    continue
                attempt_deadline = min(attempt_deadline, usb_deadline)
            if usb_id is not None:
                self._usb_attempted.add(usb_id)
            self._deadline = min(operation_deadline, attempt_deadline) if operation_deadline else attempt_deadline
            try:
                self._connect_endpoint(transport, endpoint)
                return self
                # 返回对象自身，支持 with StackChanBridge(...) as bridge 的用法。
            except Exception as error:
                self.close()
                errors.append(f"{endpoint}: {error}")
            finally:
                self._deadline = operation_deadline
        detail = "; ".join(errors[-2:]) if errors else "没有枚举到栈酱 USB"
        raise ConnectionError("等待栈酱连接；请接 USB 数据线或让电脑连接栈酱热点。" + detail)

    def _read_response(self, command: str, timeout: float = 3.0) -> str:
        # 简单命令的公共收发入口：先发一行，再读到固件返回的响应类型，返回str。
        try:
            self._send_line(command)
            deadline = time.monotonic() + timeout
            recently_seen = []
            while time.monotonic() < deadline:
                line = self._read_line(deadline)
                if not line:
                    continue
                if line.startswith("ERR "):
                    # GESTURE has a one-line response and no binary payload.
                    # A rejected preset must not tear down a healthy audio
                    # link. Other command failures retain their old recovery.
                    if command.startswith("GESTURE "):
                        # 完整的GESTURE拒绝行说明分帧仍同步；此异常可由MotionPolicy局部处理。
                        raise StackChanCommandRejected(f"StackChan rejected {command!r}: {line}")
                    raise RuntimeError(f"StackChan rejected {command!r}: {line}")
                if (line == "PONG" or line.startswith("OK FACE ")
                        or line.startswith("OK VOLUME ") or line.startswith("OK MOVE ")
                        or line.startswith("OK GESTURE ")
                        or line in ("OK HOME", "OK STOP") or line.startswith("INFO ")
                        or line == "OK CAMERA CLOSE"):
                    return line
                if len(recently_seen) < 5:
                    recently_seen.append(line)
            raise TimeoutError(
                f"No response to {command!r} on {self.port}; "
                f"check firmware, endpoint and transport ownership. Other output: {recently_seen}"
            )
        except StackChanCommandRejected:
            raise
        except Exception:
            if self.transport == "tcp":
                self.close()
            raise

    def _exchange(self, command: str, expected: str, timeout: float = 3.0) -> str:
        # set_face/头部命令等调用；在_read_response之上额外比较完整期望响应内容。
        try:
            received = self._read_response(command, timeout=timeout)
            if received != expected:
                raise RuntimeError(
                    f"Unexpected response to {command!r}: expected {expected!r}, "
                    f"received {received!r}"
                )
            return received
        except StackChanCommandRejected:
            raise
        except Exception:
            if self.transport == "tcp":
                self.close()
            raise

    def ping(self) -> bool:
        self._exchange("PING", "PONG")
        return True

    def info(self) -> str:
        # 固件能力字符串按当前端点缓存；close/换连接会清空，不把旧固件能力套给新设备。
        self._check_active()
        if self._firmware_info is not None:
            return self._firmware_info
        response = self._read_response("INFO")
        if not response.startswith("INFO "):
            raise RuntimeError(f"Unexpected INFO response: {response}")
        self._firmware_info = response[5:]
        return self._firmware_info

    def set_face(self, face: str) -> None:
        # FaceEvents工作线程调用；表情名称转协议大写token，等待明确的OK FACE响应。
        if not isinstance(face, str) or face.lower() not in self.VALID_FACES:
            raise ValueError(f"Unknown face {face!r}. Choices: {sorted(self.VALID_FACES)}")
        mode = face.upper()
        self._exchange(f"FACE {mode}", f"OK FACE {mode}")

    def gesture(self, name: str, *, placement: str = "shoulder", when: str = "now") -> None:
        """Start/stage a finite preset; ACK is not encoder verification.

        SPEECH on GESTURE_V2 starts only after AUDIO STARTED. Legacy firmware
        accepts the original single-token command with default shoulder limits.
        """
        if not isinstance(name, str) or name.lower() not in self.VALID_GESTURES:
            raise ValueError(f"Unknown head gesture: {name!r}")
        if placement not in ("shoulder", "desk"):
            raise ValueError("Gesture placement must be shoulder or desk")
        if when not in ("now", "speech"):
            raise ValueError("Gesture timing must be now or speech")
        info = self.info()
        # MotionPolicy根据回复选择name；placement/when作为已有固件参数发送，不在PC逐帧控制舵机。
        token = name.upper()
        if "GESTURE_V2" in info:
            command = f"GESTURE {token} {placement.upper()} {when.upper()}"
        else:
            if ("GESTURE_V1" not in info or name.lower() not in self.LEGACY_GESTURES
                    or placement != "shoulder" or when != "now"):
                raise ValueError("This gesture/profile/timing requires GESTURE_V2 firmware")
            command = f"GESTURE {token}"
        self._exchange(command, f"OK GESTURE {token}", timeout=3.0)

    def move_head(self, x_tenths: int, y_tenths: int, speed: int = 300) -> None:
        """Request a StackChan head pose; angles use tenths of degrees."""
        if type(x_tenths) is not int or not -900 <= x_tenths <= 900:
            raise ValueError("x_tenths must be an integer in -900..900")
        if type(y_tenths) is not int or not 50 <= y_tenths <= 850:
            raise ValueError("y_tenths must be an integer in 50..850")
        if type(speed) is not int or not 100 <= speed <= 800:
            raise ValueError("speed must be an integer in 100..800")
        if "SERVO_MOTION_V1" not in self.info():
            raise RuntimeError("Firmware does not advertise SERVO_MOTION_V1")
        self._exchange(f"MOVE {x_tenths} {y_tenths} {speed}",
                       f"OK MOVE {x_tenths} {y_tenths} {speed}", timeout=5.0)

    def home_head(self) -> None:
        if "SERVO_MOTION_V1" not in self.info():
            raise RuntimeError("Firmware does not advertise SERVO_MOTION_V1")
        self._exchange("HOME", "OK HOME", timeout=5.0)

    def stop_head(self) -> None:
        if "SERVO_MOTION_V1" not in self.info():
            raise RuntimeError("Firmware does not advertise SERVO_MOTION_V1")
        self._exchange("STOP", "OK STOP", timeout=5.0)

    def _expect_line(self, expected: str, *, timeout: float) -> str:
        """Wait for one response, skipping unrelated startup/debug lines."""
        try:
            # 与_exchange不同，此函数只等待响应，不主动发送命令，用于多阶段二进制事务。
            deadline = time.monotonic() + timeout
            seen = []
            while time.monotonic() < deadline:
                line = self._read_line(deadline)
                if not line:
                    continue
                if line == expected:
                    return line
                if line.startswith("ERR "):
                    raise RuntimeError(f"StackChan: {line}")
                if len(seen) < 5:
                    seen.append(line)
            raise TimeoutError(
                f"Waiting for {expected!r} timed out on {self.port}. Other lines: {seen}"
            )
        except Exception:
            if self.transport == "tcp":
                self.close()
            raise

    def _require_capability(self, capability: str) -> None:
        if capability not in self.info():
            raise RuntimeError(
                f"栈酱固件缺少 {capability}；请确认设备使用本项目的摄像头/麦克风固件。"
            )

    def _receive_blob(self, command: str, *, kind: str, dimensions: tuple[int, int],
                      max_bytes: int, header_timeout: float) -> tuple[bytes, dict]:
        """Download one sensor result; no line reads are allowed inside its bytes."""
        started = time.perf_counter()
        # capture_jpeg/record_pcm调用；返回(二进制bytes,耗时dict)，共用一套DATA分帧协议。
        # 本协议头是ASCII文字，不用struct打包：DATA 类型 字节数 CRC32 尺寸1 尺寸2。
        sent_command = False
        try:
            self._send_line(command)
            sent_command = True
            header_deadline = time.monotonic() + header_timeout
            while True:
                line = self._read_line(header_deadline)
                if line.startswith("ERR ") or line == "MIC CANCELLED":
                    raise RuntimeError(f"StackChan: {line}")
                if line == "MIC EMPTY" and kind == "PCM16":
                    return b"", {"capture_s": time.perf_counter() - started,
                                  "transfer_s": 0.0, "bytes": 0}
                if line == "MIC STARTED" and kind == "PCM16":
                    print("[StackChanMIC] 正在录音，说完后稍作停顿。", flush=True)
                    continue
                if line.startswith("DATA "):
                    break
                # MIC STARTED and short startup/status lines are not payload.

            fields = line.split()
            # split按空白分成str列表；int把十进制长度/尺寸转整数，CRC用基数16解析。
            if len(fields) != 6 or fields[:2] != ["DATA", kind]:
                raise RuntimeError(f"Invalid StackChan DATA header: {line!r}")
            count = int(fields[2])
            if (len(fields[3]) != 8
                    or any(c not in "0123456789abcdefABCDEF" for c in fields[3])):
                raise RuntimeError("Invalid DATA CRC32")
            expected_crc = int(fields[3], 16)
            actual_dimensions = (int(fields[4]), int(fields[5]))
            # JPEG尺寸为(宽,高)，PCM尺寸为(采样率,声道数)，由调用者传dimensions核对。
            if actual_dimensions != dimensions or not 1 <= count <= max_bytes:
                raise RuntimeError(f"Unexpected DATA dimensions/size: {line!r}")
            if kind == "PCM16" and count % 2:
                raise RuntimeError("Microphone PCM16 byte count must be even")

            header_ready = time.perf_counter()
            self._send_line("READ DATA")
            # 告诉固件开始发二进制；从这里到收齐count字节，都不能调用_read_line猜换行。
            # USB CDC and local TCP use the same CRC/ACK flow control. These
            # are failure bounds, not advertised throughput or frame rates.
            transfer_deadline = time.monotonic() + max(5.0, count / 16000.0 + 3.0)
            received = bytearray()
            crc = 0
            # received保存累计载荷；crc保存逐块滚动校验结果，不需要每次重算全部数据。
            while len(received) < count:
                self._check_active()
                chunk_size = min(1024, count - len(received))
                # 最后一块可能不足1024字节；按协议总长计算精确剩余量，避免读入后面的文本。
                chunk = bytearray()
                chunk_deadline = min(transfer_deadline, time.monotonic() + 3.0)
                while len(chunk) < chunk_size:
                    self._check_active()
                    if time.monotonic() >= chunk_deadline:
                        raise TimeoutError(
                            f"Sensor transfer stopped at {len(received)+len(chunk)}/{count} bytes"
                        )
                    part = self._serial.read(chunk_size - len(chunk))
                    # read可能只返回一小段，内层循环先凑齐本块；外层循环再凑齐整个载荷。
                    if part:
                        chunk.extend(part)
                received.extend(chunk)
                crc = zlib.crc32(chunk, crc)
                # zlib是标准库；第二参数传上一块CRC，使最终结果等价于对全部载荷计算CRC。
                self._send_line(f"ACK {len(received)}")
                # ACK带“累计收到多少字节”，固件收到确认后再推进，不是另一个并发读取线程。
            self._expect_line("DONE DATA", timeout=3.0)
            # 精确收齐二进制之后才恢复文本解析，等待事务尾标记。
            if (crc & 0xFFFFFFFF) != expected_crc:
                raise RuntimeError("StackChan sensor CRC32 mismatch; discarded frame/recording")
            timing = {"capture_s": header_ready - started,
                      "transfer_s": time.perf_counter() - header_ready,
                      "bytes": count}
            return bytes(received), timing
            # 转为不可变bytes交给上层，JPEG仍是压缩图片；解码到像素由摄像头适配层完成。
        except Exception:
            # At all sensor protocol waits CANCEL is a text command. Sending it
            # releases a firmware MIC wait or DATA/ACK wait before reconnecting.
            # Never do this inside the legacy host-to-device AUDIO binary path.
            if sent_command and self.transport == "serial":
                self.cancel_sensor()
            self.close()
            raise

    def capture_jpeg(self, width: int = 320) -> bytes:
        # FaceEvents._execute_core调用，最终供friston3.hardware.camera中的StackChan相机使用。
        if type(width) is not int or width not in (320, 640):
            raise ValueError("Camera width must be 320 or 640")
        self._require_capability("CAMERA_JPEG_V1")
        self.last_camera_timing = {}
        jpeg, timing = self._receive_blob(
            f"CAMERA {width}", kind="JPEG", dimensions=(width, width * 3 // 4),
            max_bytes=640 * 480 * 2, header_timeout=8.0,
        )
        self.last_camera_timing = timing
        return jpeg

    def close_camera(self) -> None:
        self._require_capability("CAMERA_JPEG_V1")
        self._exchange("CAMERA CLOSE", "OK CAMERA CLOSE", timeout=5.0)

    def cancel_sensor(self) -> None:
        """传感器失败后的尽力释放；音频上传的二进制阶段不能调用此方法。"""
        if self._serial is not None:
            # 仅owner做尽力清理。音频上传阶段固件把任何字节当载荷，不能插入文本CANCEL。
            try:
                self._serial.write(b"CANCEL\n")
            except Exception:
                pass

    def record_pcm(self, mode: str = "touch", *, wait_ms: int = 1000,
                   max_ms: int = 8000, silence_ms: int = 700,
                   threshold: int = 350) -> bytes:
        # FaceEvents._execute_core调用；等待固件录音，再按DATA协议下载原始PCM字节。
        mode = validate_microphone(mode, wait_ms, max_ms, silence_ms, threshold).upper()
        self._require_capability("MIC_PCM16_V1")
        self.last_mic_timing = {}
        pcm, timing = self._receive_blob(
            f"MIC {mode} {wait_ms} {max_ms} {silence_ms} {threshold}",
            kind="PCM16", dimensions=(SAMPLE_RATE, 1), max_bytes=MAX_PCM_BYTES,
            header_timeout=(wait_ms + max_ms) / 1000.0 + 4.0,
        )
        self.last_mic_timing = timing
        return pcm

    def set_volume(self, level: int) -> None:
        """Set moderate output volume. v0.4 firmware only."""
        if not isinstance(level, int) or not 0 <= level <= 160:
            raise ValueError("Volume must be an integer from 0 to 160")
        self._exchange(f"VOLUME {level}", f"OK VOLUME {level}")

    def play_pcm(self, pcm: bytes, *, sample_rate: int = SAMPLE_RATE) -> None:
        """Upload and play <=12s PCM16LE mono with negotiated flow control.

        Only send 1024-byte packets when the firmware explicitly advertises
        AUDIO4B support and its enlarged receive buffer. Older devices retain
        their 128-byte handshake. Never infer a safe packet size from baudrate.
        """
        self.last_audio_timing = {}
        if sample_rate != SAMPLE_RATE:
            raise ValueError("Expected 16000Hz PCM16LE mono audio")
        validate_pcm(pcm)
        if self._serial is None or not self._serial.is_open:
            raise RuntimeError("StackChan not connected")
        info = self.info()
        self._require_capability("ADPCM4")
        wire = _encode_ima_adpcm(pcm)
        # pcm是原始16bit采样，wire是4bit ADPCM压缩结果；两者长度不能混用。
        command, packet_bytes = ("AUDIO4B", 1024) if "ADPCM4_ACK1024" in info else ("AUDIO4", 128)
        header = f"{command} {len(pcm)//2} {zlib.crc32(wire) & 0xFFFFFFFF:08X}\n"
        # 头部报原始采样数量与压缩载荷CRC；&0xFFFFFFFF取32bit，:08X格式化为8位大写十六进制。
        codec_name = "ADPCM4"

        started_upload = time.perf_counter()
        self._write_bytes(header.encode("ascii"))
        self._expect_line("READY AUDIO", timeout=3.0)
        # READY表示固件已准备接收载荷；后续每包必须得到累计ACK再发送下一包。

        # One acknowledged packet in flight; CRC prevents corrupted playback.
        for offset in range(0, len(wire), packet_bytes):
            # range第三参数是步长；offset为当前包起点，切片取最多packet_bytes个字节。
            chunk = wire[offset : offset + packet_bytes]
            self._write_bytes(chunk)
            confirmed = offset + len(chunk)
            try:
                self._expect_line(f"ACK {confirmed}", timeout=5.0)
            except (RuntimeError, TimeoutError) as exc:
                raise RuntimeError(
                    f"{codec_name} transfer stopped at {confirmed}/{len(wire)} "
                    f"bytes (last confirmed {offset}): {exc}"
                ) from exc

        clip_duration = pcm_seconds(len(pcm))
        self._expect_line("AUDIO STARTED", timeout=10.0)
        # 全部ACK只表示上传；STARTED才表示开始播音，DONE才表示当前音频事务结束。
        upload_seconds = time.perf_counter() - started_upload
        started_playback = time.perf_counter()
        self._expect_line("AUDIO DONE", timeout=clip_duration + 7.0)
        playback_seconds = time.perf_counter() - started_playback
        self.last_audio_timing = {
            "codec": codec_name, "transfer_s": upload_seconds,
            "started_at": started_playback, "playback_s": playback_seconds,
            "audio_s": clip_duration,
        }
        print(
            f"[StackChan:{self.transport.upper()}] codec={codec_name} ack={packet_bytes} wire={len(wire)}B "
            f"raw={len(pcm)}B transfer={upload_seconds:.2f}s "
            f"playback={playback_seconds:.2f}s "
            f"audio={clip_duration:.2f}s",
            flush=True,
        )

    def close(self) -> None:
        # owner在退出/掉线/切换时调用；接口统一，底层是串口或TCP都使用.close释放句柄。
        self._firmware_info = None
        if self._serial is not None:
            try:
                self._serial.close()
            finally:
                self._serial = None

    def __enter__(self) -> "StackChanBridge":
        # with进入时连接，__exit__离开时关闭；异常路径也会调用__exit__。
        return self.connect()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def main() -> None:
    # 本文件独立CLI入口；正常NPC业务通过FaceEvents共享owner，不再额外开一个Bridge。
    parser = argparse.ArgumentParser(description="Friston-3 StackChan face controller")
    parser.add_argument("--port", default=None, help="USB COM port or tcp://IP:PORT")
    parser.add_argument("--transport", choices=("auto", "usb", "wifi"), help="自动发现；USB/Wi-Fi 无需环境变量")
    parser.add_argument("--host", help="Numeric device IP for offline TCP (default 192.168.4.1)")
    parser.add_argument("--tcp-port", type=int, help="Offline TCP port (default 8765)")
    parser.add_argument("--face", choices=sorted(StackChanBridge.VALID_FACES))
    parser.add_argument("--gesture", choices=sorted(StackChanBridge.VALID_GESTURES),
                        help="Run one finite head preset; extended presets require GESTURE_V2")
    parser.add_argument("--placement", choices=("shoulder", "desk"), default="shoulder",
                        help="Gesture amplitude profile; desk requires GESTURE_V2")
    parser.add_argument("--move", nargs=2, type=int, metavar=("X_TENTHS", "Y_TENTHS"))
    parser.add_argument("--speed", type=int, default=300)
    parser.add_argument("--home", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument(
        "--interactive", action="store_true",
        help="Keep the connection open and switch faces interactively (no re-flash)",
    )
    parser.add_argument(
        "--hold", type=float, default=None,
        help="Keep the connection open for N seconds; TCP presets default to 4 seconds",
    )
    args = parser.parse_args()
    if args.hold is not None and not 0 <= args.hold <= 300:
        parser.error("--hold must be between 0 and 300 seconds")

    try:
        transport = args.transport
        if args.port and not args.port.lower().startswith("tcp://") and transport is None:
            transport = "usb"
        robot = StackChanBridge(port=args.port, transport=transport, host=args.host,
                                tcp_port=args.tcp_port)
        hold_seconds = args.hold or 0.0
        with robot:
            if robot.transport == "tcp" and not args.interactive:
                if (args.move or args.home) and not hold_seconds > 0:
                    parser.error("TCP MOVE/HOME require an explicit positive --hold (or --interactive); "
                                 "disconnecting stops motion. No MOVE/HOME command was sent.")
                if args.gesture and args.gesture != "none" and args.hold is None:
                    hold_seconds = 4.0
                elif args.gesture and args.gesture != "none" and hold_seconds == 0:
                    parser.error("TCP --gesture needs --hold > 0 or omit --hold for 4 seconds; "
                                 "--hold 0 would immediately stop the gesture. No gesture command was sent.")
            print(f"Connected: {robot.port} ({robot.info()})")
            if args.face:
                robot.set_face(args.face)
                print(f"Face switched to {args.face}")
            if args.gesture:
                robot.gesture(args.gesture, placement=args.placement)
                print(f"Gesture accepted: {args.gesture}; observe the real head to verify movement")
            if args.move:
                robot.move_head(args.move[0], args.move[1], args.speed)
                print(f"Head move command accepted: X={args.move[0] / 10:.1f}, Y={args.move[1] / 10:.1f}")
            if args.home:
                robot.home_head()
                print("Head home command accepted")
            if args.stop:
                robot.stop_head()
                print("Head stop command accepted")
            if not (args.face or args.gesture or args.move or args.home or args.stop):
                print("PING -> PONG")

            if args.interactive:
                print("Keep-open mode. Type a face name, or 'quit' to exit.")
                print("Available:", ", ".join(sorted(StackChanBridge.VALID_FACES)))
                while True:
                    try:
                        face = input("face> ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        break
                    if face in ("exit", "quit", "q"):
                        break
                    if not face:
                        continue
                    if face not in StackChanBridge.VALID_FACES:
                        print(f"Unknown face: {face!r}")
                        continue
                    robot.set_face(face)
                    print(f"OK FACE {face.upper()} (connection kept open)")
            elif hold_seconds:
                print(f"Keeping {robot.port} open for {hold_seconds:g} seconds...")
                time.sleep(hold_seconds)
    except (OSError, TimeoutError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"StackChan connection failed: {exc}\n")


if __name__ == "__main__":
    main()
