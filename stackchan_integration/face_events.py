"""Single-owner USB/TCP scheduler for Friston-3 expressions, audio and sensors.

Only this worker opens the device link. Camera and audio transactions are
serialized, so binary image/audio bytes cannot be mistaken for text responses.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .stackchan_bridge import StackChanBridge
from .connection import preference_name
from .motion_events import MotionPolicy
from .audio_protocol import pcm_seconds, validate_microphone, validate_pcm
from friston3.runtime_settings import SETTINGS

LOG = logging.getLogger(__name__)


@dataclass
class _DeviceJob:
    operation: str
    arguments: dict
    deadline: float
    session_epoch: Optional[int] = None
    motion_stamp: Optional[tuple[int, int, str]] = None
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    result: object = None
    error: Optional[Exception] = None


class StackChanFaceEvents:
    """Keep one connection owner for the lifetime of the application.

    Expressions are coalesced; all sensor/audio requests have a bounded queue
    lifetime. An expired request cannot unexpectedly run later. Cancellation
    is handled by the owner, never by a second thread closing/reading its port.
    """

    def __init__(self, port: Optional[str] = None, enabled: Optional[bool] = None, *,
                 transport: Optional[str] = None, host: Optional[str] = None,
                 tcp_port: Optional[int] = None):
        # Resolve configuration before starting the worker, without opening a
        # connection. Invalid settings then reach main.py's visible error path.
        selected = transport or ("usb" if port and not port.lower().startswith("tcp://")
                                 else "wifi" if port else SETTINGS.get("connection_preference", "auto"))
        self._bridge = StackChanBridge(
            port=port, transport=selected, host=host, tcp_port=tcp_port
        )
        self.port = self._bridge.port
        self.transport = self._bridge.transport
        self.enabled = True if enabled is None else bool(enabled)
        self._cond = threading.Condition()
        self._motion = MotionPolicy(self._cond)
        self._connected = False
        self._connection_error = "正在自动发现栈酱"
        self._desired_preference = self._bridge.preference
        self._preference_dirty = False
        self._connection_epoch = 0
        self._wanted = "idle"
        self._version = 1
        self._face_hold_until = 0.0
        self._face_hold_seconds = 6.0
        self._closed = False
        self._jobs: deque[_DeviceJob] = deque()
        self._active_job: Optional[_DeviceJob] = None
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self.last_audio_timing: dict[str, float | str] = {}
        self.last_camera_timing: dict[str, float | int] = {}
        self.last_mic_timing: dict[str, float | int] = {}
        if self.enabled:
            self._worker = threading.Thread(
                target=self._run, daemon=True, name="stackchan-hardware"
            )
            self._worker.start()

    def connection_status(self) -> dict:
        """A thread-safe snapshot; checking status never probes another port."""
        with self._cond:
            return {
                "connected": self._connected,
                "transport": self.transport,
                "endpoint": self.port,
                "preference": self._desired_preference,
                "epoch": self._connection_epoch,
                "error": self._connection_error,
                "busy": self._active_job.operation if self._active_job else "",
                "motion": self._motion.last_result,
                "switch_pending": self._preference_dirty or (
                    self._connected and self._desired_preference != "auto" and
                    {"usb": "serial", "wifi": "tcp"}.get(self._desired_preference) != self.transport),
            }

    def set_connection_preference(self, preference: str) -> dict:
        preference = preference_name(preference)
        # Persist before changing the live preference; failed saves never claim
        # a preference that would disappear at the next startup.
        SETTINGS.update(connection_preference=preference)
        with self._cond:
            self._desired_preference = preference
            self._preference_dirty = True
            self._cond.notify_all()
        return self.connection_status()

    def wait_ready(self, timeout: float = 1.0) -> bool:
        """Wait in short interruptible intervals while the owner reconnects."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._cond:
            while not self._connected and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(timeout=min(0.1, remaining))
            return self._connected and not self._closed

    def set_face(self, face: str) -> None:
        face = face.lower()
        if face not in StackChanBridge.VALID_FACES:
            raise ValueError(f"Invalid StackChan expression: {face}")
        if not self.enabled:
            return
        with self._cond:
            if self._closed:
                return
            self._wanted = face
            self._version += 1
            self._cond.notify()

    def on_request(self) -> None:
        with self._cond:
            self._discard_queued_motion_locked()
            # A console request begun while disconnected has no hardware
            # session to resume on. Its text reply may still complete.
            self._motion.begin_turn(self._connection_epoch if self._connected else -1)
            self._face_hold_until = 0.0
        self.set_face("thinking")
        self._queue_state_motion("think", "thinking", cooldown=9.0)

    def prepare_reply(self, result) -> None:
        """Stage the same model turn's validated face and bounded gesture.

        The face appears immediately. A gesture starts with the first audio
        transaction, or at completion for a muted/text-only reply.
        """
        with self._cond:
            self._discard_queued_motion_locked()
            presentation = self._motion.prepare_reply(
                result, self._connection_epoch if self._connected else -1)
            if presentation is None:
                return
            face, stop_old_motion = presentation
            self._face_hold_until = 0.0
            if stop_old_motion:
                # The answer can explicitly request stillness. A THINK cue
                # already sent while planning must not linger into that reply.
                self._queue_stop_motion_locked()
        self.set_face(face)

    def on_reply_ready(self) -> None:
        with self._cond:
            self._discard_queued_motion_locked()
            self._wanted, hold_face = self._motion.finish_reply()
            self._face_hold_until = time.monotonic() + self._face_hold_seconds if hold_face else 0.0
            self._version += 1
            self._cond.notify()

    @property
    def connection_epoch(self) -> int:
        """Current hardware session; sensor caches invalidate when this changes."""
        with self._cond:
            return self._connection_epoch

    @property
    def playback_epoch(self) -> int:
        """Bind in-flight synthesis to the connection that prepared its reply."""
        return self._motion.playback_epoch

    @property
    def last_motion_result(self) -> dict:
        return self._motion.last_result

    def _discard_queued_motion_locked(self) -> None:
        """Remove decorative jobs only; sensor/audio and explicit stops survive."""
        kept = deque()
        for job in self._jobs:
            if job.operation in ("motion_event", "reply_gesture"):
                job.cancelled.set()
                job.result = {"status": "skipped", "reason": "superseded"}
                job.done.set()
            else:
                kept.append(job)
        self._jobs = kept

    def _queue_stop_motion_locked(self) -> None:
        if not self.enabled or not self._connected or self._closed:
            return
        if any(job.operation == "motion_stop" for job in self._jobs):
            return
        # Stop is bounded and uses the same owner. Putting it before queued
        # jobs never interrupts a binary transaction already in progress.
        self._jobs.appendleft(_DeviceJob("motion_stop", {}, time.monotonic() + 4.0,
                                        session_epoch=self._connection_epoch))
        self._cond.notify_all()

    def _queue_state_motion(self, gesture: str, phase: str, *, cooldown: float) -> None:
        """No wait in the caller: short-lived state cues cannot delay planning."""
        with self._cond:
            now = time.monotonic()
            if (not self.enabled or self._closed or not self._connected
                    or len(self._jobs) >= 7):
                return
            stamp = self._motion.state_cue(phase, cooldown)
            if stamp is None:
                return
            self._jobs.append(_DeviceJob(
                "motion_event", {"gesture": gesture}, now + 2.0,
                session_epoch=self._connection_epoch, motion_stamp=stamp))
            self._cond.notify_all()

    def cancel_motion(self) -> dict:
        """Clear queued cues, then wait for a stop ACK on the existing worker.

        This remains available after gestures_enabled becomes false. A lost
        connection is not re-opened to replay a stop or any earlier gesture.
        """
        with self._cond:
            self._motion.cancel()
            self._discard_queued_motion_locked()
            if not self.enabled or not self._connected or self._closed:
                return {"status": "skipped", "reason": "not_connected"}
        return self._submit("motion_stop", timeout=4.0)

    def _submit(self, operation: str, *, timeout: float,
                session_epoch: Optional[int] = None, **arguments):
        if not self.enabled:
            raise RuntimeError("栈酱设备管理器未启用")
        job = _DeviceJob(operation, arguments, time.monotonic() + timeout,
                         session_epoch=session_epoch)
        with self._cond:
            if self._closed:
                raise RuntimeError("StackChan device manager is closed")
            if not self._connected:
                raise ConnectionError("栈酱尚未连接，正在自动恢复；本次设备请求未执行")
            if session_epoch is not None and session_epoch != self._connection_epoch:
                raise ConnectionError("设备连接已中断，本轮旧语音已丢弃；不会在重连后补播")
            job.session_epoch = self._connection_epoch if session_epoch is None else session_epoch
            if operation in ("audio", "reply_gesture"):
                job.motion_stamp = self._motion.stamp
            if len(self._jobs) >= 8:
                raise RuntimeError("StackChan request queue is full")
            self._jobs.append(job)
            self._cond.notify()
        try:
            completed = False
            # Short waits let Windows Python dispatch Ctrl+C while a hardware
            # transaction is pending; use the original deadline for expiry.
            while not completed:
                remaining = job.deadline - time.monotonic()
                if remaining <= 0:
                    break
                completed = job.done.wait(timeout=min(0.1, remaining))
        except BaseException:
            # In particular Ctrl+C: never leave a queued recording or playback
            # request that can start after the caller has abandoned it.
            self._cancel(job)
            raise
        if not completed:
            with self._cond:
                # Avoid discarding a result completed at the timeout boundary.
                completed = job.done.is_set()
            if not completed:
                self._cancel(job)
                raise TimeoutError(
                    f"StackChan {operation} exceeded {timeout:.0f}s; request cancelled."
                )
        if job.error is not None:
            if isinstance(job.error, ConnectionError):
                raise ConnectionError(f"StackChan {operation}: {job.error}") from job.error
            if isinstance(job.error, TimeoutError):
                raise TimeoutError(f"StackChan {operation}: {job.error}") from job.error
            raise RuntimeError(f"StackChan {operation} failed: {job.error}") from job.error
        return job.result

    def _cancel(self, job: _DeviceJob) -> None:
        with self._cond:
            job.cancelled.set()
            try:
                self._jobs.remove(job)
            except ValueError:
                # The connection owner observes cancellation during active I/O.
                pass
            else:
                job.error = RuntimeError("StackChan request cancelled before execution")
                job.done.set()
            self._cond.notify_all()

    def get_info(self) -> str:
        return self._submit("info", timeout=12.0)

    def move_head(self, x_tenths: int, y_tenths: int, speed: int = 300) -> None:
        if type(x_tenths) is not int or not -900 <= x_tenths <= 900:
            raise ValueError("x_tenths must be an integer in -900..900")
        if type(y_tenths) is not int or not 50 <= y_tenths <= 850:
            raise ValueError("y_tenths must be an integer in 50..850")
        if type(speed) is not int or not 100 <= speed <= 800:
            raise ValueError("speed must be an integer in 100..800")
        self._submit("move_head", timeout=8.0, x_tenths=x_tenths,
                     y_tenths=y_tenths, speed=speed)

    def home_head(self) -> None:
        self._submit("home_head", timeout=8.0)

    def stop_head(self) -> None:
        self._submit("stop_head", timeout=8.0)

    def capture_jpeg(self, width: int = 320) -> bytes:
        if type(width) is not int or width not in (320, 640):
            raise ValueError("Camera width must be 320 or 640")
        self.last_camera_timing = {}
        return self._submit("camera", timeout=55.0, width=width)

    def capture_identity_jpeg(self, width: int = 320, *, timeout: float = 2.5) -> bytes:
        """One bounded capture-and-release job, shared with MIC/TTS/photo I/O."""
        if type(width) is not int or width not in (320, 640):
            raise ValueError("Camera width must be 320 or 640")
        if not isinstance(timeout, (int, float)) or not 0.5 <= timeout <= 8:
            raise ValueError("Identity camera timeout must be 0.5..8 seconds")
        return self._submit("identity_camera", timeout=float(timeout), width=width)

    def close_camera(self) -> None:
        self._submit("camera_close", timeout=12.0)

    def record_pcm(self, mode: str = "touch", *, wait_ms: int = 1000,
                   max_ms: int = 8000, silence_ms: int = 700,
                   threshold: int = 350) -> bytes:
        mode = validate_microphone(mode, wait_ms, max_ms, silence_ms, threshold)
        self.last_mic_timing = {}
        with self._cond:
            self._motion.begin_listening()
        if mode == "touch":
            # Firmware may finish this quiet cue while waiting for a touch,
            # but actual recording cancels it. VAD is always armed in stillness.
            self._queue_state_motion("attend", "listening", cooldown=24.0)
        return self._submit(
            "microphone", timeout=(wait_ms + max_ms) / 1000.0 + 40.0,
            mode=mode, wait_ms=wait_ms, max_ms=max_ms,
            silence_ms=silence_ms, threshold=threshold,
        )

    def play_pcm(self, pcm: bytes, *, volume: int = 60,
                 session_epoch: Optional[int] = None) -> bool:
        """Upload and play through the same connection; wait for AUDIO DONE."""
        if not self.enabled:
            return False
        validate_pcm(pcm)
        if type(volume) is not int or not 0 <= volume <= 160:
            raise ValueError("Volume must be 0..160")
        self.last_audio_timing = {}
        duration = pcm_seconds(len(pcm))
        timeout = max(30.0, min(180.0, duration * 16.0 + 20.0))
        self._submit("audio", timeout=timeout, session_epoch=session_epoch,
                     pcm=pcm, volume=volume)
        return True

    # Motion, speech and sensors share the same transaction worker.
    def on_task_completed(self, result) -> None:
        if not self.enabled or not self._motion.needs_reply_gesture(result):
            return
        try:
            self._submit("reply_gesture", timeout=4.0)
        except Exception as error:
            # 不重放不确定的动作回执，也不覆盖原任务结果。
            LOG.warning("表演动作未确认，保留原任务结果：%s", error)

    def _execute(self, bridge: StackChanBridge, job: _DeviceJob):
        if job.operation == "motion_event":
            with self._cond:
                current = (job.session_epoch == self._connection_epoch
                           and not job.cancelled.is_set() and time.monotonic() < job.deadline)
                stamp = job.motion_stamp if current else None
            return self._motion.execute_state(bridge, job.arguments["gesture"], stamp)
        if job.operation == "motion_stop":
            return self._motion.execute_stop(bridge)
        if job.operation == "reply_gesture":
            return self._motion.execute_reply(bridge, job.motion_stamp)
        # Keep camera ownership conservative after failures. v3 MOVE can close
        # the camera, so automatic gestures must never run during its session.
        if job.operation in ("camera", "identity_camera"):
            with self._cond:
                self._motion.camera_session(True)
                self._discard_queued_motion_locked()
        result = self._execute_core(bridge, job)
        if job.operation in ("camera_close", "identity_camera", "audio", "microphone"):
            self._motion.camera_session(False)
        return result

    def _execute_core(self, bridge: StackChanBridge, job: _DeviceJob):
        if job.operation == "info":
            return bridge.info()
        if job.operation == "move_head":
            return bridge.move_head(**job.arguments)
        if job.operation == "home_head":
            return bridge.home_head()
        if job.operation == "stop_head":
            return bridge.stop_head()
        if job.operation == "camera":
            result = bridge.capture_jpeg(**job.arguments)
            self.last_camera_timing = dict(bridge.last_camera_timing)
            return result
        if job.operation == "identity_camera":
            try:
                result = bridge.capture_jpeg(**job.arguments)
                self.last_camera_timing = dict(bridge.last_camera_timing)
                bridge.close_camera()
                return result
            except Exception:
                # JPEG may already be complete when close_camera hits the job
                # deadline. Its transfer cleanup no longer runs at that point.
                # Only this connection owner sends CANCEL; firmware releases
                # camera/I2C without adding another connection or waiting loop.
                bridge.cancel_sensor()
                raise
        if job.operation == "camera_close":
            return bridge.close_camera()
        if job.operation == "microphone":
            result = bridge.record_pcm(**job.arguments)
            self.last_mic_timing = dict(bridge.last_mic_timing)
            return result
        if job.operation == "audio":
            self._motion.execute_reply(bridge, job.motion_stamp, audio_bytes=len(job.arguments["pcm"]))
            bridge.set_volume(job.arguments["volume"])
            bridge.play_pcm(job.arguments["pcm"])
            self.last_audio_timing = dict(bridge.last_audio_timing)
            return True
        raise ValueError(f"Unknown StackChan operation: {job.operation}")

    def _invalidate_connection(self, error) -> None:
        with self._cond:
            if self._connected:
                self._connection_epoch += 1
            self._connected = False
            self._connection_error = str(error)
            self._motion.connection_changed(connected=False)
            self._face_hold_until = 0.0
            self._wanted = "idle"
            self._version += 1
            while self._jobs:
                abandoned = self._jobs.popleft()
                abandoned.cancelled.set()
                abandoned.error = ConnectionError("连接已切换或断开，旧设备请求已丢弃")
                abandoned.done.set()
            self._cond.notify_all()

    def _mark_connected(self, bridge) -> None:
        with self._cond:
            newly_connected = not self._connected
            self.port, self.transport = bridge.port, bridge.transport
            self._connected = True
            self._connection_error = ""
            self._cond.notify_all()
        if newly_connected:
            label = "USB" if bridge.transport == "serial" else "Wi-Fi"
            print(f"[StackChan] 已连接 {label}：{bridge.port}；已确认 Friston 栈酱。", flush=True)
            capabilities = bridge.info()
            with self._cond:
                self._motion.connection_changed(connected=True)
            if "GESTURE_V2" in capabilities:
                print("[StackChanMotion] GESTURE_V2 已启用：丰富预设、肩上/桌面幅度、首声同步。", flush=True)
            elif "GESTURE_V1" in capabilities:
                print("[StackChanMotion] 当前为 GESTURE_V1：使用基础动作降级；更新固件可启用丰富预设。", flush=True)
            else:
                print("[StackChanMotion] 固件尚无动作预设，表情/语音照常；更新固件后启用舵机表演。", flush=True)

    def _expire_jobs(self, now) -> None:
        """Called under the condition lock, before claiming the next job."""
        while self._jobs and (
                self._jobs[0].cancelled.is_set() or self._jobs[0].deadline <= now
                or self._jobs[0].session_epoch != self._connection_epoch):
            job = self._jobs.popleft()
            job.error = (ConnectionError("旧连接的设备请求已丢弃")
                         if job.session_epoch != self._connection_epoch
                         else TimeoutError("设备请求在执行前已过期"))
            job.done.set()

    def _run(self) -> None:
        bridge = self._bridge
        delivered_revision = 0
        next_connect_time = next_probe = last_warning = 0.0
        try:
            while True:
                with self._cond:
                    now = time.monotonic()
                    if self._closed:
                        break
                    if self._face_hold_until and now >= self._face_hold_until:
                        self._face_hold_until = 0.0
                        self._wanted = "idle"
                        self._version += 1
                    self._expire_jobs(now)
                    if now < next_connect_time:
                        self._cond.wait(timeout=min(0.25, next_connect_time - now))
                        continue
                    maintain = not self._connected or now >= next_probe or self._preference_dirty
                    if not maintain and not self._jobs and self._version == delivered_revision:
                        # A timed idle wakeup is essential: plugging USB in must
                        # be discovered without an unrelated face/audio request.
                        self._cond.wait(timeout=min(0.25, max(0.01, next_probe - now)))
                        continue
                    preference, preference_dirty = self._desired_preference, self._preference_dirty
                    can_handoff = self._motion.can_handoff

                if maintain:
                    # Discovery is separate from any user job's deadline, and
                    # only this worker ever opens or closes a device endpoint.
                    bridge._set_operation(self._stop_event, time.monotonic() + 10.0)
                    try:
                        if preference_dirty:
                            bridge.set_preference(preference)
                            with self._cond:
                                if self._desired_preference == preference:
                                    self._preference_dirty = False
                        if can_handoff and bridge.handoff_needed():
                            with self._cond:
                                # on_request may arrive while USB metadata is
                                # being enumerated. Reserve the handoff under
                                # the same lock that begins a dialogue turn.
                                can_handoff = self._motion.can_handoff
                                if can_handoff:
                                    self._invalidate_connection("正在安全切换栈酱连接")
                            if can_handoff:
                                bridge.release_for_handoff()
                                delivered_revision = 0
                        if not bridge.is_connected:
                            bridge.connect()
                            self._mark_connected(bridge)
                        elif now >= next_probe:
                            bridge._exchange("PING", "PONG", timeout=1.0)
                        next_probe = time.monotonic() + 1.0
                    except Exception as error:
                        bridge.close()
                        self._invalidate_connection(error)
                        delivered_revision = 0
                        next_connect_time = time.monotonic() + 2.0
                        if not self._stop_event.is_set() and (
                                not last_warning or time.monotonic() - last_warning >= 15.0):
                            LOG.warning("StackChan 等待自动连接：%s", error)
                            last_warning = time.monotonic()
                        continue
                    finally:
                        bridge._set_operation()

                with self._cond:
                    if self._closed:
                        break
                    self._expire_jobs(time.monotonic())
                    job = self._jobs.popleft() if self._jobs else None
                    self._active_job = job
                    revision = self._version
                    face = self._wanted if revision != delivered_revision else None
                if job is None and face is None:
                    continue
                bridge._set_operation(
                    job.cancelled if job else self._stop_event,
                    job.deadline if job else time.monotonic() + 5.0,
                )
                try:
                    if face is not None:
                        if job is not None and job.operation == "motion_event":
                            # A decorative cue's short queue lifetime must not
                            # shorten an otherwise healthy expression exchange.
                            bridge._set_operation(self._stop_event, time.monotonic() + 5.0)
                        bridge.set_face(face)
                        delivered_revision = revision
                    if job is not None:
                        if job.operation == "motion_event":
                            bridge._set_operation(job.cancelled, job.deadline)
                        job.result = self._execute(bridge, job)
                except Exception as error:
                    old_transport = bridge.transport
                    bridge.close()
                    if job is not None:
                        job.error = ConnectionError(str(error))
                    self._invalidate_connection(error)
                    delivered_revision = 0
                    # A broken USB AUDIO upload has no line-level CANCEL: wait
                    # for firmware's bounded RX timeout before sending PING.
                    delay = 15.0 if job is not None and job.operation == "audio" and old_transport == "serial" else 2.0
                    next_connect_time = time.monotonic() + delay
                    if not self._stop_event.is_set():
                        LOG.warning("StackChan 本次请求已中止，将自动恢复连接：%s", error)
                finally:
                    bridge._set_operation()
                    with self._cond:
                        self._active_job = None
                        if job is not None:
                            if job.operation in ("audio", "microphone", "identity_camera"):
                                self._version += 1
                            job.done.set()
                        self._cond.notify_all()
        finally:
            with self._cond:
                self._connected = False
                while self._jobs:
                    job = self._jobs.popleft()
                    job.error = ConnectionError("设备管理器已关闭")
                    job.done.set()
                self._cond.notify_all()
            bridge.close()

    def close(self) -> None:
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._stop_event.set()
            if self._active_job is not None:
                self._active_job.cancelled.set()
            while self._jobs:
                job = self._jobs.popleft()
                job.cancelled.set()
                job.error = RuntimeError("Device manager closed before execution")
                job.done.set()
            self._cond.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=2.0)

    def __enter__(self) -> "StackChanFaceEvents":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
