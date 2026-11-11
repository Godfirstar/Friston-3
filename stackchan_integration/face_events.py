"""Single-owner USB/TCP scheduler for Friston-3 expressions, audio and sensors.

Only this worker opens the device link. Camera and audio transactions are
serialized, so binary image/audio bytes cannot be mistaken for text responses.
"""

from __future__ import annotations

# main.py/preview.py创建本模块对象；语音输入、语音输出、摄像头适配器都复用它。
# 它是设备调度层：调用方提交任务，唯一stackchan-hardware线程调用Bridge执行协议。
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
    # dataclass自动生成构造函数；_submit把一次请求包装为此对象，在调用方与owner之间共享。
    operation: str
    arguments: dict
    deadline: float
    # operation是操作名，arguments是实参字典；deadline是排队+执行共用的绝对截止时间。
    session_epoch: Optional[int] = None
    motion_stamp: Optional[tuple[int, int, str]] = None
    # session_epoch绑定设备连接版本；motion_stamp绑定(对话轮次,修订号,阶段)，两者用途不同。
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    # default_factory为每个job创建独立Event；不能把同一个Event实例用作所有任务默认值。
    result: object = None
    error: Optional[Exception] = None
    # owner写result/error后set done；提交方等待done，再返回结果或抛出相应异常。


class StackChanFaceEvents:
    """Keep one connection owner for the lifetime of the application.

    Expressions are coalesced; all sensor/audio requests have a bounded queue
    lifetime. An expired request cannot unexpectedly run later. Cancellation
    is handled by the owner, never by a second thread closing/reading its port.
    """

    def __init__(self, port: Optional[str] = None, enabled: Optional[bool] = None, *,
                 transport: Optional[str] = None, host: Optional[str] = None,
                 tcp_port: Optional[int] = None):
        # self是当前设备管理器对象；port/transport来自调用参数或runtime_settings.SETTINGS。
        # Resolve configuration before starting the worker, without opening a
        # connection. Invalid settings then reach main.py's visible error path.
        selected = transport or ("usb" if port and not port.lower().startswith("tcp://")
                                 else "wifi" if port else SETTINGS.get("connection_preference", "auto"))
        self._bridge = StackChanBridge(
            port=port, transport=selected, host=host, tcp_port=tcp_port
        )
        # Bridge构造只解析配置；下方工作线程_run才调用connect打开实际设备。
        self.port = self._bridge.port
        self.transport = self._bridge.transport
        self.enabled = True if enabled is None else bool(enabled)
        self._cond = threading.Condition()
        # Condition默认使用可重入锁；with负责保护共享字段，wait等待时释放锁，notify唤醒等待者。
        self._motion = MotionPolicy(self._cond)
        # MotionPolicy定义在motion_events.py；共用这把锁，让换轮、队列变更和动作状态保持一致。
        self._connected = False
        self._connection_error = "正在自动发现栈酱"
        self._desired_preference = self._bridge.preference
        self._preference_dirty = False
        self._connection_epoch = 0
        self._wanted = "idle"
        self._version = 1
        # 表情只保留最新wanted和版本号，不为每次set_face排长队；旧表情可被新表情合并替代。
        self._face_hold_until = 0.0
        self._face_hold_seconds = 6.0
        self._closed = False
        self._jobs: deque[_DeviceJob] = deque()
        # deque是标准库双端队列；普通任务append排尾，owner用popleft取头部。
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
            # target传入方法本身（没有括号）；start才让_run在新线程执行。
            self._worker.start()

    def connection_status(self) -> dict:
        """A thread-safe snapshot; checking status never probes another port."""
        with self._cond:
            # 返回缓存状态dict，不为查询另开连接。此处self是管理器，不是底层Bridge。
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
        # 设备设置技能调用；先保存用户偏好，再通知owner在事务边界切换，不由调用者抢占socket。
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
                # 等待时释放锁，让owner能够更新_connected；短间隔也允许主线程处理Ctrl+C。
            return self._connected and not self._closed

    def set_face(self, face: str) -> None:
        # main.py及本类状态回调调用；只更新目标表情与版本，不在这里发送FACE命令。
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
        # main.py开始处理一轮用户输入时调用：旧装饰动作失效，新轮进入thinking。
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
            # main.py拿到Agent结果后调用。result是技能结果dict，不直接把回复文本当动作代码执行。
            self._discard_queued_motion_locked()
            presentation = self._motion.prepare_reply(
                result, self._connection_epoch if self._connected else -1)
            if presentation is None:
                return
            face, stop_old_motion = presentation
            # 元组解包得到表情str及是否停止旧动作bool；回复动作先暂存，首段语音时才下发。
            self._face_hold_until = 0.0
            if stop_old_motion:
                # The answer can explicitly request stillness. A THINK cue
                # already sent while planning must not linger into that reply.
                self._queue_stop_motion_locked()
        self.set_face(face)

    def on_reply_ready(self) -> None:
        # main.py在本轮呈现结束后调用；完成动作状态，表情从此时开始保留一段时间再回idle。
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
        # 只筛掉尚未执行的装饰动作；保留相机/麦克风/音频与显式stop，避免误删业务请求。
        for job in self._jobs:
            if job.operation in ("motion_event", "reply_gesture"):
                job.cancelled.set()
                job.result = {"status": "skipped", "reason": "superseded"}
                job.done.set()
            else:
                kept.append(job)
        self._jobs = kept

    def _queue_stop_motion_locked(self) -> None:
        # prepare_reply需要静止时调用；方法名locked表明调用者已经持有_cond。
        if not self.enabled or not self._connected or self._closed:
            return
        if any(job.operation == "motion_stop" for job in self._jobs):
            return
        # Stop is bounded and uses the same owner. Putting it before queued
        # jobs never interrupts a binary transaction already in progress.
        self._jobs.appendleft(_DeviceJob("motion_stop", {}, time.monotonic() + 4.0,
                                        session_epoch=self._connection_epoch))
        # appendleft放到排队头部，但不会打断owner当前正在读写的二进制事务。
        self._cond.notify_all()

    def _queue_state_motion(self, gesture: str, phase: str, *, cooldown: float) -> None:
        """No wait in the caller: short-lived state cues cannot delay planning."""
        with self._cond:
            now = time.monotonic()
            if (not self.enabled or self._closed or not self._connected
                    or len(self._jobs) >= 7):
                return
            stamp = self._motion.state_cue(phase, cooldown)
            # state_cue定义在MotionPolicy；返回本轮状态快照或None，后者表示本次不需要动作。
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
        # 本类get_info/capture_jpeg/record_pcm/play_pcm等公共方法共用；返回具体任务的结果。
        # **arguments把任意关键字实参收进dict，例如width=320变为{"width":320}。
        if not self.enabled:
            raise RuntimeError("栈酱设备管理器未启用")
        job = _DeviceJob(operation, arguments, time.monotonic() + timeout,
                         session_epoch=session_epoch)
        # 创建内存任务对象；调用方随后等待，不直接与设备进行任何read/write。
        with self._cond:
            if self._closed:
                raise RuntimeError("StackChan device manager is closed")
            if not self._connected:
                raise ConnectionError("栈酱尚未连接，正在自动恢复；本次设备请求未执行")
            if session_epoch is not None and session_epoch != self._connection_epoch:
                raise ConnectionError("设备连接已中断，本轮旧语音已丢弃；不会在重连后补播")
            job.session_epoch = self._connection_epoch if session_epoch is None else session_epoch
            # 没显式传版本就绑定当前连接；重连后旧语音不能被当作新连接的待播内容。
            if operation in ("audio", "reply_gesture"):
                job.motion_stamp = self._motion.stamp
            if len(self._jobs) >= 8:
                raise RuntimeError("StackChan request queue is full")
            self._jobs.append(job)
            self._cond.notify()
            # 入队和通知在同一把锁下完成；owner在拿到锁后能看到完整任务对象。
        try:
            completed = False
            # Short waits let Windows Python dispatch Ctrl+C while a hardware
            # transaction is pending; use the original deadline for expiry.
            while not completed:
                remaining = job.deadline - time.monotonic()
                if remaining <= 0:
                    break
                completed = job.done.wait(timeout=min(0.1, remaining))
                # Event.wait返回bool：True表示已set，False只是本次短等待到期，不一定是总超时。
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
            # 把owner线程中捕获的异常转回提交方线程；from保留原始异常作为原因，便于排查。
            if isinstance(job.error, ConnectionError):
                raise ConnectionError(f"StackChan {operation}: {job.error}") from job.error
            if isinstance(job.error, TimeoutError):
                raise TimeoutError(f"StackChan {operation}: {job.error}") from job.error
            raise RuntimeError(f"StackChan {operation} failed: {job.error}") from job.error
        return job.result

    def _cancel(self, job: _DeviceJob) -> None:
        # _submit等待超时或被Ctrl+C中断时调用；仅改任务标记，绝不从第二线程关闭串口。
        with self._cond:
            job.cancelled.set()
            try:
                self._jobs.remove(job)
                # 还在队列就移除；若owner已取走，remove会抛ValueError，靠cancelled通知正在执行者。
            except ValueError:
                # The connection owner observes cancellation during active I/O.
                pass
            else:
                job.error = RuntimeError("StackChan request cancelled before execution")
                job.done.set()
            self._cond.notify_all()

    def get_info(self) -> str:
        # 提交info任务；最终由_execute_core调用Bridge.info，返回固件能力字符串。
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
        # friston3.hardware.camera中的StackChan相机适配器调用；返回JPEG压缩字节，不是像素数组。
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
        # voice_input.py调用，返回PCM16LE音频bytes；真正录音与下载在owner执行的microphone任务中。
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
        # voice_output.py调用；用原始PCM长度推算时长，超时预算包含排队、上传与完整播放。
        timeout = max(30.0, min(180.0, duration * 16.0 + 20.0))
        self._submit("audio", timeout=timeout, session_epoch=session_epoch,
                     pcm=pcm, volume=volume)
        return True

    # Motion, speech and sensors share the same transaction worker.
    def on_task_completed(self, result) -> None:
        # main.py呈现任务结果时调用；给静音/纯文字回复提供一次动作机会，不覆盖原业务结果。
        if not self.enabled or not self._motion.needs_reply_gesture(result):
            return
        try:
            self._submit("reply_gesture", timeout=4.0)
        except Exception as error:
            # 不重放不确定的动作回执，也不覆盖原任务结果。
            LOG.warning("表演动作未确认，保留原任务结果：%s", error)

    def _execute(self, bridge: StackChanBridge, job: _DeviceJob):
        # 只由本类_run调用；bridge是共享协议对象，job为从队列取出的任务。
        # 本层先处理动作状态/相机占用，再交给_execute_core完成具体设备事务。
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
        # operation分流到stackchan_bridge.py的方法；这些调用均发生在同一个owner线程。
        if job.operation == "info":
            return bridge.info()
        if job.operation == "move_head":
            return bridge.move_head(**job.arguments)
            # **dict把保存的参数字典重新展开为命名实参，与_submit收集参数的过程相反。
        if job.operation == "home_head":
            return bridge.home_head()
        if job.operation == "stop_head":
            return bridge.stop_head()
        if job.operation == "camera":
            result = bridge.capture_jpeg(**job.arguments)
            self.last_camera_timing = dict(bridge.last_camera_timing)
            return result
        if job.operation == "identity_camera":
            # 身份识别抓拍是“捕获+释放相机”一个完整任务，避免后续MIC/TTS被相机占用影响。
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
            # 首段语音前准备回复动作；V2的speech命令先暂存，固件在AUDIO STARTED时触发。
            bridge.set_volume(job.arguments["volume"])
            bridge.play_pcm(job.arguments["pcm"])
            self.last_audio_timing = dict(bridge.last_audio_timing)
            return True
        raise ValueError(f"Unknown StackChan operation: {job.operation}")

    def _invalidate_connection(self, error) -> None:
        # owner连接失败/切换时调用；连接版本递增，使旧传感器缓存、旧音频请求失效。
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
                # 给所有被丢弃任务设置error并完成通知，防止提交方一直等一个永远不会执行的请求。
            self._cond.notify_all()

    def _mark_connected(self, bridge) -> None:
        # owner完成Bridge.connect握手后调用；同步端点与能力状态，通知wait_ready调用方。
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
        # __init__创建的stackchan-hardware线程入口；整个连接的open/read/write/close都由它执行。
        bridge = self._bridge
        delivered_revision = 0
        # delivered_revision是已发到设备的表情版本；与self._version比较就知道是否需要更新表情。
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
                    # maintain决定本轮是否做连接/心跳/偏好维护；与具体用户任务截止时间分开。
                    if not maintain and not self._jobs and self._version == delivered_revision:
                        # A timed idle wakeup is essential: plugging USB in must
                        # be discovered without an unrelated face/audio request.
                        self._cond.wait(timeout=min(0.25, max(0.01, next_probe - now)))
                        continue
                    preference, preference_dirty = self._desired_preference, self._preference_dirty
                    can_handoff = self._motion.can_handoff
                    # 离开with后才做慢I/O；否则其他线程连提交取消或查询状态都得等待网络收包。

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
                            # 这里只在完整事务之间切换连接，不会把一张JPEG前半段从USB读、后半段从TCP读。
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
                        # 不传参数恢复为空上下文，避免把连接维护阶段的截止时间留给下一项业务。

                with self._cond:
                    if self._closed:
                        break
                    self._expire_jobs(time.monotonic())
                    job = self._jobs.popleft() if self._jobs else None
                    # 从左边取出最早排队项；空队列时job=None，仍可单独同步新表情。
                    self._active_job = job
                    revision = self._version
                    face = self._wanted if revision != delivered_revision else None
                if job is None and face is None:
                    continue
                bridge._set_operation(
                    job.cancelled if job else self._stop_event,
                    job.deadline if job else time.monotonic() + 5.0,
                )
                # 当前job的取消Event与总截止时间传入Bridge，底层每次收发轮询都会检查。
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
                        # 同步完成一整项事务再取下一个；保证音频/图片二进制不与FACE文本交错。
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
                            # 不论成功还是异常，owner都发布完成通知；_submit醒来检查result/error。
                        self._cond.notify_all()
        finally:
            # owner循环退出时通知队列里的等待者，然后由owner自己关闭底层连接。
            with self._cond:
                self._connected = False
                while self._jobs:
                    job = self._jobs.popleft()
                    job.error = ConnectionError("设备管理器已关闭")
                    job.done.set()
                self._cond.notify_all()
            bridge.close()

    def close(self) -> None:
        # main.py/preview.py退出时调用；发取消和退出通知，等待owner清理，不从此处直接抢关端口。
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
            # join等待线程结束；超时不会强制杀线程，当前收发由其取消检查/截止时间结束。

    def __enter__(self) -> "StackChanFaceEvents":
        # 支持with设备管理器as device；__exit__保证离开with时调用close。
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
