"""回复表情与有限动作的状态机。

本模块只管理动作策略，不持有设备管理器、队列或串口。设备管理器传入
同一把锁，使换轮和动作选择保持同步；实际命令仍在原设备线程执行。
"""
from __future__ import annotations

# 上游是face_events.StackChanFaceEvents：它持有队列和线程，本文件只保存回复/动作状态。
# execute_*由设备owner线程调用；最终通过传入的StackChanBridge发送固件命令。
import logging
import time

from friston3.runtime_settings import SETTINGS
from .stackchan_bridge import StackChanCommandRejected

LOG = logging.getLogger(__name__)
_ALLOWED_SKILLS = {"take_photo", "auto_photo", "npc_talk", "control_volume", "configure_device"}
# set集合用于判断技能名是否支持回复表演；下面各dict是已有情绪/固件预设名称映射。
_EMOTION_FACES = {
    "neutral": "neutral", "warm": "happy", "amused": "happy",
    "proud": "proud", "thinking": "thinking", "curious": "doubt",
    "surprised": "surprised", "concerned": "sad", "sad": "sad",
    "firm": "angry", "sleepy": "sleepy",
}
_GESTURES = frozenset({
    "none", "auto", "nod", "agree", "shake", "tilt", "greet", "think",
    "proud", "chuckle", "surprised", "concerned", "bow", "look_owner",
    "look_left", "look_right", "scan",
})
# A little variation comes from the selected emotion and turn counter, never
# keyword matching in the reply. Neutral responses deliberately include rests.
_AUTO_GESTURES = {
    "neutral": ("nod", "none", "tilt", "none"),
    "warm": ("greet", "nod", "agree"),
    "amused": ("chuckle", "tilt"),
    "proud": ("proud", "nod"),
    "thinking": ("think", "tilt"),
    "curious": ("tilt", "think"),
    "surprised": ("surprised",),
    "concerned": ("concerned",),
    "sad": ("concerned", "bow"),
    "firm": ("nod",),
    "sleepy": ("settle",),
}
_LEGACY_GESTURES = {
    "none": "none", "nod": "nod", "agree": "nod", "shake": "shake",
    "tilt": "tilt", "greet": "nod", "think": "tilt", "proud": "nod",
    "chuckle": "nod", "surprised": "tilt", "concerned": "tilt", "bow": "nod",
    "look_left": "look_left", "look_right": "look_right", "scan": "shake",
    "attend": "tilt", "settle": "none", "beat": "none",
}


def reply_presentation(result) -> tuple[str, str]:
    """Accept only bounded metadata; task failures never add reply movement."""
    if not isinstance(result, dict) or result.get("success") is not True:
        # 输入为Agent/技能结果dict，输出(表情名,动作名)；不读设备、不创建线程。
        return "sad", "none"
    skill = result.get("executed_skill")
    if skill in ("take_photo", "auto_photo"):
        return "happy", "agree"
    if skill in ("control_volume", "configure_device"):
        if (skill == "configure_device" and result.get("setting") == "owner_side"
                and result.get("choice") in ("left", "right")):
            return "happy", "look_owner"
        return "neutral", "nod"
    if skill != "npc_talk":
        return "neutral", "none"
    presentation = result.get("presentation")
    # presentation来自NPC输出中的结构化呈现元数据；不是按回复中文关键词猜动作。
    if not isinstance(presentation, dict):
        return "neutral", "none"
    emotion, gesture = presentation.get("emotion"), presentation.get("gesture")
    face = _EMOTION_FACES.get(emotion, "neutral") if isinstance(emotion, str) else "neutral"
    return face, gesture if isinstance(gesture, str) and gesture in _GESTURES else "none"


class MotionPolicy:
    """集中管理换轮、回复动作、冷却和相机占用；不创建线程。

    lock 使用设备管理器的 Condition（默认 RLock），允许换轮时原子更新队列和状态。
    """

    def __init__(self, lock, settings=SETTINGS):
        # FaceEvents.__init__传入它的Condition；self是动作状态对象，settings提供用户运行设置。
        self._lock = lock
        self._settings = settings
        self._turn = 0
        self._revision = 0
        self._phase = "idle"
        # turn表示对话轮次，revision表示本轮阶段/意图修订号，phase为idle/thinking/reply等阶段。
        self._turn_active = False
        self._reply_epoch = -1
        # epoch是设备连接版本；断线前生成但尚未播出的语音不能在新连接上补播。
        self._reply_face = "idle"
        self._reply_prepared = False
        self._pending_gesture = "none"
        # pending只暂存待执行动作；prepare_reply本身不会给舵机发指令。
        self._camera_active = False
        # 与固件相机/头部共享资源的使用状态关联；由FaceEvents._execute更新，不由本类开关相机。
        self._gesture_faulted = False
        self._next_state_at = {"listening": 0.0, "thinking": 0.0}
        # 字典值是下一次允许该状态动作的monotonic时刻；是现有调度规则，不是新建限位。
        self._next_reply_at = self._next_beat_at = 0.0
        self._emotion = "neutral"
        self._variation = self._auto_counter = 0
        self._audio_segments = self._beats = 0
        # audio_segments记录本轮进入播放的分段数；beats记录后续语音小节拍次数。
        self._reply_motion_allowed = False
        self._owner_side_reported = False
        self._last_result = {}

    @property
    def stamp(self) -> tuple[int, int, str]:
        """队列只保存这张快照；换轮或阶段变化后，旧动作自动失效。"""
        with self._lock:
            return self._turn, self._revision, self._phase
            # @property允许写motion.stamp；返回不可变三元组，队列任务保存它作为有效性凭据。

    @property
    def playback_epoch(self) -> int:
        with self._lock:
            return self._reply_epoch

    @property
    def can_handoff(self) -> bool:
        # FaceEvents._run读取，判断此刻是否可在事务边界切换USB/TCP连接。
        with self._lock:
            return not (self._turn_active or self._reply_prepared or self._camera_active)

    @property
    def last_result(self) -> dict:
        with self._lock:
            return dict(self._last_result)
            # 返回新dict避免上层直接改内部结果；这里字段都是简单值，不保存设备对象。

    def begin_turn(self, epoch: int) -> None:
        # FaceEvents.on_request调用；轮次和修订号变化后，旧队列任务的stamp不再匹配。
        with self._lock:
            self._turn += 1
            self._revision += 1
            self._phase = "thinking"
            self._turn_active = True
            self._reply_epoch = epoch
            self._clear_reply()

    def _clear_reply(self) -> None:
        # 内部复用的重置方法；调用者已持同一把锁，不代表发送“停止舵机”的物理命令。
        self._reply_prepared = False
        self._pending_gesture = "none"
        self._audio_segments = self._beats = 0
        self._reply_motion_allowed = False

    def prepare_reply(self, result, epoch: int) -> tuple[str, bool] | None:
        """返回要显示的表情和是否需停止旧动作；旧连接的回复不再表演。"""
        face, gesture = reply_presentation(result)
        # 拆开纯函数返回的两个字符串；之后在锁下将其绑定到当前回复/连接。
        with self._lock:
            self._revision += 1
            self._phase = "reply"
            self._clear_reply()
            if epoch < 0 or self._reply_epoch != epoch:
                # 无有效连接或连接已换代时返回None，上层不再表演这段旧回复。
                return None
            self._reply_face = face
            self._reply_prepared = True
            self._pending_gesture = gesture
            presentation = result.get("presentation", {}) if isinstance(result, dict) else {}
            emotion = presentation.get("emotion") if isinstance(presentation, dict) else None
            self._emotion = emotion if isinstance(emotion, str) else "neutral"
            self._variation = self._auto_counter
            if gesture == "auto":
                self._auto_counter += 1
            return face, gesture == "none"
            # 第二个bool通知FaceEvents：回复明确要求不动时，需清理已经发出的thinking动作。

    def finish_reply(self) -> tuple[str, bool]:
        """播完后开始保留表情；此前不要让长回复中途失去表情。"""
        with self._lock:
            self._turn_active = False
            self._phase = "idle"
            self._revision += 1
            prepared = self._reply_prepared
            face = self._reply_face if prepared else "idle"
            self._clear_reply()
            return face, prepared
            # FaceEvents根据prepared决定是否开始保留表情计时，返回值不等于动作执行结果。

    def begin_listening(self) -> None:
        # FaceEvents.record_pcm调用；只改变空闲时的阶段，不抢占正在准备的回复。
        with self._lock:
            if not self._turn_active and not self._reply_prepared:
                self._phase = "listening"

    def cancel(self) -> None:
        # FaceEvents.cancel_motion调用；修订号使排队动作失效，实际停止另由owner排队执行。
        with self._lock:
            self._revision += 1
            self._pending_gesture = "none"
            self._reply_motion_allowed = False

    def connection_changed(self, *, connected: bool) -> None:
        # FaceEvents连接成功/失效时调用；*表示connected只能按名字传参，便于读清含义。
        with self._lock:
            if connected:
                self._gesture_faulted = False
                return
            self._turn_active = False
            self._revision += 1
            self._phase = "idle"
            self._clear_reply()
            self._reply_face = "idle"
            self._camera_active = False

    def camera_session(self, active: bool) -> None:
        with self._lock:
            self._camera_active = active

    def state_cue(self, phase: str, cooldown: float) -> tuple[int, int, str] | None:
        # _queue_state_motion调用；返回当前stamp或None，供队列决定要不要加入think/attend提示动作。
        with self._lock:
            now = time.monotonic()
            if (not self._motion_enabled() or self._phase != phase
                    or now < self._next_state_at[phase]):
                return None
            self._next_state_at[phase] = now + cooldown
            return self.stamp
            # 读取property时会再次进入同一把锁；Condition默认RLock允许同一线程重入。

    def needs_reply_gesture(self, result) -> bool:
        # FaceEvents.on_task_completed调用；判断静音/纯文字结果是否还有未消费的回复动作。
        with self._lock:
            return (self._settings.get("gestures_enabled", True)
                    and isinstance(result, dict) and result.get("success") is True
                    and result.get("executed_skill") in _ALLOWED_SKILLS
                    and self._pending_gesture != "none")

    def _motion_enabled(self) -> bool:
        # 内部谓词只返回bool，不操作硬件；综合用户开关、当前连接动作故障和相机占用。
        return (self._settings.get("gestures_enabled", True)
                and not self._gesture_faulted and not self._camera_active)

    def _result(self, **value):
        # **value收集关键字为dict；同时保存最近动作结果，供设备状态界面读取。
        with self._lock:
            self._last_result = dict(value)
        return value

    def _gesture_token(self, gesture: str) -> str:
        # 将auto/look_owner等抽象选择落到固件预设名；此函数仍不发送指令。
        if gesture == "auto":
            choices = _AUTO_GESTURES.get(self._emotion, _AUTO_GESTURES["neutral"])
            return choices[self._variation % len(choices)]
            # %取余让计数在候选列表内循环；同一回复已保存variation，所以分段不会反复重新抽取。
        if gesture == "look_owner":
            return {"left": "look_left", "right": "look_right"}.get(
                self._settings.get("owner_side", "none"), "none")
        return gesture

    def _send_preset(self, bridge, token, *, when="now"):
        # 仅owner经execute_*调用；bridge来自FaceEvents，类定义在stackchan_bridge.py。
        capabilities = bridge.info()
        if "GESTURE_V2" in capabilities:
            # INFO返回的能力字符串决定协议版本；V2可指定摆放方式和与首声同步的时机。
            selected = token
            options = {"placement": self._settings.get("placement", "shoulder"), "when": when}
        elif "GESTURE_V1" in capabilities:
            selected = _LEGACY_GESTURES.get(token, "none")
            options = {}
            when = "now"
            if selected == "none" and token != "none":
                return self._result(status="skipped", reason="preset_requires_v2", gesture=token)
        else:
            return self._result(status="skipped", reason="gesture_firmware_required")
        try:
            bridge.gesture(selected, **options)
            # **options把dict展开为placement/when等命名参数；此行才真正开始设备协议事务。
        except (StackChanCommandRejected, ValueError) as error:
            # 完整拒绝和参数问题可局部处理；连接/超时异常继续向上交给FaceEvents恢复连接。
            # 只有完整拒绝行能保留当前连接；不确定的 I/O 故障交给设备线程恢复。
            with self._lock:
                self._gesture_faulted = True
                self._reply_motion_allowed = False
            LOG.warning("表演预设被拒绝，已停用本连接的后续动作：%s", error)
            return self._result(status="rejected", gesture=selected, error=str(error))
        print(f"[StackChanMotion] {selected} / {when} 已接受；到位未校验。")
        return self._result(status="command_accepted", gesture=selected,
                            requested_gesture=token, when=when, verified=False)

    def execute_state(self, bridge, token: str, stamp):
        # FaceEvents._execute处理motion_event时调用；stamp来自入队当时的状态快照。
        with self._lock:
            if stamp != self.stamp:
                # 排队期间可能已经换轮；比较快照，避免旧thinking动作跑进新回复。
                return self._result(status="skipped", reason="superseded_state")
            if not self._motion_enabled():
                return self._result(status="skipped", reason="disabled_faulted_or_camera")
            if token not in {"attend", "think"}:
                return self._result(status="skipped", reason="invalid_state_preset")
        return self._send_preset(bridge, token)
        # 慢协议发送放到锁外，其他线程仍可更新取消/阶段；设备读写仍只有owner执行。

    def execute_reply(self, bridge, stamp, *, audio_bytes: int | None = None):
        """首段执行语义动作，后续最多添加两次有间隔的小幅节拍。"""
        now = time.monotonic()
        audio = audio_bytes is not None
        # audio_bytes由FaceEvents音频任务传原始PCM字节数；None表示纯文字/静音的动作机会。
        with self._lock:
            if stamp != self.stamp or self._phase != "reply" or not self._reply_prepared:
                return self._result(status="skipped", reason="superseded_reply")
            first = self._audio_segments == 0
            # 首段消费_pending_gesture，后续段只可能按既有规则产生少量beat，不重复完整语义动作。
            if audio:
                self._audio_segments += 1
            if not self._motion_enabled():
                self._pending_gesture = "none"
                self._reply_motion_allowed = False
                return self._result(status="skipped", reason="disabled_faulted_or_camera")
            if first:
                requested = self._pending_gesture
                self._pending_gesture = "none"
                token = self._gesture_token(requested)
                # requested是模型/技能给的抽象选择，token是最终固件预设，两者分开保留可解释性。
                self._reply_motion_allowed = token != "none"
                if token == "none":
                    if requested == "look_owner" and not self._owner_side_reported:
                        self._owner_side_reported = True
                        print("[StackChanMotion] 主人方向尚未设置，本轮不猜朝向。")
                    return self._result(status="skipped", reason="none_or_unset_owner_side")
                if requested == "auto" and now < self._next_reply_at:
                    self._reply_motion_allowed = False
                    return self._result(status="skipped", reason="reply_cooldown")
                self._next_reply_at = now + 4.0
                self._next_beat_at = now + 4.5
            elif (audio and self._reply_motion_allowed and self._beats < 2
                  and now >= self._next_beat_at and audio_bytes >= 38400):
                token = "beat"
                self._beats += 1
                self._next_beat_at = now + 4.5
            else:
                return self._result(status="skipped", reason="rest_between_speech_beats")
        result = self._send_preset(bridge, token, when="speech" if audio else "now")
        # speech让固件等AUDIO STARTED再动；now用于无语音的即时呈现，PC不靠sleep猜开始时刻。
        if first and result["status"] != "command_accepted":
            with self._lock:
                self._reply_motion_allowed = False
        return result

    def execute_stop(self, bridge):
        """关闭表演后仍可停止；相机占用期间不抢用其 I2C 会话。"""
        capabilities = bridge.info()
        # FaceEvents处理motion_stop时调用；关闭表演后仍可执行停止，不会自动重新打开旧连接。
        if "GESTURE_V1" in capabilities or "GESTURE_V2" in capabilities:
            return self._send_preset(bridge, "none")
        with self._lock:
            can_stop_head = "SERVO_MOTION_V1" in capabilities and not self._camera_active
        if can_stop_head:
            bridge.stop_head()
            return self._result(status="command_accepted", gesture="stop", verified=False)
        return self._result(status="skipped", reason="no_active_preset_capability")
