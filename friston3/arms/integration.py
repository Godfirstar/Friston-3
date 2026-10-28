"""Optional, nonblocking dialogue gestures over the local arm service.

This module owns no serial port and imports no robotics/model dependency.
The service remains the authority for enabling, limits and motion completion.
"""
from __future__ import annotations

import threading
import time

from .config import GESTURES
from friston3.runtime_settings import SETTINGS


_STOP_COMMANDS = frozenset((
    "停止", "停下", "急停", "紧急停止", "停止机械臂", "机械臂停止",
    "停止双臂", "双臂停止", "停下机械臂", "/arms stop",
))


def is_immediate_stop(instruction: str) -> bool:
    """An explicit stop protocol, never a keyword-based gesture classifier."""
    return (isinstance(instruction, str)
            and instruction.strip().rstrip("。！？!?.,，；; ").lower() in _STOP_COMMANDS)


def _job_text(job) -> str:
    if not isinstance(job, dict):
        return "无"
    states = {"accepted": "已接受，尚未确认完成", "running": "执行中",
              "completed": "已完成", "cancelled": "已取消", "failed": "失败"}
    state = str(job.get("status", "unknown"))
    summary = f"{job.get('name') or job.get('kind') or '任务'}：{states.get(state, state)}"
    if job.get("id"):
        summary += f"（{job['id']}）"
    if job.get("message"):
        summary += f"；{job['message']}"
    return summary


def describe_status(result: dict) -> str:
    """Retain the structured snapshot while giving the user a useful summary."""
    if result.get("success") is False:
        return f"机械臂状态={result.get('status', 'unknown')}；{result.get('message', '状态未取得。')}"
    mode = "模拟服务（无真实机械臂）" if result.get("mode") == "mock" else "实机服务"
    sides = {"left": "左臂", "right": "右臂"}
    available = "、".join(sides.get(str(arm), str(arm))
                         for arm in result.get("available_arms", [])) or "无"
    status = str(result.get("status", "unknown"))
    states = {"starting": "正在连接", "stopping": "正在确认停止持姿", "idle": "就绪空闲",
              "running": "执行中", "accepted": "任务已接受", "ready": "就绪", "faulted": "故障"}
    summary = (f"{mode}；服务状态={states.get(status, status)}；配置={available}；"
            f"{'已使能' if result.get('armed') else '未使能'}；"
            f"对话联动={'开启' if result.get('auto_enabled') else '关闭'}；"
            f"{'忙碌' if result.get('busy') else '空闲'}。"
            f"当前任务：{_job_text(result.get('current_job'))}。"
            f"上一任务：{_job_text(result.get('last_job'))}。")
    hold = result.get("hold_result")
    if isinstance(hold, dict) and hold.get("message"):
        summary += f"停止持姿：{hold['message']}"
    return summary


def describe_motions(result: dict) -> str:
    if result.get("success") is False:
        return str(result.get("message", "动作列表未取得。"))
    items = result.get("motions", [])
    invalid = result.get("invalid_motions", [])
    errors = "；".join(f"{item.get('name', '?')}：{item.get('error', '无效')}"
                      for item in invalid if isinstance(item, dict))
    if not items:
        return ("没有可用的机械臂动作；无效记录：" + errors if errors else
                "尚未录制机械臂动作。请先在 lerobot 服务控制台示教。")
    labels = []
    for motion in items:
        if not isinstance(motion, dict):
            continue
        sides = "+".join(str(side) for side in motion.get("arms", []))
        mode = "可用于对话联动" if motion.get("looped") else "仅手动播放"
        labels.append(f"{motion.get('name', '?')}（{sides}，{motion.get('duration_s', '?')}秒，{mode}）")
    prefix = "模拟动作（无真实机械臂）：" if result.get("mode") == "mock" else "已录制动作："
    return prefix + "；".join(labels) + ("；无效记录：" + errors if errors else "")


class ArmIntegration:
    """Explicit operations are bounded; optional automatic gestures never wait.

    Only one automatic request may be in flight. There is no cross-turn motion
    queue, retry, automatic arming, or hardware startup. A fresh NPC result
    contributes at most one callback, consumed at the first audio clip (or at
    the text-only fallback).
    """

    def __init__(self, config_path=None, *, client_factory=None, notify=None):
        self.config_path = config_path
        self._client_factory = client_factory
        self._notify = notify or (lambda message: print(f"[Arms] {message}", flush=True))
        self._lock = threading.Lock()
        self._generation = 0
        self._inflight = False
        self._closed = False
        self._retry_after = 0.0
        self._worker = None
        self._last_job_id = None
        self._last_dialogue_job_id = None
        self._cleanup_ids = set()

    def _client(self):
        if self._client_factory is not None:
            return self._client_factory(self.config_path)
        from .client import ArmClient
        return ArmClient(self.config_path)

    def _request(self, operation: str, **arguments) -> dict:
        try:
            client = self._client()
            result = client.status() if operation == "status" else client.command(operation, **arguments)
            if not isinstance(result, dict):
                raise RuntimeError("机械臂服务未返回对象")
            if "mode" not in result and getattr(client, "mode", None) in ("real", "mock"):
                result = {**result, "mode": client.mode}
            return dict(result)
        except FileNotFoundError:
            return {"success": False, "status": "unconfigured",
                    "message": "机械臂服务尚未配置；请先在 lerobot 环境运行 python -m friston3.arms init。"}
        except (OSError, ValueError, RuntimeError) as error:
            return {"success": False, "status": "unavailable",
                    "message": f"机械臂服务未确认响应：{error}"}

    def cancel_pending(self) -> None:
        with self._lock:
            self._generation += 1

    def perform(self, operation: str, *, name: str = "") -> dict:
        """The planner can never arm, teach, or enable automatic execution."""
        if operation not in ("status", "list", "stop", "play"):
            raise ValueError("对话只开放 status、list、stop 和 play")
        if operation == "stop":
            self.cancel_pending()
        arguments = {"name": name, "source": "manual"} if operation == "play" else {}
        result = self._request(operation, **arguments)
        result["operation"] = operation
        if operation == "play":
            self._remember_job(result)
        elif operation == "stop" and result.get("success") is True:
            with self._lock:
                self._last_job_id = None
                self._last_dialogue_job_id = None
        if operation in ("status", "list"):
            if "message" in result:
                result["service_message"] = result["message"]
            result["message"] = describe_status(result) if operation == "status" else describe_motions(result)
        elif operation == "play" and result.get("status") in ("accepted", "running"):
            result["service_message"] = result.get("message", "")
            result["message"] = (f"已接受动作 {name}，实际执行结果尚未确认；"
                                 "可查询机械臂状态查看进度。")
            if result.get("mode") == "mock":
                result["message"] = "模拟服务（无真实机械臂）：" + result["message"]
        # Query and acceptance are not physical completion. Preserve status,
        # current_job, last_job, positions and every other service field.
        result["speak_confirmation"] = True
        return result

    def _remember_job(self, result: dict, *, dialogue=False) -> None:
        job_id = result.get("job_id")
        if (result.get("success") is True and result.get("status") in ("accepted", "running")
                and isinstance(job_id, str) and job_id):
            with self._lock:
                self._last_job_id = job_id
                if dialogue:
                    self._last_dialogue_job_id = job_id

    def _stop_owned(self, job_id: str):
        with self._lock:
            if job_id in self._cleanup_ids:
                return None
            self._cleanup_ids.add(job_id)
        return self._request("stop", job_id=job_id)

    def cancel_dialogue(self):
        """An explicit performance-off choice cancels only our automatic job."""
        self.cancel_pending()
        with self._lock:
            job_id = self._last_dialogue_job_id
        return self._stop_owned(job_id) if job_id else None

    def reply_start(self, result: dict, *, allow: bool = True):
        """Return a once-only callback for this successful, fresh NPC reply."""
        with self._lock:
            self._generation += 1
            generation = self._generation
        if (not allow or not SETTINGS.get("gestures_enabled", True) or not isinstance(result, dict)
                or result.get("executed_skill") != "npc_talk"
                or result.get("success") is not True):
            return None
        presentation = result.get("presentation")
        name = presentation.get("arm_gesture") if isinstance(presentation, dict) else None
        if not isinstance(name, str) or name not in GESTURES:
            return None
        called = False
        created_at = time.monotonic()

        def start_once():
            nonlocal called
            with self._lock:
                if called:
                    return
                called = True
                if (self._closed or generation != self._generation or self._inflight
                        or not SETTINGS.get("gestures_enabled", True)
                        or time.monotonic() < self._retry_after
                        or time.monotonic() - created_at > 15.0):
                    return
                self._inflight = True
                self._worker = threading.Thread(target=self._automatic, args=(name, generation),
                                                name="friston-arm-gesture", daemon=True)
                self._worker.start()

        return start_once

    def _automatic(self, name: str, generation: int) -> None:
        try:
            with self._lock:
                if (self._closed or generation != self._generation
                        or not SETTINGS.get("gestures_enabled", True)):
                    return
            result = self._request("play", name=name, source="dialogue")
            self._remember_job(result, dialogue=True)
            with self._lock:
                closed = self._closed
                stale = generation != self._generation
            if ((closed or not SETTINGS.get("gestures_enabled", True)) and result.get("job_id")
                    and result.get("status") in ("accepted", "running")):
                stopped = self._stop_owned(result["job_id"])
                if stopped is not None:
                    self._notify(str(stopped.get("message", "本轮动作停止未确认。")))
                return
            if stale:
                return
            if result.get("status") in ("unconfigured", "unavailable"):
                with self._lock:
                    self._retry_after = time.monotonic() + 15.0
            if result.get("status") == "unconfigured":
                return
            if result.get("status") in ("accepted", "running"):
                prefix = "模拟服务：" if result.get("mode") == "mock" else ""
                self._notify(f"{prefix}对话动作 {name} 已接受，完成状态以服务查询为准。")
            elif result.get("message"):
                self._notify(str(result["message"]))
        finally:
            with self._lock:
                self._inflight = False

    def close(self) -> None:
        """Cancel this integration's job without stopping later CLI work."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=1.0)
        with self._lock:
            job_id = self._last_job_id
            inflight = self._inflight
        if job_id:
            result = self._stop_owned(job_id)
            if result is not None:
                self._notify(str(result.get("message", "本轮动作停止未确认。")))
        if inflight:
            self._notify("退出时仍有机械臂请求等待确认；如需立即全局停止，请在独立终端运行 python -m friston3.arms stop。")
