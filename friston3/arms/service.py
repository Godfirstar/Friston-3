"""机械臂服务：HTTP 线程接收请求，唯一控制线程持有串口。

入口为 serve -> ArmService.start -> _run；_Handler 只调用状态/命令接口。
普通轨迹交给 trajectories，策略任务交给 PolicySession 生成目标；两条路径
最终都由本模块的控制线程调用 hardware，不让模型线程直接操作机械臂。
"""
from __future__ import annotations

# 这些都是 Python 标准库，不是机器人框架：OrderedDict 保留插入顺序，
# deepcopy 复制嵌套对象，threading 提供线程，time 提供时钟与等待。
from collections import OrderedDict
from copy import deepcopy
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import time
import uuid

from .config import GESTURES, JOINTS, load_config
# 开头的点表示从同一个 arms 包导入；这里导入的轨迹函数定义在 trajectories.py。
from .trajectories import (check_compatibility, interpolate, load_motion, motion_catalog,
                           motion_summary, save_recording, validate_name, within_pose)


def _result(success, status, message, **extra):
    # **extra 把额外关键字实参收进字典；返回值里的 **extra 再把这些键值展开。
    # 调用如 _result(True, "accepted", "已接收", job_id="...")，返回普通 dict。
    return {"success": success, "status": status, "message": message, **extra}


class ArmService:
    """HTTP threads never call the hardware. There is no waiting action queue."""

    def __init__(self, config, hardware=None):
        # __init__ 在 ArmService(config) 创建对象时执行；self 是“本次创建的对象”。
        # config 来自 config.load_config，是 dict；hardware 可传入一个驱动对象。
        # self.xxx 是对象保存的字段，方法返回后仍存在；普通局部变量只供本次调用使用。
        # 可注入同一接口的 MockHardware；替换硬件实现即可离线验证调度，
        # 调用方不需要知道真实串口细节。构造驱动本身还不会打开串口。
        self.config = config
        if hardware is None:
            # None 表示未提供；延迟到真正构造服务时才导入驱动工厂。
            from .hardware import make_hardware
            hardware = make_hardware(config)
        self.hardware = hardware
        # RLock 保护共享任务/状态，Event 用来通知线程；两者职责不同。
        # 必须可重入，因为持锁的命令处理可能再调用同样持锁的 status/_finish。
        self._lock = threading.RLock()
        # Event 是一个可置位/清零的通知标志；set/clear/wait 都是标准库方法。
        # wake 叫醒控制循环，closing 通知退出，cancel 取消动作，startup 通知连接已尝试。
        self._wake = threading.Event()
        self._closing = threading.Event()
        self._cancel = threading.Event()
        self._startup = threading.Event()
        self._thread = None
        # 布尔值记录运行状态；空字符串表示暂无错误；空字典稍后填入实际读回的数据。
        self._ready = False
        self._fault = ""
        self._armed = False
        self._auto = False
        self._positions = {}
        self._profiles = {}
        # A reconnect may observe a pose left by native policy execution.
        # Each legacy job re-enters its original rules for its own arms only.
        self._native_policy_arms = set(config["arms"])
        self._observed_at = 0.0
        self._active = None
        # _active：当前任务 dict 或 None；_last_job：最近终态的公开副本。
        # _jobs：job_id -> 任务；_requests：request_id -> (请求指纹, 首次响应)。
        self._last_job = None
        self._jobs = OrderedDict()
        self._requests = OrderedDict()
        self._generation = 0
        self._need_hold = False
        self._hold_result = None
        self._policy = None
        self._shutdown_requested = False
        self._motions, self._invalid = motion_catalog(config)

    def start(self):
        # 由本文件 serve 调用；只启动后台线程，本方法不执行硬件连接。
        if self._thread is not None:
            raise RuntimeError("机械臂服务已经启动")
        self._thread = threading.Thread(target=self._run, name="friston-arm-owner", daemon=True)
        # target=self._run 传“方法本身”，没有括号；写 self._run() 会在当前线程立即运行。
        # daemon=True 表示这是后台线程；下行 start() 才让它开始执行 _run。
        self._thread.start()

    def wait_ready(self, timeout=10.0):
        # 供调用者/离线验证等待启动；Event.wait 返回是否在 timeout 秒内被 set。
        if not self._startup.wait(timeout):
            raise TimeoutError("机械臂连接仍未完成")
        with self._lock:
            # with 自动调用锁的进入/退出操作；即使内部 raise，也会释放锁。
            if not self._ready:
                raise RuntimeError(self._fault or "机械臂未连接")

    def _public_job(self, job):
        # 内部 job 还带模型/轨迹等细节；这里只挑允许给客户端看的字段并复制。
        if job is None:
            return None
        return deepcopy({key: job[key] for key in (
            "id", "kind", "name", "status", "message", "source", "elapsed_s",
            "duration_s", "hold_confirmed", "motion_saved", "limited", "arms",
            "instruction", "policy_status", "metrics", "scene", "task_success", "verification",
            "object_task_success_verified", "global_goal", "current_subtask",
            "termination_reason") if key in job})

    def status(self):
        # 返回控制线程最近更新的快照；HTTP 查询不会为了“最新”而自行访问串口。
        # deepcopy 避免调用方继续修改服务内部可变字典。
        with self._lock:
            # a if 条件 else b 是条件表达式；这里按优先级把多个内部标志归成一个状态名。
            status = ("shutting_down" if self._shutdown_requested else "faulted" if self._fault else "starting" if not self._ready else
                      "stopping" if self._need_hold else
                      self._active["status"] if self._active else "idle")
            return _result(not bool(self._fault), status,
                self._fault or ("模拟服务：不会控制真实机械臂。" if self.config["mode"] == "mock" else
                                 "机械臂服务已连接。" if self._ready else "正在连接机械臂。"),
                mode=self.config["mode"], armed=self._armed, auto_enabled=self._auto,
                policy_resident=(self._policy is not None or
                                 bool(self._active and self._active["kind"] == "task")),
                busy=self._active is not None or self._need_hold,
                ready=self._ready, stop_pending=self._need_hold,
                available_arms=list(self._profiles) if self._ready else [],
                # list(dict) 取字典的键；例如 ["right"]。positions 是“臂名 -> 六关节字典”。
                configured_arms=list(self.config["arms"]), positions=deepcopy(self._positions),
                observation_age_s=round(time.monotonic() - self._observed_at, 3) if self._observed_at else None,
                current_job=self._public_job(self._active), last_job=deepcopy(self._last_job),
                motions=deepcopy(self._motions), invalid_motions=deepcopy(self._invalid),
                hold_result=deepcopy(self._hold_result))

    def job_status(self, job_id):
        # _Handler.do_GET('/jobs/...') 调用；dict.get 找不到键时返回 None，不抛 KeyError。
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return _result(False, "not_found", "找不到该任务；记录最多保留最近64项。")
            return _result(True, job["status"], job["message"], job=self._public_job(job))

    def command(self, request):
        """Validate/deduplicate before publishing one job to the owner thread."""
        # 此方法由 HTTP 请求线程调用，只登记任务/取消意图，不执行硬件 I/O。
        if not isinstance(request, dict):
            # isinstance 判断对象类型；HTTP 的 JSON 也可能是列表/null，所以先辨认。
            return _result(False, "rejected", "请求必须是JSON对象")
        operation = request.get("op")
        request_id = request.get("request_id")
        # operation 是操作名 str；request_id 是客户端生成的 str，用来识别同一次提交。
        if not isinstance(request_id, str) or not 8 <= len(request_id) <= 80:
            return _result(False, "rejected", "请求缺少有效 request_id")
        try:
            fingerprint = json.dumps(request, sort_keys=True, ensure_ascii=True, allow_nan=False)
            # json.dumps 把 dict 转字符串；按键排序让内容相同的请求得到相同指纹。
        except (ValueError, TypeError):
            return _result(False, "rejected", "请求含不可序列化或非有限数值")
        with self._lock:
            if request_id in self._requests:
                original, answer = self._requests[request_id]
                # 元组解包：一次取出首次请求字符串 original 和首次响应 dict answer。
                return deepcopy(answer) if original == fingerprint else _result(
                    False, "rejected", "相同 request_id 不可用于不同命令")
            try:
                # self._command_locked 是下方定义的方法；它假定调用者已经持有 _lock。
                answer = self._command_locked(operation, request)
            except (OSError, ValueError, KeyError, TypeError) as error:
                answer = _result(False, "rejected", str(error))
            self._requests[request_id] = (fingerprint, deepcopy(answer))
            # Only bounded, in-memory metadata; never write logs to the project.
            while len(self._requests) > 256:
                # OrderedDict.popitem(last=False) 删除最早登记的项，不是删除磁盘文件。
                self._requests.popitem(last=False)
            return answer

    def _command_locked(self, operation, request):
        # 由 command 调用。输入操作名和请求 dict，输出统一响应 dict；登记任务后立即返回。
        fields = {
            "arm": set(), "stop": {"job_id"}, "auto": {"enabled"},
            "play": {"name", "source"}, "record": {"name", "arm", "duration_s"},
            "task": {"instruction", "arm"},
            "list": set(), "status": set(), "shutdown": set(),
        }
        if operation not in fields or set(request) - {"op", "request_id"} - fields[operation]:
            # set(dict) 得到键集合，集合相减筛出不允许的额外参数。
            raise ValueError("未知操作或多余参数")
        if operation == "status":
            return self.status()
        if operation == "list":
            self._motions, self._invalid = motion_catalog(self.config)
            return _result(True, "listed", "已读取本地示教动作。", mode=self.config["mode"],
                           motions=deepcopy(self._motions), invalid_motions=deepcopy(self._invalid))
        if operation == "shutdown":
            self._shutdown_requested = True
            self._request_stop_locked()
            return _result(True, "shutting_down", "服务关闭请求已接收；停止当前任务后断开机械臂串口。",
                           mode=self.config["mode"])
        if operation == "stop":
            specified = request.get("job_id")
            # 指定 job_id 只取消那一项；未指定则表示操作整个服务当前动作。
            if specified is not None:
                if not isinstance(specified, str):
                    raise ValueError("job_id 必须是字符串")
                if not self._active or self._active["id"] != specified:
                    return _result(True, "no_op", "该任务已不在执行；未干预其他任务。")
            return self._request_stop_locked()
        if self._shutdown_requested:
            raise ValueError("机械臂服务正在关闭")
        if not self._ready or self._fault or self._closing.is_set():
            raise ValueError(self._fault or "机械臂尚未准备好")
        if self._need_hold:
            raise ValueError("正在确认停止持姿，请稍后查询状态")
        if operation == "arm":
            if self._active:
                raise ValueError("已有任务正在执行")
            self._armed = True
            self._cancel.clear()
            return _result(True, "armed", "已允许手动动作；自动对话动作仍由 auto 设置控制。")
        if operation == "auto":
            enabled = request.get("enabled")
            if type(enabled) is not bool:
                raise ValueError("enabled 必须是布尔值")
            if enabled and not self._armed:
                raise ValueError("先在本机运行 arm 使能动作")
            self._auto = enabled
            if not enabled and self._active and self._active.get("source") == "dialogue":
                return self._request_stop_locked()
            return _result(True, "configured", "自动对话动作已" + ("开启。" if enabled else "关闭。"))
        if operation != "task" and not self._armed:
            raise ValueError("动作未使能；在机械臂终端运行 python -m friston3.arms arm")
        if self._active:
            return _result(False, "busy", "机械臂正在执行任务，本次动作未排队。",
                           job_id=self._active["id"])
        if operation == "task":
            instruction = request.get("instruction")
            # instruction：完整任务文字；arm："left"/"right"，和物体在画面哪边不同。
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("任务需要完整的自然语言 instruction")
            arm = request.get("arm")
            if arm is None and len(self.config["arms"]) == 1:
                # iter(dict) 逐个给出键；只有一个臂时 next(...) 取出唯一臂名。
                arm = next(iter(self.config["arms"]))
            if arm not in self.config["arms"]:
                raise ValueError("请用 arm 指定执行任务的已配置机械臂：" + "、".join(self.config["arms"]))
            name = "policy_task"
        else:
            name = validate_name(request.get("name"))
        now = time.monotonic()
        # monotonic 是只用于计算时间间隔的单调时钟，不会因系统校时倒退。
        if now - self._observed_at > max(0.5, 3.0 / self.config["fps"]):
            raise ValueError("机械臂状态过旧，拒绝开始动作")
        job = {"id": uuid.uuid4().hex, "kind": operation, "name": name,
               "status": "accepted", "message": "任务已接收，尚未完成。", "source": "manual",
               "generation": self._generation, "accepted_at": now, "elapsed_s": 0.0}
        # job 是一个 dict；保存接收时的 generation，后来可判断它是否已被取消。
        # uuid.uuid4().hex 生成任务编号字符串；它不同于标识“本次请求”的 request_id。
        if operation == "task":
            job.update(instruction=instruction, global_goal=None, arm=arm, arms=[arm],
                       object_task_success_verified=False)
        elif operation == "play":
            source = request.get("source", "manual")
            if source not in ("manual", "dialogue"):
                raise ValueError("source 只支持 manual 或 dialogue")
            if source == "dialogue" and (not self._auto or name not in GESTURES):
                return _result(False, "disabled", "自动对话动作未开启，或动作未登记为对话预设。")
            motion = load_motion(self.config["motions_dir"], name, self.config["max_duration_s"])
            check_compatibility(motion, self._profiles, self.config)
            if not within_pose(self._positions, motion["frames"][0]["positions"], self.config):
                raise ValueError("当前姿态离录制起点较远；请用遥操作恢复到起点后再播放。未自动移动到起点。")
            if source == "dialogue" and not motion_summary(motion, self.config)["looped"]:
                raise ValueError("自动对话动作必须回到接近录制起点的姿态；请重新录制首尾闭合的动作")
            job.update(motion=motion, source=source, duration_s=motion["frames"][-1]["t"],
                       arms=list(motion["profiles"]))
        elif operation == "record":
            arm, duration = request.get("arm"), request.get("duration_s", 8.0)
            if arm not in self.config["arms"]:
                raise ValueError("示教所选从臂未配置")
            if (type(duration) not in (int, float) or not math.isfinite(duration)
                    or not 1 <= duration <= self.config["max_duration_s"]):
                raise ValueError("示教时长必须为1..30秒且不超过配置上限")
            if not self.config.get("leader"):
                raise ValueError("配置中没有 leader；先填写已标定主臂的 port 和 id")
            if (Path(self.config["motions_dir"]) / (name + ".json")).exists():
                raise ValueError("同名动作已存在，未覆盖；请换一个名称或先手动备份并移走旧文件")
            job.update(arm=arm, arms=[arm], duration_s=float(duration), frames=[], limited=False)
        if operation == "task":
            self._native_policy_arms.add(arm)
        else:
            try:
                self._validate_legacy_positions(self._positions, job["arms"])
            except RuntimeError as error:
                raise ValueError(str(error)) from error
            if operation == "record":
                self._auto = False
        self._active = job
        # 这里只有一个活动任务槽，busy 时拒绝新任务；不是无限排队的任务队列。
        # PolicySession 内部的 action deque 是“同一任务的动作帧”，不要混淆。
        self._jobs[job["id"]] = job
        # 这里保存的是同一个 job 对象，owner 后续更新它时，按 ID 查询也能看到进度。
        while len(self._jobs) > 64:
            self._jobs.popitem(last=False)
        self._cancel.clear()
        self._wake.set()
        return _result(True, "accepted", "动作任务已接收，执行结果请按 job_id 查询。",
                       job_id=job["id"], mode=self.config["mode"])

    def _request_stop_locked(self):
        self._armed = False
        self._auto = False
        # 递增版本使 owner 已取出的旧 job 失效；此处不强杀线程，也不写串口。
        self._generation += 1
        self._cancel.set()
        if self._fault and not self._ready:
            self._need_hold = False
            self._hold_result = _result(False, "unconfirmed", "控制连接已故障，无法确认持姿。",
                                        hold_confirmed=False)
            return deepcopy(self._hold_result)
        self._need_hold = True
        self._hold_result = None
        if self._active:
            self._active.update(status="cancelling", message="已取消后续动作，正在确认持姿。")
        self._wake.set()
        return _result(True, "stop_requested", "停止请求已接收；未回起点、未卸力，等待执行器确认持姿。",
                       hold_confirmed=False)

    def _validate_legacy_positions(self, positions, arms):
        for arm in arms:
            for joint, number in positions[arm].items():
                low, high = self._profiles[arm]["limits"][joint]
                if not low - 0.01 <= number <= high + 0.01:
                    raise RuntimeError(f"{arm}.{joint} 实测超出已标定范围")

    def _observe(self):
        # 仅由 owner 调用；返回 {臂名: {关节名: float}}，同时更新服务缓存。
        positions = self.hardware.read()
        # self.hardware.read 的定义位于 hardware.py，实际对象决定调用 real 还是 mock 实现。
        if set(positions) != set(self.config["arms"]):
            raise RuntimeError("机械臂状态缺失")
        for arm, pose in positions.items():
            if (set(pose) != set(JOINTS) or any(type(v) not in (int, float) or not math.isfinite(v)
                                              for v in pose.values())):
                raise RuntimeError(f"{arm} 状态包含无效关节数值")
        with self._lock:
            legacy_arms = set(positions) - self._native_policy_arms
        self._validate_legacy_positions(positions, legacy_arms)
        with self._lock:
            self._positions = positions
            self._observed_at = time.monotonic()
        return positions

    def _finish(self, job, status, message, **extra):
        # owner 在完成/取消/失败时调用；更新共享 job，并把当前任务槽恢复为 None。
        with self._lock:
            job.update(status=status, message=message, **extra)
            if job["kind"] == "task" and status in ("failed", "cancelled"):
                job.update(termination_reason=status, task_success=None)
            self._last_job = self._public_job(job)
            if self._active is job:
                # is 比较“是否同一个对象”；不是 == 那样比较两个字典内容是否相同。
                self._active = None
        print(f"[Arms] {job['name']}: {status} — {message}", flush=True)

    def _hold(self):
        # Called only by the serial owner; no stale cached pose is substituted.
        positions = self._observe()
        with self._lock:
            # 字典推导式逐项筛选：native/legacy 都是“臂名 -> 姿态”的新字典。
            native = {arm: pose for arm, pose in positions.items() if arm in self._native_policy_arms}
            legacy = {arm: pose for arm, pose in positions.items() if arm not in self._native_policy_arms}
        if native:
            self.hardware.send_policy(native)
        if legacy:
            self.hardware.send(legacy)
        return _result(True, "hold_command_sent", "已用当前实测姿态下发持姿目标；持续监控连接。",
                       hold_confirmed=True)

    def _cancel_current(self):
        # _run 看到 _need_hold 后调用；从取消标记走到实际下发持姿的过程发生在 owner。
        with self._lock:
            job = self._active
        if job and job["kind"] == "task" and "started_at" in job and self._policy is not None:
            # 策略有自己的版本号：清空待执行帧，并使仍在计算的旧结果失效。
            self._policy.stop()
            self._update_policy_job(job)
        try:
            result = self._hold()
        except Exception as error:
            result = _result(False, "unconfirmed", f"停止下发未确认：{type(error).__name__}: {error}",
                             hold_confirmed=False)
            with self._lock:
                self._fault = result["message"]
                self._ready = False
        try:
            self.hardware.close_leader()
        except Exception as error:
            result["leader_close_error"] = str(error)
        with self._lock:
            self._hold_result = result
            self._need_hold = False
            if job:
                self._finish(job, "cancelled" if result["success"] else "failed",
                             result["message"], hold_confirmed=result["hold_confirmed"])

    def _start_job(self, job, positions):
        # _tick_job 第一次遇到 accepted 状态时调用；positions 是这一轮 _observe 的结果。
        with self._lock:
            if self._cancel.is_set() or job["generation"] != self._generation or self._closing.is_set():
                return
        if job["kind"] == "task":
            if self._policy is None:
                from .policy import PolicySession
                # 第一次任务才创建推理线程；后续任务复用同一模型，避免反复冷加载。
                self._policy = PolicySession(self.config)
            self._policy.start(job["instruction"], job["arm"], deepcopy(positions[job["arm"]]))
            # PolicySession.start 只发布任务和观察给模型线程，不在此处同步完成模型推理。
            with self._lock:
                job["started_at"] = time.monotonic()
                job.update(status="running", message="任务正在加载策略与场景；尚未确认取物结果。")
            self._update_policy_job(job)
            return
        self._validate_legacy_positions(positions, job["arms"])
        with self._lock:
            self._native_policy_arms.difference_update(job["arms"])
        if job["kind"] == "record":
            self.hardware.connect_leader()
            leader_start = self.hardware.read_leader()
            positions = self._observe()
            job["leader_start"] = leader_start
            job["follower_start"] = deepcopy(positions[job["arm"]])
            job["frames"] = [{"t": 0.0, "positions": {job["arm"]: deepcopy(positions[job["arm"]])}}]
        else:
            if not within_pose(positions, job["motion"]["frames"][0]["positions"], self.config):
                raise RuntimeError("执行前姿态已改变，动作未开始")
        with self._lock:
            if self._cancel.is_set() or job["generation"] != self._generation:
                return
            job["reference"] = {arm: deepcopy(positions[arm]) for arm in job["arms"]}
            job["started_at"] = time.monotonic()
            job["last_tick"] = job["started_at"]
            job["tracking_bad_since"] = None
            message = (f"正在录制 {job['arm']}，请移动主臂，并在 {job['duration_s']:g} 秒内回到起始姿态。"
                       if job["kind"] == "record" else "正在执行；尚未确认完成。")
            job.update(status="running", message=message)
        print(f"[Arms] {job['name']}: running ({job['duration_s']:g}s)", flush=True)

    def _update_policy_job(self, job):
        # 从 policy.PolicySession.status 获取模型线程的快照，再挑字段同步到服务 job。
        snapshot = self._policy.status()
        with self._lock:
            for key in ("metrics", "scene", "instruction", "task_success", "verification",
                        "global_goal", "current_subtask", "termination_reason"):
                if key in snapshot:
                    job[key] = deepcopy(snapshot[key])
            job["policy_status"] = snapshot.get("status", "running")
            if snapshot.get("message"):
                job["message"] = str(snapshot["message"])
        return snapshot

    def _tick_policy_job(self, job, positions):
        # _tick_job 每个控制周期调用；最多取一个目标并发给电机，没有等待整块推理。
        if self._cancel.is_set() or job["generation"] != self._generation or self._closing.is_set():
            return
        target = self._policy.tick(deepcopy(positions[job["arm"]]))
        # target 是六关节目标 dict 或 None；None 常表示模型仍在加载/动作队列暂时为空。
        self._update_policy_job(job)
        if self._cancel.is_set() or job["generation"] != self._generation or self._closing.is_set():
            return
        if target is not None:
            # 模型线程只产出目标，串口仍由当前 owner 线程发送；发送成功后才记账。
            # acknowledge 确认 SDK 接受的目标，不是关节到达或物体任务成功的证明。
            applied = self.hardware.send_policy({job["arm"]: target})
            # 驱动输入按臂分组，返回 SDK 实际接受的同结构目标；下行只取当前臂。
            self._policy.acknowledge(applied[job["arm"]])
        snapshot = self._update_policy_job(job)
        with self._lock:
            job["elapsed_s"] = round(time.monotonic() - job["started_at"], 3)
            if self._cancel.is_set() or job["generation"] != self._generation or self._closing.is_set():
                return
        state = snapshot.get("status", "running")
        if state == "failed":
            raise RuntimeError(snapshot.get("message") or "策略执行失败")
        if state in ("completed", "cancelled"):
            message = (str(snapshot.get("message") or "策略执行周期已结束；物体是否成功取放尚未确认。")
                       if state == "completed" else str(snapshot.get("message") or "策略任务已取消。"))
            self._finish(job, state, message, object_task_success_verified=False)
        else:
            with self._lock:
                job["status"] = "loading" if state == "loading" else "running"

    def _bounded_reference(self, job, target, dt):
        # 仅普通 record/play 调用的旧轨迹处理；策略 task 分支不会经过这个函数。
        # dt 是本次控制周期耗时，target/返回值都是“臂名 -> 六关节目标”的 dict。
        answer = {}
        for arm, pose in target.items():
            answer[arm] = {}
            for joint, requested in pose.items():
                low, high = self._profiles[arm]["limits"][joint]
                bounded = min(high, max(low, requested))
                speed = self.config["gripper_speed_s"] if joint == "gripper.pos" else self.config["speed_deg_s"]
                previous = job["reference"][arm][joint]
                step = speed * dt
                answer[arm][joint] = previous + max(-step, min(step, bounded - previous))
                if abs(bounded - requested) > 1e-6 or abs(answer[arm][joint] - requested) > 0.1:
                    job["limited"] = True
        return answer

    def _tick_job(self, job, positions):
        # _run 调用的分流入口：首次先初始化；策略走 _tick_policy_job；普通轨迹走下方插值。
        if job["status"] == "accepted":
            self._start_job(job, positions)
            # return 结束本次函数调用；并非结束后台线程，下一个周期还会再进入。
            return
        if job["kind"] == "task":
            self._tick_policy_job(job, positions)
            return
        if job["status"] != "running":
            return
        now = time.monotonic()
        elapsed = now - job["started_at"]
        dt = min(max(now - job["last_tick"], 0.001), 2.0 / self.config["fps"])
        job["last_tick"] = now
        job["elapsed_s"] = round(elapsed, 3)
        tracking_bad = any(abs(positions[arm][joint] - value) > (
            self.config["gripper_tolerance"] * 1.5 if joint == "gripper.pos" else self.config["tracking_error_deg"])
            for arm, pose in job["reference"].items() for joint, value in pose.items())
        if tracking_bad:
            if job["tracking_bad_since"] is None:
                job["tracking_bad_since"] = now
            elif now - job["tracking_bad_since"] >= self.config["tracking_error_s"]:
                raise RuntimeError("实测位置持续跟不上轨迹，已停止本次动作")
        else:
            job["tracking_bad_since"] = None

        if job["kind"] == "record":
            arm = job["arm"]
            # Record the measured follower, not an unreachable leader target.
            if elapsed <= job["duration_s"]:
                job["frames"].append({"t": round(elapsed, 6), "positions": {arm: deepcopy(positions[arm])}})
            if elapsed >= job["duration_s"]:
                with self._lock:
                    if self._cancel.is_set() or job["generation"] != self._generation:
                        return
                # Do not hold the state mutex during serial or filesystem I/O:
                # stop must publish its cancellation while either is blocked.
                self.hardware.send({arm: positions[arm]})
                self.hardware.close_leader()
                if self._cancel.is_set():
                    return
                saved = save_recording(self.config, job["name"], {arm: self._profiles[arm]}, job["frames"])
                if self._cancel.is_set():
                    saved.unlink(missing_ok=True)
                    return
                motions, invalid = motion_catalog(self.config)
                with self._lock:
                    if self._cancel.is_set() or job["generation"] != self._generation:
                        cancelled = True
                    else:
                        cancelled = False
                        self._motions, self._invalid = motions, invalid
                        self._finish(job, "completed", "示教已保存；请先手动回放检查，再开启自动联动。",
                                     motion_saved=str(saved))
                if cancelled:
                    saved.unlink(missing_ok=True)
                return
            leader = self.hardware.read_leader()
            target = {arm: {joint: job["follower_start"][joint] + leader[joint] - job["leader_start"][joint]
                            for joint in JOINTS}}
            # 相对遥操作：从臂起点 + 主臂相对起点的变化；不是把两臂绝对读数直接相等。
        else:
            target = interpolate(job["motion"], elapsed)
            # interpolate 定义在 trajectories.py，按经过时间插值出目标，不涉及神经网络。
            finish_config = {**self.config, "start_tolerance_deg": self.config["finish_tolerance_deg"],
                             "gripper_tolerance": self.config["finish_gripper_tolerance"]}
            at_finish = (elapsed >= job["duration_s"] and job.get("final_goal_sent", False)
                         and within_pose(positions, target, finish_config))
            if at_finish:
                job.setdefault("settled_since", now)
            else:
                job.pop("settled_since", None)
            if at_finish and now - job["settled_since"] >= 0.1:
                with self._lock:
                    if self._cancel.is_set() or job["generation"] != self._generation:
                        return
                # Entering the tolerance alone is insufficient: an earlier
                # motor goal might still be moving the arm in another direction.
                self.hardware.send({arm: positions[arm] for arm in job["arms"]})
                with self._lock:
                    if self._cancel.is_set() or job["generation"] != self._generation:
                        return
                    self._finish(job, "completed", "轨迹已结束，关节实测进入终点容差；未验证任何物品任务。")
                return
            if elapsed > job["duration_s"] + 3.0:
                raise RuntimeError("轨迹结束后仍未到达终点容差")
        reference = self._bounded_reference(job, target, dt)
        with self._lock:
            if (self._cancel.is_set() or job["generation"] != self._generation
                    or not self._armed or self._closing.is_set()):
                return
        # This is one in-flight serial transaction. A concurrent stop may set
        # cancellation immediately; the next owner iteration performs readback
        # hold, and only then acknowledges stop completion.
        applied = self.hardware.send(reference)
        with self._lock:
            # Keep the time-based reference, not the driver's readback-relative
            # clamp, so a stalled joint cannot hide behind tiny sent targets.
            job["reference"] = reference
            if job["kind"] == "play" and elapsed >= job["duration_s"]:
                job["final_goal_sent"] = all(abs(applied[arm][joint] - number) < 1e-3
                    for arm, pose in target.items() for joint, number in pose.items())

    def _job_failed(self, job, error):
        # _run 捕获当前任务异常后调用；error 是异常对象，type(error).__name__ 取异常类名。
        message = f"{type(error).__name__}: {error}"
        if job["kind"] == "task" and self._policy is not None:
            self._policy.stop()
        with self._lock:
            self._armed = self._auto = False
            self._cancel.set()
        try:
            hold = self._hold()
        except Exception as hold_error:
            hold = _result(False, "unconfirmed", f"无法确认持姿：{hold_error}", hold_confirmed=False)
            with self._lock:
                self._fault = hold["message"]
                self._ready = False
        try:
            self.hardware.close_leader()
        except Exception:
            pass
        with self._lock:
            self._hold_result = hold
        self._finish(job, "failed", message, hold_confirmed=hold["hold_confirmed"])

    def _run(self):
        # 本函数是 friston-arm-owner 线程入口：连接、读取、发送、关闭都在此线程。
        # 慢模型在 PolicySession 的另一个线程执行，因此不会直接阻塞此控制循环。
        connected = False
        # connected 是本线程局部标记，用于 finally 判断是否应尝试读回持姿。
        try:
            self.hardware.connect(native_policy=True)
            # 连接操作在此线程发生，hardware._claim 会把此线程 ID 记录为唯一 owner。
            connected = True
            self._profiles = deepcopy(self.hardware.profiles)
            self._observe()
            with self._lock:
                self._ready = True
            self._startup.set()
            print(f"[Arms] {self.config['mode']} 服务就绪；动作未使能。", flush=True)
            interval = 1.0 / self.config["fps"]
            # fps=20 表示目标周期 0.05 秒；这是调度目标，USB I/O 和系统调度会引入波动。
            while not self._closing.is_set():
                began = time.monotonic()
                self._wake.clear()
                # 清掉上一轮唤醒；本轮若有新 stop/command，其他线程可再次 set。
                with self._lock:
                    stopping, faulted = self._need_hold, bool(self._fault)
                if stopping:
                    self._cancel_current()
                elif not faulted:
                    positions = self._observe()
                    with self._lock:
                        job = self._active
                    if job:
                        try:
                            self._tick_job(job, positions)
                        except Exception as error:
                            self._job_failed(job, error)
                self._wake.wait(max(0.0, interval - (time.monotonic() - began)))
                # Event.wait 在剩余周期内休息；其他线程 set 时可提前醒来，减少取消等待。
                # 扣掉本轮已耗时间，按配置 fps 调度；普通线程调度并不保证硬实时。
        except Exception as error:
            with self._lock:
                self._ready = self._armed = self._auto = False
                self._fault = f"{type(error).__name__}: {error}"
                if self._active:
                    self._finish(self._active, "failed", self._fault, hold_confirmed=False)
                self._hold_result = _result(False, "unconfirmed", "连接失败，未确认持姿。", hold_confirmed=False)
                self._need_hold = False
            print(f"[Arms] 服务故障：{self._fault}", flush=True)
        finally:
            # finally 无论正常退出还是发生异常都会执行，避免只在成功路径关闭资源。
            self._startup.set()
            if self._policy is not None:
                try:
                    self._policy.stop()
                    self._policy.close()
                except Exception as error:
                    print(f"[Arms] 策略关闭未确认：{error}", flush=True)
            # Final readback hold is attempted only after a completed connect;
            # the driver closes all ports without a surprise torque release.
            if connected:
                try:
                    self._hold_result = self._hold()
                except Exception:
                    self._hold_result = _result(False, "unconfirmed", "关闭时未确认持姿。", hold_confirmed=False)
            try:
                self.hardware.close()
            except Exception as error:
                print(f"[Arms] 关闭未确认：{error}", flush=True)
            with self._lock:
                self._ready = self._armed = self._auto = False
                self._need_hold = False
                if self._active:
                    self._finish(self._active, "cancelled", "服务关闭；已取消后续动作。",
                                 hold_confirmed=bool((self._hold_result or {}).get("hold_confirmed")))
                elif self._last_job and self._last_job["status"] == "failed":
                    held = bool((self._hold_result or {}).get("hold_confirmed"))
                    self._last_job["hold_confirmed"] = held
                    self._jobs[self._last_job["id"]]["hold_confirmed"] = held

    def close(self):
        # serve 的 finally 调用；调用方只发退出通知，并等待 owner 自己关串口。
        with self._lock:
            self._request_stop_locked()
        self._closing.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=4.0)
            # Thread.join 等待线程结束；timeout 到期不等于强制终止，所以还检查 is_alive。
            if self._thread.is_alive():
                print("[Arms] 控制线程尚未退出，串口状态未确认。", flush=True)


class _Handler(BaseHTTPRequestHandler):
    # 继承标准库 BaseHTTPRequestHandler；HTTPServer 在请求到来时调用 do_GET/do_POST。
    # 这里的 self 是“请求处理器”，不是上面的 ArmService；服务保存在 self.server.engine。
    protocol_version = "HTTP/1.0"

    def log_message(self, *_args):
        pass

    def setup(self):
        super().setup()
        self.connection.settimeout(1.5)

    def _send(self, value, status=200):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def _authorized(self):
        supplied = self.headers.get("Authorization", "")
        expected = "Bearer " + self.server.engine.config["token"]
        if self.headers.get("Origin") is not None or not hmac.compare_digest(
                supplied.encode("utf-8"), expected.encode("utf-8")):
            self._send(_result(False, "unauthorized", "本机接口认证失败。"), 403)
            return False
        return True

    def do_GET(self):
        # GET 只读状态。self.path 来自标准库解析的 URL 路径。
        if not self._authorized():
            return
        if self.path == "/status":
            self._send(self.server.engine.status())
        elif self.path.startswith("/jobs/") and len(self.path) <= 100:
            self._send(self.server.engine.job_status(self.path[len("/jobs/"):]))
        else:
            self._send(_result(False, "not_found", "接口不存在。"), 404)

    def do_POST(self):
        # POST 解析 JSON 后交给 engine.command；HTTP 请求线程不拿驱动对象执行动作。
        if not self._authorized():
            return
        if self.path != "/command":
            self._send(_result(False, "not_found", "接口不存在。"), 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 16_384 or self.headers.get("Transfer-Encoding"):
                raise ValueError("请求长度无效")
            if self.headers.get_content_type() != "application/json":
                raise ValueError("仅支持 application/json")
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("请求体不完整")
            request = json.loads(raw.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError("不接受非有限数值")))
            answer = self.server.engine.command(request)
        except (ValueError, UnicodeError, TimeoutError, OSError) as error:
            self._send(_result(False, "rejected", str(error)), 400)
            return
        self._send(answer)
        # JSON 语法有效也可能得到列表/null；command 已拒绝后，不能再无条件 .get。
        if (isinstance(request, dict) and request.get("op") == "shutdown"
                and answer.get("status") == "shutting_down"):
            self.server.request_shutdown()


class _Server(ThreadingHTTPServer):
    # 标准库 ThreadingHTTPServer 为请求创建线程；继承只补充服务关闭协调。
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shutdown_lock = threading.Lock()
        self._shutdown_thread = None

    def request_shutdown(self):
        with self._shutdown_lock:
            if self._shutdown_thread is None:
                self._shutdown_thread = threading.Thread(target=self.shutdown,
                                                         name="friston-arm-shutdown", daemon=True)
                self._shutdown_thread.start()


def serve(config_path=None, *, policy_overrides=None):
    # arms.__main__.main 的 serve 分支调用。这里组装配置、业务对象、HTTP 对象并启动。
    config = load_config(config_path)
    if policy_overrides:
        config["policy"] = {**config.get("policy", {}), **policy_overrides}
        from .policy import policy_options
        policy_options(config)  # 启动前校验命令行覆盖项；只影响本次服务进程。
    engine = ArmService(config)
    # Bind first: a second service fails before touching any serial port.
    # 这是同一监听端口上的服务去重；不是覆盖任意外部录制脚本的全局串口锁。
    server = _Server((config["host"], config["port"]), _Handler)
    server.engine = engine
    print(f"[Arms] 仅本机监听 {config['host']}:{config['port']}；模式={config['mode']}。", flush=True)
    if config["mode"] == "real":
        print("[Arms] 请在机械臂放稳后连接：LeRobot 初始化可能短暂关闭再启用力矩。", flush=True)
        print("[Arms] 停止/退出保留持姿力矩；手动断电前请支撑好机械臂。", flush=True)
    else:
        print("[Arms] 模拟模式：结果仅验证软件，不能证明实机动作可用。", flush=True)
    try:
        engine.start()
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\n[Arms] 正在停止服务。", flush=True)
    finally:
        engine.close()
        server.server_close()
    return 0
