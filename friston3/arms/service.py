"""One serial owner, bounded taught motions, and a loopback-only command API."""
from __future__ import annotations

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

from .config import ARM_NAMES, GESTURES, JOINTS, load_config
from .trajectories import (check_compatibility, interpolate, load_motion, motion_catalog,
                           motion_summary, save_recording, validate_name, within_pose)


def _result(success, status, message, **extra):
    return {"success": success, "status": status, "message": message, **extra}


class ArmService:
    """HTTP threads never call the hardware. There is no waiting action queue."""

    def __init__(self, config, hardware=None):
        self.config = config
        if hardware is None:
            from .hardware import make_hardware
            hardware = make_hardware(config)
        self.hardware = hardware
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._closing = threading.Event()
        self._cancel = threading.Event()
        self._startup = threading.Event()
        self._thread = None
        self._ready = False
        self._fault = ""
        self._armed = False
        self._auto = False
        self._positions = {}
        self._profiles = {}
        self._observed_at = 0.0
        self._active = None
        self._last_job = None
        self._jobs = OrderedDict()
        self._requests = OrderedDict()
        self._generation = 0
        self._need_hold = False
        self._hold_result = None
        self._motions, self._invalid = motion_catalog(config)

    def start(self):
        if self._thread is not None:
            raise RuntimeError("机械臂服务已经启动")
        self._thread = threading.Thread(target=self._run, name="friston-arm-owner", daemon=True)
        self._thread.start()

    def wait_ready(self, timeout=10.0):
        if not self._startup.wait(timeout):
            raise TimeoutError("机械臂连接仍未完成")
        with self._lock:
            if not self._ready:
                raise RuntimeError(self._fault or "机械臂未连接")

    def _public_job(self, job):
        if job is None:
            return None
        return deepcopy({key: job[key] for key in (
            "id", "kind", "name", "status", "message", "source", "elapsed_s",
            "duration_s", "hold_confirmed", "motion_saved", "limited", "arms") if key in job})

    def status(self):
        with self._lock:
            status = ("faulted" if self._fault else "starting" if not self._ready else
                      "stopping" if self._need_hold else
                      self._active["status"] if self._active else "idle")
            return _result(not bool(self._fault), status,
                self._fault or ("模拟服务：不会控制真实机械臂。" if self.config["mode"] == "mock" else
                                 "机械臂服务已连接。" if self._ready else "正在连接机械臂。"),
                mode=self.config["mode"], armed=self._armed, auto_enabled=self._auto,
                busy=self._active is not None or self._need_hold,
                ready=self._ready, stop_pending=self._need_hold,
                available_arms=list(self._profiles) if self._ready else [],
                configured_arms=list(self.config["arms"]), positions=deepcopy(self._positions),
                observation_age_s=round(time.monotonic() - self._observed_at, 3) if self._observed_at else None,
                current_job=self._public_job(self._active), last_job=deepcopy(self._last_job),
                motions=deepcopy(self._motions), invalid_motions=deepcopy(self._invalid),
                hold_result=deepcopy(self._hold_result))

    def job_status(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return _result(False, "not_found", "找不到该任务；记录最多保留最近64项。")
            return _result(True, job["status"], job["message"], job=self._public_job(job))

    def command(self, request):
        """Validate/deduplicate before publishing one job to the owner thread."""
        if not isinstance(request, dict):
            return _result(False, "rejected", "请求必须是JSON对象")
        operation = request.get("op")
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not 8 <= len(request_id) <= 80:
            return _result(False, "rejected", "请求缺少有效 request_id")
        try:
            fingerprint = json.dumps(request, sort_keys=True, ensure_ascii=True, allow_nan=False)
        except (ValueError, TypeError):
            return _result(False, "rejected", "请求含不可序列化或非有限数值")
        with self._lock:
            if request_id in self._requests:
                original, answer = self._requests[request_id]
                return deepcopy(answer) if original == fingerprint else _result(
                    False, "rejected", "相同 request_id 不可用于不同命令")
            try:
                answer = self._command_locked(operation, request)
            except (OSError, ValueError, KeyError, TypeError) as error:
                answer = _result(False, "rejected", str(error))
            self._requests[request_id] = (fingerprint, deepcopy(answer))
            # Only bounded, in-memory metadata; never write logs to the project.
            while len(self._requests) > 256:
                self._requests.popitem(last=False)
            return answer

    def _command_locked(self, operation, request):
        fields = {
            "arm": set(), "stop": {"job_id"}, "auto": {"enabled"},
            "play": {"name", "source"}, "record": {"name", "arm", "duration_s"},
            "list": set(), "status": set(),
        }
        if operation not in fields or set(request) - {"op", "request_id"} - fields[operation]:
            raise ValueError("未知操作或多余参数")
        if operation == "status":
            return self.status()
        if operation == "list":
            self._motions, self._invalid = motion_catalog(self.config)
            return _result(True, "listed", "已读取本地示教动作。", mode=self.config["mode"],
                           motions=deepcopy(self._motions), invalid_motions=deepcopy(self._invalid))
        if operation == "stop":
            specified = request.get("job_id")
            if specified is not None:
                if not isinstance(specified, str):
                    raise ValueError("job_id 必须是字符串")
                if not self._active or self._active["id"] != specified:
                    return _result(True, "no_op", "该任务已不在执行；未干预其他任务。")
            return self._request_stop_locked()
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
        if not self._armed:
            raise ValueError("动作未使能；在机械臂终端运行 python -m friston3.arms arm")
        if self._active:
            return _result(False, "busy", "机械臂正在执行任务，本次动作未排队。",
                           job_id=self._active["id"])
        name = validate_name(request.get("name"))
        now = time.monotonic()
        if now - self._observed_at > max(0.5, 3.0 / self.config["fps"]):
            raise ValueError("机械臂状态过旧，拒绝开始动作")
        job = {"id": uuid.uuid4().hex, "kind": operation, "name": name,
               "status": "accepted", "message": "任务已接收，尚未完成。", "source": "manual",
               "generation": self._generation, "accepted_at": now, "elapsed_s": 0.0}
        if operation == "play":
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
            self._auto = False
        self._active = job
        self._jobs[job["id"]] = job
        while len(self._jobs) > 64:
            self._jobs.popitem(last=False)
        self._cancel.clear()
        self._wake.set()
        return _result(True, "accepted", "动作任务已接收，执行结果请按 job_id 查询。",
                       job_id=job["id"], mode=self.config["mode"])

    def _request_stop_locked(self):
        self._armed = False
        self._auto = False
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

    def _observe(self):
        positions = self.hardware.read()
        if set(positions) != set(self.config["arms"]):
            raise RuntimeError("机械臂状态缺失")
        for arm, pose in positions.items():
            if (set(pose) != set(JOINTS) or any(type(v) not in (int, float) or not math.isfinite(v)
                                              for v in pose.values())):
                raise RuntimeError(f"{arm} 状态包含无效关节数值")
            for joint, number in pose.items():
                low, high = self._profiles[arm]["limits"][joint]
                if not low - 0.01 <= number <= high + 0.01:
                    raise RuntimeError(f"{arm}.{joint} 实测超出已标定范围")
        with self._lock:
            self._positions = positions
            self._observed_at = time.monotonic()
        return positions

    def _finish(self, job, status, message, **extra):
        with self._lock:
            job.update(status=status, message=message, **extra)
            self._last_job = self._public_job(job)
            if self._active is job:
                self._active = None
        print(f"[Arms] {job['name']}: {status} — {message}", flush=True)

    def _hold(self):
        # Called only by the serial owner; no stale cached pose is substituted.
        positions = self._observe()
        self.hardware.send(positions)
        return _result(True, "hold_command_sent", "已用当前实测姿态下发持姿目标；持续监控连接。",
                       hold_confirmed=True)

    def _cancel_current(self):
        with self._lock:
            job = self._active
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
        with self._lock:
            if self._cancel.is_set() or job["generation"] != self._generation or self._closing.is_set():
                return
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

    def _bounded_reference(self, job, target, dt):
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
        if job["status"] == "accepted":
            self._start_job(job, positions)
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
        else:
            target = interpolate(job["motion"], elapsed)
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
        message = f"{type(error).__name__}: {error}"
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
        connected = False
        try:
            self.hardware.connect()
            connected = True
            self._profiles = deepcopy(self.hardware.profiles)
            self._observe()
            with self._lock:
                self._ready = True
            self._startup.set()
            print(f"[Arms] {self.config['mode']} 服务就绪；动作未使能。", flush=True)
            interval = 1.0 / self.config["fps"]
            while not self._closing.is_set():
                began = time.monotonic()
                self._wake.clear()
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
            self._startup.set()
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
        with self._lock:
            self._request_stop_locked()
        self._closing.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=4.0)
            if self._thread.is_alive():
                print("[Arms] 控制线程尚未退出，串口状态未确认。", flush=True)


class _Handler(BaseHTTPRequestHandler):
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
        if not self._authorized():
            return
        if self.path == "/status":
            self._send(self.server.engine.status())
        elif self.path.startswith("/jobs/") and len(self.path) <= 100:
            self._send(self.server.engine.job_status(self.path[len("/jobs/"):]))
        else:
            self._send(_result(False, "not_found", "接口不存在。"), 404)

    def do_POST(self):
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


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


def serve(config_path=None):
    config = load_config(config_path)
    engine = ArmService(config)
    # Bind first: a second service fails before touching any serial port.
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
