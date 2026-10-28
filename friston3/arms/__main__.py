"""Operator commands for the standalone arm service; no serial I/O in this CLI."""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

from .client import ArmClient
from .config import create_config, default_config_path


_NAME = re.compile(r"[a-z][a-z0-9_-]{0,47}\Z")
_TERMINAL_STATES = frozenset(("completed", "cancelled", "failed"))


def _motion_name(value: str) -> str:
    if not _NAME.fullmatch(value):
        raise argparse.ArgumentTypeError("动作名应为小写字母开头，随后为小写字母、数字、_ 或 -，最多 48 字。")
    return value


def _record_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("录制秒数必须是数字。") from error
    if not math.isfinite(seconds) or not 1.0 <= seconds <= 30.0:
        raise argparse.ArgumentTypeError("录制时长应在 1–30 秒之间。")
    return seconds


def _wait_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("等待秒数必须是数字。") from error
    if not math.isfinite(seconds) or not 1.0 <= seconds <= 120.0:
        raise argparse.ArgumentTypeError("等待时长应在 1–120 秒之间。")
    return seconds


def _shared_options(parser: argparse.ArgumentParser) -> None:
    # SUPPRESS lets both `--config X status` and `status --config X` work,
    # without a subparser default discarding a global option already supplied.
    parser.add_argument("--config", default=argparse.SUPPRESS,
                        help="配置文件路径；默认使用 FRISTON_ARMS_CONFIG 或用户配置目录")
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="以 JSON 输出结果")


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description="Friston-3 双臂动作与示教控制台")
    parser.set_defaults(config=None, json=False)
    _shared_options(parser)
    commands = parser.add_subparsers(dest="operation", required=True)

    init = commands.add_parser("init", help="创建配置；保留已有配置，不连接硬件")
    _shared_options(init)
    for side in ("right", "left", "leader"):
        init.add_argument("--" + side + "-port", default="", help=side + " 串口")
        init.add_argument("--" + side + "-id", default="", help=side + " 已有标定 ID")
    init.add_argument("--mock", action="store_true", help="创建纯模拟配置，不接任何真实机械臂")

    for name, help_text in (
            ("serve", "在 lerobot 环境启动唯一硬件控制服务"),
            ("status", "查看实际服务状态与当前姿态"),
            ("list", "列出已录制动作"),
            ("arm", "本地启用后续动作；不播放轨迹"),
            ("stop", "请求停止并禁用后续动作；不卸力、不回起点")):
        child = commands.add_parser(name, help=help_text)
        _shared_options(child)

    auto = commands.add_parser("auto", help="允许或禁止对话触发已录制的闭合动作")
    _shared_options(auto)
    auto.add_argument("setting", choices=("on", "off"))

    play = commands.add_parser("play", help="回放已录动作，默认等待实际终态")
    _shared_options(play)
    play.add_argument("name", type=_motion_name)
    play.add_argument("--no-wait", action="store_true", help="只显示接收结果和任务编号")
    play.add_argument("--wait-timeout", type=_wait_seconds, default=45.0)

    record = commands.add_parser("record", help="用主臂示教选定从臂，按实际反馈记录动作")
    _shared_options(record)
    record.add_argument("name", type=_motion_name)
    record.add_argument("--arm", choices=("left", "right"), default="right")
    record.add_argument("--seconds", type=_record_seconds, default=8.0)
    record.add_argument("--countdown", type=int, choices=(0, 1, 2, 3), default=3,
                        help="提交录制前准备倒计时，默认 3 秒")
    record.add_argument("--no-wait", action="store_true")
    record.add_argument("--wait-timeout", type=_wait_seconds, default=45.0)
    return parser.parse_args(argv)


def _emit(result: dict, *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        return
    print(result.get("message") or result.get("status") or "未返回状态", flush=True)
    mode = result.get("mode")
    if mode == "mock":
        print("[模拟] 此结果仅验证软件流程，不代表实机执行或实机验收。", flush=True)
    elif mode == "real":
        print("[实机] 由独立控制服务连接机械臂。", flush=True)
    if "armed" in result:
        print("动作执行：" + ("已启用" if result["armed"] else "未启用 / 已停止")
              + "；对话动作：" + ("开启" if result.get("auto_enabled") else "关闭"), flush=True)
    arms = result.get("available_arms")
    if isinstance(arms, list):
        print("已接入：" + ("、".join(str(arm) for arm in arms) or "无"), flush=True)
    if result.get("job_id"):
        print("任务编号：" + str(result["job_id"]), flush=True)
    finished_job = result.get("job")
    if isinstance(finished_job, dict) and finished_job.get("motion_saved"):
        print("动作已保存：" + str(finished_job["motion_saved"]), flush=True)
    current = result.get("current_job")
    if isinstance(current, dict):
        print("当前任务：{name} / {status} / {id}".format(
            name=current.get("name", ""), status=current.get("status", ""),
            id=current.get("id", "")), flush=True)
    motions = result.get("motions")
    if isinstance(motions, list):
        if not motions:
            print("尚无动作。使用 record 录制你自己的轨迹。", flush=True)
        for motion in motions:
            if not isinstance(motion, dict):
                continue
            print("  {name} | {arms} | {duration}s | {looped}".format(
                name=motion.get("name", ""),
                arms=",".join(str(arm) for arm in motion.get("arms", [])),
                duration=motion.get("duration_s", "?"),
                looped="起终点相近，可检查后用于对话" if motion.get("looped") else "非闭合轨迹"), flush=True)
    positions = result.get("positions")
    if isinstance(positions, dict):
        for arm, values in positions.items():
            if isinstance(values, dict):
                readings = ", ".join(f"{key}={value:.2f}" for key, value in values.items()
                                     if type(value) in (int, float))
                print(f"  {arm}: {readings}", flush=True)
        if positions:
            print("姿态单位：转动关节为度，gripper.pos 为 0–100。", flush=True)


def _confirm_stop(client: ArmClient, *, timeout: float = 5.0) -> dict:
    """Wait for the serial owner to acknowledge its readback-hold command.

    A POST acknowledgement only means cancellation has been requested. It
    does not prove the worker is alive or that a bus write succeeded.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            snapshot = client.status()
        except (OSError, ValueError, RuntimeError) as error:
            return {"success": False, "status": "unconfirmed", "hold_confirmed": False,
                    "mode": getattr(client, "mode", None),
                    "message": "停止请求后无法取得服务状态，持姿尚未确认。" + str(error)}
        if snapshot.get("success") is not True or snapshot.get("status") == "faulted":
            return {"success": False, "status": "unconfirmed", "hold_confirmed": False,
                    "mode": snapshot.get("mode", getattr(client, "mode", None)),
                    "message": "服务故障，停止持姿尚未确认：" + str(snapshot.get("message", "未知故障")),
                    "hold_result": snapshot.get("hold_result")}
        if snapshot.get("stop_pending") is False:
            hold = snapshot.get("hold_result")
            confirmed = (isinstance(hold, dict) and hold.get("success") is True
                         and hold.get("hold_confirmed") is True
                         and snapshot.get("armed") is False
                         and snapshot.get("current_job") is None)
            if confirmed:
                return {**hold, "mode": snapshot.get("mode", getattr(client, "mode", None)),
                        "armed": False, "auto_enabled": snapshot.get("auto_enabled", False)}
            detail = hold.get("message") if isinstance(hold, dict) else "没有取得执行器持姿回执。"
            if snapshot.get("armed") or snapshot.get("current_job") is not None:
                detail = "服务已重新使能或仍有任务，不能将当前状态视为已停止。"
            return {"success": False, "status": "unconfirmed", "hold_confirmed": False,
                    "mode": snapshot.get("mode", getattr(client, "mode", None)),
                    "message": "停止持姿未确认：" + str(detail), "hold_result": hold}
        time.sleep(0.1)
    return {"success": False, "status": "unconfirmed", "hold_confirmed": False,
            "mode": getattr(client, "mode", None),
            "message": "停止请求已提交，但等待执行器持姿确认超时；请检查服务终端和机械臂状态。"}


def _stop_after_wait_failure(client: ArmClient) -> str:
    try:
        stop = client.command("stop")
        if stop.get("success") is not True:
            return str(stop.get("message", "停止请求未被接受。"))
        confirmed = _confirm_stop(client)
        return str(confirmed["message"])
    except (OSError, ValueError, RuntimeError):
        return "停止请求也未获得确认；请检查服务终端和机械臂当前状态。"


def _wait_for_job(client: ArmClient, accepted: dict, *, timeout: float, as_json: bool) -> int:
    job_id = accepted.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError("服务接收了操作但未返回任务编号，不能确认执行结果。")
    deadline = time.monotonic() + timeout
    previous = None
    try:
        while time.monotonic() < deadline:
            result = client.job(job_id)
            if result.get("success") is not True:
                _emit(result, as_json=as_json)
                return 1
            job = result.get("job")
            if not isinstance(job, dict) or job.get("id") != job_id:
                raise RuntimeError("服务返回的任务编号不匹配，不能确认执行结果。")
            status = job.get("status")
            if status != previous:
                _emit({"mode": getattr(client, "mode", None), **result}, as_json=as_json)
                previous = status
            if status in _TERMINAL_STATES:
                return 0 if status == "completed" else (130 if status == "cancelled" else 1)
            time.sleep(0.2)
    except KeyboardInterrupt:
        message = _stop_after_wait_failure(client)
        _emit({"success": False, "status": "stop_requested", "message": message,
               "job_id": job_id}, as_json=as_json)
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        stop_message = _stop_after_wait_failure(client)
        raise RuntimeError(str(error) + " " + stop_message) from error
    stop_message = _stop_after_wait_failure(client)
    raise RuntimeError("等待任务结果超时；未将其视为完成。" + stop_message)


def main(argv=None) -> int:
    args = _arguments(argv)
    client = None
    motion_submitted = False
    try:
        if args.operation == "init":
            path = create_config(args.config, right_port=args.right_port, right_id=args.right_id,
                                 left_port=args.left_port, left_id=args.left_id,
                                 leader_port=args.leader_port, leader_id=args.leader_id,
                                 mock=args.mock)
            _emit({"success": True, "status": "configured", "mode": "mock" if args.mock else "real",
                   "message": f"配置已创建：{path}。尚未连接硬件或启用动作。"}, as_json=args.json)
            return 0
        if args.operation == "serve":
            from .service import serve
            outcome = serve(args.config)
            return outcome if type(outcome) is int else 0

        client = ArmClient(args.config)
        if args.operation == "status":
            result = client.status()
        elif args.operation == "auto":
            result = client.command("auto", enabled=args.setting == "on")
        elif args.operation == "play":
            motion_submitted = True
            result = client.command("play", name=args.name, source="manual")
        elif args.operation == "record":
            # Check availability before the preparation countdown. The service
            # remains authoritative if state changes while the operator waits.
            snapshot = client.status()
            if snapshot.get("success") is not True:
                _emit(snapshot, as_json=args.json)
                return 1
            if snapshot.get("mode") == "mock" and not args.json:
                print("[模拟录制] 使用模拟主臂，结果不能用于真实机械臂。", flush=True)
            if snapshot.get("armed") is not True or snapshot.get("busy"):
                _emit({"success": False, "status": "not_ready",
                       "message": "录制前请确认服务空闲，并在本地执行 arm 启用动作。"}, as_json=args.json)
                return 1
            for remaining in range(args.countdown, 0, -1):
                if not args.json:
                    print(f"准备录制 {args.arm} / {args.name}：{remaining}…", flush=True)
                time.sleep(1.0)
            motion_submitted = True
            result = client.command("record", name=args.name, arm=args.arm,
                                    duration_s=float(args.seconds))
        else:
            result = client.command(args.operation)

        _emit({"mode": getattr(client, "mode", None), **result}, as_json=args.json)
        if result.get("success") is not True:
            return 1
        if args.operation == "stop":
            confirmed = _confirm_stop(client)
            _emit(confirmed, as_json=args.json)
            return 0 if confirmed.get("success") is True else 1
        if args.operation in ("record", "play") and not args.no_wait:
            return _wait_for_job(client, result, timeout=args.wait_timeout, as_json=args.json)
        return 0
    except KeyboardInterrupt:
        # Ctrl+C can arrive while a POST is in flight, before its job_id is
        # returned. Such an operation may already be running on the service.
        message = (_stop_after_wait_failure(client) if motion_submitted and client is not None
                   else "本地操作已取消，尚未提交动作。")
        _emit({"success": False, "status": "stop_requested" if motion_submitted else "cancelled",
               "message": message}, as_json=args.json)
        return 130
    except FileNotFoundError as error:
        config_path = Path(args.config) if args.config else default_config_path()
        message = str(error)
        if not config_path.is_file():
            message += ("。先执行 python -m friston3.arms init；"
                        f"默认配置位置：{default_config_path()}。")
        _emit({"success": False, "status": "missing_file",
               "message": message}, as_json=args.json)
        return 1
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        _emit({"success": False, "status": "error", "message": str(error)}, as_json=args.json)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
