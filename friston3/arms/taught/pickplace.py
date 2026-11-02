#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZCD first real skill: teach A/B/C hover/contact and execute one pick/place.

No ACT, no camera localization, no implicit pose guessing.
Depends on the move, pose and exchange modules in this package.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from dataclasses import replace

from . import move as motion
from . import pose as poses
from . import exchange

HERE = Path(__file__).resolve().parent
SKILLS_FILE = HERE / "so101_zcd_grasp.json"
SLOTS = ("A", "B", "C")
PHASES = ("hover", "contact")
GRIPS = ("open", "close")


def load_record(path, robot_id, calibration, *, allow_missing=False):
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("技能记录文件不是 JSON 对象")
        if data.get("robot_id") != robot_id or data.get("calibration_sha256") != poses.signature(calibration):
            raise ValueError("技能记录与当前 COM9 机械臂/校准不一致")
        if not isinstance(data.get("slots"), dict) or not isinstance(data.get("gripper"), dict):
            raise ValueError("技能记录数据结构错误")
    elif allow_missing:
        data = {"robot_id": robot_id, "calibration_sha256": poses.signature(calibration),
                "slots": {}, "gripper": {}}
    else:
        raise FileNotFoundError(f"还没有记录基础技能：{path}")
    return data


def write_record(path, data):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def valid_raw(raw):
    if (not isinstance(raw, dict) or set(raw) != set(motion.BODY)
            or any(type(v) is not int or not 0 <= v <= 4095 for v in raw.values())):
        raise ValueError("抓放关键位姿的编码器记录无效")
    return raw


def capture_slot(bus, cal, path, robot_id, slot, phase):
    raw, _ = motion.prepare_live(bus, cal)
    data = load_record(path, robot_id, cal, allow_missing=True)
    data["slots"].setdefault(slot, {})[phase] = {
        "raw_position": raw,
        "joints_deg": motion.to_degrees(raw, cal),
        "model_position_mm": motion.fk.position_mm(motion.to_degrees(raw, cal)),
    }
    write_record(path, data)
    return {"status": "SLOT_POSE_SAVED_NO_MOTION", "slot": slot, "phase": phase,
            "pose": data["slots"][slot][phase]}


def capture_grip(bus, cal, path, robot_id, name):
    if not bus.is_calibrated:
        raise RuntimeError("当前校准不匹配")
    actual = exchange.gripper_state(bus)
    data = load_record(path, robot_id, cal, allow_missing=True)
    data["gripper"][name] = actual["percent"]
    write_record(path, data)
    return {"status": "GRIP_VALUE_SAVED_NO_MOTION", "name": name,
            "percent": actual["percent"]}


def grip_to(bus, percent, execute):
    """Set recorded normalized gripper target; not a force-control grasp."""
    if not math.isfinite(percent) or not 0 <= percent <= 100:
        raise ValueError("夹爪目标应在校准的 0..100 之间")
    current = exchange.gripper_state(bus)
    if not execute:
        return {"status": "GRIPPER_PREVIEW_NO_WRITES", "from": current,
                "target_percent": percent}
    if not bus.is_calibrated:
        raise RuntimeError("当前校准不匹配")
    try:
        if not current["torque_enabled"]:
            bus.write("Goal_Position", "gripper", current["percent"], num_retry=0)
            bus.enable_torque(motors=["gripper"], num_retry=0)
            if not exchange.gripper_state(bus)["torque_enabled"]:
                raise RuntimeError("夹爪电机力矩状态无法确认")
        span = percent - current["percent"]
        points = max(1, math.ceil(abs(span) / 2.0))
        for k in range(1, points + 1):
            goal = current["percent"] + span * k / points
            bus.sync_write("Goal_Position", {"gripper": goal}, num_retry=0)
            time.sleep(0.10)
        time.sleep(0.25)
        final = exchange.gripper_state(bus)
        return {"status": "GRIPPER_TARGET_SENT", "target_percent": percent,
                "after": final, "grasp_confirmed": False}
    except BaseException:
        print("夹爪控制中断：" + exchange.hold_gripper(bus), file=sys.stderr)
        raise


def pose_targets(data, robot_id, cal, slot_from, slot_to):
    p = poses.load_file(poses.DEFAULT_FILE)
    poses.check_identity(p, robot_id, cal)
    ready = valid_raw(poses.checked_target(p, "ready", cal))
    slots = data["slots"]
    targets = {"ready": ready}
    for s in (slot_from, slot_to):
        if s not in slots:
            raise ValueError(f"未标定槽位 {s}")
        for phase in PHASES:
            record = slots[s].get(phase)
            if not isinstance(record, dict):
                raise ValueError(f"尚未标定 {s}.{phase}")
            targets[f"{s}.{phase}"] = valid_raw(record.get("raw_position"))
    grips = data["gripper"]
    for name in GRIPS:
        v = grips.get(name)
        if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 100:
            raise ValueError(f"请先标定 gripper {name}")
    if grips["open"] == grips["close"]:
        raise ValueError("夹爪的 open 和 close 不能是同一个位置")
    return targets, grips


def pick_actions(slot):
    return [
        ("arm", "ready"), ("grip", "open"),
        ("arm", f"{slot}.hover"), ("arm", f"{slot}.contact"),
        ("grip", "close"), ("arm", f"{slot}.hover"),
        ("confirm", "lifted"), ("arm", "ready"),
    ]


def place_actions(slot):
    return [
        ("arm", f"{slot}.hover"), ("arm", f"{slot}.contact"),
        ("grip", "open"), ("arm", f"{slot}.hover"),
        ("confirm", "placed"), ("arm", "ready"),
    ]


def steps_between(a, c):
    return pick_actions(a) + place_actions(c)


def preflight(
    start_raw, cal, targets, grips,
    source, dest, policy, actions=None
):
    raw = dict(start_raw)
    trace = []

    if actions is None:
        actions = steps_between(source, dest)

    for category, name in actions:
        if category == "arm":
            goal = targets[name]
            plan = motion.make_plan(raw, [0, 0, 0], cal,
                                    execution_policy=policy, joint_target_raw=goal)
            change = motion.max_error(
                motion.to_degrees(raw, cal), motion.to_degrees(goal, cal)
            )
            trace.append({"type": "arm", "pose": name,
                          "max_joint_change_deg": round(change, 1),
                          "model_seconds": round(plan["nominal_seconds"], 1)})
            raw = dict(goal)
        elif category == "grip":
            trace.append({"type": "gripper", "command": name,
                          "percent": round(grips[name], 2)})
        else:
            trace.append({"type": "manual_confirmation", "item": name})
    return trace



def go(bus, cal, raw_target, policy, name):
    """
    执行一个已标定的关节姿态。

    Ready / hover：过渡姿态，允许适当定位误差。
    contact：抓放姿态，保留原本较严格的到位要求。
    """

    is_transit = (
        name == "ready"
        or name.endswith(".hover")
    )

    if is_transit:
        actual_policy = replace(
            policy,
            final_joint_tolerance_deg=2.0,
            endpoint_tolerance_mm=8.0,
        )
    else:
        actual_policy = policy

    actual_policy.validate()

    print(
        f"MOVE -> {name} "
        f"(tolerance: "
        f"{actual_policy.final_joint_tolerance_deg:g}deg / "
        f"{actual_policy.endpoint_tolerance_mm:g}mm)",
        flush=True,
    )

    outcome = motion.move_ee(
        bus,
        cal,
        joint_target_raw=raw_target,
        execution_policy=actual_policy,
        execute=True,
    )

    if outcome.get("status") not in (
        "JOINT_TARGET_REACHED",
        "NO_MOTION_NEEDED",
    ):
        raise RuntimeError(
            f"{name} 未确认到位：{outcome.get('status')}"
        )

    print("  arm:", outcome["status"], flush=True)



def ask_confirm(message):
    answer = input(message + " [y=确认，其它=停止] ").strip().lower()
    if answer != "y":
        raise RuntimeError("操作员未确认本次物理动作结果；已停止后续步骤。")


def run_actions(bus, cal, targets, grips, actions, policy, source, dest):
    for category, name in actions:
        if category == "arm":
            go(bus, cal, targets[name], policy, name)
        elif category == "grip":
            print("GRIP ->", name, flush=True)
            motion.emit(grip_to(bus, grips[name], execute=True))
        elif name == "lifted":
            ask_confirm(f"检查积木是否已从 {source} 随夹爪抬起")
        elif name == "placed":
            ask_confirm(f"检查积木是否已释放到 {dest}")


def pick(bus, cal, targets, grips, source, policy):
    """Pick at an already taught slot, not an object detector."""
    run_actions(bus, cal, targets, grips, pick_actions(source), policy, source, None)


def place(bus, cal, targets, grips, dest, policy):
    """Place the currently held object in an already taught empty slot."""
    run_actions(bus, cal, targets, grips, place_actions(dest), policy, None, dest)


def transfer(bus, cal, targets, grips, source, dest, policy):
    pick(bus, cal, targets, grips, source, policy)
    place(bus, cal, targets, grips, dest, policy)
    return {"status": "TRANSFER_OPERATOR_CONFIRMED", "from": source, "to": dest,
            "observation": "manual confirmation; no camera verification"}


def rehearse(bus, cal, targets, source, dest, policy):
    """Visit the taught arm waypoints, without gripper commands or an object."""
    for category, name in steps_between(source, dest):
        if category == "arm":
            go(bus, cal, targets[name], policy, name)
    return {"status": "ARM_WAYPOINTS_REHEARSED", "gripper_acted": False,
            "object_moved": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="COM9")
    ap.add_argument("--id", default="friston_follower_12v")
    ap.add_argument("--calibration-dir", default=None)
    ap.add_argument("--file", type=Path, default=SKILLS_FILE)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="只读记录槽位抓取点或悬停点")
    c.add_argument("slot", choices=SLOTS)
    c.add_argument("phase", choices=PHASES)
    g = sub.add_parser("grip-save", help="只读记录夹爪 open/close 位置")
    g.add_argument("name", choices=GRIPS)
    gs = sub.add_parser("grip-to", help="把夹爪移动到指定归一化开度")
    gs.add_argument("percent", type=float)
    gs.add_argument("--execute", action="store_true")
    sub.add_parser("show", help="显示保存的槽位与夹爪标定")
    t = sub.add_parser("transfer", help="按已有技能完成一个物体的抓放")
    t.add_argument("source", choices=SLOTS)
    t.add_argument("destination", choices=SLOTS)
    t.add_argument("--execute", action="store_true")
    t.add_argument("--rehearse", action="store_true", help="只走手臂关键位姿，不操作夹爪")
    t.add_argument("--speed-deg-s", type=float, default=8.0)
    t.add_argument(
    "--resume-after-grasp",
    action="store_true",
    help="从夹爪闭合后的 source.hover 继续",
)
    args = ap.parse_args()
    if args.cmd == "show":
        if not args.file.is_file():
            raise SystemExit(f"尚未创建 {args.file}")
        motion.emit(json.loads(args.file.read_text(encoding="utf-8-sig")))
        return 0
    if args.cmd == "transfer" and args.source == args.destination:
        raise SystemExit("来源槽位不能等于目标槽位。")
    if args.cmd == "transfer" and args.rehearse and not args.execute:
        raise SystemExit("--rehearse 要配合 --execute；不加 --execute 就是纯预览。")
    bus = None
    try:
        robot, calibration = motion.connect_robot(args)
        bus = robot.bus
        bus.connect()
        if args.cmd == "capture":
            out = capture_slot(bus, calibration, args.file, args.id, args.slot, args.phase)
        elif args.cmd == "grip-save":
            out = capture_grip(bus, calibration, args.file, args.id, args.name)
        elif args.cmd == "grip-to":
            out = grip_to(bus, args.percent, args.execute)
        else:
            data = load_record(args.file, args.id, calibration)
            targets, grips = pose_targets(data, args.id, calibration,
                                          args.source, args.destination)
            policy = motion.ExecutionPolicy(speed_deg_s=args.speed_deg_s).validate()
            start_raw, _ = motion.prepare_live(bus, calibration)
            actions = steps_between(
                args.source, args.destination
            )

            if args.resume_after_grasp:
                # 第6个动作：夹爪闭合后返回 A.hover
                # 前5个动作已经执行，不再重复。
                actions = actions[5:]

            trace = preflight(
                start_raw, calibration, targets, grips,
                args.source, args.destination, policy,
                actions=actions,
            )
            motion.emit({"status": "PREFLIGHT_ONLY" if not args.execute else "EXECUTING_PREVIEW",
                         "steps": trace, "collision_checked": False,
                         "note": "所有姿态均从 Leader 录制；此处只验证关节规划，不验证篮沿/另一臂/线缆的碰撞。"})
            if not args.execute:
                return 0
            
            if args.rehearse:
                if args.resume_after_grasp:
                    raise ValueError(
                        "不能同时使用 rehearse 和 resume"
                    )

                out = rehearse(
                    bus, calibration, targets,
                    args.source, args.destination, policy,
                )

            elif args.resume_after_grasp:
                ask_confirm(
                    f"确认夹爪仍夹住来自 {args.source} 的积木，"
                    f"{args.destination} 为空，"
                    "后续运动路径没有障碍物"
                )

                # 接着执行剩余的动作
                # 不再重复张开夹爪或抓取
                run_actions(
                    bus,
                    calibration,
                    targets,
                    grips,
                    actions,
                    policy,
                    args.source,
                    args.destination,
                )

                out = {
                    "status": "TRANSFER_OPERATOR_CONFIRMED_RESUMED",
                    "from": args.source,
                    "to": args.destination,
                    "observation": "manual_confirmation",
                }

            else:
                out = transfer(
                    bus, calibration, targets, grips,
                    args.source, args.destination, policy,
                )

        motion.emit(out)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        motion.emit({"status": "STOPPED", "reason": str(exc) or "Ctrl+C",
                     "details": getattr(exc, "details", getattr(exc, "motion_report", {}))})
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        if bus is not None and bus.is_connected:
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    sys.exit(main())
