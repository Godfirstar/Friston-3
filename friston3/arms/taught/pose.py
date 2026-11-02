
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZCD: 保存 Ready / Park 姿态并执行关节空间归位。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import move as motion


NAMES = ("ready", "park")
DEFAULT_FILE = Path(__file__).with_name("so101_zcd_poses.json")


def signature(calibration):
    # 与现有 move.py 的 home_reference 签名算法一致
    value = json.dumps(calibration, sort_keys=True)
    return hashlib.sha256(value.encode()).hexdigest()


def load_file(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"姿态记录不存在：{path}；请先 capture。"
        )

    data = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(data, dict) or not isinstance(
        data.get("poses"), dict
    ):
        raise ValueError("姿态文件格式错误。")

    return data


def check_identity(data, robot_id, calibration):
    if data.get("robot_id") != robot_id:
        raise ValueError("姿态文件与机械臂 ID 不匹配。")

    if data.get("calibration_sha256") != signature(calibration):
        raise ValueError(
            "校准版本已改变，原来的姿态记录不能继续使用。"
        )


def checked_target(data, name, calibration):
    row = data["poses"].get(name)

    if not isinstance(row, dict):
        raise ValueError(f"还没有保存 {name} 姿态。")

    raw = row.get("raw_position")

    if not isinstance(raw, dict) or set(raw) != set(motion.BODY):
        raise ValueError("关节名称不匹配。")

    if not all(
        type(v) is int and 0 <= v <= 4095
        for v in raw.values()
    ):
        raise ValueError("保存的编码器目标无效。")


    return raw


def capture(bus, calibration, robot_id, path, name, replace):
    """只读保存当前真实姿态，不发送运动命令。"""

    raw, _ = motion.prepare_live(bus, calibration)
    q = motion.to_degrees(raw, calibration)


    if path.exists():
        data = load_file(path)
        check_identity(data, robot_id, calibration)
    else:
        data = {
            "robot_id": robot_id,
            "calibration_sha256": signature(calibration),
            "poses": {},
        }

    if name in data["poses"] and not replace:
        raise ValueError(
            f"{name} 已保存；添加 --replace 才能覆盖。"
        )

    data["poses"][name] = {
        "raw_position": raw,
        "joints_deg": q,
        "model_position_mm": motion.fk.position_mm(q),
        "recorded_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    # 原子替换，保留另一个已记录姿态
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)

    return {
        "status": "POSE_CAPTURED_NO_MOTION",
        "name": name,
        "file": str(path),
        "pose": data["poses"][name],
    }



def go_to(bus, calibration, data, name, execute, speed):
    """执行已保存的关节姿态，不附加人为运动跨度上限。"""

    target_raw = checked_target(data, name, calibration)

    policy = motion.ExecutionPolicy(
        speed_deg_s=speed
    ).validate()

    result = motion.move_ee(
        bus,
        calibration,
        joint_target_raw=target_raw,
        execution_policy=policy,
        execute=execute,
    )

    return {
        "name": name,
        "result": result,
    }



def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--port", default="COM9")
    parser.add_argument(
        "--id", default="friston_follower_12v"
    )
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument(
        "--file", type=Path, default=DEFAULT_FILE
    )

    sub = parser.add_subparsers(
        dest="command", required=True
    )

    # capture ready / capture park
    p = sub.add_parser("capture")
    p.add_argument("name", choices=NAMES)
    p.add_argument("--replace", action="store_true")

    # show：不打开串口
    sub.add_parser("show")

    # goto ready / goto park
    p = sub.add_parser("goto")
    p.add_argument("name", choices=NAMES)
    p.add_argument("--execute", action="store_true")
    p.add_argument(
        "--speed-deg-s", type=float, default=8.0
    )

    args = parser.parse_args()
    bus = None

    try:
        if args.command == "show":
            data = load_file(args.file)
            motion.emit({
                "status": "SAVED_POSES_FILE_ONLY",
                **data,
            })
            return 0

        robot, calibration = motion.connect_robot(args)
        bus = robot.bus

        # 直接连接电机总线，不调用 robot.connect()
        bus.connect()

        if args.command == "capture":
            result = capture(
                bus, calibration, args.id,
                args.file, args.name, args.replace
            )
        else:
            data = load_file(args.file)
            check_identity(data, args.id, calibration)

            result = go_to(
    bus, calibration, data, args.name,
    args.execute, args.speed_deg_s,
)

        motion.emit(result)
        return 0

    except (Exception, KeyboardInterrupt) as exc:
        motion.emit({
            "status": (
                "INTERRUPTED"
                if isinstance(exc, KeyboardInterrupt)
                else "STOPPED"
            ),
            "reason": str(exc) or "Ctrl+C",
            "code": getattr(exc, "code", "POSE_ERROR"),
            **getattr(
                exc,
                "motion_report",
                getattr(exc, "details", {}),
            ),
        })
        return (
            130 if isinstance(exc, KeyboardInterrupt)
            else 1
        )

    finally:
        if bus is not None and bus.is_connected:
            try:
                bus.disconnect(disable_torque=False)
            except Exception as exc:
                print(
                    f"串口关闭失败，力矩状态未确认：{exc}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    sys.exit(main())
