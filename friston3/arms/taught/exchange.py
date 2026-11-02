
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZCD 第一步：交换计划检查 + SO-101 夹爪控制。

依赖同包中的 start 和 move 模块。

本文件不发送五个机械臂关节的运动命令。
"""

from __future__ import annotations

import argparse
import math
import sys
import time

from . import start as symbol
from . import move as motion


# 独立于模型声明的真实任务目标
EXPECTED = {
    "red_block": "B",
    "blue_block": "A",
}


def check_exchange(plan: dict) -> dict:
    """检查模型是否真的完成了交换任务。"""

    result = symbol.validate_plan(plan)

    if result.get("status") != "symbolic_check_passed":
        raise ValueError(f"模型没有产生合法计划：{result}")

    if plan["goal"] != EXPECTED:
        raise ValueError(
            f"目标理解错误：需要 {EXPECTED}，"
            f"模型却声明 {plan['goal']}"
        )

    if result["predicted_final_state"] != EXPECTED:
        raise ValueError("计划没有实现红蓝积木交换。")

    return result


def self_test():
    # 这仅仅是测试用例，不会作为答案提供给 LLM。
    actions = [
        ("pick", "red_block"),
        ("place", "C"),
        ("pick", "blue_block"),
        ("place", "A"),
        ("pick", "red_block"),
        ("place", "B"),
    ]

    good = {
        "status": "plan",
        "reason": "离线自测",
        "goal": dict(EXPECTED),
        "steps": [
            {"skill": skill, "arg": arg}
            for skill, arg in actions
        ],
    }

    assert check_exchange(good)["predicted_final_state"] == EXPECTED

    # 特意测试：模型错误地认为不需要交换
    wrong = {
        "status": "plan",
        "reason": "没有交换",
        "goal": dict(symbol.INITIAL),
        "steps": [],
    }

    try:
        check_exchange(wrong)
    except ValueError:
        pass
    else:
        raise AssertionError("没有拒绝错误目标！")

    print("PASS: 正确交换计划通过；错误目标被拒绝。")
    print("没有连接机器人。")


def gripper_state(bus) -> dict:
    """读取夹爪真实位置和力矩状态。"""

    raw = motion.read_register(
        bus, "Present_Position", ("gripper",)
    )["gripper"]

    position = bus.sync_read(
        "Present_Position",
        motors=["gripper"],
        num_retry=0,
    )

    if set(position) != {"gripper"}:
        raise RuntimeError("夹爪位置读数缺失。")

    value = float(position["gripper"])

    if not math.isfinite(value) or not -0.5 <= value <= 100.5:
        raise RuntimeError(f"夹爪读数异常：{value}")

    torque = motion.read_register(
        bus, "Torque_Enable", ("gripper",)
    )["gripper"]

    mode = motion.read_register(
        bus, "Operating_Mode", ("gripper",)
    )["gripper"]

    if torque not in (0, 1) or mode != 0:
        raise RuntimeError("夹爪不在预期的位置控制模式。")

    return {
        "raw": raw,
        "percent": value,
        "torque_enabled": bool(torque),
    }


def hold_gripper(bus):
    """异常时尝试保持当前夹爪位置。"""

    try:
        now = gripper_state(bus)["percent"]

        bus.sync_write(
            "Goal_Position",
            {"gripper": now},
            num_retry=0,
        )

        return "已尝试保持当前位置（不是硬件急停）"

    except Exception as exc:
        return f"保持未确认：{exc}"


def run_gripper(bus, delta: float, execute: bool) -> dict:
    """只控制第六个夹爪电机。"""

    if not bus.is_calibrated:
        raise RuntimeError("电机校准与文件不匹配，停止。")

    if not math.isfinite(delta) or abs(delta) > 10:
        raise ValueError("单次位移不能超过10个百分比点。")

    before = gripper_state(bus)

    if delta == 0:
        return {
            "status": "GRIPPER_READ_ONLY",
            "before": before,
        }

    goal = before["percent"] + delta

    if not 0 <= goal <= 100:
        raise ValueError(
            f"目标 {goal:.2f} 超出标定的0..100范围。"
        )

    summary = {
        "before": before,
        "target_percent": round(goal, 2),
        "note": "正负方向尚未与实际张开/闭合对应。",
    }

    if not execute:
        return {
            "status": "GRIPPER_PREVIEW_NO_WRITES",
            **summary,
        }

    try:
        # 只准备夹爪电机，不改变五个机械臂关节目标
        if not before["torque_enabled"]:

            bus.write(
                "Goal_Position",
                "gripper",
                before["percent"],
                num_retry=0,
            )

            bus.enable_torque(
                motors=["gripper"],
                num_retry=0,
            )

            if not gripper_state(bus)["torque_enabled"]:
                raise RuntimeError("夹爪力矩开启未确认。")

        else:
            bus.sync_write(
                "Goal_Position",
                {"gripper": before["percent"]},
                num_retry=0,
            )

        current = before["percent"]

        # 分段下发，不一次发送较大的位置变化
        steps = max(1, math.ceil(abs(delta) / 2))

        for i in range(1, steps + 1):
            target = (
                before["percent"]
                + delta * i / steps
            )

            actual = gripper_state(bus)["percent"]

            if abs(actual - current) > 7:
                raise RuntimeError(
                    "夹爪未跟上前一目标，停止。"
                )

            bus.sync_write(
                "Goal_Position",
                {"gripper": target},
                num_retry=0,
            )

            current = target
            time.sleep(0.12)

        # 读取实际反馈
        deadline = time.monotonic() + 1.5

        while time.monotonic() < deadline:
            final = gripper_state(bus)

            if abs(final["percent"] - goal) <= 3:
                return {
                    "status": "GRIPPER_FEEDBACK_OK",
                    **summary,
                    "after": final,
                }

            time.sleep(0.1)

        raise RuntimeError(
            f"夹爪未到目标；实际={final['percent']:.2f}，"
            f"目标={goal:.2f}"
        )

    except BaseException:
        print(
            "夹爪中止：" + hold_gripper(bus),
            file=sys.stderr,
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    sub.add_parser("self-test")

    p = sub.add_parser("plan")
    p.add_argument(
        "instruction",
        nargs="?",
        default="把红色积木和蓝色积木的位置交换一下",
    )
    p.add_argument("--model", default="qwen3.5:9b")
    p.add_argument("--timeout", type=float, default=180)

    g = sub.add_parser("gripper")
    g.add_argument("--delta", type=float, default=0)
    g.add_argument("--port", default="COM9")
    g.add_argument("--id", default="friston_follower_12v")
    g.add_argument("--calibration-dir", default=None)
    g.add_argument("--execute", action="store_true")

    args = parser.parse_args()
    bus = None

    try:
        if args.command == "self-test":
            self_test()

        elif args.command == "plan":
            plan = symbol.request_plan(
                args.instruction,
                args.model,
                args.timeout,
            )

            checked = check_exchange(plan)

            symbol.show({
                "status": "EXCHANGE_PLAN_OK_NOT_EXECUTED",
                "plan": plan,
                "check": checked,
                "expected_goal": EXPECTED,
            })

        else:
            robot, _ = motion.connect_robot(args)
            bus = robot.bus
            bus.connect()

            symbol.show(
                run_gripper(
                    bus,
                    args.delta,
                    args.execute,
                )
            )

        return 0

    except (Exception, KeyboardInterrupt) as exc:
        print(
            f"STOP: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 130 if isinstance(exc, KeyboardInterrupt) else 1

    finally:
        if bus is not None and bus.is_connected:
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    raise SystemExit(main())
