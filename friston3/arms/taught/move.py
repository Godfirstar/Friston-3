#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-101 ZCD MVP 4: Cartesian waypoint paths and recorded joint-space returns.

  python -m friston3.arms.taught.move --delta-mm 50 0 0 --execute
  python -m friston3.arms.taught.move --home --execute
  python -m friston3.arms.taught.move --offline-demo
Without --execute, live mode reads and plans only. Model XYZ is base_link,
not a measured camera/table frame. Position only: tool attitude may change. No collision geometry or gripper commands.
The recorded-reference return can traverse a LARGE joint arc; clear the entire
swept arm volume, not only the endpoint. Basket rims, the other arm and cables
are NOT modeled. The <=2 degree inward-only reference-return exception is a
software policy, NOT evidence that a geometric calibration mismatch is safe.
Ctrl+C attempts fresh-position hold. Communication failure can prevent it.
Disconnect preserves torque. Rate limiting applies to targets, not a measured
firmware speed guarantee. See friston3.arms.taught.start for offline pick/place recipes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import inspect
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from . import fk, start

BODY = fk.JOINT_NAMES
MOTORS = (*BODY, "gripper")
DEG_PER_TICK = 360.0 / 4095.0
PERIOD = 0.1  # Initial stationary-pose sampling interval.
MAX_IO_S = 0.30
MAX_GAP_S = 0.35
MAX_FOLLOW_DEG = 1.0


@dataclass(frozen=True)
class ExecutionPolicy:
    """Command trajectory/feedback settings, not certified physical limits."""
    speed_deg_s: float = 8.0
    rate_hz: float = 20.0
    following_error_deg: float = 5.0
    path_tolerance_mm: float = 3.0
    endpoint_tolerance_mm: float = 5.0
    min_duration_s: float = 1.0
    final_joint_tolerance_deg: float = 1.5
    settle_timeout_s: float = 6.0

    def validate(self):
        for name, value in asdict(self).items():
            require(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value > 0, f"{name} 必须为有限正数。")
        require(5 <= self.rate_hz <= 50, "rate_hz 必须在 5..50 Hz。")
        require(self.speed_deg_s / self.rate_hz >= DEG_PER_TICK,
                "速度/频率小于一个编码器刻度；降低 --rate-hz 或提高 --speed-deg-s。")
        require(self.speed_deg_s / self.rate_hz <= self.following_error_deg,
                "每周期允许步长超过跟随误差阈值；降低速度或提高频率。")
        return self

    @property
    def period(self):
        return 1.0 / self.rate_hz


class MotionError(RuntimeError):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code, self.details = code, details


def fail(code, message, **details):
    raise MotionError(code, message, **details)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def emit(data):
    print(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


def max_error(a, b):
    return max(abs(a[n] - b[n]) for n in BODY)


def read_register(bus, register, names=BODY):
    began = time.monotonic()
    values = bus.sync_read(register, motors=list(names), normalize=False, num_retry=0)
    require(time.monotonic() - began <= MAX_IO_S, f"读取 {register} 超时，停止推进。")
    require(set(values) == set(names), f"{register} 读数不完整。")
    require(all(isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v) and v == int(v) for v in values.values()),
            f"{register} 含无效读数。")
    return {n: int(values[n]) for n in names}


def read_pose(bus):
    raw = read_register(bus, "Present_Position")
    require(all(0 <= v <= 4095 for v in raw.values()), "编码器读数超出 0..4095。")
    return raw


def to_degrees(raw, calibration):
    return {n: (raw[n] - (calibration[n]["range_min"] +
                          calibration[n]["range_max"]) / 2) * DEG_PER_TICK for n in BODY}


def checked_bounds(calibration, margin_deg=1.0):
    require(math.isfinite(margin_deg) and margin_deg >= 0, "限位裕量必须是有限非负数。")
    bounds = {}
    for name, _, _, _, _, limits in fk.CHAIN:
        if limits is None:
            continue
        row = calibration[name]
        lo, hi = row["range_min"], row["range_max"]
        require(row["id"] == BODY.index(name) + 1 and row["drive_mode"] == 0,
                f"{name}: 电机 ID 或方向约定不匹配。")
        require(isinstance(lo, int) and isinstance(hi, int) and 0 <= lo < hi <= 4095,
                f"{name}: 校准范围无效。")
        lower, upper = map(math.degrees, limits)
        if not (name == "wrist_roll" and lo == 0 and hi == 4095):
            half_range = (hi - lo) / 2 * DEG_PER_TICK
            lower, upper = max(lower, -half_range), min(upper, half_range)
        require(lower + margin_deg < upper - margin_deg, f"{name}: 限位裕量消除了可用范围。")
        bounds[name] = (lower + margin_deg, upper - margin_deg)
    return bounds


def check_pose(q, bounds):
    bad = [f"{n}={q[n]:.3f}°, 允许 {bounds[n][0]:.3f}..{bounds[n][1]:.3f}°"
           for n in BODY if not math.isfinite(q[n]) or not bounds[n][0] <= q[n] <= bounds[n][1]]
    require(not bad, "START_OR_FEEDBACK_NEAR_LIMIT: " + "; ".join(bad) +
            "。先用已有遥操作离开端点，再运行；不修改校准。")


def mvp_ik_policy():
    """Position-only workspace prototype, not the old 5 mm diagnostic policy."""
    return fk.IKPolicy(margin_deg=0.5, max_joint_delta_deg=60.0,
        max_orientation_deg=180.0, position_tolerance_mm=0.5,
        max_delta_mm=100.0, max_iterations=240)


def recorded_bounds(calibration):
    """Recorded encoder sweep is never enlarged or rewritten."""
    result = {}
    for n in BODY:
        row = calibration[n]
        lo, hi = row['range_min'], row['range_max']
        require(type(lo) is int and type(hi) is int and 0 <= lo < hi <= 4095,
                f'{n}: invalid encoder calibration')
        half = (hi-lo)/2 * DEG_PER_TICK
        result[n] = (-half, half)
    return result


def _excess(x, bounds):
    return max(bounds[0]-x, 0.0, x-bounds[1])


def _raw_from_q(q, calibration):
    return {n: round(q[n]/DEG_PER_TICK +
            (calibration[n]['range_min']+calibration[n]['range_max'])/2) for n in BODY}


def _interpolate_raw(a, b, settings):
    """Integer, quintic, speed-limited setpoints; solver iterates aren't sent."""
    max_ticks = math.floor(settings.speed_deg_s * settings.period / DEG_PER_TICK)
    require(max_ticks >= 1, 'Command rate/speed is below one encoder tick.')
    span = max_error(a, b)
    steps = max(math.ceil(settings.min_duration_s/settings.period),
                math.ceil(1.875 * span/max_ticks)+1)
    points = [dict(a)]
    for i in range(1, steps+1):
        u = i/steps
        w = 10*u**3-15*u**4+6*u**5
        point = {n: round(a[n]+w*(b[n]-a[n])) for n in BODY}
        require(max_error(point, points[-1]) <= max_ticks, 'Quantized speed limit exceeded.')
        points.append(point)
    return points, max_ticks


def make_plan(raw, delta, calibration, ik_policy=None, execution_policy=None, joint_target_raw=None):
    """Cartesian: solve all 5 mm waypoints BEFORE motion, then time parameterize.

    Recorded joint return: do not impose Cartesian straightness or a 50 mm
    distance cap. A measured start <=2 deg outside nominal URDF bounds may
    return INWARD only, inside the unchanged recorded encoder sweep. This is
    a reference-return exception, never a general relaxation of motor limits.
    """
    ik_policy = ik_policy or mvp_ik_policy()
    require(isinstance(ik_policy, fk.IKPolicy), 'Invalid IK policy.')
    settings = (execution_policy or ExecutionPolicy()).validate()
    q0 = to_degrees(raw, calibration)
    physical = recorded_bounds(calibration)
    check_pose(q0, physical)
    nominal = checked_bounds(calibration, 0.0)
    interior = checked_bounds(calibration, ik_policy.margin_deg)
    p0 = fk.position_mm(q0)
    t0, _ = fk.forward_kinematics(q0)
    recovery = []
    
    if joint_target_raw is not None:
        require(
            set(joint_target_raw) == set(BODY)
            and all(
                type(v) is int and 0 <= v <= 4095
                for v in joint_target_raw.values()
            ),
            "Invalid recorded joint target."
        )

        q_target = to_degrees(joint_target_raw, calibration)

        # 记录姿态使用电机校准范围，不再使用 URDF 内缩范围
        check_pose(q_target, physical)

        # 直接在关节空间规划
        knots = [dict(raw), dict(joint_target_raw)]

        delta = [
            t - p
            for t, p in zip(fk.position_mm(q_target), p0)
        ]

        excursion = max_error(q_target, q0)

        ik_policy = replace(
            ik_policy,
            max_joint_delta_deg=min(
                180.0,
                max(ik_policy.max_joint_delta_deg, excursion + 3)
            ),
            max_orientation_deg=180.0,
        )

        result = {
            "status": "RECORDED_JOINT_PATH",
            "waypoint_count": 1,
            "cartesian_straightness_required": False,
            "recovery": [],
        }

        bounds = physical

    else:
        require(isinstance(delta, (list,tuple)) and len(delta)==3 and
                all(type(v) in (float,int) and math.isfinite(v) for v in delta), 'Invalid Cartesian delta.')
        require(fk.norm(delta) <= ik_policy.max_delta_mm+1e-8,
                f'Request exceeds {ik_policy.max_delta_mm:g} mm per Cartesian skill.')
        check_pose(q0, interior)
        count = max(1, math.ceil(fk.norm(delta)/5.0))
        knots, residuals = [dict(raw)], []
        current_q = dict(q0)
        # Each waypoint is warm-started from its predecessor. No guessed or
        # clipped live pose is substituted for the input observation.
        for k in range(1, count+1):
            target = [p0[i]+delta[i]*k/count for i in range(3)]
            offset = [target[i]-fk.position_mm(current_q)[i] for i in range(3)]
            local = replace(ik_policy, max_joint_delta_deg=min(20.0, ik_policy.max_joint_delta_deg),
                            max_delta_mm=max(10.0, fk.norm(offset)+1.0))
            r = fk.ik_probe(current_q, offset, calibration, policy=local)
            if r['status'] != 'MODEL_ONLY_IK_OK_NOT_MOTION_CLEARANCE':
                fail('WAYPOINT_IK_FAILED', f'Cartesian waypoint {k}/{count} failed; NO trajectory executed.',
                     failed_waypoint_mm=target, ik_report=r, start_joints_deg=q0)
            candidate = r['model_candidate_deg_NOT_motor_command']
            point = _raw_from_q(candidate, calibration)
            point['wrist_roll'] = raw['wrist_roll']
            current_q = to_degrees(point, calibration)
            check_pose(current_q, interior)
            require(max_error(current_q,q0) <= ik_policy.max_joint_delta_deg,
                    'Whole motion joint excursion exceeds configured bound.')
            knots.append(point)
            residuals.append(fk.norm([t-p for t,p in zip(target,fk.position_mm(current_q))]))
        bounds = interior
        result = {'status': 'CARTESIAN_WAYPOINTS_SOLVED', 'waypoint_count': count,
                  'max_quantized_waypoint_error_mm': max(residuals),
                  'orientation_mode': 'POSITION_ONLY; wrist_roll fixed; tool attitude may change'}
    path = [dict(raw)]
    # Short segments stop smoothly at each waypoint; this prioritizes simple,
    # inspectable control over a time-optimal blended trajectory.
    segment_settings = replace(settings, min_duration_s=0.15) if len(knots)>2 else settings
    for a,b in zip(knots,knots[1:]):
        segment, max_ticks = _interpolate_raw(a,b,segment_settings)
        path.extend(segment[1:])
    require(len(path) <= 100000, 'Too many trajectory setpoints.')
    length_sq = sum(v*v for v in delta)
    max_drift = max_deviation = 0.0
    previous_excess = {n: _excess(q0[n],nominal[n]) for n in BODY}
    for i, point in enumerate(path):
        q = to_degrees(point, calibration)
        check_pose(q, physical)
        check_pose(q, bounds)
        require(max_error(q,q0) <= ik_policy.max_joint_delta_deg+1e-8, 'Joint excursion exceeded.')
        t,_ = fk.forward_kinematics(q)
        drift = fk.rotation_difference_deg(t0,t)
        max_drift = max(max_drift,drift)
        require(drift <= ik_policy.max_orientation_deg+1e-8, 'Requested orientation constraint not satisfied.')
        if joint_target_raw is None:
            disp = [a-b for a,b in zip(fk.position_mm(q),p0)]
            frac = max(0.,min(1.,sum(a*b for a,b in zip(disp,delta))/length_sq)) if length_sq else 0.
            deviation = fk.norm([a-frac*b for a,b in zip(disp,delta)])
            max_deviation = max(max_deviation,deviation)
            require(deviation <= settings.path_tolerance_mm, 'Cartesian interpolation exceeds path error tolerance.')
    final_q = to_degrees(path[-1],calibration)
    error = fk.norm([a-b-d for a,b,d in zip(fk.position_mm(final_q),p0,delta)])
    require(error <= settings.endpoint_tolerance_mm, 'Quantized endpoint error exceeded.')
    result.update(max_model_orientation_change_deg=max_drift, max_model_path_deviation_mm=max_deviation)
    return {'raw_start':dict(raw),'q_start':q0,'bounds':bounds,'hard_bounds':nominal,
            'recorded_bounds':physical,'nominal_bounds':nominal,
            'path':path,'delta':list(delta),'p_start':p0,'target_q':final_q,
            'nominal_seconds':(len(path)-1)*settings.period,'ik_policy':ik_policy,
            'execution_policy':settings,'ik_report':result,'max_command_step_ticks':max_ticks,
            'quantized_model_error_mm':error,'joint_space_reference_return':joint_target_raw is not None,
            'max_start_excess':{n:_excess(q0[n],nominal[n]) for n in BODY}}


def check_feedback(raw, last_goal, plan, calibration):
    settings = plan["execution_policy"]
    q = to_degrees(raw, calibration)
    check_pose(q, plan['recorded_bounds'])
    feedback_bounds = {n: (max(plan['recorded_bounds'][n][0],lo-0.35),
                           min(plan['recorded_bounds'][n][1],hi+0.35))
                       for n,(lo,hi) in plan['bounds'].items()}
    check_pose(q, feedback_bounds)
    actual_t, _ = fk.forward_kinematics(q)
    start_t, _ = fk.forward_kinematics(plan["q_start"])
    require(fk.rotation_difference_deg(start_t, actual_t) <= plan["ik_policy"].max_orientation_deg,
            "实际反馈对应的模型姿态变化超过配置值。")
    require(max_error(q, plan["q_start"]) <= plan["ik_policy"].max_joint_delta_deg
            + settings.following_error_deg, "实际关节偏离起点过多。")
    error = max_error(raw, last_goal) * DEG_PER_TICK
    if error > settings.following_error_deg:
        fail("FOLLOWING_ERROR", f"跟随误差 {error:.3f}° 超过 {settings.following_error_deg:g}°。",
             actual_deg=error, limit_deg=settings.following_error_deg)
    return q


def fresh_goal_write(bus, target, current, following_error_deg=MAX_FOLLOW_DEG):
    require(max_error(target, current) * DEG_PER_TICK <= following_error_deg,
            f"下一个目标与当前读数相差超过 {following_error_deg:g}°，停止推进。")
    began = time.monotonic()
    bus.sync_write("Goal_Position", target, normalize=False, num_retry=0)
    require(time.monotonic() - began <= MAX_IO_S, "目标写入耗时过长，停止推进。")


def prime_and_enable(bus, plan, torque, mark_mutation):
    # Re-check the live pose before ANY mutation. No stale preview execution.
    raw = read_pose(bus)
    require(max_error(raw, plan["raw_start"]) * DEG_PER_TICK <= 0.5,
            "规划后机械臂发生移动，尚未下发；稳定后重试。")
    mark_mutation()
    fresh_goal_write(bus, raw, raw)
    require(read_register(bus, "Goal_Position") == raw, "当前位置保持目标读回不一致。")
    for name in BODY:
        if torque[name] == 0:
            # Re-prime THIS motor immediately before enabling it, with an
            # acknowledged write and goal readback. Never disable other motors.
            current = read_pose(bus)
            require(max_error(current, raw) * DEG_PER_TICK <= 0.5,
                    "开启力矩前手臂不稳定；停止。")
            began = time.monotonic()
            bus.write("Goal_Position", name, current[name], normalize=False, num_retry=0)
            require(time.monotonic() - began <= MAX_IO_S, f"{name}: 保持目标写入耗时过长。")
            require(read_register(bus, "Goal_Position", (name,))[name] == current[name],
                    f"{name}: 保持目标未确认。")
            again = read_pose(bus)
            require(max_error(again, current) * DEG_PER_TICK <= 0.5,
                    "保持目标确认期间手臂移动，停止。")
            began = time.monotonic()
            bus.enable_torque(motors=[name], num_retry=0)
            require(time.monotonic() - began <= MAX_IO_S, f"{name}: 开启力矩耗时过长。")
    require(all(v == 1 for v in read_register(bus, "Torque_Enable").values()),
            "手臂力矩未全部开启，停止。")


def run_path(bus, plan, calibration):
    settings = plan["execution_policy"]
    nominal = plan["path"][-1]
    last = plan["raw_start"]
    last_tick = time.monotonic()
    last_read = last_tick
    require(read_register(bus, "Goal_Position") == last,
            "执行前保持目标与最新起点不一致。")
    for target in plan["path"][1:]:
        time.sleep(max(0.0, settings.period - (time.monotonic() - last_tick)))
        require(time.monotonic() - last_read <= MAX_GAP_S, "控制循环延迟；停止，不跳步追赶。")
        raw = read_pose(bus)
        require(time.monotonic() - last_read <= MAX_GAP_S, "反馈间隔过长；停止推进。")
        last_read = time.monotonic()
        check_feedback(raw, last, plan, calibration)
        fresh_goal_write(bus, target, raw, settings.following_error_deg)
        last = target
        last_tick = time.monotonic()
    deadline, stable = time.monotonic() + settings.settle_timeout_s, 0
    recent_raw = []
    while time.monotonic() < deadline:
        time.sleep(settings.period)
        raw = read_pose(bus)
        recent_raw = (recent_raw + [raw])[-4:]
        quiet = len(recent_raw) == 4 and all(
            max(p[n] for p in recent_raw)-min(p[n] for p in recent_raw) <= 1 for n in BODY)
        require(time.monotonic() - last_read <= MAX_GAP_S, "等待到位时反馈间隔过长。")
        last_read = time.monotonic()
        q = check_feedback(raw, last, plan, calibration)
        displacement = [a - b for a, b in zip(fk.position_mm(q), plan["p_start"])]
        model_error = fk.norm([a - b for a, b in zip(displacement, plan["delta"])])
        length_sq = sum(v*v for v in plan["delta"])
        progress = (sum(a * b for a, b in zip(displacement, plan["delta"])) / length_sq
                    if length_sq > 1e-12 else 1.0)
        # 关节角反馈是否达到目标精度
        joint_ok = (
            max_error(q, plan["target_q"])
            <= settings.final_joint_tolerance_deg
        )

        if plan["joint_space_reference_return"]:
            # 示教记录的关节目标：
            # 直接根据真实关节反馈判断是否到位。
            # FK 末端误差仍保留用于诊断，不作为额外门槛。
            reached = joint_ok
        else:
            # 笛卡尔运动继续使用末端位置判定
            reached = (
                joint_ok
                and model_error <= settings.endpoint_tolerance_mm
                and (
                    length_sq <= settings.endpoint_tolerance_mm**2
                    or progress >= 0.6
                )
            )
            stable = stable + 1 if reached else 0
        if stable >= max(3, math.ceil(0.5/settings.period)):
            require(read_register(bus, "Goal_Position") == last, "最终目标读回不一致。")
            require(all(v == 1 for v in read_register(bus, "Torque_Enable").values()),
                    "到位时发现力矩关闭。")
            return {"status": "JOINT_TARGET_REACHED", "joints_deg": q,
                    "model_position_mm": fk.position_mm(q),
                    "model_delta_mm": displacement, "model_endpoint_error_mm": model_error,
                    "model_displacement_norm_mm": fk.norm(displacement),
                    "physical_cartesian_accuracy_verified": False,
                    "settle_correction_deg": {n: 0.0 for n in BODY},
                    "torque_on_exit": "ARM_REMAINS_ENABLED; gripper unchanged"}
        # Do not integrate more effort into a blocked/stationary arm. The
        # existing position servo keeps its nominal goal; timeout reports failure.
    fail("TARGET_NOT_REACHED", "反馈未满足到位条件。",
         final_joints_deg=q, target_joints_deg=plan["target_q"],
         joint_error_deg={n: q[n]-plan["target_q"][n] for n in BODY},
         model_endpoint_error_mm=model_error, model_progress=progress,
         final_raw=raw, goal_raw=last,
         goal_readback=read_register(bus, "Goal_Position"),
         motor_velocity_raw=read_register(bus, "Present_Velocity"),
         motor_load_raw=read_register(bus, "Present_Load"))


def attempt_hold(bus, calibration, bounds=None):
    try:
        raw = read_pose(bus)
        # A stop may cross a configurable planning margin, never the hard bounds.
        hold_bounds = recorded_bounds(calibration)
        check_pose(to_degrees(raw, calibration), hold_bounds)
        fresh_goal_write(bus, raw, raw)
        require(read_register(bus, "Goal_Position") == raw, "保持目标读回不一致。")
        return "FRESH_HOLD_GOAL_CONFIRMED; physical stop is not guaranteed"
    except Exception as exc:
        return f"HOLD_NOT_CONFIRMED: {exc}; 电机可能仍追向最后目标，请使用现场断电手段。"


def connect_robot(args):
    Robot, Config, _ = start.load_driver()
    paths = ([Path(args.calibration_dir).expanduser() / f"{args.id}.json"]
             if args.calibration_dir else start.calibration_candidates(Robot, args.id))
    paths = [p for p in paths if p.is_file()]
    require(len(paths) == 1, "需要唯一的已有校准文件；可用 --calibration-dir 指定。")
    robot = Robot(Config(port=args.port, id=args.id, calibration_dir=paths[0].parent,
                         cameras={}, use_degrees=True, disable_torque_on_disconnect=False))
    require(tuple(robot.bus.motors) == MOTORS, "电机名称/顺序不是预期 SO-101。")
    require(all(m.id == i and m.model == "sts3215"
                for i, m in enumerate(robot.bus.motors.values(), 1)), "电机 ID/型号不匹配。")
    for method, required in (("sync_read", ("motors", "normalize", "num_retry")),
                             ("sync_write", ("normalize", "num_retry")),
                             ("write", ("normalize", "num_retry")),
                             ("enable_torque", ("motors", "num_retry")),
                             ("disconnect", ("disable_torque",))):
        require(all(n in inspect.signature(getattr(robot.bus, method)).parameters for n in required),
                f"当前驱动 {method} 接口不兼容；停止，不猜测调用。")
    calibration = {n: asdict(robot.calibration[n]) for n in BODY}
    checked_bounds(calibration, 0.0)
    return robot, calibration


def plan_summary(plan):
    settings = plan["execution_policy"]
    ik = plan["ik_report"]
    return {"start_joints_deg": plan["q_start"], "target_joints_deg": plan["target_q"],
            "delta_mm_in_base_link": plan["delta"], "minimum_seconds": plan["nominal_seconds"],
            "quantized_model_error_mm": plan["quantized_model_error_mm"],
            "ik_report": {k: ik[k] for k in ("status", "position_residual_mm", "orientation_change_deg",
                "candidate_delta_deg", "termination", "failure_reasons", "waypoint_count",
                "recovery", "orientation_mode", "max_model_orientation_change_deg",
                "max_model_path_deviation_mm") if k in ik},
            "ik_settings": asdict(plan["ik_policy"]), "execution_settings": asdict(settings),
            "command_speed_cap_after_quantization_deg_s": plan["max_command_step_ticks"] * DEG_PER_TICK / settings.period,
            "collision_checked": False, "gripper_commands": False,
            "warning": "篮沿、另一只机械臂和线缆未建模；需要现场看护。FK 是模型反馈，不是外部测量。"}


def prepare_live(bus, calibration):
    require(bool(bus.is_calibrated), "校准不匹配；停止，不写回。")
    require(all(v == 0 for v in read_register(bus, "Operating_Mode").values()),
            "存在非位置模式电机；停止，不修改模式。")
    torque = read_register(bus, "Torque_Enable")
    require(all(v in (0, 1) for v in torque.values()), "力矩状态异常。")
    raw = read_pose(bus)
    began = time.monotonic()
    normalized = bus.sync_read("Present_Position", motors=list(BODY), num_retry=0)
    require(time.monotonic() - began <= MAX_IO_S and set(normalized) == set(BODY)
            and all(math.isfinite(float(v)) for v in normalized.values()), "角度读取失败或耗时过长。")
    require(max_error(normalized, to_degrees(raw, calibration)) <= 0.5,
            "原始读数与驱动角度不一致或手臂正在移动。")
    time.sleep(PERIOD)
    newer = read_pose(bus)
    require(max_error(newer, raw) * DEG_PER_TICK <= 0.5, "手臂尚未稳定。")
    return newer, torque


def move_ee(bus, calibration, *, delta_mm=None, target_mm=None, ik_policy=None,
            execution_policy=None, execute=False, on_mutation=None, reporter=None, joint_target_raw=None):
    """One measured-start move, with hold-on-error. Caller owns bus connection.

    XYZ is the model base frame, not a camera/table frame. No gripper commands.
    """
    require(sum(v is not None for v in (delta_mm, target_mm, joint_target_raw)) == 1,
            "必须指定一个位移、绝对位置或已记录关节目标。")
    ik_policy = ik_policy or mvp_ik_policy()
    settings = (execution_policy or ExecutionPolicy()).validate()
    mutation_started = False
    plan = None
    def mark():
        nonlocal mutation_started
        mutation_started = True
        if on_mutation is not None:
            on_mutation()
    try:
        raw, torque = prepare_live(bus, calibration)
        p0 = fk.position_mm(to_degrees(raw, calibration))
        if joint_target_raw is not None:
            target_mm = fk.position_mm(to_degrees(joint_target_raw, calibration))
        supplied = target_mm if target_mm is not None else delta_mm
        require(isinstance(supplied, (tuple, list)) and len(supplied) == 3
                and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                        and math.isfinite(v) for v in supplied), "目标必须是三个有限数值。")
        absolute_target = list(target_mm) if target_mm is not None else [p+d for p,d in zip(p0, delta_mm)]
        delta = [t-p for t,p in zip(absolute_target, p0)]
        plan = make_plan(raw, delta, calibration, ik_policy, settings, joint_target_raw)
        summary = plan_summary(plan)
        if not execute:
            return {"status": "PREVIEW_ONLY_NO_WRITES", **summary}
        if plan["path"][-1] == raw:
            return {"status": "NO_MOTION_NEEDED", **summary, "motion_was_attempted": False}
        if reporter:
            reporter({"status": "READY_TO_EXECUTE", **summary})
        prime_and_enable(bus, plan, torque, mark)
        time.sleep(settings.period)
        after = read_pose(bus)
        require(max_error(after, raw) * DEG_PER_TICK <= 0.5, "开启力矩后位置变化过大，停止。")
        # Preserve the originally requested absolute endpoint after torque capture.
        after_position = fk.position_mm(to_degrees(after, calibration))
        plan = make_plan(after, [t-p for t,p in zip(absolute_target, after_position)],
                         calibration, ik_policy, settings, joint_target_raw)
        latest = read_pose(bus)
        require(max_error(latest, after) * DEG_PER_TICK <= 0.5,
                f"重新规划期间位置变化 {max_error(latest, after) * DEG_PER_TICK:.3f}°，超过 0.5°。")
        fresh_goal_write(bus, after, latest)
        result = run_path(bus, plan, calibration)
        result["model_delta_mm"] = [p-s for p,s in zip(result["model_position_mm"], p0)]
        result["requested_target_mm"] = absolute_target
        return {**result, "motion_was_attempted": True, "plan": plan_summary(plan)}
    except (Exception, KeyboardInterrupt) as exc:
        details = dict(getattr(exc, "details", {}))
        details["motion_was_attempted"] = mutation_started
        if mutation_started and bus.is_connected:
            details["hold_result"] = attempt_hold(bus, calibration)
        if isinstance(exc, KeyboardInterrupt):
            exc.motion_report = details
            raise
        raise MotionError(getattr(exc, "code", "EXECUTION_STOPPED"), str(exc), **details) from exc


def home_reference(bus, calibration, robot_id, path, save=False):
    """Record or load a measured pose; return uses a joint-space path."""
    signature = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
    if save:
        raw, _ = prepare_live(bus, calibration)
        check_pose(to_degrees(raw, calibration), checked_bounds(calibration, 0.5))
        data = {"robot_id": robot_id, "calibration_sha256": signature, "raw_position": raw,
                "joints_deg": to_degrees(raw, calibration),
                "model_position_mm": fk.position_mm(to_degrees(raw, calibration)),
                "note": "现场记录参考位；关节空间返回，未进行碰撞规划；夹爪不归位。"}
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
        return {"status": "HOME_SAVED_NO_MOTION", "file": str(path), **data}
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    require(data["robot_id"] == robot_id and data["calibration_sha256"] == signature,
            "归位文件的机械臂 ID 或校准版本不匹配。")
    raw = data["raw_position"]
    require(set(raw) == set(BODY) and all(type(v) is int and 0 <= v <= 4095 for v in raw.values()),
            "归位记录无效。")
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="COM9")
    parser.add_argument("--id", default="friston_follower_12v")
    parser.add_argument("--calibration-dir")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--delta-mm", type=float, nargs=3, metavar=("DX", "DY", "DZ"))
    target.add_argument("--target-mm", type=float, nargs=3, metavar=("X", "Y", "Z"))
    target.add_argument("--home", action="store_true", help="返回已记录参考位；加 --execute 执行")
    target.add_argument("--save-home", action="store_true", help="只读记录当前参考位，拒绝覆盖")
    parser.add_argument("--home-file", default=str(Path(__file__).with_name("so101_zcd_home.json")))
    parser.add_argument("--execute", action="store_true", help="实际移动；省略时只读规划")
    parser.add_argument("--offline-demo", action="store_true", help="合成姿态验证，不打开串口")
    parser.add_argument("--speed-deg-s", type=float, default=8.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--following-error-deg", type=float, default=3.0)
    parser.add_argument("--path-tolerance-mm", type=float, default=3.0)
    parser.add_argument("--endpoint-tolerance-mm", type=float, default=3.0)
    fk.add_policy_arguments(parser)
    parser.set_defaults(margin_deg=0.5, max_joint_delta_deg=60.0, max_orientation_deg=180.0,
                        ik_tolerance_mm=0.5, max_delta_mm=100.0, ik_iterations=240)
    descriptions = {
        '--margin-deg': '目标关节的附加边界裕量，MVP默认0.5度；不修改校准',
        '--max-joint-delta-deg': '一次笛卡尔技能的关节变化上限，MVP默认60度',
        '--max-orientation-deg': '默认180度：位置任务不保持夹爪朝向；不适用于抓取姿态约束',
        '--ik-tolerance-mm': 'MVP默认0.5 mm的模型求解容差，不是真机精度',
        '--max-delta-mm': '一次笛卡尔技能位移长度上限，MVP默认100 mm',
        '--ik-iterations': '每个局部路点的最大IK迭代次数，MVP默认240'}
    for action in parser._actions:
        for flag in action.option_strings:
            if flag in descriptions:
                action.help = descriptions[flag]
    args = parser.parse_args()
    bus = None
    try:
        ik_policy = fk.policy_from_args(args)
        settings = ExecutionPolicy(speed_deg_s=args.speed_deg_s, rate_hz=args.rate_hz,
            following_error_deg=args.following_error_deg, path_tolerance_mm=args.path_tolerance_mm,
            endpoint_tolerance_mm=args.endpoint_tolerance_mm).validate()
        if args.delta_mm is None and args.target_mm is None and not (args.home or args.save_home):
            args.delta_mm = [50.0, 0.0, 0.0]
        if args.offline_demo:
            require(not (args.execute or args.home or args.save_home), "离线演示不能执行或访问归位记录。")
            q = dict(zip(BODY, (0.0, -30.0, 30.0, -30.0, 0.0)))
            calibration = {n: {"id": i+1, "drive_mode": 0, "range_min": 0, "range_max": 4095}
                           for i,n in enumerate(BODY)}
            raw = {n: round(2047.5+q[n]/DEG_PER_TICK) for n in BODY}
            p0 = fk.position_mm(to_degrees(raw, calibration))
            delta = args.delta_mm if args.target_mm is None else [t-p for t,p in zip(args.target_mm, p0)]
            emit({"status": "SYNTHETIC_PLAN_ONLY_NO_HARDWARE", **plan_summary(
                make_plan(raw, delta, calibration, ik_policy, settings))})
            return 0
        robot, calibration = connect_robot(args)
        bus = robot.bus
        bus.connect()  # Avoid robot.connect(): no configure/calibrate/PID writes.
        if args.save_home:
            require(not args.execute, "保存参考位是只读操作，不使用 --execute。")
            emit(home_reference(bus, calibration, args.id, args.home_file, save=True))
            return 0
        home_raw = home_reference(bus, calibration, args.id, args.home_file) if args.home else None
        emit(move_ee(bus, calibration, delta_mm=args.delta_mm, target_mm=args.target_mm,
                     ik_policy=ik_policy, execution_policy=settings, execute=args.execute,
                     reporter=emit, joint_target_raw=home_raw))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        details = getattr(exc, "motion_report", getattr(exc, "details", {}))
        emit({"status": "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "STOPPED",
              "reason": str(exc) or "Ctrl+C", "code": getattr(exc, "code", "INPUT_OR_CONNECTION_ERROR"),
              "motion_was_attempted": False, **details})
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        if bus is not None and bus.is_connected:
            try:
                bus.disconnect(disable_torque=False)
            except Exception as exc:
                print(f"串口关闭失败；力矩状态未改变：{exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
