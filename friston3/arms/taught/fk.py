#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-101 ZCD: forward-kinematics diagnostics only; no motion controller.

Usage (run from the Friston-3 folder):
  python -m friston3.arms.taught.fk reported
  python -m friston3.arms.taught.fk live --port COM9 --id friston_follower_12v
  python -m friston3.arms.taught.fk file so101_pose.json
  python -m friston3.arms.taught.fk self-test
  python -m friston3.arms.taught.fk ik-demo
  python -m friston3.arms.taught.fk live --probe-mm 0 0 5

reported/file/self-test use only the Python standard library and never import
LeRobot or open a serial port. 'reported' uses the historical joint readings
pasted into the conversation, NOT a fresh hardware observation.

'live' delegates ONLY to read_arm() in the start module next to
this file. That original reader blocks common motor writes and retains torque
state on disconnect. Close other COM9 clients and support the arm beforehand.

The model below is a kinematic-parameter excerpt, NOT the full official URDF.
Joint origins, axes and limits come from TheRobotStudio/SO-ARM100's
Simulation/SO101/so101_new_calib.urdf, checked on 2026-10-30. No meshes,
inertias, collision geometry, dynamics or moving-jaw geometry are loaded.
Identity mapping of the five LeRobot angles to URDF angles is an UNVERIFIED
assumption. No ad-hoc offsets, clipping, calibration writes or motion commands
are performed. Output must not be used as a motion clearance.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from typing import Any

VERSION = "3.0-configurable-ik"

MODEL_SOURCE = (
    "https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/"
    "Simulation/SO101/so101_new_calib.urdf"
)
JOINT_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"
)
# (joint name, parent link, child link, origin xyz in metres, origin rpy in
# radians, revolute limits in radians or None for the fixed tool frame).
# All FIVE revolute axes in this exact excerpt are (0, 0, 1) in joint frame.
CHAIN = (
    ("shoulder_pan", "base_link", "shoulder_link",
     (0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159),
     (-1.91986, 1.91986)),
    ("shoulder_lift", "shoulder_link", "upper_arm_link",
     (-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0.0),
     (-1.74533, 1.74533)),
    ("elbow_flex", "upper_arm_link", "lower_arm_link",
     (-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708),
     (-1.69, 1.69)),
    ("wrist_flex", "lower_arm_link", "wrist_link",
     (-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -1.5708),
     (-1.65806, 1.65806)),
    ("wrist_roll", "wrist_link", "gripper_link",
     (5.55112e-17, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159),
     (-2.74385, 2.84121)),
    ("gripper_frame_joint", "gripper_link", "gripper_frame_link",
     (-0.0079, -0.000218121, -0.0981274), (0.0, 3.14159, 0.0), None),
)
# Historical observation supplied by the user; not synthetic, not live.
REPORTED = {
    "time_utc": "2026-10-30T12:47:26.464240+00:00",
    "port": "COM9", "id": "friston_follower_12v",
    "mode": "read_only_no_motion_commands",
    "calibration_matches_motor": True,
    "joints_deg": {
        "shoulder_pan": -2.10989010989011,
        "shoulder_lift": -98.06593406593407,
        "elbow_flex": 97.0989010989011,
        "wrist_flex": 71.38461538461539,
        "wrist_roll": 1.5384615384615385,
    },
    "gripper_0_100": 0.27137042062415195,
}
Matrix = list[list[float]]


def identity() -> Matrix:
    return [[float(i == j) for j in range(4)] for i in range(4)]


def multiply(a: Matrix, b: Matrix) -> Matrix:
    return [[sum(a[i][k] * b[k][j] for k in range(4))
             for j in range(4)] for i in range(4)]


def origin_transform(xyz: tuple[float, ...], rpy: tuple[float, ...]) -> Matrix:
    """URDF convention: translation followed by Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr, xyz[0]],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr, xyz[1]],
        [-sp, cp*sr, cp*cr, xyz[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_z(angle_rad: float) -> Matrix:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数值，不能是布尔值或字符串。")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} 含 NaN 或无穷大。")
    return result


def checked_joints(joints: object) -> dict[str, float]:
    if not isinstance(joints, dict) or set(joints) != set(JOINT_NAMES):
        raise ValueError("joints_deg 必须恰好包含五个手臂关节；不要把夹爪开合量当角度。")
    values = {name: finite_number(joints[name], name) for name in JOINT_NAMES}
    if any(abs(q) > 180.0 for q in values.values()):
        raise ValueError("角度绝对值超过 180 度；检查是否误用了编码器原始值。")
    return values


def forward_kinematics(joints: object) -> tuple[Matrix, dict[str, list[float]]]:
    """Compute an UNVERIFIED pose in base_link coordinates; never clip angles."""
    q = checked_joints(joints)
    transform = identity()
    frames = {"base_link": [0.0, 0.0, 0.0]}
    last_link = "base_link"
    for name, parent, child, xyz, rpy, limits in CHAIN:
        if parent != last_link:
            raise ValueError("内置运动链不连续。")
        # Each link: parent-to-joint origin, then rotation in the joint frame.
        transform = multiply(transform, origin_transform(xyz, rpy))
        if limits is not None:
            transform = multiply(transform, rotation_z(math.radians(q[name])))
        frames[child] = [transform[i][3] for i in range(3)]
        last_link = child
    return transform, frames


def model_limit_notes(joints: dict[str, float]) -> list[dict[str, Any]]:
    """URDF limits are model data, not verified hardware safety limits."""
    notes = []
    for name, _, _, _, _, limits in CHAIN:
        if limits is None:
            continue
        low, high = (math.degrees(value) for value in limits)
        margin = min(joints[name] - low, high - joints[name])
        if margin < 5.0:
            notes.append({
                "joint": name, "value_deg": round(joints[name], 6),
                "model_range_deg": [round(low, 6), round(high, 6)],
                "signed_margin_deg": round(margin, 6),
                "status": "outside_model_limits" if margin < 0 else "within_5deg_of_model_limit",
            })
    return notes


def make_report(snapshot: object, source_mode: str) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        raise ValueError("输入必须是 read 子命令生成的 JSON 对象。")
    if snapshot.get("mode") != "read_only_no_motion_commands":
        raise ValueError("输入不是已知只读读数格式；请使用原脚本 read --output 保存。")
    if snapshot.get("calibration_matches_motor") is not True:
        raise ValueError("输入未确认电机校准信息匹配；停止，不计算可用于后续核对的位姿。")
    q = checked_joints(snapshot.get("joints_deg"))
    grip = finite_number(snapshot.get("gripper_0_100"), "gripper_0_100")
    if not 0.0 <= grip <= 100.0:
        raise ValueError("夹爪归一化读数超出 0..100。")
    transform, frames = forward_kinematics(q)
    notes = model_limit_notes(q)
    warnings = [
        "q_urdf=q_lerobot 仅是假设；尚未实测确认零点、方向和工具偏移。",
        "这是 base_link 到 gripper_frame_link 的模型估计，不是相机或尺子实测。",
        "基座原点不一定在桌面；本程序没有定义桌面，也没有检查碰撞。",
        "夹爪开合量不参与这条固定参考点运动链；未计算真实夹持中心随开合的变化。",
    ]
    if transform[2][3] < 0:
        warnings.append("模型末端参考点 z<0：先核对实际姿态和基座/桌面关系，不据此自动修正关节。")
    if notes:
        warnings.append("有读数接近或超出 URDF 标称范围；只报告，不截断、不改标定、不下发动作。")
    return {
        "mode": source_mode,
        "status": "MODEL_ESTIMATE_UNVERIFIED",
        "motion_allowed": False,
        "source_time_utc": snapshot.get("time_utc"),
        "robot_id": snapshot.get("id"),
        "base_frame": "base_link", "target_frame": "gripper_frame_link",
        "mapping_assumption": "q_urdf_deg = q_lerobot_deg (not physically verified)",
        "joints_deg": q,
        "gripper_0_100_excluded_from_fk": grip,
        "tool_position_mm": {a: round(1000*transform[i][3], 3)
                             for i, a in enumerate(("x", "y", "z"))},
        "model_limit_notes": notes,
        "warnings": warnings,
        "T_base_tool_metres": [[round(v, 9) for v in row] for row in transform],
        "link_origins_mm": {k: [round(1000*v, 3) for v in pos] for k, pos in frames.items()},
        "model_source": MODEL_SOURCE,
        "model_source_checked_date": "2026-10-30",
        "model_excerpt_sha256": hashlib.sha256(
            json.dumps(CHAIN, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def read_live(args: argparse.Namespace) -> dict[str, Any]:
    """Reuse the existing read-only reader. Never create a new motor controller."""
    from .start import read_arm

    capture = io.StringIO()
    options = argparse.Namespace(
        port=args.port, id=args.id, calibration_dir=args.calibration_dir, output=None
    )
    try:
        with contextlib.redirect_stdout(capture):
            snapshot = read_arm(options)
    except BaseException:
        print(capture.getvalue(), end="", file=sys.stderr)
        raise
    return snapshot


def calibration_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Read existing local calibration for inspection; never modify it."""
    path = Path(snapshot["calibration_file"])
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    result = {}
    for name in JOINT_NAMES:
        row = data[name]
        result[name] = {k: row[k] for k in
                        ("id", "drive_mode", "homing_offset", "range_min", "range_max")}
    return result


def self_test() -> None:
    count = 0
    def check(condition: bool, message: str) -> None:
        nonlocal count
        if not condition:
            raise AssertionError(message)
        count += 1
    t = rotation_z(math.pi/2)
    check(abs(t[0][0]) < 1e-12 and abs(t[1][0]-1) < 1e-12, "Rz sign")
    a = origin_transform((1.0, 2.0, 3.0), (0.0, 0.0, 0.0))
    check([a[i][3] for i in range(3)] == [1.0, 2.0, 3.0], "translation")
    check(multiply(identity(), a) == a, "matrix product")
    transform, frames = forward_kinematics(REPORTED["joints_deg"])
    # Independent SciPy Rotation('xyz') reference, computed from the same
    # official model parameters; this validates software, not the real arm.
    reference = (0.134507677, 0.003302153, -0.016662668)
    check(max(abs(transform[i][3]-reference[i]) for i in range(3)) < 1e-8, "FK reference")
    check(transform[3] == [0.0, 0.0, 0.0, 1.0], "homogeneous matrix")
    check(len(frames) == 7, "frame count")
    report = make_report(REPORTED, "offline_reported_snapshot")
    check(report["motion_allowed"] is False, "motion disabled")
    check(any(n["joint"] == "elbow_flex" and n["status"] == "outside_model_limits"
              for n in report["model_limit_notes"]), "outside limit reporting")
    changed = {**REPORTED, "gripper_0_100": 99.0}
    check(make_report(changed, "test")["tool_position_mm"] == report["tool_position_mm"],
          "gripper is excluded from fixed-frame FK")
    invalid = [
        {**REPORTED, "calibration_matches_motor": False},
        {**REPORTED, "joints_deg": {}},
        {**REPORTED, "joints_deg": {**REPORTED["joints_deg"], "shoulder_pan": float("nan")}},
        {**REPORTED, "joints_deg": {**REPORTED["joints_deg"], "shoulder_pan": True}},
        {**REPORTED, "joints_deg": {**REPORTED["joints_deg"], "shoulder_pan": 2077}},
        {**REPORTED, "joints_deg": {**REPORTED["joints_deg"], "gripper": 0.0}},
        {**REPORTED, "gripper_0_100": -1.0},
    ]
    for item in invalid:
        try:
            make_report(item, "test")
        except ValueError:
            count += 1
        else:
            raise AssertionError("invalid observation accepted")
    print(f"{count} 项离线测试通过；未导入 LeRobot，未连接串口，未验证真机几何精度。")


ACTIVE_IK_JOINTS = JOINT_NAMES[:4]  # Keep wrist_roll and gripper unchanged.


@dataclass(frozen=True)
class IKPolicy:
    """Application planning thresholds, not measured mechanical safety limits."""
    margin_deg: float = 1.0
    max_joint_delta_deg: float = 15.0
    max_orientation_deg: float = 15.0
    position_tolerance_mm: float = 0.25
    max_delta_mm: float = 50.0
    max_iterations: int = 160

    def __post_init__(self):
        for name in ("margin_deg", "max_joint_delta_deg", "max_orientation_deg",
                     "position_tolerance_mm", "max_delta_mm"):
            value = finite_number(getattr(self, name), name)
            if value < 0 or (name not in ("margin_deg", "max_orientation_deg") and value == 0):
                raise ValueError(f"{name}: 必须为有限的非负值（位移及容差必须大于零）。")
        if self.max_joint_delta_deg > 180 or self.max_orientation_deg > 180:
            raise ValueError("关节变化/朝向筛查阈值不可超过 180°。")
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int) or not 1 <= self.max_iterations <= 2000:
            raise ValueError("max_iterations 必须为 1..2000 的整数。")


def add_policy_arguments(parser) -> None:
    parser.add_argument("--margin-deg", type=float, help="模型/标定边界内的额外裕量，默认 1°")
    parser.add_argument("--max-joint-delta-deg", type=float, help="一次规划的关节变化上限，默认 15°")
    parser.add_argument("--max-orientation-deg", type=float, help="位置 IK 的朝向变化筛查上限，默认 15°；不是姿态控制")
    parser.add_argument("--ik-tolerance-mm", type=float, help="连续角度模型的位置残差容差，默认 0.25 mm")
    parser.add_argument("--max-delta-mm", type=float, help="一次规划的末端位移长度上限，默认 50 mm")
    parser.add_argument("--ik-iterations", type=int, help="局部 IK 最大迭代次数，默认 160")


def policy_from_args(args) -> IKPolicy:
    policy = IKPolicy()
    names = {"margin_deg": "margin_deg", "max_joint_delta_deg": "max_joint_delta_deg",
             "max_orientation_deg": "max_orientation_deg", "ik_tolerance_mm": "position_tolerance_mm",
             "max_delta_mm": "max_delta_mm", "ik_iterations": "max_iterations"}
    changes = {field: getattr(args, arg) for arg, field in names.items()
               if getattr(args, arg, None) is not None}
    return replace(policy, **changes)


def norm(values: list[float]) -> float:
    return math.sqrt(sum(x*x for x in values))


def solve_3x3(a: Matrix, b: list[float]) -> list[float]:
    """Small pivoted Gaussian solve; no third-party linear algebra dependency."""
    rows = [list(a[i]) + [b[i]] for i in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda i: abs(rows[i][col]))
        if abs(rows[pivot][col]) < 1e-15:
            raise ValueError("IK 线性方程接近奇异；停止，不产生运动命令。")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = rows[col][col]
        rows[col] = [v / scale for v in rows[col]]
        for i in range(3):
            if i != col:
                factor = rows[i][col]
                rows[i] = [rows[i][k] - factor*rows[col][k] for k in range(4)]
    return [rows[i][3] for i in range(3)]


def position_mm(q: dict[str, float]) -> list[float]:
    t, _ = forward_kinematics(q)
    return [1000.0*t[i][3] for i in range(3)]


def rotation_difference_deg(a: Matrix, b: Matrix) -> float:
    trace = sum(a[i][j]*b[i][j] for i in range(3) for j in range(3))
    return math.degrees(math.acos(max(-1.0, min(1.0, (trace-1.0)/2.0))))


def recorded_range_info(q: dict[str, float], calibration: dict[str, Any],
                        raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Inspect, never edit calibration. Full-turn roll range is not a hard stop."""
    result: dict[str, Any] = {}
    for name in JOINT_NAMES:
        row = calibration[name]
        low = finite_number(row["range_min"], name+".range_min")
        high = finite_number(row["range_max"], name+".range_max")
        if not 0 <= low < high <= 4095:
            raise ValueError(f"{name}: 不支持的校准范围。")
        if row.get("drive_mode") != 0:
            raise ValueError(f"{name}: 本诊断尚未验证 drive_mode != 0 的映射。")
        mid = (low+high)/2.0
        half_range = (high-low)/2.0*360.0/4095.0
        full_turn_placeholder = name == "wrist_roll" and low == 0 and high == 4095
        item: dict[str, Any] = {
            "recorded_range_deg": [-half_range, half_range],
            "angle_margin_deg": round(half_range-abs(q[name]), 6),
            "full_turn_placeholder_not_measured_stops": full_turn_placeholder,
        }
        if raw is not None:
            val = finite_number(raw[name], "raw_position."+name)
            if not 0 <= val <= 4095:
                raise ValueError(f"{name}: 原始位置超出已验证的编码器范围。")
            expected_deg = (val-mid)*360.0/4095.0
            item.update({
                "raw_position": val,
                "raw_margin_ticks": min(val-low, high-val),
                "raw_vs_degrees_discrepancy_deg": round(abs(expected_deg-q[name]), 6),
            })
            # Original reader performs raw and degree reads separately. Do not
            # silently combine observations taken while the arm was moving.
            if abs(expected_deg-q[name]) > 0.25:
                raise ValueError(f"{name}: 两次读取不一致，机械臂可能移动了；保持稳定后重读。")
        result[name] = item
    return result


def ik_probe(joints: object, delta_mm: object,
             calibration: dict[str, Any] | None = None,
             raw: dict[str, Any] | None = None,
             policy: IKPolicy | None = None) -> dict[str, Any]:
    """POSITION-ONLY local IK in an UNVERIFIED model. Never sends motor commands.

    Uses damped least squares with central-difference position Jacobian (mm/deg).
    Virtual trial iterates are projected into the configured bounds. The supplied
    observation is NEVER clipped. A candidate endpoint is NOT a trajectory.
    """
    q0 = checked_joints(joints)
    policy = policy if policy is not None else IKPolicy()
    if not isinstance(policy, IKPolicy):
        raise TypeError("policy 必须为 IKPolicy。")
    if not isinstance(delta_mm, (list, tuple)) or len(delta_mm) != 3:
        raise ValueError("probe-mm 必须为三个数，单位毫米。")
    delta = [finite_number(v, "probe_mm") for v in delta_mm]
    if norm(delta) > policy.max_delta_mm + 1e-9:
        raise ValueError(f"REQUEST_EXCEEDS_MAX_DELTA: {norm(delta):.6g} mm > {policy.max_delta_mm:g} mm。")
    initial_t, _ = forward_kinematics(q0)
    start = position_mm(q0)
    target = [start[i]+delta[i] for i in range(3)]
    report: dict[str, Any] = {
        "status": "UNSOLVED",
        "motion_allowed": False,
        "mode": "OFFLINE_POSITION_IK_NO_MOTOR_COMMANDS",
        "reference_frame": "base_link_NOT_measured_world_or_table",
        "mapping": "q_urdf=q_lerobot_UNVERIFIED",
        "start_position_mm": [round(v, 6) for v in start],
        "requested_delta_mm": delta,
        "target_position_mm": [round(v, 6) for v in target],
        "orientation_constraint": "NONE; wrist_roll_fixed; orientation_drift_reported_and_screened",
        "collision_check": "NOT_IMPLEMENTED",
        "path_check": "NOT_IMPLEMENTED; optimizer_iterates_are_NOT_a_motion_path",
        "diagnostic_policy": asdict(policy),
        "failure_reasons": [],
        "mapping_validation": "NOT_TESTABLE_BY_IK_CONVERGENCE; needs measured geometry/pose",
        "warning": "即使数值求解通过，也不代表真实位置准确、姿态保持或运动安全。",
    }
    recorded = recorded_range_info(q0, calibration, raw) if calibration is not None else None
    report["recorded_ranges_available"] = recorded is not None
    if recorded is not None:
        report["recorded_range_checks"] = recorded
    bounds: dict[str, tuple[float, float]] = {}
    hard_bounds: dict[str, tuple[float, float]] = {}
    blockers = []
    for name, _, _, _, _, limits in CHAIN:
        if limits is None:
            continue
        lo, hi = [math.degrees(v) for v in limits]
        # Compare both model bounds and the user's recorded sweep. Neither is a
        # hardware safety guarantee. wrist_roll's full-range record is a sentinel.
        if recorded is not None and not recorded[name]["full_turn_placeholder_not_measured_stops"]:
            rlo, rhi = recorded[name]["recorded_range_deg"]
            lo, hi = max(lo, rlo), min(hi, rhi)
        hard_bounds[name] = (lo, hi)
        bounds[name] = (lo+policy.margin_deg, hi-policy.margin_deg)
        if not bounds[name][0] <= q0[name] <= bounds[name][1]:
            blockers.append({"joint": name, "current_deg": q0[name],
                             "diagnostic_interior_deg": list(bounds[name]),
                             "model_and_recorded_bounds_deg": [lo, hi],
                             "signed_boundary_margin_deg": min(q0[name]-lo, hi-q0[name]),
                             "code": "START_OUTSIDE_JOINT_BOUNDS" if not lo <= q0[name] <= hi
                                     else "START_INSIDE_CONFIGURED_MARGIN"})
    report["model_and_recorded_bounds_deg"] = hard_bounds
    report["planning_bounds_deg"] = bounds
    if blockers:
        report.update({"status": "START_POSE_NEAR_OR_OUTSIDE_LIMITS", "blockers": blockers,
                       "failure_reasons": blockers, "termination": "START_REJECTED_NO_SOLVE",
                       "next_step": "用已验证的遥操作缓慢离开折叠/端点位，停止占用串口后重读；不要直接发送指定关节角。"})
        return report
    search_bounds = {n: (max(bounds[n][0], q0[n]-policy.max_joint_delta_deg),
                         min(bounds[n][1], q0[n]+policy.max_joint_delta_deg)) for n in JOINT_NAMES}
    q = dict(q0)
    h = 0.001  # degrees; only virtual FK perturbations, never hardware dither.
    damping = 0.05  # mm/degree
    stalled = False
    termination = "ITERATION_LIMIT"
    iteration = 0
    projected_trials = 0
    for iteration in range(policy.max_iterations+1):
        current = position_mm(q)
        err = [target[i]-current[i] for i in range(3)]
        if norm(err) <= policy.position_tolerance_mm:
            termination = "POSITION_TOLERANCE_REACHED"
            break
        if iteration == policy.max_iterations:
            break
        jac = [[0.0]*len(ACTIVE_IK_JOINTS) for _ in range(3)]
        for j, name in enumerate(ACTIVE_IK_JOINTS):
            plus, minus = dict(q), dict(q)
            plus[name] += h
            minus[name] -= h
            pp, pm = position_mm(plus), position_mm(minus)
            for i in range(3):
                jac[i][j] = (pp[i]-pm[i])/(2*h)
        normal = [[sum(jac[i][k]*jac[j][k] for k in range(len(ACTIVE_IK_JOINTS)))
                   + (damping*damping if i == j else 0.0) for j in range(3)] for i in range(3)]
        y = solve_3x3(normal, err)
        dq = [sum(jac[i][j]*y[i] for i in range(3)) for j in range(len(ACTIVE_IK_JOINTS))]
        accepted = False
        # Projection lets free joints continue improving after one joint reaches
        # a bound. Projected gradient is a descent fallback when DLS points out
        # of the feasible region. Neither operation touches hardware or q0.
        gradient = [sum(jac[i][j]*err[i] for i in range(3)) for j in range(len(ACTIVE_IK_JOINTS))]
        for direction in (dq, gradient):
            scale = min(1.0, 0.5/max(1e-12, max(abs(v) for v in direction)))
            for _ in range(18):
                trial = dict(q)
                for name, change in zip(ACTIVE_IK_JOINTS, direction):
                    unbounded = q[name] + change*scale
                    trial[name] = max(search_bounds[name][0], min(search_bounds[name][1], unbounded))
                    projected_trials += int(trial[name] != unbounded)
                if norm([target[i]-v for i, v in enumerate(position_mm(trial))]) < norm(err)-1e-12:
                    q, accepted = trial, True
                    break
                scale *= 0.5
            if accepted:
                break
        if not accepted:
            stalled = True
            termination = "LOCAL_OPTIMIZER_STALLED"
            break
    final_t, _ = forward_kinematics(q)
    final_pos = position_mm(q)
    residual = norm([target[i]-final_pos[i] for i in range(3)])
    rotation_change = rotation_difference_deg(initial_t, final_t)
    solved = residual <= policy.position_tolerance_mm
    if not solved:
        status = "NO_LOCAL_SOLUTION_WITHIN_DIAGNOSTIC_POLICY"
    elif rotation_change > policy.max_orientation_deg:
        status = "POSITION_SOLVED_BUT_ORIENTATION_DRIFT_TOO_LARGE"
    else:
        status = "MODEL_ONLY_IK_OK_NOT_MOTION_CLEARANCE"
    active = []
    for n in JOINT_NAMES:
        for side, value in zip(("lower", "upper"), search_bounds[n]):
            if abs(q[n]-value) <= 1e-5:
                source = "JOINT_EXCURSION_LIMIT" if abs(abs(value-q0[n])-policy.max_joint_delta_deg) <= 1e-6 else "JOINT_BOUND_WITH_MARGIN"
                active.append({"joint": n, "side": side, "constraint": source, "value_deg": value})
    reasons = []
    if not solved:
        reasons.append({"code": "POSITION_RESIDUAL_EXCEEDS_TOLERANCE", "actual_mm": residual,
                        "allowed_mm": policy.position_tolerance_mm})
        reasons.extend({"code": a["constraint"]+"_ACTIVE", **a} for a in active)
        reasons.append({"code": termination, "iterations": iteration,
                        "note": "局部搜索未通过；不证明全局无解，也不证明零点模型正确或错误。"})
    if rotation_change > policy.max_orientation_deg:
        reasons.append({"code": "ORIENTATION_DRIFT_EXCEEDS_LIMIT", "actual_deg": rotation_change,
                        "allowed_deg": policy.max_orientation_deg,
                        "note": "本求解器只优化位置；朝向为筛查项，可能存在其他朝向更合适的解。"})
    report.update({"status": status, "iterations": iteration, "line_search_stalled": stalled,
                   "termination": termination, "active_constraints": active,
                   "projected_virtual_trial_components": projected_trials,
                   "failure_reasons": reasons,
                   "checks": {
                       "position_residual": {"passed": solved, "actual_mm": residual,
                                             "allowed_mm": policy.position_tolerance_mm},
                       "orientation_drift": {"passed": rotation_change <= policy.max_orientation_deg,
                                             "actual_deg": rotation_change, "allowed_deg": policy.max_orientation_deg},
                       "max_joint_change": {"passed": max(abs(q[n]-q0[n]) for n in JOINT_NAMES) <= policy.max_joint_delta_deg+1e-9,
                                            "actual_deg": max(abs(q[n]-q0[n]) for n in JOINT_NAMES),
                                            "allowed_deg": policy.max_joint_delta_deg}},
                   "position_residual_mm": round(residual, 8),
                   "orientation_change_deg": round(rotation_change, 6),
                   "model_candidate_deg_NOT_motor_command": q,
                   "candidate_delta_deg": {n: round(q[n]-q0[n], 6) for n in JOINT_NAMES},
                   "candidate_position_mm": [round(v, 6) for v in final_pos]})
    return report


def ik_demo(policy: IKPolicy | None = None) -> dict[str, Any]:
    # Synthetic interior pose: NOT a home pose or a request to place the real arm.
    q = dict(zip(JOINT_NAMES, (0.0, -30.0, 30.0, -30.0, 0.0)))
    return {"source": "SYNTHETIC_MATH_TEST_NOT_YOUR_ARM_NOT_A_HOME_COMMAND",
            "input_model_joints_deg": q, "ik_probe": ik_probe(q, [0.0, 0.0, 5.0], policy=policy)}


def ik_self_test() -> None:
    count = 0
    def check(condition: bool, message: str) -> None:
        nonlocal count
        if not condition:
            raise AssertionError(message)
        count += 1
    check(max(abs(x-y) for x, y in zip(solve_3x3(
        [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]], [2.0, 6.0, 12.0]), [1, 2, 3])) < 1e-12,
        "linear solver")
    result = ik_demo()["ik_probe"]
    check(result["status"] == "MODEL_ONLY_IK_OK_NOT_MOTION_CLEARANCE", "IK demo")
    check(result["position_residual_mm"] < IKPolicy().position_tolerance_mm, "IK residual")
    check(result["motion_allowed"] is False, "motion remains prohibited")
    check(result["candidate_delta_deg"]["wrist_roll"] == 0.0, "roll fixed")
    q = dict(zip(JOINT_NAMES, (0.0, -30.0, 30.0, -30.0, 0.0)))
    zero = ik_probe(q, [0.0, 0.0, 0.0])
    check(zero["model_candidate_deg_NOT_motor_command"] == q, "zero displacement")
    blocked = ik_probe(REPORTED["joints_deg"], [0.0, 0.0, 5.0])
    check(blocked["status"] == "START_POSE_NEAR_OR_OUTSIDE_LIMITS", "historical start blocked")
    check("model_candidate_deg_NOT_motor_command" not in blocked, "no clipped seed")
    check(REPORTED["joints_deg"]["elbow_flex"] == 97.0989010989011, "input unchanged")
    for bad in ([0, 0, 50.01], [40, 40, 0], [float("nan"), 0, 0], [True, 0, 0], [0, 0]):
        try:
            ik_probe(q, bad)
        except ValueError:
            count += 1
        else:
            raise AssertionError("bad offset accepted")
    # Regression fixture reproduced against v2.0: wrist_flex hit its 3-degree
    # excursion cap and a generic status hid the 2.78541mm residual.
    constrained_q = dict(zip(JOINT_NAMES, (0, -60, -60, -30, 0)))
    strict = IKPolicy(margin_deg=5.0, max_joint_delta_deg=3.0, max_orientation_deg=2.0,
                      position_tolerance_mm=0.02, max_delta_mm=5.0, max_iterations=80)
    constrained = ik_probe(constrained_q, [0, 0, 5], policy=strict)
    check(constrained["status"] == "NO_LOCAL_SOLUTION_WITHIN_DIAGNOSTIC_POLICY", "bounded failure fixture")
    check(any(r["code"] == "JOINT_EXCURSION_LIMIT_ACTIVE" and r["joint"] == "wrist_flex"
              for r in constrained["failure_reasons"]), "specific limiting joint exposed")
    check(any(r["code"] == "POSITION_RESIDUAL_EXCEEDS_TOLERANCE"
              for r in constrained["failure_reasons"]), "residual failure exposed")
    mvp = ik_probe(constrained_q, [0, 0, 5])
    check(mvp["status"] == "MODEL_ONLY_IK_OK_NOT_MOTION_CLEARANCE", "MVP fixture converges")
    check(mvp["checks"]["position_residual"]["passed"] and mvp["failure_reasons"] == [], "MVP metrics agree")
    check(mvp["candidate_delta_deg"]["wrist_roll"] == 0, "MVP roll fixed")
    orientation = ik_probe(constrained_q, [0, 0, 5], policy=replace(IKPolicy(), max_orientation_deg=2))
    check(orientation["status"] == "POSITION_SOLVED_BUT_ORIENTATION_DRIFT_TOO_LARGE", "orientation remains independent")
    one_step = ik_probe(q, [0, 0, 5], policy=replace(IKPolicy(), max_iterations=1))
    check(one_step["termination"] == "ITERATION_LIMIT", "iteration limit exposed")
    near = dict(q, shoulder_lift=-99.5)
    check(ik_probe(near, [0, 0, 0])["blockers"][0]["code"] == "START_INSIDE_CONFIGURED_MARGIN", "margin distinguished")
    zero_margin = ik_probe(near, [0, 0, 0], policy=replace(IKPolicy(), margin_deg=0))
    check(zero_margin["status"] == "MODEL_ONLY_IK_OK_NOT_MOTION_CLEARANCE", "margin configurable")
    outside = dict(q, shoulder_lift=-101.0)
    check(ik_probe(outside, [0, 0, 0], policy=replace(IKPolicy(), margin_deg=0))["blockers"][0]["code"]
          == "START_OUTSIDE_JOINT_BOUNDS", "hard model limit retained")
    calibration = {n: {"drive_mode": 0, "range_min": 1000, "range_max": 3000} for n in JOINT_NAMES}
    check(ik_probe(near, [0, 0, 0], calibration, policy=replace(IKPolicy(), margin_deg=0))["blockers"][0]["code"]
          == "START_OUTSIDE_JOINT_BOUNDS", "recorded range retained")
    check(constrained_q == dict(zip(JOINT_NAMES, (0, -60, -60, -30, 0))), "solver leaves observation unchanged")
    for overrides in ({"margin_deg": -1}, {"max_joint_delta_deg": float("nan")},
                      {"position_tolerance_mm": 0}, {"max_delta_mm": -1},
                      {"max_orientation_deg": 181}, {"max_iterations": True}):
        try:
            replace(IKPolicy(), **overrides)
        except ValueError:
            count += 1
        else:
            raise AssertionError(f"bad policy accepted: {overrides}")
    json.dumps(mvp, allow_nan=False)
    check(True, "JSON contains only finite metrics")
    print(f"{count} 项新增 IK 离线测试通过；不连接串口，不包含动作下发。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    p = sub.add_parser("ik-demo", help="合成姿态上的离线 IK 自测；不要把测试角度发给真机")
    add_policy_arguments(p)
    p = sub.add_parser("reported", help="复算对话中的历史读数；不连接机械臂")
    p.add_argument("--output", help="保存报告 JSON；拒绝覆盖")
    p = sub.add_parser("file", help="读取原脚本 read --output 保存的 JSON")
    p.add_argument("snapshot")
    p.add_argument("--output", help="保存报告 JSON；拒绝覆盖")
    p = sub.add_parser("live", help="调用原脚本，只读当前关节并计算 FK")
    p.add_argument("--port", default="COM9")
    p.add_argument("--id", default="friston_follower_12v")
    p.add_argument("--calibration-dir")
    p.add_argument("--output", help="保存报告 JSON；拒绝覆盖")
    for name in ("reported", "file", "live"):
        add_policy_arguments(sub.choices[name])
        sub.choices[name].add_argument("--probe-mm", type=float, nargs=3, metavar=("DX", "DY", "DZ"),
                                      help="只计算基座坐标位移；默认上限 50 mm，可用参数调整，永不下发")
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
            ik_self_test()
            return 0
        if args.command == "ik-demo":
            print(json.dumps(ik_demo(policy_from_args(args)), ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        if args.output and Path(args.output).exists():
            raise FileExistsError(f"拒绝覆盖已有文件：{args.output}")
        if args.command == "reported":
            snapshot, mode = REPORTED, "offline_reported_snapshot_NOT_live"
        elif args.command == "file":
            snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8-sig"))
            mode = "offline_snapshot_file_NOT_live"
        else:
            snapshot, mode = read_live(args), "live_read_only_fk"
        report = make_report(snapshot, mode)
        report["script_version"] = VERSION
        calibration = None
        if args.command == "live":
            try:
                calibration = calibration_summary(snapshot)
                report["calibration_summary_read_only"] = calibration
                report["recorded_range_checks"] = recorded_range_info(
                    snapshot["joints_deg"], calibration, snapshot.get("raw_position"))
            except (OSError, KeyError, TypeError, ValueError) as exc:
                report["calibration_summary_error"] = str(exc)
        if args.probe_mm is not None:
            if args.command == "live" and "calibration_summary_error" in report:
                report["ik_probe"] = {"status": "CALIBRATION_INSPECTION_FAILED",
                                      "motion_allowed": False,
                                      "reason": report["calibration_summary_error"]}
            else:
                report["ik_probe"] = ik_probe(snapshot["joints_deg"], args.probe_mm,
                                               calibration, snapshot.get("raw_position"), policy_from_args(args))
        text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        print(text)
        if args.output:
            with Path(args.output).open("x", encoding="utf-8") as f:
                f.write(text + "\n")
        return 0
    except KeyboardInterrupt:
        print("已中断；本程序没有运动控制功能。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"STOP [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
