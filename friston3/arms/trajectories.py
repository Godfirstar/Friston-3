"""Validated, locally taught joint trajectories; never contains stock real poses."""
from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from .config import ARM_NAMES, JOINTS

FORMAT = "friston-arm-trajectory-v1"
UNITS = "degrees_gripper_percent"
_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,47}$")


def validate_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError("动作名须以小写英文字母开头，只含小写字母、数字、_、-，最多48字符")
    return name


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_motion(value, name=None, max_duration_s=30.0):
    if not isinstance(value, dict) or value.get("format") != FORMAT or value.get("units") != UNITS:
        raise ValueError("动作格式或单位不匹配（转动关节为度，夹爪为0..100）")
    actual_name = validate_name(value.get("name"))
    if name is not None and actual_name != name:
        raise ValueError("动作文件名与内部名称不一致")
    profiles = value.get("profiles")
    if (not isinstance(profiles, dict) or not profiles or set(profiles) - set(ARM_NAMES)):
        raise ValueError("动作必须绑定已示教的 left/right 从臂")
    for profile in profiles.values():
        if (not isinstance(profile, dict) or not isinstance(profile.get("id"), str)
                or not isinstance(profile.get("calibration_digest"), str)
                or not profile["calibration_digest"]):
            raise ValueError("动作缺少机械臂 ID 或标定摘要")
    frames = value.get("frames")
    if not isinstance(frames, list) or not 3 <= len(frames) <= 3000:
        raise ValueError("动作必须包含3..3000个有效采样点")
    previous = -1.0
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or set(frame) != {"t", "positions"}:
            raise ValueError("动作采样点格式错误")
        timestamp = frame["t"]
        if (not _finite(timestamp) or timestamp <= previous or timestamp > max_duration_s + 0.1
                or (index == 0 and timestamp != 0)):
            raise ValueError("动作时间须从0开始、严格递增且不超过时长限制")
        previous = timestamp
        positions = frame["positions"]
        if not isinstance(positions, dict) or set(positions) != set(profiles):
            raise ValueError("每一帧必须包含相同的从臂集合")
        for pose in positions.values():
            if (not isinstance(pose, dict) or set(pose) != set(JOINTS)
                    or any(not _finite(number) for number in pose.values())):
                raise ValueError("每只从臂必须具有完整六关节有限数值")
            if not -0.01 <= pose["gripper.pos"] <= 100.01:
                raise ValueError("夹爪超出0..100范围")
    if previous < 0.1:
        raise ValueError("动作时间过短")
    if value.get("mode") not in ("real", "mock"):
        raise ValueError("动作缺少 real/mock 模式标记")
    return value


def load_motion(directory, name, max_duration_s=30.0):
    path = Path(directory) / (validate_name(name) + ".json")
    if path.stat().st_size > 4_000_000:
        raise ValueError("动作文件过大")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"动作 {name} 不是有效JSON") from error
    return validate_motion(value, name, max_duration_s)


def within_pose(actual, target, config):
    for arm, pose in target.items():
        if arm not in actual:
            return False
        for joint, value in pose.items():
            tolerance = config["gripper_tolerance"] if joint == "gripper.pos" else config["start_tolerance_deg"]
            if abs(actual[arm][joint] - value) > tolerance:
                return False
    return True


def motion_summary(value, config):
    return {"name": value["name"], "arms": list(value["profiles"]),
            "duration_s": round(value["frames"][-1]["t"], 3), "mode": value["mode"],
            "looped": within_pose(value["frames"][-1]["positions"], value["frames"][0]["positions"], config)}


def motion_catalog(config):
    result, invalid = [], []
    directory = Path(config["motions_dir"])
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            try:
                result.append(motion_summary(load_motion(directory, path.stem, config["max_duration_s"]), config))
            except (OSError, ValueError, KeyError, TypeError) as error:
                invalid.append({"name": path.stem, "error": str(error)})
    return result, invalid


def check_compatibility(value, profiles, config):
    if value["mode"] != config["mode"]:
        raise ValueError("模拟动作与实机动作不可混用")
    for arm, saved in value["profiles"].items():
        actual = profiles.get(arm)
        if not actual or any(saved.get(key) != actual.get(key) for key in ("id", "calibration_digest")):
            raise ValueError(f"{arm} 的ID或标定与录制时不一致；请使用原标定或重新示教")
        for frame in value["frames"]:
            for joint, number in frame["positions"][arm].items():
                low, high = actual["limits"][joint]
                if not low - 0.01 <= number <= high + 0.01:
                    raise ValueError(f"动作中的 {arm}.{joint} 超出本机标定范围")


def interpolate(value, elapsed):
    frames = value["frames"]
    if elapsed >= frames[-1]["t"]:
        return deepcopy(frames[-1]["positions"])
    if elapsed <= 0:
        return deepcopy(frames[0]["positions"])
    index = bisect_right([frame["t"] for frame in frames], elapsed)
    left, right = frames[index - 1], frames[index]
    weight = (elapsed - left["t"]) / (right["t"] - left["t"])
    return {arm: {joint: number + (right["positions"][arm][joint] - number) * weight
                  for joint, number in pose.items()}
            for arm, pose in left["positions"].items()}


def save_recording(config, name, profiles, frames):
    value = validate_motion({"format": FORMAT, "units": UNITS, "name": validate_name(name),
        "mode": config["mode"], "created_at": datetime.now(timezone.utc).isoformat(),
        "profiles": {arm: {key: entry[key] for key in ("id", "calibration_digest")}
                     for arm, entry in profiles.items()}, "frames": frames}, name, config["max_duration_s"])
    check_compatibility(value, profiles, config)
    directory = Path(config["motions_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (name + ".json")
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    # A temporary file and an exclusive hard link give complete/no-overwrite
    # publication on NTFS and normal local POSIX filesystems.
    import os
    import tempfile
    descriptor, temporary = tempfile.mkstemp(prefix=".recording-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination
