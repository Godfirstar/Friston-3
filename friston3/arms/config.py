"""Configuration shared by the LeRobot service and dependency-free clients."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import secrets

JOINTS = tuple(name + ".pos" for name in (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"))
ARM_NAMES = ("left", "right")
GESTURES = ("rest", "wave", "welcome", "point_pickup", "shrug")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


def default_config_path() -> Path:
    explicit = os.environ.get("FRISTON_ARMS_CONFIG", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    local = os.environ.get("LOCALAPPDATA")
    root = Path(local) if local else Path(os.environ.get(
        "XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return (root / "Friston3" / "arms" / "config.json").resolve()


def _number(value, label, minimum, maximum):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not minimum <= value <= maximum):
        raise ValueError(f"{label} 必须是 {minimum}..{maximum} 范围的有限数值")
    return float(value)


def validate_config(value: dict, path: Path) -> dict:
    if not isinstance(value, dict) or value.get("protocol") != 1:
        raise ValueError("机械臂配置 protocol 必须是 1")
    if value.get("host") != "127.0.0.1":
        raise ValueError("机械臂服务仅绑定本机 127.0.0.1")
    port = value.get("port")
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("服务 port 必须是 1024..65535 的整数")
    token = value.get("token")
    if not isinstance(token, str) or not 24 <= len(token) <= 256 or not token.isascii() or any(c.isspace() for c in token):
        raise ValueError("机械臂服务需要有效的本机访问 token；请重新生成配置")
    if value.get("mode") not in ("real", "mock"):
        raise ValueError("mode 仅支持 real 或 mock")
    arms = value.get("arms")
    if not isinstance(arms, dict) or not arms or set(arms) - set(ARM_NAMES):
        raise ValueError("arms 必须配置 left 和/或 right")
    ports = []
    for name, entry in [*arms.items(), ("leader", value.get("leader"))]:
        if name == "leader" and not entry:
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"{name} 配置必须是对象")
        arm_port, arm_id = entry.get("port"), entry.get("id")
        if not isinstance(arm_port, str) or not arm_port.strip() or len(arm_port) > 256:
            raise ValueError(f"{name} 缺少串口")
        if not isinstance(arm_id, str) or not _IDENTIFIER.fullmatch(arm_id):
            raise ValueError(f"{name} 的 id 必须是已有标定使用的英文、数字、下划线或连字符名称")
        ports.append(arm_port.strip().casefold())
        calibration_dir = entry.get("calibration_dir")
        if calibration_dir is not None:
            if not isinstance(calibration_dir, str) or not calibration_dir.strip():
                raise ValueError(f"{name}.calibration_dir 应是目录路径或 null")
            candidate = Path(calibration_dir).expanduser()
            entry["calibration_dir"] = str((path.parent / candidate).resolve()
                                           if not candidate.is_absolute() else candidate.resolve())
    if len(ports) != len(set(ports)):
        raise ValueError("左右从臂和主臂必须使用不同的串口")
    fields = {
        "fps": (20.0, 5.0, 50.0),
        "speed_deg_s": (20.0, 1.0, 60.0),
        "gripper_speed_s": (40.0, 1.0, 100.0),
        "max_relative_target": (2.0, 0.1, 10.0),
        "start_tolerance_deg": (8.0, 0.1, 15.0),
        "gripper_tolerance": (12.0, 0.1, 20.0),
        "finish_tolerance_deg": (2.0, 0.1, 5.0),
        "finish_gripper_tolerance": (5.0, 0.1, 10.0),
        "tracking_error_deg": (15.0, 1.0, 30.0),
        "tracking_error_s": (0.75, 0.1, 2.0),
        "max_duration_s": (30.0, 1.0, 30.0),
    }
    for key, (default, low, high) in fields.items():
        value[key] = _number(value.get(key, default), key, low, high)
    motion_dir = value.get("motions_dir", str(path.parent / "motions"))
    if not isinstance(motion_dir, str) or not motion_dir.strip():
        raise ValueError("motions_dir 应为目录路径")
    candidate = Path(motion_dir).expanduser()
    value["motions_dir"] = str((path.parent / candidate).resolve()
                               if not candidate.is_absolute() else candidate.resolve())
    value["config_path"] = str(path)
    return value


def load_config(path=None) -> dict:
    selected = Path(path).expanduser().resolve() if path else default_config_path()
    try:
        content = selected.read_text(encoding="utf-8-sig")
    except FileNotFoundError as error:
        raise FileNotFoundError("尚未配置机械臂；先运行 python -m friston3.arms init。"
                                f"配置位置：{selected}") from error
    if len(content) > 64_000:
        raise ValueError("机械臂配置文件过大")
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"机械臂配置不是有效 JSON：{error.msg}") from error
    return validate_config(value, selected)


def create_config(path=None, *, right_port="", right_id="", left_port="", left_id="",
                  leader_port="", leader_id="", mock=False) -> Path:
    selected = Path(path).expanduser().resolve() if path else default_config_path()
    if selected.exists():
        raise FileExistsError(f"配置已存在，未覆盖：{selected}")
    arms = {}
    for name, arm_port, arm_id in (("right", right_port, right_id), ("left", left_port, left_id)):
        if bool(arm_port) != bool(arm_id):
            raise ValueError(f"{name} 的 port 和 id 要一起提供")
        if arm_port:
            arms[name] = {"port": arm_port, "id": arm_id, "calibration_dir": None}
    if mock and not arms:
        arms = {"right": {"port": "MOCK_RIGHT", "id": "mock_right", "calibration_dir": None}}
    if not arms:
        raise ValueError("请提供至少一只从臂的 --right-port/--right-id 或 --left-port/--left-id")
    if bool(leader_port) != bool(leader_id):
        raise ValueError("主臂的 --leader-port 和 --leader-id 要一起提供")
    leader = ({"port": leader_port, "id": leader_id, "calibration_dir": None}
              if leader_port else None)
    if mock and leader is None:
        leader = {"port": "MOCK_LEADER", "id": "mock_leader", "calibration_dir": None}
    config = validate_config({
        "protocol": 1, "host": "127.0.0.1", "port": 8766,
        "token": secrets.token_urlsafe(32), "mode": "mock" if mock else "real",
        "arms": arms, "leader": leader, "motions_dir": str(selected.parent / "motions"),
    }, selected)
    config.pop("config_path", None)
    selected.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects a concurrently created configuration.
    with selected.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if os.name != "nt":
        selected.chmod(0o600)
    return selected

