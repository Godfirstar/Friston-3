"""SO-101 drivers for the local arm service.

Importing this module does not import LeRobot or open a serial port.  The real
driver imports the installed LeRobot only in ``connect``.  Every hardware call
must stay on the thread which opened the driver.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import threading
import time
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

from .config import ARM_NAMES, JOINTS


_MOTORS = tuple(joint.removesuffix(".pos") for joint in JOINTS)
_CALIBRATION_FIELDS = ("id", "drive_mode", "homing_offset", "range_min", "range_max")


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} 必须是有限数值。")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} 必须是有限数值。")
    return value


def _positions(values: Any, label: str) -> dict[str, float]:
    if not isinstance(values, Mapping) or set(values) != set(JOINTS):
        raise ValueError(f"{label} 必须且只能包含全部六个 .pos 关节键。")
    return {joint: _number(values[joint], f"{label}.{joint}") for joint in JOINTS}


def _check_limits(values: Mapping[str, float], limits: Mapping[str, list[float]], label: str) -> None:
    for joint, value in values.items():
        lower, upper = limits[joint]
        if not lower <= value <= upper:
            raise ValueError(
                f"{label}.{joint}={value:.4f} 超出标定范围 [{lower:.4f}, {upper:.4f}]。"
            )


def _device_info(info: Any, label: str) -> dict[str, Any]:
    if not isinstance(info, Mapping):
        raise ValueError(f"{label} 配置必须是对象。")
    result = dict(info)
    for key in ("port", "id"):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError(f"{label} 缺少非空 {key}。")
        result[key] = result[key].strip()
    identifier = result["id"]
    if (
        identifier in (".", "..")
        or len(identifier) > 128
        or any(char in identifier for char in '/\\\x00<>:"|?*')
    ):
        raise ValueError(f"{label}.id 不是有效的标定文件名。")
    if result.get("calibration_dir") is not None:
        directory = result["calibration_dir"]
        if not isinstance(directory, (str, Path)) or not str(directory).strip():
            raise ValueError(f"{label}.calibration_dir 必须是目录路径或 null。")
    return result


def _canonical_calibration(value: Any, label: str) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping) or set(value) != set(_MOTORS):
        raise ValueError(f"{label} 标定必须完整包含 SO-101 的六个电机。")
    result: dict[str, dict[str, int]] = {}
    for motor_id, motor in enumerate(_MOTORS, start=1):
        entry = value[motor]
        if not isinstance(entry, Mapping):
            entry = {key: getattr(entry, key, None) for key in _CALIBRATION_FIELDS}
        if set(entry) != set(_CALIBRATION_FIELDS):
            raise ValueError(f"{label}: {motor} 的标定字段不完整或不受支持。")
        fields = {}
        for field in _CALIBRATION_FIELDS:
            number = entry[field]
            if isinstance(number, bool) or not isinstance(number, int):
                raise ValueError(f"{label}: {motor}.{field} 必须是整数。")
            fields[field] = number
        if fields["id"] != motor_id:
            raise ValueError(f"{label}: {motor} 应为电机 ID {motor_id}。")
        if fields["drive_mode"] not in (0, 1):
            raise ValueError(f"{label}: {motor}.drive_mode 必须为 0 或 1。")
        if fields["range_min"] >= fields["range_max"]:
            raise ValueError(f"{label}: {motor} 标定范围无效。")
        result[motor] = fields
    return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 中出现重复键 {key!r}。")
        result[key] = value
    return result


def _read_calibration(path: Path) -> dict[str, dict[str, int]]:
    try:
        if not path.is_file():
            raise FileNotFoundError(f"未找到标定文件：{path}")
        if path.stat().st_size > 256 * 1024:
            raise ValueError(f"标定文件过大：{path}")
        raw = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)
        return _canonical_calibration(raw, str(path))
    except (OSError, ValueError) as exc:
        raise ValueError(f"无法使用标定文件 {path}：{exc}") from exc


def _calibration_file(info: Mapping[str, Any], role: str) -> tuple[Path, dict[str, dict[str, int]]]:
    """An explicit directory resolves ambiguity; implicit old/new files must agree."""
    identifier = info["id"]
    if info.get("calibration_dir") is not None:
        path = Path(info["calibration_dir"]).expanduser().resolve() / f"{identifier}.json"
        return path, _read_calibration(path)
    constants = importlib.import_module("lerobot.utils.constants")
    root = Path(constants.HF_LEROBOT_CALIBRATION).expanduser()
    group = "robots" if role == "follower" else "teleoperators"
    names = (f"so101_{role}", f"so_{role}")
    candidates = [root / group / name / f"{identifier}.json" for name in names]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        checked = "；".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"未找到 {identifier} 的已有标定文件。检查过：{checked}。"
            "请沿用之前测试时的 ID，或显式设置 calibration_dir；服务不会重新标定。"
        )
    calibrations = [_read_calibration(path) for path in existing]
    if any(calibration != calibrations[0] for calibration in calibrations[1:]):
        raise ValueError(
            f"{identifier} 的旧版/新版标定内容不同："
            + "；".join(str(path) for path in existing)
            + "。请用 calibration_dir 明确指定已验证的目录。"
        )
    return existing[0].resolve(), calibrations[0]


def _classes(role: str) -> tuple[type, type]:
    parent = "robots" if role == "follower" else "teleoperators"
    class_name = "SO101Follower" if role == "follower" else "SO101Leader"
    for suffix in (f"so101_{role}", f"so_{role}"):
        module_name = f"lerobot.{parent}.{suffix}"
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == "lerobot":
                raise RuntimeError("未安装 LeRobot；请在已测试过机械臂的 lerobot 环境启动服务。") from exc
            if exc.name == module_name:
                continue
            raise RuntimeError(
                f"加载 {module_name} 时缺少依赖 {exc.name}；请检查当前 lerobot 环境。"
            ) from exc
        try:
            return getattr(module, class_name), getattr(module, class_name + "Config")
        except AttributeError as exc:
            raise RuntimeError(f"当前 LeRobot 的 {module_name} 不提供所需 SO-101 API。") from exc
    raise RuntimeError("当前 LeRobot 没有受支持的 SO-101 驱动；不会自动升级或更改环境。")


def _profile(device: Any, info: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    calibration = _canonical_calibration(device.calibration, info["id"])
    if calibration != expected:
        raise ValueError(f"{info['id']} 加载后的标定与选中的文件不一致。")
    bus = device.bus
    if set(bus.motors) != set(_MOTORS):
        raise ValueError(f"{info['id']} 的驱动关节集合与 SO-101 不一致。")
    normalize = getattr(bus, "_normalize", None)
    if not callable(normalize):
        raise RuntimeError("当前 LeRobot 不提供兼容的标定坐标转换；拒绝猜测角度范围。")
    limits = {}
    for joint, motor in zip(JOINTS, _MOTORS):
        definition = bus.motors[motor]
        mode = getattr(definition.norm_mode, "value", definition.norm_mode)
        wanted = "range_0_100" if motor == "gripper" else "degrees"
        if mode != wanted or definition.id != calibration[motor]["id"]:
            raise ValueError(f"{info['id']} 的 {motor} 单位或电机 ID 不一致。")
        raw = calibration[motor]
        endpoints = [
            _number(normalize({definition.id: raw[key]})[definition.id], f"{joint} 边界")
            for key in ("range_min", "range_max")
        ]
        lower, upper = sorted(endpoints)
        if lower >= upper:
            raise ValueError(f"{info['id']} 的 {motor} 转换后范围无效。")
        if motor == "gripper" and not (math.isclose(lower, 0.0) and math.isclose(upper, 100.0)):
            raise ValueError("夹爪必须使用 0–100 的标定百分比。")
        limits[joint] = [lower, upper]
    fingerprint = {"calibration": calibration, "limits": limits, "units": "degrees_and_gripper_percent"}
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"id": info["id"], "calibration_digest": digest, "limits": limits}


def _close_without_release(device: Any) -> None:
    """Close even a partly connected driver without invoking its torque-off default."""
    bus = getattr(device, "bus", None)
    if bus is None:
        return
    try:
        if not bus.is_connected:
            return
    except Exception:
        pass
    try:
        bus.disconnect(disable_torque=False)
        return
    except Exception as original:
        # Older Feetech uses an SDK port; newer SerialMotorsBus has _close().
        port = getattr(bus, "port_handler", None)
        for target, method in ((port, "closePort"), (bus, "_close")):
            closer = getattr(target, method, None)
            if callable(closer):
                try:
                    closer()
                    return
                except Exception:
                    pass
        raise RuntimeError("串口关闭失败；未发送卸力指令，请检查该机械臂连接。") from original


class _OwnedDriver:
    mode: str

    def __init__(self, config: Mapping[str, Any]):
        self.config = copy.deepcopy(dict(config))
        self._owner: int | None = None
        self._connected = False
        self._profiles: dict[str, dict[str, Any]] = {}
        cap = _number(self.config.get("max_relative_target", 2.0), "max_relative_target")
        if cap <= 0:
            raise ValueError("max_relative_target 必须大于 0。")
        self._caps = {motor: 5.0 if motor == "gripper" else cap for motor in _MOTORS}

    @property
    def profiles(self) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(self._profiles)

    def _claim(self) -> None:
        if self._owner is None:
            self._owner = threading.get_ident()
        self._check_owner()

    def _check_owner(self) -> None:
        if self._owner is not None and self._owner != threading.get_ident():
            raise RuntimeError("机械臂 I/O 只能由打开连接的控制线程调用。")

    def _require_connected(self) -> None:
        self._check_owner()
        if not self._connected:
            raise RuntimeError("机械臂驱动尚未连接。")

    def _targets(self, targets: Any) -> dict[str, dict[str, float]]:
        self._require_connected()
        if not isinstance(targets, Mapping) or not targets or not set(targets) <= set(self._profiles):
            raise ValueError("目标必须指定至少一只已连接机械臂。")
        checked = {}
        for arm in ARM_NAMES:
            if arm in targets:
                checked[arm] = _positions(targets[arm], arm)
                _check_limits(checked[arm], self._profiles[arm]["limits"], arm)
        return checked


class RealHardware(_OwnedDriver):
    """A version-adapted SO-101 driver; construction never touches hardware."""

    mode = "real"

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        arms = self.config.get("arms")
        if not isinstance(arms, Mapping) or not arms or not set(arms) <= set(ARM_NAMES):
            raise ValueError("real 模式必须配置 left/right 中至少一只机械臂。")
        self._definitions = {arm: _device_info(arms[arm], arm) for arm in ARM_NAMES if arm in arms}
        ports = [info["port"].casefold() for info in self._definitions.values()]
        ids = [info["id"] for info in self._definitions.values()]
        if len(set(ports)) != len(ports) or len(set(ids)) != len(ids):
            raise ValueError("左右从臂必须使用不同串口和不同标定 ID。")
        self._devices: dict[str, Any] = {}
        self._leader: Any = None
        self._leader_profile: dict[str, Any] | None = None

    def _build(self, info: Mapping[str, Any], role: str) -> tuple[Any, dict[str, Any]]:
        cls, config_cls = _classes(role)
        path, calibration = _calibration_file(info, role)
        kwargs = {
            "port": info["port"], "id": info["id"], "calibration_dir": path.parent, "use_degrees": True,
        }
        if role == "follower":
            kwargs.update(cameras={}, max_relative_target=dict(self._caps), disable_torque_on_disconnect=False)
        device = cls(config_cls(**kwargs))
        try:
            return device, _profile(device, info, calibration)
        except BaseException:
            _close_without_release(device)
            raise

    def connect(self) -> None:
        self._claim()
        if self._connected or self._devices:
            raise RuntimeError("机械臂驱动已有连接；请先关闭现有连接。")
        profiles = {}
        try:
            # Validate every file and coordinate system before opening any port.
            for arm, info in self._definitions.items():
                device, profile = self._build(info, "follower")
                self._devices[arm] = device
                profiles[arm] = profile
            for arm, device in self._devices.items():
                device.connect(calibrate=False)
                if not device.is_calibrated:
                    raise RuntimeError(
                        f"{arm} 臂 ({self._definitions[arm]['id']}) 的控制器标定与文件不一致；"
                        "请用原 LeRobot 流程检查，服务不会写回标定。"
                    )
                initial = _positions(device.get_observation(), arm)
                _check_limits(initial, profiles[arm]["limits"], arm)
            self._profiles = profiles
            self._connected = True
        except BaseException as original:
            try:
                self.close()
            except Exception as cleanup:
                raise RuntimeError(f"连接失败：{original}；清理连接时也出错：{cleanup}") from original
            raise

    def read(self) -> dict[str, dict[str, float]]:
        self._require_connected()
        return {arm: _positions(device.get_observation(), arm) for arm, device in self._devices.items()}

    def send(self, targets: Any) -> dict[str, dict[str, float]]:
        checked = self._targets(targets)
        # All arguments have been checked before the first serial write.
        return {
            arm: _positions(self._devices[arm].send_action(values), f"{arm} 下发结果")
            for arm, values in checked.items()
        }

    def connect_leader(self) -> None:
        self._require_connected()
        if self._leader is not None:
            raise RuntimeError("主臂已连接。")
        info = _device_info(self.config.get("leader"), "leader")
        if info["port"].casefold() in {item["port"].casefold() for item in self._definitions.values()}:
            raise ValueError("主臂串口不能与从臂串口相同。")
        device, profile = self._build(info, "leader")
        self._leader = device
        try:
            device.connect(calibrate=False)
            if not device.is_calibrated:
                raise RuntimeError("主臂控制器标定与文件不一致；服务不会重新标定。")
            initial = _positions(device.get_action(), "leader")
            _check_limits(initial, profile["limits"], "leader")
            self._leader_profile = profile
        except BaseException:
            self.close_leader()
            raise

    def read_leader(self) -> dict[str, float]:
        self._require_connected()
        if self._leader is None or self._leader_profile is None:
            raise RuntimeError("主臂尚未连接。")
        values = _positions(self._leader.get_action(), "leader")
        _check_limits(values, self._leader_profile["limits"], "leader")
        return values

    def close_leader(self) -> None:
        self._check_owner()
        if self._leader is not None:
            _close_without_release(self._leader)
            self._leader = None
        self._leader_profile = None

    def close(self) -> None:
        self._check_owner()
        self._connected = False
        failures = []
        try:
            self.close_leader()
        except Exception as exc:
            failures.append(str(exc))
        for arm, device in list(self._devices.items()):
            try:
                _close_without_release(device)
                del self._devices[arm]
            except Exception as exc:
                failures.append(f"{arm}: {exc}")
        self._profiles.clear()
        if failures:
            raise RuntimeError("；".join(failures))
        self._owner = None


class MockHardware(_OwnedDriver):
    """Synthetic positions for integration checks, never a real calibration."""

    mode = "mock"

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        arms = self.config.get("arms") or {"right": {"id": "mock_right"}}
        if not isinstance(arms, Mapping) or not set(arms) <= set(ARM_NAMES):
            raise ValueError("mock arms 必须使用 left/right 名称。")
        self._definitions = copy.deepcopy(dict(arms))
        self._state: dict[str, dict[str, float]] = {}
        self._leader_start: float | None = None

    def connect(self) -> None:
        self._claim()
        if self._connected:
            raise RuntimeError("mock 驱动已连接。")
        for arm in ARM_NAMES:
            if arm not in self._definitions:
                continue
            info = self._definitions[arm]
            identifier = str(info.get("id") or f"mock_{arm}")
            limits = {joint: [0.0, 100.0] if joint == "gripper.pos" else [-90.0, 90.0] for joint in JOINTS}
            self._profiles[arm] = {
                "id": identifier,
                "calibration_digest": "mock:" + hashlib.sha256(identifier.encode()).hexdigest(),
                "limits": limits,
            }
            self._state[arm] = {joint: 50.0 if joint == "gripper.pos" else 0.0 for joint in JOINTS}
        self._connected = True

    def read(self) -> dict[str, dict[str, float]]:
        self._require_connected()
        return copy.deepcopy(self._state)

    def send(self, targets: Any) -> dict[str, dict[str, float]]:
        checked = self._targets(targets)
        result = {}
        for arm, values in checked.items():
            applied = {}
            for joint, target in values.items():
                current = self._state[arm][joint]
                cap = self._caps[joint.removesuffix(".pos")]
                applied[joint] = current + max(-cap, min(cap, target - current))
            self._state[arm] = applied
            result[arm] = dict(applied)
        return result

    def connect_leader(self) -> None:
        self._require_connected()
        if self._leader_start is not None:
            raise RuntimeError("mock 主臂已连接。")
        self._leader_start = time.monotonic()

    def read_leader(self) -> dict[str, float]:
        self._require_connected()
        if self._leader_start is None:
            raise RuntimeError("mock 主臂尚未连接。")
        phase = 2.0 * math.pi * (time.monotonic() - self._leader_start) / 4.0
        result = {joint: 50.0 if joint == "gripper.pos" else 0.0 for joint in JOINTS}
        result["shoulder_pan.pos"] = 5.0 * math.sin(phase)
        result["wrist_roll.pos"] = 8.0 * math.sin(phase)
        return result

    def close_leader(self) -> None:
        self._check_owner()
        self._leader_start = None

    def close(self) -> None:
        self._check_owner()
        self._connected = False
        self._leader_start = None
        self._state.clear()
        self._profiles.clear()
        self._owner = None


def make_hardware(config: Mapping[str, Any]) -> RealHardware | MockHardware:
    mode = config.get("mode", "real")
    if mode == "real":
        return RealHardware(config)
    if mode == "mock":
        return MockHardware(config)
    raise ValueError("机械臂 mode 必须为 real 或 mock。")
