"""用于本地 ARM 服务的 SO-101 驱动程序。

导入此模块不会导入 LeRobot 或打开串口。真正的

驱动程序仅在 ``connect`` 中导入已安装的 LeRobot。所有硬件调用
都必须在打开驱动程序的线程中执行。
"""

from __future__ import annotations

# 这些导入来自 Python 标准库：importlib 按字符串导入模块，hashlib 计算摘要，
# Mapping/Real 是类型判断的抽象类别，Any/类型标注帮助阅读，不自动做运行时校验。
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
# JOINTS 定义于 arms/config.py；如 shoulder_pan.pos -> 电机名 shoulder_pan。
_CALIBRATION_FIELDS = ("id", "drive_mode", "homing_offset", "range_min", "range_max")


def _number(value: Any, label: str) -> float:
    # 本文件其他转换函数调用：把数值对象转成普通 float，label 是报错时使用的字段名。
    # value: Any 表示接受任意类型；-> float 是返回值标注，真正检查由下面 if 执行。
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} 必须是有限数值。")
    value = float(value)
    # 重新绑定局部变量，不会修改调用方原来的对象。math.isfinite 排除 NaN/正负无穷。
    if not math.isfinite(value):
        raise ValueError(f"{label} 必须是有限数值。")
    return value


def _positions(values: Any, label: str) -> dict[str, float]:
    # 统一 SDK 返回的数据结构；Mapping 包括普通 dict，但要求恰好有六个关节键。
    if not isinstance(values, Mapping) or set(values) != set(JOINTS):
        raise ValueError(f"{label} 必须且只能包含全部六个 .pos 关节键。")
    return {joint: _number(values[joint], f"{label}.{joint}") for joint in JOINTS}
    # 字典推导式：按 JOINTS 顺序逐项转换。f"...{表达式}..." 将表达式结果嵌入字符串。


def _check_limits(values: Mapping[str, float], limits: Mapping[str, list[float]], label: str) -> None:
    # 旧轨迹/普通 send 路径使用的已有范围检查；策略 send_policy 不调用它。
    # -> None 表示函数用于检查，成功时不返回业务数据，失败时 raise 抛出异常。
    for joint, value in values.items():
        lower, upper = limits[joint]
        if not lower <= value <= upper:
            raise ValueError(
                f"{label}.{joint}={value:.4f} 超出标定范围 [{lower:.4f}, {upper:.4f}]。"
            )


def _device_info(info: Any, label: str) -> dict[str, Any]:
    # RealHardware 构造和连接 leader 时调用；复制并规范化配置，不连接设备。
    if not isinstance(info, Mapping):
        raise ValueError(f"{label} 配置必须是对象。")
    result = dict(info)
    # dict(info) 浅复制键值；后续改 result 的 port/id 不会直接改原配置同名字段。
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
    # 把 JSON dict 或 SDK 标定对象整理为“电机名 -> 标定字段 -> int”的统一嵌套字典。
    # _read_calibration 与 _profile 都调用它，统一形状之后才能比较是否同一份标定。
    if not isinstance(value, Mapping) or set(value) != set(_MOTORS):
        raise ValueError(f"{label} 标定必须完整包含 SO-101 的六个电机。")
    result: dict[str, dict[str, int]] = {}
    for motor_id, motor in enumerate(_MOTORS, start=1):
        # enumerate 同时给出序号和元素；start=1 对齐 SO-101 电机 ID 1..6。
        entry = value[motor]
        if not isinstance(entry, Mapping):
            entry = {key: getattr(entry, key, None) for key in _CALIBRATION_FIELDS}
            # getattr(对象, 属性名字符串, 缺省值) 用于读取 SDK 对象属性，类似 entry.id。
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
    # _calibration_file 调用；读取已有文件并返回规范化字典，不新建/修改标定。
    try:
        if not path.is_file():
            raise FileNotFoundError(f"未找到标定文件：{path}")
        if path.stat().st_size > 256 * 1024:
            raise ValueError(f"标定文件过大：{path}")
        raw = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)
        # Path.read_text 是标准库读文件方法；JSON 解码时用 _unique_object 检查重复键。
        return _canonical_calibration(raw, str(path))
    except (OSError, ValueError) as exc:
        raise ValueError(f"无法使用标定文件 {path}：{exc}") from exc


def _calibration_file(info: Mapping[str, Any], role: str) -> tuple[Path, dict[str, dict[str, int]]]:
    """An explicit directory resolves ambiguity; implicit old/new files must agree."""
    identifier = info["id"]
    # role 是 follower/leader；info 来自服务配置，identifier 是与标定文件关联的机器人 ID。
    if info.get("calibration_dir") is not None:
        path = Path(info["calibration_dir"]).expanduser().resolve() / f"{identifier}.json"
        return path, _read_calibration(path)
    constants = importlib.import_module("lerobot.utils.constants")
    # import_module 动态导入已安装的 LeRobot 模块；不在项目里复制一套 SDK。
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
    # 返回二元组：(选中的绝对文件路径 Path，标定内容 dict)。


def _classes(role: str) -> tuple[type, type]:
    # RealHardware._build 调用；返回“设备类、配置类”，这里还没有创建设备实例。
    parent = "robots" if role == "follower" else "teleoperators"
    class_name = "SO101Follower" if role == "follower" else "SO101Leader"
    for suffix in (f"so101_{role}", f"so_{role}"):
        module_name = f"lerobot.{parent}.{suffix}"
        try:
            module = importlib.import_module(module_name)
            # module 是模块对象；模块里保存 SO101Follower/SO101FollowerConfig 等类定义。
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
            # type 标注表示返回的是类本身；调用方稍后通过 cls(...) 创建实例。
        except AttributeError as exc:
            raise RuntimeError(f"当前 LeRobot 的 {module_name} 不提供所需 SO-101 API。") from exc
    raise RuntimeError("当前 LeRobot 没有受支持的 SO-101 驱动；不会自动升级或更改环境。")


def _profile(device: Any, info: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    # _build 调用；输入 SDK 设备对象、配置、文件标定，输出 id/摘要/单位范围的资料 dict。
    # 校验“这些数字代表什么”及标定一致性，不发送电机目标。
    calibration = _canonical_calibration(device.calibration, info["id"])
    if calibration != expected:
        raise ValueError(f"{info['id']} 加载后的标定与选中的文件不一致。")
    bus = device.bus
    # device.bus 是 LeRobot 的总线对象；.motors/.calibration 等属性由 SDK 定义。
    if set(bus.motors) != set(_MOTORS):
        raise ValueError(f"{info['id']} 的驱动关节集合与 SO-101 不一致。")
    normalize = getattr(bus, "_normalize", None)
    # 保存 SDK 的转换方法；下方 normalize({...}) 把原始编码器数值变成工程单位。
    if not callable(normalize):
        raise RuntimeError("当前 LeRobot 不提供兼容的标定坐标转换；拒绝猜测角度范围。")
    limits = {}
    for joint, motor in zip(JOINTS, _MOTORS):
        # zip 把关节键 shoulder_pan.pos 和电机名 shoulder_pan 一一配对。
        definition = bus.motors[motor]
        mode = getattr(definition.norm_mode, "value", definition.norm_mode)
        # SDK 可能使用枚举：枚举的 .value 才是 "degrees" 等字符串；否则沿用原值。
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
    # 先稳定序列化，再算 SHA256 十六进制摘要；同标定/单位应有同摘要，供轨迹检查身份。
    return {"id": info["id"], "calibration_digest": digest, "limits": limits}


def _close_without_release(device: Any) -> None:
    """Close even a partly connected driver without invoking its torque-off default."""
    bus = getattr(device, "bus", None)
    # 由 RealHardware.close/失败清理调用；设备可能只构造一半，所以用 getattr 查询。
    if bus is None:
        return
    try:
        if not bus.is_connected:
            return
    except Exception:
        pass
    try:
        bus.disconnect(disable_torque=False)
        # disconnect 定义在安装的 LeRobot 总线实现；此参数沿用项目“关闭串口保留力矩”的语义。
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
    # RealHardware/MockHardware 的公共父类：复用线程归属与数值结构处理，不创建串口。
    # 下划线表示“本模块内部使用”的命名约定，不是 Python 强制的私有权限。
    mode: str

    def __init__(self, config: Mapping[str, Any]):
        # 子类 super().__init__(config) 调用这里；self 仍是那个子类对象，并非另造一个父对象。
        self.config = copy.deepcopy(dict(config))
        self._owner: int | None = None
        # int | None 是联合类型：尚未连接时为 None，之后保存线程 ID 整数。
        self._connected = False
        self._profiles: dict[str, dict[str, Any]] = {}
        cap = _number(self.config.get("max_relative_target", 2.0), "max_relative_target")
        if cap <= 0:
            raise ValueError("max_relative_target 必须大于 0。")
        self._caps = {motor: 5.0 if motor == "gripper" else cap for motor in _MOTORS}

    @property
    def profiles(self) -> dict[str, dict[str, Any]]:
        # @property 把方法作为只读属性使用：service 读 hardware.profiles，不写 profiles()。
        return copy.deepcopy(self._profiles)
        # deepcopy 连内层字典一起复制，服务对副本的改动不会污染驱动内部资料。

    def _claim(self) -> None:
        # connect 所在线程成为 owner。HTTP/推理线程即使拿到对象，也不能越过
        # _check_owner 读写串口；这使“单一所有者”成为可检查的代码约束。
        if self._owner is None:
            self._owner = threading.get_ident()
        self._check_owner()

    def _check_owner(self) -> None:
        # threading.get_ident 是标准库函数，返回当前执行线程 ID；与连接时记录值比较。
        if self._owner is not None and self._owner != threading.get_ident():
            raise RuntimeError("机械臂 I/O 只能由打开连接的控制线程调用。")

    def _require_connected(self) -> None:
        # read/send/read_leader 等入口复用此检查；正常时继续执行，否则抛出 RuntimeError。
        self._check_owner()
        if not self._connected:
            raise RuntimeError("机械臂驱动尚未连接。")

    def _targets(self, targets: Any, *, native_policy: bool = False) -> dict[str, dict[str, float]]:
        # 输入 {臂名: {六关节目标}}，返回规范化的新字典；由两种驱动的 send/send_policy 调用。
        # * 后的 native_policy 必须通过名字传参，便于读代码时辨认执行路径。
        self._require_connected()
        if not isinstance(targets, Mapping) or not targets or not set(targets) <= set(self._profiles):
            raise ValueError("目标必须指定至少一只已连接机械臂。")
        checked = {}
        for arm in ARM_NAMES:
            if arm in targets:
                checked[arm] = _positions(targets[arm], arm)
                if not native_policy:
                    _check_limits(checked[arm], self._profiles[arm]["limits"], arm)
        return checked


class RealHardware(_OwnedDriver):
    # 括号表示继承；本类复用父类的 _claim/_targets 等方法，并实现真实硬件操作。
    """A version-adapted SO-101 driver; construction never touches hardware."""

    mode = "real"

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        arms = self.config.get("arms")
        if not isinstance(arms, Mapping) or not arms or not set(arms) <= set(ARM_NAMES):
            raise ValueError("real 模式必须配置 left/right 中至少一只机械臂。")
        self._definitions = {arm: _device_info(arms[arm], arm) for arm in ARM_NAMES if arm in arms}
        # 定义资料与实际设备分开：_definitions 是配置 dict，_devices 稍后保存 SDK 实例。
        ports = [info["port"].casefold() for info in self._definitions.values()]
        ids = [info["id"] for info in self._definitions.values()]
        if len(set(ports)) != len(ports) or len(set(ids)) != len(ids):
            raise ValueError("左右从臂必须使用不同串口和不同标定 ID。")
        self._devices: dict[str, Any] = {}
        self._leader: Any = None
        self._leader_profile: dict[str, Any] | None = None

    def _build(self, info: Mapping[str, Any], role: str) -> tuple[Any, dict[str, Any]]:
        # connect/connect_leader 调用；根据配置创建 SDK 对象和 profile，尚未 device.connect。
        cls, config_cls = _classes(role)
        # cls 是设备类，config_cls 是对应配置类；名称是局部变量，可以像普通类那样调用。
        path, calibration = _calibration_file(info, role)
        kwargs = {
            "port": info["port"], "id": info["id"], "calibration_dir": path.parent, "use_degrees": True,
        }
        if role == "follower":
            kwargs.update(cameras={}, max_relative_target=dict(self._caps), disable_torque_on_disconnect=False)
        device = cls(config_cls(**kwargs))
        # **kwargs 把 dict 展开成命名参数：先构造 SDK 配置对象，再传给设备构造函数。
        try:
            return device, _profile(device, info, calibration)
            # _profile 返回校验后的身份/单位资料；两项一同返回给 connect 保存。
        except BaseException:
            _close_without_release(device)
            raise

    def connect(self, *, native_policy: bool = False) -> None:
        # ArmService._run 在 owner 线程调用；成功后 _connected=True，失败则清理已打开部分。
        self._claim()
        # _claim 继承自 _OwnedDriver，第一次调用时记录当前线程为唯一硬件所有者。
        if self._connected or self._devices:
            raise RuntimeError("机械臂驱动已有连接；请先关闭现有连接。")
        profiles = {}
        try:
            # Validate every file and coordinate system before opening any port.
            for arm, info in self._definitions.items():
                device, profile = self._build(info, "follower")
                # 元组解包接收 _build 返回的 SDK 对象和 profile 字典。
                self._devices[arm] = device
                profiles[arm] = profile
            for arm, device in self._devices.items():
                device.connect(calibrate=False)
                # 此处才调用 LeRobot 设备方法打开串口；calibrate=False 使用已有标定。
                if not device.is_calibrated:
                    raise RuntimeError(
                        f"{arm} 臂 ({self._definitions[arm]['id']}) 的控制器标定与文件不一致；"
                        "请用原 LeRobot 流程检查，服务不会写回标定。"
                    )
                initial = _positions(device.get_observation(), arm)
                # SDK get_observation 读取实测关节；_positions 在本文件统一为六维 float 字典。
                if not native_policy:
                    _check_limits(initial, profiles[arm]["limits"], arm)
            self._profiles = profiles
            self._connected = True
        except BaseException as original:
            # 包括用户中断在内的连接异常也要清理；raise（不带参数）继续向上抛出原异常。
            try:
                self.close()
            except Exception as cleanup:
                raise RuntimeError(f"连接失败：{original}；清理连接时也出错：{cleanup}") from original
            raise

    def read(self) -> dict[str, dict[str, float]]:
        # ArmService._observe 调用；每只臂通过 SDK 读取状态，返回按臂名分组的嵌套 dict。
        self._require_connected()
        return {arm: _positions(device.get_observation(), arm) for arm, device in self._devices.items()}

    def send(self, targets: Any) -> dict[str, dict[str, float]]:
        # ArmService 普通 record/play 与对应持姿路径调用；targets 是按臂分组的目标。
        checked = self._targets(targets)
        # All arguments have been checked before the first serial write.
        return {
            arm: _positions(self._devices[arm].send_action(values), f"{arm} 下发结果")
            for arm, values in checked.items()
        }
        # SDK send_action 返回实际接受的目标；这里把返回值统一格式后交回 Service。

    def send_policy(self, targets: Any) -> dict[str, dict[str, float]]:
        """Use the installed SDK's native target semantics for policy commands."""
        # ArmService 的策略任务专用；与旧轨迹 send 路径分开，保留现有实验语义。
        # 临时更改 SDK 配置必须放在 try/finally 中恢复，异常也不能污染后续调用。
        checked = self._targets(targets, native_policy=True)
        result = {}
        for arm, values in checked.items():
            device = self._devices[arm]
            # device 是对应臂的 LeRobot SO101Follower 实例；config 是该实例持有的配置对象。
            previous = device.config.max_relative_target
            device.config.max_relative_target = None
            try:
                result[arm] = _positions(device.send_action(values), f"{arm} 下发结果")
            finally:
                # 即使 SDK 发送抛异常，finally 也恢复旧配置，避免影响之后的普通轨迹调用。
                device.config.max_relative_target = previous
        return result

    def connect_leader(self) -> None:
        # 服务开始普通 record 时调用；使用与从臂相同的 owner 线程管理主臂。
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
        # 普通 record 控制循环调用；get_action 是 LeRobot 主臂接口名，返回主臂实测关节。
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
        # ArmService._run 的 finally 调用；逐个关设备，收集错误，最后释放线程归属。
        self._check_owner()
        self._connected = False
        failures = []
        try:
            self.close_leader()
        except Exception as exc:
            failures.append(str(exc))
        for arm, device in list(self._devices.items()):
            # list(...) 先复制键值对列表，允许循环中 del 删除原字典项而不改变迭代器。
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
    # 与 RealHardware 具有相同方法名/参数约定；Service 可直接换实现，无需改调度流程。
    # 它只在内存里保存数字，不能据此验证真实机械臂运动或抓取成功。
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

    def connect(self, *, native_policy: bool = False) -> None:
        # 同样记录 owner，但不打开串口；下方创建模拟 profile 和初始姿态。
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
        # 模拟“读回”就是复制内存状态；仍走同样的线程归属检查。
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

    def send_policy(self, targets: Any) -> dict[str, dict[str, float]]:
        # 模拟发送直接更新内部状态，下一次 read 就能读到；真实硬件不会瞬时到达目标。
        checked = self._targets(targets, native_policy=True)
        for arm, values in checked.items():
            self._state[arm] = dict(values)
        return copy.deepcopy(checked)

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
    # ArmService.__init__ 调用的工厂函数：根据模式选类并创建对象，不负责 connect。
    mode = config.get("mode", "real")
    if mode == "real":
        return RealHardware(config)
    if mode == "mock":
        return MockHardware(config)
    raise ValueError("机械臂 mode 必须为 real 或 mock。")
