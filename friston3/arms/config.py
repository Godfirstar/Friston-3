"""LeRobot 服务和无依赖客户端共享的配置。"""
from __future__ import annotations
# future annotations让类型注解延后处理；Path、dict等注解描述接口，不替代下面的运行时检查。

import json
import math
import os
from pathlib import Path
import re
import secrets

JOINT_FORMAT = "friston_pi05_joint_v1"
JOINTS = tuple(name + ".pos" for name in (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"))
# tuple(...)消费生成器，为每个关节名加.pos后得到固定顺序元组；策略张量/驱动均沿用此顺序。
DATASET_JOINTS = [key.removesuffix(".pos") for key in JOINTS]
STATE, ACTION = "observation.state", "action"
ARM_NAMES = ("left", "right")
GESTURES = ("rest", "wave", "welcome", "point_pickup", "shrug")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


def default_config_path() -> Path:
    # load_config/create_config等调用；返回Path对象，不读取文件。-> Path是返回类型提示。
    # os.environ是进程环境变量映射，get第二项是缺键默认值；strip去两端空白。
    explicit = os.environ.get("FRISTON_ARMS_CONFIG", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    local = os.environ.get("LOCALAPPDATA")
    # Windows优先LOCALAPPDATA；其他环境回退到XDG_DATA_HOME或用户.local/share目录。
    root = Path(local) if local else Path(os.environ.get(
        "XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return (root / "Friston3" / "arms" / "config.json").resolve()


def _number(value, label, minimum, maximum):
    # validate_config的内部辅助：通过就返回float，失败抛ValueError并带字段名label。
    # type精确检查会排除bool；math.isfinite排除NaN/无穷，链式比较要求落在既有区间。
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not minimum <= value <= maximum):
        raise ValueError(f"{label} 必须是 {minimum}..{maximum} 范围的有限数值")
    return float(value)


def validate_config(value: dict, path: Path) -> dict:
    # load_config/create_config调用；value来自JSON/构造字典，path是该配置文件路径。
    # 会在原value中补默认项、展开路径，再返回同一个字典；这不是“只检查不修改”的函数。
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
    # *arms.items()把(名称,配置)各项展开进列表，再附加可选leader；for每次解包两项。
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
        # casefold统一大小写，用于发现COM9/com9重复；并未在此打开串口。
        calibration_dir = entry.get("calibration_dir")
        if calibration_dir is not None:
            if not isinstance(calibration_dir, str) or not calibration_dir.strip():
                raise ValueError(f"{name}.calibration_dir 应是目录路径或 null")
            candidate = Path(calibration_dir).expanduser()
            # expanduser展开~；相对路径以配置文件的父目录为基准，不以当前终端目录为基准。
            entry["calibration_dir"] = str((path.parent / candidate).resolve()
                                           if not candidate.is_absolute() else candidate.resolve())
    if len(ports) != len(set(ports)):
        # set集合会去重；元素数变少表示有多个角色填写了同一个串口。
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
    # fields每项是(默认值,下界,上界)；这些是当前已有的配置约定，本轮注释不改变数值。
    # 普通轨迹与策略发送走不同方法，具体使用哪些参数要继续看service/hardware，不能混同。
    for key, (default, low, high) in fields.items():
        # 嵌套解包：先拆键和值，再把三元组拆为三个变量；缺字段才使用default。
        value[key] = _number(value.get(key, default), key, low, high)
    motion_dir = value.get("motions_dir", str(path.parent / "motions"))
    if not isinstance(motion_dir, str) or not motion_dir.strip():
        raise ValueError("motions_dir 应为目录路径")
    candidate = Path(motion_dir).expanduser()
    value["motions_dir"] = str((path.parent / candidate).resolve()
                               if not candidate.is_absolute() else candidate.resolve())
    value["config_path"] = str(path)
    # config_path是加载时补的来源说明；create_config落盘前会删掉这项。
    return value


def load_config(path=None) -> dict:
    # 调用方包括ArmClient、ArmIntegration和服务CLI；返回经过validate_config的字典。
    # 条件表达式在有显式path时用它，没有时才调用default_config_path。
    selected = Path(path).expanduser().resolve() if path else default_config_path()
    try:
        content = selected.read_text(encoding="utf-8-sig")
        # Path.read_text实际打开/读取文件；utf-8-sig同时兼容带BOM和普通UTF-8文本。
    except FileNotFoundError as error:
        raise FileNotFoundError("尚未配置机械臂；先运行 python -m friston3.arms init。"
                                f"配置位置：{selected}") from error
    if len(content) > 64_000:
        raise ValueError("机械臂配置文件过大")
    try:
        value = json.loads(content)
        # loads解析一个字符串为Python对象；json.load则是直接从文件流读取，注意最后的s。
    except json.JSONDecodeError as error:
        raise ValueError(f"机械臂配置不是有效 JSON：{error.msg}") from error
    return validate_config(value, selected)


def create_config(path=None, *, right_port="", right_id="", left_port="", left_id="",
                  leader_port="", leader_id="", mock=False) -> Path:
    # arms CLI的init分支调用；*后的硬件选项要求按名称传入；生成新JSON并返回文件Path。
    # 配置创建不会连接电机；mock也只是选择模拟配置，实际驱动由服务稍后构造。
    selected = Path(path).expanduser().resolve() if path else default_config_path()
    if selected.exists():
        raise FileExistsError(f"配置已存在，未覆盖：{selected}")
    arms = {}
    for name, arm_port, arm_id in (("right", right_port, right_id), ("left", left_port, left_id)):
        # 外层元组包含两组配置；每轮解包成三个局部变量，避免复制左右臂的组装代码。
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
    # secrets.token_urlsafe生成本机服务认证用随机token；调用validate_config一次补齐默认项。
    config.pop("config_path", None)
    # pop删除键并返回其值；第二参数None让缺键时不报错。mkdir可以补齐多级父目录。
    selected.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects a concurrently created configuration.
    with selected.open("x", encoding="utf-8", newline="\n") as stream:
        # 'x'是排他新建模式：文件已存在就失败；with退出时关闭文件流。
        json.dump(config, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if os.name != "nt":
        # 非Windows才设置POSIX权限；0o600是八进制，表示仅拥有者可读写。
        selected.chmod(0o600)
    return selected

