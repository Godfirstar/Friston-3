"""普通关节轨迹的数据层：供 ArmService 的 record/play 分支调用。

load_motion 读取并校验 JSON；interpolate 按时间计算目标；save_recording 保存
本次录到的从臂实测位置。本模块不连接串口、不运行 π0.5，也不是已删除的
taught 包。JSON 中的标定摘要用于确认“同一组数字仍表示同一套关节坐标”。
"""
from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from .config import ARM_NAMES, JOINTS
# 前导点表示同一个arms包中的config；这里复用关节顺序，不另造一套名称。

FORMAT = "friston-arm-trajectory-v1"
UNITS = "degrees_gripper_percent"
_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,47}$")


def validate_name(name: str) -> str:
    # 服务record/play及文件读取调用；name:str/->str是类型提示，实际检查仍由if完成。
    # _NAME为re.compile返回的正则对象；fullmatch要求整个名字符合格式，不只是开头。
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError("动作名须以小写英文字母开头，只含小写字母、数字、_、-，最多48字符")
    return name


def _finite(value):
    # validate_motion调用，返回bool；精确type判断排除布尔值，isfinite排除NaN/无穷。
    return type(value) in (int, float) and math.isfinite(value)


def validate_motion(value, name=None, max_duration_s=30.0):
    # load_motion/save_recording调用；value是轨迹JSON解析出的dict，返回通过检查的原dict。
    # 轨迹结构：profiles保存各臂身份，frames是按时间排序的[{t,positions}, ...]列表。
    if not isinstance(value, dict) or value.get("format") != FORMAT or value.get("units") != UNITS:
        raise ValueError("动作格式或单位不匹配（转动关节为度，夹爪为0..100）")
    actual_name = validate_name(value.get("name"))
    if name is not None and actual_name != name:
        raise ValueError("动作文件名与内部名称不一致")
    profiles = value.get("profiles")
    # dict.get缺键返回None；set(dict)取得键集合，差集检测是否出现left/right之外的臂名。
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
    # enumerate同时给出0起始索引和当前帧；previous逐轮保存上一帧秒数，用来检查时间顺序。
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or set(frame) != {"t", "positions"}:
            raise ValueError("动作采样点格式错误")
        timestamp = frame["t"]
        if (not _finite(timestamp) or timestamp <= previous or timestamp > max_duration_s + 0.1
                or (index == 0 and timestamp != 0)):
            raise ValueError("动作时间须从0开始、严格递增且不超过时长限制")
        previous = timestamp
        positions = frame["positions"]
        # positions是“臂名→关节名→数值”的嵌套字典；每帧必须沿用profiles中的同一组臂。
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
    # 服务play和motion_catalog调用；返回经过校验的轨迹字典，未发送任何目标给电机。
    path = Path(directory) / (validate_name(name) + ".json")
    if path.stat().st_size > 4_000_000:
        raise ValueError("动作文件过大")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        # 内层先读字符串，外层再解析为Python对象；JSON格式错误被包装成业务ValueError。
    except json.JSONDecodeError as error:
        raise ValueError(f"动作 {name} 不是有效JSON") from error
    return validate_motion(value, name, max_duration_s)


def within_pose(actual, target, config):
    # 服务在轨迹开始/结束时、motion_summary判断首尾时调用；返回是否逐关节接近目标。
    # actual/target均是“臂→关节→数值”；config提供本次比较所用的已有容差。
    for arm, pose in target.items():
        if arm not in actual:
            return False
        for joint, value in pose.items():
            # .items()每次给(关节名,目标值)，夹爪单位是百分比，其他关节是度。
            tolerance = config["gripper_tolerance"] if joint == "gripper.pos" else config["start_tolerance_deg"]
            if abs(actual[arm][joint] - value) > tolerance:
                return False
    return True


def motion_summary(value, config):
    # motion_catalog和服务调用；将整段轨迹压成展示字典，不把所有采样帧发给状态查看者。
    # frames[-1]取末帧，frames[0]取首帧；looped只说明首尾位置接近，不表示已经循环播放。
    return {"name": value["name"], "arms": list(value["profiles"]),
            "duration_s": round(value["frames"][-1]["t"], 3), "mode": value["mode"],
            "looped": within_pose(value["frames"][-1]["positions"], value["frames"][0]["positions"], config)}


def motion_catalog(config):
    # ArmService初始化/刷新动作列表调用；返回(有效摘要列表,无效文件原因列表)。
    result, invalid = [], []
    directory = Path(config["motions_dir"])
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            # glob寻找目录中匹配的文件，sorted固定顺序；path.stem是不含.json的文件名。
            try:
                result.append(motion_summary(load_motion(directory, path.stem, config["max_duration_s"]), config))
            except (OSError, ValueError, KeyError, TypeError) as error:
                # 某个文件坏了记到invalid，for仍检查其余文件；不会默默把坏文件当有效动作。
                invalid.append({"name": path.stem, "error": str(error)})
    return result, invalid


def check_compatibility(value, profiles, config):
    # ArmService接收play、save_recording保存时调用；通过时隐式返回None，失败抛异常。
    # value内是录制时的身份，profiles内是当前设备身份；比对ID和标定摘要保持坐标含义一致。
    if value["mode"] != config["mode"]:
        raise ValueError("模拟动作与实机动作不可混用")
    for arm, saved in value["profiles"].items():
        actual = profiles.get(arm)
        if not actual or any(saved.get(key) != actual.get(key) for key in ("id", "calibration_digest")):
            # any消费生成器：任一字段不同就为True；这不是在计算两个姿态的距离。
            raise ValueError(f"{arm} 的ID或标定与录制时不一致；请使用原标定或重新示教")
        for frame in value["frames"]:
            for joint, number in frame["positions"][arm].items():
                low, high = actual["limits"][joint]
                if not low - 0.01 <= number <= high + 0.01:
                    raise ValueError(f"动作中的 {arm}.{joint} 超出本机标定范围")


def interpolate(value, elapsed):
    """由 ArmService._tick_job 调用，返回 elapsed 时刻的六关节目标。"""
    frames = value["frames"]
    # elapsed是动作开始后的秒数；超出两端时直接返回端点深拷贝，避免调用方改到原始数据。
    if elapsed >= frames[-1]["t"]:
        return deepcopy(frames[-1]["positions"])
    if elapsed <= 0:
        return deepcopy(frames[0]["positions"])
    # 找到第一个时间大于 elapsed 的采样点；左右两帧夹住当前时刻。
    # validate_motion 已确保时间严格递增，所以后面的分母不为零。
    index = bisect_right([frame["t"] for frame in frames], elapsed)
    # 列表推导提取所有时间；bisect_right来自标准库，在有序时间轴里找插入位置。
    left, right = frames[index - 1], frames[index]
    weight = (elapsed - left["t"]) / (right["t"] - left["t"])
    # weight是0到1之间的时间比例，例如相邻帧正中间为0.5，关节目标也取两端中间值。
    # 线性插值 q(t)=q_left + weight*(q_right-q_left)，不是神经网络推理。
    # 返回新字典，后续控制代码不会改写从磁盘读到的原始示教。
    return {arm: {joint: number + (right["positions"][arm][joint] - number) * weight
                  for joint, number in pose.items()}
            for arm, pose in left["positions"].items()}


def save_recording(config, name, profiles, frames):
    """由服务完成 record 时调用；返回已发布的轨迹路径。"""
    # frames来自所有者线程录到的实测位置；字典推导只保留profiles中ID和标定摘要两项。
    # datetime.now(timezone.utc).isoformat()生成带UTC时区的时间字符串，供追溯录制时刻。
    value = validate_motion({"format": FORMAT, "units": UNITS, "name": validate_name(name),
        "mode": config["mode"], "created_at": datetime.now(timezone.utc).isoformat(),
        "profiles": {arm: {key: entry[key] for key in ("id", "calibration_digest")}
                     for arm, entry in profiles.items()}, "frames": frames}, name, config["max_duration_s"])
    check_compatibility(value, profiles, config)
    directory = Path(config["motions_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (name + ".json")
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    # dumps把dict转JSON字符串；ensure_ascii=False保留中文，separators去掉不必要的空格。
    # 先把完整内容写入同目录临时文件，再用硬链接发布目标名字。
    # 调用方看到的是完整 JSON；目标若已存在，os.link 会失败而非覆盖旧示教。
    import os
    import tempfile
    descriptor, temporary = tempfile.mkstemp(prefix=".recording-", suffix=".tmp", dir=directory)
    # mkstemp返回(文件描述符整数,临时文件路径字符串)；fdopen把描述符包装成文本写入流。
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            # flush 推入操作系统缓冲区，fsync 再请求同步文件内容。
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        # unlink只删除临时文件名；成功硬链接后destination仍指向完整文件内容。
        Path(temporary).unlink(missing_ok=True)
    return destination
