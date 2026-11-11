"""SO-101 示教采集入口：沿用 LeRobot 数据格式，在外层提供手动分段面板。

由 python -m friston3.arms.learning.record 转入 main；build_config 把项目配置转换成
LeRobot 的 RecordConfig。真正的读臂、跟随、相机采集和写数据由官方录制线程执行。
录制前先让占用同一串口的 ArmService 退出，保证硬件在这段时间只有一个所有者。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
from threading import RLock, Thread
import time

from ..models import assets_root, resolve_asset_path
from ..config import load_config
# .表示当前learning包；..回到父包arms，因此load_config定义在arms/config.py。
# assets_root/resolve_asset_path定义在learning/__init__.py，统一定位模型与数据资产。


def parse_args(argv=None):
    # 【调用方】main；输入argv字符串list或None，返回带属性的argparse.Namespace。
    # None让argparse读取真实命令行；本函数只整理参数/路径，还没有连接机械臂。
    parser = argparse.ArgumentParser(description=__doc__)
    # ArgumentParser来自Python标准库；__doc__是本文件顶部说明，显示在--help里。
    parser.add_argument("--config", type=Path)
    # 声明--config PATH；type=Path把输入字符串转成pathlib.Path，不会因此读取文件。
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    # choices限制语法上的可选值；default是未显式传参时的值，随后用args.arm读取。
    parser.add_argument("--follower-port")
    parser.add_argument("--follower-id")
    parser.add_argument("--leader-port")
    parser.add_argument("--leader-id")
    parser.add_argument("--task", required=True, help="Task actually demonstrated in these episodes")
    # required=True要求提供任务文本；PowerShell中带空格的文本需用引号作为一个参数。
    parser.add_argument("--camera", action="append", metavar="KEY=INDEX",
                        help="Repeat for each view; default DirectShow base_0_rgb=0 and right_wrist_0_rgb=1")
    # append表示该选项可重复，每次值追加到args.camera列表；不是后一次覆盖前一次。
    # 例如--camera base_0_rgb=0 --camera right_wrist_0_rgb=1；具体索引以当前配置/枚举为准。
    parser.add_argument("--camera-backend", choices=("any", "msmf", "dshow"), default="dshow")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-seconds", type=float, default=60)
    parser.add_argument("--reset-seconds", type=float, default=10)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--resume", action="store_true", help="Append new episodes/tasks to the local dataset")
    # store_true是布尔开关：传--resume即True，不传默认False，无需再跟一个True字符串。
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--service-shutdown-timeout", type=float, default=15.0,
                        help="Seconds to wait for the arm service to release shared serial ports")
    args = parser.parse_args(argv)
    # 实际解析发生于这里；--follower-port自动变成属性名args.follower_port。
    # 遇到--help会打印帮助并退出，因此main不会继续构造真实硬件配置。
    if not args.task.strip():
        # str.strip去除首尾空白；空字符串在if中为False，not反转为True。
        parser.error("--task must describe the demonstration")
    if min(args.width, args.height, args.fps, args.episodes, args.episode_seconds) <= 0 or args.reset_seconds < 0:
        parser.error("Dimensions, fps, episode count and duration must be positive")
    if args.resume and args.root is None:
        parser.error("--resume requires the existing dataset --root")
    if not math.isfinite(args.service_shutdown_timeout) or args.service_shutdown_timeout <= 0:
        parser.error("--service-shutdown-timeout must be finite and positive")
    if args.root is None:
        args.root = assets_root() / "datasets" / ("so101_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        # Path的/拼路径；strftime把当前时间格式化为目录名；这里只算路径，不开始录制。
    # 续写定位已迁移的旧数据集；新建录制始终尊重用户明确指定的输出路径。
    args.root = resolve_asset_path(args.root) if args.resume else args.root.expanduser().resolve()
    return args


def build_config(args):
    # 【调用方】main；args来自parse_args，返回官方LeRobot RecordConfig配置对象。
    # 下游是arms.config.load_config、policy_options，以及LeRobot各配置类构造函数。
    # 配置对象只描述怎样连接/保存；实际连接由后面的record_demonstrations进行。
    # 参数优先级：本次 CLI 明确值 > 机械臂配置；相机键名还要与训练/推理保持一致。
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.scripts.lerobot_record import RecordConfig
    from lerobot.teleoperators.so_leader import SO101LeaderConfig
    # 函数内延迟导入：显示CLI帮助不必先导入整套LeRobot/相机/机器人依赖。

    try:
        config = load_config(args.config)
        # args.config可为None，此时load_config使用项目默认配置位置；返回嵌套dict。
    except FileNotFoundError:
        if args.config is not None:
            raise
        config = {}
    follower = dict(config.get("arms", {}).get(args.arm) or {})
    # get("arms",{})在缺键时给空dict；再按left/right取臂；dict(...)做浅复制以便本次覆盖。
    leader = dict(config.get("leader") or {})
    # or {}让缺失/空配置统一成空字典，后面的CLI参数可以补齐port/id。
    for destination, prefix in ((follower, "follower"), (leader, "leader")):
        # 外层tuple含两个二元组；for解包成“要修改的dict”和“CLI属性前缀”。
        for field in ("port", "id"):
            # 同一逻辑处理port、id，避免follower/leader各复制一份代码。
            explicit = getattr(args, prefix + "_" + field)
            # getattr按动态字符串取属性，例如getattr(args,"follower_port")。
            if explicit:
                destination[field] = explicit
                # CLI明确传值时覆盖这份本次配置；destination是follower/leader字典的引用。
            if not destination.get(field):
                raise ValueError(f"Configure {prefix} {field} or supply --{prefix}-{field}")
    cameras = {}
    # 未来的{相机键名: OpenCVCameraConfig实例}，不是图像帧字典。
    backend = {"any": 0, "msmf": 1400, "dshow": 700}[args.camera_backend]
    # 字典字面量建立名字→OpenCV后端编号映射，紧接[]按用户所选名字取一个值。
    from ..policy import policy_options
    mappings = policy_options(config)["cameras"]
    # policy_options定义在arms/policy.py，补齐机械臂策略的默认选项；取其中相机映射。
    specifications = args.camera or [f"{key}={index}" for key, index in mappings.items()]
    # 有CLI相机列表就使用它；否则列表推导把配置映射转成相同KEY=INDEX文本格式。
    for specification in specifications:
        key, separator, index = specification.partition("=")
        # str.partition只按第一个=切成三个字符串；没有=时separator为空，用来识别格式错误。
        key = key.removeprefix("observation.images.")
        # 去除可选公共前缀；录制配置使用base_0_rgb，数据feature保存时再有完整前缀。
        if not separator or not key or key in cameras:
            raise ValueError("Each camera needs a unique KEY=INDEX")
        cameras[key] = OpenCVCameraConfig(index_or_path=int(index), fps=args.fps,
                                        width=args.width, height=args.height, backend=backend)
        # 实例化官方相机配置；int把索引文本转整数；命名参数明确每个值的用途。
        # 这是描述分辨率/帧率/设备的对象，此行不等于已经取得一帧画面。
    for definition in (follower, leader):
        if definition.get("calibration_dir"):
            definition["calibration_dir"] = Path(definition["calibration_dir"])
            # 校准目录从JSON字符串转Path，以符合官方机器人配置的接口。
    robot = SO101FollowerConfig(**follower, cameras=cameras, use_degrees=True,
                                disable_torque_on_disconnect=False)
    # **follower把dict展开为命名参数，例如port="COM9",id="..."。
    # use_degrees让前五个关节采用角度表示；夹爪仍有自己的百分比归一化语义。
    teleop = SO101LeaderConfig(**leader, use_degrees=True)
    # teleop是主臂的遥操作配置；robot是从臂配置，它们不是同一个设备对象。
    # --task 是这段示教实际完成的任务标签，之后通过 task_index 与每帧绑定。
    dataset = DatasetRecordConfig(
        repo_id=args.repo_id or f"local/{args.root.name}", root=args.root,
        single_task=args.task.strip(), fps=args.fps, num_episodes=args.episodes,
        episode_time_s=args.episode_seconds, reset_time_s=args.reset_seconds,
        push_to_hub=False, no_stamp=True, streaming_encoding=True, encoder_threads=2,
    )
    # dataset配置规定任务名、路径、采集fps、每段时长与段数；不是已经写好的数据集。
    # repo_id是数据集标识，即使以local/开头仍是字符串；push_to_hub=False不上传Hub。
    # streaming_encoding=True让录制侧使用流式视频编码，避免把整段原始图像堆在内存。
    return RecordConfig(robot=robot, teleop=teleop, dataset=dataset, resume=args.resume,
                        display_data=args.display, play_sounds=False)


def release_service_for_recording(args, recording, *, client_factory=None):
    """Ask the configured service to exit before opening one of its serial ports."""
    # 【调用方】main在正式采集前调用；recording是build_config返回的RecordConfig。
    # 仅当配置端口重叠才请求服务退出；返回None或抛异常，不直接打开串口。
    # *后的client_factory必须按名字传；默认None用ArmClient，离线检查可传替代构造函数。
    from ..client import ArmClient

    try:
        service = load_config(args.config)
    except FileNotFoundError:
        if args.config is not None:
            raise
        return
    if service.get("mode") != "real":
        return
    configured = list(service["arms"].values()) + [service.get("leader") or {}]
    service_ports = {str(item.get("port", "")).strip().casefold() for item in configured}
    # 集合推导式收集端口，strip去空白、casefold统一大小写；set便于求交集。
    recording_ports = {str(device.port).strip().casefold() for device in (recording.robot, recording.teleop)}
    if not service_ports.intersection(recording_ports):
        # intersection返回两个端口集合的交集；空集表示不存在需要交接的同一端口。
        return
    # 采集器需要直接读写 leader/follower，不能与服务线程同时争用同一串口。
    # client_factory 是可替换依赖，便于离线验证这一交接过程而不连接真机。
    client = (client_factory or ArmClient)(args.config, timeout=2.0)
    # 先选择一个可调用构造器，再用(...)创建客户端实例；客户端封装本地HTTP交互。
    # Windows can take just over two seconds to report a refused localhost
    # connection. A short HTTP timeout alone cannot establish that it has exited.
    if not client.is_listening(timeout=3.0):
        return
    try:
        snapshot = client.status()
    except (OSError, RuntimeError):
        if not client.is_listening(timeout=3.0):
            return
        raise
    arms = snapshot.get("configured_arms")
    if (snapshot.get("mode") != service["mode"] or not isinstance(arms, list)
            or set(arms) != set(service["arms"])):
        raise RuntimeError("本机端点未返回当前机械臂配置的身份信息，未切换录制。")
    last_error = "服务仍在退出"
    try:
        result = client.command("shutdown")
    except (OSError, RuntimeError) as error:
        # The request may have reached the server. Do not submit it a second time.
        last_error = str(error)
    else:
        if result.get("success") is not True or result.get("status") != "shutting_down":
            raise RuntimeError(str(result.get("message") or "机械臂服务未接受关闭请求。"))
    deadline = time.monotonic() + args.service_shutdown_timeout
    # monotonic是不会随系统校时倒退的秒计时，适合计算等待超时；不是时间戳日期。
    while time.monotonic() < deadline:
        try:
            if not client.is_listening(timeout=3.0):
                return
        except (OSError, RuntimeError) as error:
            last_error = str(error)
        time.sleep(0.1)
    raise RuntimeError(f"机械臂服务关闭未确认，尚未开始录制：{last_error}")


class _RecordingControls:
    """Gate each official recording loop and keep controls within their phase."""

    def __init__(self):
        # self是这次采集的控制对象；面板、键盘监听和录制线程共享它的状态。
        self.events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
        self._lock = RLock()
        # threading.RLock是可重入锁，同一线程持锁时可再次进入；保护共享阶段/事件。
        self._pressed = set()
        self._phase = "connecting"
        self._message = "正在准备数据集与设备；连接完成后请点击开始本段。"
        self._start_requested = False
        self._episode = None
        self._initial_episode = None
        self._frames_at_end = 0
        self._dataset = None
        self._original_clear = None
        self._clear_was_local = False
        self.listener = None

    @staticmethod
    def _key_name(key):
        # staticmethod不自动传self；它只规范化输入key，不读取当前实例状态。
        name = key if isinstance(key, str) else getattr(key, "name", None) or getattr(key, "char", None)
        return name.lower() if isinstance(name, str) else None

    def press(self, key):
        name = self._key_name(key)
        control = name if name in ("right", "left", "esc") else None
        if control is None:
            return
        with self._lock:
            if name in self._pressed:
                return
            self._pressed.add(name)
            self.command({"right": "save", "left": "rerecord", "esc": "finish"}[control],
                         source={"right": "→", "left": "←", "esc": "Esc"}[control])

    def command(self, command, *, source="按钮"):
        # 【调用方】GUI按钮/press；把开始/保存/重录/结束意图写入共享events与提示文本。
        from lerobot.utils.keyboard_input import apply_recording_control

        with self._lock:
            if self._phase in ("done", "error"):
                return
            if command == "finish":
                self._message = f"已收到{source}：结束并保存已有帧，正在等待录制器退出。"
                self._start_requested = False
                apply_recording_control("esc", self.events)
                # Esc keeps frames even when Left was pressed just beforehand.
                self.events["rerecord_episode"] = (
                    self._dataset is not None and not self._dataset.has_pending_frames())
            elif self.events["stop_recording"]:
                self._message = "已请求结束，正在保存和关闭设备。"
            elif command == "start" and self._phase == "waiting":
                self._start_requested = True
                self.events["exit_early"] = True
                self._message = f"已收到{source}：开始本段。"
            elif command in ("save", "rerecord") and self._phase in ("record", "reset"):
                control = "right" if command == "save" else "left"
                apply_recording_control(control, self.events)
                self._message = (f"已收到{source}：保存本段，下一段等待手动开始。" if command == "save"
                                 else f"已收到{source}：丢弃本段，等待手动重新开始。")
            else:
                self._message = (f"已收到{source}，当前未录制；请点击开始本段。" if self._phase == "waiting"
                                 else f"已收到{source}，当前阶段不接受此操作，请按面板状态操作。")
            print(self._message, flush=True)

    def release(self, key):
        with self._lock:
            self._pressed.discard(self._key_name(key))

    def listen(self):
        from lerobot.utils import keyboard_input

        if keyboard_input.pynput_can_capture() and keyboard_input.keyboard is not None:
            self.listener = keyboard_input.keyboard.Listener(on_press=self.press, on_release=self.release)
            self.listener.start()
            if keyboard_input.pynput_listener_is_trusted(self.listener):
                return self.listener, self.events
            self.listener.stop()

        # Terminal backends expose discrete names, without key-release events.
        def tap(name):
            name = {"n": "right", "r": "left", "q": "esc"}.get(name.lower(), name.lower())
            self.press(name)
            self.release(name)

        self.listener = keyboard_input.create_key_listener(tap, controls_help="Right/Left/Esc, or n/r/q")
        return self.listener, self.events

    def run_loop(self, original, *args, **kwargs):
        # 【调用方】_record_with_controls临时安装的包装函数；original是官方record_loop。
        # *args收集额外位置参数成tuple，**kwargs收集命名参数成dict，便于原样转发。
        if kwargs.get("events") is not self.events:
            return original(*args, **kwargs)
            # 调用处的*和**是解包，把收集到的参数恢复为普通函数调用。
        dataset = kwargs.get("dataset")
        with self._lock:
            if dataset is not None:
                if self._dataset is None:
                    self._dataset = dataset
                    self._initial_episode = dataset.num_episodes
                    self._original_clear = dataset.clear_episode_buffer
                    self._clear_was_local = "clear_episode_buffer" in vars(dataset)
                    dataset.clear_episode_buffer = self.clear_episode_buffer
                self._episode = dataset.num_episodes
                self._frames_at_end = 0
                self._start_requested = False
                self._phase = "waiting"
                self._message = "等待开始：可以移动主臂复位，当前不记录。准备好后点击开始本段。"
            else:
                self._phase = "reset"
                outcome = "丢弃" if self.events["rerecord_episode"] else "保存"
                self._message = f"复位中：主臂继续跟随，不记录；本段随后{outcome}，下一段等待手动开始。"
            self.events["exit_early"] = self.events["stop_recording"]
        try:
            if dataset is not None:
                # Keep official device/camera ownership and teleoperation while
                # waiting, without frames or pollution of the recording timer.
                waiting = dict(kwargs, dataset=None, timer=None, control_time_s=float("inf"))
                original(*args, **waiting)
                with self._lock:
                    if self.events["stop_recording"]:
                        return
                    if not self._start_requested:
                        raise RuntimeError("等待开始阶段意外结束，未启动录制")
                    self._phase = "record"
                    self._message = "正在录制；→或保存本段结束，←重录，Esc结束并保存。"
                    self.events["exit_early"] = False
            return original(*args, **kwargs)
        finally:
            with self._lock:
                if dataset is not None:
                    self._frames_at_end = self._frame_count()
                self._phase = "saving"
                self.events["exit_early"] = self.events["stop_recording"]
                if self._dataset is not None:
                    pending = self._dataset.has_pending_frames()
                    if self.events["stop_recording"]:
                        self.events["rerecord_episode"] = not pending
                    elif dataset is not None and not pending:
                        # The official outer loop retries rerecords without saving
                        # an empty episode or consuming an episode-count slot.
                        self.events["rerecord_episode"] = True
                    if not self.events["stop_recording"]:
                        self._message = ("本段将丢弃，下一段等待手动开始。" if self.events["rerecord_episode"]
                                         else "本段已结束，正在处理与保存；下一段等待手动开始。")

    def _frame_count(self):
        writer = getattr(self._dataset, "writer", None)
        buffer = getattr(writer, "episode_buffer", None)
        return int(buffer.get("size", 0)) if isinstance(buffer, dict) else 0

    def snapshot(self):
        # 【调用方】面板刷新；持锁取一份状态dict供GUI展示，不把硬件对象传给GUI。
        with self._lock:
            saved = (self._dataset.num_episodes - self._initial_episode
                     if self._initial_episode is not None else 0)
            return {"phase": self._phase, "message": self._message, "episode": self._episode,
                    "frames": self._frame_count() if self._phase == "record" else self._frames_at_end,
                    "saved": saved, "stopping": self.events["stop_recording"]}

    def finished(self, error=None):
        with self._lock:
            self._phase = "error" if error is not None else "done"
            self._message = f"录制失败：{error}" if error is not None else "录制已结束，设备已关闭。"


    def clear_episode_buffer(self, *args, **kwargs):
        with self._lock:
            # Esc can arrive after the official loop chose its Left/discard
            # branch. Decide at the actual discard boundary, on the record thread.
            if self.events["stop_recording"] and self._dataset.has_pending_frames():
                self._dataset.save_episode()
            return self._original_clear(*args, **kwargs)

    def close(self):
        if self._original_clear is not None:
            if self._clear_was_local:
                self._dataset.clear_episode_buffer = self._original_clear
            else:
                del self._dataset.clear_episode_buffer
        if self.listener is not None:
            self.listener.stop()


def _record_with_controls(config, controls):
    """Run official recording on its one device-owning thread."""
    # 【调用方】record_demonstrations交给面板的worker；config为RecordConfig，controls为控制实例。
    # 下游官方lerobot_record.record负责真实设备与数据写入；本函数只包装入口/恢复函数。
    from lerobot.scripts import lerobot_record

    # 这里临时替换官方模块的函数（monkey patch），把事件交给面板；finally 必须恢复。
    # 这依赖安装版 LeRobot 的内部接口，升级时应核对 record_loop 的参数与保存行为。
    original_listener = lerobot_record.init_keyboard_listener
    # 不加()表示保存函数对象，而不是现在就启动监听；finally用它恢复原函数。
    original_loop = lerobot_record.record_loop
    lerobot_record.init_keyboard_listener = controls.listen
    lerobot_record.record_loop = lambda *args, **kwargs: controls.run_loop(original_loop, *args, **kwargs)
    # lambda匿名函数保留original_loop引用，再把未来收到的位置/命名参数转交run_loop。
    try:
        return lerobot_record.record(config)
    finally:
        lerobot_record.init_keyboard_listener = original_listener
        lerobot_record.record_loop = original_loop
        controls.close()


class _RecordingPanel:
    """Small main-thread Tk view; all device work stays in official recording."""
    # 【创建方】record_demonstrations；GUI只发controls命令和读snapshot，硬件操作交给worker。

    def __init__(self, controls):
        import tkinter as tk
        from tkinter import ttk

        self.controls = controls
        self.root = tk.Tk()
        self.root.title("Friston 示教录制控制")
        self.root.geometry("650x300")
        self.root.minsize(600, 280)
        self.root.attributes("-topmost", True)
        self._finished = False
        self._close_requested = False
        self._result = None
        self._error = None
        frame = ttk.Frame(self.root, padding=18)
        frame.pack(fill="both", expand=True)
        self.state = tk.StringVar()
        self.counts = tk.StringVar()
        self.feedback = tk.StringVar()
        ttk.Label(frame, textvariable=self.state, font=("Microsoft YaHei", 16, "bold")).pack(anchor="w")
        ttk.Label(frame, textvariable=self.counts, font=("Microsoft YaHei", 12)).pack(anchor="w", pady=8)
        ttk.Label(frame, textvariable=self.feedback, wraplength=600).pack(anchor="w", pady=8)
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=12)
        self.buttons = {}
        for command, label in (("start", "开始本段"), ("save", "保存本段 →"),
                               ("rerecord", "重录 ←"), ("finish", "结束并保存 Esc")):
            button = ttk.Button(row, text=label, command=lambda name=command: self._command(name))
            # lambda默认参数在创建时保存本轮command；避免所有按钮最后都读到循环最后一个值。
            button.pack(side="left", padx=(0, 8))
            self.buttons[command] = button
        ttk.Label(frame, text="只有“正在录制”时才写入帧。每段都需手动开始；等待/复位时仍可跟随主臂。",
                  wraplength=600).pack(anchor="w")
        self.root.protocol("WM_DELETE_WINDOW", self._request_close)
        for key, name in (("Right", "right"), ("Left", "left"), ("Escape", "esc")):
            self.root.bind(f"<KeyPress-{key}>", lambda event, n=name: controls.press(n))
            self.root.bind(f"<KeyRelease-{key}>", lambda event, n=name: controls.release(n))
        self._refresh()

    def _command(self, name):
        if self._finished and name == "finish":
            self.root.destroy()
        else:
            self.controls.command(name)

    def _request_close(self):
        if self._finished:
            self.root.destroy()
        else:
            self._close_requested = True
            self.controls.command("finish", source="关闭窗口")

    def _refresh(self):
        # 【调用方】初始化与Tk after回调；从controls取快照，更新文字/按钮，不连接硬件。
        state = self.controls.snapshot()
        phase = state["phase"]
        labels = {"connecting": "正在连接", "waiting": "等待开始 · 不记录", "record": "正在录制",
                  "reset": "正在复位 · 不记录", "saving": "正在保存 / 处理本段",
                  "done": "录制结束", "error": "录制失败"}
        self.state.set(labels[phase])
        episode = "—" if state["episode"] is None else state["episode"]
        self.counts.set(f"当前 episode：{episode}    本段帧数：{state['frames']}    本次已保存：{state['saved']} 段")
        self.feedback.set(state["message"])
        for command, button in self.buttons.items():
            enabled = (command == "finish" or (not state["stopping"] and
                       ((command == "start" and phase == "waiting") or
                        (command in ("save", "rerecord") and phase in ("record", "reset")))))
            button.configure(state="normal" if enabled else "disabled")
        if self._finished:
            self.buttons["finish"].configure(text="关闭")
            if self._close_requested:
                self.root.destroy()
                return
        self.root.after(100, self._refresh)
        # tkinter.after登记100毫秒后再调用这个函数；不加()表示传函数，不是现在递归执行。

    def run(self, record):
        # 【调用方】record_demonstrations；record是无参函数，由它在worker内启动官方录制。
        def worker():
            # 内部闭包记住self与record；线程结束时将结果/异常写回供主线程读取。
            try:
                self._result = record()
            except BaseException as error:
                self._error = error
            finally:
                self.controls.finished(self._error)
                self._finished = True

        # Tk 界面留在主线程；这个 worker 是录制期间实际持有硬件的线程。
        # 按钮只更新 controls/events，不能从 UI 回调绕过 worker 直接操作串口。
        thread = Thread(target=worker, name="lerobot-record")
        # threading.Thread构造线程对象；target=worker传函数，构造本身还没开始运行。
        thread.start()
        # start启动线程，在新线程调用worker；不要写worker()，那会在当前线程直接执行。
        try:
            self.root.mainloop()
            # Tk事件循环占据主线程处理按钮/窗口；耗时设备录制在上面的worker执行。
        finally:
            if thread.is_alive():
                self.controls.command("finish", source="控制面板退出")
            thread.join()
            # join等待录制线程完成清理后再返回，防止UI已退出而设备仍被后台线程占用。
        if self._error is not None:
            raise self._error
        return self._result


def record_demonstrations(config):
    """Require explicit episode starts in a visible panel around official recording."""
    # 【调用方】main；创建一次采集控制器与GUI，返回panel.run的结果。
    controls = _RecordingControls()
    # ClassName()构造实例并执行__init__；controls保存该实例引用。
    panel = _RecordingPanel(controls)
    # 把已创建的控制器传给GUI：依赖注入让两个对象操作同一份事件状态。
    return panel.run(lambda: _record_with_controls(config, controls))
    # 传入的是“稍后执行的函数”；此处lambda闭包记住config/controls，不提前调用录制器。


def main():
    # 【入口顺序】解析参数→组装配置→释放服务串口所有权→启动面板与正式采集。
    args = parse_args()
    # args是Namespace，后续使用args.arm/args.root等属性，而不是用字典[]。
    config = build_config(args)
    # config是RecordConfig实例，与原始JSON dict不同；配置类把不同模块的接口连接起来。
    release_service_for_recording(args, config)
    # 同一串口的旧服务退出后，官方录制线程才接管；这体现Single Owner的生命周期交接。
    record_demonstrations(config)
    # 到这里进入GUI和设备录制，函数返回后main结束。


if __name__ == "__main__":
    # 仅python -m ...record直接运行模块时执行；import本模块不会自动录制。
    main()
