"""Record SO-101 demonstrations with both configured cameras through LeRobot."""
from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
from threading import RLock, Thread
import time

from . import assets_root
from ..config import load_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--follower-port")
    parser.add_argument("--follower-id")
    parser.add_argument("--leader-port")
    parser.add_argument("--leader-id")
    parser.add_argument("--task", required=True, help="Task actually demonstrated in these episodes")
    parser.add_argument("--camera", action="append", metavar="KEY=INDEX",
                        help="Repeat for each view; default DirectShow base_0_rgb=0 and right_wrist_0_rgb=1")
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
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--service-shutdown-timeout", type=float, default=15.0,
                        help="Seconds to wait for the arm service to release shared serial ports")
    args = parser.parse_args(argv)
    if not args.task.strip():
        parser.error("--task must describe the demonstration")
    if min(args.width, args.height, args.fps, args.episodes, args.episode_seconds) <= 0 or args.reset_seconds < 0:
        parser.error("Dimensions, fps, episode count and duration must be positive")
    if args.resume and args.root is None:
        parser.error("--resume requires the existing dataset --root")
    if not math.isfinite(args.service_shutdown_timeout) or args.service_shutdown_timeout <= 0:
        parser.error("--service-shutdown-timeout must be finite and positive")
    if args.root is None:
        args.root = assets_root() / "datasets" / ("so101_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    args.root = args.root.expanduser().resolve()
    return args


def build_config(args):
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.scripts.lerobot_record import RecordConfig
    from lerobot.teleoperators.so_leader import SO101LeaderConfig

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        if args.config is not None:
            raise
        config = {}
    follower = dict(config.get("arms", {}).get(args.arm) or {})
    leader = dict(config.get("leader") or {})
    for destination, prefix in ((follower, "follower"), (leader, "leader")):
        for field in ("port", "id"):
            explicit = getattr(args, prefix + "_" + field)
            if explicit:
                destination[field] = explicit
            if not destination.get(field):
                raise ValueError(f"Configure {prefix} {field} or supply --{prefix}-{field}")
    cameras = {}
    backend = {"any": 0, "msmf": 1400, "dshow": 700}[args.camera_backend]
    from ..policy import policy_options
    mappings = policy_options(config)["cameras"]
    specifications = args.camera or [f"{key}={index}" for key, index in mappings.items()]
    for specification in specifications:
        key, separator, index = specification.partition("=")
        key = key.removeprefix("observation.images.")
        if not separator or not key or key in cameras:
            raise ValueError("Each camera needs a unique KEY=INDEX")
        cameras[key] = OpenCVCameraConfig(index_or_path=int(index), fps=args.fps,
                                        width=args.width, height=args.height, backend=backend)
    for definition in (follower, leader):
        if definition.get("calibration_dir"):
            definition["calibration_dir"] = Path(definition["calibration_dir"])
    robot = SO101FollowerConfig(**follower, cameras=cameras, use_degrees=True,
                                disable_torque_on_disconnect=False)
    teleop = SO101LeaderConfig(**leader, use_degrees=True)
    dataset = DatasetRecordConfig(
        repo_id=args.repo_id or f"local/{args.root.name}", root=args.root,
        single_task=args.task.strip(), fps=args.fps, num_episodes=args.episodes,
        episode_time_s=args.episode_seconds, reset_time_s=args.reset_seconds,
        push_to_hub=False, no_stamp=True, streaming_encoding=True, encoder_threads=2,
    )
    return RecordConfig(robot=robot, teleop=teleop, dataset=dataset, resume=args.resume,
                        display_data=args.display, play_sounds=False)


def release_service_for_recording(args, recording, *, client_factory=None):
    """Ask the configured service to exit before opening one of its serial ports."""
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
    recording_ports = {str(device.port).strip().casefold() for device in (recording.robot, recording.teleop)}
    if not service_ports.intersection(recording_ports):
        return
    client = (client_factory or ArmClient)(args.config, timeout=2.0)
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
        self.events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
        self._lock = RLock()
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
        if kwargs.get("events") is not self.events:
            return original(*args, **kwargs)
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
    from lerobot.scripts import lerobot_record

    original_listener = lerobot_record.init_keyboard_listener
    original_loop = lerobot_record.record_loop
    lerobot_record.init_keyboard_listener = controls.listen
    lerobot_record.record_loop = lambda *args, **kwargs: controls.run_loop(original_loop, *args, **kwargs)
    try:
        return lerobot_record.record(config)
    finally:
        lerobot_record.init_keyboard_listener = original_listener
        lerobot_record.record_loop = original_loop
        controls.close()


class _RecordingPanel:
    """Small main-thread Tk view; all device work stays in official recording."""

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

    def run(self, record):
        def worker():
            try:
                self._result = record()
            except BaseException as error:
                self._error = error
            finally:
                self.controls.finished(self._error)
                self._finished = True

        thread = Thread(target=worker, name="lerobot-record")
        thread.start()
        try:
            self.root.mainloop()
        finally:
            if thread.is_alive():
                self.controls.command("finish", source="控制面板退出")
            thread.join()
        if self._error is not None:
            raise self._error
        return self._result


def record_demonstrations(config):
    """Require explicit episode starts in a visible panel around official recording."""
    controls = _RecordingControls()
    panel = _RecordingPanel(controls)
    return panel.run(lambda: _record_with_controls(config, controls))


def main():
    args = parse_args()
    config = build_config(args)
    release_service_for_recording(args, config)
    record_demonstrations(config)


if __name__ == "__main__":
    main()
