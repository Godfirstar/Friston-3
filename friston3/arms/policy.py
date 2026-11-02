"""Persistent Pi0.5 inference, fresh camera frames and asynchronous action chunks.

Only ArmService owns the motors. This module passes measured observations to a
single model worker and returns one target per control tick, without disk logs.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import threading
import time

from .config import JOINTS


def policy_options(config):
    from .learning import assets_root

    root = assets_root()
    options = {
        "model_dir": str(root / "pi05_base"),
        "tokenizer_dir": str(root / "paligemma_tokenizer"),
        "adapter_dir": str(root / "checkpoints" / "step80"),
        "cameras": {"observation.images.base_0_rgb": 0,
                    "observation.images.right_wrist_0_rgb": 1},
        "camera_backend": "dshow", "camera_timeout": 10.0,
        "gpu": 0, "gpu_layers": 8, "chunk_mib": 64.0,
        "num_steps": 10, "cycles": 50, "rtc": True,
        "blend_steps": 8, "ground_scene": True, "verify_scene": True,
        "fps": config.get("fps", 20.0),
    }
    options.update(config.get("policy", {}))
    if float(options["fps"]) != float(config.get("fps", 20.0)):
        raise ValueError("policy.fps must match the arm service fps; configure the service control rate")
    for key in ("fps", "camera_timeout", "chunk_mib"):
        if not math.isfinite(float(options[key])) or float(options[key]) <= 0:
            raise ValueError(f"policy.{key} must be finite and positive")
    for key in ("num_steps", "cycles", "blend_steps"):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f"policy.{key} must be a positive integer")
    return options


class CameraRig:
    """Drain each USB camera continuously; copy the newest completed frames."""

    def __init__(self, mappings, backend="dshow", timeout=10.0):
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._frames = {}
        self._errors = {}
        self._threads = []
        self.timeout = timeout
        for key, index in mappings.items():
            worker = threading.Thread(target=self._capture, args=(key, index, backend),
                                      name=f"policy-camera-{index}", daemon=True)
            self._threads.append(worker)
            worker.start()
        self.keys = tuple(mappings)
        try:
            self.read()
        except BaseException:
            self.close()
            raise

    def _capture(self, key, index, backend):
        import cv2

        cap = None
        try:
            api = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW, "any": cv2.CAP_ANY}[backend]
            cap = cv2.VideoCapture(index, api)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open {key}: camera {index}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Camera {index} returned no frame")
                with self._condition:
                    self._frames[key] = (frame, time.perf_counter())
                    self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._errors[key] = str(error)
                self._condition.notify_all()
        finally:
            if cap is not None:
                cap.release()

    def read(self, *, newer_than=None):
        deadline = time.perf_counter() + self.timeout
        with self._condition:
            while not self._errors and any(
                    key not in self._frames or (newer_than is not None and self._frames[key][1] <= newer_than)
                    for key in self.keys):
                remaining = deadline - time.perf_counter()
                if remaining <= 0 or self._stop.is_set():
                    raise TimeoutError("Waiting for policy camera frames")
                self._condition.wait(remaining)
            if self._errors:
                raise RuntimeError(str(self._errors))
            now = time.perf_counter()
            frames, ages = {}, {}
            for key in self.keys:
                frame, stamp = self._frames[key]
                if now - stamp > self.timeout:
                    raise TimeoutError(f"Camera stopped updating: {key}")
                frames[key] = frame.copy()
                ages[key] = now - stamp
            return frames, ages

    def close(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        for worker in self._threads:
            worker.join(timeout=1.0)


class Pi05Policy:
    """Load the existing expert-LoRA checkpoint with its own training statistics."""

    def __init__(self, options):
        import numpy as np
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        from lerobot.policies.pi05 import processor_pi05
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy as OfficialPolicy
        from .learning import checkpoint_reader, loader, training

        self.options = options
        self.device = torch.device(f"cuda:{options['gpu']}")
        if not torch.cuda.is_available():
            raise RuntimeError("Pi0.5 needs the installed CUDA LeRobot environment")
        torch.cuda.set_device(self.device)
        adapter = Path(options["adapter_dir"])
        self.checkpoint = cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8-sig"))
        if cfg.get("dataset_fps") is not None and float(cfg["dataset_fps"]) != float(options["fps"]):
            raise ValueError("The checkpoint dataset_fps differs from the service control rate")
        if cfg.get("format") != "friston_pi05_expert_lora_v1":
            raise ValueError("Unsupported Pi0.5 adapter format")
        if cfg.get("joint_order") != [joint.removesuffix(".pos") for joint in JOINTS]:
            raise ValueError("Adapter joint order differs from the robot")
        if cfg.get("model_slots") != list(range(6)) or cfg.get("supervised_action_dims") != 6:
            raise ValueError("Adapter action mapping differs from the robot")
        for path, key in ((Path(options["model_dir"]) / "config.json", "base_config_sha256"),
                          (Path(loader.__file__), "loader_sha256")):
            if hashlib.sha256(path.read_bytes()).hexdigest() != cfg[key]:
                raise ValueError(f"Checkpoint dependency differs: {path.name}")
        with checkpoint_reader():
            self.model, _ = loader.load_pi05_lowmem(
                options["model_dir"], gpu=options["gpu"], gpu_layers=options["gpu_layers"],
                chunk_mib=options["chunk_mib"])
        targets = training.attach_trainables(self.model, int(cfg["rank"]), float(cfg["alpha"]))
        if targets != cfg["target_modules"]:
            raise ValueError("Adapter target modules differ")
        parameters = training.trainable_dict(self.model)
        saved = load_file(str(adapter / "trainable.safetensors"), device="cpu", backend="pread")
        if parameters.keys() != saved.keys():
            raise ValueError("Adapter parameter names differ")
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(saved[name].to(device=parameter.device, dtype=parameter.dtype))
        del saved
        self.model.eval().requires_grad_(False)
        self.horizon = self.model.config.chunk_size
        if cfg["horizon"] != self.horizon:
            raise ValueError("Adapter action horizon differs")
        self.image_keys = cfg.get("image_keys", [cfg["image_key"]])
        self.stats = tuple(np.asarray(cfg["statistics"][key][quantile], dtype=np.float32)
                           for key in ("observation.state", "action") for quantile in ("q01", "q99"))
        if any(v.shape != (6,) or not np.isfinite(v).all() for v in self.stats):
            raise ValueError("Adapter statistics must contain six finite values")
        if any(np.any(hi <= lo) for lo, hi in (self.stats[:2], self.stats[2:])):
            raise ValueError("Adapter quantile spans must be positive")
        self.tokenizer = AutoTokenizer.from_pretrained(options["tokenizer_dir"], local_files_only=True)
        self.tokenizer.padding_side = "right"
        self.processor = processor_pi05
        self.state_step = processor_pi05.Pi05PrepareStateTokenizerProcessorStep(
            max_state_dim=self.model.config.max_state_dim)
        from types import SimpleNamespace
        self.proxy = SimpleNamespace(config=self.model.config,
                                     parameters=lambda: iter([torch.empty(0, device=self.device)]))
        self.image_fn = OfficialPolicy._preprocess_images
        if options["rtc"]:
            from lerobot.policies.rtc.configuration_rtc import RTCConfig
            from lerobot.policies.rtc.modeling_rtc import RTCProcessor
            rtc = RTCConfig(enabled=True, mode="guided", execution_horizon=options["blend_steps"])
            self.model.config.rtc_config = rtc
            self.model.rtc_processor = RTCProcessor(rtc)

    def predict(self, frames, observation, task, previous=None, delay=0):
        import cv2
        import numpy as np
        import torch

        state = np.asarray([observation[k] for k in JOINTS], dtype=np.float32)
        if not np.isfinite(state).all():
            raise ValueError("Nonfinite robot observation")
        lo, hi, action_lo, action_hi = self.stats
        normalized = 2 * (state - lo) / (hi - lo) - 1
        padded = torch.nn.functional.pad(torch.from_numpy(normalized)[None],
                                         (0, self.model.config.max_state_dim - 6))
        p = self.processor
        transition = {p.TransitionKey.OBSERVATION: {p.OBS_STATE: padded},
                      p.TransitionKey.COMPLEMENTARY_DATA: {"task": [task]}}
        prompt = self.state_step(transition)[p.TransitionKey.COMPLEMENTARY_DATA]["task"]
        encoded = self.tokenizer(prompt, padding=True, truncation=True,
                                 max_length=self.model.config.tokenizer_max_length, return_tensors="pt")
        images = {}
        for key in self.image_keys:
            rgb = cv2.cvtColor(frames[key], cv2.COLOR_BGR2RGB)
            images[key] = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float() / 255
        with torch.no_grad():
            images, masks = self.image_fn(self.proxy, images)
        # Missing views contain only masked padding tokens. They contribute no
        # keys to attention; omit their vision tower and prefix computation.
        valid = [(image, mask) for image, mask in zip(images, masks) if bool(mask.any())]
        inputs = {"images": [v[0] for v in valid], "img_masks": [v[1] for v in valid],
                  "tokens": encoded["input_ids"].to(self.device),
                  "masks": encoded["attention_mask"].to(self.device, dtype=torch.bool)}
        kwargs = {}
        if self.options["rtc"] and previous:
            prefix = torch.as_tensor(np.asarray(previous), device=self.device, dtype=torch.float32)
            kwargs = {"prev_chunk_left_over": prefix, "inference_delay": min(delay, len(previous)),
                      "execution_horizon": min(len(previous), delay + self.options["blend_steps"])}
        # Guided RTC needs its internal Jacobian; inference_mode would disable it.
        raw = self.model.sample_actions(**inputs, num_steps=self.options["num_steps"], **kwargs)
        raw = raw.detach().float().cpu().numpy()[0]
        if raw.shape != (self.horizon, self.model.config.max_action_dim) or not np.isfinite(raw).all():
            raise ValueError("Invalid Pi0.5 action chunk")
        actions = (raw[:, :6] + 1) * (action_hi - action_lo) / 2 + action_lo
        return raw.tolist(), actions.tolist()


class PolicySession:
    """CPU control queue and one persistent model worker, with task generations."""

    def __init__(self, config, *, policy_factory=None, camera_factory=None):
        self.options = policy_options(config)
        self._policy_factory = policy_factory or Pi05Policy
        self._camera_factory = camera_factory or CameraRig
        self._condition = threading.Condition()
        self._closed = False
        self._generation = 0
        self._queue = deque()
        self._state = {}
        self._status = {"status": "idle", "message": "策略尚未接收任务。", "metrics": {}}
        self._sent = self._predictions = 0
        self._acknowledged = 0
        self._last_ack = self._state_clock = 0.0
        self._last_send = self._first_send = None
        self._latency = 0.0
        self._warm_latencies = deque(maxlen=4)
        self._worker = threading.Thread(target=self._run, name="pi05-inference", daemon=True)
        self._worker.start()

    def start(self, instruction, arm, positions):
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("任务指令不能为空")
        with self._condition:
            if self._closed:
                raise RuntimeError("策略服务已经关闭")
            self._generation += 1
            self._queue.clear()
            self._state = dict(positions)
            self._sent = self._predictions = 0
            self._warm_latencies.clear()
            self._acknowledged = 0
            self._last_ack = self._state_clock = 0.0
            self._last_send = self._first_send = None
            self._status = {"status": "loading", "message": "正在读取场景并准备策略。",
                            "instruction": instruction, "arm": arm, "scene": {},
                            "metrics": {"commands": 0, "chunks": 0, "max_send_gap_s": 0.0,
                                        "queue_empty_ticks": 0}, "task_success": None}
            self._condition.notify_all()

    def tick(self, positions):
        with self._condition:
            self._state = dict(positions)
            self._state_clock = time.perf_counter()
            if self._status["status"] not in ("loading", "running"):
                return None
            self._condition.notify_all()
            if not self._queue:
                if self._last_send is not None and self._predictions < self.options["cycles"]:
                    self._status["metrics"]["queue_empty_ticks"] += 1
                return None
            _, target = self._queue.popleft()
            now = time.perf_counter()
            if self._last_send is not None:
                metrics = self._status["metrics"]
                metrics["max_send_gap_s"] = max(metrics["max_send_gap_s"], now - self._last_send)
            self._first_send = self._first_send or now
            self._last_send = now
            self._sent += 1
            self._status["metrics"].update(commands_requested=self._sent, queued_actions=len(self._queue))
            return dict(zip(JOINTS, target))

    def acknowledge(self, applied):
        """Called by the serial owner only after a successful hardware send."""
        with self._condition:
            self._acknowledged = self._sent
            self._last_ack = time.perf_counter()
            self._status["metrics"]["commands"] = self._acknowledged
            self._status["metrics"]["commands_sent"] = self._acknowledged
            self._status["metrics"]["last_applied"] = dict(applied)
            self._condition.notify_all()

    def status(self):
        with self._condition:
            return deepcopy(self._status)

    def stop(self):
        with self._condition:
            self._generation += 1
            self._queue.clear()
            if self._status["status"] in ("loading", "running"):
                self._status.update(status="cancelled", message="策略任务已取消。")
            self._condition.notify_all()

    def close(self):
        self.stop()
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._worker.join(timeout=2.0)

    def _current(self, generation):
        return generation == self._generation and not self._closed

    def _run(self):
        policy = camera = None
        current = -1
        task = ""
        before = None
        try:
            while True:
                with self._condition:
                    idle_camera = camera if self._status["status"] not in ("loading", "running") else None
                    if idle_camera is not None:
                        camera = None
                if idle_camera is not None:
                    idle_camera.close()
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or self._status["status"] in ("loading", "running"))
                    if self._closed:
                        break
                    generation = self._generation
                    instruction = self._status["instruction"]
                    if current == generation:
                        if self._predictions >= self.options["cycles"]:
                            if (self._queue or self._acknowledged < self._sent
                                    or self._state_clock <= self._last_ack):
                                self._condition.wait(0.02)
                                continue
                            finishing = True
                        else:
                            finishing = False
                            # Prefetch for recent warm jitter; RTC still uses the
                            # last measured delay rather than this queue reserve.
                            latency = max(self._warm_latencies, default=self._latency)
                            refill = math.ceil(latency * self.options["fps"]) + 4
                            if len(self._queue) > refill:
                                self._condition.wait(0.02)
                                continue
                    else:
                        finishing = False
                try:
                    if current != generation:
                        if camera is None:
                            metadata = json.loads((Path(self.options["adapter_dir"]) / "adapter_config.json")
                                                  .read_text(encoding="utf-8-sig"))
                            keys = metadata.get("image_keys", [metadata["image_key"]])
                            mappings = {key: self.options["cameras"][key] for key in keys}
                            camera = self._camera_factory(mappings, self.options["camera_backend"],
                                                          self.options["camera_timeout"])
                        frames, _ = camera.read()
                        task, scene = instruction, {}
                        if self.options["ground_scene"]:
                            from .scene import ground_task
                            scene_options = dict(self.options)
                            scene_options.setdefault("scene_num_gpu", -1 if policy is None else 0)
                            scene = ground_task(instruction, frames, scene_options)
                            task = scene["task"]
                        with self._condition:
                            if not self._current(generation):
                                continue
                            self._status.update(scene=scene, message="正在加载策略。")
                        if policy is None:
                            from .scene import release_dialogue_gpu
                            release_dialogue_gpu(self.options)
                            policy = self._policy_factory(self.options)
                        before = scene
                        current = generation
                    if finishing:
                        verification = {"success": None, "evidence": "尚未验证抓取结果。"}
                        if self.options["verify_scene"]:
                            from .scene import inspect_completion
                            frames, _ = camera.read(newer_than=self._last_ack)
                            verification = inspect_completion(instruction, before, frames, self.options)
                        with self._condition:
                            if self._current(generation):
                                detail = verification.get("fact") or verification.get("uncertainty") or "尚未验证取物结果。"
                                self._status.update(status="completed", task_success=verification.get("success"),
                                                    verification=verification,
                                                    message="策略执行周期结束。" + detail)
                        continue
                    began = time.perf_counter()
                    frames, ages = camera.read()
                    with self._condition:
                        if not self._current(generation):
                            continue
                        observed = dict(self._state)
                        sent_before = self._sent
                        previous = [raw for raw, _ in self._queue]
                    expected_delay = math.ceil(self._latency * self.options["fps"])
                    original, actions = policy.predict(frames, observed, task, previous, expected_delay)
                    elapsed = time.perf_counter() - began
                    with self._condition:
                        if not self._current(generation):
                            continue
                        consumed = self._sent - sent_before
                        # Drop exactly the prefix already executed while this
                        # prediction was in flight; never replay expired actions.
                        self._queue = deque(zip(original[consumed:], actions[consumed:]))
                        self._latency = elapsed
                        if self._predictions:
                            self._warm_latencies.append(elapsed)
                        self._predictions += 1
                        self._status.update(status="running", message="策略正在根据实时画面执行任务。")
                        self._status["metrics"].update(
                            chunks=self._predictions, inference_s=elapsed,
                            inference_delay_frames=consumed, queued_actions=len(self._queue),
                            camera_age_s=ages, model_horizon=policy.horizon,
                            horizon_seconds=policy.horizon / self.options["fps"])
                        self._condition.notify_all()
                except Exception as error:
                    with self._condition:
                        if self._current(generation):
                            self._queue.clear()
                            self._status.update(status="failed", message=f"{type(error).__name__}: {error}")
                    if camera is not None:
                        camera.close()
                        camera = None
        finally:
            if camera is not None:
                camera.close()
