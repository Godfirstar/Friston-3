"""机械臂策略会话。ArmService 按控制频率提供观察并取出下一帧关节目标。

CameraRig 持续采集图像，PolicySession 的后台线程用最新画面请求动作块或高层子任务。
模型计算由 models.pi05.Pi05Policy 负责。串口由 ArmService 的控制线程持有。
"""
from __future__ import annotations

# 标准库：deque 是双端队列；deepcopy 复制嵌套对象；threading 管理线程与同步。
# 相机采集线程按需导入 OpenCV，模型工作线程按需导入 π₀.₅。
from collections import deque
from copy import deepcopy
import json
import math
from pathlib import Path
import threading
import time

from .config import JOINT_FORMAT, JOINTS


def __getattr__(name):
    # 保留已有的显式导入路径，读取策略配置或创建会话时仍可只用标准库。
    if name == "Pi05Policy":
        from .models.pi05 import Pi05Policy
        return Pi05Policy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def policy_options(config):
    # PolicySession.__init__ 调用；输入服务配置 dict，返回合并默认值后的策略配置 dict。
    from .models import assets_root, resolve_asset_path

    root = assets_root()
    # root 是 pathlib.Path，/ 在这里拼接路径。
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
        "subtask_mode": "direct", "replan_steps": 50, "semantic_adapter_dir": None,
        "fps": config.get("fps", 20.0),
    }
    provided = config.get("policy", {})
    options.update(provided)
    # dict.get 第二个参数是缺省值；update 用用户配置覆盖同名默认项。
    # 配置可能仍保存迁移前的绝对路径；仅定位新根下同名资产，不选择别的检查点。
    for key in ("model_dir", "tokenizer_dir", "adapter_dir"):
        options[key] = str(resolve_asset_path(options[key]))
    action_dir = Path(options["adapter_dir"])
    metadata_file = action_dir / "adapter_config.json"
    metadata = {}
    if metadata_file.is_file() or options["subtask_mode"] == "trained":
        metadata = json.loads(metadata_file.read_text(encoding="utf-8-sig"))
    options["checkpoint_format"] = metadata.get("format")
    joint = options["checkpoint_format"] == JOINT_FORMAT
    if joint:
        if "subtask_mode" not in provided:
            options["subtask_mode"] = "trained"
        if "rtc" not in provided:
            options["rtc"] = False
        if "replan_steps" not in provided:
            # 每两个完整动作块复看一次；论文只规定高层频率较低，具体帧数可覆盖。
            options["replan_steps"] = 2 * int(metadata["horizon"])
    if options["subtask_mode"] not in ("direct", "external", "trained"):
        raise ValueError("policy.subtask_mode must be direct, external or trained")
    if options["subtask_mode"] == "trained":
        if joint:
            directory = action_dir
        elif options["semantic_adapter_dir"]:
            directory = resolve_asset_path(options["semantic_adapter_dir"])
        else:
            raise ValueError("Legacy action checkpoint requires a separate --semantic-adapter-dir in trained mode")
        for filename in ("adapter_config.json", "trainable.safetensors"):
            if not (directory / filename).is_file():
                raise FileNotFoundError(f"Missing trained semantic checkpoint file: {directory / filename}")
        options["semantic_adapter_dir"] = str(directory)
    if float(options["fps"]) != float(config.get("fps", 20.0)):
        raise ValueError("policy.fps must match the arm service fps; configure the service control rate")
    for key in ("fps", "camera_timeout", "chunk_mib"):
        if not math.isfinite(float(options[key])) or float(options[key]) <= 0:
            raise ValueError(f"policy.{key} must be finite and positive")
    for key in ("num_steps", "cycles", "blend_steps", "replan_steps"):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f"policy.{key} must be a positive integer")
    return options


class CameraRig:
    """Drain each USB camera continuously; copy the newest completed frames."""

    def __init__(self, mappings, backend="dshow", timeout=10.0):
        # PolicySession._run 创建此对象。mappings 是“模型图像键 -> 摄像头编号”的 dict。
        # self 指这个相机组对象；backend 是后端名 str，timeout 是等待帧的秒数。
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._frames = {}
        self._errors = {}
        self._threads = []
        self.timeout = timeout
        for key, index in mappings.items():
            # items 每次给出 (键, 值)，for key,index 自动拆开这个二元组。
            worker = threading.Thread(target=self._capture, args=(key, index, backend),
                                      name=f"policy-camera-{index}", daemon=True)
            # target 传方法，args 是调用它的位置参数元组；start 才创建实际执行流。
            self._threads.append(worker)
            worker.start()
        self.keys = tuple(mappings)
        # tuple(dict) 固定保存所有视角键；不是图像内容，帧稍后写入 self._frames。
        try:
            self.read()
        except BaseException:
            self.close()
            raise

    def _capture(self, key, index, backend):
        # 每个相机线程的入口；cv2 是 opencv-python 包的 Python 模块名。
        import cv2

        cap = None
        try:
            api = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW, "any": cv2.CAP_ANY}[backend]
            cap = cv2.VideoCapture(index, api)
            # cap 是 OpenCV 的视频采集对象；它在此线程创建、读取并最终 release。
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open {key}: camera {index}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            while not self._stop.is_set():
                ok, frame = cap.read()
                # OpenCV 返回 (是否成功 bool, 图像 ndarray)；图像通常为 uint8 [H,W,3] BGR。
                if not ok or frame is None:
                    raise RuntimeError(f"Camera {index} returned no frame")
                with self._condition:
                    # 每路只保留最新帧，不积压旧画面；Condition 唤醒等待图像的推理线程。
                    self._frames[key] = (frame, time.perf_counter())
                    # 字典值存 (图像, 单调时间戳)，让 read 能计算这帧已经存在多久。
                    self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._errors[key] = str(error)
                self._condition.notify_all()
        finally:
            # finally 在正常结束/异常后都会执行，不把相机占用留给下一次任务。
            if cap is not None:
                cap.release()

    def read(self, *, newer_than=None):
        # PolicySession._run 调用，返回 (frames, ages) 两个 dict；不会在这里访问串口。
        # 参数中的 * 表示 newer_than 必须按名字传，例如 read(newer_than=某时间)。
        deadline = time.perf_counter() + self.timeout
        with self._condition:
            while not self._errors and any(
                    key not in self._frames or (newer_than is not None and self._frames[key][1] <= newer_than)
                    for key in self.keys):
                remaining = deadline - time.perf_counter()
                if remaining <= 0 or self._stop.is_set():
                    raise TimeoutError("Waiting for policy camera frames")
                self._condition.wait(remaining)
                # Condition.wait 等待时会释放条件锁，采集线程才能拿锁写帧；醒来后重新持锁。
            if self._errors:
                raise RuntimeError(str(self._errors))
            now = time.perf_counter()
            frames, ages = {}, {}
            for key in self.keys:
                frame, stamp = self._frames[key]
                if now - stamp > self.timeout:
                    raise TimeoutError(f"Camera stopped updating: {key}")
                frames[key] = frame.copy()
                # 复制使调用方可独立处理图像；age 衡量新鲜度，不代表图像与关节硬同步。
                ages[key] = now - stamp
            return frames, ages

    def close(self):
        # PolicySession 结束任务/退出时调用；通知采集线程退出，再等待其清理相机。
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        for worker in self._threads:
            worker.join(timeout=1.0)


class PolicySession:
    """CPU control queue and one persistent model worker, with task generations."""

    def __init__(self, config, *, policy_factory=None, camera_factory=None):
        # ArmService._start_job 创建此对象；factory 是“可调用的构造函数/类”，不是已加载模型。
        self.options = policy_options(config)
        self._full_chunks = self.options["checkpoint_format"] == JOINT_FORMAT and not self.options["rtc"]
        self._policy_factory = policy_factory
        self._camera_factory = camera_factory or CameraRig
        # 注入工厂可换成假相机/假模型，验证队列与取消逻辑时无需加载大模型或连接设备。
        self._condition = threading.Condition()
        # Condition 同时提供共享锁与等待/唤醒；以下队列和状态由它协调两个线程的访问。
        self._closed = False
        self._generation = 0
        self._queue = deque()
        # deque 每项是 (raw_action, physical_target)：前者给 RTC，后者给真实发送。
        self._state = {}
        # _state 是最新实测六关节 dict；_status 是供 ArmService 查询的状态 dict。
        self._status = {"status": "idle", "message": "策略尚未接收任务。", "metrics": {}}
        self._sent = self._predictions = 0
        # _sent 记录 tick 已取走多少目标；_acknowledged 记录 send_policy 成功确认到第几个。
        # _predictions 记录已完成多少次动作块生成，和单帧计数不同。
        self._acknowledged = 0
        self._last_ack = self._state_clock = 0.0
        self._last_send = self._first_send = None
        self._latency = 0.0
        self._warm_latencies = deque(maxlen=4)
        # maxlen 使这个小队列自动淘汰最早样本，用最近4次热推理耗时估算预取时机。
        self._worker = threading.Thread(target=self._run, name="pi05-inference", daemon=True)
        self._worker.start()

    def start(self, instruction, arm, positions):
        # ArmService owner 调用；输入任务str、臂名str、实测姿态dict，返回 None。
        # 真正的相机与模型工作留给 _run；这里仅更新共享字段并唤醒它。
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("任务指令不能为空")
        with self._condition:
            if self._closed:
                raise RuntimeError("策略服务已经关闭")
            self._generation += 1
            self._queue.clear()
            # clear 清空旧任务尚未执行的目标；dict(positions) 复制六个标量值组成的新字典。
            self._state = dict(positions)
            self._sent = self._predictions = 0
            self._warm_latencies.clear()
            self._acknowledged = 0
            self._next_review = 0
            self._last_ack = self._state_clock = 0.0
            self._last_send = self._first_send = None
            self._status = {"status": "loading", "message": "正在读取场景并准备策略。",
                            "instruction": instruction, "arm": arm, "scene": {},
                            "global_goal": instruction, "current_subtask": None,
                            "subtask_mode": self.options["subtask_mode"], "termination_reason": None,
                            "metrics": {"commands": 0, "chunks": 0, "max_send_gap_s": 0.0,
                                        "queue_empty_ticks": 0}, "task_success": None}
            self._condition.notify_all()
            # 通知等待这个 Condition 的线程重新检查状态；当前 with 退出后它才能获取锁。

    def tick(self, positions):
        # 消费者：ArmService 控制线程每轮调用。只取一个目标，不在这里做慢推理。
        with self._condition:
            self._state = dict(positions)
            self._state_clock = time.perf_counter()
            # state_clock 标记本轮观察更新时间；结束判定用它确认最后发送后又观察过一次。
            if self._status["status"] not in ("loading", "running"):
                return None
            self._condition.notify_all()
            # 已到观察时刻，保留剩余队列直至高层返回；等待期间 owner 仍持续刷新实测关节。
            if self._review_due():
                return None
            if not self._queue:
                # 空队列的 bool 值为 False；没有可执行帧就返回 None，不在控制线程等待模型。
                if self._last_send is not None and self._predictions < self.options["cycles"]:
                    self._status["metrics"]["queue_empty_ticks"] += 1
                return None
            _, target = self._queue.popleft()
            # popleft 取出并删除最早一项；_ 按惯例表示此处不使用该项的 raw_action 部分。
            now = time.perf_counter()
            if self._last_send is not None:
                metrics = self._status["metrics"]
                metrics["max_send_gap_s"] = max(metrics["max_send_gap_s"], now - self._last_send)
            self._first_send = self._first_send or now
            self._last_send = now
            self._sent += 1
            self._status["metrics"].update(commands_requested=self._sent, queued_actions=len(self._queue))
            return dict(zip(JOINTS, target))
            # zip 配对关节名与六个数字，dict 变成 send_policy 所需的具名目标。

    def acknowledge(self, applied):
        """Called by the serial owner only after a successful hardware send."""
        with self._condition:
            # 由 owner 在 send_policy 成功后调用；applied 是驱动实际接受的六关节目标 dict。
            self._acknowledged = self._sent
            self._last_ack = time.perf_counter()
            self._status["metrics"]["commands"] = self._acknowledged
            self._status["metrics"]["commands_sent"] = self._acknowledged
            self._status["metrics"]["last_applied"] = dict(applied)
            self._condition.notify_all()

    def status(self):
        # 调用方读取副本，不得到可直接修改内部共享状态的引用。
        with self._condition:
            return deepcopy(self._status)

    def stop(self):
        with self._condition:
            # 不终止正在锁外运行的模型前向；发布结果前的 _current 会丢弃旧结果。
            # 这个计数器属于 PolicySession，和 ArmService 的 generation 各自独立。
            self._generation += 1
            self._queue.clear()
            if self._status["status"] in ("loading", "running"):
                self._status.update(status="cancelled", termination_reason="cancelled", message="策略任务已取消。")
                self._status["metrics"]["queued_actions"] = 0
            self._condition.notify_all()

    def close(self):
        # ArmService 退出时调用；通知 _run 退出，join 只等待，不强行杀掉模型计算。
        self.stop()
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._worker.join(timeout=2.0)

    def _current(self, generation):
        # 小型谓词：返回 bool，判断某次锁外计算是否仍属于当前任务。
        return generation == self._generation and not self._closed

    def _review_due(self):
        # 发送帧数只决定何时再看图；只有高层读过新图后才有完成/切换结论。
        return self.options["subtask_mode"] != "direct" and self._acknowledged >= self._next_review

    def _next_work(self):
        """在 Condition 锁内决定下一次模型工作；None 表示等待 owner 的通知。"""
        if self._status["status"] == "loading":
            return "prepare"
        exhausted = self._predictions >= self.options["cycles"]
        if self._review_due() or (exhausted and not self._queue):
            if self._acknowledged != self._sent or (self._sent and self._state_clock <= self._last_ack):
                return None
            return "finish" if self.options["subtask_mode"] == "direct" else "review"
        if exhausted:
            return None
        if self._full_chunks:
            # 用下一次真实观察生成新动作块；同一动作块逐帧按次序发送。
            if self._queue or self._acknowledged != self._sent or (self._sent and self._state_clock <= self._last_ack):
                return None
            return "predict"
        # 同一子任务内继续根据近期实际延迟预取，为 RTC 保留旧块尚未消费的轨迹。
        latency = max(self._warm_latencies, default=self._latency)
        refill = math.ceil(latency * self.options["fps"]) + 4
        return "predict" if len(self._queue) <= refill else None

    def _prepare(self, policy, camera, generation, global_goal):
        scene = {}
        current_subtask = None
        if self.options["subtask_mode"] == "direct":
            current_subtask = global_goal
            if self.options["ground_scene"]:
                from .scene import ground_task
                frames, _ = camera.read()
                scene_options = dict(self.options)
                scene_options.setdefault("scene_num_gpu", -1 if policy is None else 0)
                scene = ground_task(global_goal, frames, scene_options)
                current_subtask = scene["task"]
        elif self.options["subtask_mode"] == "trained":
            scene = {"assessment": "trained_semantic_policy",
                     "checkpoint": self.options["semantic_adapter_dir"],
                     "capability_validation": "子任务与完成信号来自模型预测；本服务未验证该检查点的真机成功率。"}
        with self._condition:
            if not self._current(generation):
                return policy
            self._status.update(scene=scene, message="正在加载策略。")
        if policy is None:
            from .scene import release_dialogue_gpu
            release_dialogue_gpu(self.options)
            if self._policy_factory is None:
                from .models.pi05 import Pi05Policy
                self._policy_factory = Pi05Policy
            policy = self._policy_factory(self.options)
        with self._condition:
            if self._current(generation):
                self._status.update(status="running", current_subtask=current_subtask,
                                    message="策略已加载，正在读取当前观测。")
        return policy

    def _review(self, policy, camera, generation, global_goal, current_subtask, last_ack):
        # tick 在观察期间暂停出队。每路图像都来自上次确认发送之后，此时也有新的关节读回。
        frames, _ = camera.read(newer_than=last_ack)
        mode = self.options["subtask_mode"]
        free_text = mode == "trained" and getattr(policy, "open_subtasks", False)
        boxes = None
        if mode == "external":
            from .scene import select_subtask
            selected = select_subtask(global_goal, current_subtask, frames, self.options, policy.subtasks)
        elif free_text:
            # _next_work 已等到最后一帧执行后的关节读回；与本次相机帧配对。
            with self._condition:
                if not self._current(generation):
                    return
                observation = dict(self._state)
            if getattr(policy, "predicts_boxes", False):
                selected, boxes = policy.select_subtask(frames, global_goal, observation, return_boxes=True)
            else:
                selected = policy.select_subtask(frames, global_goal, observation)
        else:
            selected = policy.select_subtask(frames, global_goal)
        if selected is not None and not free_text and selected not in policy.subtasks:
            raise ValueError(f"Subtask has no action training label: {selected!r}")
        with self._condition:
            if not self._current(generation):
                return
            if selected != current_subtask or selected is None:
                # 切换/结束同时作废预测版本、旧执行帧与对应 RTC 前缀；预算按总任务累计。
                self._generation += 1
                self._queue.clear()
                self._status["metrics"]["queued_actions"] = 0
            self._status["current_subtask"] = selected
            if boxes is not None:
                self._status["scene"]["subtask_boxes"] = boxes
            else:
                self._status["scene"].pop("subtask_boxes", None)
            self._next_review = self._acknowledged + self.options["replan_steps"]
            if selected is None:
                self._status.update(status="completed", termination_reason="goal_complete", task_success=True,
                                    verification={"assessment": f"{mode}_subtask_prediction", "success": True},
                                    message="高层策略根据最新观测判断总目标已完成。")
            elif self._predictions >= self.options["cycles"] and not self._queue:
                self._status.update(status="completed", termination_reason="budget_exhausted",
                                    message="动作块预算已耗尽，高层仍预测有待执行的子任务。")
            self._condition.notify_all()

    def _finish_direct(self, camera, generation, global_goal, last_ack):
        verification = {"success": None, "evidence": "尚未验证抓取结果。"}
        if self.options["verify_scene"]:
            from .scene import inspect_completion
            frames, _ = camera.read(newer_than=last_ack)
            with self._condition:
                if not self._current(generation):
                    return
                before = self._status["scene"]
            verification = inspect_completion(global_goal, before, frames, self.options)
        with self._condition:
            if self._current(generation):
                detail = verification.get("fact") or verification.get("uncertainty") or "尚未验证取物结果。"
                self._status.update(status="completed", termination_reason="budget_exhausted",
                                    task_success=verification.get("success"), verification=verification,
                                    message="策略执行周期结束。" + detail)

    def _predict(self, policy, camera, generation, current_subtask, last_ack):
        began = time.perf_counter()
        frames, ages = camera.read(newer_than=last_ack) if self._full_chunks else camera.read()
        with self._condition:
            if not self._current(generation) or self._review_due():
                return
            observed = dict(self._state)
            sent_before = self._sent
            previous = [raw for raw, _ in self._queue]
        expected_delay = math.ceil(self._latency * self.options["fps"])
        original, actions = policy.predict(frames, observed, current_subtask, previous, expected_delay)
        elapsed = time.perf_counter() - began
        with self._condition:
            if not self._current(generation):
                return
            consumed = self._sent - sent_before
            self._queue = deque(zip(original[consumed:], actions[consumed:]))
            self._latency = elapsed
            if self._predictions:
                self._warm_latencies.append(elapsed)
            self._predictions += 1
            self._status.update(message="策略正在根据实时画面执行任务。")
            self._status["metrics"].update(
                chunks=self._predictions, inference_s=elapsed, inference_delay_frames=consumed,
                queued_actions=len(self._queue), camera_age_s=ages, model_horizon=policy.horizon,
                horizon_seconds=policy.horizon / self.options["fps"])
            self._condition.notify_all()

    def _run(self):
        # 语义和动作共用一个推理线程；两者串行使用同一份骨干，串口始终由 owner 管理。
        policy = camera = None
        try:
            while True:
                with self._condition:
                    if self._closed:
                        break
                    if self._status["status"] in ("loading", "running"):
                        work = self._next_work()
                    else:
                        work = "close_camera" if camera is not None else None
                    if work is None:
                        self._condition.wait()
                        continue
                    generation = self._generation
                    global_goal = self._status.get("global_goal")
                    current_subtask = self._status.get("current_subtask")
                    last_ack = self._last_ack
                if work == "close_camera":
                    camera.close()
                    camera = None
                    continue
                try:
                    if work == "prepare":
                        if camera is None:
                            metadata = json.loads((Path(self.options["adapter_dir"]) / "adapter_config.json")
                                                  .read_text(encoding="utf-8-sig"))
                            keys = metadata["image_keys"] if "image_keys" in metadata else [metadata["image_key"]]
                            if self.options["subtask_mode"] == "trained" and metadata.get("format") == JOINT_FORMAT:
                                keys = dict.fromkeys([*keys, *metadata.get("semantic_image_keys", keys)])
                            mappings = {key: self.options["cameras"][key] for key in keys}
                            camera = self._camera_factory(mappings, self.options["camera_backend"],
                                                          self.options["camera_timeout"])
                        policy = self._prepare(policy, camera, generation, global_goal)
                    elif work == "review":
                        self._review(policy, camera, generation, global_goal, current_subtask, last_ack)
                    elif work == "predict":
                        self._predict(policy, camera, generation, current_subtask, last_ack)
                    else:
                        self._finish_direct(camera, generation, global_goal, last_ack)
                except Exception as error:
                    # 将模型/相机故障发给 ArmService 的错误与持姿路径；过期错误不得结束新任务。
                    with self._condition:
                        if self._current(generation):
                            self._queue.clear()
                            self._status["metrics"]["queued_actions"] = 0
                            self._status.update(status="failed", termination_reason="failed",
                                                message=f"{type(error).__name__}: {error}")
                    if camera is not None:
                        camera.close()
                        camera = None
        finally:
            if camera is not None:
                camera.close()
