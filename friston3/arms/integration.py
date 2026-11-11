"""NPC/CLI 与机械臂服务的集成层：显式任务、状态查询和可选对话手势。

ArmsControlSkill 和 arms.__main__ 调用 perform；main.py 在回复播放时使用
reply_start 返回的回调。显式 task 可按需启动独立服务，自动手势只连接已有
服务。本模块管理调用生命周期，不持有串口，也不导入模型或机器人 SDK。
"""
# 类型标注只说明输入/输出约定，不会让字符串自动变成机械臂对象。
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

# .config 是同包文件；GESTURES 是允许的手势名集合，load_config 读取配置字典。
from .config import GESTURES, load_config
from friston3.config import PROJECT_ROOT
from friston3.runtime_settings import SETTINGS


_STOP_COMMANDS = frozenset((
    "停止", "停下", "急停", "紧急停止", "停止机械臂", "机械臂停止",
    "停止双臂", "双臂停止", "停下机械臂", "/arms stop",
))


def is_immediate_stop(instruction: str) -> bool:
    """An explicit stop protocol, never a keyword-based gesture classifier."""
    # main 的会话循环调用：输入原始文字，返回是否属于明确的停止命令。
    # 点号串联字符串方法：去两端空白 -> 去末尾标点 -> 转小写 -> 检查集合成员。
    # and 会短路：instruction 不是 str 时，后面的字符串方法不会执行。
    return (isinstance(instruction, str)
            and instruction.strip().rstrip("。！？!?.,，；; ").lower() in _STOP_COMMANDS)


def _job_text(job) -> str:
    # describe_status 的格式化辅助函数；job 为任务字典或 None，返回供人阅读的文字。
    if not isinstance(job, dict):
        return "无"
    states = {"accepted": "已接受，尚未确认完成", "loading": "加载策略与场景中", "running": "执行中",
              "completed": "已完成", "cancelled": "已取消", "failed": "失败"}
    # get(键, 默认值) 允许结果缺字段；第二次 get 将英文状态映射为中文，未知状态保留原文。
    state = str(job.get("status", "unknown"))
    summary = f"{job.get('name') or job.get('kind') or '任务'}：{states.get(state, state)}"
    if job.get("id"):
        summary += f"（{job['id']}）"
    if job.get("message"):
        summary += f"；{job['message']}"
    return summary


def describe_status(result: dict) -> str:
    """Retain the structured snapshot while giving the user a useful summary."""
    # perform("status") 调用：读取服务返回的 dict，生成一句摘要；不修改服务或机械臂。
    if result.get("success") is False:
        return f"机械臂状态={result.get('status', 'unknown')}；{result.get('message', '状态未取得。')}"
    mode = "模拟服务（无真实机械臂）" if result.get("mode") == "mock" else "实机服务"
    sides = {"left": "左臂", "right": "右臂"}
    # for ... 是生成器表达式，逐个将 left/right 翻译；join 用顿号连接这些字符串。
    available = "、".join(sides.get(str(arm), str(arm))
                         for arm in result.get("available_arms", [])) or "无"
    status = str(result.get("status", "unknown"))
    states = {"starting": "正在连接", "loading": "加载策略与场景中", "stopping": "正在确认停止持姿", "idle": "就绪空闲",
              "running": "执行中", "accepted": "任务已接受", "ready": "就绪", "faulted": "故障"}
    summary = (f"{mode}；服务状态={states.get(status, status)}；配置={available}；"
            f"示教动作{'已使能' if result.get('armed') else '未使能'}；"
            f"对话联动={'开启' if result.get('auto_enabled') else '关闭'}；"
            f"{'忙碌' if result.get('busy') else '空闲'}。"
            f"当前任务：{_job_text(result.get('current_job'))}。"
            f"上一任务：{_job_text(result.get('last_job'))}。")
    hold = result.get("hold_result")
    if isinstance(hold, dict) and hold.get("message"):
        summary += f"停止持姿：{hold['message']}"
    return summary


def describe_motions(result: dict) -> str:
    # perform("list") 调用：输入动作列表响应，返回可播报的说明；无网络和串口操作。
    if result.get("success") is False:
        return str(result.get("message", "动作列表未取得。"))
    items = result.get("motions", [])
    invalid = result.get("invalid_motions", [])
    errors = "；".join(f"{item.get('name', '?')}：{item.get('error', '无效')}"
                      for item in invalid if isinstance(item, dict))
    if not items:
        return ("没有可用的机械臂动作；无效记录：" + errors if errors else
                "尚未录制机械臂动作。请先在 lerobot 服务控制台示教。")
    # labels 是本函数新建的 list；append 在下面逐个加入每条动作说明。
    labels = []
    for motion in items:
        if not isinstance(motion, dict):
            continue
        sides = "+".join(str(side) for side in motion.get("arms", []))
        mode = "可用于对话联动" if motion.get("looped") else "仅手动播放"
        labels.append(f"{motion.get('name', '?')}（{sides}，{motion.get('duration_s', '?')}秒，{mode}）")
    prefix = "模拟动作（无真实机械臂）：" if result.get("mode") == "mock" else "已录制动作："
    return prefix + "；".join(labels) + ("；无效记录：" + errors if errors else "")


class ArmIntegration:
    """Explicit operations are bounded; optional automatic gestures never wait.

    Only one automatic request may be in flight. There is no cross-turn motion
    queue, retry, or automatic arming. Explicit policy tasks can start their
    configured background service; optional gestures never do. A fresh NPC result
    contributes at most one callback, consumed at the first audio clip (or at
    the text-only fallback).
    """

    def __init__(self, config_path=None, *, client_factory=None, notify=None):
        # main.py 与机械臂 CLI 创建 ArmIntegration 时调用；self 是这个集成对象。
        # config_path 是配置路径；client_factory/notify 是可选函数，不是函数执行结果。
        # * 后的参数须按名传入，便于阅读调用意图，例如 notify=print。
        self.config_path = config_path
        # 保存工厂函数可替换客户端实现；正常运行未传入时，_client() 使用真实 HTTP 客户端。
        self._client_factory = client_factory
        # lambda 创建匿名函数；message 是未来调用它时传入的文字。这里不会立即 print。
        self._notify = notify or (lambda message: print(f"[Arms] {message}", flush=True))
        # Lock 是线程锁；多个线程读写下面这些属性时，with self._lock 让它们逐个进入。
        self._lock = threading.Lock()
        # generation 是递增的“轮次编号”；旧回调保存的编号可与当前编号比较，识别过期回复。
        self._generation = 0
        # inflight 表示自动手势的 HTTP 请求线程尚未结束，不等于整个机械臂任务已经完成。
        self._inflight = False
        self._closed = False
        # retry_after 保存单调时钟时间；worker 是线程对象，尚未创建时先放 None。
        self._retry_after = 0.0
        self._worker = None
        # 保存任务编号是为了退出/关闭联动时能精确请求停止本对象提交的任务。
        self._last_job_id = None
        self._last_dialogue_job_id = None
        # set() 创建空集合；记录已发出清理请求的编号，避免本对象对同一任务重复清理。
        self._cleanup_ids = set()
        # 启动锁专门串行化“检查服务→必要时创建进程”，与短时间保护状态的 _lock 分开。
        self._startup_lock = threading.Lock()
        # 这里先占位；Popen 创建的进程句柄和供排查的启动命令，稍后由 _ensure_task_service 填入。
        self._service_process = None
        self._service_command = None

    def _client(self):
        # _request 每次调用本函数取得一个客户端；返回对象，尚未发送 HTTP 请求。
        if self._client_factory is not None:
            # self._client_factory(...) 是调用已保存的函数，参数是本对象的配置路径。
            return self._client_factory(self.config_path)
        from .client import ArmClient
        return ArmClient(self.config_path)

    def _request(self, operation: str, **arguments) -> dict:
        # NPC 技能和 CLI task 的共同出口；集成层管理服务生命周期，客户端只管通信。
        # **arguments 将 name=... 等额外关键字收集为 dict；返回服务结果或整理后的错误 dict。
        try:
            # client 是 _client() 刚返回的 ArmClient/替代实现对象；.status/.command 是其方法。
            client = self._client()
            if operation == "task":
                self._ensure_task_service(client)
            # A if 条件 else B 只执行一个分支；调用处 **arguments 又把字典展开为关键字参数。
            result = client.status() if operation == "status" else client.command(operation, **arguments)
            if not isinstance(result, dict):
                raise RuntimeError("机械臂服务未返回对象")
            # getattr(obj, 属性名, 默认值) 兼容没有 mode 属性的替代客户端。
            if "mode" not in result and getattr(client, "mode", None) in ("real", "mock"):
                # {**result, ...} 复制原字典的键值，再增加 mode；没有修改 result 原来指向的字典。
                result = {**result, "mode": client.mode}
            # dict(result) 是浅复制：外层字典独立，内部若有嵌套对象仍可能是同一个对象。
            return dict(result)
        except FileNotFoundError:
            return {"success": False, "status": "unconfigured",
                    "message": "机械臂服务尚未配置；请先在 lerobot 环境运行 python -m friston3.arms init。"}
        except (OSError, ValueError, RuntimeError) as error:
            return {"success": False, "status": "unavailable",
                    "message": f"机械臂服务未确认响应：{error}"}

    @staticmethod
    def _service_python(config):
        # staticmethod 表示函数不自动接收 self；启动函数传入配置 dict，得到 Python 可执行文件 Path。
        # 目的：NPC 所在环境未必装了 LeRobot，后台服务要使用装有机械臂依赖的解释器。
        # os.environ 是环境变量映射；or 先取非空的环境变量，否则读取配置中的 python。
        explicit = os.environ.get("FRISTON_LEROBOT_PYTHON", "").strip() or config.get("python")
        if explicit:
            if not isinstance(explicit, str):
                raise ValueError("机械臂 python 配置必须是解释器路径")
            # Path 封装路径；expanduser 展开 ~，此时仍未启动文件或解释器。
            candidate = Path(explicit).expanduser()
            if not candidate.is_absolute():
                # Path 的 / 运算符拼接路径；相对路径以配置文件所在目录为基准。
                candidate = Path(config["config_path"]).parent / candidate
            # resolve 规范化成绝对路径；is_file 检查是否实际存在，return 返回 Path 对象。
            candidate = candidate.resolve()
            if not candidate.is_file():
                raise RuntimeError(f"配置的 LeRobot Python 不存在：{candidate}")
            return candidate
        # 没有显式配置时收集候选：sys.executable 是当前 Python 路径，sys.prefix 是环境根目录。
        executable = Path(sys.executable).resolve()
        prefixes = [Path(sys.prefix), Path(sys.base_prefix), executable.parent,
                    Path.home() / "anaconda3", Path.home() / "miniconda3"]
        for key in ("CONDA_PREFIX", "_CONDA_ROOT"):
            # 这些变量有时存在、有时不存在；append 把找到的目录加入候选根列表。
            if os.environ.get(key):
                prefixes.append(Path(os.environ[key]))
        for key in ("CONDA_EXE", "CONDA_PYTHON_EXE"):
            if os.environ.get(key):
                prefix = Path(os.environ[key]).parent
                prefixes.append(prefix.parent if prefix.name.lower() in ("scripts", "bin", "condabin") else prefix)
        # candidates 存最终 python.exe/bin/python 路径；prefixes 只存用于拼接的环境根目录。
        candidates = []
        if Path(sys.prefix).name.lower() == "lerobot":
            candidates.append(executable)
        # Windows 与类 Unix 环境的解释器相对位置不同，用条件表达式选对应形式。
        binary = Path("python.exe") if sys.platform == "win32" else Path("bin/python")
        for prefix in prefixes:
            if prefix.parent.name.lower() == "envs":
                candidates.append(prefix.parent / "lerobot" / binary)
            candidates.append(prefix / "envs" / "lerobot" / binary)
        try:
            # find_spec 检查当前解释器能否找到 lerobot 包，不在这里加载模型或创建机器人。
            if importlib.util.find_spec("lerobot") is not None:
                candidates.append(executable)
        except (ImportError, ValueError):
            pass
        # seen 记录已检查的规范化路径文字，避免多种环境线索得到同一路径时重复处理。
        seen = set()
        for candidate in candidates:
            candidate = candidate.resolve()
            key = os.path.normcase(str(candidate))
            if key not in seen and candidate.is_file():
                return candidate
            seen.add(key)
        raise RuntimeError("没有找到已安装的 LeRobot Python；请在机械臂配置中填写 python，"
                           "或设置 FRISTON_LEROBOT_PYTHON 指向现有环境解释器。")

    @staticmethod
    def _service_absent(error):
        # _ensure_task_service 调用；输入异常对象，沿 __cause__ 检查是否明确“连接被拒绝”。
        # 客户端用 raise ... from error 包装过异常，因此真正的网络原因可能在下一层。
        cause = error
        while cause is not None:
            if isinstance(cause, ConnectionRefusedError) or (
                    isinstance(cause, OSError) and
                    (cause.errno in (61, 111, 10061) or getattr(cause, "winerror", None) == 10061)):
                return True
            cause = cause.__cause__
        return False

    def _ensure_task_service(self, client):
        """Health-check before POST; a submitted task is never retried."""
        # 先确认现有服务，再决定是否启动独立 Python 进程。启动锁防止并发请求
        # 各自拉起一个服务；只有 task 走这里，自动手势不会隐式占用机械臂。
        # 输入现成客户端；就绪时直接 return（等价返回 None），失败则抛异常给 _request。
        # with 进入时获取锁、离开时释放锁；即使 raise/return 也会释放。
        with self._startup_lock:
            with self._lock:
                if self._closed:
                    raise RuntimeError("机械臂对话入口已关闭")
            # snapshot 用于保存 status() 返回的状态字典；None 表示尚未取得有效状态。
            snapshot = None
            try:
                snapshot = client.status()
            except (OSError, RuntimeError) as error:
                # 单纯 HTTP 超时不表示服务不存在；若端口仍监听，raise 重抛原异常。
                if not self._service_absent(error) and client.is_listening():
                    raise
            if snapshot is not None:
                if snapshot.get("success") is not True:
                    raise RuntimeError(str(snapshot.get("message", "现有机械臂服务故障")))
                if snapshot.get("ready") is True:
                    # 已有服务 ready 时立即结束；不会再创建另一套服务进程。
                    return
            # 只读取配置和启动等待参数；尚未向服务提交真正的 task 命令。
            config = load_config(self.config_path)
            timeout = config.get("startup_timeout_s", 30.0)
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("startup_timeout_s 必须是有限正数")
            # process 是 Popen 进程句柄或 None。poll() 为 None 表示该子进程尚未退出。
            process = self._service_process if snapshot is None else None
            if snapshot is None and (process is None or process.poll() is not None):
                from friston3.runtime_bootstrap import interpreter_environment
                # python 是前面解析出的 Path；argv 是按参数分项的 list，不是拼好的 shell 脚本。
                python = self._service_python(config)
                # -B 不写 .pyc；-m 按模块运行 __main__.py；最后的 serve 选中 CLI 服务分支。
                argv = [str(python), "-B", "-m", "friston3.arms", "--config",
                        str(Path(config["config_path"]).resolve()), "serve"]
                # 返回适合该解释器的环境变量 dict，交给 Popen 的 env 参数供子进程使用。
                environment = interpreter_environment(python)
                environment["PYTHONDONTWRITEBYTECODE"] = "1"
                # Popen 启动独立进程并立即返回句柄，不等服务加载完成；ready 的等待在下方。
                # cwd 决定子进程工作目录；输入接空设备、输出和错误输出被丢弃；Windows 不弹新控制台。
                self._service_process = subprocess.Popen(
                    argv, cwd=PROJECT_ROOT, env=environment,
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
                process = self._service_process
                # 将参数列表转换为可显示的命令行，仅用于错误提示，不在这里再次执行。
                self._service_command = subprocess.list2cmdline(argv)
                self._notify("正在后台启动已配置的 LeRobot 机械臂服务。")
            # monotonic 是不会因系统校时倒退的计时器；deadline 为本次等待结束的时刻。
            deadline = time.monotonic() + timeout
            last_error = "服务尚未就绪"
            # 每轮只查询状态；不能把“进程已经创建”当作“模型与硬件都已就绪”。
            while time.monotonic() < deadline:
                with self._lock:
                    if self._closed:
                        raise RuntimeError("机械臂对话入口已关闭；后台服务保持独立运行。")
                try:
                    snapshot = client.status()
                except (OSError, RuntimeError) as error:
                    # 启动中暂时连不上就保留最近错误，继续下一轮；没有重发机器人动作。
                    last_error = str(error)
                else:
                    # try/except/else 中的 else 只在 try 没抛异常时执行。
                    if snapshot.get("success") is not True:
                        raise RuntimeError(str(snapshot.get("message", "机械臂服务启动失败")))
                    if snapshot.get("ready") is True:
                        return
                    last_error = str(snapshot.get("message", "机械臂服务正在连接硬件"))
                if process is not None:
                    # poll 非 None 是已经退出的整数退出码，此时继续等 ready 没有意义。
                    code = process.poll()
                    if code is not None:
                        raise RuntimeError(f"后台机械臂服务已退出，退出码 {code}。"
                                           f"请运行 {self._service_command} 查看启动错误。"
                                           f"最近状态：{last_error}")
                # 暂停 0.1 秒让出 CPU，避免高速空循环；这不是机械臂的控制周期。
                time.sleep(0.1)
            raise RuntimeError(f"等待机械臂服务就绪超过 {timeout:g} 秒：{last_error}。"
                               "任务尚未提交；可查询机械臂状态后再次发出任务。")

    def cancel_pending(self) -> None:
        # perform 的 stop/task 分支与 cancel_dialogue 调用；令已保存的旧回调编号过期。
        # 它只改本对象状态，不发 HTTP，也不等于停止已经运行的机械臂任务。
        with self._lock:
            self._generation += 1

    def perform(self, operation: str, *, name: str = "", instruction: str = "", arm=None) -> dict:
        """Submit explicit tasks while keeping automatic gestures independent."""
        # ArmsControlSkill.execute 与 CLI task 的入口：输入操作和按名参数，返回完整结果字典。
        # name 是录制动作名，instruction 是自然语言任务，arm 为 left/right 或 None。
        if operation not in ("status", "list", "stop", "play", "task"):
            raise ValueError("对话支持 status、list、stop、play 和 task")
        if operation == "stop":
            self.cancel_pending()
        # arguments 是将传给客户端的参数 dict；查询操作无需 name，因而使用空字典 {}。
        arguments = {"name": name, "source": "manual"} if operation == "play" else {}
        if operation == "task":
            self.cancel_pending()
            # arm 没指定时展开空字典，响应协议里就没有 arm 字段，交给服务按配置选择。
            arguments = {"instruction": instruction, **({"arm": arm} if arm is not None else {})}
        # **arguments 把字典还原为命名参数；_request 内部再调用 ArmClient 发 HTTP。
        result = self._request(operation, **arguments)
        # result[...] = ... 为结果增加调用来源信息；后续展示仍可读取原来的 job_id/status。
        result["operation"] = operation
        if operation in ("play", "task"):
            # 记录已接受的任务编号，供 close 使用；记录编号不表示任务已经完成。
            self._remember_job(result)
            if operation == "task" and result.get("success") is True and result.get("status") in ("accepted", "running"):
                from friston3.llm import reserve_policy_memory
                # 通知语言模型层为机器人策略协调内存；该函数不是 π₀.₅ 的 forward。
                reserve_policy_memory(self.config_path)
        elif operation == "stop" and result.get("success") is True:
            with self._lock:
                self._last_job_id = None
                self._last_dialogue_job_id = None
        if operation in ("status", "list"):
            # 人类可读的 message 单独组织；原服务说明先放到 service_message 保留。
            if "message" in result:
                result["service_message"] = result["message"]
            result["message"] = describe_status(result) if operation == "status" else describe_motions(result)
        elif operation in ("play", "task") and result.get("status") in ("accepted", "running"):
            result["service_message"] = result.get("message", "")
            result["message"] = (("已接收取放任务，将根据当前画面和策略执行；成功取物尚未确认。"
                                  if operation == "task" else f"已接受动作 {name}，实际执行结果尚未确认；")
                                 + "可查询机械臂状态查看进度。")
            if result.get("mode") == "mock":
                result["message"] = "模拟服务（无真实机械臂）：" + result["message"]
        # Query and acceptance are not physical completion. Preserve status,
        # current_job, last_job, positions and every other service field.
        result["speak_confirmation"] = True
        # 上层 NPC/CLI 拿到同一组结构化字段和可读摘要，决定如何展示/播报。
        return result

    def _remember_job(self, result: dict, *, dialogue=False) -> None:
        # perform 与 _automatic 调用；只从成功接收结果中保存 job_id，不进行任何网络请求。
        job_id = result.get("job_id")
        if (result.get("success") is True and result.get("status") in ("accepted", "running")
                and isinstance(job_id, str) and job_id):
            with self._lock:
                self._last_job_id = job_id
                if dialogue:
                    self._last_dialogue_job_id = job_id

    def _stop_owned(self, job_id: str):
        # close/cancel_dialogue/_automatic 的清理出口；输入编号，只请求停止该编号对应的任务。
        with self._lock:
            if job_id in self._cleanup_ids:
                return None
            self._cleanup_ids.add(job_id)
        # 网络请求放在锁外：其他线程不必在一次 HTTP 等待期间一直等这把状态锁。
        return self._request("stop", job_id=job_id)

    def cancel_dialogue(self):
        """An explicit performance-off choice cancels only our automatic job."""
        # 用户关闭表现/联动时由 main 调用；先废弃未执行回调，再停止本对象记住的自动任务。
        self.cancel_pending()
        with self._lock:
            job_id = self._last_dialogue_job_id
        return self._stop_owned(job_id) if job_id else None

    def reply_start(self, result: dict, *, allow: bool = True):
        """Return a once-only callback for this successful, fresh NPC reply."""
        # main._run_session 把返回的函数交给语音 on_start。闭包保存本轮 generation，
        # 让“开始说话”与“开始手势”对齐；下一轮已开始时，旧回调就失效。
        # result 来自本轮 Agent 输出；返回值可能是 None，也可能是稍后才执行的函数对象。
        with self._lock:
            self._generation += 1
            # generation 是本轮局部整数；self._generation 是对象属性，后续轮次仍会改变。
            generation = self._generation
        if (not allow or not SETTINGS.get("gestures_enabled", True) or not isinstance(result, dict)
                or result.get("executed_skill") != "npc_talk"
                or result.get("success") is not True):
            return None
        # presentation 是回复表现字段 dict；arm_gesture 是规划器选择的已注册手势名字。
        presentation = result.get("presentation")
        name = presentation.get("arm_gesture") if isinstance(presentation, dict) else None
        if not isinstance(name, str) or name not in GESTURES:
            return None
        # 下面的嵌套函数会记住 called/generation/name/created_at；这组捕获环境叫闭包。
        # reply_start 返回后，变量仍由返回的函数引用，因此不会立即消失。
        called = False
        created_at = time.monotonic()

        def start_once():
            # nonlocal 修改外层局部变量；即使语音和文字兜底都调用，也只执行一次。
            # 若没有 nonlocal，赋值 called=True 会变成本函数的新局部变量，无法更新外层状态。
            nonlocal called
            with self._lock:
                if called:
                    # 同一回调第二次被调用时直接返回；return 没写值时结果是 None。
                    return
                called = True
                # 逐项检查本对象是否关闭、轮次是否过期、是否已有请求、联动是否开启、
                # 上次失败后的等待是否结束，以及这个回调是否已经等待了太久。
                if (self._closed or generation != self._generation or self._inflight
                        or not SETTINGS.get("gestures_enabled", True)
                        or time.monotonic() < self._retry_after
                        or time.monotonic() - created_at > 15.0):
                    return
                # 先标记请求正在处理，再启动线程，避免下一次回调在间隙再创建一个请求。
                self._inflight = True
                # target 传方法对象，不写 ()；args 的 tuple 在新线程中变成 _automatic(name, generation)。
                # daemon=True 表示这个辅助线程本身不会阻止解释器退出，不表示其工作已经成功。
                self._worker = threading.Thread(target=self._automatic, args=(name, generation),
                                                name="friston-arm-gesture", daemon=True)
                # .start() 真正启动后台执行；不会等待 HTTP 返回，因此语音播放可继续。
                self._worker.start()

        # 返回函数本身，不能写 start_once()；后者会立即执行，并返回 None 给语音层。
        return start_once

    def _automatic(self, name: str, generation: int) -> None:
        # 仅由 start_once 创建的线程调用；提交一次对话手势、保存编号，关闭联动时清理已接受的任务。
        # name/generation 来自 Thread(args=...)，不是这里重新让 LLM 生成的动作。
        try:
            with self._lock:
                if (self._closed or generation != self._generation
                        or not SETTINGS.get("gestures_enabled", True)):
                    return
            # source 区分自动手势与人工 play；_request 对 play 不会自动启动服务。
            result = self._request("play", name=name, source="dialogue")
            self._remember_job(result, dialogue=True)
            with self._lock:
                closed = self._closed
                stale = generation != self._generation
            if ((closed or not SETTINGS.get("gestures_enabled", True)) and result.get("job_id")
                    and result.get("status") in ("accepted", "running")):
                stopped = self._stop_owned(result["job_id"])
                if stopped is not None:
                    self._notify(str(stopped.get("message", "本轮动作停止未确认。")))
                return
            if stale:
                # 轮次过期时不再提示旧结果；这一分支本身不会请求停止已经接受的动作。
                return
            if result.get("status") in ("unconfigured", "unavailable"):
                with self._lock:
                    self._retry_after = time.monotonic() + 15.0
            if result.get("status") == "unconfigured":
                return
            if result.get("status") in ("accepted", "running"):
                prefix = "模拟服务：" if result.get("mode") == "mock" else ""
                self._notify(f"{prefix}对话动作 {name} 已接受，完成状态以服务查询为准。")
            elif result.get("message"):
                self._notify(str(result["message"]))
        finally:
            # finally 保证正常结束、提前 return 或抛异常都清除“请求处理中”的标记。
            with self._lock:
                self._inflight = False

    def close(self) -> None:
        """Cancel this integration's job without stopping later CLI work."""
        # main 会话退出时调用：废弃待执行回调，短暂等待请求线程，按记录的编号清理本次任务。
        # 此处不终止 self._service_process；独立服务的生命周期与 NPC 进程不同。
        with self._lock:
            if self._closed:
                return
            # 先在锁内标记关闭，后台线程随后会看到；generation 增加让旧回调失效。
            self._closed = True
            self._generation += 1
            worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            # is/not is 比较是否为同一个对象；current_thread() 返回当前正在执行 close 的线程。
            # join 等待线程结束；timeout=1.0 限制本次等待，不会强行杀线程，也不能等待自己。
            worker.join(timeout=1.0)
        with self._lock:
            # join 后重新读属性，因为请求线程可能刚刚取得 job_id；复制到局部变量后释放锁。
            job_id = self._last_job_id
            inflight = self._inflight
        if job_id:
            # 仅按本对象记住的编号停止，不向独立服务发送“退出进程”的 shutdown。
            result = self._stop_owned(job_id)
            if result is not None:
                self._notify(str(result.get("message", "本轮动作停止未确认。")))
        if inflight:
            self._notify("退出时仍有机械臂请求等待确认；如需立即全局停止，请在独立终端运行 python -m friston3.arms stop。")
