"""机械臂命令行入口：python -m friston3.arms 会执行本文件。

serve 创建独立服务；task 经 ArmIntegration 按需启动该服务；其余命令用
ArmClient 发 HTTP 请求。本文件负责解析参数和展示结果，不直接读写串口。
先读 main，再按需读 _wait_for_job；全链路说明见同目录 README.md。
"""
# 本文件作为包的 __main__.py，运行 python -m friston3.arms 时会成为程序入口。
# 类型标注用于说明参数和返回值，不会替代下面显式执行的格式检查。
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

# .client 是当前包里的 client.py；ArmClient 是类，调用 ArmClient(...) 才创建对象。
from .client import ArmClient
from .config import create_config, default_config_path


# compile 把正则文字预先转成可复用的匹配对象；r 前缀保留反斜杠给正则解释。
_NAME = re.compile(r"[a-z][a-z0-9_-]{0,47}\Z")
# frozenset 是不能增删的集合；_wait_for_job 用 in 判断任务是否进入结束状态。
_TERMINAL_STATES = frozenset(("completed", "cancelled", "failed"))


def _motion_name(value: str) -> str:
    # argparse 在解析 name 参数时调用；value 是命令行传来的 str，返回通过检查的原字符串。
    # value: str 与 -> str 分别是输入/返回类型说明；fullmatch 要求整个字符串匹配。
    if not _NAME.fullmatch(value):
        raise argparse.ArgumentTypeError("动作名应为小写字母开头，随后为小写字母、数字、_ 或 -，最多 48 字。")
    return value


def _record_seconds(value: str) -> float:
    # record --seconds 的转换函数：把终端中的数字文字转为 float 秒数，供 main 提交。
    try:
        seconds = float(value)
    except ValueError as error:
        # float("abc") 会抛 ValueError；转成 ArgumentTypeError 后 argparse 能显示参数错误。
        # from error 保留原始异常原因，不会再次执行输入内容。
        raise argparse.ArgumentTypeError("录制秒数必须是数字。") from error
    if not math.isfinite(seconds) or not 1.0 <= seconds <= 30.0:
        # isfinite 排除 inf/nan；链式比较 1.0 <= x <= 30.0 等价于两个比较用 and 连接。
        raise argparse.ArgumentTypeError("录制时长应在 1–30 秒之间。")
    return seconds


def _wait_seconds(value: str) -> float:
    # play/record --wait-timeout 共用；输入字符串，返回秒数或抛 argparse 的参数异常。
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("等待秒数必须是数字。") from error
    if not math.isfinite(seconds) or not 1.0 <= seconds <= 120.0:
        raise argparse.ArgumentTypeError("等待时长应在 1–120 秒之间。")
    return seconds


def _task_wait_seconds(value: str) -> float:
    # task/shutdown 的等待参数使用它；这里只解析现有协议的正数要求，不提交任务。
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("等待秒数必须是数字。") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("等待秒数必须是有限正数。")
    return seconds


def _shared_options(parser: argparse.ArgumentParser) -> None:
    # _arguments 多次调用：把同一组 --config/--json 选项加到传进来的解析器对象上。
    # parser 是函数参数，不是类名；parser.add_argument(...) 是调用这个对象的方法。
    # 本函数直接修改解析器，返回 None，所以调用方不需要 parser = _shared_options(parser)。
    # SUPPRESS lets both `--config X status` and `status --config X` work,
    # without a subparser default discarding a global option already supplied.
    # SUPPRESS 的意思是“未填写时不写入这个属性”，避免子命令默认值覆盖前面已给的全局值。
    parser.add_argument("--config", default=argparse.SUPPRESS,
                        help="配置文件路径；默认使用 FRISTON_ARMS_CONFIG 或用户配置目录")
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="以 JSON 输出结果")


def _arguments(argv=None):
    # main 调用：输入可选字符串列表 argv，返回 argparse.Namespace（解析结果对象）。
    # argv=None 时 parse_args 读取真实终端参数；传列表则解析该列表，适合离线检查解析行为。
    # ArgumentParser(...) 创建总解析器；description=... 是关键字参数，不是字典赋值。
    parser = argparse.ArgumentParser(description="Friston-3 双臂动作与示教控制台")
    # 先规定默认结果：没给路径时 args.config=None，没给 --json 时 args.json=False。
    parser.set_defaults(config=None, json=False)
    _shared_options(parser)
    # add_subparsers 让 serve/status/task 等各有一套参数；选中的名字存入 args.operation。
    # required=True 表示必须选一个子命令；不是同时执行所有下方定义的命令。
    commands = parser.add_subparsers(dest="operation", required=True)

    # init 是“init 子命令的解析器对象”，不是运行 init 的结果；此处只登记界面。
    init = commands.add_parser("init", help="创建配置；保留已有配置，不连接硬件")
    _shared_options(init)
    for side in ("right", "left", "leader"):
        # 循环拼出 --right-port 等六个选项；argparse 会把连字符变成属性名中的下划线。
        # 例如 --right-port COM9 解析后可从 args.right_port 读取字符串 "COM9"。
        init.add_argument("--" + side + "-port", default="", help=side + " 串口")
        init.add_argument("--" + side + "-id", default="", help=side + " 已有标定 ID")
    # action="store_true" 是无值开关：写 --mock 得到 True；没有写则默认 False。
    init.add_argument("--mock", action="store_true", help="创建纯模拟配置，不接任何真实机械臂")

    # 每次循环从一个二元素 tuple 解包出 name/help_text，创建一个同结构的子命令。
    for name, help_text in (
            ("serve", "在 lerobot 环境启动唯一硬件控制服务"),
            ("status", "查看实际服务状态与当前姿态"),
            ("list", "列出已录制动作"),
            ("arm", "本地启用后续动作；不播放轨迹"),
            ("stop", "请求停止并禁用后续动作；不卸力、不回起点"),
            ("shutdown", "停止任务并关闭服务，交还机械臂串口")):
        child = commands.add_parser(name, help=help_text)
        _shared_options(child)
        if name == "serve":
            child.add_argument("--trained-goal-language", choices=("en", "original"),
                               help="总目标语言：en 将中文整句译成英文（默认），original 保留原话")
            child.add_argument("--adapter-dir", help="联合训练生成的检查点目录")
            child.add_argument("--replan-steps", type=int, help="两次语义观察之间确认发送的动作帧数")
            child.add_argument("--rtc", action=argparse.BooleanOptionalAction, default=None,
                               help="动作块之间使用 RTC 引导；联合检查点默认逐块执行")
        if name == "shutdown":
            # --no-wait 对应 args.no_wait；type=函数名 将转换函数交给 argparse 稍后调用。
            child.add_argument("--no-wait", action="store_true", help="只显示关闭请求的接收结果")
            child.add_argument("--wait-timeout", type=_task_wait_seconds, default=10.0)

    auto = commands.add_parser("auto", help="允许或禁止对话触发已录制的闭合动作")
    _shared_options(auto)
    # 不以 -- 开头的是位置参数；auto on 中 on 存到 args.setting。
    # choices 只约束这个参数的可选值，不是让程序同时执行两种选项。
    auto.add_argument("setting", choices=("on", "off"))

    play = commands.add_parser("play", help="回放已录动作，默认等待实际终态")
    _shared_options(play)
    play.add_argument("name", type=_motion_name)
    # 默认会等待终态；写 --no-wait 才跳过等待。这个开关不会改变服务是否执行命令。
    play.add_argument("--no-wait", action="store_true", help="只显示接收结果和任务编号")
    play.add_argument("--wait-timeout", type=_wait_seconds, default=45.0)

    task = commands.add_parser("task", help="根据当前场景运行自然语言 Pi0.5 取放任务")
    _shared_options(task)
    task.add_argument("instruction", help="完整任务，例如：把红色积木放到蓝色盒子里")
    # --arm 是可选参数；不给时 None 表示交由配置选择，不是字符串 "None"。
    task.add_argument("--arm", choices=("left", "right"), default=None,
                      help="单臂配置自动选择；双臂配置请指定执行臂")
    task.add_argument("--no-wait", action="store_true", help="只显示接收结果和任务编号")
    task.add_argument("--wait-timeout", type=_task_wait_seconds, default=None,
                      help="等待结果的秒数；默认持续等待，Ctrl+C 可停止任务")

    record = commands.add_parser("record", help="用主臂示教选定从臂，按实际反馈记录动作")
    _shared_options(record)
    record.add_argument("name", type=_motion_name)
    record.add_argument("--arm", choices=("left", "right"), default="right")
    record.add_argument("--seconds", type=_record_seconds, default=8.0)
    # type=int 先把文字转为整数，再由 choices 检查可选范围；默认 3 由程序直接提供。
    record.add_argument("--countdown", type=int, choices=(0, 1, 2, 3), default=3,
                        help="提交录制前准备倒计时，默认 3 秒")
    record.add_argument("--no-wait", action="store_true")
    record.add_argument("--wait-timeout", type=_wait_seconds, default=45.0)
    # 真正的解析发生在这里；--help 会由 argparse 显示帮助并退出，不进入 main 的硬件分支。
    # 当前选项用子命令和 choices 组织；没有使用 add_mutually_exclusive_group 创建互斥开关组。
    return parser.parse_args(argv)


def _emit(result: dict, *, as_json: bool = False) -> None:
    # main 和轮询函数调用：读取结果 dict 并打印；返回 None，不修改任务、不发请求。
    # * 后的 as_json 只能按名字传，调用形式是 _emit(result, as_json=args.json)。
    if as_json:
        # dumps 将 Python 字典变成 JSON 字符串；flush=True 立即刷新输出，便于终端/上层读取。
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        return
    # get 缺键时返回 None；or 依次取第一个非空值，因此优先显示 message，其次 status。
    print(result.get("message") or result.get("status") or "未返回状态", flush=True)
    mode = result.get("mode")
    if mode == "mock":
        print("[模拟] 此结果仅验证软件流程，不代表实机执行或实机验收。", flush=True)
    elif mode == "real":
        print("[实机] 由独立控制服务连接机械臂。", flush=True)
    if "armed" in result:
        # "armed" in result 检查键是否存在；result["armed"] 取值，不能把两者混成一回事。
        print("动作执行：" + ("已启用" if result["armed"] else "未启用 / 已停止")
              + "；对话动作：" + ("开启" if result.get("auto_enabled") else "关闭"), flush=True)
    arms = result.get("available_arms")
    # HTTP 回包解析后类型仍需核对；isinstance(arms, list) 确认后才能按列表遍历。
    if isinstance(arms, list):
        print("已接入：" + ("、".join(str(arm) for arm in arms) or "无"), flush=True)
    if result.get("job_id"):
        print("任务编号：" + str(result["job_id"]), flush=True)
    finished_job = result.get("job")
    if isinstance(finished_job, dict) and finished_job.get("motion_saved"):
        print("动作已保存：" + str(finished_job["motion_saved"]), flush=True)
    current = result.get("current_job")
    if isinstance(current, dict):
        # 字符串.format(name=...) 将花括号占位符替换成指定值；它和 f-string 都用于格式化。
        print("当前任务：{name} / {status} / {id}".format(
            name=current.get("name", ""), status=current.get("status", ""),
            id=current.get("id", "")), flush=True)
    # 每轮把“文字标签、任务字典”解包到 label/job；同一段展示逻辑可复用于三种任务位置。
    for label, job in (("当前", current), ("上一项", result.get("last_job")),
                       ("任务", finished_job)):
        if isinstance(job, dict) and job.get("kind") == "task":
            print(f"{label}策略任务：{job.get('instruction', '')}", flush=True)
            if job.get("global_goal") and job["global_goal"] != job.get("instruction"):
                print(f"  策略总目标：{job['global_goal']}", flush=True)
            print(f"  {job.get('policy_status', job.get('status', ''))}：{job.get('message', '')}", flush=True)
            if job.get("current_subtask"):
                print(f"  当前子任务：{job['current_subtask']}", flush=True)
            for key in ("scene", "metrics"):
                if job.get(key):
                    print(f"  {key}: " + json.dumps(job[key], ensure_ascii=False, allow_nan=False), flush=True)
    motions = result.get("motions")
    if isinstance(motions, list):
        if not motions:
            print("尚无动作。使用 record 录制你自己的轨迹。", flush=True)
        for motion in motions:
            if not isinstance(motion, dict):
                continue
            print("  {name} | {arms} | {duration}s | {looped}".format(
                name=motion.get("name", ""),
                arms=",".join(str(arm) for arm in motion.get("arms", [])),
                duration=motion.get("duration_s", "?"),
                looped="起终点相近，可检查后用于对话" if motion.get("looped") else "非闭合轨迹"), flush=True)
    positions = result.get("positions")
    if isinstance(positions, dict):
        for arm, values in positions.items():
            # dict.items() 遍历键值对；外层 arm 是臂名，values 是该臂各关节读数字典。
            if isinstance(values, dict):
                # f-string 的 :.2f 将数字显示为两位小数；只改变展示，不改变真实关节读数。
                readings = ", ".join(f"{key}={value:.2f}" for key, value in values.items()
                                     if type(value) in (int, float))
                print(f"  {arm}: {readings}", flush=True)
        if positions:
            print("姿态单位：转动关节为度，gripper.pos 为 0–100。", flush=True)


def _confirm_stop(client: ArmClient, *, timeout: float = 5.0) -> dict:
    """Wait for the serial owner to acknowledge its readback-hold command.

    A POST acknowledgement only means cancellation has been requested. It
    does not prove the worker is alive or that a bus write succeeded.
    """
    # main(stop) 与失败清理函数调用：输入客户端，查询停止持姿回执，返回已确认/未确认字典。
    # monotonic 提供单调计时；deadline 是未来时刻，不能用系统日期变化判断是否超时。
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            # snapshot 是服务当下状态的 dict；这里只查询，不是再次发送 stop。
            snapshot = client.status()
        except (OSError, ValueError, RuntimeError) as error:
            return {"success": False, "status": "unconfirmed", "hold_confirmed": False,
                    "mode": getattr(client, "mode", None),
                    "message": "停止请求后无法取得服务状态，持姿尚未确认。" + str(error)}
        if snapshot.get("success") is not True or snapshot.get("status") == "faulted":
            return {"success": False, "status": "unconfirmed", "hold_confirmed": False,
                    "mode": snapshot.get("mode", getattr(client, "mode", None)),
                    "message": "服务故障，停止持姿尚未确认：" + str(snapshot.get("message", "未知故障")),
                    "hold_result": snapshot.get("hold_result")}
        if snapshot.get("stop_pending") is False:
            hold = snapshot.get("hold_result")
            # confirmed 是多个布尔条件的组合；and 短路保证 hold 不是字典时不再调用 .get。
            confirmed = (isinstance(hold, dict) and hold.get("success") is True
                         and hold.get("hold_confirmed") is True
                         and snapshot.get("armed") is False
                         and snapshot.get("current_job") is None)
            if confirmed:
                # **hold 展开原回执，再补充 mode/armed/auto_enabled；右侧同名键覆盖左侧值。
                return {**hold, "mode": snapshot.get("mode", getattr(client, "mode", None)),
                        "armed": False, "auto_enabled": snapshot.get("auto_enabled", False)}
            detail = hold.get("message") if isinstance(hold, dict) else "没有取得执行器持姿回执。"
            if snapshot.get("armed") or snapshot.get("current_job") is not None:
                detail = "服务已重新使能或仍有任务，不能将当前状态视为已停止。"
            return {"success": False, "status": "unconfirmed", "hold_confirmed": False,
                    "mode": snapshot.get("mode", getattr(client, "mode", None)),
                    "message": "停止持姿未确认：" + str(detail), "hold_result": hold}
        time.sleep(0.1)
    return {"success": False, "status": "unconfirmed", "hold_confirmed": False,
            "mode": getattr(client, "mode", None),
            "message": "停止请求已提交，但等待执行器持姿确认超时；请检查服务终端和机械臂状态。"}


def _stop_after_wait_failure(client: ArmClient) -> str:
    # 等待超时、通信失败或 Ctrl+C 时调用：先请求 stop，再查询回执，返回用于显示的文字。
    try:
        stop = client.command("stop")
        if stop.get("success") is not True:
            return str(stop.get("message", "停止请求未被接受。"))
        confirmed = _confirm_stop(client)
        # confirmed 为上一步返回的 dict；本函数只取 message 文字供调用方拼接输出。
        return str(confirmed["message"])
    except (OSError, ValueError, RuntimeError):
        return "停止请求也未获得确认；请检查服务终端和机械臂当前状态。"


def _wait_for_shutdown(client: ArmClient, timeout: float):
    # main(shutdown) 调用：只轮询监听端口是否关闭，返回确认结果 dict；不重复提交 shutdown。
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            # min 取“单次最多等待时间”和“总剩余时间”中的较小值。
            if not client.is_listening(timeout=min(3.0, deadline - time.monotonic())):
                return {"success": True, "status": "shutdown", "message": "机械臂服务已停止监听。"}
        except (OSError, RuntimeError):
            pass
        time.sleep(0.1)
    return {"success": False, "status": "shutdown_unconfirmed",
            "message": "关闭请求已提交，但服务停止监听尚未确认。"}


def _wait_for_job(client: ArmClient, accepted: dict, *, timeout: float | None, as_json: bool) -> int:
    # POST 的 accepted 只代表登记任务；轮询同一个 job_id 才能知道终态。
    # 分开“接收”和“完成”，使慢推理不必占用一个持续数分钟的 HTTP 连接。
    # main 在 play/record/task 后调用；accepted 是首次提交的结果 dict，返回整数进程退出码。
    # timeout 的 float | None 表示“秒数或不设期限”；不是必须传两种值。
    job_id = accepted.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError("服务接收了操作但未返回任务编号，不能确认执行结果。")
    # None 用作无期限的标记；不能写 time.monotonic() + None。
    deadline = time.monotonic() + timeout if timeout is not None else None
    # previous 保存上次看到的状态，避免每 0.2 秒都重复打印相同文字。
    previous = None
    try:
        while deadline is None or time.monotonic() < deadline:
            # 查询的是同一个 job_id；轮询 GET，不会重新执行 POST 提交任务。
            result = client.job(job_id)
            if result.get("success") is not True:
                _emit(result, as_json=as_json)
                return 1
            job = result.get("job")
            # result 是 HTTP 外层结果，job 才是内层任务；外层 success 不等于任务 completed。
            if not isinstance(job, dict) or job.get("id") != job_id:
                raise RuntimeError("服务返回的任务编号不匹配，不能确认执行结果。")
            status = job.get("status")
            progress = (status, job.get("current_subtask"))
            if progress != previous:
                _emit({"mode": getattr(client, "mode", None), **result}, as_json=as_json)
                previous = progress
            if status in _TERMINAL_STATES:
                # 约定退出码：0 成功，130 取消，其余失败用 1；交给 main 再交给操作系统。
                return 0 if status == "completed" else (130 if status == "cancelled" else 1)
            time.sleep(0.2)
    except KeyboardInterrupt:
        # 键盘 Ctrl+C 会抛此异常；这里转为停止请求和退出码，不把中断记作执行成功。
        message = _stop_after_wait_failure(client)
        _emit({"success": False, "status": "stop_requested", "message": message,
               "job_id": job_id}, as_json=as_json)
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        stop_message = _stop_after_wait_failure(client)
        raise RuntimeError(str(error) + " " + stop_message) from error
    stop_message = _stop_after_wait_failure(client)
    raise RuntimeError("等待任务结果超时；未将其视为完成。" + stop_message)


def main(argv=None) -> int:
    # 最下方 SystemExit 调用此函数；先解析参数，再只执行选中分支，返回 CLI 退出码。
    # args 是 Namespace；点号读取解析器创建的属性，例如 args.operation 为 "status"。
    args = _arguments(argv)
    # 先放 None/False，保证即使创建客户端或发请求中途失败，异常处理也能判断做到了哪一步。
    client = None
    motion_submitted = False
    try:
        if args.operation == "init":
            # 第一个 args.config 按位置传，后面 right_port=... 等按参数名传给 create_config。
            # 返回的 path 是配置文件路径；创建配置本身不会开启串口。
            path = create_config(args.config, right_port=args.right_port, right_id=args.right_id,
                                 left_port=args.left_port, left_id=args.left_id,
                                 leader_port=args.leader_port, leader_id=args.leader_id,
                                 mock=args.mock)
            _emit({"success": True, "status": "configured", "mode": "mock" if args.mock else "real",
                   "message": f"配置已创建：{path}。尚未连接硬件或启用动作。"}, as_json=args.json)
            return 0
        if args.operation == "serve":
            # 延迟导入：查看帮助、查询状态不必导入 LeRobot 或初始化硬件。
            from .service import serve
            # 此分支进入长期运行的服务；不同于后面的客户端查询，它会持有硬件工作线程。
            policy = {key: getattr(args, key) for key in (
                "trained_goal_language", "adapter_dir", "replan_steps", "rtc")
                if getattr(args, key) is not None}
            outcome = serve(args.config, policy_overrides=policy)
            # 某些结束路径未显式返回整数时使用 0；type(... ) is int 排除 bool 被当作退出码。
            return outcome if type(outcome) is int else 0

        # 类名后加 (...) 创建客户端对象；此时只读 HTTP 配置，尚未发送命令或打开机械臂串口。
        client = ArmClient(args.config)
        if args.operation == "status":
            result = client.status()
        elif args.operation == "auto":
            # args.setting == "on" 是比较表达式，结果为 bool，传给 enabled 关键字参数。
            result = client.command("auto", enabled=args.setting == "on")
        elif args.operation == "play":
            # 提交前先置 True；即使回包未到就 Ctrl+C，也知道服务可能已经收到动作请求。
            motion_submitted = True
            result = client.command("play", name=args.name, source="manual")
        elif args.operation == "task":
            # 与 NPC 复用同一集成入口，因此 CLI task 也能按需启动后台服务。
            from .integration import ArmIntegration
            motion_submitted = True
            # JSON 模式传入一个不输出文字的回调，避免普通提示混入机器读取的 JSON。
            # lambda _: None 是接收一个参数、返回 None 的小函数；_ 表示此参数不需要使用。
            integration = ArmIntegration(args.config, notify=(lambda _: None) if args.json else None)
            # perform 返回“接受/拒绝”的结果；是否完成由下面的 _wait_for_job 继续查询。
            result = integration.perform("task", instruction=args.instruction, arm=args.arm)
            # The independent service persists after this CLI exits. Closing
            # this integration would cancel the task submitted with --no-wait.
        elif args.operation == "record":
            # Check availability before the preparation countdown. The service
            # remains authoritative if state changes while the operator waits.
            # snapshot 来自查询服务，里面的 armed/busy 是此刻状态，不是本地猜测。
            snapshot = client.status()
            if snapshot.get("success") is not True:
                _emit(snapshot, as_json=args.json)
                return 1
            if snapshot.get("mode") == "mock" and not args.json:
                print("[模拟录制] 使用模拟主臂，结果不能用于真实机械臂。", flush=True)
            if snapshot.get("armed") is not True or snapshot.get("busy"):
                _emit({"success": False, "status": "not_ready",
                       "message": "录制前请确认服务空闲，并在本地执行 arm 启用动作。"}, as_json=args.json)
                return 1
            # range(起点, 不包含的终点, 步长) 依次产生 3/2/1 等倒计时整数。
            for remaining in range(args.countdown, 0, -1):
                if not args.json:
                    print(f"准备录制 {args.arm} / {args.name}：{remaining}…", flush=True)
                time.sleep(1.0)
            motion_submitted = True
            result = client.command("record", name=args.name, arm=args.arm,
                                    duration_s=float(args.seconds))
        else:
            # 其余已由 argparse 限定的命令（如 arm/stop/list/shutdown）直接使用协议同名操作。
            result = client.command(args.operation)

        # **result 展开响应字段；若响应本来就有 mode，右侧值覆盖左侧的客户端默认 mode。
        _emit({"mode": getattr(client, "mode", None), **result}, as_json=args.json)
        if result.get("success") is not True:
            return 1
        if args.operation == "stop":
            confirmed = _confirm_stop(client)
            _emit(confirmed, as_json=args.json)
            return 0 if confirmed.get("success") is True else 1
        if args.operation == "shutdown" and not args.no_wait:
            confirmed = _wait_for_shutdown(client, args.wait_timeout)
            _emit(confirmed, as_json=args.json)
            return 0 if confirmed.get("success") is True else 1
        if args.operation in ("record", "play", "task") and not args.no_wait:
            # 这是“已接收→等待实际终态”的入口；--no-wait 会跳过，但不会取消后台任务。
            return _wait_for_job(client, result, timeout=args.wait_timeout, as_json=args.json)
        return 0
    except KeyboardInterrupt:
        # Ctrl+C can arrive while a POST is in flight, before its job_id is
        # returned. Such an operation may already be running on the service.
        message = (_stop_after_wait_failure(client) if motion_submitted and client is not None
                   else "本地操作已取消，尚未提交动作。")
        _emit({"success": False, "status": "stop_requested" if motion_submitted else "cancelled",
               "message": message}, as_json=args.json)
        return 130
    except FileNotFoundError as error:
        # 配置或动作文件缺失时，把异常转成 CLI 输出；Path 仅表示路径，不会在这里建文件。
        config_path = Path(args.config) if args.config else default_config_path()
        message = str(error)
        if not config_path.is_file():
            message += ("。先执行 python -m friston3.arms init；"
                        f"默认配置位置：{default_config_path()}。")
        _emit({"success": False, "status": "missing_file",
               "message": message}, as_json=args.json)
        return 1
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        _emit({"success": False, "status": "error", "message": str(error)}, as_json=args.json)
        return 1


if __name__ == "__main__":
    # 直接作为程序运行才进入此块；被 import 时不会自动 main()。
    # main() 的整数交给 SystemExit，成为终端里的进程退出码。
    raise SystemExit(main())
