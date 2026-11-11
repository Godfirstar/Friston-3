"""程序入口：创建共享设备，再按输入、身份、技能和回复呈现运行接待循环。

控制台语法在 friston3.console；模型规划与技能分发仍交给 Agent。
"""
# import 把模块载入当前程序，并绑定名字 argparse；它是 Python 自带的命令行解析库，
# 不是本仓库中的文件。后面的 argparse.ArgumentParser 就是读取该模块里的类。
import argparse
# 只把指定名字绑定到当前文件，之后直接写 ExitStack 即可。
# ExitStack 是标准库 contextlib 的类，用来登记退出时要执行的清理函数。
from contextlib import ExitStack

# IDENTITY 是 friston3/npc/identity.py 中已经创建的 Identity 实例，不是字符串或类名。
# 对象.name 表示读取该对象的 name 属性，例如下文用 IDENTITY.name 取得显示名称。
from friston3.npc.identity import IDENTITY
# 括号让这条 import 可以跨多行；其中每个名字都是 console.py 定义的函数。
# 此处只是取得函数，并没有调用它们；出现 函数名(...) 时才执行函数体。
from friston3.console import (
    discard_input, read_instruction, new_visitor, handle_identity_command,
    run_device_command, handle_session_command, print_help,
)


# 【函数作用】定义本程序接受哪些启动选项，再把本次终端输入解析成属性对象。
# 【在哪里调用】本文件 main() 的第一条语句 arguments = _arguments()。
# 【输入/返回】不接收显式参数；parse_args() 默认读取进程的命令行参数，
# 返回 argparse.Namespace。这里只解析选项，具体下载/查询/删除/录入在 main() 分支执行。
# def 定义函数；冒号后的缩进组成函数体。定义不会马上执行，要用 _arguments() 调用。
# 函数名前的单下划线是“供模块内部使用”的命名约定，不是 Python 强制访问权限。
def _arguments():
    # argparse.ArgumentParser 是标准库中的类；类名(...) 创建一个解析器实例。
    # description="..." 是关键字参数：左边是构造器的参数名，右边是传入的字符串。
    # = 把右侧返回的对象绑定给左侧局部变量 parser。parser 在这里首次定义，
    # 名字由作者自取，不是 Python 关键字；它只在这次 _arguments() 调用中使用。
    # 这个对象负责保存参数规则、生成 --help 内容、检查和解析终端输入。
    parser = argparse.ArgumentParser(description="Friston-3 终端")
    # parser.方法名()：取出这个实例的方法并调用；该方法由 argparse 库实现，
    # 不需要在 main.py 再定义 add_mutually_exclusive_group。
    # mutually exclusive 意为“互斥”：在同一组里登记的选项，最多只能同时选一个。
    # 方法返回“参数组对象”，= 将它绑定为新的局部变量 operation；并不是返回某次操作。
    # 未设置 required=True，所以允许一个都不选：直接 python main.py 会进入正常会话。
    operation = parser.add_mutually_exclusive_group()
    # operation.add_argument(...) 向刚才的互斥组登记一个选项，此刻不会下载模型。
    # "--download-owner-models" 是第一个位置参数：靠所在位置传给 add_argument。
    # choices= 和 help= 是关键字参数：靠参数名字传值；它们不是新建的本地变量。
    # ("face", "voice", "all") 是包含三个字符串的元组，choices 要求用户取其中一个。
    # help 是在 python main.py --help 中展示的解释；解析器默认还提供 -h/--help。
    # 一对调用括号内部可以自然换行，因此下一行 help=... 仍属于这一次方法调用。
    # 用例：python main.py --download-owner-models face；缺少 face 等取值会报参数错误。
    operation.add_argument("--download-owner-models", choices=("face", "voice", "all"),
                           help="显式联网下载身份模型；不启动硬件或对话")
    # action="store_true" 表示布尔开关：出现 --owner-status 就得到 True，未出现则 False。
    # 它不再接一个取值，所以写 --owner-status 即可，不要再写 --owner-status True。
    operation.add_argument("--owner-status", action="store_true",
                           help="查看本地身份模板和模型状态；不启动硬件")
    # 另一个布尔开关；这里仅登记“用户是否选了删除”，真正删除在 main() 中发生。
    operation.add_argument("--forget-owner", action="store_true",
                           help="删除主人人脸与声纹模板")
    # 这是带字符串取值的选项，用法如 --enroll-owner both；没有指定时默认是 None。
    # 四个选项都放在 operation 组里，所以 --owner-status 和 --enroll-owner 不能同用。
    operation.add_argument("--enroll-owner", choices=("face", "voice", "both"),
                           help="在本地控制台录入主人；不加载 ASR、对话或 TTS")
    # parse_args() 才真正读取 sys.argv[1:]（脚本名之后的终端参数），检查并生成 Namespace。
    # argparse 自动把选项名里的 - 换成属性名里的 _：--owner-status → owner_status。
    # 例如 --owner-status 对应 Namespace(download_owner_models=None, owner_status=True,
    # forget_owner=False, enroll_owner=None)，后文用 arguments.owner_status 读取。
    # return 把该对象交回 main()，结束本函数；--help 会显示帮助并正常退出，
    # 非法取值/互斥冲突则由 argparse 显示错误并退出，不会继续启动 main 的设备分支。
    return parser.parse_args()


# 【作用/调用】文件末尾 main() 启动本函数；它按启动选项选择维护操作或正常会话。
# 【返回】整数 0 表示正常结束，1 表示失败，最外层把它作为操作系统可读取的退出码。
# 【下游】_arguments 解析 → 身份/设备对象初始化 → _run_session 交互 → finally 清理。
def main():
    # = 右侧先调用 _arguments，再把返回的 Namespace 绑定给本地变量 arguments。
    # 名字 arguments 与上面 parser/operation 并非同一个变量，也不需要提前声明类型。
    arguments = _arguments()
    # 连续赋值：四个变量初始都指向 None（“暂时没有对象”），此刻没有创建任何设备。
    # 这样即使初始化中途失败，finally 也能判断哪些对象已经创建成功、需要关闭。
    face = camera = speech_input = owner = None
    # try 包住可能失败的初始化/运行；异常跳到对应 except，finally 负责收尾。
    try:
        # vars(arguments) 把 Namespace 的属性取成字典，.values() 取得其所有值。
        # any(...) 判断是否至少有一项为真；not 取反。四项均为 None/False 表示未选维护模式。
        if not any(vars(arguments).values()):
            # 函数内 import 只在走到这个分支时执行；维护命令无需先探测完整聊天环境。
            from friston3.runtime_bootstrap import use_installed_runtime
            # 项目函数来自 runtime_bootstrap.py：当前环境可用时返回 None；
            # 若交给另一个已有 Python 运行，则等子进程结束，返回它的整数退出码。
            child_status = use_installed_runtime()
            # is not None 判断“有返回码”，0 也是有效返回码；不能简写 if child_status。
            if child_status is not None:
                return child_status
        # Setup/status paths return before creating device workers or chat models.
        # 点号读取 Namespace 属性；此属性是字符串或 None，不是同名函数调用。
        if arguments.download_owner_models:
            from friston3.owner_setup import download_owner_models
            # 传入 face/voice/all，得到结果字典 prepared；下载是在这一行发生的。
            prepared = download_owner_models(arguments.download_owner_models)
            # 字典[键] 取该键的值；“A if 条件 else B”是条件表达式，按结果选择返回 0 或 1。
            return 0 if prepared["success"] else 1

        from friston3.owner_identity import OwnerIdentityManager, OwnerProfile

        if arguments.forget_owner:
            # OwnerProfile 是 owner_identity.py 的类；先创建实例，再调用其 forget 方法。
            # 连写 Class().method() 与先用变量保存实例再调 method 的基本步骤相同。
            OwnerProfile().forget()
            print("主人人脸与声纹模板已删除；对话记忆保持原状。")
            return 0
        if arguments.owner_status:
            # owner 由 None 更新为身份管理器对象，finally 因而知道它需要 close()。
            owner = OwnerIdentityManager()
            # 从内向外求值：先调用 status_text() 得到文字，再由内置 print 输出到终端。
            print(owner.status_text())
            return 0

        from friston3.hardware.camera import create_camera
        from stackchan_integration import StackChanFaceEvents

        # Camera, microphone, expressions and playback share one USB/TCP owner.
        # face 是 StackChanFaceEvents 实例（定义在 stackchan_integration/face_events.py），
        # 它封装栈酱设备交互，不只是“脸的图像”；camera/语音稍后复用它，避免重复连接。
        face = StackChanFaceEvents()
        # create_camera 是 hardware/camera.py 的工厂函数：接收已有设备，返回相机接口对象。
        camera = create_camera(face)
        # 关键字参数 camera=camera：左边是构造器的形参名，右边是本函数的局部变量。
        # 两边恰好同名，但含义不同；这里只传对象引用，没有复制一份硬件。
        owner = OwnerIdentityManager(camera=camera, device=face)
        if arguments.enroll_owner:
            # enroll 方法来自 OwnerIdentityManager；完成指定身份录入后立即结束本次进程。
            return 0 if owner.enroll(arguments.enroll_owner) else 1

        # f"...{表达式}..." 是格式化字符串：先求 IDENTITY.name，再把它插入文字中。
        print(f"{IDENTITY.name} 启动（栈酱摄像头 / 栈酱麦克风 / 自动连接）")
        print("[StackChan] 自动发现 USB 或本地 Wi-Fi；设备尚未连接时会继续等待。")
        owner.warmup()  # 调用身份管理器的预热方法；方法本身决定需要准备哪些已有模型。
        from stackchan_integration.voice_input import StackChanSpeechInput
        # speech_input 是 voice_input.py 定义的语音输入对象，仍接收同一个 face 设备对象。
        speech_input = StackChanSpeechInput(face)
        # 这里没有 begin_utterance() 的括号：把“方法对象”保存为回调，而非现在执行。
        # 语音模块拿到整段麦克风音频后、开始 ASR（语音转文字）前再调用它，
        # 让身份识别与这句录音关联，并可与随后进行的 ASR 重叠执行。
        speech_input.audio_observer = owner.begin_utterance
        speech_input.preload()  # 语音输入对象的方法，提前准备识别所需资源。
        # 四个位置参数按顺序传给本文件下方 _run_session；正常会话循环在那个函数内。
        _run_session(face, camera, speech_input, owner)
        return 0
    # except (A, B) 可以捕获任意一种指定异常；EOF 表示输入结束，KeyboardInterrupt 常来自 Ctrl+C。
    except (EOFError, KeyboardInterrupt):
        print(f"\n{IDENTITY.name} 已退出。")
        return 0
    except Exception as error:
        # as error 把捕获的异常对象绑定给局部变量；不是一个提前声明的全局错误变量。
        # type(error).__name__ 取异常类的名字，{error} 则显示这次异常的具体说明。
        # Include dependency/configuration failures in the visible startup
        # boundary. No device operation or model download is retried silently.
        print(f"[{IDENTITY.name}] 启动或运行未完成：{type(error).__name__}: {error}")
        print("请检查上面指明的本地模型或依赖；普通设备断线会自动恢复。")
        return 1
    finally:
        # finally 在正常完成、return、或异常离开 try 时都会执行（进程被强制终止等情况除外）。
        # for 每轮从这个元组取一个对象，绑定为 resource；这是按顺序关闭，不是四线程同时关闭。
        for resource in (owner, speech_input, camera, face):
            # 某项还是 None，说明它没有创建成功；不能对 None 调用 close。
            if resource is not None:
                try:
                    # 四种对象分别实现 close()；相同方法名是调用约定，不要求它们是同一个类。
                    resource.close()
                except Exception as error:
                    # 一个 close 失败只报告这次错误，for 仍继续尝试关闭其他对象。
                    print(f"[{IDENTITY.name}] 资源释放未确认：{type(error).__name__}: {error}")


def _run_session(face, camera, speech_input, owner):
    """由 main() 调用的组装入口：把输入、技能和输出连接成完整会话。

    本函数持有 voice/arms 的生命周期；具体规划交给 Agent/LLMPlanner，
    设备能力交给已注册的技能。face/camera/owner 则由外层 main() 释放。
    阅读顺序：注册对象 → 读取输入 → 取得身份 → 执行技能 → 呈现真实结果。
    """
    # 括号里的四个名字是形参，由 main() 按位置传入已有实例；这里不会重新创建这四个对象。
    # 下面 from ... import ... 的最后一段分别是类/函数名；实际定义就在对应模块的 .py 中。
    # Agent 负责一次请求，SkillManager 保存技能，LLMPlanner 选择技能；不要把这三个类混为模型。
    from friston3.agent import Agent
    from friston3.skills.auto_photo import AutoPhotoSkill
    from friston3.skills.photo import PhotoSkill
    from friston3.skills.manager import SkillManager
    from friston3.skills.npc import NpcTalkSkill
    from friston3.skills.volume import VolumeControlSkill
    from friston3.skills.device_settings import DeviceSettingsSkill
    from friston3.skills.arms import ArmsControlSkill
    from friston3.arms.integration import ArmIntegration, is_immediate_stop
    from friston3.vision.yunet_observer import YuNetObserver
    from friston3.planner import LLMPlanner
    from friston3.input_text import prepare_input_text, is_memory_command
    from stackchan_integration.voice_output import AgentSpeechOutput

    # ExitStack 是动态的 with：创建一个资源就登记一个清理回调，退出时倒序执行。
    # 即使下一项初始化失败，或某个 close 抛异常，先前登记的其余清理仍会执行。
    # with 对象 as 名字 是上下文管理语法：进入时取得资源，退出缩进块时执行清理协议。
    # ExitStack() 创建标准库对象，as resources 把进入该上下文得到的对象绑定给这个变量。
    # 本例 resources 就是这个 ExitStack；所有登记操作只在 with 块结束时才被调用。
    with ExitStack() as resources:
        # callback(函数, 参数1, 参数2) 登记未来的调用 discard_input(owner, speech_input)。
        # 这里传的是函数本身；若写 discard_input(...)，就会现在调用并登记它的返回值。
        # 此项最先登记，所以在后面的 arms.close、voice.close 之后最后执行。
        resources.callback(discard_input, owner, speech_input)
        voice = AgentSpeechOutput(face)  # 异步预热 TTS；音量技能复用同一个播音对象。
        resources.callback(voice.close)  # 传方法对象；此刻不加 ()，否则会立即关闭。
        arms = ArmIntegration()  # 交互进程的集成对象；真实串口由独立服务持有。
        resources.callback(arms.close)

        # 依赖注入：把已创建的相机/语音/机械臂对象交给技能，避免各自重复占用设备。
        # SkillManager 类在 skills/manager.py；manager 是本次会话唯一的技能注册表实例。
        # Python 的 Class(...) 会创建实例并调用初始化逻辑 __init__，但不等于执行技能。
        manager = SkillManager()
        # 从内向外执行：PhotoSkill(camera) 先创建拍照技能，再传给 manager.register。
        # register 在 SkillManager 中定义，保存“技能名称 → 技能实例”，不在这里拍照。
        manager.register(PhotoSkill(camera))
        # AutoPhotoSkill(camera, observer=...) 混合位置参数与关键字参数。
        # YuNetObserver() 创建观察器；自动拍照技能稍后用它分析相机帧。
        manager.register(AutoPhotoSkill(camera, observer=YuNetObserver()))
        # 注册语音音量、设备设置、机械臂三种能力；传入各自已经创建好的共享依赖。
        manager.register(VolumeControlSkill(voice))
        manager.register(DeviceSettingsSkill(face))
        manager.register(ArmsControlSkill(arms))
        # NpcTalkSkill 来自 skills/npc.py；构造器会准备会话/知识/语义等依赖，
        # 因而“创建对象”不保证完全没有耗时或副作用，具体要看它的 __init__。
        npc = NpcTalkSkill()
        manager.register(npc)
        # Planner 从注册表读技能描述，Agent 用同一注册表执行模型选中的技能。
        # 先用同一个 manager 创建 LLMPlanner，再把 manager 和规划器作为两项实参传给 Agent。
        # 此后调用 agent.handle(...) 才会处理某一条具体用户请求。
        agent = Agent(manager, LLMPlanner(manager))

        # speech_input is not None 的结果是 bool，传给 console.print_help 的 speech_enabled 参数。
        print_help(speech_enabled=speech_input is not None)

        # while True 是持续循环；正常每轮处理一句话，break 才离开循环，continue 则开始下一轮。
        while True:
            try:
                # console.read_instruction 返回二元组 (文本, 来源)，逗号左侧是“解包赋值”。
                # raw_instruction 得到字符串，input_source 得到 "text" 或 "speech"。
                raw_instruction, input_source = read_instruction(owner, speech_input)
            except (EOFError, KeyboardInterrupt):
                print(f"\n{IDENTITY.name} 已退出。")
                break
            # 空字符串在条件判断中为假；not 将结果取反。空输入不进入模型规划。
            if not raw_instruction:
                # console.discard_input 清本轮音频/身份观察，不删除长期记忆或身份模板。
                discard_input(owner, speech_input)
                continue
            # /t is an optional local spelling for one typed utterance.
            # == 比较值，and 要求两边都成立；.lower/.startswith 是 Python 字符串的方法。
            # 先忽略大小写判断是否以 "/t " 开头；这里不会修改 raw_instruction 本身。
            if input_source == "text" and raw_instruction.lower().startswith("/t "):
                # [3:] 是切片：取索引 3 起到末尾（Python 从 0 编号），去掉三个字符 "/t "。
                # .strip() 返回去掉两侧空白的新字符串，= 才更新 raw_instruction。
                raw_instruction = raw_instruction[3:].strip()
            # 规范化文本用于本地命令识别；原始文本稍后仍传给 Agent 保留输入证据。
            # prepare_input_text 在 input_text.py；返回字典，然后用 ["normalized_text"] 取字段。
            # 一次表达式既能调用函数，也能继续对返回对象取值；不是访问函数自身的属性。
            instruction = prepare_input_text(
                raw_instruction, input_source=input_source)["normalized_text"]
            # rstrip(字符集合) 去掉右端属于这些字符的部分；lower() 再统一英文大小写。
            # command 是专门匹配控制台命令的副本；instruction 和原始输入仍各自保留。
            command = instruction.rstrip("。！？!?.,，；; ").lower()
            if not instruction:
                discard_input(owner, speech_input)
                continue

            # Stop has no identity/LLM precondition once input has arrived.
            # The microphone remains half-duplex: use the independent service
            # CLI while this thread is occupied by planning or playback.
            # is_immediate_stop 来自 arms/integration.py，识别直接停止请求。
            # 分支位于身份和LLM之前；符合时直接分发技能，处理完 continue，不再生成聊天回答。
            if is_immediate_stop(instruction):
                discard_input(owner, speech_input)
                # {"operation": "stop"} 是字典：冒号左边为键，右边为值。
                # manager.execute 在 skills/manager.py，根据名字找到技能实例并调用其 execute。
                # stopped 是结果字典，真正的机械臂请求由 ArmsControlSkill → ArmIntegration 发送。
                stopped = manager.execute("control_arms", {"operation": "stop"})
                print(f"[Arms] {stopped['message']}")
                continue

            if input_source == "text":
                # split() 按空白拆成单词列表，[:1] 取前 0～1 个元素，空列表也不会越界。
                # in 判断这个小列表是否等于括号中的任意一项，如 ["/arms"]；并非子串搜索。
                if command.split()[:1] in (["/arms"], ["/voice"], ["/device"]):
                    discard_input(owner, speech_input)
                    # console.run_device_command 把明确命令转成技能参数，直接调用 manager。
                    run_device_command(command, manager, face)
                    continue
                # The physical console can inspect the actual previous result
                # without replacing a visitor's trace with a new owner turn.
                if command in ("查看证据", "/evidence"):
                    discard_input(owner, speech_input)
                    # 此方法在 NpcTalkSkill 内定义；查询上一次回答的实际依据，不发新规划请求。
                    print(npc.explain_last_response())
                    continue
                # console.handle_identity_command 返回 bool：True 表示已识别并处理，跳过剩余分支。
                if handle_identity_command(command, owner, speech_input, npc):
                    continue
            # elif 只有前面的 if input_source == "text" 不成立时才会检查。
            elif command.split()[:1] in (["/owner"], ["/visitor"]):
                discard_input(owner, speech_input)
                print(f"{IDENTITY.name}: 这项管理操作需要在电脑控制台输入。")
                continue

            if command in ("下一位来访者", "下一位访客", "新访客"):
                # console.new_visitor 完成会话交接；接收的是实例，函数内可修改这些对象的状态。
                new_visitor(owner, speech_input, npc)
                continue

            # 条件表达式只求被选中的一侧；文字输入时不读取 speech_input.last_audio_pcm。
            # pcm 是最近一句的原始音频数据，sample_rate 是每秒采样数（Hz）；不等于识别文本。
            pcm = speech_input.last_audio_pcm if input_source == "speech" else None
            sample_rate = speech_input.last_audio_rate if input_source == "speech" else 16000
            try:
                # OwnerIdentityManager.context_for 综合本轮输入来源和观测，返回身份字典 context。
                # 这里 context 是后文可信身份数据；不是 LLM 自己填写的模型“上下文窗口”。
                context = owner.context_for(input_source=input_source, pcm=pcm, sample_rate=sample_rate)
            finally:
                # 无论身份判断成功或失败都清录音，避免下一句话借用本轮音频。
                if speech_input is not None:
                    speech_input.clear_last_audio()
            # 身份来自本轮识别结果；不要让模型生成的技能参数决定“谁在说话”。
            npc.set_interlocutor(context)
            # 两层条件表达式按 role 选择显示标签；括号只分组/续行，没有逗号就不是元组。
            label = "主人 / 可露希尔" if context["role"] == "owner" else (
                "来访者" if context["role"] == "visitor" else "身份未确认")
            # f 字符串中 {context['wait_ms']:.0f} 把数值显示成零位小数；只影响显示，不改原值。
            print(f"[身份] {label}；依据={context['source']}；额外等待={context['wait_ms']:.0f}ms")

            # Spoken management sees this utterance's context, never the prior
            # speaker's private history or authorization.
            # console.handle_session_command 返回 "exit" / "handled" / None，表示退出/已处理/未命中。
            # 这里 action 是普通局部变量，不是前面 argparse 里 action= 的特殊值。
            action = handle_session_command(command, npc, input_source=input_source, context=context)
            if action == "exit":
                break
            if action == "handled":
                continue

            face.on_request()  # 栈酱事件接口：开始呈现“正在处理请求”的状态，定义在 face_events.py。
            try:
                # 业务执行入口：规划是“打算做什么”，result 是“实际调用后返回什么”。
                # agent.handle 在 agent.py；返回字典。interlocutor=context 是把本轮身份传给它。
                result = agent.handle(raw_instruction, input_source=input_source, interlocutor=context)
                # 字典.get(key) 在键缺失时返回 None；与 result[key] 缺失时抛 KeyError 不同。
                # executed 是管理器记录的实际技能名称，不能仅凭该字段断言技能执行成功。
                executed = result.get("executed_skill")
                # 多个 and 条件同时成立才取消对话手势；is True 要求实际布尔 True。
                # 外层括号允许条件跨多行，不改变布尔运算含义。
                if (executed == "configure_device" and result.get("success") is True
                        and result.get("setting") == "gestures" and result.get("choice") == "off"):
                    # ArmIntegration.cancel_dialogue 取消自己负责的对话动作；可能没有任务而返回 None。
                    arm_stop = arms.cancel_dialogue()
                    if arm_stop is not None:
                        # 字典[新键] = 值 会增加/替换一个字段，把机械臂取消结果并入本次设备设置结果。
                        result["arm_motion_stop"] = arm_stop
                        if arm_stop.get("success") is not True or arm_stop.get("status") == "stopping":
                            # += 在这里把字符串拼接后重新赋给同一字段，用于补充尚未完成的停止说明。
                            result["message"] += " 机械臂：" + arm_stop["message"]
                if executed in ("take_photo", "auto_photo"):
                    # NpcTalkSkill.record_execution 保存真实拍照执行结果，供后续对话追问使用。
                    npc.record_execution(executed, result)
                # get 的第二个参数只在键不存在时使用；str(...) 将取得的值转为显示用字符串。
                reply_text = str(result.get("message", "任务结束"))
                if (executed in ("take_photo", "auto_photo")
                        and result.get("success") is True and context["role"] == "owner"):
                    # Presentation only: retain original device result, saved
                    # path, success flag and execution record without changes.
                    reply_text = "可露希尔，" + reply_text
                print(f"{IDENTITY.name}:", reply_text)
                # StackChanFaceEvents.prepare_reply 读取表情/动作提示，准备本轮呈现。
                face.prepare_reply(result)
                # reply_start 返回闭包，保存本轮动作与“是否已触发”；这里还未执行动作。
                # ArmIntegration.reply_start 可能返回一个可调用函数，也可能返回 None。
                # allow= 后面的 not 是布尔取反；记忆管理语句不触发这次对话联动动作。
                arm_start = arms.reply_start(result, allow=not is_memory_command(instruction))
                # 播报条件分两层：外层请求成功或属于机械臂；内层是普通NPC回答，
                # 或设备技能明确要求 speak_confirmation。and/or 有短路特性：结果已确定时不再算右侧。
                # get("success", True) 表示字段缺失时此处默认 True；这不等于给结果写入 success 字段。
                if ((result.get("success", True) or executed == "control_arms")
                        and ((executed == "npc_talk" and not is_memory_command(instruction))
                             or (executed in ("control_volume", "configure_device", "control_arms")
                                 and result.get("speak_confirmation")))):
                    # AgentSpeechOutput.speak 来自 voice_output.py；on_start 接收函数对象，
                    # 由播音过程在适当时机调用，因此不要写成 on_start=arm_start()。
                    voice.speak(reply_text, on_start=arm_start)
                if arm_start is not None:
                    # 为静音/无语音补触发；闭包内部保证最多一次，因此不是重复动作。
                    # 变量也可以保存函数；arm_start() 就是执行它保存的函数，而非调用某个类。
                    arm_start()
                # Speech consumes the gesture just before the first clip.
                # A muted/unavailable voice still gets one bounded gesture.
                face.on_task_completed(result)
            finally:
                # 即使规划/技能/播音抛异常，也通知设备结束当前回复状态；异常会继续向外传播。
                face.on_reply_ready()


# Python 为每个模块提供 __name__：直接执行本文件时为 "__main__"，被 import 时为 "main"。
# 因而“导入函数供别处使用”不会自动进入设备主循环；直接 python main.py 才走这里。
if __name__ == "__main__":
    # 先调用 main() 得到退出码，再构造 SystemExit 并 raise；这是明确结束进程的方式。
    # main 内部的 finally 已先执行，所以正常退出时会先清理资源，再交出退出码。
    raise SystemExit(main())
