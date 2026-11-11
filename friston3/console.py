"""直接控制指令与输入切换，不调用模型规划。

设备命令仅由键盘入口调用；语音会话命令必须先取得本轮身份。
这里只解释指令，设备参数仍交给 SkillManager 统一校验。
"""
import sqlite3

from friston3.npc.identity import IDENTITY


def discard_input(owner, speech_input):
    """丢弃上一轮身份观察和录音，避免键盘操作沿用旧说话人的信息。"""
    # 调用方：main和本模块的命令分支；owner来自main创建的OwnerIdentityManager实例。
    # invalidate定义在owner_identity.py，只作废本轮观察，不删除磁盘中的主人模板。
    owner.invalidate()
    if speech_input is not None:
        speech_input.clear_last_audio()


def read_instruction(owner, speech_input):
    # main._run_session每轮调用；返回(输入字符串,来源字符串)，由main用两个变量解包。
    # speech_input为None时用内置input阻塞等键盘；否则使用voice_input.py的read方法。
    if speech_input is None:
        # return后逗号把两个值组成元组；strip去除两端空白，不是修改用户单词。
        return input("\n你: ").strip(), "text"
    try:
        return speech_input.read().strip(), "speech"
    except KeyboardInterrupt:
        # 这里捕获Ctrl+C并切入键盘，不把这次中断继续抛给main退出程序。
        # 调度器会取消正在等待的录音；被中断的录音不能确认键盘操作者身份。
        discard_input(owner, speech_input)
        print("\n已进入电脑控制台。输入命令或一句话；直接按 Enter 返回麦克风。")
        return input("电脑控制台: ").strip(), "text"


def new_visitor(owner, speech_input, npc):
    # 调用方：main识别“下一位”或handle_identity_command处理/visitor new。
    # 参数是三个已经存在的对象；下面调用它们的方法，会改变对应实例的会话状态。
    discard_input(owner, speech_input)
    # 交接会结束主人演示模式，只清理上一位匿名访客的临时会话。
    profile = owner.handover_visitor()
    # profile是handover_visitor返回的会话标识；set_interlocutor在skills/npc.py定义。
    # 字典中的role和来源由实际控制台/身份流程提供，不是让模型任意填写。
    npc.set_interlocutor({
        "role": "visitor", "profile_id": profile, "source": "visitor_handover",
        "authenticated": False, "local_operator": False,
        "reason": "现场明确开始接待下一位来访者",
    })
    print(f"{IDENTITY.name}: 已开始新访客会话。")


def handle_identity_command(command, owner, speech_input, npc):
    """只从键盘入口调用；这些身份管理能力不登记为模型技能。"""
    # main只对键盘输入调用此函数；返回True表示命令已识别并处理，False表示交给其他分支。
    # split是字符串方法，按空白拆出列表，如'/owner enroll face'→三项字符串。
    words = command.split()
    # or短路：words为空时不再访问words[0]，避免空列表索引错误。
    if not words or words[0] not in ("/owner", "/visitor"):
        return False
    discard_input(owner, speech_input)
    try:
        if words == ["/owner", "status"]:
            print(owner.status_text())
        elif words[:2] == ["/owner", "enroll"] and len(words) in (2, 3):
            # [:2]取前两项；words[2]是第三项。条件表达式在省略录入类型时提供'both'。
            owner.enroll(words[2] if len(words) == 3 else "both")
        elif words[:2] == ["/owner", "mode"] and len(words) == 3:
            owner.set_mode(words[2])
            if words[2] == "owner":
                print("已启用主人演示模式，120 秒后自动恢复识别；可用 /owner mode auto 提前结束。")
            elif words[2] == "once":
                print("30 秒内下一条对话按主人处理，随后恢复自动识别。")
            elif words[2] == "visitor":
                new_visitor(owner, speech_input, npc)
                print("已固定为访客接待模式；用 /owner mode auto 恢复自动识别。")
            else:
                print("已恢复自动识别；电脑键盘输入按主人处理。")
        elif words == ["/owner", "forget"]:
            # owner.profile是对象的属性；再取.forget方法并调用才执行模板删除。
            owner.profile.forget()
            owner.set_mode("auto")
            print("主人人脸与声纹模板已删除；对话记忆保持原状。")
        elif words == ["/visitor", "new"]:
            new_visitor(owner, speech_input, npc)
        else:
            print("身份命令：/owner status | /owner enroll face|voice|both | "
                  "/owner mode auto|owner|visitor|once | /owner forget | /visitor new")
    except (EOFError, KeyboardInterrupt):
        discard_input(owner, speech_input)
        print("电脑控制台操作已取消。")
    except Exception as error:
        # as把异常实例绑定为error；type(error).__name__用于显示异常类别。
        discard_input(owner, speech_input)
        print(f"身份命令未完成：{type(error).__name__}: {error}")
    # 识别了命令但执行失败也返回True，表示“已处理该输入”，并非报告设备操作成功。
    return True


def _device_request(parts):
    """把有限的控制台语法转成技能参数，不执行设备操作。"""
    # 由run_device_command调用；parts为split后的非空列表；返回(技能名,参数字典)。
    # 前导_表示内部辅助函数。它只翻译控制台语法，manager.execute才会执行技能。
    if parts[0] == "/voice":
        if len(parts) == 2 and parts[1] in ("off", "on", "status"):
            operation = {"off": "mute", "on": "unmute", "status": "status"}[parts[1]]
            # 先建立键→值字典，再用[parts[1]]查表；不是调用名为off/on的Python函数。
            return "control_volume", {"operation": operation}
        if len(parts) == 3 and parts[1] == "volume":
            # 终端的'50'是字符串，int将它转为整数；无法转换时抛ValueError给外层处理。
            return "control_volume", {"operation": "set", "value": int(parts[2])}
        raise ValueError("用法：/voice off | /voice on | /voice status | /voice volume 0..100")
    if parts[0] == "/device":
        if parts == ["/device", "status"]:
            return "configure_device", {"setting": "status"}
        if len(parts) == 2 and parts[1] in ("auto", "usb", "wifi"):
            return "configure_device", {"setting": "connection", "choice": parts[1]}
        raise ValueError("用法：/device status | /device auto | /device usb | /device wifi")
    if len(parts) == 2 and parts[1] in ("status", "list", "stop"):
        return "control_arms", {"operation": parts[1]}
    if len(parts) == 3 and parts[1] == "play":
        return "control_arms", {"operation": "play", "name": parts[2]}
    raise ValueError("用法：/arms status | /arms list | /arms stop | /arms play 动作名；使能与示教请在独立服务控制台操作。")


def run_device_command(command, manager, face):
    """调用前，主循环已确认键盘来源并丢弃上一轮录音。"""
    # 调用方：main键盘命令分支；manager为SkillManager，face为StackChanFaceEvents。
    parts = command.split()
    label = {"/voice": "StackChanTTS", "/device": "StackChan", "/arms": "Arms"}[parts[0]]
    try:
        name, arguments = _device_request(parts)
        # 解包元组：name取得技能名，arguments取得dict；交给同一个注册表校验并执行。
        result = manager.execute(name, arguments)
        print(f"[{label}] {result['message']}")
        if parts == ["/device", "status"] and result["success"]:
            # connection_status在face_events.py中定义；返回已有连接状态，不是创建新连接。
            print(face.connection_status())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"[{label}] 操作未完成：{error}")


def handle_session_command(command, npc, *, input_source, context):
    """取得本轮身份后调用；返回 exit、handled 或 None。"""
    # 调用方：main完成本轮身份识别之后；*规定后两项必须用input_source=、context=传入。
    # context是身份结果字典，npc是NpcTalkSkill；这里读取实际身份，而不是猜说话者。
    if command in ("退出", "exit", "quit"):
        if input_source == "text" or context["role"] == "owner":
            return "exit"
        print(f"{IDENTITY.name}: 我会继续接待。结束运行需要由主人操作。")
    elif command in ("查看证据", "/evidence"):
        print(npc.explain_last_response())
    elif command in ("清空记忆", "/clear"):
        npc.clear_history()
        print(f"{IDENTITY.name}: 当前会话的短期记忆已清空。")
    elif command == "清空长期记忆":
        # NPC 自己检查 local_operator 和 owner；语音身份不能授予本地删除权限。
        try:
            npc.clear_all_memory()
        except (OSError, sqlite3.Error, RuntimeError) as error:
            print(f"{IDENTITY.name}: 长期记忆未能清空：{error}")
        else:
            # try/except之后的else只在try没有抛异常时执行，因此此处才报告长期记忆清空。
            print(f"{IDENTITY.name}: 已清空保存的对话、长期记录和设备执行记录。")
    else:
        return None
    return "handled"


def print_help(*, speech_enabled):
    # main初始化会话后调用；只输出用法，不执行这些文字里提到的操作。
    # 形参speech_enabled是bool，决定是否显示语音输入下的Ctrl+C说明。
    print("可用指令：")
    print("  帮我拍张照 / 自动拍照 / 自动拍照，倒计时5秒")
    print("  声音大一点 / 小一点 / 音量调到百分之五十 / 静音 / 恢复声音 / 现在音量多少")
    print("  自动选择连接 / 切换到 Wi-Fi / 切换到 USB / 现在怎么连接的")
    print("  改成自动收音 / 改成触屏收音 / 关闭表演动作 / 现在放在桌上")
    print("  跟我打个招呼 / 点两下头 / 得意一下 / 看看左边 / 看看主人")
    print("  主人在你左边 / 主人在你右边：按机器人自身视角设置主人方向")
    print("  下一位来访者：结束当前访客临时会话，开始下一位")
    print("  清空记忆：清空当前互动对象的短期记忆")
    print("  记住：内容 / 忘记：关键词 / 清空长期记忆：在主人电脑控制台管理")
    print("  查看证据：查看当前会话上一轮的身份与回复依据")
    print("  /owner status：查看身份状态；/owner enroll face|voice|both：录入主人")
    print("  /owner mode auto|owner|visitor|once：自动、主人演示、访客、下一句主人")
    print("  /owner forget：删除身份模板；/visitor new：开始新访客会话")
    print("  /voice off | on | status | volume 0..100：电脑控制台播音控制（百分比）")
    print("  /device status | auto | usb | wifi：连接状态和连接偏好")
    print("  /arms status | list | stop | play 动作名：独立机械臂服务；使能/示教在 lerobot 控制台操作")
    print("  停止机械臂 / 急停：收到输入后直接停止；播音或规划期间请在另一个终端运行 python -m friston3.arms stop")
    print("  退出：在电脑控制台或由已确认的主人结束程序")
    if speech_enabled:
        print("  等待说话时按 Ctrl+C 进入电脑控制台；控制台空行返回麦克风。")
