"""直接控制指令与输入切换，不调用模型规划。

设备命令仅由键盘入口调用；语音会话命令必须先取得本轮身份。
这里只解释指令，设备参数仍交给 SkillManager 统一校验。
"""
import sqlite3

from friston3.npc.identity import IDENTITY


def discard_input(owner, speech_input):
    """丢弃上一轮身份观察和录音，避免键盘操作沿用旧说话人的信息。"""
    owner.invalidate()
    if speech_input is not None:
        speech_input.clear_last_audio()


def read_instruction(owner, speech_input):
    if speech_input is None:
        return input("\n你: ").strip(), "text"
    try:
        return speech_input.read().strip(), "speech"
    except KeyboardInterrupt:
        # 调度器会取消正在等待的录音；被中断的录音不能确认键盘操作者身份。
        discard_input(owner, speech_input)
        print("\n已进入电脑控制台。输入命令或一句话；直接按 Enter 返回麦克风。")
        return input("电脑控制台: ").strip(), "text"


def new_visitor(owner, speech_input, npc):
    discard_input(owner, speech_input)
    # 交接会结束主人演示模式，只清理上一位匿名访客的临时会话。
    profile = owner.handover_visitor()
    npc.set_interlocutor({
        "role": "visitor", "profile_id": profile, "source": "visitor_handover",
        "authenticated": False, "local_operator": False,
        "reason": "现场明确开始接待下一位来访者",
    })
    print(f"{IDENTITY.name}: 已开始新访客会话。")


def handle_identity_command(command, owner, speech_input, npc):
    """只从键盘入口调用；这些身份管理能力不登记为模型技能。"""
    words = command.split()
    if not words or words[0] not in ("/owner", "/visitor"):
        return False
    discard_input(owner, speech_input)
    try:
        if words == ["/owner", "status"]:
            print(owner.status_text())
        elif words[:2] == ["/owner", "enroll"] and len(words) in (2, 3):
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
        discard_input(owner, speech_input)
        print(f"身份命令未完成：{type(error).__name__}: {error}")
    return True


def _device_request(parts):
    """把有限的控制台语法转成技能参数，不执行设备操作。"""
    if parts[0] == "/voice":
        if len(parts) == 2 and parts[1] in ("off", "on", "status"):
            operation = {"off": "mute", "on": "unmute", "status": "status"}[parts[1]]
            return "control_volume", {"operation": operation}
        if len(parts) == 3 and parts[1] == "volume":
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
    parts = command.split()
    label = {"/voice": "StackChanTTS", "/device": "StackChan", "/arms": "Arms"}[parts[0]]
    try:
        name, arguments = _device_request(parts)
        result = manager.execute(name, arguments)
        print(f"[{label}] {result['message']}")
        if parts == ["/device", "status"] and result["success"]:
            print(face.connection_status())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"[{label}] 操作未完成：{error}")


def handle_session_command(command, npc, *, input_source, context):
    """取得本轮身份后调用；返回 exit、handled 或 None。"""
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
            print(f"{IDENTITY.name}: 已清空保存的对话、长期记录和设备执行记录。")
    else:
        return None
    return "handled"


def print_help(*, speech_enabled):
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
