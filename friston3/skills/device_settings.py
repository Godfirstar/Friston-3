"""设备设置的描述、模型参数格式和入口校验都在本文件维护。"""
from friston3.runtime_settings import SETTINGS
# SETTINGS是runtime_settings.py末尾创建的共享RuntimeSettings实例，不是普通配置常量字典。


class DeviceSettingsSkill:
    # main._run_session创建并注册本类；Planner读取说明，SkillManager最终调用execute。
    name = "configure_device"
    choices = {
        "connection": ("auto", "usb", "wifi"),
        "listening": ("touch", "vad"),
        "placement": ("shoulder", "desk"),
        "owner_side": ("left", "right", "none"),
        "gestures": ("on", "off"),
        "status": (),
    }
    # 类属性choices是“设置项→允许值元组”字典；status的空元组表示查询无需choice。
    description = (
        "读取/修改本机运行配置。arguments.setting必填：status查全部当前配置（无choice）、connection连接(auto/usb/wifi)、"
        "listening收音(touch触屏/vad自动)、placement位置(shoulder/desk)、"
        "owner_side主人方向(left/right/none，机器人视角)、gestures动作(on/off)。"
        "修改时括号内值填choice；方向不能猜。"
    )
    def __init__(self, face):
        # self是当前技能实例，face是main传来的StackChanFaceEvents；保存引用，不新开设备。
        self.face = face

    @classmethod
    def normalize_arguments(cls, arguments, *, defaults=False):
        # Planner/SkillManager调用；@classmethod自动传入类本身cls，故能读cls.choices。
        # arguments来自模型/控制台，返回规范化的新字典；*后defaults必须按名称传参。
        # set(arguments)取字典键，减去允许键集合后非空就说明有未知字段。
        if type(arguments) is not dict or set(arguments) - {"setting", "choice"}:
            raise ValueError("设备设置只接受 setting 和 choice")
        setting = arguments.get("setting", "status" if defaults else None)
        # dict.get第二项只在缺键时使用；defaults=True的直接调用可省略setting而视作查询。
        if type(setting) is not str or setting not in cls.choices:
            raise ValueError("未知设备设置")
        choice = arguments.get("choice", "")
        if type(choice) is not str:
            raise ValueError("设置值必须是字符串")
        if setting == "status":
            if choice:
                raise ValueError("查询设备状态不接受设置值")
        elif choice not in cls.choices[setting]:
            raise ValueError("该设置值无效，请明确要使用的设置")
        return {"setting": setting, **({"choice": choice} if choice else {})}
        # **在字典字面量中展开另一个字典；choice为空就展开{}，不加入choice键。

    @classmethod
    def argument_schema(cls):
        # LLMPlanner._schema_for_routes调用；返回JSON Schema描述数据，不执行设备操作。
        # 外层列表推导对choices每项生成一套规则；anyOf要求模型输出满足其中一个分支。
        # enum是允许的字符串值，required是必填键；查询分支省去choice，修改分支要求它。
        return {"anyOf": [
            {"type": "object", "properties": {
                "setting": {"type": "string", "enum": [setting]},
                **({"choice": {"type": "string", "enum": list(choices)}} if choices else {})},
             "required": ["setting", "choice"] if choices else ["setting"],
             "additionalProperties": False}
            for setting, choices in cls.choices.items()
        ]}

    def status_text(self):
        # execute的status分支调用；self.face.connection_status在face_events.py定义，返回状态dict。
        state = self.face.connection_status()
        # 查表把底层transport代码变成显示名称；.get的默认值只用于缺键/未匹配情况。
        transport = {"serial": "USB", "usb": "USB", "tcp": "Wi-Fi", "wifi": "Wi-Fi"}.get(
            state.get("transport"), "设备")
        link = f"已连接{transport}" if state.get("connected") else "正在等待栈酱连接"
        preference = {"auto": "自动选择连接", "usb": "固定使用USB", "wifi": "固定使用Wi-Fi"}[
            state.get("preference", "auto")]
        pending = "，当前操作结束后切换" if state.get("switch_pending") else ""
        # SETTINGS.get是RuntimeSettings的方法，并非直接读取某个配置文件；内部负责同步读取。
        listening = "触屏收音" if SETTINGS.get("mic_mode") == "touch" else "自动收音"
        placement = "肩上" if SETTINGS.get("placement") == "shoulder" else "桌面"
        gestures = "开启" if SETTINGS.get("gestures_enabled") else "关闭"
        side = SETTINGS.get("owner_side", "none")
        owner_side = "主人方向尚未设置" if side == "none" else f"按我的视角主人在{'左' if side == 'left' else '右'}边"
        return (f"{link}，{preference}{pending}。使用{listening}，位置设为{placement}，"
                f"{owner_side}，表演动作{gestures}。")

    def execute(self, setting="status", choice=""):
        # SkillManager校验参数后调用；结果沿Manager→Agent→main返回，main决定是否播报。
        # setting/choice是规范化的字符串；motion_stop初始None，只有需要取消头部动作时才填结果。
        motion_stop = None
        if setting == "connection":
            # set_connection_preference记录连接偏好；当前事务可能还未结束，不能据此断言已连上。
            state = self.face.set_connection_preference(choice)
            target = {"auto": "自动选择连接", "usb": "使用USB", "wifi": "使用Wi-Fi"}[choice]
            message = f"已设为{target}，在当前设备操作结束后切换。"
            # Selecting a transport does not prove that connection succeeded.
            return {"success": True, "message": message, "setting": setting,
                    "choice": choice, "connection": state, "speak_confirmation": True}
        if setting == "listening":
            # update在runtime_settings.py定义，更新共享设置并保存；这里不直接重建语音模型。
            SETTINGS.update(mic_mode=choice)
            message = "已改为触屏后收音。" if choice == "touch" else "已改为自动检测说话后收音。"
        elif setting == "placement":
            SETTINGS.update(placement=choice)
            message = "已记住，现在在你肩上。" if choice == "shoulder" else "已记住，现在放在桌面。"
        elif setting == "owner_side":
            SETTINGS.update(owner_side=choice)
            message = ("已清除主人方向设置。" if choice == "none" else
                       f"已记住，按我的视角，主人在{'左' if choice == 'left' else '右'}边。")
        elif setting == "gestures":
            # choice == 'on'产生bool，再按关键字参数传入update，不把'on'字符串当布尔值使用。
            SETTINGS.update(gestures_enabled=choice == "on")
            message = "已开启表演动作。" if choice == "on" else "已关闭表演动作。"
        else:
            message = self.status_text()
        if setting == "placement" or (setting == "gestures" and choice == "off"):
            # 此处取消栈酱头部动作；机械臂的对话动作取消还由main收到结果后交给ArmIntegration。
            # Persist the choice first: a failed stop must not re-enable motion.
            # Profile changes also discard an older, potentially wider preset.
            try:
                motion_stop = self.face.cancel_motion()
                # 方法可能返回拒绝/未确认等状态；“设置已保存”与“动作已停止”分别记录。
            except Exception as error:
                motion_stop = {"status": "unconfirmed", "error": str(error)}
            if motion_stop.get("status") in ("rejected", "unconfirmed"):
                message += "当前动作停止尚未确认。"
        return {"success": True, "message": message, "setting": setting,
                "choice": choice, "speak_confirmation": True,
                **({"motion_stop": motion_stop} if motion_stop is not None else {})}
