"""设备设置的描述、模型参数格式和入口校验都在本文件维护。"""
from friston3.runtime_settings import SETTINGS


class DeviceSettingsSkill:
    name = "configure_device"
    choices = {
        "connection": ("auto", "usb", "wifi"),
        "listening": ("touch", "vad"),
        "placement": ("shoulder", "desk"),
        "owner_side": ("left", "right", "none"),
        "gestures": ("on", "off"),
        "status": (),
    }
    description = (
        "读取/修改本机运行配置。arguments.setting必填：status查全部当前配置（无choice）、connection连接(auto/usb/wifi)、"
        "listening收音(touch触屏/vad自动)、placement位置(shoulder/desk)、"
        "owner_side主人方向(left/right/none，机器人视角)、gestures动作(on/off)。"
        "修改时括号内值填choice；方向不能猜。"
    )
    def __init__(self, face):
        self.face = face

    @classmethod
    def normalize_arguments(cls, arguments, *, defaults=False):
        if type(arguments) is not dict or set(arguments) - {"setting", "choice"}:
            raise ValueError("设备设置只接受 setting 和 choice")
        setting = arguments.get("setting", "status" if defaults else None)
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

    @classmethod
    def argument_schema(cls):
        return {"anyOf": [
            {"type": "object", "properties": {
                "setting": {"type": "string", "enum": [setting]},
                **({"choice": {"type": "string", "enum": list(choices)}} if choices else {})},
             "required": ["setting", "choice"] if choices else ["setting"],
             "additionalProperties": False}
            for setting, choices in cls.choices.items()
        ]}

    def status_text(self):
        state = self.face.connection_status()
        transport = {"serial": "USB", "usb": "USB", "tcp": "Wi-Fi", "wifi": "Wi-Fi"}.get(
            state.get("transport"), "设备")
        link = f"已连接{transport}" if state.get("connected") else "正在等待栈酱连接"
        preference = {"auto": "自动选择连接", "usb": "固定使用USB", "wifi": "固定使用Wi-Fi"}[
            state.get("preference", "auto")]
        pending = "，当前操作结束后切换" if state.get("switch_pending") else ""
        listening = "触屏收音" if SETTINGS.get("mic_mode") == "touch" else "自动收音"
        placement = "肩上" if SETTINGS.get("placement") == "shoulder" else "桌面"
        gestures = "开启" if SETTINGS.get("gestures_enabled") else "关闭"
        side = SETTINGS.get("owner_side", "none")
        owner_side = "主人方向尚未设置" if side == "none" else f"按我的视角主人在{'左' if side == 'left' else '右'}边"
        return (f"{link}，{preference}{pending}。使用{listening}，位置设为{placement}，"
                f"{owner_side}，表演动作{gestures}。")

    def execute(self, setting="status", choice=""):
        motion_stop = None
        if setting == "connection":
            state = self.face.set_connection_preference(choice)
            target = {"auto": "自动选择连接", "usb": "使用USB", "wifi": "使用Wi-Fi"}[choice]
            message = f"已设为{target}，在当前设备操作结束后切换。"
            # Selecting a transport does not prove that connection succeeded.
            return {"success": True, "message": message, "setting": setting,
                    "choice": choice, "connection": state, "speak_confirmation": True}
        if setting == "listening":
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
            SETTINGS.update(gestures_enabled=choice == "on")
            message = "已开启表演动作。" if choice == "on" else "已关闭表演动作。"
        else:
            message = self.status_text()
        if setting == "placement" or (setting == "gestures" and choice == "off"):
            # Persist the choice first: a failed stop must not re-enable motion.
            # Profile changes also discard an older, potentially wider preset.
            try:
                motion_stop = self.face.cancel_motion()
            except Exception as error:
                motion_stop = {"status": "unconfirmed", "error": str(error)}
            if motion_stop.get("status") in ("rejected", "unconfirmed"):
                message += "当前动作停止尚未确认。"
        return {"success": True, "message": message, "setting": setting,
                "choice": choice, "speak_confirmation": True,
                **({"motion_stop": motion_stop} if motion_stop is not None else {})}
