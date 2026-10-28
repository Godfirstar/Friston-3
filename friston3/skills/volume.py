"""音量参数只在入口校验；execute 使用校验后的百分比执行业务。"""


class VolumeControlSkill:
    """用户使用 0..100 百分比，设备继续使用 0..160 的音量值。

    这里只修改播音配置。已有的设备连接会在下次播音时发送音量。
    """

    name = "control_volume"
    operations = ("set", "increase", "decrease", "mute", "unmute", "status")
    description = (
        "读取/修改本机播音配置（音量百分比、静音），不测环境分贝。operation必填："
        "status查当前状态（无value），set设百分比（必填value），"
        "increase/decrease调大/小（value默认10个百分点），mute/unmute静音/恢复（无value）。value为0..100整数。"
    )
    def __init__(self, voice):
        self.voice = voice

    @classmethod
    def normalize_arguments(cls, arguments, *, defaults=False):
        """拒绝未知字段，补齐相对调节的默认步长；不使用特殊数值作占位符。"""
        if type(arguments) is not dict or set(arguments) - {"operation", "value"}:
            raise ValueError("音量操作只接受 operation 和 value")
        operation = arguments.get("operation", "status" if defaults else None)
        if type(operation) is not str or operation not in cls.operations:
            raise ValueError("不支持的音量操作")
        value = arguments.get("value")
        if "value" in arguments and (type(value) is not int or not 0 <= value <= 100):
            raise ValueError("音量百分比/调节百分点必须为 0 至 100 的整数")
        if operation == "set" and "value" not in arguments:
            raise ValueError("设置音量需要明确的百分比")
        if operation in ("mute", "unmute", "status") and "value" in arguments:
            raise ValueError("静音、恢复和查询不接受音量数值")
        checked = {"operation": operation}
        if operation in ("set", "increase", "decrease"):
            checked["value"] = arguments.get("value", 10)
        return checked

    @staticmethod
    def argument_schema():
        percent = {"type": "integer", "minimum": 0, "maximum": 100}
        return {"anyOf": [
            {"type": "object", "properties": {
                "operation": {"type": "string", "enum": ["set"]}, "value": percent},
             "required": ["operation", "value"], "additionalProperties": False},
            {"type": "object", "properties": {
                "operation": {"type": "string", "enum": ["increase", "decrease"]}, "value": percent},
             "required": ["operation"], "additionalProperties": False},
            {"type": "object", "properties": {
                "operation": {"type": "string", "enum": ["mute", "unmute", "status"]}},
             "required": ["operation"], "additionalProperties": False},
        ]}

    def state(self):
        level = self.voice.volume
        return {
            "volume": level,
            "volume_percent": level * 100 / 160,
            "speech_enabled": self.voice.enabled,
            "muted": not self.voice.enabled or level == 0,
        }

    def status_text(self):
        state = self.state()
        percent = f"{state['volume_percent']:g}"
        return (f"{'静音' if state['muted'] else '启用'}；音量={percent}%"
                f"（设备值 {state['volume']}/160）")

    def execute(self, operation="status", value=None):
        if operation == "mute":
            self.voice.configure_output(enabled=False, volume=0)
        elif operation == "unmute":
            self.voice.unmute()
        elif operation == "set":
            self.voice.configure_output(enabled=value > 0, volume=round(value * 160 / 100))
        elif operation in ("increase", "decrease"):
            step = 10 if value is None else value
            if step:
                delta = round(step * 160 / 100)
                level = max(0, min(160, self.voice.volume + (delta if operation == "increase" else -delta)))
                enabled = level > 0 and (self.voice.enabled or operation == "increase")
                self.voice.configure_output(enabled=enabled, volume=level)
                # 调低音量不会自动恢复已被主动关闭的播音。

        state = self.state()
        percent = f"{state['volume_percent']:g}"
        if state["muted"]:
            message = "当前已静音。" if operation == "status" else "已静音。"
            if state["volume"]:
                message += f"保存的播音音量是百分之{percent}。"
        elif operation == "status":
            message = f"当前播音音量是百分之{percent}。"
        elif operation == "unmute":
            message = f"已恢复播音，音量为百分之{percent}。"
        else:
            message = f"播音音量已设为百分之{percent}。"
        return {
            "success": True, "message": message,
            "operation": operation, **state,
            "speak_confirmation": not state["muted"],
        }
