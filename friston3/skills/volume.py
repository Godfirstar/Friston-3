"""音量参数只在入口校验；execute 使用校验后的百分比执行业务。"""


class VolumeControlSkill:
    """用户使用 0..100 百分比，设备继续使用 0..160 的音量值。

    这里只修改播音配置。已有的设备连接会在下次播音时发送音量。
    """

    name = "control_volume"
    # main创建技能并注册；name是注册表键，description供Planner读；operations是允许操作的元组。
    operations = ("set", "increase", "decrease", "mute", "unmute", "status")
    description = (
        "读取/修改本机播音配置（音量百分比、静音），不测环境分贝。operation必填："
        "status查当前状态（无value），set设百分比（必填value），"
        "increase/decrease调大/小（value默认10个百分点），mute/unmute静音/恢复（无value）。value为0..100整数。"
    )
    def __init__(self, voice):
        # voice来自main创建的AgentSpeechOutput（voice_output.py）；self.voice保存同一实例引用。
        self.voice = voice

    @classmethod
    def normalize_arguments(cls, arguments, *, defaults=False):
        """拒绝未知字段，补齐相对调节的默认步长；不使用特殊数值作占位符。"""
        # Planner/Manager调用；@classmethod传类本身cls，使用cls.operations检查已知操作。
        # set(dict)得到键集合，集合相减找出未知字段；输入必须为普通dict。
        if type(arguments) is not dict or set(arguments) - {"operation", "value"}:
            raise ValueError("音量操作只接受 operation 和 value")
        operation = arguments.get("operation", "status" if defaults else None)
        if type(operation) is not str or operation not in cls.operations:
            raise ValueError("不支持的音量操作")
        value = arguments.get("value")
        # get缺键返回None；'value' in arguments是在查键是否存在，能区分缺键和显式给None。
        if "value" in arguments and (type(value) is not int or not 0 <= value <= 100):
            raise ValueError("音量百分比/调节百分点必须为 0 至 100 的整数")
        if operation == "set" and "value" not in arguments:
            raise ValueError("设置音量需要明确的百分比")
        if operation in ("mute", "unmute", "status") and "value" in arguments:
            raise ValueError("静音、恢复和查询不接受音量数值")
        checked = {"operation": operation}
        # 新建结果字典，避免原地改写调用者的arguments；相对调节未填value时取10个百分点。
        if operation in ("set", "increase", "decrease"):
            checked["value"] = arguments.get("value", 10)
        return checked

    @staticmethod
    def argument_schema():
        # @staticmethod不自动接收self/cls；Planner调用它获取模型输出格式说明。
        percent = {"type": "integer", "minimum": 0, "maximum": 100}
        # percent是复用的Schema字典，描述允许的整数值，并非正在设置的真实音量。
        # anyOf的三个分支分别用于绝对设置、相对增减、无数值操作；required控制必填键。
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
        # execute/status_text调用；读取AgentSpeechOutput属性并组装新dict，不测麦克风分贝。
        level = self.voice.volume
        # level为设备0..160刻度；除以160再乘100得到用户界面的百分比。
        return {
            "volume": level,
            "volume_percent": level * 100 / 160,
            "speech_enabled": self.voice.enabled,
            "muted": not self.voice.enabled or level == 0,
        }

    def status_text(self):
        # 将state()的字典转成文字；f字符串中的:g选择简洁数字格式，例如50而非50.0。
        state = self.state()
        percent = f"{state['volume_percent']:g}"
        return (f"{'静音' if state['muted'] else '启用'}；音量={percent}%"
                f"（设备值 {state['volume']}/160）")

    def execute(self, operation="status", value=None):
        # SkillManager调用；前面normalize_arguments已校验。此函数修改语音输出对象配置，
        # 下一次真正播音才会向设备应用音量；返回success不表示已经发出声音。
        # configure_output还会持久化SETTINGS，在启用语音且设备功能启用时可启动TTS worker预热；
        # 因此这次设置并非只有内存赋值，但worker启动也不代表已经合成或播放这句确认文字。
        if operation == "mute":
            self.voice.configure_output(enabled=False, volume=0)
        elif operation == "unmute":
            # unmute在voice_output.py定义，恢复其维护的播音状态，而不是这里猜一个固定音量。
            self.voice.unmute()
        elif operation == "set":
            # value为0..100；round映射到设备整数刻度，value>0产生是否启用的布尔值。
            self.voice.configure_output(enabled=value > 0, volume=round(value * 160 / 100))
        elif operation in ("increase", "decrease"):
            step = 10 if value is None else value
            # step指“百分点”，例如50%加10得到60%；不是把当前值乘以1.1。
            if step:
                delta = round(step * 160 / 100)
                level = max(0, min(160, self.voice.volume + (delta if operation == "increase" else -delta)))
                # 先按操作加/减delta，min限定上端，max限定下端；这是已有设备刻度映射。
                enabled = level > 0 and (self.voice.enabled or operation == "increase")
                self.voice.configure_output(enabled=enabled, volume=level)
                # 调低音量不会自动恢复已被主动关闭的播音。

        state = self.state()
        # 操作后重新读取真实配置，再据此生成message，避免仅根据请求数值声称结果。
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
            # **把state的各键展开到结果；speak_confirmation由main读取，静音后不再要求播报。
            "speak_confirmation": not state["muted"],
        }
