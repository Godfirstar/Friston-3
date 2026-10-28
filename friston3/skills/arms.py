"""机械臂技能只开放已定义的操作；模型和控制台共用入口校验。"""
import re


class ArmsControlSkill:
    name = "control_arms"
    operations = ("status", "list", "stop", "play")
    description = (
        "机械臂服务。operation必填：status查连接/任务，list列动作，stop停止并撤销使能，"
        "play播放已示教name。仅play必填name，其余不带。"
        "常用rest休息、wave挥手、welcome欢迎、point_pickup取物指示、shrug摊手；"
        "须服务已使能且动作存在。使能/示教/自动开关/任意角度未开放。"
        "明确机械臂操作用本技能，闲聊附带动作才用presentation.arm_gesture。"
    )
    def __init__(self, integration):
        self.integration = integration

    @classmethod
    def normalize_arguments(cls, arguments, *, defaults=False):
        if type(arguments) is not dict or set(arguments) - {"operation", "name"}:
            raise ValueError("机械臂操作只接受 operation 和 name")
        operation = arguments.get("operation", "status" if defaults else None)
        if type(operation) is not str or operation not in cls.operations:
            raise ValueError("机械臂只开放 status、list、stop 和 play")
        name = arguments.get("name", "")
        if operation == "play":
            if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name) is None:
                raise ValueError("播放需要合法的已示教动作名：小写字母开头，最长48字符，只含字母数字_-。")
        elif "name" in arguments:
            raise ValueError("查询、列表和停止不接受动作名")
        return {"operation": operation, **({"name": name} if operation == "play" else {})}

    @classmethod
    def argument_schema(cls):
        return {"anyOf": [
            {"type": "object", "properties": {
                "operation": {"type": "string", "enum": ["status", "list", "stop"]}},
             "required": ["operation"], "additionalProperties": False},
            {"type": "object", "properties": {
                "operation": {"type": "string", "enum": ["play"]},
                "name": {"type": "string", "pattern": "^[a-z][a-z0-9_-]{0,47}$",
                         "minLength": 1, "maxLength": 48}},
             "required": ["operation", "name"], "additionalProperties": False},
        ]}

    def execute(self, operation="status", name=""):
        return self.integration.perform(operation, name=name)
