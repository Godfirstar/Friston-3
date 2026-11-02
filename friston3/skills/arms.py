"""机械臂技能只开放已定义的操作；模型和控制台共用入口校验。"""
import re


class ArmsControlSkill:
    name = "control_arms"
    operations = ("status", "list", "stop", "play", "task")
    description = (
        'task取放：instruction照抄整句请求（物体/目的地/顺序），多步一次调用；arm可选。status查任务，list列示教，stop停止，play(name)播已使能的示教：rest/wave/welcome/point_pickup/shrug。结束≠抓取成功。'
    )
    def __init__(self, integration):
        self.integration = integration

    @classmethod
    def normalize_arguments(cls, arguments, *, defaults=False):
        if type(arguments) is not dict or set(arguments) - {"operation", "name", "instruction", "arm"}:
            raise ValueError("机械臂操作只接受 operation、name、instruction 和 arm")
        operation = arguments.get("operation", "status" if defaults else None)
        if type(operation) is not str or operation not in cls.operations:
            raise ValueError("机械臂支持 status、list、stop、play 和 task")
        if operation == "task":
            if "name" in arguments:
                raise ValueError("任务不接受已示教动作名")
            instruction = arguments.get("instruction")
            if type(instruction) is not str or not instruction.strip():
                raise ValueError("任务需要完整的自然语言 instruction")
            arm = arguments.get("arm")
            if "arm" in arguments and arm not in ("left", "right"):
                raise ValueError("arm 只能是 left 或 right")
            return {"operation": operation, "instruction": instruction,
                    **({"arm": arm} if arm is not None else {})}
        if "instruction" in arguments or "arm" in arguments:
            raise ValueError("只有 task 接受 instruction 和 arm")
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
                "operation": {"type": "string", "enum": ["task"]},
                "instruction": {"type": "string", "minLength": 1},
                "arm": {"type": "string", "enum": ["left", "right"]}},
             "required": ["operation", "instruction"], "additionalProperties": False},
            {"type": "object", "properties": {
                "operation": {"type": "string", "enum": ["status", "list", "stop"]}},
             "required": ["operation"], "additionalProperties": False},
            {"type": "object", "properties": {
                "operation": {"type": "string", "enum": ["play"]},
                "name": {"type": "string", "pattern": "^[a-z][a-z0-9_-]{0,47}$",
                         "minLength": 1, "maxLength": 48}},
             "required": ["operation", "name"], "additionalProperties": False},
        ]}

    def execute(self, operation="status", name="", instruction="", arm=None):
        return self.integration.perform(operation, name=name, instruction=instruction, arm=arm)
