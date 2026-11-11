"""机械臂技能只开放已定义的操作；模型和控制台共用入口校验。"""
# re 是 Python 标准库正则表达式模块，下面用 re.fullmatch 检查动作名格式。
import re


class ArmsControlSkill:
    # main._run_session 用 ArmsControlSkill(arms) 创建实例并 register 到技能管理器。
    # 类属性 name 是注册键；operations 是 tuple（不可变序列），列出已接入的操作名。
    name = "control_arms"
    operations = ("status", "list", "stop", "play", "task")
    # description 是给 Planner 的工具说明字符串；相邻字符串/括号仅组织代码，不执行操作。
    description = (
        'task取放：instruction照抄整句请求（物体/目的地/顺序），多步 一次调用；arm可选。status查任务，list列示教，stop停止，play(name)播已使能的示教：rest/wave/welcome/point_pickup/shrug。结束≠抓取成功。'
    )
    def __init__(self, integration):
        # 构造时 Python 自动传当前实例为 self；integration   是 main 创建的 ArmIntegration。
        # 点号用于给实例保存属性：之后 execute 用同一个对象，不在技能里另开串口。
        self.integration = integration

    # @classmethod 是内置装饰器：调用本方法时自动传入类作为 cls，而非实例 self。
    # 因为下面要读 cls.operations，这里使用类方法；无需访问某一次连接的 integration。
    @classmethod
    def normalize_arguments(cls, arguments, *, defaults=False):
        # 调用方：Planner.plan/_plan_resident_arms 和 SkillManager.execute；返回规范化 dict。
        # arguments 是外部参数 dict；* 后的 defaults 必须用名字传入，省略时为 False。
        # type(x) is not dict 要求实际是字典；set(dict) 得到所有键构成的集合。
        # 集合 A - B 是差集：剩余的就是未知字段；非空集合在 if 中为真，触发报错。
        if type(arguments) is not dict or set(arguments) - {"operation", "name", "instruction", "arm"}:
            # raise ValueError 抛出“参数值不符合约定”，由 Planner 或 Manager 转成失败结果。
            raise ValueError("机械臂操作只接受 operation、name、instruction 和 arm")
        # operation 在这里首次定义：从字典读取操作名 str；dict.get 缺键时返回第二个实参。
        # 条件表达式在 defaults=True 时给默认 status，否则给 None 表示未提供必要值。
        operation = arguments.get("operation", "status" if defaults else None)
        # not in 判断值不在允许的 tuple 中；cls.operations 是本类前面定义的操作集合。
        if type(operation) is not str or operation not in cls.operations:
            raise ValueError("机械臂支持 status、list、stop、play 和 task")
        if operation == "task":
            # 'name' in arguments 检查“键是否出现”，即使 name 值为空也算出现。
            if "name" in arguments:
                raise ValueError("任务不接受已示教动作名")
            # instruction 是原始任务文本 str 或 None；它稍后作为自然语言交给策略执行链路。
            instruction = arguments.get("instruction")
            # or 短路：不是字符串时不再调用 strip；strip 返回去首尾空白后的字符串副本。
            if type(instruction) is not str or not instruction.strip():
                raise ValueError("任务需要完整的自然语言 instruction")
            # arm 是可选的 left/right 字符串，缺键为 None，表示让下游使用自己的选择方式。
            arm = arguments.get("arm")
            if "arm" in arguments and arm not in ("left", "right"):
                raise ValueError("arm 只能是 left 或 right")
            # ** 在字典中展开另一字典的键值：有 arm 才加 arm 字段；否则展开空字典 {}。
            # return 立即结束函数，后面的 play/status 校验不再执行。
            return {"operation": operation, "instruction": instruction,
                    **({"arm": arm} if arm is not None else {})}
        # 到这里必定不是 task，因而拒绝只属于 task 的字段，避免参数被悄悄忽略。
        if "instruction" in arguments or "arm" in arguments:
            raise ValueError("只有 task 接受 instruction 和 arm")
        # name 是已录制轨迹的标识 str，缺少时用空串；它不是任意自然语言任务。
        name = arguments.get("name", "")
        if operation == "play":
            # re.fullmatch 必须整串匹配，成功返回 Match 对象，失败返回 None。
            # r 前缀是原始字符串；[a-z] 要求首字符小写字母，后面 {0,47} 允许再有 0~47 字符。
            if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name) is None:
                raise ValueError("播放需要合法的已示教动作名：小写字母开头，最长48字符，只含字母数字_-。")
        elif "name" in arguments:
            raise ValueError("查询、列表和停止不接受动作名")
        # 查询/列表/停止只返回 operation；play 才额外返回 name。没有在此操作任何设备。
        return {"operation": operation, **({"name": name} if operation == "play" else {})}

    @classmethod
    def argument_schema(cls):
        # 调用方：Planner._schema_for_routes；无外部参数，返回 JSON Schema 描述字典。
        # cls 因为 @classmethod 自动传入，本方法暂不读取它，保留与同类技能一致的接口。
        # anyOf 后的 [] 是分支列表：模型参数只要满足 task/查询组/play 三种形状之一即可。
        # properties 声明字段；required 列必填字段；additionalProperties=False 拒绝未知字段。
        return {"anyOf": [
            # task 分支：instruction 必填，minLength=1；arm 不在 required 中，因此可省略。
            {"type": "object", "properties": {
                "operation": {"type": "string", "enum": ["task"]},
                "instruction": {"type": "string", "minLength": 1},
                "arm": {"type": "string", "enum": ["left", "right"]}},
             "required": ["operation", "instruction"], "additionalProperties": False},
            # 查询组：enum 表示 operation 只能选 status/list/stop，不需要名字或任务文本。
            {"type": "object", "properties": {
                "operation": {"type": "string", "enum": ["status", "list", "stop"]}},
             "required": ["operation"], "additionalProperties": False},
            # play 分支：必须给 name；pattern/minLength/maxLength 向模型声明名称格式。
            {"type": "object", "properties": {
                "operation": {"type": "string", "enum": ["play"]},
                "name": {"type": "string", "pattern": "^[a-z][a-z0-9_-]{0,47}$",
                         "minLength": 1, "maxLength": 48}},
             "required": ["operation", "name"], "additionalProperties": False},
        ]}

    def execute(self, operation="status", name="", instruction="", arm=None):
        # 调用方：SkillManager.execute 用 **checked 把校验后的字典展开为这些命名实参。
        # operation/name/instruction/arm 分别对应动作种类、轨迹名、任务原文、可选左右臂。
        # self.integration 是 __init__ 保存的对象；perform 定义在 arms/integration.py。
        # perform 负责请求服务并补充可读结果；返回 dict，直接沿 Manager/Agent 交给 main。
        # task 的 accepted/running 只表示任务已接收/正在运行，不能据此断言实际抓取成功。
        return self.integration.perform(operation, name=name, instruction=instruction, arm=arm)
