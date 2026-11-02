"""登记技能、调用技能自己的参数校验，再附上真实执行来源。"""


class SkillManager:
    """技能各自解释参数；管理器只负责分发和处理执行异常。"""

    def __init__(self):
        self.skills = {}

    def register(self, skill):
        self.skills[skill.name] = skill

    def execute(self, name, arguments=None, *, context=None):
        """context 由 Agent 传入可信的单轮上下文，不属于模型可填的 arguments。"""
        if not isinstance(name, str):
            return {"success": False, "message": "技能名称必须是字符串"}
        skill = self.skills.get(name)
        if skill is None:
            return {"success": False, "message": f"找不到技能：{name}"}
        try:
            # 本地直接调用仍可省略 status；模型规划使用同一校验的严格模式。
            checked = skill.normalize_arguments({} if arguments is None else arguments, defaults=True)
        except (TypeError, ValueError) as error:
            return {"success": False, "message": str(error)}
        print(f"[SkillManager] 调用: {name}, 参数: {checked}")
        try:
            result = (skill.execute(**checked, context=context) if context is not None
                      else skill.execute(**checked))
            if not isinstance(result, dict):
                return {"success": False, "message": "技能返回格式错误", "executed_skill": name}
            # 只有真正调用技能之后才标注，参数错误不算设备执行。
            return {**result, "executed_skill": name}
        except Exception as error:
            return {"success": False, "message": f"技能执行异常：{error}", "executed_skill": name}
