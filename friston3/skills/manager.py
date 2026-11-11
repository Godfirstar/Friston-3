"""登记技能、调用技能自己的参数校验，再附上真实执行来源。"""


# main._run_session 使用 SkillManager() 创建一个管理器实例；它是普通 Python 类。
# 类定义下面的方法；每个实例另外保存自己的 skills 字典，不是文件级全局注册表。
class SkillManager:
    """技能各自解释参数；管理器只负责分发和处理执行异常。"""

    def __init__(self):
        # __init__ 由 SkillManager() 自动调用；self 指向这次新建的管理器。
        # self.skills 左侧的点号表示“这个对象的 skills 属性”；{} 创建空字典 dict。
        # 注册表：名称 → 技能实例；这里没有自动扫描文件或加载插件的机制。
        self.skills = {}

    def register(self, skill):
        # 输入 skill 是已创建的对象，例如 photo.py 的 PhotoSkill 实例；不是类名字符串。
        # 调用 manager.register(photo) 时，Python 自动传 manager 作为 self。
        # main._run_session 调用。约定对象具有相应方法，无需强制继承共同父类。
        # 同名会覆盖旧实例，这是当前字典赋值行为，并非显式重复注册检查。
        # skill.name 读取技能的类属性；字典[键] = 值 将整个对象保存到对应键下。
        self.skills[skill.name] = skill
        # 函数走到结尾没有写 return 时，Python 自动返回 None；调用方无需接收结果。

    def execute(self, name, arguments=None, *, context=None):
        """context 由 Agent 传入可信的单轮上下文，不属于模型可填的 arguments。"""
        # 调用方：Agent.handle、console.run_device_command 和 main 中的立即停止分支。
        # name 是技能名 str；arguments 是 dict 或 None；context 是 DialogueTurn 或 None。
        # None 表示未提供值；参数中 * 之后的 context 必须写 context=... 才能传入。
        # 下游是选中技能的 normalize_arguments/execute；本函数返回结果 dict，交给调用方。
        # isinstance 是 Python 内置类型检查；not 对布尔结果取反。
        if not isinstance(name, str):
            # return 立即结束函数；此处没找到/调用技能，因此不能标记 executed_skill。
            return {"success": False, "message": "技能名称必须是字符串"}
        # get 查不到时返回 None；skills[name] 则会抛 KeyError。
        # skill 是本函数局部变量，来自本实例的注册表；对象类型由 name 对应条目决定。
        skill = self.skills.get(name)
        # is None 比较的是是否为同一个 None 单例，不是调用技能的相等运算。
        if skill is None:
            # f 前缀允许 {name} 插入变量值；仅构造失败文本，不会执行这个名字。
            return {"success": False, "message": f"找不到技能：{name}"}
        try:
            # 本地直接调用仍可省略 status；模型规划使用同一校验的严格模式。
            # 具体参数规则由技能维护；控制台可绕过 Planner，因此执行前仍需校验。
            # 条件表达式 A if 条件 else B：没传 arguments 才新建 {}，否则保留实际参数。
            # .normalize_arguments 根据实际对象找到方法，例如 photo.py / arms.py 的同名方法。
            # checked 是校验方法返回的规范化字典；defaults=True 允许技能采用已有本地默认值。
            checked = skill.normalize_arguments({} if arguments is None else arguments, defaults=True)
        except (TypeError, ValueError) as error:
            # 圆括号中的异常类型元组表示“捕获任意一种”；as error 保存这次异常。
            # str(error) 用内置 str 转成错误文本；返回后不会继续执行下面的技能。
            return {"success": False, "message": str(error)}
        print(f"[SkillManager] 调用: {name}, 参数: {checked}")
        try:
            # **checked 把字典展开为关键字实参，例如 {"operation": "status"}。
            # 例如 skill.execute(**{"operation": "status"}) 等价于 execute(operation="status")。
            # 这是在调用处展开；若定义处写 def f(**kwargs)，含义才是把多余命名实参收成字典。
            # 只有传入 context 时才附加它；正常 Agent 链路只有 NPC 消费此参数。
            # 条件表达式只执行选中的一支，不会先执行两次再选结果。
            # result 是本次技能返回值，拍照是完成结果，机械臂 task 也可能仅表示接收任务。
            result = (skill.execute(**checked, context=context) if context is not None
                      else skill.execute(**checked))
            # 技能约定返回 dict；不满足约定也要向上返回结构化失败，避免 main 访问字段报错。
            if not isinstance(result, dict):
                return {"success": False, "message": "技能返回格式错误", "executed_skill": name}
            # 只有真正调用技能之后才标注，参数错误不算设备执行。
            # executed_skill 表示调用过谁，不代表物理成功；还需查看 success/status。
            # 后写的同名键覆盖前面的值，执行来源因此由管理器统一提供。
            # 此处 **result 在字典字面量里展开键值，与上面调用函数时展开实参是两种用途。
            # 新建外层字典，不直接给技能自己的 result 增加键；嵌套对象仍可能共享引用。
            return {**result, "executed_skill": name}
        except Exception as error:
            # 在调度接口把技能异常转为统一结果，错误信息保留供主循环显示。
            return {"success": False, "message": f"技能执行异常：{error}", "executed_skill": name}
