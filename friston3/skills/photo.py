"""普通摄影 Skill：只调用摄像头 Adapter。"""

# class 定义一类对象的结构与方法；main._run_session 中的 PhotoSkill(camera)
# 才会创建一个实例。类名后加 () 是创建对象，不是开始拍照。
class PhotoSkill:
    """普通单张拍照技能。"""

    # 类属性由所有实例共同查找：SkillManager.register 读取 skill.name 作为字典键。
    name = "take_photo"
    # LLMPlanner.plan 读取 description 拼入工具说明；这段文本本身不执行任何动作。
    description = "立即拍一张照片；arguments必须为{}。"

    # @staticmethod 是 Python 内置装饰器：下面的方法不需要访问某一个拍照实例，
    # 所以 Python 不自动传 self；PhotoSkill.argument_schema() 与实例调用都可用。
    @staticmethod
    def argument_schema():
        # 调用方：planner.py 的 _schema_for_routes；无参数，返回一个 dict（字典）。
        # JSON Schema 是“要求模型输出什么形状”的描述数据，不是相机函数。
        # type=object 表示参数要是 JSON 对象；properties={} 表示没有可填写的字段；
        # required=[] 是空列表，表示没有必填字段；False 禁止再添加未知字段。
        return {"type": "object", "properties": {}, "required": [],
                "additionalProperties": False}

    @staticmethod
    def normalize_arguments(arguments, *, defaults=False):
        # 调用方：Planner.plan 和 SkillManager.execute；arguments 是待检查的参数字典。
        # * 是关键字专用参数分隔符：defaults 必须写成 defaults=True，而非位置参数。
        # defaults=False 是省略该实参时的默认值；拍照没有可默认的字段，故这里未使用它。
        # type(x) 是 Python 内置函数；is not dict 要求实际类型就是内置字典。
        # 非空字典在 if 中为真，所以后半句 arguments 会拒绝任何额外参数。
        if type(arguments) is not dict or arguments:
            # raise 抛出内置 ValueError（值不符合约定），立刻退出本函数，由调用方捕获。
            raise ValueError("普通拍照不接受参数")
        # return 结束函数，把新建的空字典交回调用方；不会改变原 arguments 对象。
        return {}

    # 初始化入口：main._run_session 创建 PhotoSkill(camera) 时由 Python 自动调用。
    # self 是正在初始化的 PhotoSkill 实例，camera 是 main 已创建好的相机对象。
    def __init__(self, camera):
        # 点号访问对象属性；把参数 camera 保存为 self.camera，后续 execute 才能用它。
        # 两个名字指向同一相机对象；这里没有复制相机或重新连接设备。
        self.camera = camera

    # 调用方：SkillManager.execute；无额外参数，返回 success/message/path 等结果字段。
    # 下游：hardware/camera.py 中 StackChanCamera.capture；拍照 I/O 从这条调用开始。
    def execute(self):
        # print 是 Python 内置输出函数；写到终端，方便看到技能真正开始执行的时刻。
        print("[PhotoSkill] 开始执行拍照技能")
        # try 包含可能抛异常的相机调用；except 负责把异常转换为技能统一的结果字典。
        try:
            # self.camera 查到前面保存的对象；.capture 找到它的方法；() 才执行该方法。
            # result 是这里首次定义的局部变量，保存 capture() 返回的 dict。
            result = self.camera.capture()
        except Exception as error:
            # Exception 是大多数普通异常的基类；as error 把异常对象绑定给局部变量 error。
            # f"...{error}" 是格式化字符串，会把 error 的文字说明插入花括号处。
            return {
                "success": False,
                "message": f"拍照发生异常：{error}",
            }

        # dict.get 是内置字典的方法：读取 success，缺少这个键则返回 None；不会抛 KeyError。
        # not 把真假反转；success=False/None 时进入失败分支。
        if not result.get("success"):
            return {
                "success": False,
                # get 的第二个参数是“键不存在时的默认值”，保留底层相机给出的错误原因。
                "message": result.get("error", "拍照失败"),
            }

        # result["path"] 按键读取；与 get 不同，缺少 path 会抛 KeyError，由 Manager 处理。
        # path 是保存位置（字符串），来自相机返回结果，不是这个技能临时猜出来的路径。
        path = result["path"]
        # 成功结果沿 SkillManager -> Agent -> main 返回；main 决定如何显示结果。
        return {
            "success": True,
            "path": path,
            "message": f"真实拍照完成！保存至：{path}",
        }
