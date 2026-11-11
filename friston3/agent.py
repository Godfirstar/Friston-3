"""输入规范化 → 准备背景 → 一次规划 → 校验并执行技能。

本轮可信上下文随调用传递，NPC 无需预先注入或事后清理。
"""
# from 模块 import 名字：把标准库 time 的 perf_counter 函数引入本文件。
# 后面写 perf_counter() 即调用它；得到适合计算耗时的浮点秒数，而非日历时间。
from time import perf_counter

# 这两个函数定义在 friston3/input_text.py：分别处理输入格式和识别显式记忆命令。
from friston3.input_text import prepare_input_text, is_memory_command
# DialogueTurn 是 npc/turn.py 的 dataclass；后面的 DialogueTurn(...) 会创建一个本轮资料对象。
from friston3.npc.turn import DialogueTurn


# 阅读入口：main._run_session 创建本类，并在每轮输入时调用 handle。
# 本层编排流程：Planner 选择操作，SkillManager 执行，NPC 管理对话状态。
class Agent:
    # 整体作用：接住 main 传来的管理器与规划器，保存供以后 handle 使用；不会立刻规划。
    # Agent(manager, planner) 会自动调用 __init__；self 是这次创建的 Agent 实例。
    # skill_manager 来自 skills/manager.py 的 SkillManager；planner 来自 planner.py 的 LLMPlanner。
    def __init__(self, skill_manager, planner):
        # 构造器依赖注入：保存传入的对象，多个模块共用同一技能表。
        # 等号右边是参数，左边 self.xxx 是保存在实例上的属性；两个名字指向同一个对象。
        self.skill_manager = skill_manager
        self.planner = planner
        # 当前实现明确依赖 NPC；注册 npc_talk 必须先于创建 Agent。
        # skill_manager.skills 先取管理器的字典，再用 ['npc_talk'] 取得已注册的 NPC 对象。
        # self.npc 的真实类型是 skills/npc.py 的 NpcTalkSkill；找不到键会抛 KeyError。
        self.npc = skill_manager.skills['npc_talk']

    # 调用方 main._run_session；本函数把“一个用户输入”变成“一次技能执行结果 dict”。
    # instruction 是原文 str；input_source 是 text/speech；interlocutor 是身份上下文字典或 None。
    # 下游依次是 input_text.py、NpcTalkSkill、LLMPlanner.plan 和 SkillManager.execute。
    def handle(self, instruction, *, input_source='text', interlocutor=None):
        # * 后面的参数必须按名称传入，减少输入来源/身份传错位置的问题。
        # isinstance/str 是 Python 内置对象；.strip() 是字符串方法，返回去首尾空白的副本。
        # or 是短路运算：若 instruction 不是字符串，就不会再尝试调用它的 strip。
        if not isinstance(instruction, str) or not instruction.strip():
            # {} 创建字典，键是字符串，False 是布尔值；return 把字典交回 main 并结束本函数。
            return {'success': False, 'message': '请输入具体内容。'}
        # 保留 raw_text 与 normalized_text：原文用于追踪，规范化文本用于处理。
        # input_text 是 prepare_input_text 返回的 dict；函数名后的 () 才真正执行规范化。
        # input_source=input_source 左边是被调用函数的参数名，右边是本函数的局部变量值。
        input_text = prepare_input_text(instruction, input_source=input_source)
        # 通过字典键取规范化字符串，并让局部名字 instruction 改为指向这个工作文本。
        # 不会修改调用方 main 的 raw_instruction 字符串，也不会丢掉 input_text 中保存的原文。
        instruction = input_text['normalized_text']
        # len 是内置函数；这里返回 Python 字符串的字符数，而非模型 token 数。
        if len(instruction) > 500:
            return {'success': False, 'message': '单次输入请控制在500字以内。'}
        # started 首次在此定义为 float；finally 中用结束时刻减它，得到本轮处理耗时。
        started = perf_counter()
        # print 输出到终端；f 字符串把 instruction 的实际内容插入 {instruction}。
        print(f'[Agent] 收到指令: {instruction}')
        # 身份由主程序传入，不由 LLM 在 arguments 中自报权限。
        # self.npc 找到 NPC 对象；.set_interlocutor 方法定义在 skills/npc.py，按身份切换会话。
        self.npc.set_interlocutor(interlocutor)
        # try/finally 保证内部任何 return 或异常离开时，都执行最后的计时语句。
        try:
            # is_memory_command 返回 bool；例如“记住：...”进入这条不需要 LLM 规划的分支。
            if is_memory_command(instruction):
                # 显式记忆命令由 NPC 的权限检查处理，无需模型规划。
                # turn 是本轮 DialogueTurn 实例；用关键字实参填入它的字段。
                turn = DialogueTurn(input_text=input_text, started_at=started)
                # call 是“待调用计划”dict；arguments 又是嵌套 dict，不是在这里调用 NPC。
                call = {'skill': 'npc_talk', 'arguments': {'user_text': instruction}}
            else:
                # 准备会话、实体候选、语义意图提示和记忆；提示不等于最终路由。
                # route_context 是 NPC.prepare_route_context 返回的 dict，保存给规划器看的背景。
                route_context = self.npc.prepare_route_context(
                    instruction, input_source=input_text['input_source'])
                # 字典[键] = 值 会新增/覆盖这个键；这里附上本轮规范化记录，不是拼接字符串。
                route_context['input_text'] = input_text
                # plan 返回“准备调用什么”，此刻还没有执行被选中的设备操作。
                # .plan 定义在 planner.py；call 返回 dict 或 None，None 表示没有可执行计划。
                call = self.planner.plan(instruction, dialogue_context=route_context)
                # dataclass 把本轮资料成组传递，避免 NPC 从上轮残留字段猜上下文。
                # decision 让 NPC 复用对话路由；last_* 保留本轮模型调用诊断。
                # A if 条件 else B 是条件表达式：只有存在 call 才调用 dict.get，避免 None.get。
                # get('npc_decision') 缺键时返回 None；设备计划通常就没有 NPC 的子路由决策。
                # last_input 是模型消息 list，last_call 是诊断 dict，均由刚才的 plan 更新。
                turn = DialogueTurn(
                    input_text=input_text, route_context=route_context,
                    decision=call.get('npc_decision') if call else None,
                    planner_input=self.planner.last_input,
                    planner_call=self.planner.last_call, started_at=started)
                if call is None:
                    # 无有效计划就返回失败，不能把任意模型文本当可执行操作。
                    # record_planning_failure 定义在 skills/npc.py，保存诊断并返回失败结果 dict。
                    return self.npc.record_planning_failure(turn)
                if call['skill'] == 'npc_talk':
                    # NPC 接收用户请求，模型草稿仍在 turn.decision 中等待检查。
                    call = {'skill': 'npc_talk', 'arguments': {'user_text': instruction}}
                elif call['skill'] == 'control_arms' and call['arguments'].get('operation') == 'task':
                    # 保留完整指令，避免规划模型改写时丢失物体、位置或动作顺序。
                    # 两层 [] 依次取计划中的 arguments 字典，再修改其 instruction 字段。
                    call['arguments']['instruction'] = instruction
            print(f'[Agent] 工具调用: {call}')
            # context 是可信上下文，arguments 是技能参数；两者不要混成模型输入权限。
            # 结果交回 main，再安排语音/表情，因而下方计时不包含播报。
            # self.skill_manager.execute 定义在 skills/manager.py：按名字找对象，再真正调用技能。
            # 前两个是位置实参；context=... 是命名实参，仅 NPC 分支收到 turn，其余收到 None。
            # return 直接转交技能结果；返回前 Python 仍先运行下面的 finally。
            return self.skill_manager.execute(
                call['skill'], call['arguments'],
                context=turn if call['skill'] == 'npc_talk' else None)
        finally:
            # 即使 try 内提前 return，finally 仍会执行，成功和失败均能记录耗时。
            # elapsed 是 float 秒数；f 字符串里的 :.3f 把它格式化为小数点后三位。
            elapsed = perf_counter() - started
            print(f'[Agent] 本轮总耗时 {elapsed:.3f}s（含规划与执行，不含语音合成/播放）')
