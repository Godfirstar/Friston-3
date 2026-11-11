"""规划器只选择一个已注册技能；技能自己提供说明、参数格式和校验。"""
# copy/json/re/time 都是 Python 标准库：分别处理复制、JSON 文本、正则匹配和计时。
# from ... import ... 允许直接用函数名；import json 则要通过 json.loads 这样的点号访问。
from copy import deepcopy
import json
import re
from time import perf_counter

# config.py 提供默认模型名、HTTP 地址和选项字典；这些大写名字只是项目约定的常量。
from friston3.config import OLLAMA_MODEL, OLLAMA_CHAT_URL, OLLAMA_OPTIONS
# llm.py 的 chat 真正发送请求；estimate_tokens 估算长度；performance_stats 提取耗时信息；
# _policy_memory_status 查询策略内存状态。前导 _ 是“内部使用”的命名约定，不是访问权限语法。
from friston3.llm import chat, estimate_tokens, performance_stats, _policy_memory_status
# npc/agentic_rag.py 定义对话 schema、消息构造和决策校验；括号允许 import 跨多行排版。
from friston3.npc.agentic_rag import (
    AgenticSettings, ROUTE_SCHEMA, ROUTE_TOKENS, _unique_object,
    build_route_input, normalize_entity_choices, parse_decision, route_schema_for_signals,
)
# npc/persona.py 的 normalize_presentation 把表情/动作提示整理成约定结构。
from friston3.npc.persona import normalize_presentation


class LLMPlanner:
    """统一规划；策略驻留时先用轻量模型识别机械臂指令。"""

    def __init__(self, skill_manager, model=OLLAMA_MODEL, url=OLLAMA_CHAT_URL):
        # main._run_session 中 LLMPlanner(manager) 创建实例时自动调用；self 是这个新实例。
        # skill_manager 是 SkillManager 对象，model/url 是可覆盖默认值的字符串实参。
        # 构造器只保存引用与诊断字段，不在这里请求 LLM；请求发生在后面的 plan 方法中。
        # 多重赋值按对应位置保存：等价于 self.model=model 然后 self.url=url。
        self.model, self.url = model, url
        # 引用同一个 dict，而非复制：main 注册的技能供规划和执行共同使用。
        # 动态来源是注册表；注册动作本身仍由 main 显式编写。
        # 点号先取管理器的 skills 字典，再保存到本实例的 skills 属性，没有复制这个字典。
        self.skills = skill_manager.skills
        # Agent 会把本轮诊断写入 DialogueTurn，供“查看证据”解释失败阶段。
        # {} 新建空 dict（按键查值）；[] 新建空 list（按顺序存元素），两者用途不同。
        self.last_call = {}
        self.last_input = []

    def _schema_for_routes(self, dialogue_schema, dialogue_routes, action_routes):
        """将对话格式与各技能的参数格式组合成一次模型输出。

        anyOf 的每个分支都是完整对象，避免依赖模型后端的条件语法支持。
        JSON 格式约束只帮助生成，技能仍负责执行前的实际校验。
        """
        # 调用方是本文件 plan；三个参数来自对话路由schema、允许的对话名tuple、设备名list。
        # 返回 dict，稍后传给 llm.chat 的 schema 参数；这里仅创建数据，不执行任何技能。
        # def 写在方法内部叫嵌套函数，仅本次 _schema_for_routes 调用中可直接使用。
        def closed_object(properties, required=()):
            # properties 是“字段名->字段规则”的 dict；required 是可迭代的必填字段名。
            # () 是空 tuple，作为默认值；list(required) 会把传来的元组/字典键变成列表。
            # required 是必填字段，additionalProperties=False 排除未知字段。
            return {"type": "object", "properties": properties,
                    "required": list(required), "additionalProperties": False}

        # empty_arguments 首次定义为上述函数返回的 dict，描述“只能是空对象”的参数规则。
        empty_arguments = closed_object({})

        def branch(routes, arguments, *, dialogue=False):
            # 另一个嵌套函数：routes 是该分支允许的名字，arguments 是技能提供的参数schema。
            # * 后 dialogue 必须写成 dialogue=True；False 默认表示设备/UNSUPPORTED分支。
            # deepcopy 防止某一路由修改 schema 时连带改动共享的嵌套字典。
            # properties 是本分支所有字段的规则字典；list(routes) 生成 route 的枚举值列表。
            # 下方 dict.items() 逐个给出 (键,值)；for key,value ... 将二元组拆给两个局部变量。
            # {key: ... for ... if ...} 是字典推导式：复制对话字段，但排除已由本分支定义的 route。
            # **{...} 在字典里展开键值；deepcopy 来自标准库 copy，递归复制嵌套容器。
            properties = {
                "route": {"type": "string", "enum": list(routes)},
                "arguments": deepcopy(arguments),
                **{key: deepcopy(value) for key, value in dialogue_schema["properties"].items()
                   if key != "route"},
            }
            # not dialogue 只让设备分支清空对话专用字段，避免一个输出又执行设备又夹带台词。
            if not dialogue:
                # 两层 [] 先取 entity_choices 规则字典，再设置 maxItems=0，要求返回空列表。
                properties["entity_choices"]["maxItems"] = 0
                # for 依次读取外层元组中的每个 (字段名,固定值)，分别绑定局部 key 与 value。
                for key, value in (("query", ""), ("reply", ""), ("followup", False),
                                   ("detail", False), ("intent", "general")):
                    # A if 条件 else B 是条件表达式；布尔默认值对应 boolean，其余对应 string。
                    # enum=[value] 是只有一个元素的列表，意味着这个字段只能取这个固定值。
                    properties[key] = {"type": "boolean" if type(value) is bool else "string",
                                       "enum": [value]}
                # 调用上面的 closed_object 创建 presentation 规则；末尾元组列出三个必填键。
                properties["presentation"] = closed_object({
                    "emotion": {"type": "string", "enum": ["neutral"]},
                    "gesture": {"type": "string", "enum": ["none"]},
                    "arm_gesture": {"type": "string", "enum": ["none"]}},
                    ("emotion", "gesture", "arm_gesture"))
            # 第二个 properties 传给 required；遍历 dict 得到键，所以本分支全部字段都必填。
            return closed_object(properties, properties)

        # branches 是 schema 分支列表：有 NPC 路由时先加入一个对话分支，否则从空列表开始。
        branches = ([branch(dialogue_routes, empty_arguments, dialogue=True)]
                    if dialogue_routes else [])
        # 参数 schema 由各技能提供，规划器无需复制其业务字段定义。
        # extend 是 list 方法，会把可迭代对象中的每一项加入列表；这里传入的是生成器表达式。
        # for route in action_routes 逐个取技能名；self.skills[route] 取技能对象；
        # .argument_schema() 调用对应 skills/*.py 的方法，例如 PhotoSkill.argument_schema。
        # (route,) 是只含一个元素的 tuple；逗号不能省，(route) 仍只是 route 本身。
        branches.extend(branch((route,), self.skills[route].argument_schema()) for route in action_routes)
        # append 与 extend 不同，只添加一个元素；UNSUPPORTED 描述无法执行的请求。
        branches.append(branch(("UNSUPPORTED",), empty_arguments))
        # anyOf 是 JSON Schema 的关键字，表示模型输出需满足列表中的至少一个分支。
        return {"anyOf": branches}

    def _plan_resident_arms(self, instruction):
        """Classify arm commands on CPU without loading the dialogue persona."""
        # 本文件 plan 在 π0.5 已占用内存时调用；instruction 为用户原文 str。
        # 下游是 llm.chat 和 ArmsControlSkill.normalize_arguments；返回计划 dict 或 None。
        # None 可能表示“不适用”或“失败”，调用方用 self.last_call['status'] 区分这两种情况。
        # Pi0.5 驻留时的轻量文本路由，不是预测关节动作的神经网络。
        # 下方 num_gpu=0 让此请求使用 CPU，输出 task/status/play 等操作参数。
        # model 是本次局部模型名，不覆盖 self.model 保存的普通对话模型名。
        model = "qwen3.5:2b-q4_K_M"
        # messages 是 list[dict] 形状的数据：第一条 system 定义任务，第二条 user 提供原文。
        # 相邻字符串在 Python 中自动拼接；下面大段引号内容是给模型读的文本，不是 Python 指令。
        messages = [{"role": "system", "content": (
            "只判断本轮是否要求实际机械臂操作，仅输出JSON。"
            "operation:task=要求抓取/移动/放置/整理物体，多步抓放仍是一次task；"
            "status=查询机械臂连接/任务进度/结果；list=列示教动作；"
            "stop=明确要求停止机械臂任务；play=播放已示教动作。"
            "普通聊天、知识/能力解释、假设、引用、被否定的操作均为none；"
            "不要抓取不是停止正在执行的任务，除非明确要求停止。"
            "play的name用已示教名，常用rest休息/wave挥手/welcome欢迎/point_pickup取物指示/shrug摊手；"
            "其他operation的name必须空串。"
            "arm仅task且明确指定左机械臂/右机械臂才为left/right，否则空串；"
            "物体或目的地在左/右侧不表示选择左/右机械臂。"
            "仅依据当前请求，不猜执行结果，不回复聊天内容。")},
            {"role": "user", "content": instruction}]
        # schema 是输出格式 dict；enum 限可选值，required 列必填键，不允许额外字段。
        # 它限制生成格式，后面仍用 Python 检查 operation 与 name/arm 的组合是否合法。
        schema = {"type": "object", "properties": {
            "operation": {"type": "string", "enum": ["none", "task", "status", "list", "stop", "play"]},
            "name": {"type": "string", "maxLength": 48},
            "arm": {"type": "string", "enum": ["", "left", "right"]}},
            "required": ["operation", "name", "arm"], "additionalProperties": False}
        # started 是标准库 perf_counter 返回的 float 秒数，结束时相减得到耗时。
        started = perf_counter()
        # 将输入和诊断保存到实例上，Agent.handle 才能在 plan 返回后取到本轮信息。
        self.last_input = messages
        self.last_call = {"phase": "resident_arms_plan", "model": model, "status": "pending"}
        # stage 记录运行到哪一步；发生异常时用于报告失败位置，避免只留下一个错误文本。
        stage = "request"
        try:
            # chat 定义在 llm.py；这是本方法首次真正发出模型 HTTP 请求。
            # result 是其响应 dict，模型文本位于 result['message']['content']。
            # options 用 ** 复制默认选项的键值，后写的 temperature/num_gpu 等覆盖同名默认值。
            # diagnostics 接收同一个 last_call 字典，chat 可在请求过程写入诊断信息。
            result = chat(messages, model=model, url=self.url, timeout=20.0,
                          options={**OLLAMA_OPTIONS, "temperature": 0, "num_gpu": 0,
                                   "num_ctx": 2048, "num_predict": 80},
                          schema=schema, diagnostics=self.last_call, reject_truncated=True)
            # performance_stats 在 llm.py 提取响应性能字段，dict.update 原地加入这些键值。
            self.last_call.update(performance_stats(result, perf_counter() - started))
            stage = "validate_plan"
            # 能解析 JSON 不代表参数可执行，之后还检查字段、类型和操作组合。
            # json.loads 是标准库函数：把 JSON 字符串解析成 Python 对象，call 预期为 dict。
            # object_pairs_hook 传的是函数对象（没有括号），JSON 解码器会逐对象调用它。
            # _unique_object 定义在 npc/agentic_rag.py，遇到重复键抛错，防止值被悄悄覆盖。
            call = json.loads(result["message"]["content"], object_pairs_hook=_unique_object)
            # set(dict) 取键集合；顺序无关地比较是否恰好有 operation/name/arm 三个键。
            if type(call) is not dict or set(call) != {"operation", "name", "arm"}:
                raise ValueError("arm router fields do not match the schema")
            # 多重赋值把三个字段分别绑定给局部变量；此刻只是解包，类型由下一句检查。
            operation, name, arm = call["operation"], call["name"], call["arm"]
            # *sequence 在 tuple 中展开元素；把 'none' 与技能声明的 operations 拼成允许值。
            # or 连接任一违规条件：只有 play 可带 name，只有 task 可带 arm。
            if (type(operation) is not str or operation not in ("none", *self.skills["control_arms"].operations)
                    or type(name) is not str or type(arm) is not str or arm not in ("", "left", "right")
                    or (operation != "play" and name) or (operation != "task" and arm)):
                raise ValueError("invalid arm router arguments")
            # 保存深拷贝快照，后面对 arm/arguments 的规范化不会改掉“模型原本输出了什么”。
            self.last_call["parsed_plan"] = deepcopy(call)
            # 非空字符串 arm 在 if 中为真；空字符串代表模型没有指定左右臂。
            if arm:
                # A model's object-location label is not an explicit arm choice.
                # arm_reference 是“左右臂名->正则规则”的局部字典；r 前缀保留正则中的反斜杠。
                arm_reference = {
                    "left": r"左(?:边|侧)?的?(?:机械|手)?臂|\bleft\s+(?:robot(?:ic)?\s+)?arm\b",
                    "right": r"右(?:边|侧)?的?(?:机械|手)?臂|\bright\s+(?:robot(?:ic)?\s+)?arm\b",
                }
                # re.search 在原始指令中查明确左右臂表述；IGNORECASE 让英文字母忽略大小写。
                # 返回 None 表示没匹配到；不能拿“物体在右边”直接认作“使用右臂”。
                if re.search(arm_reference[arm], instruction, flags=re.IGNORECASE) is None:
                    # setdefault 缺键时放入 [] 并返回该列表；append 再向列表追加一条诊断字符串。
                    self.last_call.setdefault("normalizations", []).append("discarded_arm_without_explicit_reference")
                    # 仅清空本地 arm 变量，后面不再生成 arm 实参；原始计划快照仍保留。
                    arm = ""
            if operation == "none":
                # update 也接受 key=value 命名实参；此状态让外层 plan 继续普通统一规划。
                self.last_call.update(status="not_applicable", selected_skill="none")
                return None
            # arguments 是准备交给机械臂技能的参数 dict，先放入所有操作共有的 operation。
            arguments = {"operation": operation}
            if operation == "task":
                # task 带用户原句；左右臂只有确实选中时才作为可选字段添加。
                arguments["instruction"] = instruction
                if arm:
                    arguments["arm"] = arm
            elif operation == "play":
                # play 使用已有轨迹名；status/list/stop 不需要额外字段。
                arguments["name"] = name
            # 到 skills/arms.py 的类方法再校验，返回新的规范化 dict；仍未执行 ArmIntegration。
            arguments = self.skills["control_arms"].normalize_arguments(arguments)
            self.last_call.update(status="ok", selected_skill="control_arms",
                                  normalized_arguments=deepcopy(arguments))
            # 计划交给 Agent.handle，那里才会请求 SkillManager 真正执行技能。
            return {"skill": "control_arms", "arguments": arguments}
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            # except 元组列出可处理的错误类型，as error 接住异常对象。
            # getattr(obj,'reason',None) 是内置取属性函数；没有属性时返回 None，不抛 AttributeError。
            # timed_out 是 bool；既识别直接超时，也识别网络异常包装在 reason 内的超时。
            timed_out = (isinstance(error, TimeoutError)
                         or isinstance(getattr(error, "reason", None), TimeoutError))
            # last_call.get 缺少 failure_stage 时才回退到本函数的 stage；保留传输层更具体的信息。
            # type(error).__name__ 得到异常类名字；f 字符串再拼上具体错误内容。
            self.last_call.update(status="error", failure_stage=self.last_call.get("failure_stage", stage),
                                  error_kind="timeout" if timed_out else "planning_error",
                                  error=f"{type(error).__name__}: {error}")
            return None
        finally:
            # 无论成功 return、无关 return 或捕获异常，finally 都执行；round(...,3) 保留三位小数。
            self.last_call["wall_s"] = round(perf_counter() - started, 3)
            # 字典推导式过滤冗长的原始输出和计划字段，只打印简短诊断，原字段仍保存在对象中。
            print("[LLMPlanner] 机械臂轻量规划：", {key: value for key, value in self.last_call.items()
                  if key not in ("raw_output", "parsed_plan")})

    def plan(self, instruction, *, dialogue_context=None):
        # Agent.handle 的调用入口：只返回计划或 None，不在这里执行技能。
        # instruction 是用户字符串；dialogue_context 是 NPC.prepare_route_context 生成的 dict。
        # * 使 dialogue_context 只能按名字传；未传时 None，下面按空背景处理。
        # 成功返回 {'skill':名称,'arguments':参数}，对话还带 npc_decision；失败/不支持返回 None。
        # 下游：本文件轻量规划/schema组装、agentic_rag.py 输入与解析、llm.chat、各技能校验。
        # 重置两个实例字段，避免上一轮的输入/诊断被本轮失败路径误当成新结果。
        self.last_call, self.last_input = {}, []
        # isinstance 检查类型；or 短路保证非字符串不调用 strip；空白文本也没有可规划任务。
        if not isinstance(instruction, str) or not instruction.strip():
            return None
        # arms_probe 是轻量机械臂规划的诊断快照，初始 None 表示本轮尚未走过这条路径。
        arms_probe = None
        # 'control_arms' in self.skills 检查字典中是否注册这个键。
        # _policy_memory_status 在 llm.py 返回状态元组，[0] 读取第一个值：策略是否占用内存。
        # and 是短路运算：没注册机械臂技能时，不再读取第二个条件。
        if "control_arms" in self.skills and _policy_memory_status()[0]:
            # 小模型明确判断“非机械臂”后才继续统一规划；错误直接交回调用者。
            # 所以“统一计划”不意味着所有情形都只发生一次 LLM 请求。
            # answer 是上方方法返回的计划 dict 或 None；方法同时更新本对象 last_call。
            answer = self._plan_resident_arms(instruction)
            if self.last_call["status"] != "not_applicable":
                return answer
            # 只有明确“不适用机械臂”才保存快照并继续；下一步会重建 last_call。
            arms_probe = deepcopy(self.last_call)
        # or 返回第一个真值对象，不只是 True/False；None 或空字典会替换成新 {}。
        # context 是本函数局部背景字典，不复制其中已有的历史等嵌套对象。
        context = dialogue_context or {}
        # dict.get(key,default) 缺键时回退；signals 包含实体候选、语义提示、输入来源等。
        signals = context.get("signals", {})
        # route_schema_for_signals 定义在 npc/agentic_rag.py，按本轮信号准备对话输出格式。
        dialogue_schema = route_schema_for_signals(signals)
        # 三层 [] 逐层取 properties -> route -> enum，得到对话名列表；tuple 转为元组。
        # 没注册 NPC 时使用空元组 ()，后面的 schema 就不会允许对话分支。
        dialogue_routes = (tuple(dialogue_schema["properties"]["route"]["enum"])
                           if "npc_talk" in self.skills else ())
        # CHAT/RETRIEVE 等是 NPC 内部路由；设备 route 使用已注册的技能名。
        # 遍历 dict 得到键名；生成器表达式滤掉 NPC；内置 sorted 返回按名字排序的新 list。
        # name 是该表达式内部循环变量；action_routes 保存设备技能名，不是技能对象。
        action_routes = sorted(name for name in self.skills if name != "npc_talk")
        # route 是唯一任务标识；不再让模型同时输出可能互相冲突的 skill。
        # 只有明确选中的技能才可执行，不从回复台词或改写稿推测设备动作。
        # 非空 {...} 没有键:值时是 set（单独 {} 仍是 dict）；* 展开名字，集合自动去重。
        allowed_routes = {*dialogue_routes, *action_routes, "UNSUPPORTED"}
        # 调用本类上面的 schema 组装方法，得到约束整个模型返回对象的 dict。
        schema = self._schema_for_routes(dialogue_schema, dialogue_routes, action_routes)
        # started 是本次统一规划的开始时刻；stage 是进度字符串，供异常诊断使用。
        started = perf_counter()
        stage = "build_input"
        # 请求尝试/收头/收响应分开记录：能区分“没发出去”“等待响应”“返回后解析失败”。
        self.last_call = {"phase": "unified_plan", "model": self.model,
                          "status": "pending", "request_attempted": False,
                          "response_headers_received": False, "response_received": False}
        if arms_probe is not None:
            # 把先前机械臂轻量路由的诊断嵌入本轮诊断，保留“一轮曾请求两次”的事实。
            self.last_call["arms_route"] = arms_probe
        try:
            # 当前请求、历史线索、身份和工具说明分开组装，便于追踪信息来源。
            # messages 为 build_route_input 返回的消息 list，函数定义在 npc/agentic_rag.py。
            # 每个 context.get 都提供该字段的默认形状，例如 history 默认 []、topic 默认 {}。
            # estimate_tokens=estimate_tokens 传函数对象（无括号），让组装函数需要时自己调用。
            # tool_help 的字典推导式把每个技能名映射到技能对象的 description 类属性。
            messages = build_route_input(
                instruction, history=context.get("history", []), topic=context.get("topic", {}),
                memory_context=context.get("memory_context", ""),
                signals=context.get("signals", {}), pending_reference=context.get("pending_reference", {}),
                interlocutor_context=context.get("interlocutor"),
                estimate_tokens=estimate_tokens,
                tool_help={name: self.skills[name].description for name in action_routes})
            # 把实际准备发送的消息存到实例属性，便于 Agent 后续放进 DialogueTurn 留痕。
            self.last_input = messages
            # AgenticSettings 是 npc/agentic_rag.py 的 dataclass；defaults 类方法返回配置实例。
            # settings.route_timeout 用点号读取此实例的超时秒数字段。
            settings = AgenticSettings.defaults()
            # dict.update 原地补诊断字段；sum 将生成器给出的每条消息估算长度加总。
            # m 是循环中当前消息 dict；m['content'] 是内容字符串，+8 是每条消息附加估算开销。
            self.last_call.update(
                timeout_s=settings.route_timeout, stream=False,
                output_token_limit=ROUTE_TOKENS,
                estimated_prompt_tokens=sum(
                    estimate_tokens(m["content"]) + 8 for m in messages))
            # :g 是 f字符串的数字格式：按有效数字显示，避免把 8.0 打印成多余小数。
            print(f"[LLMPlanner] 统一规划等待配置：通常 {settings.route_timeout:g}s；"
                  "Pi0.5 驻留时 CPU 至少 60s（故障等待预算，不是响应速度目标）")
            stage = "request"
            # 共用传输代码只发送一次请求；规划失败时绝不重新猜设备动作。
            # result 是 llm.chat 返回的响应 dict；普通计划用 self.model，不用上面的局部小模型。
            # schema 帮助约束 JSON；reject_truncated=True 由传输层拒绝截断响应。
            # policy_cpu_timeout 是策略驻留时的等待配置，并不是用户体验目标用时。
            result = chat(messages, model=self.model, url=self.url, timeout=settings.route_timeout,
                          options={**OLLAMA_OPTIONS, "temperature": 0, "num_predict": ROUTE_TOKENS},
                          schema=schema, diagnostics=self.last_call, reject_truncated=True,
                          policy_cpu_timeout=60.0)
            # 这里 status=ok 只表示请求阶段已拿到响应；后续校验仍可能抛错并改成 error。
            self.last_call.update(performance_stats(result, perf_counter() - started), status="ok")
            stage = "validate_plan"
            # raw 是模型返回的 JSON 文本 str，尚不是 Python 可直接按键读取的 dict。
            raw = result["message"]["content"]
            # object_pairs_hook 拦截重复键，否则后一个同名值会悄悄覆盖前一个。
            # json.loads 将 raw 解码；重复键检查函数来自 npc/agentic_rag.py，详见上方轻量规划。
            call = json.loads(raw, object_pairs_hook=_unique_object)
            self.last_call["parsed_plan"] = deepcopy(call)
            # expected 是预期键的 set：一个 arguments，加上对话 schema 所列必填键。
            # set(call) 读取实际键，比较集合不受 JSON 字段顺序影响。
            expected = {"arguments", *ROUTE_SCHEMA["required"]}
            if not isinstance(call, dict) or set(call) != expected:
                raise ValueError("planner fields do not match the schema")
            # 表演提示不能赋予模型额外的设备权限。
            # normalize_presentation 定义在 npc/persona.py，返回规范化后的表演提示 dict。
            call["presentation"] = normalize_presentation(call["presentation"])
            # route 是选择的路由 str，arguments 是技能实参 dict；多重赋值分别取出这两个字段。
            route, arguments = call["route"], call["arguments"]
            if not isinstance(route, str) or type(arguments) is not dict:
                raise ValueError("invalid route/argument types")
            if route not in allowed_routes:
                raise ValueError("route is not registered")
            # 对话路由 CHAT/RETRIEVE/... 最终都由同一个 npc_talk 技能处理；设备路由直接用技能名。
            if route in dialogue_routes:
                skill = "npc_talk"
                # 保留已有的无操作兼容格式；真实设备参数不能在这里被丢弃。
                # 例如 owner_side=none 仍是 configure_device 的真实设置。
                # discarded_noop_arguments 是 bool，记录是否遇到这一个兼容用的无操作参数形状。
                discarded_noop_arguments = arguments == {"choice": "none"}
                if discarded_noop_arguments:
                    # 重绑局部 arguments，并同步 call 中的字段；原 parsed_plan 深拷贝保留旧值。
                    arguments = {}
                    call["arguments"] = arguments
                elif arguments:
                    # 非空字典为真；其他对话参数不是受支持的兼容形式，因此直接报错。
                    raise ValueError("dialogue routes may not supply execution arguments")
                # 先用字典推导式只取对话字段，再 json.dumps 编成 JSON 文本，交给既有解析接口。
                # ensure_ascii=False 保留中文字符；parse_decision 再校验并规范化对话分支。
                # decision 是结果 dict；函数来自 npc/agentic_rag.py，使用程序提供的 signals，
                # 而非让模型自己声称“分类器给了什么分数”。这里没有再次调用 LLM。
                decision = parse_decision(json.dumps({key: call[key] for key in ROUTE_SCHEMA["required"]},
                                                     ensure_ascii=False),
                                          signals=signals,
                                          pending_reference=context.get("pending_reference"),
                                          diagnostics=self.last_call)
                # 同文件的 normalize_entity_choices 复核语音候选选择，返回规范化 decision 字典。
                decision = normalize_entity_choices(decision, signals, diagnostics=self.last_call)
                # 该 decision 经 Agent 的 DialogueTurn 交给 NPC，避免重复模型路由。
                if discarded_noop_arguments:
                    # setdefault 取已有诊断列表或新建 []；append 加一条记录，不改变任务选择。
                    self.last_call.setdefault("normalizations", []).append(
                        "discarded_noop_dialogue_arguments")
                    self.last_call["normalized_arguments"] = {}
            else:
                # 非对话路由：UNSUPPORTED 转成内部的 none，其余直接保留注册技能名。
                skill = "none" if route == "UNSUPPORTED" else route
                # 设备调用必须把对话专用字段留空；is not False 也避免把整数 0 当成布尔 False。
                if (call["query"] != "" or call["reply"] != ""
                        or call["entity_choices"] != []
                        or call["followup"] is not False or call["intent"] != "general"
                        or call["detail"] is not False):
                    raise ValueError("non-dialogue routes may not contain dialogue fields")
                if skill == "none":
                    if arguments:
                        raise ValueError("unsupported routes take no arguments")
                else:
                    # schema 管输出格式，技能的校验方法管参数是否满足业务约定。
                    # self.skills[skill] 找到对象，方法定义在对应 skills/*.py；默认 defaults=False。
                    # arguments 被替换成该方法返回的规范化 dict，不直接执行方法名为 execute 的函数。
                    arguments = self.skills[skill].normalize_arguments(arguments)
                # 设备计划没有 NPC 子路由，后面就不会把 npc_decision 加入返回计划。
                decision = None
            # 这两条是“选择结果/规划耗时”的诊断，selected_skill 并不证明设备已经完成操作。
            self.last_call["selected_skill"] = skill
            self.last_call["wall_s"] = round(perf_counter() - started, 3)
            print("[LLMPlanner] 统一决策性能：", {key: value for key, value in self.last_call.items()
                  if key not in ("raw_output", "parsed_plan", "normalized_decision")})
            print("[LLMPlanner] 模型原始输出:", raw)
            if skill == "none":
                return None
            # answer 是交给 Agent.handle 的计划 dict；它与模型原始 call 的字段结构有意不同。
            answer = {"skill": skill, "arguments": arguments}
            if decision is not None:
                answer["npc_decision"] = decision
            # 返回后由 Agent -> SkillManager -> 对应技能 execute，真正执行被选中的操作。
            return answer
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            # 捕获本流程支持处理的异常；超时判断方法与 _plan_resident_arms 中相同。
            timed_out = (isinstance(error, TimeoutError)
                         or isinstance(getattr(error, "reason", None), TimeoutError))
            # 用 ** 先复制已有诊断，再覆盖失败字段；类型名和异常文字组成可读 error 字符串。
            # 与 update 不同，这里是创建新外层 dict 再让属性指向它。
            self.last_call = {**self.last_call, "phase": "unified_plan", "model": self.model,
                              "status": "error", "wall_s": round(perf_counter() - started, 3),
                              "failure_stage": self.last_call.get("failure_stage", stage),
                              "error_kind": "timeout" if timed_out else "planning_error",
                              "error": f"{type(error).__name__}: {error}"}
            print("[LLMPlanner] 统一决策失败；未执行任何技能：", self.last_call)
            return None
