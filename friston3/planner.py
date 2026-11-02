"""规划器只选择一个已注册技能；技能自己提供说明、参数格式和校验。"""
from copy import deepcopy
import json
import re
from time import perf_counter

from friston3.config import OLLAMA_MODEL, OLLAMA_CHAT_URL, OLLAMA_OPTIONS
from friston3.llm import chat, estimate_tokens, performance_stats, _policy_memory_status
from friston3.npc.agentic_rag import (
    AgenticSettings, ROUTE_SCHEMA, ROUTE_TOKENS, _unique_object,
    build_route_input, normalize_entity_choices, parse_decision, route_schema_for_signals,
)
from friston3.npc.persona import normalize_presentation


class LLMPlanner:
    """统一规划；策略驻留时先用轻量模型识别机械臂指令。"""

    def __init__(self, skill_manager, model=OLLAMA_MODEL, url=OLLAMA_CHAT_URL):
        self.model, self.url = model, url
        self.skills = skill_manager.skills
        self.last_call = {}
        self.last_input = []

    def _schema_for_routes(self, dialogue_schema, dialogue_routes, action_routes):
        """将对话格式与各技能的参数格式组合成一次模型输出。

        anyOf 的每个分支都是完整对象，避免依赖模型后端的条件语法支持。
        JSON 格式约束只帮助生成，技能仍负责执行前的实际校验。
        """
        def closed_object(properties, required=()):
            return {"type": "object", "properties": properties,
                    "required": list(required), "additionalProperties": False}

        empty_arguments = closed_object({})

        def branch(routes, arguments, *, dialogue=False):
            properties = {
                "route": {"type": "string", "enum": list(routes)},
                "arguments": deepcopy(arguments),
                **{key: deepcopy(value) for key, value in dialogue_schema["properties"].items()
                   if key != "route"},
            }
            if not dialogue:
                properties["entity_choices"]["maxItems"] = 0
                for key, value in (("query", ""), ("reply", ""), ("followup", False),
                                   ("detail", False), ("intent", "general")):
                    properties[key] = {"type": "boolean" if type(value) is bool else "string",
                                       "enum": [value]}
                properties["presentation"] = closed_object({
                    "emotion": {"type": "string", "enum": ["neutral"]},
                    "gesture": {"type": "string", "enum": ["none"]},
                    "arm_gesture": {"type": "string", "enum": ["none"]}},
                    ("emotion", "gesture", "arm_gesture"))
            return closed_object(properties, properties)

        branches = ([branch(dialogue_routes, empty_arguments, dialogue=True)]
                    if dialogue_routes else [])
        branches.extend(branch((route,), self.skills[route].argument_schema()) for route in action_routes)
        branches.append(branch(("UNSUPPORTED",), empty_arguments))
        return {"anyOf": branches}

    def _plan_resident_arms(self, instruction):
        """Classify arm commands on CPU without loading the dialogue persona."""
        model = "qwen3.5:2b-q4_K_M"
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
        schema = {"type": "object", "properties": {
            "operation": {"type": "string", "enum": ["none", "task", "status", "list", "stop", "play"]},
            "name": {"type": "string", "maxLength": 48},
            "arm": {"type": "string", "enum": ["", "left", "right"]}},
            "required": ["operation", "name", "arm"], "additionalProperties": False}
        started = perf_counter()
        self.last_input = messages
        self.last_call = {"phase": "resident_arms_plan", "model": model, "status": "pending"}
        stage = "request"
        try:
            result = chat(messages, model=model, url=self.url, timeout=20.0,
                          options={**OLLAMA_OPTIONS, "temperature": 0, "num_gpu": 0,
                                   "num_ctx": 2048, "num_predict": 80},
                          schema=schema, diagnostics=self.last_call, reject_truncated=True)
            self.last_call.update(performance_stats(result, perf_counter() - started))
            stage = "validate_plan"
            call = json.loads(result["message"]["content"], object_pairs_hook=_unique_object)
            if type(call) is not dict or set(call) != {"operation", "name", "arm"}:
                raise ValueError("arm router fields do not match the schema")
            operation, name, arm = call["operation"], call["name"], call["arm"]
            if (type(operation) is not str or operation not in ("none", *self.skills["control_arms"].operations)
                    or type(name) is not str or type(arm) is not str or arm not in ("", "left", "right")
                    or (operation != "play" and name) or (operation != "task" and arm)):
                raise ValueError("invalid arm router arguments")
            self.last_call["parsed_plan"] = deepcopy(call)
            if arm:
                # A model's object-location label is not an explicit arm choice.
                arm_reference = {
                    "left": r"左(?:边|侧)?的?(?:机械|手)?臂|\bleft\s+(?:robot(?:ic)?\s+)?arm\b",
                    "right": r"右(?:边|侧)?的?(?:机械|手)?臂|\bright\s+(?:robot(?:ic)?\s+)?arm\b",
                }
                if re.search(arm_reference[arm], instruction, flags=re.IGNORECASE) is None:
                    self.last_call.setdefault("normalizations", []).append("discarded_arm_without_explicit_reference")
                    arm = ""
            if operation == "none":
                self.last_call.update(status="not_applicable", selected_skill="none")
                return None
            arguments = {"operation": operation}
            if operation == "task":
                arguments["instruction"] = instruction
                if arm:
                    arguments["arm"] = arm
            elif operation == "play":
                arguments["name"] = name
            arguments = self.skills["control_arms"].normalize_arguments(arguments)
            self.last_call.update(status="ok", selected_skill="control_arms",
                                  normalized_arguments=deepcopy(arguments))
            return {"skill": "control_arms", "arguments": arguments}
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            timed_out = (isinstance(error, TimeoutError)
                         or isinstance(getattr(error, "reason", None), TimeoutError))
            self.last_call.update(status="error", failure_stage=self.last_call.get("failure_stage", stage),
                                  error_kind="timeout" if timed_out else "planning_error",
                                  error=f"{type(error).__name__}: {error}")
            return None
        finally:
            self.last_call["wall_s"] = round(perf_counter() - started, 3)
            print("[LLMPlanner] 机械臂轻量规划：", {key: value for key, value in self.last_call.items()
                  if key not in ("raw_output", "parsed_plan")})

    def plan(self, instruction, *, dialogue_context=None):
        self.last_call, self.last_input = {}, []
        if not isinstance(instruction, str) or not instruction.strip():
            return None
        arms_probe = None
        if "control_arms" in self.skills and _policy_memory_status()[0]:
            answer = self._plan_resident_arms(instruction)
            if self.last_call["status"] != "not_applicable":
                return answer
            arms_probe = deepcopy(self.last_call)
        context = dialogue_context or {}
        signals = context.get("signals", {})
        dialogue_schema = route_schema_for_signals(signals)
        dialogue_routes = (tuple(dialogue_schema["properties"]["route"]["enum"])
                           if "npc_talk" in self.skills else ())
        action_routes = sorted(name for name in self.skills if name != "npc_talk")
        # route 是唯一任务标识；不再让模型同时输出可能互相冲突的 skill。
        # 只有明确选中的技能才可执行，不从回复台词或改写稿推测设备动作。
        allowed_routes = {*dialogue_routes, *action_routes, "UNSUPPORTED"}
        schema = self._schema_for_routes(dialogue_schema, dialogue_routes, action_routes)
        started = perf_counter()
        stage = "build_input"
        self.last_call = {"phase": "unified_plan", "model": self.model,
                          "status": "pending", "request_attempted": False,
                          "response_headers_received": False, "response_received": False}
        if arms_probe is not None:
            self.last_call["arms_route"] = arms_probe
        try:
            messages = build_route_input(
                instruction, history=context.get("history", []), topic=context.get("topic", {}),
                memory_context=context.get("memory_context", ""),
                signals=context.get("signals", {}), pending_reference=context.get("pending_reference", {}),
                interlocutor_context=context.get("interlocutor"),
                estimate_tokens=estimate_tokens,
                tool_help={name: self.skills[name].description for name in action_routes})
            self.last_input = messages
            settings = AgenticSettings.defaults()
            self.last_call.update(
                timeout_s=settings.route_timeout, stream=False,
                output_token_limit=ROUTE_TOKENS,
                estimated_prompt_tokens=sum(
                    estimate_tokens(m["content"]) + 8 for m in messages))
            print(f"[LLMPlanner] 统一规划等待配置：通常 {settings.route_timeout:g}s；"
                  "Pi0.5 驻留时 CPU 至少 60s（故障等待预算，不是响应速度目标）")
            stage = "request"
            # 共用传输代码只发送一次请求；规划失败时绝不重新猜设备动作。
            result = chat(messages, model=self.model, url=self.url, timeout=settings.route_timeout,
                          options={**OLLAMA_OPTIONS, "temperature": 0, "num_predict": ROUTE_TOKENS},
                          schema=schema, diagnostics=self.last_call, reject_truncated=True,
                          policy_cpu_timeout=60.0)
            self.last_call.update(performance_stats(result, perf_counter() - started), status="ok")
            stage = "validate_plan"
            raw = result["message"]["content"]
            call = json.loads(raw, object_pairs_hook=_unique_object)
            self.last_call["parsed_plan"] = deepcopy(call)
            expected = {"arguments", *ROUTE_SCHEMA["required"]}
            if not isinstance(call, dict) or set(call) != expected:
                raise ValueError("planner fields do not match the schema")
            # 表演提示不能赋予模型额外的设备权限。
            call["presentation"] = normalize_presentation(call["presentation"])
            route, arguments = call["route"], call["arguments"]
            if not isinstance(route, str) or type(arguments) is not dict:
                raise ValueError("invalid route/argument types")
            if route not in allowed_routes:
                raise ValueError("route is not registered")
            if route in dialogue_routes:
                skill = "npc_talk"
                # 保留已有的无操作兼容格式；真实设备参数不能在这里被丢弃。
                # 例如 owner_side=none 仍是 configure_device 的真实设置。
                discarded_noop_arguments = arguments == {"choice": "none"}
                if discarded_noop_arguments:
                    arguments = {}
                    call["arguments"] = arguments
                elif arguments:
                    raise ValueError("dialogue routes may not supply execution arguments")
                decision = parse_decision(json.dumps({key: call[key] for key in ROUTE_SCHEMA["required"]},
                                                     ensure_ascii=False),
                                          signals=signals,
                                          pending_reference=context.get("pending_reference"),
                                          diagnostics=self.last_call)
                decision = normalize_entity_choices(decision, signals, diagnostics=self.last_call)
                if discarded_noop_arguments:
                    self.last_call.setdefault("normalizations", []).append(
                        "discarded_noop_dialogue_arguments")
                    self.last_call["normalized_arguments"] = {}
            else:
                skill = "none" if route == "UNSUPPORTED" else route
                if (call["query"] != "" or call["reply"] != ""
                        or call["entity_choices"] != []
                        or call["followup"] is not False or call["intent"] != "general"
                        or call["detail"] is not False):
                    raise ValueError("non-dialogue routes may not contain dialogue fields")
                if skill == "none":
                    if arguments:
                        raise ValueError("unsupported routes take no arguments")
                else:
                    arguments = self.skills[skill].normalize_arguments(arguments)
                decision = None
            self.last_call["selected_skill"] = skill
            self.last_call["wall_s"] = round(perf_counter() - started, 3)
            print("[LLMPlanner] 统一决策性能：", {key: value for key, value in self.last_call.items()
                  if key not in ("raw_output", "parsed_plan", "normalized_decision")})
            print("[LLMPlanner] 模型原始输出:", raw)
            if skill == "none":
                return None
            answer = {"skill": skill, "arguments": arguments}
            if decision is not None:
                answer["npc_decision"] = decision
            return answer
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            timed_out = (isinstance(error, TimeoutError)
                         or isinstance(getattr(error, "reason", None), TimeoutError))
            self.last_call = {**self.last_call, "phase": "unified_plan", "model": self.model,
                              "status": "error", "wall_s": round(perf_counter() - started, 3),
                              "failure_stage": self.last_call.get("failure_stage", stage),
                              "error_kind": "timeout" if timed_out else "planning_error",
                              "error": f"{type(error).__name__}: {error}"}
            print("[LLMPlanner] 统一决策失败；未执行任何技能：", self.last_call)
            return None
