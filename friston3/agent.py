"""输入规范化 → 准备背景 → 一次规划 → 校验并执行技能。

本轮可信上下文随调用传递，NPC 无需预先注入或事后清理。
"""
from time import perf_counter

from friston3.input_text import prepare_input_text, is_memory_command
from friston3.npc.turn import DialogueTurn


class Agent:
    def __init__(self, skill_manager, planner):
        self.skill_manager = skill_manager
        self.planner = planner
        self.npc = skill_manager.skills['npc_talk']

    def handle(self, instruction, *, input_source='text', interlocutor=None):
        if not isinstance(instruction, str) or not instruction.strip():
            return {'success': False, 'message': '请输入具体内容。'}
        input_text = prepare_input_text(instruction, input_source=input_source)
        instruction = input_text['normalized_text']
        if len(instruction) > 500:
            return {'success': False, 'message': '单次输入请控制在500字以内。'}
        started = perf_counter()
        print(f'[Agent] 收到指令: {instruction}')
        self.npc.set_interlocutor(interlocutor)
        try:
            if is_memory_command(instruction):
                # 显式记忆命令由 NPC 的权限检查处理，无需模型规划。
                turn = DialogueTurn(input_text=input_text, started_at=started)
                call = {'skill': 'npc_talk', 'arguments': {'user_text': instruction}}
            else:
                route_context = self.npc.prepare_route_context(
                    instruction, input_source=input_text['input_source'])
                route_context['input_text'] = input_text
                call = self.planner.plan(instruction, dialogue_context=route_context)
                turn = DialogueTurn(
                    input_text=input_text, route_context=route_context,
                    decision=call.get('npc_decision') if call else None,
                    planner_input=self.planner.last_input,
                    planner_call=self.planner.last_call, started_at=started)
                if call is None:
                    return self.npc.record_planning_failure(turn)
                if call['skill'] == 'npc_talk':
                    call = {'skill': 'npc_talk', 'arguments': {'user_text': instruction}}
                elif call['skill'] == 'control_arms' and call['arguments'].get('operation') == 'task':
                    call['arguments']['instruction'] = instruction
            print(f'[Agent] 工具调用: {call}')
            return self.skill_manager.execute(
                call['skill'], call['arguments'],
                context=turn if call['skill'] == 'npc_talk' else None)
        finally:
            elapsed = perf_counter() - started
            print(f'[Agent] 本轮总耗时 {elapsed:.3f}s（含规划与执行，不含语音合成/播放）')
