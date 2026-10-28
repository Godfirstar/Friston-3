"""一轮对话的输入和诊断，由 Agent 显式传入，不在 NPC 中排队。

ConversationState 保存多轮历史，DialogueTurn 只描述当前一次调用。
这里的上下文由程序提供，不属于模型可以填写的技能参数。
"""
from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter

from friston3.npc.identity import runtime_relationship_evidence


@dataclass
class DialogueTurn:
    input_text: dict
    route_context: dict = field(default_factory=dict)
    decision: dict | None = None
    planner_input: list = field(default_factory=list)
    planner_call: dict = field(default_factory=dict)
    started_at: float = field(default_factory=perf_counter)


def planning_failure(turn, interlocutor):
    """规划失败留下诊断，不产生正常对话历史或设备执行记录。"""
    call = turn.planner_call
    failed = call.get('status') == 'error'
    timed_out = call.get('error_kind') == 'timeout'
    if timed_out:
        reply = '本轮规划请求超时，尚未执行任何设备动作。输入“查看证据”可查看诊断。'
    elif failed:
        reply = '本轮规划未完成，尚未执行任何设备动作。输入“查看证据”可查看诊断。'
    else:
        reply = '这条指令暂不能执行；没有调用设备。'
    calls = [call] if call.get('request_attempted') else []
    signals = turn.route_context.get('signals', {})
    elapsed = round(perf_counter() - turn.started_at, 3)
    trace = {
        'question': turn.input_text['raw_text'],
        'normalized_question': turn.input_text['normalized_text'],
        'input_text': deepcopy(turn.input_text), 'interlocutor': deepcopy(interlocutor),
        'route': 'planning_failed' if failed else 'unsupported_skill',
        'output_origin': 'planner_failure' if failed else 'unsupported_skill',
        'model_called': bool(calls), 'generation_accepted': False,
        'fallback_used': False, 'execution_started': False,
        'reply': reply, 'subjects': [], 'sent_sources': [],
        'retrieved_source_ids': [], 'used_source_ids': [],
        'semantic_review_performed': False, 'model_input': deepcopy(turn.planner_input),
        'planner_call': deepcopy(call), 'ollama_calls': deepcopy(calls),
        'agentic_rag': {
            'decision': 'UNAVAILABLE' if failed else 'UNSUPPORTED',
            'reason': 'planner_timeout' if timed_out else 'planner_failure' if failed else 'unsupported_skill',
            'model_decision': None,
            'semantic': deepcopy(signals.get('semantic_evidence_demand', {})),
            'entity_resolution': deepcopy(signals.get('entity_resolution', {})),
            'retrieval_calls': 0, 'model_calls': len(calls),
            'model_phases': [item['phase'] for item in calls], 'elapsed_s': elapsed,
        },
        'elapsed_s': elapsed, 'agent_elapsed_s': elapsed,
    }
    return {'success': False, 'message': reply}, trace


def finish_trace(turn, trace, route, model_calls, retrieval_calls, interlocutor):
    """所有回复路径共用的诊断收尾，返回适合打印的简短摘要。"""
    calls = ([turn.planner_call] if turn.decision is not None else []) + model_calls
    elapsed = round(perf_counter() - turn.started_at, 3)
    route.update(retrieval_calls=retrieval_calls, model_calls=len(calls),
                 npc_model_calls=len(model_calls), model_phases=[item['phase'] for item in calls],
                 elapsed_s=elapsed, fallback_used=bool(route.get('fallback_used') or trace.get('fallback_used')))
    trace.update(agentic_rag=route, ollama_calls=calls, elapsed_s=elapsed,
                 agent_elapsed_s=elapsed, model_called=bool(calls),
                 question=turn.input_text['raw_text'],
                 normalized_question=turn.input_text['normalized_text'],
                 input_text=deepcopy(turn.input_text), interlocutor=deepcopy(interlocutor),
                 runtime_relationship_evidence=runtime_relationship_evidence())
    if turn.input_text['input_source'] == 'speech':
        trace['speech_span_basis'] = 'normalized_question'
    trace.setdefault('semantic_review_performed', False)
    trace.setdefault('used_source_ids', [])
    hidden = {'input', 'chat_input', 'rejected_chat_reply', 'recovery_query',
              'entity_resolution', 'speech_entity_candidates', 'speech_selections'}
    summary = {key: value for key, value in route.items() if key not in hidden}
    summary.update(output_route=trace.get('route'), output_origin=trace.get('output_origin'),
                   generation_accepted=trace.get('generation_accepted', False),
                   additional_retrievals=trace.get('additional_retrievals', 0),
                   semantic_review_performed=trace['semantic_review_performed'])
    for key in ('generation_error', 'repair_stop_reason'):
        if trace.get(key):
            summary[key] = trace[key]
    return summary
