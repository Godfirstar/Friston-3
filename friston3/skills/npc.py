"""对话技能：组织一轮回复，并把状态、规划和证据交给对应模块。

入口 execute 接收本轮上下文；session 保存当前说话者的多轮历史。
主要流程在 _execute：记忆命令 → 设备经历 → 路由 → 检索 → 生成与核对。
模型 HTTP 在 friston3.llm，检索在 knowledge，诊断在 npc.turn。
"""
import json
import re
import sqlite3
from hashlib import sha256
from contextlib import closing
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from time import perf_counter

from friston3.config import (OLLAMA_CHAT_URL, NPC_CONTEXT_TOKENS,
                            NPC_REPLY_TOKENS, NPC_HISTORY_TURNS, KNOWLEDGE_DB,
                            OLLAMA_MODEL, OLLAMA_OPTIONS)
from friston3.knowledge.terra_kb import retrieve_packet, reply_entity_mentions, connect as connect_lore
from friston3.npc.dialogue import (EXPLANATIONS, bare_text, discourse_act,
                                 reference_description, conversation_act)
from friston3.npc.identity import (
    IDENTITY, SELF_SUBJECT, IDENTITY_SOURCE, OWNER_SUBJECT,
    normalize_interlocutor, interlocutor_prompt, runtime_relationship_evidence,
    configured_identity_evidence,
)
from friston3.npc.memory import MemoryStore
from friston3.npc.persona import (
    PRESENTATION_PROMPT, normalize_presentation, spoken_response_schema,
)
from friston3.npc.response_guard import classify_self_question, clean_reply, validate_reply
from friston3.knowledge.lore_text import source_spans
from friston3.knowledge.entity_resolver import resolve_entity_resolution
from friston3.knowledge.speech_entities import (
    SpeechEntityDependencyError, candidates as speech_entity_candidates,
    prepare as prepare_speech_entities, vocabulary_stats,
)
from friston3.npc.agentic_rag import (
    AgenticSettings, ROUTE_TOKENS, build_route_input, parse_decision, _unique_object,
    normalize_entity_choices, route_schema_for_signals,
)
from friston3.npc.intent_router import SemanticIntentRouter, SemanticGuess
from friston3.input_text import prepare_input_text
from friston3.llm import chat, estimate_tokens, performance_stats
from friston3.npc.session import ConversationState
from friston3.npc.turn import DialogueTurn, planning_failure, finish_trace
from friston3.npc.prompts import (
    PromptBuilder, PRESENTED_REPLY_TOKENS, history_messages, subject_key,
)


class NpcTalkSkill:
    name = "npc_talk"
    @staticmethod
    def normalize_arguments(arguments, *, defaults=False):
        if set(arguments) != {'user_text'}:
            raise ValueError('对话只接受 user_text')
        text = arguments['user_text']
        if type(text) is not str or not 1 <= len(text) <= 500:
            raise ValueError('对话内容必须为 1 至 500 字的文字')
        return {'user_text': text}

    def __init__(self, model=None, url=OLLAMA_CHAT_URL, *, memory_path=None, db_file=KNOWLEDGE_DB):
        # Routing and dialogue share the resident model; camera execution
        # remains behind the existing skill interfaces.
        self.model = model or OLLAMA_MODEL
        self.url = url
        self.db_file = db_file
        self.session = ConversationState()
        self._model_calls = []
        self.agentic = AgenticSettings.defaults()
        self._route_trace = {}
        self._retrieval_calls = 0
        self._turn = None
        self._speech_entity_error = ''
        self.interlocutor = normalize_interlocutor()
        self._active_profile = self.interlocutor['profile_id']
        self._owner_session = None
        try:
            self._owner_memory = MemoryStore(memory_path)
        except (OSError, sqlite3.Error) as error:
            self._owner_memory = None
            print(f"[NpcTalkSkill] 长期记忆不可用，本次仅保留短期话题：{error}")
        # Legacy records belong to the private owner scope. An unidentified
        # speaker never gets them just because this is the same Python process.
        # Load weights before the CLI accepts its first question. Model loading
        # is startup work; it must not consume a visitor's answer budget.
        self.startup_error = ''
        print(f'[NpcTalkSkill] 模型={self.model}；路由等待上限={self.agentic.route_timeout:g}s；'
              f'回答等待配置={self.agentic.turn_timeout:g}s（故障预算，不是响应速度）')
        warm_started = perf_counter()
        try:
            chat(self._prompts().chat('你好', ''), model=self.model, url=self.url, timeout=60,
                 options={**OLLAMA_OPTIONS, 'num_predict': 1, 'temperature': 0})
            print(f'[NpcTalkSkill] 模型预热完成，耗时 {perf_counter() - warm_started:.2f}s')
        except (OSError, ValueError) as error:
            self.startup_error = f'{type(error).__name__}: {error}'
            print(f'[NpcTalkSkill] 模型预热失败：{self.startup_error}')
        # Eagerly prepare the small, shared-encoder semantic router BEFORE the
        # interactive prompt, so visitors do not pay its first-use embedding cost.
        # Missing FastEmbed/models are non-fatal: the existing LLM route remains.
        self.semantic_router = SemanticIntentRouter()
        if not self.semantic_router.error:
            started = perf_counter()
            try:
                self.semantic_router.prepare()
                # Load the immutable lore index and alias tables before the
                # first visitor asks about a character; neither call writes data.
                try:
                    from friston3.knowledge.semantic_index import search as warm_lore_index
                    warm_lore_index(IDENTITY.name, top_k=1, db_file=self.db_file)
                    with closing(connect_lore(self.db_file)) as db:
                        resolve_entity_resolution(db, IDENTITY.name)
                except (OSError, ValueError, sqlite3.Error, RuntimeError) as error:
                    print(f'[NpcTalkSkill] 知识索引预热未完成，将按需检索：{error}')
                print(f"[IntentRouter] 语义意图索引已准备好，启动耗时 {perf_counter()-started:.2f}s")
            except Exception as error:
                self.semantic_router.error = f"{type(error).__name__}: {error}"
                print(f"[IntentRouter] 语义层不可用，交给现有 LLM 路由：{self.semantic_router.error}")

        try:
            with closing(connect_lore(self.db_file)) as db:
                prepare_speech_entities(db)
        except (SpeechEntityDependencyError, OSError, ValueError, sqlite3.Error, RuntimeError) as error:
            self._speech_entity_error = f'{type(error).__name__}: {error}'
            print(f'[NpcTalkSkill] 语音实体候选未启用，原始转写仍可使用：{error}')

    def set_interlocutor(self, context=None):
        """切换整个会话：保存主人状态，新访客使用独立临时记忆。"""
        checked = normalize_interlocutor(context)
        profile = checked['profile_id']
        if profile != self._active_profile:
            if self._active_profile == 'owner':
                self._owner_session = self.session
            if profile == 'owner':
                self.session = self._owner_session or ConversationState(memory=self._owner_memory)
                self._owner_session = None
            else:
                self.session = ConversationState()
            self._active_profile = profile
        self.interlocutor = checked


    def _prompts(self):
        return PromptBuilder(self.session, self.interlocutor, self._source_stamp())

    def clear_history(self):
        """重建短期会话，保留显式保存的记忆。"""
        self.session = ConversationState(memory=self.session.memory)


    def _raw_user_text(self, user_text):
        current = self._turn.input_text if self._turn else {}
        return current['raw_text'] if current.get('normalized_text') == user_text else user_text


    def clear_all_memory(self):
        if not self.interlocutor['local_operator'] or self.interlocutor['role'] != 'owner':
            raise RuntimeError("长期记忆管理需在主人本地控制台操作")
        if self._owner_memory is None:
            raise RuntimeError("长期记忆文件未成功打开，无法确认已经清空")
        counts = self._owner_memory.clear_all()
        self._owner_session = None
        self.clear_history()
        return counts

    def record_execution(self, skill_name, result):
        if self.session.memory is None:
            return
        try:
            self.session.memory.record_execution(skill_name, result)
        except (OSError, sqlite3.Error) as error:
            print(f"[NpcTalkSkill] 执行记录未保存：{error}")

    def explain_last_response(self):
        if not self.session.trace:
            return "本次会话还没有可查看的回复依据。"
        return "【当前会话的上轮身份、实际输入与处理路径；相似度不是身份概率，规则通过不等于事实认证】\n" + json.dumps(
            self.session.trace, ensure_ascii=False, indent=2)

    @staticmethod
    def _real_history_kind(user_text, kind):
        """Past physical events belong to execution records, even if asserted by a user."""
        past = re.search(r'刚才|刚刚|上次|昨天|前天|已经|之前|实际|现实', user_text)
        device = re.search(r'你|我们|这台设备|这个终端', user_text)
        if past and device:
            if re.search(r'拍(?:了|过|好|完|到|的|成功)|拍摄|拍照|照片.{0,10}(?:保存|成功|完成|失败)', user_text):
                return 'self_task_history'
            if re.search(r'见面|会面|见过|见了|遇见|遇到|接待', user_text):
                return 'self_encounter'
            if re.search(r'(?:一起|一同|共同).{0,5}(?:工作|共事|逛|玩|聊天)', user_text):
                return 'self_encounter'
            if re.search(r'去过|去了|到过|旅行|出差', user_text):
                return 'self_travel'
        if kind == 'self_task_history' or (kind.startswith('self_') and
                any(w in user_text for w in ('这台设备', '现实', '实际', '上次', '刚才', '昨天', '今天', '我们'))):
            return kind
        return ''


    def _clarified_question(self, user_text, packet):
        """Resolve a bare selection in the original task, preserving both endpoints.

        An explicit new question must keep its own intent. Only bounded name
        selections are substituted; neither model text nor an unrelated earlier
        topic can supply a query rewrite.
        """
        if not self.session.pending_reference or packet.get('resolution', {}).get('ambiguities'):
            return None
        pending_ids = set(self.session.pending_reference.get('candidate_ids', []))
        mentions = list(packet.get('resolution', {}).get('mentions', []))
        mentions.extend({'id': item['id'], 'name': item['name'], 'alias': item['name']}
                        for item in packet.get('speech_selections', []))
        choices = [item for item in mentions
                   if not pending_ids or item['id'] in pending_ids]
        if len({item['id'] for item in choices}) != 1:
            return None
        aliases = '|'.join(re.escape(bare_text(item['alias'])) for item in choices)
        if not re.fullmatch(r'(?:我(?:说|指)(?:的)?(?:是)?|是|应该是|就是|选择)?(?:' + aliases
                            + r')(?:吧|呀|啊|呢|才对|对吧)?', bare_text(user_text)):
            return None
        selected = choices[0]
        original = self.session.pending_reference['question']
        span = self.session.pending_reference.get('speech_span')
        if span is not None:
            start, end = span.get('start'), span.get('end')
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(original)
                    or original[start:end] != span.get('heard')):
                return None
            return original[:start] + selected['name'] + original[end:]
        rewritten = original
        for ambiguity in self.session.pending_reference.get('ambiguities', []):
            if any(candidate['id'] == selected['id'] for candidate in ambiguity['candidates']):
                rewritten = rewritten.replace(ambiguity['alias'], selected['name'])
        description = self.session.pending_reference.get('description')
        if description and not pending_ids:
            rewritten = rewritten.replace(description, selected['name'])
        return rewritten if rewritten != original else None

    def _confirmed_speech_question(self, db, user_text):
        """An explicit yes answers only a unique, still-valid pending name.

        This is a bounded confirmation protocol, not another ASR correction.
        The current utterance keeps its own raw text and has no invented span.
        """
        pending = self.session.pending_reference
        ids = pending.get('candidate_ids', [])
        if (not pending.get('speech_confirmable') or len(ids) != 1
                or bare_text(user_text) not in ('对', '是的', '没错')):
            return None, []
        choices = [item for group in pending.get('ambiguities', [])
                   for item in group.get('candidates', []) if item.get('id') == ids[0]]
        row = db.execute('SELECT id,name FROM entities WHERE id=?', (ids[0],)).fetchone()
        if not row or not choices or any(item.get('name') != row['name'] for item in choices):
            return None, []
        original, span = pending.get('question', ''), pending.get('speech_span', {})
        start, end = span.get('start'), span.get('end')
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(original)
                or original[start:end] != span.get('heard')):
            return None, []
        return original[:start] + row['name'] + original[end:], [dict(row)]


    def _source_stamp(self):
        paths = (Path(self.db_file), Path(str(self.db_file) + '-wal'),
                 Path(__file__).resolve().parents[2] / 'data/knowledge/npc_character.json')
        return tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.is_file() else None for p in paths)

    def _remember(self, user_text, reply, packet=None, sent_sources=(), *, persist=True):
        raw_text = self._raw_user_text(user_text)
        record = {"role": "user", "content": raw_text}
        if raw_text != user_text:
            record['normalized_content'] = user_text
        self.session.history.append(record)
        self.session.history.append({"role": "assistant", "content": reply})
        packet = packet or {}
        subjects = packet.get("subjects", [])
        # A named participant in an actually used self-relation is a valid
        # antecedent too ("who built you" -> "did she create your personality").
        if subjects and all(s['id'] == IDENTITY.entity_id for s in subjects):
            subjects = list(subjects)
            for source in sent_sources:
                target = source.get('object_id')
                name = source.get('object_name')
                if (target and name and name in reply
                        and target not in {s['id'] for s in subjects}):
                    subjects.append({'id': target, 'name': name, 'kind': 'character'})
        acknowledgement = (conversation_act(user_text) == 'thanks' or
                           bare_text(user_text) in ('好的', '好', '哈哈', '嗯', '明白了'))
        keep_topic = not subjects and acknowledgement and bool(self.session.topic)
        if not subjects and acknowledgement:
            subjects = self.session.topic.get('subjects', [])
        knowledge_topic = bool(packet.get('lore_question') or sent_sources)
        topic_key = subject_key(subjects)
        if not subjects and (knowledge_topic or keep_topic):
            query = (self.session.topic.get('retrieval_question', '') if keep_topic else
                     packet.get('retrieval_question', user_text))
            topic_key = frozenset({'query:' + query.removesuffix('；进一步解释原因与依据')})
        self.session.history_topics.append(topic_key)
        if keep_topic:
            pass  # A social acknowledgement does not replace the pending lore task/evidence.
        elif subjects or knowledge_topic:
            same_topic = subject_key(subjects) == subject_key(self.session.topic.get('subjects', []))
            # Preserve only a selector for trusted configuration, never model prose.
            persona_hint = self.session.topic.get("persona_hint", "") if same_topic else ""
            self.session.topic = {
                "question": user_text, "intent": packet.get("intent", "general"),
                "retrieval_question": packet.get('retrieval_question', user_text),
                "context_root": packet.get('context_root', packet.get('retrieval_question', user_text)),
                "subjects": subjects, "source_ids": [item["id"] for item in sent_sources],
                "sources": list(sent_sources),
                "source_stamp": self._source_stamp(),
                "persona_hint": (persona_hint or user_text) if packet.get("intent") == "personality" else "",
            }
        elif not acknowledgement:
            self.session.topic.clear()
        if (subjects or knowledge_topic) and not keep_topic:
            # Referents live with short-term conversation, independently of
            # whether old evidence may still be reused.
            self.session.discourse_topic = {key: deepcopy(self.session.topic.get(key)) for key in
                                     ('question', 'retrieval_question', 'context_root', 'intent', 'subjects')}
            self.session.discourse_turn = self.session.turn_number
        if self.session.memory is not None and persist:
            try:
                self.session.memory.log_turn(raw_text, reply)
            except (OSError, sqlite3.Error) as error:
                print(f"[NpcTalkSkill] 对话记录未保存：{error}")

    def _direct_reply(self, user_text, reply, route, subjects=(), records=(), sources=(), **trace):
        """Only real device/memory status may bypass dialogue generation."""
        packet = {"subjects": list(subjects), "intent": route}
        self.session.trace = {
            "question": user_text, "route": route, "model_called": False,
            "generation_accepted": False, "output_origin": "execution_record", "fallback_used": False,
            "subjects": list(subjects), "sent_sources": [], "used_source_ids": [source["id"] for source in sources],
            "history_sent": [], "raw_assistant_history_sent": False,
            "identity_source": IDENTITY_SOURCE,
            "rule_source": "friston3/skills/npc.py",
            "execution_records": list(records), "reply": reply, **trace,
        }
        return self._finish_reply(user_text, reply, packet, sources=sources)

    def _memory_command(self, user_text):
        match = re.fullmatch(r"\s*(记住|忘记)\s*[:：]\s*(.*?)\s*", user_text, flags=re.S)
        if match is None:
            return None
        action, statement = match.groups()
        if not self.interlocutor['local_operator'] or self.interlocutor['role'] != 'owner':
            reply = "长期记忆由主人在本地控制台管理；这句话可以继续聊，但不会改动已保存的记录。"
            self.session.trace = {'route': 'memory_permission', 'model_called': False,
                               'output_origin': 'local_permission', 'reply': reply,
                               'generation_accepted': False, 'fallback_used': False}
            return {'success': False, 'message': reply}
        if not statement:
            return {"success": False, "message": f"请在“{action}：”后写下具体内容。"}
        if self.session.memory is None:
            return {"success": False, "message": "长期记忆文件未成功打开，这条内容还没有保存或删除。"}
        try:
            if action == "记住":
                self.session.memory.remember(statement)
                reply = "记下了，这是你告诉我的信息。以后相关时我会参考它。"
                self._remember(user_text, reply)
                self.session.pending_reference.clear()
            else:
                count = self.session.memory.forget(statement)
                self.clear_history()
                reply = f"已删除 {count} 条匹配的长期记忆，并清空本次短期对话。" if count else "没有找到匹配的长期记忆；本次短期对话已清空。"
            self.session.trace = {"route": "explicit_memory_command", "model_called": False, "reply": reply,
                               "generation_accepted": False, "output_origin": "memory_command", "fallback_used": False}
            return {"success": True, "message": reply}
        except (OSError, sqlite3.Error) as error:
            return {"success": False, "message": f"记忆操作未完成：{error}"}

    def _self_history_reply(self, user_text, kind):
        subjects = [dict(SELF_SUBJECT)]
        if kind in ("self_encounter", "self_travel"):
            # Camera events and user reports prove neither a meeting nor a journey.
            thing = "会面" if kind == "self_encounter" else "出行"
            return self._direct_reply(user_text, f"我这里没有可核实的{thing}记录。资料里的故事我能聊，但不能说成自己亲历过。", kind, subjects)
        if self.session.memory is None:
            return self._direct_reply(user_text, "执行记录现在读不到，我不能确认自己做过哪些任务。", kind, subjects)
        try:
            events = self.session.memory.recent_execution_events(limit=1)
        except (OSError, sqlite3.Error) as error:
            print(f"[NpcTalkSkill] 执行记录读取失败：{error}")
            return self._direct_reply(user_text, "执行记录现在读不到，我不能确认自己做过哪些任务。", kind, subjects)
        if not events:
            return self._direct_reply(user_text, "我还没找到这次任务的实际执行记录，不能确认拍照或保存成功。", kind, subjects)
        event = events[0]
        label = "自动拍照" if event["skill"] == "auto_photo" else "普通拍照"
        outcome = "成功" if event["success"] else "失败"
        reply = f"最近保存的一次执行记录是{label}，结果为{outcome}。"
        return self._direct_reply(user_text, reply, kind, subjects, events)


    def _normalize_grounded_reply(self, reply, packet, sent_sources, user_text):
        """Normalize wording without inventing a fact or another cause.

        In an introduction, a bare acknowledgement means 'I know of the
        person', not that the speaker met them. Its removal is limited to
        introductions supported by an explicit fact about that person.
        """
        proof = '\n'.join(s.get('text', '') for s in sent_sources)
        if '只为' in reply and sent_sources and not re.search(r'只为|唯一|仅仅为了|只是为了', proof):
            # Weakening unsupported exclusivity preserves the underlying
            # statement. The raw model draft remains in the evidence trace.
            reply = reply.replace('只为', '为了')
            self.session.trace['wording_adjustment'] = 'Removed unsupported motive exclusivity: 只为 -> 为了'
        subjects = [s for s in packet.get('subjects', []) if s.get('id') != IDENTITY.entity_id]
        if (packet.get('intent') != 'introduction' or len(subjects) != 1
                or not any(w in user_text for w in ('认识', '了解', '知道'))
                or not any(s.get('subject_id') == subjects[0]['id'] or
                           (s.get('kind') == 'relation' and s.get('object_id') == subjects[0]['id'])
                           for s in sent_sources)):
            return reply
        candidate = re.sub(r'^\s*(?:当然)?(?:认识|知道|了解)(?:这个名字)?[，,。！!]\s*', '', reply, count=1)
        if candidate != reply and subjects[0]['name'] in candidate:
            return candidate
        return reply

    @staticmethod
    def _only_evidence_gap(reply):
        """An empty source selection may explain a gap, never add a fact."""
        return (isinstance(reply, str)
                and bool(re.search(r'未找到|缺少|没有足够|无法确认|不能确认|尚不清楚|资料未提及', reply))
                and len(re.findall(r'[^。！？!?]+', reply.strip('。！？!?'))) == 1
                and not re.search(r'但是|不过|然而|因为|所以|实际上|，但', reply))

    def _draft_issues(self, reply, packet, sent_sources, *, source_excerpt=False, diagnostics=None):
        user_text = self.session.trace.get('answer_question', self.session.trace.get('question', ''))
        # Reliable snippets cannot license a denial that any information
        # about the requested person exists. A disclaimer cannot erase
        # facts that are actually present in the supplied evidence.
        subjects = [s for s in packet.get('subjects', []) if s.get('id') != IDENTITY.entity_id]
        if packet.get('intent') == 'introduction' and len(subjects) == 1:
            name = subjects[0]['name']
            has_subject_evidence = any(s.get('subject_id') == subjects[0]['id'] or
                                       s.get('kind') == 'relation' and s.get('object_id') == subjects[0]['id']
                                       for s in sent_sources)
            if (has_subject_evidence and re.search(
                    r'(?:不明确[。！!，,]\s*)?(?:资料(?:中|里)?(?:没有|无)|(?:没有|无|未找到))\s*(?:关于|涉及)?'
                    + re.escape(name) + r'.{0,18}(?:资料|信息|关联)|'
                    r'^\s*(?:不认识|不清楚|不知道|不明确)[。！!，,]\s*(?:资料|没有|无)', reply)):
                intro_gap_issue = True
            else:
                intro_gap_issue = False
        else:
            intro_gap_issue = False
        # Source separates the designed body from the inherited personality.
        # 'She designed me' is therefore too broad; 'my body' is supported.
        # The possessive particle is optional: 我机体 and 我的机体 name the same object.
        body_designer = (any(s.get('id') == IDENTITY.creator_id for s in subjects)
                         or IDENTITY.creator in reply)
        body_source = any('人格数据不是' in s.get('text', '') and '机体' in s.get('text', '')
                          for s in sent_sources)
        if body_designer and body_source and re.search(
                r'(?:' + re.escape(IDENTITY.creator) + r'|她|他|对方).{0,10}(?:设计|组装|制造|创造|研发)(?:了)?我(?!的?(?:机体|身体)|这具机体)', reply):
            body_identity_issue = True
        else:
            body_identity_issue = False
        runtime_chat = (not packet.get('subjects') and not packet.get('lore_question')
                        and packet.get('intent') == 'general' and not sent_sources)
        guard_reply = reply
        if self.interlocutor['role'] == 'owner' and not runtime_chat:
            # A verified addressee's vocative is not a claim about the game
            # character. Remove only a sentence-leading address before the
            # lore fact check, keeping every factual clause and the raw reply.
            # E.g. "可露希尔，多萝西..." still checks everything about 多萝西.
            guard_reply = re.sub(
                r'(^|[。！？!?])\s*(?:' + re.escape(OWNER_SUBJECT['name'])
                + r'|总工程师|主人)[，,]\s*', r'\1', reply)
            if guard_reply != reply:
                self.session.trace['guard_vocative_normalized'] = True
        mentions = reply_entity_mentions(guard_reply, db_file=self.db_file) if guard_reply else []
        candidates = [c for a in packet.get('resolution', {}).get('ambiguities', [])
                      for c in a['candidates']]
        guard_subjects = list(packet['subjects'])
        guard_sources = list(sent_sources)
        # The current exhibit maker is a runtime person, not the game's
        # canonical character. Plain social CHAT has this relation in its
        # actual system input; use that same fact without inventing a wiki edge
        # or disabling the guard for unrelated character claims.
        if runtime_chat:
            guard_sources += configured_identity_evidence() + runtime_relationship_evidence()
            mentions = [
                {**mention, 'id': OWNER_SUBJECT['id'], 'scope': 'runtime'}
                if mention.get('name') == OWNER_SUBJECT['name'] else mention
                for mention in mentions]
            if self.interlocutor['role'] == 'owner':
                guard_subjects.append(dict(OWNER_SUBJECT))
        issues = validate_reply(guard_reply, guard_subjects, guard_sources, mentions,
                                user_text=self.session.trace.get('answer_question', self.session.trace.get('question', '')),
                                act=packet.get('conversation_act', ''), candidate_entities=candidates,
                                source_excerpt=source_excerpt)
        if self.interlocutor['role'] != 'owner' and re.search(
                r'(?:^|[。！？!?])\s*(?:主人|可露希尔|总工程师)[，,！!：:]|'
                r'(?:你好|您好|早安|晚安|谢谢你?|欢迎回来|辛苦了)[，,\s]*(?:主人|可露希尔|总工程师)(?=[。.!！?？\s]|$)|'
                r'(?:你|您)(?:就是|是)(?:我的?|本机的?)?(?:主人|制造者|维护者|可露希尔)', reply):
            issues.append('unassigned_user_role:owner')
        if intro_gap_issue:
            issues.append('intro_ignores_available_evidence')
        if body_identity_issue:
            issues.append('body_personality_conflation')
        # Definition syntax is a retrieval hint, not a truth criterion.
        # _review_grounded_draft checks synthesis against the delivered text.
        if sent_sources:
            unquoted = re.sub(r'[“「『\"][^”」』\"]*[”」』\"]', '', reply)
            chronology = r'后来|随后|接着|继而|转任|改任|之后|此前|最初|最终|先.{0,35}再'
            proof = '\n'.join(source.get('text', '') for source in sent_sources)
            if (re.search(r'只为|唯一(?:的)?(?:原因|目的|动机)', reply)
                    and not re.search(r'只为|唯一|仅仅为了|只是为了', proof)):
                issues.append('unsupported_exclusive_motive')
            # A statement about a gap in the sources is not an assertion of
            # chronology. Keep the exception local, so a disclaimer before a
            # comma cannot license an unsupported assertion after it.
            uncertain = r'是否|不清楚|未说明|没有说明|未交代|未提及|没有提及|无法确定|不能确定|不确定|尚不明确|缺少|缺乏|未提供'
            temporal_claims = [clause for clause in re.split(r'[，,。！？!?；;]|然而|但是|不过|可是|但|却', unquoted)
                               if re.search(chronology, clause) and not re.search(uncertain, clause)]
            if (temporal_claims and not re.search(
                    chronology + r'|后.{0,12}(?:担任|成为|加入|离开)|开端|转折|转变|\d{4}年', proof)):
                issues.append('unsupported_temporal_order')
                if diagnostics is not None:
                    diagnostics.append({
                        'code': 'unsupported_temporal_order', 'fragments': temporal_claims,
                        'triggers': [m.group() for clause in temporal_claims for m in re.finditer(chronology, clause)],
                        'source_ids': [source['id'] for source in sent_sources],
                        'instruction': '所选资料没有说明这些片段中的先后顺序。删除无依据的时间限定，'
                                       '保留有依据的动作和关系；不要换成另一时间词，也不要把未说明改成否定事件发生。'})
            for temporal in re.finditer(r'(?:曾任|曾经担任|过去担任)\s*([^，,。！？!?；;]{2,35})', unquoted):
                role = temporal.group(1).strip()
                if (role and role in proof and not re.search(
                        r'(?:曾任|曾经担任|过去担任|原任|此前担任|前)\s*' + re.escape(role), proof)):
                    issues.append('unsupported_temporal_order')
                    if diagnostics is not None:
                        diagnostics.append({
                            'code': 'unsupported_temporal_order', 'fragments': [temporal.group()],
                            'source_ids': [source['id'] for source in sent_sources],
                            'instruction': '资料只支持这项职务，没有支持曾任或过去任职。保留有依据的职责，去掉新增任职时期。'})
                    break
        if candidates and not (all(c['name'] in reply for c in candidates)
                               and re.search(r'谁|哪位|哪个|还是|[？?]', reply)):
            issues.append('missing_reference_clarification')
        current_act = packet.get('conversation_act', 'general')
        current_topic = subject_key(packet['subjects'])
        history = history_messages(self.session)
        if (len(history) >= 2 and len(reply) >= 12 and reply == history[-1]['content']
                and user_text != history[-2]['content']):
            issues.append('repeated_previous_reply')
        # Repeating a fact within its topic is normal. Only reject copying
        # across clearly different people or different explicit social acts.
        for index, old_topic in enumerate(self.session.history_topics):
            old_act = conversation_act(history[2 * index]['content'])
            changed_people = (bool(current_topic and old_topic)
                and current_topic.isdisjoint(old_topic)
                and not any(str(key).startswith('query:') for key in old_topic))
            changed_social = (not current_topic and not old_topic
                and current_act != 'general' and old_act != 'general' and current_act != old_act)
            if len(reply) < 40 or not (changed_people or changed_social):
                continue
            old_reply = history[2 * index + 1]['content']
            match = SequenceMatcher(None, old_reply, reply, autojunk=False).find_longest_match()
            if match.size >= 30 and match.size / len(reply) > 0.55:
                issues.append('repeated_previous_reply')
                break
        return list(dict.fromkeys(issues))

    @staticmethod
    def _repair_requirements(issues, diagnostics):
        """Turn local issue codes into actionable, question-independent feedback."""
        requirements = list(diagnostics)
        explained = {item['code'] for item in requirements}
        guidance = {
            'missing_source_selection': '为回答中的事实选择实际支持它的来源编号；无法支持的内容删去或限定。',
            'body_personality_conflation': '明确区分机体制造与人格来源；不要把机体的制造者说成人格或意识的创造者。',
            'unsupported_exclusive_motive': '资料没有支持唯一动机，删除唯一或只为等排他限定，保留有依据的原因。',
            'intro_ignores_available_evidence': '已有关于所问主体的资料，直接回答其中有依据的信息，不要笼统声称没有资料。',
            'repeated_previous_reply': '回应当前问题的具体内容，不照搬上一轮不同话题的回答。',
        }
        for issue in issues:
            if issue in explained:
                continue
            instruction = guidance.get(issue, '逐项核对所选资料的主体、限定及实际设备边界，改写不受支持的部分。')
            if issue.startswith('unsupported_named_entity:'):
                instruction = '所选来源没有支持回答中关于此人物的断言；核对来源归属，或去掉该断言：' + issue.split(':', 1)[1]
            requirements.append({'code': issue, 'instruction': instruction})
        return requirements

    def _model_result(self, messages, *, schema=None, max_tokens=NPC_REPLY_TOKENS,
                      phase='generation', timeout=None):
        """管理当前回合预算；HTTP 与响应解析交给共同模型入口。"""
        needed = sum(estimate_tokens(m['content']) + 8 for m in messages) + max_tokens + 256
        if needed > NPC_CONTEXT_TOKENS:
            raise ValueError('本轮资料超过可安全核对的上下文预算')
        remaining = self.agentic.turn_timeout - (perf_counter() - self._turn.started_at)
        if remaining < 0.25:
            raise TimeoutError('对话时限内没有足够的生成时间')
        request_timeout = min(remaining, timeout) if timeout else remaining
        options = {**OLLAMA_OPTIONS,
            'temperature': 0 if schema or phase.endswith('_repair') else 0.15,
            'top_p': 0.9, 'repeat_penalty': 1.0, 'presence_penalty': 0.0,
            'num_ctx': NPC_CONTEXT_TOKENS, 'num_predict': max_tokens}
        started = perf_counter()
        record = {'phase': phase, 'model': self.model, 'num_ctx': NPC_CONTEXT_TOKENS,
                  'timeout_s': round(request_timeout, 3)}
        try:
            result = chat(messages, model=self.model, url=self.url, timeout=request_timeout,
                          options=options, schema=schema, diagnostics=record)
        except (OSError, ValueError) as error:
            record.update(status='error', wall_s=round(perf_counter() - started, 3),
                          error=f'{type(error).__name__}: {error}')
            self._model_calls.append(record)
            print(f'[NpcTalkSkill] Ollama调用失败: {record}')
            raise
        record.update(status='ok', **performance_stats(result, perf_counter() - started))
        self._model_calls.append(record)
        print('[NpcTalkSkill] Ollama性能:', {key: value for key, value in record.items()
                                             if key != 'raw_output'})
        return result


    def _review_grounded_draft(self, question, reply, sent_sources, proposed, packet, act, history, *, can_search):
        """Same-model review/revision, not an independent verification of a revision."""
        messages, schema = self._prompts().review(
            question, reply, sent_sources, proposed, packet, act, history, can_search=can_search)
        result = self._model_result(messages, schema=schema, max_tokens=256, phase='evidence_review')
        raw_output = result['message']['content']
        trace = {'raw_output': raw_output, 'model_input': deepcopy(messages),
                 'normalizations': [], 'status': 'pending'}
        self.session.trace.setdefault('evidence_reviews', []).append(trace)

        def invalid(reason):
            trace.update(status='invalid', error=reason)
            raise ValueError(reason)

        try:
            review = json.loads(raw_output, object_pairs_hook=_unique_object)
        except (ValueError, TypeError) as error:
            trace.update(status='invalid', error=f'{type(error).__name__}: {error}')
            raise ValueError('证据复核没有返回有效JSON') from error
        trace['raw_review'] = deepcopy(review)
        labels = [source['prompt_label'] for source in sent_sources]
        if (not isinstance(review, dict) or set(review) != set(schema['required'])
                or review.get('verdict') not in schema['properties']['verdict']['enum']
                or not isinstance(review.get('sources'), list)
                or any(not isinstance(label, str) or label not in labels for label in review['sources'])
                or result.get('done_reason') in ('length', 'max_tokens', 'max_output_tokens', 'token_limit')):
            invalid('证据复核输出格式无效、来源编号无效、动作不可用或输出被截断')
        normalizations = trace['normalizations']
        if len(review['sources']) != len(set(review['sources'])):
            normalizations.append('deduplicated_source_labels')
        review['sources'] = list(dict.fromkeys(review['sources']))
        if review['verdict'] == 'accept':
            candidate = review['reply'].strip() if isinstance(review['reply'], str) else ''
            if candidate and candidate != reply.strip():
                # A different line is a same-model revision, not acceptance of
                # the original draft. It must use the existing rewrite guards.
                review['verdict'], review['reply'] = 'rewrite', candidate
                normalizations.append('accept_with_changed_reply_normalized_to_rewrite')
            else:
                if review['reply'] != '':
                    normalizations.append('discarded_unused_reply_for_accept')
                review['reply'] = ''
        if review['verdict'] == 'rewrite':
            if not isinstance(review['reply'], str) or not review['reply'].strip():
                invalid('证据复核rewrite缺少有效台词')
            review['reply'] = review['reply'].strip()
            if len(review['reply']) > schema['properties']['reply']['maxLength']:
                invalid('证据复核修订台词超过长度约定')
        elif review['verdict'] == 'search':
            if review['reply'] != '':
                normalizations.append('discarded_unused_reply_for_search')
            review['reply'] = ''
        if review['verdict'] == 'search':
            if (not can_search or not isinstance(review['query'], str)
                    or not 2 <= len(review['query'].strip()) <= 160):
                invalid('证据复核search不可用或缺少有效查询')
            review['query'] = review['query'].strip()
        else:
            if review['query'] != '':
                normalizations.append('discarded_unused_query_for_' + review['verdict'])
            review['query'] = ''
        # Feedback is advisory metadata, never an evidence source or an action
        # payload. Its shape cannot invalidate an otherwise usable decision.
        feedback = review['feedback'].strip() if isinstance(review['feedback'], str) else ''
        feedback = feedback[:schema['properties']['feedback']['maxLength']]
        if feedback != review['feedback']:
            normalizations.append('normalized_advisory_feedback')
        review['feedback'] = feedback
        trace.update(status='validated', normalized_review=deepcopy(review))
        self.session.trace['semantic_review_performed'] = True
        self.session.trace['semantic_review_scope'] = (
            '同一模型依据已送资料复核或修订；修订稿仅再经程序校验，不是独立事实认证')
        return review, messages

    def _extend_grounded_evidence(self, question, packet, query):
        """Query existing hybrid retrieval; preserve the user's subject and task."""
        self._retrieval_calls += 1
        started = perf_counter()
        try:
            extra = retrieve_packet(packet.get('resolved_question', question),
                                    resolution=packet['resolution'], search_query=query,
                                    intent=packet.get('intent'), max_chars=1500, db_file=self.db_file)
        except (OSError, sqlite3.Error, RuntimeError) as error:
            self.session.trace.setdefault('retrieval_rounds', []).append({
                'query': query, 'sources': [], 'error': f'{type(error).__name__}: {error}',
                'elapsed_s': round(perf_counter() - started, 3)})
            raise
        previous = {(source.get('original_evidence_id', source['id']), source.get('text', ''))
                    for source in packet.get('sources', [])}
        previous_ids = {key for key, _ in previous}
        # Keep different complete windows independently; never splice text or
        # discard earlier qualifications just because a passage ID repeats.
        fresh = []
        for source in extra.get('sources', []):
            base_id, text = source['id'], source.get('text', '')
            if (base_id, text) in previous:
                continue
            if base_id in previous_ids:
                source = {**source, 'original_evidence_id': base_id,
                          'id': base_id + ':excerpt:' + sha256(text.encode('utf-8')).hexdigest()[:12]}
            fresh.append(source)
            previous.add((base_id, text))
        packet['sources'] = fresh + list(packet.get('sources', []))
        packet['context'] = '\n'.join(source.get('block', source['text']) for source in packet['sources'])
        if packet['sources']:
            packet['missing_knowledge'] = False
        record = {'query': query, 'sources': deepcopy(extra.get('sources', [])),
                  'new_source_ids': [s['id'] for s in fresh if 'original_evidence_id' not in s],
                  'additional_excerpt_ids': [s['id'] for s in fresh if 'original_evidence_id' in s],
                  'retrieval': deepcopy(extra.get('retrieval', {})),
                  'elapsed_s': round(perf_counter() - started, 3)}
        self.session.trace.setdefault('retrieval_rounds', []).append(record)
        # Per-query ranking/coverage belongs to that round, not the merged pool.
        packet['retrieval'] = {
            'strategy': 'bounded_agentic_hybrid', 'additional_queries': len(self.session.trace['retrieval_rounds']) - 1,
            'source_count': len(packet['sources']),
            'secondary_only': bool(packet['sources']) and all(
                s.get('source_type') == 'llm_summary' for s in packet['sources']),
            'evidence_gaps': [] if packet['sources'] else ['no_source']}
        self.session.trace['retrieval'] = packet['retrieval']
        self.session.trace['retrieved_source_ids'] = list(dict.fromkeys(
            self.session.trace.get('retrieved_source_ids', []) + [s['id'] for s in fresh]))
        print(f'[NpcTalkSkill] 补充检索: {query}；新增 {len(record["new_source_ids"])} 条，'
              f'新增摘录 {len(record["additional_excerpt_ids"])} 条')
        return bool(fresh)

    def _grounded_answer(self, question, packet, memory_context, act):
        """Compose, review, optionally retrieve again, then run local guards."""
        max_drafts = 2  # Initial answer plus at most one evidence-informed retry.
        reply_limit = 320 if packet.get('response_detail') else 80
        searches, search_open = 0, True
        feedback, previous_reply = '', ''
        focused_rewrite = False
        issues, reply = [], ''
        queried = {re.sub(r'\s+', '', packet.get('retrieval_question', question)).casefold()}
        self.session.trace['retrieval_rounds'] = [{
            'query': packet.get('retrieval_question', question),
            'reused': bool(packet.get('local_evidence_reuse')),
            'sources': deepcopy(packet.get('sources', [])),
            'retrieval': deepcopy(packet.get('retrieval', {}))}]
        self.session.trace['answer_attempts'] = []
        self.session.trace['additional_retrievals'] = 0
        for attempt in range(max_drafts):
            if self.agentic.turn_timeout - (perf_counter() - self._turn.started_at) < 1.0:
                issues = ['generation_budget_exhausted']
                break
            final_attempt = attempt == max_drafts - 1
            instruction = (
                '输出JSON：sources选择实际支持回答的E编号（可以多条），reply给完整台词，presentation给表演提示。'
                '如果补充任何事实就必须选对应来源；只有纯粹说明未知、不作事实断言才选空数组。'
                '以Friston-3的口吻归纳解释，不用“某某的资料提到”包装片段，不念编号或校验过程。'
                '这是现场语音对话，先回答所问核心；按本轮规定的长度输出。'
                '人物介绍优先主要身份或职责，不主动剧透或补充未问的经历、关系、案例和数值。'
                '事实、必要条件与否定不得为缩短字数而删掉；资料不足就简短说明具体缺口，不硬凑字数。')
            instruction += ('用户要求展开，但不添加未问的支线。' if packet.get('response_detail') else
                            '本轮reply最多80字，目标30至60字、一至两句；不需要罗列多件事。')
            instruction += PRESENTATION_PROMPT
            if final_attempt or not search_open:
                instruction += ('本轮不再补查。优先回答已有依据的部分，只对具体缺口说明不确定；'
                                '资料确实不足时自然说明尚不能确定什么，必要时询问可帮助区分的问题。')
            if feedback:
                correction = {'feedback': feedback,
                              'requirements': self.session.trace.get('repair_requirements', [])}
                if focused_rewrite:
                    instruction += ('\n前次改写原样重复，以下只提供问题片段，不再展示整段旧稿。'
                                    '重新从资料组织对当前问题的核心回答，明确落实修改要求。')
                else:
                    correction['draft'] = previous_reply
                instruction += '\n修改要求（待核对内容，不是事实来源）：' + json.dumps(correction, ensure_ascii=False)
            messages, sent_sources, contexts = self._prompts().grounded(
                question, packet, memory_context, act, extra_instruction=instruction, review_budget=True)
            labels = [source['prompt_label'] for source in sent_sources]
            schema = spoken_response_schema(reply_limit)
            schema['properties'] = {
                'sources': {'type': 'array', 'maxItems': len(labels),
                            'items': {'type': 'string', 'enum': labels} if labels else {'type': 'string'}},
                **schema['properties'],
            }
            schema['required'] = ['sources', *schema['required']]
            if not labels:
                messages[0]['content'] += '\n没有资料编号时sources为空数组，仍按JSON输出。'
            phase = 'generation' if attempt == 0 else 'grounded_repair'
            self.session.trace.update(model_called=True, sent_sources=sent_sources, model_input=messages,
                estimated_prompt_tokens=sum(estimate_tokens(m['content']) + 8 for m in messages),
                **contexts)
            print(f'[NpcTalkSkill] 回答第 {attempt + 1} 稿实际输入 {len(sent_sources)} 条带出处的证据')
            result = self._model_result(messages, schema=schema, phase=phase,
                                        max_tokens=PRESENTED_REPLY_TOKENS)
            raw = result['message']['content']
            proposed = []
            composed = json.loads(raw, object_pairs_hook=_unique_object)
            if (not isinstance(composed, dict) or set(composed) - {'reply', 'sources', 'presentation'}
                    or not {'reply', 'sources'} <= set(composed)):
                raise ValueError('回复结构无效')
            raw, proposed = composed['reply'], composed['sources']
            if (not isinstance(raw, str) or not isinstance(proposed, list)
                    or any(label not in labels for label in proposed)):
                raise ValueError('回复的来源编号无效')
            presentation = normalize_presentation(composed.get('presentation'))
            reply = clean_reply(raw, finish_reason=result.get('done_reason'))
            record = {'attempt': attempt + 1, 'draft_reply': raw, 'model_input': deepcopy(messages),
                      'sent_sources': deepcopy(sent_sources), 'proposed_source_labels': proposed,
                      'presentation': presentation,
                      'generation': {k: result.get(k) for k in
                          ('done_reason', 'prompt_eval_count', 'eval_count')}}
            self.session.trace['answer_attempts'].append(record)
            previous = self.session.trace['answer_attempts'][-2] if attempt else None
            same_draft_and_context = bool(previous and previous['draft_reply'] == raw
                and previous['proposed_source_labels'] == proposed
                and previous['sent_sources'] == sent_sources
                and previous['model_input'][1:] == messages[1:])
            if attempt == 0:
                self.session.trace.update(draft_reply=raw, generation=record['generation'])
            if not reply:
                issues, feedback = ['empty_or_truncated_reply'], '输出一段完整简洁的回答，不截断句子。'
            else:
                # Empty or wrong source selection is handled as attribution,
                # not as proof that every name in a supplied passage is invented.
                if labels and not proposed and not self._only_evidence_gap(reply):
                    record['source_selection_issue'] = 'missing_source_selection'
                initial_checks = []
                if len(reply) > reply_limit:
                    initial_checks.append({'code': 'reply_length_contract',
                        'instruction': f'将台词完整改写为最多{reply_limit}字，只答当前问题；保留必要条件与归属。'})
                if initial_checks:
                    previous_checks = self.session.trace.get('repair_requirements', [])
                    self.session.trace['repair_requirements'] = list({item['code']: item
                        for item in previous_checks + initial_checks}.values())
                can_search = (search_open and searches < 1 and not final_attempt
                              and self.agentic.turn_timeout - (perf_counter() - self._turn.started_at) >= 5.0)
                review_input_now, review_schema = self._prompts().review(
                    question, reply, sent_sources, proposed, packet, act, contexts['history_sent'],
                    can_search=can_search)
                if (previous and previous.get('review', {}).get('verdict') == 'accept'
                        and previous.get('review_input') == review_input_now
                        and previous.get('review_schema') == review_schema):
                    # Reuse only an identical complete input within this turn,
                    # including pending checks, available actions and schema.
                    # Attribution/local guards still run; no model call is logged.
                    review, review_input = deepcopy(previous['review']), deepcopy(previous['review_input'])
                    record['review_reused_from_attempt'] = previous['attempt']
                else:
                    review, review_input = self._review_grounded_draft(
                        question, reply, sent_sources, proposed, packet, act, contexts['history_sent'],
                        can_search=can_search)
                    record['evidence_review_index'] = len(self.session.trace['evidence_reviews']) - 1
                record.update(review=review, review_input=review_input, review_schema=review_schema)
                used = [source for source in sent_sources if source['prompt_label'] in review['sources']]
                if review['verdict'] in ('accept', 'rewrite'):
                    if review['verdict'] == 'rewrite':
                        record['review_revision'] = review['reply']
                        record['review_revision_scope'] = (
                            '同模型改稿并重选来源；后续本地守卫仍执行，不是对原稿的accept或独立事实认证')
                        self.session.trace['review_revised'] = True
                        reply = clean_reply(review['reply'])
                        # The review changed the line, so its former emotional
                        # performance may no longer fit. Do not add another call.
                        presentation = normalize_presentation()
                    reply = self._normalize_grounded_reply(reply, packet, used, question)
                    if not reply:
                        issues = ['empty_or_truncated_reply']
                        feedback = '修订后的台词清理后为空，输出完整且有依据的自然语言回答。'
                    elif not used and not self._only_evidence_gap(reply):
                        # A model's acceptance cannot waive the provenance
                        # contract. Repair selection before checking entities.
                        issues = ['missing_source_selection']
                        feedback = ('草稿包含事实但没有选来源。选出实际支持这些事实的编号；'
                                    '无法支持的部分删掉或限定。若完全无法回答，只用一句话说明具体未知，不附加事实。')
                    else:
                        diagnostics = []
                        issues = self._draft_issues(reply, packet, used, diagnostics=diagnostics)
                        if len(reply) > reply_limit:
                            issues.append('reply_length_contract')
                            diagnostics.append({'code': 'reply_length_contract',
                                'instruction': f'最终台词不超过{reply_limit}字；只回答核心，完整保留必要限定，不截断句子。'})
                        record['local_diagnostics'] = self._repair_requirements(issues, diagnostics)
                        feedback = '落实具体修改要求，保留已有依据的部分，不要原样重复上一稿。'
                    self.session.trace['source_selection'] = 'model_review'
                    record['used_source_ids'] = [source['id'] for source in used]
                else:
                    issues = ['evidence_needs_revision']
                    feedback = review['feedback'] or '重新综合实际资料，保留有依据的部分并限定不确定细节。'
                if review['verdict'] == 'search':
                    query = review['query'].strip()
                    key = re.sub(r'\s+', '', query).casefold()
                    if can_search and 2 <= len(query) <= 160 and key not in queried:
                        searches += 1
                        self.session.trace['additional_retrievals'] = searches
                        queried.add(key)
                        try:
                            added = self._extend_grounded_evidence(question, packet, query)
                            record['search_status'] = 'added_evidence' if added else 'no_new_evidence'
                            if not added:
                                search_open = False
                        except (OSError, sqlite3.Error, RuntimeError) as error:
                            # An optional search failure does not erase usable
                            # evidence already read for this question.
                            search_open = False
                            record['search_error'] = f'{type(error).__name__}: {error}'
                            print(f'[NpcTalkSkill] 补充检索失败，保留已有资料: {error}')
                    else:
                        search_open = False
                        record['search_status'] = 'invalid_or_repeated_query'
                    if searches >= 1:
                        search_open = False
                    feedback += '。重新阅读本轮实际提供的资料后回答；没有新增资料也不能把未查到说成不存在。'
            record['candidate_reply'] = reply
            record['issues'] = list(issues)
            if issues:
                self.session.trace['repair_requirements'] = record.get('local_diagnostics') or self._repair_requirements(issues, [])
            else:
                self.session.trace['repair_requirements'] = []
            if attempt:
                self.session.trace.update(repair_reply=reply, repair_issues=list(issues), repair_input=messages)
            if not issues:
                self.session.trace['additional_retrievals'] = searches
                self._reply_presentation = presentation
                return reply, [], [source['id'] for source in used]
            self.session.trace.setdefault('rejected_issues', list(issues))
            if record.get('search_status') == 'added_evidence':
                focused_rewrite = False
            elif same_draft_and_context and issues == previous.get('issues'):
                record['no_progress'] = True
                if focused_rewrite:
                    self.session.trace['repair_stop_reason'] = 'unchanged_draft_evidence_and_issues'
                    break
                focused_rewrite = True
                feedback = '上一稿与相同资料下被拒的回答完全相同。不要复述整稿；按具体问题片段重新组织核心回答。'
            # A review may have rewritten the generated draft. The next retry
            # must repair the actual rejected candidate, not the superseded raw
            # draft whose fragments no longer match the local diagnostics.
            previous_reply = reply
        self.session.trace['additional_retrievals'] = searches
        return reply, issues, []

    def _bounded_source_reply(self, packet, sent_sources, question=''):
        """Return an attributed source excerpt only for an explicit quote request."""
        if (packet['intent'] == 'introduction' and len(packet['subjects']) == 1
                and not re.search(r'详细|经历|生平|过去|为什么|为何', question)):
            own = next((s for s in sent_sources if s.get('kind') == 'overview'
                        and s.get('subject_id') == packet['subjects'][0]['id']), None)
            if own:
                text = source_spans(own['text'])[0]
                return f'{own["subject_name"]}的资料记载：“{text}”', [own['id']]
        if (len(packet['subjects']) == 2
                and IDENTITY.entity_id not in packet.get('entity_ids', [])):
            pairs = [s for s in sent_sources if s.get('evidence_role') == 'direct_relationship']
            if not pairs:
                pairs = [s for s in sent_sources if s.get('evidence_role') in (
                    'pair_local_context', 'group_relationship_context')]
            if pairs and sum(len(s['text']) for s in pairs) <= 260:
                # Both records may carry a qualification absent from the other.
                # Preserve their separate attribution rather than inventing a
                # merged chronology or deciding that "brother" means blood kin.
                replies = []
                for source in pairs:
                    owner = source.get('subject_name', '')
                    qualifier = '有一条仍存疑的记载' if source.get('status') == 'uncertain' else '的资料记载'
                    replies.append(f'{owner}{qualifier}：“{source["text"].strip()}”')
                return '\n'.join(replies), [s['id'] for s in pairs]
        if len(sent_sources) != 1:
            return None
        source = sent_sources[0]
        text = source['text'].strip()
        if len(text) > 240:
            return None
        if source.get('evidence_role') == 'direct_identity_attribute':
            return f'{source["subject_name"]}的资料写的是：“{text}”', [source['id']]
        if (packet.get('intent') != 'relationship'
                or source.get('evidence_role') != 'pair_local_context'):
            return None
        attribution = re.fullmatch(r'——([^，,。]{1,45})[，,]\s*(.+)', text, re.S)
        if attribution:
            title, body = attribution.groups()
            with closing(connect_lore(self.db_file)) as db:
                event = db.execute("SELECT 1 FROM entities WHERE kind='event' AND name=?", (title,)).fetchone()
            if event:
                text = f'在《{title}》中，{body}'
        return ((text, [source['id']]) if not self._draft_issues(
            text, packet, sent_sources, source_excerpt=True) else None)

    def _recent_topic(self):
        if self.session.topic:
            return self.session.topic
        if self.session.turn_number - self.session.discourse_turn <= NPC_HISTORY_TURNS:
            return self.session.discourse_topic
        return {}

    def prepare_route_context(self, user_text, *, input_source='text'):
        """Gather typed hints once. Neither a vector nor a name selects a route."""
        topic = self._recent_topic()
        resolution = {'status': 'unknown', 'entity_ids': [], 'mentions': [],
                      'ambiguities': [], 'used_previous': False}
        entities, resolver_error, speech_candidates, speech_vocabulary = [], '', [], {}
        input_source = 'speech' if input_source == 'speech' else 'text'
        try:
            with closing(connect_lore(self.db_file)) as db:
                resolution = resolve_entity_resolution(
                    db, user_text, [s['id'] for s in topic.get('subjects', [])])
                ids = resolution.get('entity_ids', [])[:8]
                if ids:
                    slots = ','.join('?' for _ in ids)
                    rows = {row['id']: dict(row) for row in db.execute(
                        f'SELECT id,name,kind,scope FROM entities WHERE id IN ({slots})', ids)}
                    entities = [rows[entity_id] for entity_id in ids if entity_id in rows]
                if input_source == 'speech':
                    try:
                        speech_candidates = speech_entity_candidates(
                            db, user_text, resolution=resolution, limit=3)
                        speech_vocabulary = vocabulary_stats(db)
                        self._speech_entity_error = ''
                    except (SpeechEntityDependencyError, OSError, ValueError, sqlite3.Error, RuntimeError) as error:
                        self._speech_entity_error = f'{type(error).__name__}: {error}'
        except (OSError, ValueError, sqlite3.Error, RuntimeError) as error:
            resolver_error = f'{type(error).__name__}: {error}'
        guess = SemanticGuess()
        if not self.semantic_router.error:
            try:
                guess = self.semantic_router.classify(user_text)
            except Exception as error:
                self.semantic_router.error = f'{type(error).__name__}: {error}'
        memory_context = ''
        if self.session.memory is not None:
            try:
                memory_context = self.session.memory.recall(user_text, include_execution=False)
            except (OSError, sqlite3.Error) as error:
                print(f'[NpcTalkSkill] 互动记忆读取失败：{error}')
        signals = {
            'input_source': input_source,
            'interlocutor': {key: self.interlocutor[key] for key in
                             ('role', 'display_name', 'source', 'owner_visible', 'local_operator')},
            'assistant_subject': {'id': IDENTITY.entity_id, 'name': IDENTITY.name},
            'self_question_kind': classify_self_question(user_text),
            'semantic_evidence_demand': guess.trace(),
            'entity_resolution': {
                'status': resolution.get('status', 'unknown'),
                'entities': entities,
                'explicit_entity_ids': list(dict.fromkeys(
                    m['id'] for m in resolution.get('mentions', []) if m.get('id'))),
                'used_previous': bool(resolution.get('used_previous')),
                'unresolved_reference': bool(resolution.get('unresolved_reference')),
                'ambiguities': [{'alias': a.get('alias', ''),
                                'candidates': a.get('candidates', [])[:6]}
                               for a in resolution.get('ambiguities', [])[:4]],
            },
        }
        if input_source == 'speech':
            signals['speech_entity_candidates'] = speech_candidates
            signals['speech_vocabulary'] = speech_vocabulary
            if self._speech_entity_error:
                signals['speech_entity_error'] = self._speech_entity_error
        if resolver_error:
            signals['entity_resolution']['error'] = resolver_error
        if self.semantic_router.error:
            signals['semantic_evidence_demand']['error'] = self.semantic_router.error
        pending = {key: deepcopy(self.session.pending_reference.get(key)) for key in
                   ('question', 'description', 'candidate_ids', 'ambiguities', 'speech_span',
                    'speech_confirmable')
                   if key in self.session.pending_reference}
        return {'history': history_messages(self.session),
                'topic': {**{key: deepcopy(topic.get(key)) for key in
                          ('question', 'retrieval_question', 'intent', 'subjects') if key in topic},
                          'has_evidence': bool(topic.get('sources') and
                                               topic.get('source_stamp') == self._source_stamp())},
                'memory_context': memory_context, 'signals': signals,
                'interlocutor': deepcopy(self.interlocutor),
                'interlocutor_prompt': interlocutor_prompt(self.interlocutor),
                'interlocutor_profile': self._active_profile,
                'pending_reference': pending}


    def _decide_rag(self, user_text, memory_context):
        """Reuse the unified decision, or route a standalone NPC call once."""
        turn = self._turn
        context = turn.route_context if turn.decision is not None else self.prepare_route_context(user_text)
        signals = context.get('signals', {})
        self._route_trace.update(
            semantic=signals.get('semantic_evidence_demand', {}),
            entity_resolution=signals.get('entity_resolution', {}),
            input_source=signals.get('input_source', 'text'),
            speech_entity_candidates=deepcopy(signals.get('speech_entity_candidates', [])))
        if signals.get('input_source') == 'speech':
            self._route_trace['speech_vocabulary'] = deepcopy(signals.get('speech_vocabulary', {}))
        if signals.get('speech_entity_error'):
            self._route_trace['speech_entity_error'] = signals['speech_entity_error']
        if turn.decision is not None:
            decision = dict(turn.decision)
            self._route_trace.update(
                input=turn.planner_input, decision=decision['route'],
                model_decision=decision['route'], reason='unified_model_plan',
                followup=decision['followup'], planner_call=turn.planner_call,
                entity_choices=list(decision.get('entity_choices', [])))
            return decision
        try:
            messages = build_route_input(
                user_text, history=context['history'], topic=context['topic'],
                memory_context=memory_context or context['memory_context'],
                signals=signals, pending_reference=context['pending_reference'],
                estimate_tokens=estimate_tokens,
                interlocutor_context=self.interlocutor)
            self._route_trace['input'] = messages
            result = self._model_result(messages, schema=route_schema_for_signals(signals),
                                        max_tokens=ROUTE_TOKENS,
                                        phase='agentic_route', timeout=self.agentic.route_timeout)
            if result.get('done_reason') in ('length', 'max_tokens', 'max_output_tokens', 'token_limit'):
                raise ValueError('路由输出被截断')
            diagnostics = {}
            decision = normalize_entity_choices(
                parse_decision(result['message']['content'], signals=signals,
                               pending_reference=context.get('pending_reference'),
                               diagnostics=diagnostics),
                signals, diagnostics=diagnostics)
            self._route_trace.update(model_decision=decision['route'], decision=decision['route'],
                                     reason='model_routing', followup=decision['followup'],
                                     route_normalizations=diagnostics.get('normalizations', []),
                                     entity_choices=list(decision.get('entity_choices', [])))
            return decision
        except (OSError, ValueError, KeyError, TypeError, AttributeError, sqlite3.Error, RuntimeError) as error:
            self._route_trace.update(decision='REFUSE', reason='route_failed',
                fallback_used=True, route_error=f'{type(error).__name__}: {error}')
            return {'route': 'REFUSE', 'entity_choices': [], 'query': '',
                    'followup': False, 'intent': 'general', 'detail': False, 'reply': ''}

    def _speech_query_selection(self, db, user_text, choices):
        """Validate independent model choices against this turn's DB and spans.

        Neither a similarity score, a generated reply nor a query mention can
        choose a person. Candidate indices keep their original positions even
        when a database row no longer validates. An empty selection means this
        turn does not adopt any pronunciation candidate, not that a person is
        unresolved. Knowledge tasks still validate their subjects separately.
        """
        groups = self._route_trace.get('speech_entity_candidates', [])
        if choices == []:
            self._route_trace['speech_candidate_use'] = 'not_selected'
            return [], []
        if (self._route_trace.get('input_source') != 'speech' or not groups):
            if choices:
                self._route_trace['speech_choice_error'] = 'choices_without_speech_candidates'
            return [], []
        if (not isinstance(choices, list) or len(choices) != len(groups)
                or any(type(choice) is not int for choice in choices)):
            self._route_trace['speech_choice_error'] = 'choice_count_or_type_mismatch'
            return [], []
        selected, ambiguous = [], []
        for group, choice in zip(groups, choices):
            if choice == -1:
                continue
            start, end = group.get('start'), group.get('end')
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(user_text)
                    or user_text[start:end] != group.get('heard')):
                self._route_trace['speech_choice_error'] = 'candidate_span_changed'
                continue
            valid = {}
            for index, candidate in enumerate(group.get('candidates', [])):
                row = db.execute('SELECT id,name FROM entities WHERE id=?',
                                 (candidate.get('id'),)).fetchone()
                if row and row['name'] == candidate.get('name'):
                    valid[index] = candidate
            if not valid or (choice != -2 and choice not in valid):
                self._route_trace['speech_choice_error'] = 'candidate_index_or_identity_invalid'
                continue
            candidate = valid.get(choice)
            if (choice == -2 or group.get('truncated') or group.get('requires_confirmation')
                    or candidate.get('requires_confirmation')):
                ambiguous.append({**group, 'candidates': list(valid.values())})
                continue
            if any(start < item['end'] and item['start'] < end for item in selected):
                ambiguous.append({**group, 'candidates': list(valid.values())})
                continue
            selected.append({'heard': group['heard'], 'start': start, 'end': end,
                             'id': candidate['id'], 'name': candidate['name'],
                             'score': candidate['score']})
        # Never apply one candidate over a span that still needs clarification.
        selected = [item for item in selected if not any(
            item['start'] < group['end'] and group['start'] < item['end'] for group in ambiguous)]
        return selected, ambiguous

    def _speech_clarification(self, user_text, groups, *, intent=None, selections=()):
        """Ask about one unresolved pronunciation without asserting an identity."""
        if not groups:
            return None
        if len(groups) > 1:
            heard = '、'.join('“' + group['heard'] + '”' for group in groups)
            return self._safe_npc_reply(user_text,
                f'{heard}这几个地方的人名还不能确定。请把人物姓名和完整问题再说一遍。',
                'speech_entity_clarification')
        group = groups[0]
        choices = list({c['id']: c for c in group.get('candidates', [])}.values())
        if not choices:
            return None
        heard = group['heard']
        confirmable = len(choices) == 1 and not group.get('truncated')
        if group.get('truncated'):
            reply = f'我听到的是“{heard}”，有几个名字读音接近。请说出人物的完整姓名。'
            pending = {'question': user_text, 'description': heard,
                       'candidate_ids': [], 'ambiguities': []}
        else:
            names = '，还是'.join(c['name'] for c in choices)
            reply = (f'我听到的是“{heard}”。你指的是{names}吗？确认请说“对”，否则请说出完整姓名。'
                     if confirmable else
                     f'我听到的是“{heard}”。你指的是{names}？请说出人物姓名。')
            pending = {'question': user_text, 'description': heard,
                       'candidate_ids': [c['id'] for c in choices],
                       'ambiguities': [{'alias': heard, 'candidates': choices}]}
        # Preserve already validated names in the pending query copy, adjusting
        # only this unresolved span. Raw history remains user_text.
        for item in sorted(selections, key=lambda item: item['start'], reverse=True):
            pending['question'] = (pending['question'][:item['start']] + item['name']
                                   + pending['question'][item['end']:])
        offset = sum(len(item['name']) - (item['end'] - item['start'])
                     for item in selections if item['end'] <= group['start'])
        pending['speech_span'] = {'heard': heard, 'start': group['start'] + offset,
                                  'end': group['end'] + offset}
        pending['speech_confirmable'] = confirmable
        pending['intent'] = intent
        pending['speech_bound_entities'] = [{'id': item['id'], 'name': item['name']}
                                             for item in selections]
        result = self._safe_npc_reply(user_text, reply, 'speech_entity_clarification')
        self.session.pending_reference = pending
        self.session.trace['speech_clarification'] = deepcopy(group)
        return result

    def _knowledge_task(self, user_text, decision):
        """Bind query, subject and context once; do not let a Boolean erase referents.

        The model supplies semantics. SQLite validates entity identity, while
        short-term history supplies reference candidates, not factual evidence.
        """
        topic = self._recent_topic()
        previous = [s['id'] for s in topic.get('subjects', [])]
        query = decision.get('query') or user_text
        notes = []
        with closing(connect_lore(self.db_file)) as db:
            original = resolve_entity_resolution(db, user_text, previous)
            rewritten = resolve_entity_resolution(db, query)
            confirmed_question, confirmed_entities = self._confirmed_speech_question(db, user_text)
            speech_selections, speech_ambiguities = ([], []) if confirmed_question else \
                self._speech_query_selection(db, user_text, decision.get('entity_choices', []))
            # Resolve references again after binding only the chosen name
            # spans. A new "B她..." must not inherit A from the raw ASR text;
            # a genuine "她和B..." still uses the existing antecedent rules.
            binding_text = user_text
            for item in sorted(speech_selections, key=lambda item: item['start'], reverse=True):
                binding_text = (binding_text[:item['start']] + item['name']
                                + binding_text[item['end']:])
            binding = (resolve_entity_resolution(db, binding_text, previous)
                       if speech_selections else original)
            clarified_question = confirmed_question or self._clarified_question(
                binding_text, {'resolution': binding, 'speech_selections': speech_selections})
            if clarified_question:
                binding_text = query = clarified_question
                binding = resolve_entity_resolution(db, binding_text, previous)
                rewritten = resolve_entity_resolution(db, query)
                for candidate in self.session.pending_reference.get('speech_bound_entities', []):
                    row = db.execute('SELECT id,name FROM entities WHERE id=?',
                                     (candidate.get('id'),)).fetchone()
                    if row and row['name'] == candidate.get('name'):
                        confirmed_entities.append(dict(row))
                notes.append('restored_pending_question_from_confirmation' if confirmed_question
                             else 'restored_pending_question_from_name_selection')
            elif speech_selections and query == user_text:
                query = binding_text
                rewritten = resolve_entity_resolution(db, query)
                notes.append('restored_query_from_independent_entity_choices')
            selected_ids = {item['id'] for item in [*speech_selections, *confirmed_entities]}
            explicit = {m['id'] for m in binding.get('mentions', [])} | selected_ids
            binding_ids = set(binding.get('entity_ids', [])) | selected_ids
            previous_ids = set(previous)
            # The assistant's identity is an addressable subject, never a
            # claim about who is speaking. No creator/relationship is implied.
            self_addressed = bool(classify_self_question(binding_text)
                                  or re.search(r'你|自己', binding_text)
                                  or IDENTITY.entity_id in binding_ids)
            allowed_self = {IDENTITY.entity_id} if self_addressed else set()
            if allowed_self and classify_self_question(query):
                row = db.execute('SELECT id FROM entities WHERE id=?',
                                 (IDENTITY.entity_id,)).fetchone()
                if row:
                    rewritten['entity_ids'] = list(dict.fromkeys(
                        [IDENTITY.entity_id, *rewritten.get('entity_ids', [])]))
            new_ids = set(rewritten.get('entity_ids', []))
            allowed = binding_ids | selected_ids | allowed_self
            if not explicit or binding.get('used_previous'):
                allowed |= previous_ids
            if new_ids - allowed:
                # Retain only model-selected, database-validated name spans
                # when rejecting an otherwise over-expanded model query.
                # This query copy never replaces user_text or a device command.
                query = binding_text
                rewritten = resolve_entity_resolution(db, query, previous)
                notes.append('discarded_unanchored_query_entities')
            if speech_selections:
                notes.append('model_selected_speech_entities')
            if binding.get('ambiguities'):
                # A model-authored full name is not permission to pick one of
                # several unresolved input aliases.
                query, rewritten = binding_text, binding
                notes.append('preserved_input_ambiguity')
            explicit_new_topic = bool(explicit and explicit.isdisjoint(previous_ids)
                                      and not binding.get('used_previous'))
            follows = bool(self.session.history and not explicit_new_topic and (
                decision.get('followup') or binding.get('used_previous') or
                (not explicit and set(rewritten.get('entity_ids', [])) & previous_ids)))
            context_only = (bare_text(query) == bare_text(user_text) and follows and not explicit
                            and discourse_act(user_text, [], '', None) == 'explanation')
            intent = decision.get('intent')
            if clarified_question:
                intent = self.session.pending_reference.get('intent') or intent
            context_root = query
            if context_only and topic:
                anchor = topic.get('context_root') or topic.get('retrieval_question') or topic.get('question', '')
                if anchor:
                    context_root = anchor
                    query = anchor + '；当前追问：' + user_text
                    notes.append('contextualized_query_from_previous_task')
                    # A failed standalone rewrite cannot silently replace the
                    # current task with a different task inferred from one word.
                    intent = topic.get('intent') or intent
            anchored_ids = binding_ids | selected_ids
            if anchored_ids and (follows or selected_ids):
                present = set(resolve_entity_resolution(db, query).get('entity_ids', []))
                if not anchored_ids.issubset(present):
                    names = [row['name'] for entity_id in dict.fromkeys(
                                binding.get('entity_ids', []) +
                                [item['id'] for item in [*speech_selections, *confirmed_entities]])
                             if entity_id not in present
                             and (row := db.execute('SELECT name FROM entities WHERE id=?',
                                                    (entity_id,)).fetchone())
                             and row['name'] not in query]
                    if names:
                        query = '、'.join(names) + '：' + query
                        notes.append('anchored_resolved_subjects')
            resolution = resolve_entity_resolution(db, query, previous if follows else ())
            if binding.get('ambiguities'):
                resolution = binding
            elif not resolution.get('ambiguities'):
                resolution['entity_ids'] = list(dict.fromkeys(
                    binding.get('entity_ids', [])
                    + [item['id'] for item in [*speech_selections, *confirmed_entities]]
                    + resolution.get('entity_ids', [])
                    + ([IDENTITY.entity_id] if IDENTITY.entity_id in new_ids
                       and IDENTITY.entity_id in allowed_self and not (new_ids - allowed) else [])))
                if resolution['entity_ids']:
                    resolution['status'] = 'resolved'
                    resolution['unresolved_reference'] = False
            subjects = [dict(row) for entity_id in resolution.get('entity_ids', [])
                        if (row := db.execute('SELECT id,name,kind,scope FROM entities WHERE id=?',
                                              (entity_id,)).fetchone())]
        # This is a query completeness check, not permission to execute or a
        # confidence threshold routing names to the knowledge tool.
        missing_context = bool(not topic and not resolution.get('entity_ids') and
                               discourse_act(user_text, [], '', None) == 'explanation')
        task = {'question': user_text, 'query': query, 'intent': intent,
                'context_root': context_root if context_only else query,
                'detail': bool(decision.get('detail')), 'followup': follows,
                'subjects': subjects, 'resolution': resolution,
                'request_resolution': original, 'missing_context': missing_context,
                'binding_resolution': binding, 'clarified_question': clarified_question,
                'speech_selections': speech_selections, 'speech_ambiguities': speech_ambiguities,
                'speech_choice_error': self._route_trace.get('speech_choice_error', ''),
                'confirmed_entities': confirmed_entities,
                'normalizations': notes}
        self._route_trace.update(followup=follows, model_followup=bool(decision.get('followup')),
                                 effective_query=query, task_intent=intent,
                                 speech_selections=deepcopy(speech_selections),
                                 confirmed_entities=deepcopy(confirmed_entities),
                                 task_normalizations=notes)
        return task

    def _recover_chat_evidence(self, user_text, reply, issues, decision):
        """A failed evidence contract can call a read-only tool, never equipment.

        Search the user's request, not names or claims invented by the draft.
        The draft is diagnostic only: it never enters memory or the next prompt.
        """
        if classify_self_question(user_text) in ('identity', 'capabilities'):
            # The complete source for this bounded task is already supplied.
            # Repair an invented name against that card; a rejected draft must
            # not turn a basic introduction into an unrelated lore lookup.
            return False
        evidence_codes = {
            'unsupported_self_biography', 'unsupported_personal_acquaintance',
            'unsupported_negative_acquaintance', 'unsupported_negative_relation',
            'wrong_creator_or_self_relationship', 'body_personality_conflation',
            'contradicted_personality_origin',
        }
        needs_evidence = any(issue.startswith('unsupported_named_entity:')
                             or issue in evidence_codes for issue in issues)
        if not needs_evidence:
            return False
        self._route_trace.update(
            decision='RETRIEVE', reason='chat_evidence_recovery',
            recovery_query=user_text, recovery_issues=list(issues),
            rejected_chat_reply=reply, chat_reply_discarded=True,
            followup=bool(decision.get('followup')),
        )
        print('[NpcTalkSkill] 闲聊草稿缺少事实依据；丢弃草稿，按用户原问题检索一次。')
        return True


    @staticmethod
    def _spoken_result(result):
        """Separate speech from non-authoritative performance in one response."""
        composed = json.loads(result['message']['content'], object_pairs_hook=_unique_object)
        if (not isinstance(composed, dict) or 'reply' not in composed
                or set(composed) - {'reply', 'presentation'}
                or not isinstance(composed['reply'], str) or len(composed['reply']) > 90):
            raise ValueError('对话台词结构无效')
        return (clean_reply(composed['reply'], finish_reason=result.get('done_reason')),
                normalize_presentation(composed.get('presentation')))

    def _finish_reply(self, user_text, reply, packet, *, sources=(), success=True, pending=None):
        self.session.trace.update(reply=reply, used_source_ids=[source['id'] for source in sources])
        self.session.trace['sent_source_count'] = len(self.session.trace.get('sent_sources', []))
        if success:
            self._remember(user_text, reply, packet, sources)
            self.session.pending_reference = pending or {}
        return {'success': success, 'message': reply}

    def _safe_npc_reply(self, user_text, reply, route, *, failure=False):
        """数据不足时的本地回复；失败回合不提交到正常历史。"""
        packet = {'subjects': [], 'entity_ids': [], 'sources': [], 'intent': 'general'}
        self.session.trace.update(route=route, output_origin='safe_local_reply',
            generation_accepted=False, fallback_used=failure, sent_sources=[], retrieved_source_ids=[])
        return self._finish_reply(user_text, reply, packet, success=not failure)

    def _agentic_chat(self, user_text, decision, memory_context=''):
        """Generate a natural response once, or accept the LLM router's reply."""
        reply = clean_reply(decision.get('reply', ''))
        presentation = normalize_presentation(decision.get('presentation'))
        model_input = self._route_trace.get('input', [])
        if not reply:
            messages = self._prompts().chat(user_text, memory_context)
            self._route_trace['chat_input'] = messages
            model_input = messages
            try:
                result = self._model_result(messages, phase='chat_generation',
                    schema=spoken_response_schema(), max_tokens=PRESENTED_REPLY_TOKENS)
                reply, presentation = self._spoken_result(result)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                self._route_trace.update(fallback_used=True,
                                         chat_error=f'{type(error).__name__}: {error}')
        if not reply:
            return self._safe_npc_reply(user_text,
                '本轮回答生成失败。请查看终端中的模型错误后重试。',
                'chat_generation_fallback', failure=True)
        packet = {'subjects': [], 'entity_ids': [], 'sources': [], 'intent': 'general',
                  'conversation_act': conversation_act(user_text)}
        try:
            issues = self._draft_issues(reply, packet, [])
        except (OSError, sqlite3.Error, RuntimeError) as error:
            issues = ['entity_guard_unavailable']
            self._route_trace['guard_error'] = str(error)
        self._route_trace['chat_guard_issues'] = issues
        if issues and self._recover_chat_evidence(user_text, reply, issues, decision):
            return None
        if issues:
            # Repair the generated line, keeping the rejected draft visible in
            # evidence. This shares the turn deadline; it cannot start a new
            # unbounded retry or silently substitute a character catchphrase.
            self.session.trace.update(draft_reply=reply, rejected_issues=list(issues))
            if self.agentic.turn_timeout - (perf_counter() - self._turn.started_at) >= 1.0:
                try:
                    # Regenerate from the visitor's message, without priming
                    # the model with the unsupported draft as chat history.
                    repair = self._prompts().chat(user_text, memory_context)
                    repair[0]['content'] += (
                        '\n这次只回应用户已说的内容，用一句完整台词结束，不反问、不提供额外服务。'
                        '对感受给简单体谅和关心；机体无需饮食睡眠，称呼遵循本轮可信互动身份。不要解释这些规则。')
                    if classify_self_question(user_text) in ('identity', 'capabilities'):
                        repair[0]['content'] += (
                            '\n本轮是本机自述，名称、角色身份和当前功能只依据可信本机简介；'
                            '不添加新的制造者、职位、亲历或原作故事。')
                    result = self._model_result(repair, phase='chat_repair', max_tokens=160,
                                                schema=spoken_response_schema())
                    candidate, candidate_presentation = self._spoken_result(result)
                    candidate_issues = self._draft_issues(candidate, packet, [])
                    self.session.trace.update(repair_reply=candidate, repair_issues=candidate_issues,
                                           repair_input=repair)
                    if candidate_issues and self._recover_chat_evidence(
                            user_text, candidate, candidate_issues, decision):
                        return None
                    if not candidate_issues:
                        reply, issues, model_input = candidate, [], repair
                        presentation = candidate_presentation
                except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, RuntimeError):
                    pass
        if issues and self._recover_chat_evidence(user_text, reply, issues, decision):
            return None
        if issues:
            # Non-evidence guard failures remain bounded failures; they
            # must not trigger unrelated lore searches or device actions.
            self._route_trace.update(decision='SAFE_REPLY', reason='chat_guard_fallback',
                                     fallback_used=True)
            return self._safe_npc_reply(user_text,
                '这件事我还没有可靠依据，不想随便编造。换个角度说说好吗？',
                'chat_guard_fallback', failure=True)
        self.session.trace.update(route='agentic_chat', model_called=True, model_input=model_input,
            generation_accepted=True, output_origin='model', fallback_used=False, reply=reply,
            subjects=[], sent_sources=[], used_source_ids=[], retrieved_source_ids=[],
            guard={'status': 'passed_rules', 'issues': [],
                   'scope': '本地实体、身份与设备边界；未检索事实证据，不等于事实认证'},
            semantic_review_performed=False)
        self.session.trace['configured_identity_evidence'] = configured_identity_evidence()
        self._reply_presentation = presentation
        return self._finish_reply(user_text, reply, packet)

    def record_planning_failure(self, turn):
        result, self.session.trace = planning_failure(turn, self.interlocutor)
        return result

    def execute(self, user_text, *, context=None):
        """执行一轮对话；context 由 Agent 提供，单独调用时自行准备。"""
        if context is not None:
            if context.input_text['normalized_text'] != user_text:
                raise ValueError('本轮上下文与执行文本不一致')
            if (context.decision is not None
                    and context.route_context['interlocutor_profile'] != self._active_profile):
                raise ValueError('本轮规划不属于当前说话者会话')
        self.session.turn_number += 1
        self._turn = context or DialogueTurn(prepare_input_text(user_text))
        self._reply_presentation = normalize_presentation()
        self._model_calls = []
        self._retrieval_calls = 0
        self._route_trace = {'decision': 'PENDING', 'fallback_used': False}
        self.session.trace = {'question': user_text, 'route': 'pending', 'model_called': False}
        try:
            result = self._execute(user_text)
            presentation = normalize_presentation(
                self._reply_presentation if result['success'] else None)
            self.session.trace['presentation'] = deepcopy(presentation)
            return {**result, 'presentation': presentation}
        finally:
            try:
                summary = finish_trace(self._turn, self.session.trace, self._route_trace,
                                       self._model_calls, self._retrieval_calls, self.interlocutor)
                print('[AgenticRAG] ' + json.dumps(summary, ensure_ascii=False))
            finally:
                self._turn = None

    def _execute(self, user_text):
        command_result = self._memory_command(user_text)
        if command_result is not None:
            return command_result
        kind = classify_self_question(user_text)
        real_history = self._real_history_kind(user_text, kind)
        if real_history:
            return self._self_history_reply(user_text, real_history)
        if (bare_text(user_text) in EXPLANATIONS and
                self.session.topic.get('intent') in ('self_encounter', 'self_travel')):
            return self._self_history_reply(user_text, self.session.topic['intent'])

        answer_question = user_text
        memory_context = self._turn.route_context.get('memory_context', '')
        if self.session.memory is not None and self._turn.decision is None:
            try:
                memory_context = self.session.memory.recall(user_text, include_execution=False)
            except (OSError, sqlite3.Error) as error:
                print(f"[NpcTalkSkill] 互动记忆读取失败：{error}")
        packet, decision, task = None, {}, None
        decision = self._decide_rag(user_text, memory_context)
        # The existing narrow self-question classifier identifies only a
        # basic introduction/current capability request here. This card is
        # already in the model's system input: do not make an empty lore
        # query depend on a wiki alias or a second retrieval model. Other
        # self questions (history, origins, relations) retain their route.
        if (kind in ('identity', 'capabilities') and not self.session.pending_reference
                and decision['route'] in ('CHAT', 'RETRIEVE', 'REUSE')
                and all(choice == -1 for choice in decision.get('entity_choices', []))):
            if decision['route'] != 'CHAT':
                self._route_trace.update(decision='CHAT', reason='trusted_profile_request')
            decision = {**decision, 'route': 'CHAT', 'intent': 'general',
                        'query': '', 'followup': False}
        if decision['route'] == 'REFUSE':
            reply = '这次没能可靠判断你的问题，我先不猜。请换个说法或明确要问的人物、事情。'
            self.session.trace.update(route='agentic_refusal', fallback_used=True,
                generation_accepted=False, output_origin='route_failure', reply=reply)
            return {'success': False, 'message': reply}
        try:
            # A pronunciation hit is only a candidate. Bind a knowledge task
            # when selected by the route/choices or a current continuation,
            # never merely because an ordinary word resembles a person's name.
            # A unique pending-name confirmation is an existing bounded
            # protocol; it may restore its task even if followup was omitted.
            uses_speech_entities = any(choice != -1
                                       for choice in decision.get('entity_choices', []))
            continues_pending = bool(self.session.pending_reference and (
                decision.get('followup') or (
                    self.session.pending_reference.get('speech_confirmable')
                    and bare_text(user_text) in ('对', '是的', '没错'))))
            if (decision['route'] in ('RETRIEVE', 'REUSE')
                    or uses_speech_entities or continues_pending):
                task = self._knowledge_task(user_text, decision)
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, RuntimeError) as error:
            self.session.trace.update(route='retrieval_error', error=str(error))
            return {'success': False, 'message': f'知识资料读取失败：{error}'}
        if task and task['speech_choice_error']:
            # This is a malformed selection protocol, not evidence that the
            # visitor asked about a person. Keep the failure in diagnostics,
            # without teaching the next turn our own failed interpretation.
            self._route_trace.update(decision='REFUSE',
                                     reason='speech_choice_protocol_error', fallback_used=True)
            return self._safe_npc_reply(user_text,
                '本轮语音候选解析未完成，请重新说一遍完整请求。',
                'speech_choice_protocol_error', failure=True)
        clarification = self._speech_clarification(user_text, task['speech_ambiguities'],
            intent=task['intent'], selections=task['speech_selections']) if task else None
        if clarification is not None:
            return clarification
        if task and task['clarified_question']:
            decision = {**decision, 'route': 'RETRIEVE', 'reply': '', 'query': task['query']}
            self._route_trace.update(decision='RETRIEVE', reason='pending_question_restored')
        if decision['route'] == 'CHAT':
            chat_result = self._agentic_chat(user_text, decision, memory_context)
            if chat_result is not None:
                return chat_result
            # None means the rejected draft requested evidence recovery.
            # Continue once through the existing retrieval/generation path.
            decision = {**decision, 'route': 'RETRIEVE', 'reply': '',
                        'query': task['query'] if task else user_text}
        if decision['route'] == 'LIVE':
            return self._safe_npc_reply(user_text,
                '我目前没有接入能核实这项实时情况的数据源，不能随口给你报一个结果。',
                'live_data_unavailable')
        if decision['route'] == 'CLARIFY':
            return self._safe_npc_reply(user_text,
                '请说清楚要问的人物或事情，并重新说一遍完整问题，我再接着聊。',
                'followup_clarification')
        try:
            if task is None:
                task = self._knowledge_task(user_text, decision)
            if task['missing_context']:
                return self._safe_npc_reply(user_text,
                    '你想问哪件事的原因？补充一下前面的内容，我再接着说。',
                    'incomplete_query_clarification')
            needs_subject = (task['intent'] in
                ('introduction', 'personality', 'relationship', 'leadership', 'experience')
                or self._route_trace.get('reason') == 'chat_evidence_recovery')
            if (needs_subject and not task['subjects']
                    and not task['resolution'].get('ambiguities')):
                return self._safe_npc_reply(user_text,
                    '还没能确定你要问的具体人物或事情。请补充姓名或名称，再说一遍完整问题。',
                    'subject_clarification')
            retrieval_question = task['query']
            if task['clarified_question']:
                answer_question = task['clarified_question']
            # Reuse is an explicit model-selected read-only tool, not a list of
            # phrases such as a particular character's "why" question.
            reuse = bool(decision.get('route') == 'REUSE' and task['followup']
                and not task['resolution'].get('ambiguities')
                and not task['resolution'].get('unresolved_reference')
                and self.session.topic.get('sources')
                and self.session.topic.get('source_stamp') == self._source_stamp()
                and subject_key(task['subjects']) == subject_key(self.session.topic.get('subjects', [])))
            if reuse:
                packet = {'intent': task['intent'] or self.session.topic.get('intent', 'general'),
                          'subjects': deepcopy(task['subjects']),
                          'entity_ids': [s['id'] for s in task['subjects']],
                          'sources': deepcopy(self.session.topic['sources']),
                          'resolution': deepcopy(task['resolution']),
                          'lore_question': True, 'local_evidence_reuse': True,
                          'retrieval': {'strategy': 'previous_answer_evidence'}}
            else:
                if decision.get('route') == 'REUSE':
                    self._route_trace['reuse_fallback'] = 'evidence_missing_changed_or_different_subject'
                self._retrieval_calls += 1
                packet = retrieve_packet(task['query'], resolution=task['resolution'],
                    intent=task['intent'], max_chars=1500, db_file=self.db_file)
            packet['retrieval_question'] = retrieval_question
            packet['original_question'] = self._raw_user_text(user_text)
            packet['resolved_question'] = retrieval_question
            packet['agentic_followup'] = bool(task['followup'])
            packet['response_detail'] = bool(task['detail'])
            packet['context_root'] = task['context_root']
            packet['request_resolution'] = task['request_resolution']
            packet['resolution']['used_previous'] = bool(
                packet['resolution'].get('used_previous') or task['binding_resolution'].get('used_previous'))
            packet['task_normalizations'] = task['normalizations']
            packet['speech_selections'] = task['speech_selections']
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, RuntimeError) as error:
            self.session.trace.update(route='retrieval_error', error=str(error))
            return {'success': False, 'message': f'知识资料读取失败：{error}'}
        # Keep the speaker identity, actual device history and world facts
        # separate, including when a user supplies a false premise.
        if kind and not packet['subjects']:
            packet['subjects'] = [dict(SELF_SUBJECT)]
            packet['entity_ids'] = [IDENTITY.entity_id]
        if kind == 'personality_origin' and IDENTITY.entity_id not in packet['entity_ids']:
            packet['subjects'].insert(0, dict(SELF_SUBJECT))
            packet['entity_ids'].insert(0, IDENTITY.entity_id)
        if kind == 'identity':
            packet['intent'] = 'introduction'
        elif kind in ('preferences', 'favorite_person', 'trusted_person'):
            packet['intent'] = 'personality'
        packet['conversation_act'] = conversation_act(user_text)
        packet['missing_knowledge'] = (not packet['sources'] and not packet['subjects']
                                       and (packet['intent'] != 'general' or packet.get('lore_question'))
                                       and packet['conversation_act'] == 'general')
        print(f"[NpcTalkSkill] 图谱命中实体: {packet.get('entity_ids') or '无'}")
        ambiguities = packet.get('resolution', {}).get('ambiguities', [])
        previous_question = history_messages(self.session)[-2]['content'] if len(self.session.history) >= 2 else ''
        act = discourse_act(user_text, packet['subjects'], previous_question, self.session.pending_reference)
        pending_ids = set(self.session.pending_reference.get('candidate_ids', []))
        current_ids = set(packet.get('entity_ids', []))
        packet['clarification_resolved'] = bool(self.session.pending_reference and current_ids and not ambiguities and
                                                (not pending_ids or current_ids & pending_ids))
        if packet['clarification_resolved']:
            act = 'correction'
        elif (act == 'correction' and packet['intent'] == 'general' and
              subject_key(packet['subjects']) == subject_key(self.session.topic.get('subjects', []))):
            packet['intent'] = self.session.topic.get('intent', 'general')
        description = reference_description(user_text, packet['subjects'])
        if packet.get('resolution', {}).get('unresolved_reference'):
            reference = re.search(r'那个人|那位|(?<!其)[他她它](?:们|俩)?', user_text)
            description = reference.group() if reference else user_text
        packet['reference_description'] = description
        next_pending = {}
        if ambiguities or description:
            next_pending = {'question': answer_question, 'description': description, 'intent': packet['intent'],
                            'candidate_ids': [c['id'] for a in ambiguities for c in a['candidates']],
                            'ambiguities': ambiguities}
            act = 'clarification'
        try:
            messages, sent_sources, contexts = self._prompts().grounded(answer_question, packet, memory_context, act)
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, RuntimeError) as error:
            self.session.trace.update(route='context_error', generation_error=f'{type(error).__name__}: {error}')
            return self._safe_npc_reply(user_text,
                '这一轮的对话上下文没有整理完整，我先不猜。请明确要追问的内容。',
                'context_error', failure=True)
        self.session.trace = {
            "question": user_text, "retrieval_question": retrieval_question, "route": "grounded_dialogue",
            "answer_question": answer_question, "model": self.model,
            "resolved_question": packet.get('resolved_question'),
            "effective_followup": packet.get('agentic_followup', False),
            "request_resolution": packet.get('request_resolution', {}),
            "response_detail": packet.get('response_detail', False),
            "task_normalizations": packet.get('task_normalizations', []),
            "speech_selections": packet.get('speech_selections', []),
            "intent": packet["intent"], "subjects": packet["subjects"], "model_called": False,
            "resolution": packet.get('resolution', {}),
            "conversation_act": packet['conversation_act'],
            "clarification_resolved": packet['clarification_resolved'],
            "generation_accepted": False, "output_origin": "pending", "fallback_used": False,
            "retrieval": packet.get("retrieval", {}),
            "retrieved_source_ids": [] if packet.get('local_evidence_reuse') else [item["id"] for item in packet["sources"]],
            "reused_source_ids": [item["id"] for item in packet["sources"]] if packet.get('local_evidence_reuse') else [],
            "sent_sources": sent_sources, **contexts, "discourse_act": act,
            "estimated_prompt_tokens": sum(estimate_tokens(m["content"]) + 8 for m in messages),
            "model_input": messages, "ollama_calls": self._model_calls,
        }
        # Local boundary checks remain separate from model evidence review.
        reply, issues, used_source_ids = '', [], []
        # Negated requests for quotations still need a natural model response.
        quote_requested = any(
            re.search(r'原文|原话|逐字|直接引用', clause)
            and not re.search(r'不要|不用|不必|无需|别|避免', clause)
            for clause in re.split(r'[，,。！？!?；;\n]', answer_question))
        bounded = (self._bounded_source_reply(packet, sent_sources, answer_question)
                   if quote_requested else None)
        if ambiguities:
            choices = list(dict.fromkeys(c['name'] for a in ambiguities for c in a['candidates']))
            reply = '你说的是' + '，还是'.join(choices) + '？'
            self.session.trace.update(route='reference_clarification', output_origin='database_candidates')
        elif description:
            reply = f'你说的“{description}”具体是哪一位？告诉我名字，我就能接着聊。'
            self.session.trace.update(route='reference_clarification', output_origin='reference_clarification')
        elif bounded:
            reply, used_source_ids = bounded
            self.session.trace.update(route='source_excerpt', output_origin='source_excerpt')
        else:
            try:
                reply, issues, used_source_ids = self._grounded_answer(
                    answer_question, packet, memory_context, act)
            except (OSError, ValueError, KeyError, TypeError, AttributeError, sqlite3.Error, RuntimeError) as error:
                self.session.trace['generation_error'] = f'{type(error).__name__}: {error}'
                issues = ['generation_unavailable']
            if issues:
                # Evidence remains inspectable; it is not a generated NPC line.
                reply, used_source_ids = '本轮未能生成通过检查的回答。请查看终端错误或输入“查看证据”。', []
                self.session.trace.update(route='generation_failed', output_origin='generation_failure',
                                       generation_accepted=False, fallback_used=True)
            else:
                self.session.trace.update(generation_accepted=True,
                    output_origin='model')
        self.session.trace['guard'] = {
            'status': 'generation_failed' if issues else 'passed_rules', 'issues': list(dict.fromkeys(issues)),
            'scope': '来源编号、实体覆盖、身份与设备边界的本地检查；不等于逐句事实认证',
        }
        if self.session.trace.get('output_origin') == 'source_excerpt':
            aliases = [f'{m["alias"]}就是{m["name"]}' for m in
                packet.get('resolution', {}).get('mentions', [])
                if m.get('alias') and m.get('name') and m['alias'] != m['name']]
            aliases += [f'{s["resolved_target_alias"]}就是{s["object_name"]}'
                        for s in sent_sources if s['id'] in used_source_ids
                        and s.get('resolved_target_alias')
                        and s.get('object_name') != s['resolved_target_alias']]
            if aliases:
                reply = '；'.join(dict.fromkeys(aliases)) + '。' + reply
        self.session.trace.update(reply=reply, used_source_ids=used_source_ids)
        self.session.trace['sent_source_count'] = len(self.session.trace.get('sent_sources', []))
        self.session.trace.setdefault('semantic_review_performed', False)
        # Keep only evidence actually selected for the final answer in the
        # topic. A later "why" can use it without trusting old model prose.
        offered_sources = self.session.trace.get('sent_sources', sent_sources)
        selected_ids = set(used_source_ids)
        actual_sources = [source for source in offered_sources if source['id'] in selected_ids]
        return self._finish_reply(user_text, reply, packet, sources=actual_sources,
                                  success=bool(reply) and not issues, pending=next_pending)
