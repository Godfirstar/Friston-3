"""Shared query meaning and source eligibility for every retrieval channel."""
import re

from friston3.knowledge.lore_text import normalize, split_sentences, narrative_excerpt
from friston3.knowledge.entity_resolver import asks_personal_acquaintance, locative_event_title


PASSAGE_KINDS = ('profile', 'character', 'relationship', 'event', 'project')


CATEGORIES = {
    'introduction': ('identity', 'background', 'personality'),
    'personality': ('personality', 'background', 'identity'),
    'relationship': ('relationship',),
    'leadership': ('identity', 'relationship', 'background'),
    'experience': ('experience', 'event', 'event_summary', 'background'),
    'concept': ('identity', 'background', 'event_summary'),
    'general': ('identity', 'personality', 'background'),
}


RELATION_TOPICS = ('父亲', '母亲', '父母', '亲属', '朋友', '好友', '老师', '导师',
                    '学生', '上司', '同事', '合作', '兄弟', '姐妹', '师徒', '制造',
                    '研发', '设计', '恋人', '搭档', '监护人', '助手', '下属', '师父', '徒弟',
                    '上下级', '师生', '同窗', '战友', '亲戚', '情侣', '敌人', '对手', '交情',
                    '相处', '指引', '指导', '亲近', '尊敬')


PERSONALITY_WORDS = ('性格', '温柔', '坚定', '信任', '善良', '理想', '愿望', '冷静',
                '认真', '孤独', '固执', '热情', '喜欢', '讨厌', '认为', '信念')


LEADERSHIP_WORDS = ('公开领袖', '领导者', '领袖', '负责人', '主管', '总辖', '董事长', '主席', '老板', '指挥', '管理')


PERSON_KINDS = {'character', 'machine', 'fan_machine', 'dialogue_speaker'}


def requested_categories(question, intent):
    """Map ordinary question facets to the categories already stored in SQLite."""
    # SearchRequest 与 retrieve_lore 使用；把问题关注点映射为按优先级排列的资料类别元组。
    # CATEGORIES 是上方字典，例如 CATEGORIES['introduction'] 取得简介需要的类别顺序。
    if intent == 'relationship':
        return CATEGORIES[intent]
    if re.search(r'能力|擅长|本领|战斗|战力|作战|打(?:起)?仗|武器|兵器|招式|技能|剑术|格斗|法术|源石技艺|怎么打|'
                 r'(?:能|可以)(?:用来)?做什么|有什么(?:用|作用)|(?:有何|什么)(?:用途|作用)|用来(?:做|干)什么', question):
        return ('ability', 'identity', 'experience')
    if intent == 'experience' and re.search(r'加入|离开|退出|脱离|投奔|入职', question):
        return ('background', 'identity', 'experience', 'event')
    if (intent == 'experience' and re.search(r'为什么|为何|怎么会|原因', question)
            and re.search(r'担任|出任|就任|(?:当|做)[^，。！？?]{1,16}(?:官|师|员|长|家|顾问)', question)):
        return ('background', 'identity', 'experience', 'event')
    if intent == 'experience' and re.search(r'做了什么|采取|应对|处理', question):
        return ('background', 'ability', 'event', 'identity')
    return CATEGORIES[intent]


def detect_intent(question: str, entities=()) -> str:
    # SearchRequest 在未收到有效 intent 时调用；返回检索意图字符串，不是 Agent 的最终技能路线。
    # re.search 是标准库正则搜索，any/sum 对现有文字线索做判断；本函数不调用 LLM。
    q = normalize(question)
    if q in ('知道了', '我知道了', '了解了', '我了解了', '你知道吗', '你知道么'):
        return 'general'
    if (sum(entity['kind'] in PERSON_KINDS for entity in entities) >= 2
            and not re.search(r'分别|各自|相比|比较|区别', q)
            and re.search(r'碰到|遇到|一起|相处|相遇|结识|互动|来往|帮助|支持|照顾|'
                          r'对.+(?:怎么看|评价|看法|态度|影响)|(?:什么|怎样|如何)态度', q)):
        return 'relationship'
    if asks_personal_acquaintance(question) or re.search(
            r'.+(?:和|与|跟).+(?:认识|相识|结识|熟悉|熟不熟|见过|见面|相遇|交集)'
            r'(?:的|过|了)?(?:吗|么|不|呢)?$', q):
        return 'relationship'
    if (any(word in q for word in ('老大', '领袖', '谁领导', '谁负责', '负责人', '老板', '谁管', '谁说了算'))
            or re.search(r'谁(?:在|来|会|能|是|现在|目前|主要)*(?:负责|领导|管理|主持|掌管)', q)):
        return 'leadership'
    relation_topics = tuple(word for word in RELATION_TOPICS if word not in ('制造', '研发', '设计'))
    if any(word in q for word in ('关系', '谁造', '谁制造', '谁设计', '谁研发', '研发者', *relation_topics)):
        return 'relationship'
    if any(word in q for word in ('怎样的人', '什么样的人', '性格', '怎么看', '看法', '评价',
                                  '你觉得', '这么觉得', '这样觉得', '喜欢谁', '最喜欢', '讨厌', '动机')):
        return 'personality'
    if any(word in q for word in ('剧情', '故事', '经历', '过去', '当年', '发生', '后来',
                                  '为什么', '为何', '怎么会', '如何会', '做了什么', '起因', '结局',
                                  '说过', '哪一章', '证据', '原文', '台词')):
        return 'experience'
    if any(word in q for word in ('是什么', '什么是', '什么意思', '定义', '概念', '原理')):
        return 'concept'
    if (any(word in q for word in ('介绍', '是谁', '谁是', '认识', '了解', '说说', '聊聊', '档案', '详细'))
            or re.search(r'^(?:那|那么|请问)?你?(?:知不知道|知道).+', q)):
        return 'introduction'
    return 'general'


def query_rows(db, sql, params=()):
    # lore_search/retrieve_lore 等的查询辅助函数；db 是 SQLite 连接，sql 是查询串，params 是参数序列。
    # 返回普通字典列表；execute 的第二参数绑定 ?，不要把用户输入拼成 SQL 语句。
    cursor = db.execute(sql, params)
    # cursor 是游标；description 提供结果列信息，其中每项 [0] 是列名。
    names = [item[0] for item in cursor.description]
    # zip 把列名与一行的值逐对配起来，dict 变为 {列名:值}，for 对所有行重复这个过程。
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def find_entity(db, entity_id):
    # 实体解析后的回查入口；返回实体字典或 None，调用者不能把“不存在”当作空实体。
    rows = query_rows(db, 'SELECT * FROM entities WHERE id=?', (entity_id,))
    return rows[0] if rows else None


def query_terms(question, names=()):
    # 词法排序与节选选择使用；从问题中移除已知主体名/常见问法，返回其余关键词列表。
    # residual 是局部中间字符串；不修改调用者保留的原始 question。
    residual = re.sub(r'最开始|一开始|最早|起初', '最初', str(question))
    # 按长名称优先移除，避免短名先破坏完整名称；re.escape 让姓名被当作字面文字。
    for name in sorted(names, key=len, reverse=True):
        residual = re.sub(re.escape(name), ' ', residual, flags=re.I)
    for word in ('发生过什么', '是怎样的人', '是什么样的人', '有什么区别', '有什么关系',
                 '都有哪些人', '都有哪些', '是做什么的', '谁领导的', '有什么', '有哪些',
                 '有何', '是否', '如何', '请解释', '请问', '分别', '什么关系', '有何关系', '是什么',
                 '什么意思', '介绍一下', '你觉得', '你认识', '你知道', '给我讲讲', '告诉我',
                 '怎么看', '为什么', '谁是', '是谁', '介绍', '说说', '聊聊', '了解', '认识',
                 '一下', '详细', '简单', '简短', '简要', '大致', '大概', '原文', '档案', '关系', '性格', '故事', '经历', '剧情',
                 '那位', '那个人', '她', '他', '它', '你', '的', '和', '与', '跟', '吗', '呢', '啊', '吧'):
        residual = residual.replace(word, ' ')
    # findall 取至少两个字符的词段，dict.fromkeys 保序去重，[:6] 最多返回六个检索词。
    return list(dict.fromkeys(re.findall(r'[\u3400-\u9fff]{2,}|[A-Za-z0-9_]{2,}', residual)))[:6]


def subject_names(db, ids):
    """Source-owned aliases, never a hand-maintained list of world factions."""
    # SearchRequest、关系描述解析调用；返回 {实体ID: [规范名, 别名, ...]}，名字均来自数据库。
    result = {}
    for entity_id in ids:
        entity = find_entity(db, entity_id)
        if not entity:
            continue
        labels = [entity['name'], *[row['alias'] for row in query_rows(
            db, 'SELECT alias FROM aliases WHERE entity_id=? AND priority>=35', (entity_id,))]]
        result[entity_id] = list(dict.fromkeys(labels))
    return result


def mentioned_subjects(item, labels):
    # item 是段落字典，labels 是 subject_names 的输出；返回本段涉及的实体 ID 集合。
    # {} 内的“表达式 for ... if ...”是集合推导式；没有 key:value，因此不是字典。
    text = normalize(item.get('content', item.get('description', '')))
    return {entity_id for entity_id, aliases in labels.items()
            if item.get('ref_id') == entity_id or any(normalize(name) in text for name in aliases)}


def event_scope_context(db, question, entities):
    """An explicitly named event bounds a person's history, not its truth status."""
    # SearchRequest 构造时调用；返回活动范围字典或 None，供后续截取明确属于指定活动的段落。
    if not any(entity['kind'] in PERSON_KINDS for entity in entities):
        return None
    requested = {entity['id']: entity['name'] for entity in entities
                 if entity['kind'] == 'event' and normalize(entity['name']) in normalize(question)}
    if not requested:
        return None
    # A question explicitly extending beyond one event needs the broader
    # chronology. Multiple named events are retained together for comparison.
    if len(requested) == 1:
        title = re.escape(next(iter(requested.values())))
        if re.search(title + r'[》”」\"]*(?:结束)?(?:之后|以后|之前|以前|前后)', question):
            return None
    event_rows = query_rows(db, "SELECT id,name,source_key FROM entities WHERE kind='event'")
    events = {row['name']: row['id'] for row in event_rows}
    titles = '|'.join(re.escape(title) for title in sorted(events, key=len, reverse=True) if title)
    locative_titles = '|'.join(re.escape(row['name']) for row in sorted(event_rows, key=lambda row: len(row['name']), reverse=True)
                              if locative_event_title(row['name'], row['source_key'])) or r'(?!)'
    return {'requested': requested, 'events': events,
            'headers': re.compile(r'(?<![\w\u3400-\u9fff])[《“「\"]?(?P<title>' + titles + r')[》”」\"]?\s*[:：]'),
            # Ordinary quoted nouns can also be event titles in the DB.
            # Require book-title marks or explicit narrative grammar, so a
            # quoted concept does not silently become a chronology boundary.
            'mentions': re.compile(r'《(?P<book>' + titles + r')》|'
                                   r'(?:在|于)[“「\"](?P<quoted>' + titles + r')[”」\"](?:的)?'
                                   r'(?:故事|剧情|活动|事件|篇章)?(?:中|里|期间)|'
                                   r'[“「\"](?P<labeled>' + titles + r')[”」\"](?:的)?(?:故事|剧情|活动|事件|篇章)|'
                                   r'在(?P<narrative>' + locative_titles + r')(?:的)?(?:故事|剧情|活动|事件)?(?:中|里|期间)|'
                                   r'(?<![\w\u3400-\u9fff])(?P<locative>' + locative_titles + r')(?:中|里|期间)')}


def event_scoped_item(item, scope):
    """Select source-written event segments without reconstructing a timeline."""
    # SearchRequest.prepare_candidate 和 retrieve_lore 调用；返回保留活动片段的新字典或 None。
    # scope 为 None 时直接返回原 item；调用方必须理解这里并非每条分支都会复制对象。
    if not scope:
        return item
    text = narrative_excerpt(item['content'])
    requested, events = scope['requested'], scope['events']
    headers = list(scope['headers'].finditer(text))
    segments, matched = [], set()
    basis = 'event_title_section'

    def keep_mentions(fragment, initial=()):
        # 内部函数形成闭包，可读取外层 events/scope；返回 (句子列表, 已覆盖活动ID集合)。
        selected, covered, active = [], set(), set(initial)
        for sentence in split_sentences(fragment):
            mentions = {events[match['book'] or match['quoted'] or match['labeled'] or match['narrative'] or match['locative']]
                        for match in scope['mentions'].finditer(sentence)}
            if mentions:
                active = mentions if mentions <= requested.keys() else set()
            if active:
                selected.append(sentence)
                covered.update(active)
        return selected, covered

    if headers:
        # Narrative introductions can precede the first colon-style heading.
        segments, matched = keep_mentions(text[:headers[0].start()],
                                           [item['ref_id']] if item.get('ref_id') in requested else [])
        for index, header in enumerate(headers):
            event_id = events[header['title']]
            if event_id in requested:
                end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
                selected, covered = keep_mentions(text[header.start():end].strip(), [event_id])
                segments.extend(selected)
                matched.update(covered)
    elif item.get('ref_id') in requested:
        segments, matched = keep_mentions(text, [item['ref_id']])
        basis = 'event_document'
    elif item.get('ref_id') not in events.values():
        # A character paragraph may introduce an event once and continue with
        # pronouns. Retain its contiguous continuation, stopping at another
        # explicitly named event. A sentence mixing scopes is not separable.
        segments, matched = keep_mentions(text)
        basis = 'event_mention_section'
    if not segments:
        return None
    return {**item, 'content': ' '.join(segments), '_event_scope_ids': sorted(matched),
            '_event_scope_basis': basis}


class SearchRequest:
    """Resolve shared search conditions once; channel scores remain separate.

    Entity resolution belongs to the caller. An empty ``ids`` list means that
    no known subject was resolved; retrieval must not guess one from similarity.
    """

    def __init__(self, db, question, ids, intent=None):
        # retrieve_lore.build_evidence_packet 或 terra_kb CLI 用 SearchRequest(...) 创建本实例。
        # self 是新对象；db 为已有连接，question 为问题，ids 为实体 ID 序列，intent 为可选检索意图。
        # 统一把共享查询条件存成属性，让词法/语义通道使用同一主体和活动范围。
        self.question = question
        # := 同时赋值并判断 find_entity 的返回值；缺失实体不进入列表，[:8] 保留前八个主体。
        self.entities = [entity for entity_id in dict.fromkeys(ids)
                         if (entity := find_entity(db, entity_id))][:8]
        self.ids = [entity['id'] for entity in self.entities]
        self.names = [entity['name'] for entity in self.entities]
        # 外部 intent 只有属于现有类别时才使用，否则由本模块规则函数从问题推断。
        self.intent = intent if intent in CATEGORIES else detect_intent(question, self.entities)
        self.categories = requested_categories(question, self.intent)
        self.labels = subject_names(db, self.ids)
        self.event_scope = event_scope_context(db, question, self.entities)
        self.event_people = {entity['id'] for entity in self.entities
                             if entity['kind'] in PERSON_KINDS}
        self.person_focus = bool(self.entities) and all(
            entity['kind'] in PERSON_KINDS for entity in self.entities)

    def prepare_candidate(self, item):
        """Keep source-owned scope and annotate a candidate before scoring it."""
        # lore_search.ranked_search 与 hybrid_search 的 dense 路径共同调用；item 是候选段落字典。
        # 返回可评分的段落（补 matched_subject_ids/evidence_role）或 None（候选不符合本次范围）。
        item = event_scoped_item(item, self.event_scope)
        if item is None:
            return None
        # owner 是布尔值：段落所属实体是否就是本次问的主体；不表示资料经过事实核验。
        owner = item['ref_id'] in self.ids
        # 集合 | 是并集，把文字提及的人物与活动范围 ID 合到一起。
        mentioned = mentioned_subjects(item, self.labels) | set(item.get('_event_scope_ids', []))
        if (self.person_focus and (self.intent in ('introduction', 'personality')
                                  or self.categories[0] == 'ability') and not owner):
            return None
        if self.ids and not mentioned:
            return None
        if self.intent == 'relationship' and len(self.ids) >= 2 and len(mentioned) < 2:
            return None
        # 给字典写键会修改当前 item；sorted 把集合变为稳定顺序的列表，方便后续输出/比较。
        item['matched_subject_ids'] = sorted(mentioned)
        item['evidence_role'] = ('owner_summary' if owner else 'event_context'
                                 if item['kind'] == 'event' else 'mentioned_context')
        return item
