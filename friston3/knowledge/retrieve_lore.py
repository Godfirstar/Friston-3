"""Question-aware retrieval over source-linked lore summaries and direct edges.

The upstream wiki is an LLM-written secondary source. Its assertions remain
source assertions, not independently verified game dialogue. Relation direction
records the narrative owner and target, never an inferred parent/child direction.
"""

from __future__ import annotations

import re
from friston3.knowledge.lore_text import normalize, tokenize, split_sentences, narrative_excerpt
from friston3.knowledge.entity_resolver import embedded_group_name, ordinary_alias_span
from friston3.knowledge.retrieval_policy import (
    PASSAGE_KINDS, PERSON_KINDS, CATEGORIES, SearchRequest, detect_intent, requested_categories,
    query_rows, find_entity, query_terms, subject_names, mentioned_subjects,
    event_scoped_item,
)
from friston3.knowledge.lore_search import related_sources, relation_text, relation_body, event_causal_score
from friston3.knowledge.hybrid_search import hybrid_ranked_search
from friston3.npc.dialogue import conversation_act
from friston3.npc.identity import IDENTITY


def concept_term(query):
    """Share the exact asked term between retrieval and answer review."""
    # concept_definition 以及回复核对路径使用；从“X是什么”提取 X，无法明确提取时返回空字符串。
    # re.sub 去常见开头，re.fullmatch 要求整句符合模式，group(1) 取正则第一捕获组。
    query = re.sub(r'^(?:请问|请解释(?:一下)?|解释一下)\s*', '', str(query).strip())
    match = re.fullmatch(
        r'(.{1,48}?)(?:的(?:定义|概念)(?:是什么)?|是什么(?:意思)?|什么意思)'
        r'(?:呢|啊|呀)?[？?。！!\s]*', query)
    if not match:
        match = re.fullmatch(r'什么是(.{1,48}?)[？?。！!\s]*', query)
    return match.group(1).strip(' \t\r\n“”「」『』\"\'') if match else ''


def concept_definition(query, content):
    """Recognize an explicit term definition, never a mere case mention.

    This is a conservative textual cue, not independent verification. The
    complete source paragraph must accompany it, including qualifications.
    """
    # 本模块与 npc.prompts 使用；判断 content 是否含明确的术语定义句，返回 bool。
    # query 是问题，content 是来源正文；只检查文本形式，不把资料的事实等级提升为已核验。
    term = concept_term(query)
    if not term:
        return False
    definition = re.compile(
        r'^[*\-\s]*(?:所谓(?:的)?)?[“「『\"\']?' + re.escape(term)
        + r'[”」』\"\']?\s*[，,]?\s*(?:通常|一般|主要)?'
        r'(?:是指|指的是|被定义为|定义为|是一种|是一类|是一系列)'
        r'[^。！？!?；;]+', re.I)
    for sentence in split_sentences(narrative_excerpt(content)):
        if not definition.match(sentence) or re.search(r'[？?]', sentence):
            continue
        if re.search(r'比喻|比作|喻为|象征|仿佛|犹如|好比|'
                     r'(?:错误地|误以为|误认为)|'
                     r'(?:说法|定义|观点).{0,12}(?:错误|不准确|不成立|误解)', sentence):
            continue
        return True
    return False


def _complete_excerpt(text, budget):
    # build_evidence_packet 的内部 add_evidence 使用；返回预算内的完整句前缀或空串。
    # budget 是字符数，ends 是句末位置列表；不会随意从半句话中间截断。
    text = re.sub(r'\s+', ' ', str(text or '')).strip()
    if len(text) <= budget:
        return text
    ends = [match.end() for match in re.finditer(r'[。！？!?；;](?:[”」』])?', text)
            if match.end() <= budget]
    return text[:max(ends)].strip() if ends else ''


def _focused_excerpt(text, budget, question, names=(), anchor_names=()):
    """Choose compact complete sentences, including adjacent qualifications."""
    # add_evidence 调用；从原文选与 question/names 有关的完整句，返回仍保留限定语的节选字符串。
    # split_sentences/narrative_excerpt 来自 lore_text；它们做文字处理，不调用生成模型。
    text = narrative_excerpt(text)
    sentences = split_sentences(text)
    # Source summaries often enumerate actors as "name: ...". Every later
    # subjectless action in that block still belongs to this local heading,
    # not to another queried person mentioned elsewhere in the passage.
    heading = re.compile(r'^[*\-\s]*[\w\u3400-\u9fff·・（）() /\-]{1,48}\s*[:：]')
    # headings 是句子下标集合；enumerate 同时提供下标和句子，用于跟踪“人物名：”标题归属。
    headings = {index for index, sentence in enumerate(sentences) if heading.match(sentence)}
    section_start, active_heading = [], None
    for index in range(len(sentences)):
        if index in headings:
            active_heading = index
        section_start.append(active_heading)
    facets = query_terms(question, names)
    facets = list(dict.fromkeys([*facets, *re.findall(
        r'擅长|能力|本领|战斗|武器|技能|源石技艺|安保|修理|厨艺|'
        r'负责|工作|职务|领导|领袖|性格|理想|动机|起源|原因|尊敬|亲近', question)]))
    tokens = set(tokenize(' '.join(facets)).split())
    sentence_tokens = [set(tokenize(sentence).split()) for sentence in sentences]
    anchors = tuple(anchor_names or names)
    explanatory = bool(re.search(r'为什么|为何|原因|起因|因果|解释|怎样|如何|经历|经过|历史|变化|转变|'
                                 r'是什么|什么意思|定义|概念|怎么会|做了什么|采取|应对|处理', question))
    if not any(tokens & values or any(term in sentence for term in facets)
               or any(name in sentence for name in anchors)
               for sentence, values in zip(sentences, sentence_tokens)):
        # Semantic synonyms may share no literal vocabulary. Absence of a
        # literal match must not discard or fabricate a semantic excerpt.
        if headings:
            first = min(headings)
            end = min((index for index in headings if index > first), default=len(sentences))
            return _complete_excerpt(''.join(sentences[first:end]), budget)
        return _complete_excerpt(text, budget)
    dependent_start = re.compile(
        r'^[*\-\s]*(?:他|她|它|这|该|其|因此|所以|但|然而|不过|相反|也就是说|'
        r'与|和|跟|并|独自|负责|曾|后来|随后|最后|最终|然后|同时|一同|共同|继续|开始|'
        r'试图|希望|决定|选择|认为|表示|指出|遭到|受到|被|为了|为此|作为|在|于|是|'
        r'拥有|带领|领导|追寻|追查|寻找|面对|参与|留守|守护|保护|利用|通过|对|将)')
    qualification = re.compile(r'^[*\-\s]*(?:但|不过|然而|相反|只是|实际上|事实上|并非|并不|不是|不等于|'
                               r'除非|尽管|即使|据说|也就是说|这意味着|因此|所以|由此)')
    # "She claims/explains that ..." qualifies the following narration too.
    # A shorter later sentence must not turn that account into an unqualified
    # setting fact. The source's own point of view remains in the excerpt.
    framed_account = bool(sentences and re.search(
        r'认为|讲述|声称|宣称|相信|坚信|推测|猜测|据说|解释了|表示|回忆',
        sentences[0][:90]))
    best, best_score = '', -1
    for start in range(len(sentences)):
        if re.match(r'^[”」』]|^——出自', sentences[start]):
            continue
        # Do not make a pronoun/contrast sentence stand alone when its local
        # antecedent is available. Earlier windows include that antecedent.
        if section_start[start] is not None and start != section_start[start]:
            continue
        if framed_account and start:
            continue
        if start not in headings and (start or headings) and dependent_start.match(sentences[start]):
            continue
        window = ''
        values = set()
        for end in range(start, len(sentences)):
            if end > start and end in headings:
                break  # Never combine actions from two different actor headings.
            sentence = sentences[end]
            if len(window) + len(sentence) > budget:
                break
            window += sentence
            values.update(sentence_tokens[end])
            if end + 1 < len(sentences) and end + 1 not in headings and qualification.match(sentences[end + 1]):
                continue
            if (end + 1 < len(sentences) and re.match(
                    r'^(?:他|她|它|其)?(?:表示|说道|说|解释|指出|回答|回应)[:：]',
                    sentences[end + 1])):
                continue
            if any(window.count(left) > window.count(right) for left, right in
                   (('“', '”'), ('「', '」'), ('『', '』'), ('（', '）'), ('(', ')'))):
                continue
            if window.count('"') % 2:
                continue
            score = 8 * sum(term in window for term in facets) + len(tokens & values)
            score += 6 * sum(name in window for name in anchors)
            # Once a claim has all its useful context, extra biography should
            # not be rewarded merely for filling the available token budget.
            if explanatory:
                # A heading is not an explanation. Retain the following
                # causal development within this same local subject block.
                score += min(len(window), budget) / max(1, budget)
            else:
                score /= 1 + len(window) / 240 + (end - start) * .03
            if score > best_score:
                best, best_score = window, score
    return best


def _simple_person_pair(question, entities, intent):
    """Only general two-person relationship questions can stop at local text."""
    if intent != 'relationship' or len(entities) != 2 or not all(
            entity['kind'] in PERSON_KINDS for entity in entities):
        return False
    if re.search(r'为什么|为何|原因|经历|经过|历史|变化|转变|发展|后来|以前|过去|'
                 r'之前|之后|前后|起初|最初|最终|一直|始终|过程|不同|分别|对比|比较|《|》', question):
        return False
    return bool(re.search(r'(?:什么关系|何种关系|何关系|啥关系|关系如何|关系怎样|关系怎么样)'
                         r'(?:呢|吗|啊|呀|吧|的)?[？?。！!\s]*$', question))


def _pair_local_candidates(db, hits, entities, *, direct_context=False):
    """Keep explicit two-name windows, never synthesize a relationship label.

    Canonical names, unique canonical name prefixes, and the explicit alias
    table are usable lexical anchors. Free-form page aliases can denote a
    group/title and are deliberately insufficient for this narrow shortcut.
    """
    # build_evidence_packet 在两人物问题中调用；从候选 hits 选局部双人语境，返回段落字典列表。
    # db 用于回查名字别名，entities 是主体字典列表；结果仍是资料片段，不是新建的人物关系边。
    aliases = {}
    for entity in entities:
        names = [entity['name']]
        prefix = re.split(r'[·•]', entity['name'], maxsplit=1)[0]
        if prefix != entity['name'] and len(prefix) >= 2:
            owners = query_rows(db, 'SELECT id FROM entities WHERE name=? OR name LIKE ?', (prefix, prefix + '·%'))
            if len(owners) == 1:
                names.append(prefix)
        names.extend(row['alias'] for row in query_rows(db, '''SELECT a.alias FROM aliases a
            WHERE a.entity_id=? AND a.origin='alias_table' AND NOT EXISTS
            (SELECT 1 FROM aliases b WHERE b.alias_norm=a.alias_norm AND b.entity_id!=a.entity_id)''', (entity['id'],)))
        aliases[entity['id']] = [normalize(name) for name in names if name]

    def names_present(sentence, owner):
        normalized = normalize(sentence)
        return all(any(not embedded_group_name(normalized, match.end())
                       and not ordinary_alias_span(normalized, alias, match.start(), match.end())
                       for alias in names for match in re.finditer(re.escape(alias), normalized))
                   or (direct_context and entity_id == owner)
                   for entity_id, names in aliases.items())

    heading = re.compile(r'^[*\-\s]*[\w\u3400-\u9fff·・（）() /\-]{1,48}\s*[:：]')
    dependent = re.compile(r'^[*\-\s]*(?:因此|所以|然而|不过|但|相反|由此|当时|此后|随后|后来|在此)')
    qualification = re.compile(r'^[*\-\s]*(?:但|不过|然而|相反|只是|实际上|事实上|并非|并不|不是|不等于|'
                               r'除非|尽管|即使|据说|也就是说|这意味着|此后|后来|随后|最终|从此)')
    selected, seen = [], set()
    pair_ids = set(aliases)
    for item in hits:
        sentences = split_sentences(narrative_excerpt(item['content']))
        headings = [index for index, sentence in enumerate(sentences) if heading.match(sentence)]
        windows = []
        for index, sentence in enumerate(sentences):
            if not names_present(sentence, item['ref_id']) or not re.search(r'[。！!][”」』\"]*$', sentence):
                continue
            if item['ref_id'] not in pair_ids:
                # A third person's biography can mention both endpoints while
                # all actions still belong to that third person. Only accept
                # it when this sentence explicitly starts with a pair actor,
                # allowing a source-written time/place prefix before them.
                leading = sentence.strip()
                leading = re.sub(r'^——[^，。]{1,48}，', '', leading)
                leading = re.sub(r'^[*\-\s]*[\w\u3400-\u9fff·・（）() /\-]{1,48}\s*[:：]', '', leading)
                leading = re.sub(r'^(?:在|当|于)[^，。]{1,60}，', '', leading.strip())
                leading = re.sub(r'^(?:后来|最终|随后|当时)[，,]?', '', leading)
                leading = normalize(leading.lstrip('“「『\"\''))
                if not any(leading.startswith(alias) for names in aliases.values() for alias in names):
                    continue
            start = max((value for value in headings if value <= index), default=index)
            end = index + 1
            if start == index and index not in headings and index and dependent.match(sentence):
                start -= 1
            section_end = min((value for value in headings if value > index), default=len(sentences))
            while end < section_end and qualification.match(sentences[end]):
                end += 1
            window = ''.join(sentences[start:end])
            if len(window) > 420 or any(window.count(left) != window.count(right)
                                        for left, right in (('“', '”'), ('「', '」'), ('『', '』'))):
                continue
            if window.count('"') % 2:
                continue
            key = (normalize(window), item['ref_id'] if item['kind'] == 'event' else '')
            if key not in seen:
                seen.add(key)
                windows.append(window)
        if windows:
            selected.append({**item, 'content': ' '.join(windows), '_pair_local': True})
    # Do not select only a convenient subset of a larger multi-event or
    # potentially conflicting account. Such queries retain the normal chain.
    if not selected or len(seen) > 4 or sum(len(item['content']) for item in selected) > 900:
        return []
    return sorted(selected, key=lambda item: len(item['content']))


def _shared_neighbor_evidence(db, ids, question):
    """At most two real edges through a shared node, not an inferred pair edge."""
    # build_evidence_packet 找不到直接关系时调用；返回已有的两条共同邻接关系作为背景。
    # A→C 和 B→C 不推出 A→B；下面仅找共同节点并挑一组已有原述。
    if len(ids) != 2 or IDENTITY.entity_id in ids:
        return []
    neighborhoods = [related_sources(db, [entity_id], 'general', question, limit=80) for entity_id in ids]
    mapped = []
    for entity_id, edges in zip(ids, neighborhoods):
        neighbors = {}
        for edge in edges:
            neighbor = edge['dst'] if edge['src'] == entity_id else edge['src']
            if neighbor not in ids:
                neighbors.setdefault(neighbor, []).append(edge)
        mapped.append(neighbors)
    # 对字典 set(...) 取键集合；& 是交集，得到双方都连接的节点 ID。
    shared = set(mapped[0]) & set(mapped[1])
    choices = [[mapped[0][neighbor][0], mapped[1][neighbor][0]] for neighbor in shared]
    choices.sort(key=lambda edges: (-sum(edge['score'] for edge in edges), edges[0]['id']))
    return choices[0] if choices else []


def _citation(item):
    # add_evidence 调用；item 是带来源字段的段落/关系字典，返回简短“文件:行号”展示字符串。
    path = item.get('source_path') or item.get('source_id') or '来源未登记'
    # Files and line numbers are enough for the small prompt. Full URLs and the
    # pinned revision are retained in the parallel source record for inspection.
    path = str(path).replace('\\', '/')
    if '/data/' in path:
        path = 'data/' + path.split('/data/', 1)[1]
    if item.get('line_start'):
        path += ':' + str(item['line_start'])
        if item.get('line_end') != item['line_start']:
            path += '-' + str(item['line_end'])
    return path


def _provenance(item):
    # add_evidence 调用；返回出处与资料等级字典，未修改原 item。
    # 下方字典 | 字典合并字段；右侧同名键优先，保证 Wiki 摘要不会继承为“已独立核验”。
    project = item.get('source_type') == 'project_config' or item.get('status') == 'project'
    return {key: item.get(key) for key in ('source', 'source_id', 'source_path', 'source_url',
            'revision', 'line_start', 'line_end')} | {
        'source_type': 'project_config' if project else 'llm_summary',
        'status': 'project' if project else item.get('status', 'source_asserted'),
        'verified': bool(item.get('verified', True)) if project else False,
    }


def _identity_attribute_focus(question, entities):
    """Conservative attribute verification, independent of names or values."""
    if len(entities) != 1 or entities[0]['kind'] not in PERSON_KINDS:
        return ''
    q = normalize(question)
    if (re.search(r'感染者|矿石病', q)
            and re.search(r'是不是|是否|还是|现在|目前|对吧|吧$|感染过|感染了|得了|患有', q)):
        return 'infection'
    if (re.search(r'种族|族属|血统', q)
            or re.search(r'(?:是|是否为|属于|算不算).{1,12}族(?:吗|么|还是|或|$)', q)):
        return 'species'
    if re.search(r'本名|真名|原名|真实身份|真正身份', q):
        return 'identity'
    if (detect_intent(question) != 'relationship'
            and re.search(r'(?:是|属于|来自|出身|出生).{1,30}(?:对吧|对吗|吗|吧)$', q)):
        return 'profile'
    return ''


def _identity_attribute_text(content, focus):
    """Keep the direct assertion and adjacent qualifications as whole text.

    A profile's ownership/category alone does not prove an attribute. This
    only recognizes explicit wording; it never extracts or invents a value.
    """
    pattern = (r'种族|族属|血统|族(?:干员|人|女|男|成员)|(?:是|为|属于)[^。！？]{0,12}族'
               if focus == 'species' else r'出身|来自|出生|种族|族属|血统|族(?:干员|人|女|男|成员)|'
               r'(?:是|为|属于)[^。！？]{0,12}族' if focus == 'profile' else
               r'本名|真名|原名|真实身份|真正身份')
    sentences = [part.strip() for part in re.findall(
        r'[^。！？!?；;]+[。！？!?；;]+[”」』\"]*|[^。！？!?；;]+$',
        narrative_excerpt(content)) if part.strip()]
    if focus == 'infection':
        # Select every status sentence in source order, including later
        # corrections. Helping patients is not the subject's own diagnosis.
        status = re.compile(
            r'非感染者|未感染|没有感染|并未感染|感染(?:了|上|矿石病)|'
            r'(?:疑似|疑有|疑为|怀疑|确诊|诊断|检查|检测|证明|确认|澄清|揭露)'
            r'[^。！？；]{0,30}(?:感染|矿石病)|'
            r'(?:患有|患上|罹患|染上|得了|得过|身患)[^。！？；]{0,12}矿石病|'
            r'(?:是|成为|身为|作为|现为|曾为|属于|系|一名|一位)[^，。！？；]{0,20}感染者|'
            r'感染者(?:身份|真相|状态)|'
            r'(?:^|[，：])(?:重度|轻度|男性|女性)?感染者(?=[，。；]|$)')
        qualifying = re.compile(
            r'^(?:但|然而|不过|实际上|事实上|后来|随后|最终|经检查|经检测|经诊断)'
            r'.*(?:并非|不是|未曾|没有|误诊|误判|澄清|属实|不实)')
        indices = {index for index, sentence in enumerate(sentences) if status.search(sentence)}
        for index in list(indices):
            following = index + 1
            while following < len(sentences) and qualifying.search(sentences[following]):
                indices.add(following)
                following += 1
        return ''.join(sentence for index, sentence in enumerate(sentences) if index in indices)
    for index, sentence in enumerate(sentences):
        if not re.search(pattern, sentence):
            continue
        selected = [sentence]
        for following in sentences[index + 1:index + 3]:
            if not re.search(r'自(?:称|视|认)|认同|身份|却|但|然而|并非|不是|不等于|实际上|事实上|所谓', following):
                break
            selected.append(following)
        return ''.join(selected)
    return ''


def _organization_intro_focus(question, entities, intent):
    """A reference-only topic can have a definition inside another profile."""
    if (len(entities) != 1 or entities[0]['kind'] != 'reference'
            or intent not in ('introduction', 'concept', 'general')):
        return False
    q = normalize(question)
    if re.search(r'成员|员工|人物|角色|哪些|谁|领导|领袖|负责人|关系|经历|历史|故事|为什么|如何|怎样', q):
        return False
    return bool(re.search(r'做什么|什么(?:公司|机构|组织|协会)|是什么|介绍|说说|聊聊|了解|业务|主营', q))


def _organization_definition_text(content, names):
    """Find a literal organization definition, not a member's job title.

    The organization name must be adjacent to its type. A person serving an
    organization does not by itself establish its type or general business.
    """
    organization_type = r'公司|机构|组织|协会|商会|研究所|研究院|集团|企业'
    sentences = [part.strip() for part in re.findall(
        r'[^。！？!?；;]+[。！？!?；;]+[”」』\"]*|[^。！？!?；;]+$',
        narrative_excerpt(content)) if part.strip()]
    best_text, best_score = '', 0
    for sentence in sentences:
        if len(sentence) > 500:
            continue
        for name in names:
            for match in re.finditer(re.escape(name), sentence, re.I):
                suffix = sentence[match.end():]
                definition = re.match(
                    r'(?:(?:是|为)(?:一[家个所])?)?[^，。；!?！？的]{0,14}?(?:' + organization_type + r')', suffix)
                if not definition:
                    continue
                if re.search(r'在|支持|帮助|合作|拥有|设立|成立|创建|创办|加入|建立|成员|员工|雇员',
                             definition.group()):
                    continue  # Another organization may be the verb's object.
                # Keep the literal sentence, including negation and the
                # employee's own scope. Never synthesize an organization card.
                score = 2 + 2 * bool(re.match(r'(?:是|为)', suffix))
                score += 3 * bool(re.search(
                    r'提供.{1,30}(?:服务|保障|咨询|支持)|业务|主营|经营|从事|专门|致力于|承担|承接', sentence))
                if score > best_score:
                    best_text, best_score = sentence, score
    return best_text, best_score


def build_evidence_packet(db, question, ids, max_chars=1900, *, intent=None):
    """Build a query-focused packet from direct edges and source-owned text.

    The graph channel is bounded local retrieval, not global GraphRAG or a
    relation inference engine. Every excerpt keeps its original source status.
    """
    # terra_kb.retrieve_packet 调用这个总入口；db 是已打开连接，question 是查询，ids 是主体 ID。
    # max_chars 是证据文字预算；返回 {context,intent,subjects,sources,lore_question,retrieval} 字典。
    # 下游依次使用 SearchRequest、related_sources、hybrid_ranked_search 和本文件的节选辅助函数。
    # 它只收集/整理证据；最终台词生成在 NpcTalkSkill._grounded_answer，不在这里训练或调用 LLM。
    # 阶段一：把问题与已解析主体组成共享请求，后面所有召回通道使用同一套查询语义。
    max_chars = max(0, int(max_chars))
    # SearchRequest 是 retrieval_policy.py 的类；调用类名创建实例，构造器会回查实体与别名。
    request = SearchRequest(db, question, ids, intent)
    # 多变量赋值按位置接收属性；focused 是确认存在的主体 ID 列表，names 是对应的名字列表。
    entities, focused, names = request.entities, request.ids, request.names
    intent, labels = request.intent, request.labels
    event_scope, event_people = request.event_scope, request.event_people
    attribute_focus = _identity_attribute_focus(question, entities)
    organization_intro = _organization_intro_focus(question, entities, intent)
    concept_search = intent == 'concept' and not focused
    # Query intent survives a retrieval miss. Otherwise an unanswered lore
    # question could be misrouted as ordinary chat and skip factual review.
    lore_request = bool(focused or intent != 'general' or re.search(
        r'哪些(?:人|角色|人物|组织|阵营)|哪几(?:位|个|名)|什么(?:人|角色|组织|阵营)|'
        r'世界观|原作|剧情|人物|角色|组织|阵营|资料|设定|'
        r'.{1,40}(?:和|与).{1,40}(?:之间|冲突|矛盾|差异)', question))
    # blocks 保存给模型看的证据文字块；sources 保存可追溯的结构化来源记录，二者按成功加入同步更新。
    blocks, sources = [], []
    # 两个 set 分别按规范化正文、段落 ID 去重，避免同一出处从不同召回通道重复占预算。
    used_text, used_passages = set(), set()
    pair_question = intent == 'relationship' and len(focused) >= 2
    # retrieval 是诊断字典：记录候选数、覆盖主体、缺口等，帮助解释“这轮为什么找到了这些资料”。
    retrieval = {'strategy': 'hybrid_with_local_graph', 'graph_scope': 'source_edges_and_shared_neighbor',
                 'direct_relation_count': 0, 'candidate_count': 0, 'covered_subject_ids': [],
                 'missing_subject_ids': list(focused), 'expansion_used': False}
    retrieval['requested_categories'] = list(requested_categories(question, intent))
    if event_scope:
        retrieval['event_scope_ids'] = list(event_scope['requested'])
        retrieval['event_scope_policy'] = 'explicit_event_documents_and_source_segments'
    if attribute_focus:
        retrieval['identity_attribute_focus'] = attribute_focus
        retrieval['direct_attribute_source_ids'] = []

    def remaining():
        # 闭包读取外层 max_chars/blocks；join 用换行拼已有块，再扣掉下一块所需的分隔换行。
        # bool(blocks) 在数值运算中是 0 或 1，不是字符串长度。
        return max_chars - len('\n'.join(blocks)) - bool(blocks)

    def add_evidence(item, label, raw_text, cap, **evidence):
        # 本函数下方各分支调用的统一入口；item 为资料记录，label 为显示标题，raw_text 为原述。
        # cap 是单段字符上限，**evidence 把其余命名参数收为字典（id/kind/subject_id 等）。
        # 返回 True 表示确实加入了 blocks/sources，False 表示未选用；调用方据此更新覆盖信息。
        metadata = _provenance(item)
        # metadata 由本文件辅助函数生成；state 是给模型看的资料等级文字，不是机器人观测 state。
        state = ('项目同人配置' if metadata['source_type'] == 'project_config' else
                 '模型生成摘要；存疑' if metadata['status'] == 'uncertain' else '模型生成摘要；未独立核验')
        note = evidence.get('context_note', '')
        prefix = f'[{label}｜{state}] '
        suffix = (f'（证据用途：{note}）' if note else '') + f' [出处：{_citation(item)}]'
        # 给出处与标签先留空间，再把剩余字数分配给正文；不是简单对全部字符串直接 [:cap]。
        budget = min(cap, remaining() - len(prefix) - len(suffix))
        # Direct assertions already carry a focused, qualified excerpt. Do
        # not optimize it for the proposed (possibly false) value in the query.
        if evidence.get('evidence_role') in (
                'direct_identity_attribute', 'pair_local_context', 'direct_concept_definition'):
            qualified_text = narrative_excerpt(raw_text)
            excerpt = qualified_text if len(qualified_text) <= budget else ''
        elif (evidence.get('kind') == 'relation'
              and len(narrative_excerpt(raw_text)) <= budget):
            excerpt = narrative_excerpt(raw_text)
        elif (item.get('category') == 'identity' and item.get('ref_id') in focused
              and intent in ('introduction', 'personality')):
            excerpt = _complete_excerpt(narrative_excerpt(raw_text), budget)
        elif event_scope and len(narrative_excerpt(raw_text)) <= budget:
            # This is already a bounded event segment; its following pronouns
            # and causal explanation belong to the same source-written span.
            excerpt = narrative_excerpt(raw_text)
        else:
            excerpt = _focused_excerpt(raw_text, budget, question, names,
                                       [entity['name'] for entity in entities
                                        if entity['id'] in event_people] if event_scope else
                                       [evidence['subject_name']] if evidence.get('kind') == 'relation' else ())
        # excerpt 是上面根据资料类别选出的原文片段；空片段或重复正文不加入来源列表。
        if not excerpt or normalize(excerpt) in used_text:
            return False
        if (evidence.get('evidence_role') == 'direct_concept_definition'
                and not concept_definition(question, excerpt)):
            return False
        if item.get('ref_id') and 'matched_subject_ids' in evidence:
            # Coverage belongs to the delivered excerpt, not to actors in a
            # different paragraph of the original multi-actor passage.
            evidence['matched_subject_ids'] = sorted(mentioned_subjects({**item, 'content': excerpt}, labels)
                                                     | set(evidence.get('event_scope_ids', [])))
            if (len(focused) == 2 and intent == 'experience' and not event_scope
                    and re.search(r'为什么|为何|原因|亲近|尊敬|冲突|矛盾', question)
                    and len(evidence['matched_subject_ids']) < 2):
                return False  # An endpoint's unrelated story cannot explain this pair.
        if event_scope and len(evidence.get('event_scope_ids', [])) > 1:
            actual_scope = event_scoped_item({**item, 'content': excerpt}, event_scope)
            if not actual_scope:
                return False  # A shortened mixed timeline needs an explicit anchor.
            evidence['event_scope_ids'] = actual_scope['_event_scope_ids']
            evidence['matched_subject_ids'] = sorted(mentioned_subjects({**item, 'content': excerpt}, labels)
                                                     | set(actual_scope['_event_scope_ids']))
            scope_names = '、'.join('《' + event_scope['requested'][event_id] + '》'
                                   for event_id in actual_scope['_event_scope_ids'])
            evidence['context_note'] = '限于' + scope_names + '的来源段落；不补入其他事件的发展，证据等级不变。'
            suffix = f"（证据用途：{evidence['context_note']}） [出处：{_citation(item)}]"
        block = prefix + excerpt + suffix
        if len(block) > remaining():
            return False
        # append/add 修改闭包捕获的列表/集合内容，所以不需要 nonlocal 重新绑定外层变量。
        blocks.append(block)
        used_text.add(normalize(excerpt))
        # ** 展开两个字典并添加最终节选；sources 保存实际送出的正文，而非未裁剪的候选全文。
        sources.append({**metadata, **evidence, 'text': excerpt, 'block': block})
        return True

    # A greeting or a request about the current speaker's unsupported memories
    # is not a reason to fill the prompt with arbitrary semantic neighbors.
    self_memory = any(word in question for word in ('你去过', '你经历', '你的经历', '你的过去', '你还记得', '我们上次'))
    self_acquaintance = bool(re.search(r'你(?:都|还|最)?(?:认识|了解|知道|熟悉)(?:谁|哪些人|什么人|多少人|哪几位|哪些角色)', question))
    social_act = conversation_act(question)
    social_only = not focused and social_act != 'general'
    if not focused and (self_memory or self_acquaintance or len(normalize(question)) <= 2
                        or social_only):
        if social_only:
            retrieval.update(skipped_reason='social_dialogue', conversation_act=social_act,
                             source_count=0, evidence_gaps=[])
        return {'context': '', 'intent': 'general' if social_only else intent, 'subjects': [], 'sources': [],
                'lore_question': False, 'retrieval': retrieval}

    # 阶段二：先查明确身份属性与直接关系，避免一条已经足够的原述被大量背景资料挤走。
    if attribute_focus:
        entity = entities[0]
        identity_rows = query_rows(db, '''SELECT p.*,s.source_type,s.path AS source_path,s.url AS source_url,s.revision
            FROM passages p LEFT JOIN sources s ON s.id=p.source_id
            WHERE p.ref_id=? AND p.category='identity' ORDER BY p.id LIMIT 8''', (entity['id'],))
        for item in identity_rows:
            raw = _identity_attribute_text(item['content'], attribute_focus)
            if not raw:
                continue
            if add_evidence(item, entity['name'] + '｜身份属性原述', raw, 620,
                    id=f"passage:{item['id']}", kind='overview', scope=item['scope'],
                    subject_id=entity['id'], subject_name=entity['name'],
                    section=item['section'], passage_id=item['id'], matched_subject_ids=[entity['id']],
                    retrieval_channels=['owner_attribute'], evidence_role='direct_identity_attribute',
                    attribute_focus=attribute_focus,
                    context_note='直接身份表述；保留否定、自称与认同限定。证据等级不变，真实冲突须说明。'):
                used_passages.add(item['id'])
                retrieval['direct_attribute_source_ids'].append(item['id'])
            if len(retrieval['direct_attribute_source_ids']) >= 2:
                break

    relational_context = intent == 'experience' and len(focused) >= 2
    # related_sources 从 lore_search 导入，返回已有关系记录；每条仍附有 source/passages 的来源。
    relations = related_sources(db, focused, 'relationship' if relational_context else intent, question, limit=18)
    if event_scope:
        scoped_relations = []
        for relation in relations:
            scoped = event_scoped_item({**relation, 'ref_id': relation['src'],
                                         'content': relation['description']}, event_scope)
            if scoped:
                scoped_relations.append({**relation, 'description': scoped['content'],
                                         '_event_scope_ids': scoped['_event_scope_ids'],
                                         '_event_scope_basis': scoped['_event_scope_basis']})
        relations = scoped_relations
    direct = relations if intent in ('relationship', 'leadership') or relational_context else []
    retrieval['direct_relation_count'] = len(direct)
    graph_context = []

    def add_relation(relation, role='direct_relationship'):
        # 把关系记录转换为 add_evidence 的统一输入；subject/object 表示原述两端，不猜关系方向。
        label = f"{relation['src_name']}的档案提及{relation['target_label']}｜关系原述"
        group_note = ''
        alias = ''
        if relation.get('target_is_group'):
            role = 'group_relationship_context'
            group_note = '群体条目原述；按每句话明确指名的行为主体理解，不能把所有成员的行动归给同一人。'
            parts = {normalize(part): part for part in re.split(r'[、,，]', relation['target_label'])}
            aliases = subject_names(db, [relation['dst']]).get(relation['dst'], [])
            alias = next((parts[normalize(name)] for name in aliases if normalize(name) in parts), '')
            if alias and normalize(alias) != normalize(relation['dst_name']):
                group_note = f"库内别名映射：{alias}即{relation['dst_name']}。" + group_note
        if role == 'graph_context':
            label += '（共同邻接，不代表两人有关系）'
        if relation.get('status') == 'uncertain':
            group_note += '关系原述存疑；保留“似乎”等限定，其他背景不能将其提升为已确认事实。'
        # relation_text 来自 lore_search，只整理原述格式；下方 add_evidence 才决定是否放入证据包。
        text = relation_text(db, relation)
        if add_evidence(relation, label, text, 350,
                id=f"relation:{relation['id']}", kind='relation', scope=relation['scope'],
                subject_id=relation['src'], subject_name=relation['src_name'],
                object_id=relation['dst'], object_name=relation['dst_name'],
                target_label=relation.get('target_label'), relation=relation['relation'],
                resolved_target_alias=alias,
                original_text=relation['target_label'] + '：' + relation_body(relation),
                passage_id=relation['passage_id'], evidence_role=role,
                **({'context_note': group_note} if group_note and not event_scope else {}),
                **({'event_scope_ids': relation['_event_scope_ids'],
                    'context_note': '仅适用于所标事件原述，不据此补入其他事件的发展。'} if event_scope else {}),
                matched_subject_ids=sorted(set(focused) & {relation['src'], relation['dst']})):
            used_passages.add(relation['passage_id'])

    # Direct pair statements are economical and precise. Single-subject
    # questions should still retrieve prose explaining the requested facet.
    for relation in direct[:3 if pair_question else 1 if intent == 'leadership' else 2]:
        add_relation(relation)

    # Explicit pair edges already answer a simple relationship question.
    # Wider/causal questions still retrieve prose and semantic context.
    direct_pair_answer = bool(sources and direct and not event_scope
                              and not any(relation.get('target_is_group') for relation in direct)
                              and _simple_person_pair(question, entities, intent))
    direct_attribute_answer = bool(retrieval.get('direct_attribute_source_ids'))
    # 阶段三：直接原述未充分覆盖的问题，才调用混合检索，取得词法+语义排序的候选段落 hits。
    # hits 中的候选尚不等于 sources 中真正交付的证据，后面还会筛选/裁剪/分配预算。
    hits = [] if direct_pair_answer or direct_attribute_answer else hybrid_ranked_search(
        db, request, limit=30 if event_scope or concept_search else 14,
        kinds=tuple(kind for kind in PASSAGE_KINDS if kind != 'relationship'))
    if direct_pair_answer:
        retrieval['strategy'] = 'direct_source_relationships'
    elif direct_attribute_answer:
        retrieval['strategy'] = 'direct_source_attributes'
    if event_scope:
        # := 同时保存活动截取结果并判断它是否有效；列表里仅保留仍属于本次活动范围的片段。
        hits = [scoped for item in hits if (scoped := event_scoped_item(item, event_scope))]
        for item in hits:
            item['matched_subject_ids'] = sorted(mentioned_subjects(item, labels)
                                                 | set(item['_event_scope_ids']))
        hits = [item for item in hits if event_people & set(item['matched_subject_ids'])]
        if re.search(r'为什么|为何|原因|起因|怎么会|如何会', question):
            actor_names = [entity['name'] for entity in entities if entity['id'] in event_people]
            facets = query_terms(question, names)
            hits.sort(key=lambda item: -event_causal_score(item['content'], actor_names, facets))
    retrieval['candidate_count'] = len(hits)
    if direct and relational_context and not event_scope:
        hits = [item for item in hits if item['ref_id'] in focused]
    if (len(entities) == 1 and entities[0]['kind'] in PERSON_KINDS and not event_scope
            and re.search(r'为什么|为何|原因|起因|怎么会|如何会', question)):
        # A source-written answer to "why" may live in the event record.
        # Owner biographies and shared narrative relevance do not outrank
        # an explicit causal answer merely through multi-channel overlap.
        facets = query_terms(question, names)
        hits.sort(key=lambda item: -event_causal_score(
            item['content'], names, facets, owned=item['ref_id'] in focused))
    if requested_categories(question, intent)[0] == 'ability' and focused:
        focused_hits = [item for item in hits if item['ref_id'] in focused
                        and item['category'] in ('ability', 'identity', 'background')]
        if focused_hits:
            hits = focused_hits
    people = [entity['id'] for entity in entities if entity['kind'] in PERSON_KINDS]
    if (len(people) == 1 and any(entity['kind'] == 'reference' for entity in entities)
            and intent == 'experience' and not event_scope):
        owner_hits = [item for item in hits if item['ref_id'] == people[0]]
        if owner_hits:
            hits = owner_hits
    # Semantic matches need not repeat the question's vocabulary.
    # Only reject pure lexical candidates with virtually no lexical support;
    # source quality and factual support remain separate from retrieval rank.
    if not focused:
        query_tokens = set(tokenize(' '.join(query_terms(question))).split())
        hits = [hit for hit in hits if 'semantic' in hit.get('retrieval_channels', [])
                or (query_tokens and len(query_tokens & set(tokenize(hit['content']).split()))
                    >= min(2, len(query_tokens)))]

    # 根据问题类型在这批候选内重排；这一步不再无条件扩大检索范围或把匹配分数当作真值。
    if concept_search:
        # Reorder only the bounded hybrid candidates. A biography's repeated
        # mentions must not outrank an explicit definition of the asked term.
        hits = [{**item, '_concept_definition': concept_definition(question, item['content'])}
                for item in hits]
        hits.sort(key=lambda item: not item['_concept_definition'])
        retrieval['concept_definition_policy'] = 'explicit_term_definition_first'

    # For a narrow organization overview, a direct definition is more useful
    # than several employee biographies. Only activate when such a sentence
    # actually exists in the retrieved candidates; otherwise retain recall.
    if organization_intro:
        definitions = []
        for index, item in enumerate(hits):
            raw, score = _organization_definition_text(item['content'], labels.get(focused[0], names))
            if raw:
                definitions.append((score, -index, item, raw))
        if definitions:
            _, _, item, raw = max(definitions, key=lambda entry: entry[:2])
            hits = [{**item, 'content': raw, '_organization_definition': True}]
            retrieval['organization_definition_source_ids'] = []

    pair_local_note = ('先说清楚原文记录的双方互动及所属事件，保留动作、时空、否定和不确定限定。'
                       '本段只支持这些具体陈述，不据此外推长期关系；证据等级不变。')
    if pair_question and direct and len(entities) == 2 and not event_scope:
        # A verified edge permits the source owner to be an implicit subject
        # in its supporting prose. Keep only local mentions of the other
        # endpoint, never a complete profile competing with a qualified edge.
        local_hits = _pair_local_candidates(db, hits, entities, direct_context=True)
        # A missing/broad local window must not restore the unfiltered owner
        # summaries. The already delivered direct sources remain sufficient
        # evidence for their assertions without unrelated dialogue snippets.
        hits = sorted(local_hits, key=lambda item: item['category'] == 'identity')
        retrieval['pair_evidence_policy'] = ('direct_edges_with_local_support'
                                             if hits else 'direct_edges_only')
    if not direct and _simple_person_pair(question, entities, intent):
        local_hits = _pair_local_candidates(db, hits, entities)
        # Activate only when every distinct local window fits. Otherwise the
        # ordinary chain retains room to explain a larger/contradictory record.
        required = sum(len(item['content']) + len(_citation(item)) + len(pair_local_note)
                       + len((find_entity(db, item['ref_id']) or {}).get('name', ''))
                       + len(item['section']) + 90 for item in local_hits)
        if local_hits and required <= remaining():
            hits = local_hits
            retrieval['pair_local_source_ids'] = []
            retrieval['pair_evidence_policy'] = 'complete_two_name_local_windows'
    if pair_question and not direct and not event_scope and 'pair_local_source_ids' not in retrieval:
        graph_context = _shared_neighbor_evidence(db, focused, question)
        retrieval['expansion_used'] = bool(graph_context)

    # Fair coverage: take one useful passage per explicit subject first, then
    # fill by relevance. A long-name entity must not consume every source slot.
    # 阶段四：建立实际送入证据包的次序，多主体/多活动先各分到有用资料，再填相关背景。
    ordered = []
    if intent == 'personality' and len(focused) == 1 and not event_scope:
        profile = next((hit for hit in hits if hit['ref_id'] == focused[0]
                        and hit['category'] == 'identity'), None)
        if profile is None:
            rows = query_rows(db, '''SELECT p.*,s.source_type,s.path AS source_path,s.url AS source_url,s.revision
                FROM passages p LEFT JOIN sources s ON s.id=p.source_id
                WHERE p.ref_id=? AND p.category='identity' ORDER BY p.id LIMIT 1''', (focused[0],))
            profile = rows[0] if rows else None
        if profile is not None:
            ordered.append(profile)
    if event_scope:
        # Each named event gets a document/segment before one character's
        # biography can use the budget. Keep all events in a comparison.
        for event_id in event_scope['requested']:
            own = next((hit for hit in hits if hit['ref_id'] == event_id), None)
            if own is None:
                own = next((hit for hit in hits if event_id in hit.get('_event_scope_ids', [])), None)
            if own is not None and all(own['id'] != item['id'] for item in ordered):
                ordered.append(own)
    if len(focused) >= 2 and not pair_question:
        for entity_id in focused:
            own = next((hit for hit in hits if hit['ref_id'] == entity_id
                        and intent in ('introduction', 'personality')
                        and hit['category'] == CATEGORIES[intent][0]), None)
            if own is None:
                own = next((hit for hit in hits if hit['ref_id'] == entity_id), None)
            if own is None:
                own = next((hit for hit in hits if entity_id in hit.get('matched_subject_ids', [])), None)
            if own is not None and all(own['id'] != item['id'] for item in ordered):
                ordered.append(own)
    # extend 消费生成器，逐条加入尚未选择的 ID；all 要求与已有每条资料都不重复。
    ordered.extend(hit for hit in hits if all(hit['id'] != item['id'] for item in ordered))
    max_sources = (len(hits) if 'pair_local_source_ids' in retrieval else
                   max(2, len(sources) + 1) if retrieval.get('pair_evidence_policy') == 'direct_edges_with_local_support' else
                   1 if 'organization_definition_source_ids' in retrieval else
                   3 if retrieval.get('direct_attribute_source_ids') else
                   max(2, len(focused)) if requested_categories(question, intent)[0] == 'ability' else
                   max(2, len(focused)) if intent == 'introduction' else
                   max(2, min(6, len(focused) * 2)) if intent == 'personality' else
                   3 if pair_question or intent == 'leadership' else
                   3 if len(focused) == 1 and intent == 'experience'
                   and re.search(r'为什么|为何|原因|起因|怎么会|如何会', question) else
                   4 if direct else max(4, min(8, len(focused) + 3)))
    # item 为一条候选字典；used_passages 防止前面已加入的直接关系/属性重复进入。
    for item in ordered:
        if item['id'] in used_passages or item['category'] == 'relationship':
            continue
        owner = find_entity(db, item['ref_id'])
        if not owner:
            continue
        # Allocate enough room for every subject while retaining useful whole
        # sentences. Unused reservations naturally flow to later evidence.
        # uncovered 是尚未被实际 sources 覆盖的主体；any 扫描已交付来源而不是全部检索命中。
        uncovered = [entity_id for entity_id in focused if not any(
            entity_id in source.get('matched_subject_ids', []) for source in sources)]
        cap = (280 if intent in ('introduction', 'personality') else 340 if direct else 420) if len(uncovered) <= 1 else max(150, min(340, remaining() // len(uncovered) - 160))
        matched = sorted(mentioned_subjects(item, labels) | set(item.get('_event_scope_ids', [])))
        direct_attribute_available = bool(retrieval.get('direct_attribute_source_ids'))
        context_note = ('仅作事件或称谓背景，不能单靠泛称覆盖直接身份表述；真实冲突须说明。'
                        if direct_attribute_available else '')
        role = ('identity_attribute_background' if direct_attribute_available
                else item.get('evidence_role', 'owner_summary'))
        if item.get('_organization_definition'):
            cap = 500
            role = 'direct_organization_definition'
            context_note = '此句直接说明组织性质或业务；人物岗位仅属于该人物，不能外推成员构成。证据等级不变。'
        if item.get('_concept_definition'):
            # Keep the whole paragraph or skip it; a short definition without
            # its later correction/uncertainty would change the source claim.
            cap = max_chars
            role = 'direct_concept_definition'
            context_note = '完整原段含术语定义表述；须保留否定、范围和不确定限定，不能由病例补出通用机制，证据等级不变。'
        if event_scope:
            role = 'event_scoped_context'
            scope_names = '、'.join('《' + event_scope['requested'][event_id] + '》'
                                   for event_id in item['_event_scope_ids'])
            context_note = '限于' + scope_names + '的来源段落；不补入其他事件的发展，证据等级不变。'
        if item.get('_pair_local'):
            role = 'pair_local_context'
            cap = len(item['content'])
            context_note = pair_local_note
            if direct:
                context_note += '其他背景不能消解直接关系原述中的推测或时间限定。'
        # 统一提交点：保存出处、主体、渠道、融合分数和用途，再从原文选取符合预算的片段。
        added = add_evidence(item, owner['name'] + '｜' + item['section'], item['content'], cap,
                id=f"passage:{item['id']}",
                kind='overview' if item['category'] == 'identity' and item['ref_id'] in focused else item['kind'],
                scope=item['scope'],
                subject_id=owner['id'], subject_name=owner['name'], section=item['section'],
                passage_id=item['id'], matched_subject_ids=matched,
                retrieval_channels=item.get('retrieval_channels', ['lexical']),
                fusion_score=item.get('fusion_score'),
                evidence_role=role,
                **({'event_scope_ids': item['_event_scope_ids'],
                    'event_scope_basis': item['_event_scope_basis']} if event_scope else {}),
                **({'attribute_focus': attribute_focus} if attribute_focus else {}),
                **({'context_note': context_note} if context_note else {}))
        if added:
            used_passages.add(item['id'])
            if item.get('_organization_definition'):
                retrieval['organization_definition_source_ids'].append(item['id'])
            if item.get('_pair_local'):
                key = 'pair_support_source_ids' if direct else 'pair_local_source_ids'
                retrieval.setdefault(key, []).append(item['id'])
        if len(sources) >= max_sources:
            break

    # Only fill genuinely missing subject context, never front-load unrelated
    # biographies before evidence answering a causal or relational question.
    for entity in entities:
        if entity['kind'] not in PERSON_KINDS or pair_question or event_scope:
            continue
        if any(entity['id'] in source.get('matched_subject_ids', []) for source in sources):
            continue
        rows = query_rows(db, """SELECT p.*,s.source_type,s.path AS source_path,s.url AS source_url,s.revision
                FROM passages p LEFT JOIN sources s ON s.id=p.source_id
                WHERE p.ref_id=? AND p.category='identity' ORDER BY p.id LIMIT 1""", (entity['id'],))
        if rows:
            add_evidence(rows[0], entity['name'] + '｜简介', rows[0]['content'], 300,
                         id=f"entity:{entity['id']}", kind='overview', scope=entity['scope'],
                         subject_id=entity['id'], subject_name=entity['name'],
                         matched_subject_ids=[entity['id']], evidence_role='entity_overview')

    for relation in graph_context:
        add_relation(relation, 'graph_context')
    if pair_question and not direct:
        note = ('已检得双方局部原述，但未据此确定长期关系类型；只回答该原述明确支持的内容。'
                if retrieval.get('pair_local_source_ids') else
                '未检得双方直接关系条目。正文只能支持其明确陈述；共同提及、共同邻接和图上路径不证明两人关系。')
        if len(note) <= remaining():
            blocks.append(note)
    # 阶段五：只按最终 sources 计算覆盖与缺口；* 展开每条来源的主体集合，再用 union 求并集。
    covered = set().union(*(set(source.get('matched_subject_ids', [])) for source in sources))
    retrieval['covered_subject_ids'] = [entity_id for entity_id in focused if entity_id in covered]
    retrieval['missing_subject_ids'] = [entity_id for entity_id in focused if entity_id not in covered]
    retrieval['source_count'] = len(sources)
    retrieval['secondary_only'] = bool(sources) and all(source['source_type'] == 'llm_summary' for source in sources)
    retrieval['evidence_gaps'] = ([] if sources else ['no_source'])
    if retrieval['missing_subject_ids']:
        retrieval['evidence_gaps'].append('missing_subject_context')
    if pair_question and not direct:
        retrieval['evidence_gaps'].append('no_explicit_relationship_type' if retrieval.get('pair_local_source_ids')
                                          else 'no_direct_relation')
    if concept_search:
        # These IDs describe delivered evidence, never discarded candidates.
        definitions = [source for source in sources
                       if source.get('evidence_role') == 'direct_concept_definition'
                       and concept_definition(question, source['text'])]
        retrieval['concept_definition_source_ids'] = [source['passage_id'] for source in definitions]
        if not definitions:
            retrieval['evidence_gaps'].append('no_explicit_definition')
    # context 是可放进提示词的文字；sources 是同一批资料的结构化记录，供引用/诊断/下轮复用。
    # 返回后 terra_kb 还会补 resolution 等字段，NPC 再交给 PromptBuilder 和模型生成回复。
    return {'context': '\n'.join(blocks), 'intent': intent,
            'subjects': [{key: e[key] for key in ('id', 'name', 'kind', 'scope')} for e in entities],
            'sources': sources, 'lore_question': bool(lore_request or sources), 'retrieval': retrieval}
