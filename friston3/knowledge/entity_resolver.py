"""Name matching and short conversational reference resolution."""
from functools import lru_cache
from pathlib import Path
import re
import sqlite3
from friston3.knowledge.lore_text import normalize
from friston3.npc.identity import IDENTITY

def _all_aliases(db):
    rows = [dict(row) for row in db.execute('''SELECT a.alias_norm,a.alias,a.entity_id,a.priority,
      a.origin,e.name,e.kind,e.source_key
      FROM aliases a JOIN entities e ON a.entity_id=e.id
      ORDER BY a.priority DESC,a.entity_id''').fetchall()]
    # Canonical entity names must resolve even when an older database omitted
    # the corresponding alias row. Use entity/name pairs: another entity's
    # alias with the same spelling must not hide this canonical candidate.
    existing = {(row['entity_id'], row['alias_norm']) for row in rows}
    people = db.execute("""SELECT id,name,kind,source_key FROM entities
        WHERE kind IN ('character','machine','fan_machine','dialogue_speaker')
        ORDER BY id""")
    for entity in people:
        word = normalize(entity['name'])
        key = (entity['id'], word)
        if not word or key in existing:
            continue
        existing.add(key)
        rows.append({'alias_norm': word, 'alias': entity['name'], 'entity_id': entity['id'],
                     'priority': 100, 'origin': 'canonical_name', 'name': entity['name'],
                     'kind': entity['kind'], 'source_key': entity['source_key']})
    # Reference nodes are imported from relation labels, not from a small
    # curated faction list. They have no independent biography. Repeated labels
    # form one lookup topic; retrieval still checks every original source edge.
    canonical = {row['alias_norm'] for row in rows}
    references = db.execute("SELECT id,name,kind FROM entities WHERE kind='reference' ORDER BY id")
    for entity in references:
        word = normalize(entity['name'])
        if (len(word) < 2 or len(word) > 24 or word in canonical
                or generic_speaker_label(word)
                or re.search(r'[（(；;]', entity['name'])):
            continue
        canonical.add(word)
        rows.append({'alias_norm': word, 'alias': entity['name'], 'entity_id': entity['id'],
                     'priority': 10, 'origin': 'reference_label',
                     'name': entity['name'], 'kind': entity['kind']})
    # A Chinese given-name prefix is useful for names such as 克丽斯腾·莱特.
    # Keep every candidate: shared prefixes must be diagnosed, never guessed.
    for row in tuple(rows):
        if row['origin'] != 'canonical_name' or row['kind'] != 'character':
            continue
        prefix = re.split(r'[·•]', row['name'], maxsplit=1)[0]
        if prefix == row['name'] or not re.fullmatch(r'[\u3400-\u9fff]{2,}', prefix):
            continue
        rows.append({**row, 'alias': prefix, 'alias_norm': normalize(prefix),
                     'priority': 35, 'origin': 'name_prefix'})
    return rows


def _group_aliases(db):
    groups = {}
    for row in _all_aliases(db):
        groups.setdefault(row['alias_norm'], []).append(row)
    return tuple(sorted(groups.items(), key=lambda item: len(item[0]), reverse=True))


@lru_cache(maxsize=2)
def _file_alias_groups(path, mtime_ns, size):
    """Reuse the immutable lore snapshot across short-lived read connections."""
    with sqlite3.connect(Path(path).as_uri() + '?mode=ro', uri=True) as snapshot:
        snapshot.row_factory = sqlite3.Row
        return _group_aliases(snapshot)


def _alias_groups(db):
    path = db.execute('PRAGMA database_list').fetchone()[2]
    if path and not db.in_transaction and not db.total_changes:
        try:
            stat = Path(path).stat()
            return _file_alias_groups(path, stat.st_mtime_ns, stat.st_size)
        except OSError:
            pass
    # In-memory or actively edited databases must observe their own snapshot.
    return _group_aliases(db)


_GENERIC_SPEAKER_ROLES = (
    '领袖', '领导', '老板', '负责人', '主任', '主管', '总工', '工程师', '研究员',
    '科学家', '医生', '护士', '病人', '患者', '干员', '队长', '队员', '士兵',
    '卫兵', '警察', '警官', '商人', '店主', '店员', '工作人员', '助手', '助理',
    '秘书', '实习生', '老师', '学生', '村民', '居民', '路人', '男人', '女人',
    '老人', '孩子', '男孩', '女孩', '青年', '少女', '少年', '父亲', '母亲',
    '哥哥', '姐姐', '弟弟', '妹妹', '客人', '观众', '记者', '市民', '感染者',
    '士官', '军官', '将军', '贵族', '代表', '仆人', '佣兵', '信使', '骑士', '教官',
)


def _explicit_speaker_reference(question, alias):
    """An occupational title is a speaker only when queried as a name/title."""
    q, word = normalize(question), normalize(alias)
    if q == word:
        return True
    escaped = re.escape(word)
    if re.search(r'[“「『"\']\s*' + re.escape(alias) + r'\s*[”」』"\']', question, re.I):
        return True
    if re.search(r'(?:叫|名叫|称作|说话人|角色|人物)' + escaped, q):
        return True
    return bool(re.fullmatch(
        r'(?:请)?(?:(?:介绍|说说|聊聊|认识|了解|查询|查查)(?:一下)?)?' + escaped +
        r'(?:是谁|是什么人|是什么角色|的档案|这个角色|这个人物|吗)?', q))


def generic_speaker_label(name):
    name = normalize(name)
    roles = '|'.join(re.escape(role) for role in _GENERIC_SPEAKER_ROLES)
    return bool(re.fullmatch(
        r'(?:(?:年轻|年老|陌生|神秘|疲惫|受伤|普通|男性|女性|年幼|中年|资深|高级|当地)的?)?'
        r'(?:' + roles + r')(?:[甲乙丙丁一二三四五0-9a-z]+)?', name))


def _self_relation_query(question):
    """Recognize either direction, without treating '你和我聊聊X' as a pair."""
    q = normalize(question)
    tail = (r'(?:到底|究竟)?(?:是|有|算是)?(?:什么关系|何关系|啥关系|怎样的?关系|何种关系|'
            r'关系如何|关系怎样|关系好吗|有关系吗|认识吗|见过吗|见过面吗|熟吗|是朋友吗|朋友吗|同事吗|搭档吗)')
    for pattern in (r'你(?:和|与|跟)(.{1,45}?)' + tail,
                    r'(.{1,45}?)(?:和|与|跟)你' + tail):
        match = re.search(pattern, q)
        if match and not any(word in match.group(1) for word in ('聊聊', '说说', '介绍', '讲讲', '谈谈', '告诉')):
            return True
    return False


def asks_personal_acquaintance(question):
    """Knowing about a person is distinct from claiming a personal encounter."""
    q = normalize(question)
    return (_self_relation_query(question)
            or bool(re.search(r'你(?:以前|曾经|曾|真的|确实|亲自|私下|本人|当面|在现实中)*(?:见过|遇见过|遇到过|结识过)', q))
            or bool(re.search(r'你(?:以前|曾经|真的|确实|亲自|私下|本人|当面)(?:认识|认不认识)', q))
            or bool(re.search(r'你(?:的)?(?:好?朋友|同伴)|你(?:都|还)?(?:认识|熟悉)(?:谁|哪些人|什么人)', q)))


def _has_person_reference(question):
    """Require pronoun grammar rather than a character inside an unknown name."""
    q = normalize(question)
    if any(word in q for word in ('那位', '那个人')) or re.search(r'那(?:几|两|些)(?:位|个人)', q):
        return True
    # '其他人' is a new/unspecified topic; '他拉' could be an unknown name.
    # A standalone ending or a predicate/particle makes the reference explicit.
    if re.search(
        r'(?<!其)[他她它](?:们(?:俩|两个)?|俩)?(?=$|的|和|与|跟|呢|吗|吧|呀|啊|是|有|会|能|'
        r'曾经|还|都|又|也|就|却|从|在|为什么|怎么|怎样|什么|说|做|去|见|'
        r'喜欢|讨厌|认识|认为|到|想|后来|以前|过去|最|究竟|到底|值得|性格|经历|情况|本人|'
        r'面对|遇到|碰到|参与|负责|担任|帮助|支持|保护|承担|采取|遭遇|应对|处理|具体|当时)', q):
        return True
    # A conversational subject at the start of a follow-up does not depend
    # on a vocabulary of every possible verb. Explicit entity names still
    # win in the caller; multiple antecedents still require clarification.
    initial = re.match(
        r'^(?:(?:那(?:么)?|所以|不过|对了|话说|至于|还有|接着|然后))*'
        r'[他她它](?:们(?:俩|两个)?|俩)?(?!人|者|乡|国|处|日|年|事|物)', q)
    name_question = re.fullmatch(
        r'(?:那(?:么)?)?[他她它](?!们|俩)[\u3400-\u9fff]{1,20}'
        r'(?:是谁|是什么人|是什么角色|叫什么|的档案)', q)
    return bool(initial and not name_question)


def ordinary_alias_span(text, alias, start, end):
    """Reject a common-noun occurrence without hiding other name occurrences."""
    if alias not in ('真理', '解真'):
        return False
    # Explicit character-reference wording wins over an ambiguous verb such as
    # "探索" (e.g. "探索真理这个角色的经历").
    if re.match(r'(?:这个|这位|这名)?(?:干员|角色|人物)|是谁|是什么人', text[end:]):
        return False
    if alias == '解真':
        # "了解/理解" + "真实/真相" crosses two ordinary words. Check this
        # occurrence only: a later explicit mention of 解真 must still resolve.
        return (start > 0 and text[start - 1] in ('了', '理')
                and text[end:end + 1] in ('实', '相'))
    return bool(re.search(r'(?:追求|追寻|探寻|探索|寻求|揭示|宇宙|科学|唯一|普遍|客观|绝对|终极)(?:的|中的)?$',
                          text[:start]))


def embedded_group_name(text, end):
    """A person alias inside a race/company name is not a named person.

    Check each occurrence independently: a later explicit mention of the
    operator must still resolve. Possessive wording and verbal '组织了' are
    not group-name suffixes; '族长' remains a personal title.
    """
    suffix = text[end:]
    if re.match(r'族(?!长)|家族|一族', suffix):
        return True
    return bool(re.match(
        r'(?:有限公司|公司|集团|组织|协会|联合会|商会|学院|研究所|研究院)'
        r'(?=$|的|是|有|在|和|与|跟|为什么|怎么|怎样|如何|哪|什么|吗|呢)', suffix))


def _embedded_job_title(text, alias, end):
    """Do not promote part of a longer occupational title to a person.

    These are language-level title forms, not names of lore characters.
    A standalone title still follows the normal alias/ambiguity path.
    """
    titles = {*_GENERIC_SPEAKER_ROLES, '首席', '副手', '副主任', '教授', '博士', '院长', '院士'}
    if alias not in titles:
        return False
    prefix = text[:end - len(alias)]
    suffix = text[end:]
    if re.search(r'(?:担任|出任|就任|任命为|受聘为|成为)(?:了|过|一名|一位|一个)?$', prefix):
        return True
    if (re.search(r'(?:当|做)(?:了|过|一名|一位|一个)?$', prefix)
            and re.match(r'$|呢|吗|吧|的(?:原因|理由|经历)|[了过]', suffix)):
        return True
    if alias == '博士' and re.match(r'后|生', suffix):
        return True
    compound = re.match(r'[\u3400-\u9fff]{1,6}(?:官|师|家|员|长)|顾问|助理|教授|助教', suffix)
    if not compound:
        return False
    # Function words/predicates introduce a new clause, not a title modifier:
    # e.g. 主任和园丁, 博士是科学家, 首席喜欢艺术家.
    return not re.search(
        r'[的和与跟是在把被让说叫问想看会能要不没也又都很更最比像呢吗吧谁哪怎如]|'
        r'喜欢|讨厌|认为|觉得|帮助|性格|经历|过去|后来|曾经|负责', compound.group())


def locative_event_title(alias, source_key=''):
    """Very short record titles need an explicit story label, not bare 'in'."""
    return len(normalize(alias)) >= 3 or bool(re.match(r'^(?:act\d|main_)', source_key or ''))


def _explicit_event_reference(question, alias, source_key=''):
    """Recognize event syntax while leaving ordinary abstract nouns alone."""
    q, word = normalize(question), normalize(alias)
    escaped = re.escape(word)
    if (re.search(r'[《“「]' + re.escape(alias) + r'[》”」]', question)
            or re.search(r'(?:活动|剧情|密录|故事|篇章)' + escaped, q)):
        return True
    if not locative_event_title(alias, source_key):
        return False
    if _explicit_speaker_reference(question, alias):
        return True
    # Both "A在故事名中..." and "故事名里A..." are natural unquoted
    # event references. No character or event-name allowlist is involved.
    return bool(re.search(escaped + r'(?:的)?(?:剧情|故事|活动|密录|篇章)(?:中|里|期间)', q)
                or re.search(r'(?:在|于)' + escaped + r'(?:中|里|期间)', q)
                or re.search(r'^(?:(?:请问|请讲讲|讲讲|说说|聊聊)(?:一下)?)?' + escaped +
                             r'(?:中|里|期间)', q))


def resolve_entity_resolution(db, question, previous_entities=()):
    """Return explicit candidates and ambiguity, plus bounded pronoun resolution."""
    q = normalize(question)
    candidates = {}
    mention_positions = {}
    mentions, ambiguities = [], []
    matched_spans = []
    for word, rows in _alias_groups(db):
        if word not in q:
            continue
        spans = [(m.start(), m.end()) for m in re.finditer(re.escape(word), q)]
        spans = [(a,b) for a,b in spans if not any(x <= a and b <= y for x,y in matched_spans)]
        spans = [(a,b) for a,b in spans if not ordinary_alias_span(q, word, a, b)]
        if not spans:
            continue
        priority = max(row['priority'] for row in rows)
        best = list({row['entity_id']: row for row in rows if row['priority'] == priority}.values())
        row = best[0]
        if all(item['kind'] in ('character', 'machine', 'fan_machine', 'dialogue_speaker') for item in best):
            spans = [(a, b) for a, b in spans if not embedded_group_name(q, b)
                     and not _embedded_job_title(q, word, b)]
            if not spans:
                continue
        if row['kind'] == 'event' and len(word) <= 4:
            if not _explicit_event_reference(question, row['alias'], row.get('source_key', '')):
                continue
        if (row['kind'] == 'dialogue_speaker' and generic_speaker_label(row['name'])
                and not _explicit_speaker_reference(question, row['alias'])):
            continue
        if len(word)==1 and '\u3400' <= word <= '\u9fff':
            # One-character operators (e.g. 陈, 令) match only in explicit
            # name-reference grammar, never by incidental substring.
            w=re.escape(word)
            inquiry = r'说说|谁是|介绍|认识|关于|聊聊|了解|知道|知不知道|怎么看|觉得|认为|评价'
            tail = (r'(?:是|的|和|与|跟|呢|吗|谁|什么|为何|为什么|怎么样|怎样|如何|'
                    r'平时|平常|通常|一般|现在|以前|过去|曾经|最初|之间|会|能|有|在|擅长|负责|'
                    r'养|种植|学习|练习|尝试|参加|参与|前往|离开|休息|$)')
            if not (re.search(r'^(?:你)?(?:' + inquiry + r')?(?:一下)?' + w + tail, q)
                    or re.search(r'(?:' + inquiry + r'|叫|和|与|跟|对|向|给|替|帮助?|让|被|由|'
                                 r'建议|劝(?:说)?|鼓励|要求|邀请|指导|引导)'
                                 r'(?:一下)?' + w + tail, q)):
                continue
        # Avoid substring false positives from short latin alias names.
        if re.fullmatch(r'[a-z0-9_]+', word) and len(word) <= 3:
            if not re.search(r'(?<![a-z0-9_])' + re.escape(word) + r'(?![a-z0-9_])', q):
                continue
        matched_spans.extend(spans)
        if len(best) != 1:
            ambiguities.append({'alias': row['alias'], 'candidates': [
                {'id': item['entity_id'], 'name': item['name'], 'alias': item['alias']}
                for item in best]})
            continue
        score = len(word) * 100 + row['priority']
        mention_positions[row['entity_id']] = min(
            mention_positions.get(row['entity_id'], len(q)), min(start for start, _ in spans))
        existing = candidates.get(row['entity_id'], -1)
        if score > existing:
            candidates[row['entity_id']] = score
            mentions.append({'id': row['entity_id'], 'name': row['name'],
                             'alias': row['alias'], 'origin': row['origin']})
    # Preserve the user's subject order so budgets do not always favor the
    # person with the longest name.
    ids = sorted(candidates, key=lambda key: (mention_positions.get(key, len(q)), -candidates[key]))
    # A fuller name in this same question can settle its accompanying title.
    # Previous discussion alone cannot decide a title shared across timelines.
    unresolved = []
    for ambiguity in ambiguities:
        named = [item for item in ambiguity['candidates'] if item['id'] in ids]
        if len(named) != 1:
            unresolved.append(ambiguity)
    ambiguities = unresolved
    self_patterns = ('你是谁', '你的研发', '你的制造', '你的主人',
                     '谁制造了你', '谁设计了你', '你谁造', '谁造了你', '你的主管', '你的老板',
                     '谁制造你', '谁设计你', '谁研发你', '谁研发了你',
                     '介绍一下自己', '介绍自己', '自我介绍', '你最喜欢', '你去过',
                     '你经历过', '你的回忆', '你的维护')
    self_relation = asks_personal_acquaintance(question)
    self_social = bool(re.search(
        r'你的(?:过去|往事|经历|亲人|家人)|你(?:见过|去过|经历过)|你有什么经历', question))
    if any(s in question for s in self_patterns) or self_relation or self_social:
        ids = [IDENTITY.entity_id] + [i for i in ids if i != IDENTITY.entity_id]
    person_reference = _has_person_reference(question)
    bare_followup = q in ('为什么', '为什么呢', '为什么会这样', '为什么这么觉得', '为什么这样说',
                         '为什么这样觉得', '为什么这么说', '怎么说', '怎么说呢',
                         '具体说说', '具体说说呢', '具体说说吧', '后来呢', '后来怎么样',
                         '继续', '继续说', '接着', '接着说', '然后呢')
    followup = person_reference or bare_followup or any(s in question for s in ('刚才', '继续', '接着', '然后呢'))
    # Explicit names replace the topic unless a pronoun clearly denotes a
    # second participant ("她和博士"). Avoid dragging an unrelated old name
    # into a fresh question such as "可露希尔她是什么人".
    needs_antecedent = not ids or bool(re.search(
        r'[他她它](?:们)?(?:和|与|跟)|(?:和|与|跟)[他她它](?:们)?', question))
    needs_antecedent = needs_antecedent or (self_relation and person_reference)
    used_previous = False
    if followup and needs_antecedent and not ambiguities:
        previous = []
        for entity_id in dict.fromkeys(previous_entities):
            if entity_id in ids:
                continue
            row = db.execute('SELECT id,name FROM entities WHERE id=?', (entity_id,)).fetchone()
            if row:
                previous.append(dict(row))
        plural = bool(re.search(r'(?<!其)[他她它](?:们|俩)|那(?:几|两|些)位|那(?:几|两|些)个人', question))
        # Bare explanations retain the task's whole subject set. Plural
        # references retain all third-person antecedents, not an arbitrary
        # first member. The current speaker is never a third-person guess.
        if bare_followup and not person_reference:
            selected = previous
        elif plural:
            selected = [item for item in previous if item['id'] != IDENTITY.entity_id]
        elif person_reference:
            choices = [item for item in previous if item['id'] != IDENTITY.entity_id]
            selected = choices if len(choices) == 1 else []
            if len(choices) > 1:
                reference = re.search(r'那个人|那位|(?<!其)[他她它](?!们)', question)
                # Keep the existing ambiguity shape: a later explicit name
                # selection can replace this exact pronoun in the original
                # question through pending_reference.
                ambiguities.append({'alias': reference.group() if reference else question,
                                    'origin': 'conversation_reference',
                                    'candidates': [{'id': item['id'], 'name': item['name'],
                                                    'alias': item['name']} for item in choices]})
        else:
            selected = previous
        for item in selected:
            ids.append(item['id'])
            used_previous = True
    ids = ids[:8]
    return {'status': 'ambiguous' if ambiguities else 'resolved' if ids else 'unknown',
            'entity_ids': ids, 'mentions': mentions, 'ambiguities': ambiguities,
            'used_previous': used_previous,
            'unresolved_reference': bool(person_reference and not ids and not ambiguities)}


def resolve_entities(db, question, previous_entities=()):
    """Compatibility API; inspect resolve_entity_resolution for ambiguity."""
    return resolve_entity_resolution(db, question, previous_entities)['entity_ids']


