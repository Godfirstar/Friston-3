"""Lexical passage ranking and direct, source-attributed relation queries."""
import math
import re

from friston3.knowledge.lore_text import normalize, phrase_query, tokenize, split_sentences
from friston3.knowledge.retrieval_policy import (
    PASSAGE_KINDS, PERSON_KINDS, RELATION_TOPICS, PERSONALITY_WORDS, LEADERSHIP_WORDS,
    query_rows, find_entity, query_terms, subject_names,
)


def _valid_relation_target(db, relation):
    """Imported narrative labels can be mislinked; do not expand such edges."""
    label = normalize(relation.get('target_label', ''))
    if not label:
        return False
    target = find_entity(db, relation['dst'])
    aliases = subject_names(db, [relation['dst']]).get(relation['dst'], [])
    labels = [normalize(part) for part in re.split(r'[、,，]', relation.get('target_label', ''))]
    if len(labels) > 1 and any(normalize(name) in labels for name in aliases):
        return True
    if (target and target['kind'] in PERSON_KINDS
            and normalize(target['name']) != label
            and re.search(r'团体|组织|群体|族群|成员们|一族|其首领', relation['description'])
            and normalize(target['name']) not in normalize(relation['description'])):
        return False
    return any(normalize(name) == label for name in aliases)


def relation_body(relation):
    """Remove only the importer's synthetic edge label before excerpt ranking."""
    description = relation['description'].strip()
    if '→' in description:
        _, separator, body = description.partition('：')
        if separator:
            return body.strip()
    return description


def relation_text(db, relation):
    """Expand only an unambiguous possessive role into a subject predicate."""
    body = relation_body(relation)
    original = relation['target_label'] + '：' + body
    if relation.get('target_is_group'):
        return original
    first = re.split(r'[，。；！？!?]', body, maxsplit=1)[0]
    owners = subject_names(db, [relation['src']]).get(relation['src'], [])
    for owner in owners:
        possessive = re.match(re.escape(owner) + r'(?:自小|自幼|儿时|童年时|昔日|现在)?的', first)
        if not possessive:
            continue
        role = first[possessive.end():]
        if (len(role) <= 30 and re.search(
                r'(?:挚友|好友|朋友|战友|同伴|旅伴|搭档|同事|护卫|保镖|雇主|监护人|'
                r'导师|老师|学生|师父|徒弟|上司|下属|助手|监管对象|监管人|监视者|'
                r'父亲|母亲|兄长|兄弟|姐姐|妹妹|弟弟|亲人|家人)$', role)
                and not re.search(r'是|为|曾|会|将|在|对|向|把|被|让|能|要|不|没|负责|担任|成为', role)):
            return relation['target_label'] + '是' + body
    return original


def _relation_describes_link(db, relation):
    """Exclude bare target labels; narrative edges retain their source limits."""
    raw = relation_body(relation)
    body = normalize(raw)
    if body in ('同上', '见上'):
        return False
    if len(raw) > 60 or len(split_sentences(raw)) > 2:
        return True
    labels = subject_names(db, [relation['src']]).get(relation['src'], [])
    if any(normalize(name) in body for name in labels):
        return True
    # A narrative, pronoun or relative role is left to its original source.
    # Only a short nominal job/group label can be rejected here.
    if re.search(r'在|于|为|向|对|与|跟|被|曾|其|他|她|们|有|是|的|'
                 r'好友|朋友|队友|同事|兄|姐|弟|妹|父|母|师|同伴', body):
        return True
    return not re.fullmatch(
        r'[^。！？!?]{1,35}(?:干员|领袖|猎人|成员|主任|医生|负责人|元凶之一)'
        r'[。！？!?]?(?:已死亡[。！？!?]?)?(?:同上[。！？!?]?)?', raw)


def _local_facet_score(text, names, facets):
    """A title belongs near the requested subject, not anywhere on its page."""
    best = 0
    for sentence in re.split(r'[。！？!?；;]', text):
        for name in names:
            for occurrence in re.finditer(re.escape(name), sentence):
                neighborhood = sentence[max(0, occurrence.start() - 18):occurrence.end() + 28]
                best = max(best, sum(facet in neighborhood for facet in facets))
    return best


def event_causal_score(text, names, facets, *, owned=False):
    """Prefer a local cause involving the asked actor over an event's epilogue."""
    heading = re.compile(r'^[*\-\s]*([\w\u3400-\u9fff·・（）() /\-]{1,48})\s*[:：]')
    causal = re.compile(r'因为|由于|导致|致使|使得|为了|源于|缘于|出于|驱使|'
                        r'原因[^。！？]{0,20}(?:是|在于|一样)|'
                        r'因[^。！？]{1,45}?(?:而|后|才)|为[^。！？]{1,45}?而')
    best, actor_heading, previous_actor = 0, owned, False
    for sentence in split_sentences(text):
        match = heading.match(sentence)
        if match:
            actor_heading = any(name in match.group(1) for name in names)
        actor = actor_heading or any(name in sentence for name in names)
        actor = actor or (previous_actor and bool(re.match(
            r'^(?:但|然而|不过|随后|后来|因此|所以)?[他她其]', sentence)))
        previous_actor = actor
        if not actor:
            continue
        causes = len(causal.findall(sentence))
        if causes:
            answer = bool(re.search(r'(?:原因|目的)[^。！？]{0,25}(?:是|在于|为了)|'
                                    r'(?:强调|解释|回答)[^。！？]{0,30}为了', sentence))
            best = max(best, 18 * min(2, causes) + 40 * answer
                       + 28 * sum(term in sentence for term in facets))
    return min(90, best)


def related_sources(db, ids, intent='relationship', question='', limit=20):
    if not ids:
        return []
    # The importer gives some unprofiled organizations several source-local
    # reference nodes. Treat equal labels as search topics, without inventing
    # relations between those nodes.
    expanded = list(ids)
    for entity_id in ids:
        entity = find_entity(db, entity_id)
        if entity and entity['kind'] == 'reference':
            expanded.extend(row['id'] for row in query_rows(db,
                "SELECT id FROM entities WHERE kind='reference' AND name=?", (entity['name'],)))
    expanded = list(dict.fromkeys(expanded))
    slots = ','.join('?' for _ in expanded)
    # Two named people ask about their own edge, never arbitrary neighbours.
    operator = 'AND' if intent == 'relationship' and len(ids) >= 2 else 'OR'
    rows = query_rows(db, f'''SELECT r.*,a.name AS src_name,b.name AS dst_name,
        s.source_type,s.path AS source_path,s.url AS source_url,s.revision,
        p.line_start,p.line_end
        FROM relations r JOIN entities a ON a.id=r.src JOIN entities b ON b.id=r.dst
        LEFT JOIN sources s ON s.id=r.source_id LEFT JOIN passages p ON p.id=r.passage_id
        WHERE r.src IN ({slots}) {operator} r.dst IN ({slots})''', (*expanded, *expanded))
    rows = [row for row in rows if _valid_relation_target(db, row)
            and _relation_describes_link(db, row)]
    terms = query_terms(question, [e['name'] for i in ids if (e := find_entity(db, i))])
    topics = [word for word in RELATION_TOPICS if word in question]
    if intent == 'leadership':
        names = [e['name'] for i in ids if (e := find_entity(db, i))]
        rows = [row for row in rows if _local_facet_score(row['description'], names, LEADERSHIP_WORDS)]
    for row in rows:
        row['target_is_group'] = bool(re.search(r'[、,，]', row.get('target_label', '')))
        text = row['description']
        score = 45 * (row['src'] in ids) + 55 * (row['src'] in ids and row['dst'] in ids)
        score += 15 * sum(word in text for word in terms)
        score += 12 * sum(word in text for word in topics)
        score += 8 if row['status'] == 'project' else 0
        score -= 18 if row['status'] == 'uncertain' else 0
        if intent == 'personality':
            score += 4 * sum(word in text for word in PERSONALITY_WORDS)
        if intent == 'leadership':
            score += 20 * sum(word in text for word in LEADERSHIP_WORDS)
        row['score'] = score
    return sorted(rows, key=lambda row: (-row['score'], row['line_start'] or 0,
                                        row['passage_id'], row['id']))[:max(1, int(limit))]


def ranked_search(db, request, *, limit=3, kinds=None):
    """Rank owner sections, direct relationships and explicit event mentions.

    Event mentions retrieve narrative context; they never establish participation
    or a personal relationship. An explicit empty ids list disables name inference.
    """
    limit = max(1, min(30, int(limit)))
    allow = tuple(dict.fromkeys(PASSAGE_KINDS if kinds is None else kinds))
    if not allow:
        return []
    query, intent = request.question, request.intent
    entities, ids, names = request.entities, request.ids, request.names
    categories, event_scope = request.categories, request.event_scope
    terms = query_terms(query, names)
    meaningful = set(tokenize(' '.join(terms)).split())
    slots = ','.join('?' for _ in allow)
    select = '''SELECT p.*,s.source_type,s.path AS source_path,s.url AS source_url,s.revision
                FROM passages p LEFT JOIN sources s ON s.id=p.source_id'''
    candidates = {}

    def remember(rows):
        candidates.update((row['id'], row) for row in rows)

    if intent == 'relationship' and len(ids) >= 2:
        direct = related_sources(db, ids, intent, query, limit=30)
        for relation in direct:
            remember(query_rows(db, select + f' WHERE p.id=? AND p.kind IN ({slots})',
                           (relation['passage_id'], *allow)))
    for entity_id in ids:
        remember(query_rows(db, select + f' WHERE p.ref_id=? AND p.kind IN ({slots}) ORDER BY p.id LIMIT 200',
                       (entity_id, *allow)))
        if intent in ('experience', 'relationship', 'general', 'leadership'):
            remember(query_rows(db, select + f''' WHERE p.id IN (
                    SELECT passage_id FROM event_mentions WHERE entity_id=? AND basis='explicit_name')
                    AND p.kind IN ({slots}) ORDER BY p.id LIMIT 80''', (entity_id, *allow)))
    # Separate subject and facet searches: an entire colloquial question is
    # rarely a literal FTS phrase. Pair conjunctions retrieve shared narrative.
    name_clauses = [phrase_query(name) for name in names if phrase_query(name)]
    clauses = [phrase_query(term) for term in terms if phrase_query(term)]
    expressions = ([' AND '.join(f'({term})' for term in name_clauses[:3])]
                   if len(name_clauses) >= 2 else []) + name_clauses + clauses[:6]
    if not names and len(clauses) >= 2:
        expressions.insert(0, ' AND '.join(f'({term})' for term in clauses[:3]))
    if intent == 'leadership' and name_clauses:
        title_expression = ' OR '.join(f'({phrase_query(word)})' for word in LEADERSHIP_WORDS if phrase_query(word))
        expressions = [f'({name}) AND ({title_expression})' for name in name_clauses] + expressions
    if not names and not expressions:
        expressions = [phrase_query(query)]
    for expression in dict.fromkeys(expression for expression in expressions if expression):
        remember(query_rows(db, f'''SELECT p.*,s.source_type,s.path AS source_path,s.url AS source_url,
                    s.revision,bm25(passage_fts) AS fts_score
                FROM passage_fts JOIN passages p ON p.id=passage_fts.rowid
                LEFT JOIN sources s ON s.id=p.source_id
                WHERE passage_fts MATCH ? AND p.kind IN ({slots}) ORDER BY fts_score LIMIT 60''',
                       (expression, *allow)))
    scored = []
    for item in candidates.values():
        item = request.prepare_candidate(item)
        if item is None:
            continue
        content = item['content']
        owner = item['ref_id'] in ids
        mentioned = set(item['matched_subject_ids'])
        if event_scope and not (mentioned & request.event_people):
            continue
        category = item['category']
        if intent != 'relationship' and category == 'relationship':
            continue  # Use typed edges so uncertainty cannot be lost in a passage.
        score = 30 * owner + 18 * len(mentioned)
        if event_scope and category in ('event_summary', 'background', 'event'):
            score += 40
        if re.search(r'为什么|为何|原因|起因|怎么会|如何会', query) and (
                event_scope or owner and any(entity['id'] == item['ref_id']
                                            and entity['kind'] in PERSON_KINDS for entity in entities)
                or item['kind'] == 'event' and mentioned):
            score += event_causal_score(
                content, [entity['name'] for entity in entities if entity['kind'] in PERSON_KINDS],
                terms, owned=owner and not event_scope)
        score += 6 * sum(name in content for name in names)
        if category in categories:
            score += 36 - 8 * categories.index(category)
        if categories[0] == 'ability' and owner and category == 'ability':
            score += 60
        if intent == 'introduction' and categories[0] == 'identity' and owner and category == 'identity':
            score += 80
        if meaningful:
            tokens = set(tokenize(content + ' ' + item['section']).split())
            score += 28 * len(meaningful & tokens) / len(meaningful)
            score += 12 * sum(word in content for word in terms)
        score += min(5, math.log1p(abs(float(item.pop('fts_score', 0) or 0))))
        if intent == 'personality':
            score += min(22, 3 * sum(word in content for word in PERSONALITY_WORDS))
        if intent == 'leadership':
            score += min(30, 10 * sum(word in content for word in LEADERSHIP_WORDS))
            score += 90 * min(2, _local_facet_score(content, names, LEADERSHIP_WORDS))
        if not ids and meaningful:
            score += 70 * sum(word in content for word in terms)
        item['score'] = round(score, 3)
        scored.append(item)
    scored.sort(key=lambda row: (-row['score'], row['id']))
    if len(ids) >= 2 and intent != 'relationship':
        reserved = []
        for entity_id in ids:
            own = next((row for row in scored if row['ref_id'] == entity_id), None)
            if own is not None:
                reserved.append(own)
        reserved_ids = {row['id'] for row in reserved}
        scored = reserved + [row for row in scored if row['id'] not in reserved_ids]
    output, seen = [], set()
    for item in scored:
        key = normalize(item['content'])
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
        if len(output) >= limit:
            break
    return output


def get_card(db, entity_id, max_sections=3):
    entity = find_entity(db, entity_id)
    if not entity:
        return None
    card = dict(entity)
    card['sections'] = {}
    for row in query_rows(db, '''SELECT p.*,s.source_type,s.path AS source_path,s.url AS source_url,s.revision
            FROM passages p LEFT JOIN sources s ON s.id=p.source_id WHERE p.ref_id=? ORDER BY p.id''', (entity_id,)):
        group = card['sections'].setdefault(row['category'], [])
        if len(group) < max(1, int(max_sections)):
            group.append(row)
    card['relations'] = related_sources(db, [entity_id], limit=1000)
    card['relationship_candidates'] = [row for row in card['relations'] if row['status'] == 'uncertain']
    card['event_mentions'] = query_rows(db, '''SELECT DISTINCT em.event_id,e.name,em.passage_id,em.basis
        FROM event_mentions em JOIN entities e ON e.id=em.event_id
        WHERE em.entity_id=? ORDER BY em.event_id,em.passage_id LIMIT 30''', (entity_id,))
    return card
