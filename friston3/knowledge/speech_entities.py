"""Read-only speech vocabulary shared by ASR hotwords and entity candidates.

Names come from the database, never a character correction table. Phonetic
matches are interpretations, not identities or lore evidence. Raw transcripts
and database rows stay unchanged. Scores are similarities, not probabilities.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
import re
import sqlite3
from threading import RLock
import unicodedata

from friston3.knowledge.entity_resolver import generic_speaker_label
from friston3.knowledge.lore_text import normalize


_PERSON_KINDS = frozenset(('character', 'machine', 'fan_machine', 'dialogue_speaker'))
_ALIAS_ORIGINS = frozenset(('canonical_name', 'alias_table', 'page_alias', 'name_prefix'))
_HAN = re.compile(r'^[\u3400-\u9fff]+$')
# Some ASR timestamp outputs separate Chinese characters with spaces. Keep
# those original offsets while never bridging punctuation or Latin tokens.
_HAN_RUN = re.compile(r'[\u3400-\u9fff]+(?:[·•\s]+[\u3400-\u9fff]+)*')
_LABEL = re.compile(r'^[\u3400-\u9fffA-Za-z0-9]+(?:[·•._-][\u3400-\u9fffA-Za-z0-9]+)*$')
_SYLLABLE = re.compile(r'^[a-z]+$')
_GENERIC_LABELS = frozenset((
    '我', '你', '您', '他', '她', '它', '我们', '你们', '他们', '她们', '它们',
    '自己', '某人', '众人', '其他人', '未知', '不详', '无名', '无名氏', '没有', '暂无',
))
# Category forms, not an allowlist of setting-specific factions or people.
_GROUP_FORM = re.compile(
    r'(?:有限公司|公司|集团|组织|委员会|协会|联合会|商会|军团|部队|小队|预备组|行动组|'
    r'研究所|研究院|实验室|学院|大学|骑士团|教会|政府|议会|家族|一族|工程部|医疗部)'
    r'|(?:共和国|公国|王国|帝国|工业|制药|学派|阵营|种族|城|市|镇|村|国|岛|族|局|署)$')
_PERSON_CONTEXT = re.compile(
    r'^[他她](?=是|曾|在|与|和|作为|担任|负责|拥有|来自|出生|成长|参与|帮助|支持|建立)|'
    r'^(?:一名|一位|这名|这位|该名|该位)(?!公司|组织|集团)|'
    r'^(?:担任|曾任|出任|出生于|出生在|成长于|成长在)')
_MAX_TEXT_CHARS = 512
_MAX_NAME_CHARS = 12
_MAX_LITERAL_CHARS = 100
_MAX_PRONUNCIATIONS = 32
_MAX_GROUPS = 3
_MAX_CANDIDATES = 3
_MAX_BUCKET = 64
_MAX_NEAR_COMPARISONS = 4096
_CACHE_SIZE = 2
_VOCAB_CACHE = OrderedDict()
_CACHE = OrderedDict()
_CACHE_LOCK = RLock()


class SpeechEntityDependencyError(ImportError):
    """The pronunciation matcher cannot import its required dependency."""


@lru_cache(maxsize=1)
def _pinyin():
    try:
        import pypinyin
    except ImportError as exc:
        raise SpeechEntityDependencyError(
            '[speech_entities:pypinyin_missing] 语音实体候选需要 pypinyin；'
            '请在项目 Python 环境执行 python -m pip install pypinyin==0.55.0。'
        ) from exc
    return pypinyin


@dataclass(frozen=True)
class _Name:
    entity_id: str
    name: str
    alias: str
    priority: int
    kind: str
    origin: str
    requires_confirmation: bool = False


@dataclass(frozen=True)
class _Vocabulary:
    names: tuple
    literals: dict
    literal_names: frozenset
    stats: dict


@dataclass(frozen=True)
class _Index:
    exact: dict
    near: dict
    deletions: dict
    lengths: tuple
    vocabulary: _Vocabulary


class _Rows:
    """Read sqlite.Row values without modifying the caller's connection."""

    def __init__(self, db):
        self.db = db

    def execute(self, *args, **kwargs):
        cursor = self.db.execute(*args, **kwargs)
        cursor.row_factory = sqlite3.Row
        return cursor


def _label(value):
    return unicodedata.normalize('NFKC', str(value or '')).strip()


def _name_rejection(value, *, weak=False):
    word = normalize(value)
    if (not value or len(word) > _MAX_NAME_CHARS or not _LABEL.fullmatch(value)
            or word.isdecimal()):
        return 'invalid_shape_or_length'
    if word in _GENERIC_LABELS or generic_speaker_label(word):
        return 'generic_label'
    if weak and ('的' in word or _GROUP_FORM.search(word)):
        return 'description_or_group'
    return ''


def _literal_add(trie, word):
    if not 2 <= len(word) <= _MAX_LITERAL_CHARS:
        return
    node = trie
    for char in word:
        node = node.setdefault(char, {})
    node[None] = True


def _name_order(name):
    origin_rank = {'canonical_name': 0, 'alias_table': 1, 'name_prefix': 2,
                   'page_alias': 3, 'reference_label': 4}
    return (origin_rank.get(name.origin, 5), name.requires_confirmation,
            -name.priority, len(name.alias), name.alias, name.entity_id)


def _reference_people(rows, entities):
    """Require source-owned person wording for otherwise untyped references.

    A relationship edge alone does not make its target a person. Even after
    this conservative source filter, these labels require confirmation.
    """
    result = set()
    for row in rows.execute('''SELECT r.dst,r.target_label,r.description,r.status,
            r.source_id,p.category FROM relations r JOIN passages p ON p.id=r.passage_id
            JOIN entities e ON e.id=r.dst WHERE e.kind='reference' '''):
        entity = entities.get(row['dst'])
        if (not entity or not row['source_id'] or row['status'] not in ('source_asserted', 'project')
                or row['category'] not in ('relationship', 'participants')):
            continue
        label = _label(entity['name'])
        if (_name_rejection(label, weak=True) or len(normalize(label)) < 2
                or normalize(row['target_label']) != normalize(label)):
            continue
        description = str(row['description'] or '').lstrip(' \t\r\n“「『')
        if description.startswith(label):
            description = description[len(label):].lstrip(' ，,：:')
        named_predicate = re.match(
            r'^(?:本人|女士|先生|是(?:一名|一位)|曾(?:是|任)|担任|出生于|出生在)', description)
        if _PERSON_CONTEXT.search(description) or named_predicate:
            result.add(entity['id'])
    return result


def _build_vocabulary(db):
    rows = _Rows(db)
    entities = {row['id']: dict(row) for row in rows.execute(
        'SELECT id,name,kind,source,source_key FROM entities ORDER BY id')}
    reference_people = _reference_people(rows, entities)
    excluded, names, literals, literal_names = Counter(), {}, {}, set()

    def add(entity, value, origin, priority, *, weak=False):
        alias = _label(value)
        reason = _name_rejection(alias, weak=weak)
        if reason:
            excluded[reason] += 1
            return
        word = normalize(alias)
        candidate = _Name(entity['id'], entity['name'], alias, priority,
                          entity['kind'], origin,
                          weak or len(word) == 1 or entity['kind'] == 'reference')
        key = (entity['id'], word)
        previous = names.get(key)
        if previous is None or _name_order(candidate) < _name_order(previous):
            names[key] = candidate
        if not weak and entity['kind'] in _PERSON_KINDS:
            literal_names.add(word)
            _literal_add(literals, word)

    # Old databases may have canonical entity names without any alias row.
    known_names = {normalize(entity['name']) for entity in entities.values()
                   if entity['kind'] in _PERSON_KINDS}
    for entity in entities.values():
        kind = entity['kind']
        if kind in _PERSON_KINDS:
            add(entity, entity['name'], 'canonical_name', 100,
                weak=kind == 'dialogue_speaker')
        elif kind == 'reference':
            if entity['id'] in reference_people and normalize(entity['name']) not in known_names:
                add(entity, entity['name'], 'reference_label', 10, weak=True)
            else:
                excluded['untyped_reference'] += 1
        else:
            excluded['non_person_entity'] += 1
            # Canonical labels of explicitly typed non-person records protect
            # complete names; short untyped reference fragments never do.
            word = normalize(entity['name'])
            if 3 <= len(word) <= _MAX_LITERAL_CHARS:
                literal_names.add(word)
                _literal_add(literals, word)

    for row in rows.execute('SELECT entity_id,alias,origin,priority FROM aliases '
                            'ORDER BY entity_id,alias_norm'):
        entity = entities.get(row['entity_id'])
        if not entity or entity['kind'] not in _PERSON_KINDS:
            continue
        if row['origin'] not in _ALIAS_ORIGINS:
            excluded['non_name_alias_origin'] += 1
            continue
        # Old importers kept page_alias after an alias_table collision. Clean
        # compact labels remain weaker candidates without rewriting the DB.
        add(entity, row['alias'], row['origin'], row['priority'],
            weak=row['origin'] == 'page_alias' or entity['kind'] == 'dialogue_speaker')

    for entity in entities.values():
        if entity['kind'] not in _PERSON_KINDS:
            continue
        prefix = re.split(r'[·•]', entity['name'], maxsplit=1)[0]
        if prefix != entity['name'] and len(prefix) >= 2 and _HAN.fullmatch(prefix):
            add(entity, prefix, 'name_prefix', 35,
                weak=entity['kind'] == 'dialogue_speaker')

    ordered = tuple(sorted(names.values(), key=_name_order))
    phonetic = [name for name in ordered if _HAN.fullmatch(normalize(name.alias))]
    stats = {
        'entity_count': len({name.entity_id for name in ordered}),
        'name_count': len(ordered),
        'unique_name_count': len({normalize(name.alias) for name in ordered}),
        'canonical_name_count': sum(name.origin == 'canonical_name' for name in ordered),
        'phonetic_name_count': len(phonetic),
        'single_character_name_count': sum(len(normalize(name.alias)) == 1 for name in phonetic),
        'confirmation_name_count': sum(name.requires_confirmation for name in ordered),
        'by_kind': dict(sorted(Counter(name.kind for name in ordered).items())),
        'excluded': dict(sorted(excluded.items())),
    }
    return _Vocabulary(ordered, literals, frozenset(literal_names), stats)


def _cache_key(db):
    # Close the program before replacing the database file, as required by the
    # existing importer. In-memory/edited connections observe their own rows.
    if db.in_transaction or db.total_changes:
        return None
    databases = db.execute('PRAGMA database_list').fetchall()
    path = next((row[2] for row in databases if row[1] == 'main'), '')
    if not path:
        return None
    try:
        path = Path(path).resolve()
        stat = path.stat()
        try:
            wal = Path(str(path) + '-wal').stat()
            wal_state = (wal.st_mtime_ns, wal.st_size)
        except FileNotFoundError:
            wal_state = None
        return (str(path), stat.st_mtime_ns, stat.st_size, wal_state)
    except OSError:
        return None


def _cached(db, cache, build):
    key = _cache_key(db)
    if key is not None:
        with _CACHE_LOCK:
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
    value = build(db)
    if key is not None and key == _cache_key(db):
        with _CACHE_LOCK:
            cache[key] = value
            cache.move_to_end(key)
            while len(cache) > _CACHE_SIZE:
                cache.popitem(last=False)
    return value


def _vocabulary(db):
    return _cached(db, _VOCAB_CACHE, _build_vocabulary)


def hotwords(db, *, max_chars=10):
    """All eligible ASR names, canonical first, without requiring pypinyin.

    Spaces, descriptions, source IDs and invented Chinese readings are excluded.
    Latin/digit names retain their spelling. Untyped page aliases remain text
    candidates but are excluded from acoustic biasing.
    """
    if type(max_chars) is not int or not 1 <= max_chars <= _MAX_NAME_CHARS:
        raise ValueError(f'max_chars 必须是 1..{_MAX_NAME_CHARS} 的整数')
    words, seen = [], set()
    for name in _vocabulary(db).names:
        key = normalize(name.alias)
        if name.origin == 'page_alias' or len(name.alias) > max_chars or key in seen:
            continue
        seen.add(key)
        words.append(name.alias)
    return words


def vocabulary_stats(db):
    """Counts for existing evidence records; never print or write name lists."""
    vocabulary = _vocabulary(db)
    return {**vocabulary.stats, 'by_kind': dict(vocabulary.stats['by_kind']),
            'excluded': dict(vocabulary.stats['excluded']),
            'hotword_count': len(hotwords(db))}


def _primary(text):
    engine = _pinyin()
    values = tuple(engine.lazy_pinyin(
        text, style=engine.Style.NORMAL, errors=lambda value: list(value)))
    if len(values) != len(text) or not all(_SYLLABLE.fullmatch(value) for value in values):
        return ()
    return values


@lru_cache(maxsize=4096)
def _readings(char):
    engine = _pinyin()
    values = engine.pinyin(char, style=engine.Style.NORMAL, heteronym=True,
                           errors=lambda value: list(value))
    if not values:
        return ()
    return tuple(dict.fromkeys(value for value in values[0]
                               if _SYLLABLE.fullmatch(value)))[:3]


def _pronunciations(word):
    primary = _primary(word)
    if not primary:
        return ()
    alternatives = [tuple(value for value in _readings(char) if value != primary[i])
                    for i, char in enumerate(word)]
    positions = [i for i, values in enumerate(alternatives) if values]
    result = [primary]
    for count in (1, 2):
        for changed in combinations(positions, count):
            for values in product(*(alternatives[i] for i in changed)):
                pronunciation = list(primary)
                for i, value in zip(changed, values):
                    pronunciation[i] = value
                result.append(tuple(pronunciation))
                if len(result) >= _MAX_PRONUNCIATIONS:
                    return tuple(result)
    return tuple(result)


def _masked(pronunciation):
    if len(pronunciation) < 3:
        return
    for count in range(1, 3 if len(pronunciation) >= 4 else 2):
        for positions in combinations(range(len(pronunciation)), count):
            masked = list(pronunciation)
            for position in positions:
                masked[position] = None
            yield tuple(masked)


def _bucket_add(index, key, value):
    if key not in index:
        index[key] = {value}
    elif index[key] is not None:
        index[key].add(value)
        if len(index[key]) > _MAX_BUCKET:
            index[key] = None


def _build_index(db):
    vocabulary = _vocabulary(db)
    exact = {}
    for name in vocabulary.names:
        word = normalize(name.alias)
        if not _HAN.fullmatch(word):
            continue
        for pronunciation in _pronunciations(word):
            group = exact.setdefault(pronunciation, {})
            previous = group.get(name.entity_id)
            if previous is None or _name_order(name) < _name_order(previous):
                group[name.entity_id] = name
    exact = {key: tuple(sorted(group.values(), key=_name_order))
             for key, group in exact.items()}
    near, deletions = {}, {}
    for pronunciation in exact:
        for masked in _masked(pronunciation):
            _bucket_add(near, masked, pronunciation)
        if len(pronunciation) >= 4:
            for i in range(len(pronunciation)):
                _bucket_add(deletions, pronunciation[:i] + pronunciation[i + 1:], pronunciation)
    for table in (near, deletions):
        for key, values in tuple(table.items()):
            if values is not None:
                table[key] = tuple(sorted(values))
    widths = {len(key) for key in exact}
    widths |= {length - 1 for length in tuple(widths) if length >= 4}
    widths |= {length + 1 for length in tuple(widths) if 3 <= length < _MAX_NAME_CHARS}
    return _Index(exact, near, deletions, tuple(sorted(widths, reverse=True)), vocabulary)


def _index(db):
    _pinyin()
    return _cached(db, _CACHE, _build_index)


def prepare(db):
    """Prepare the in-memory pronunciation index, without downloading files."""
    _index(db)


def _normalized_offsets(text):
    chars, positions = [], []
    for i, char in enumerate(text):
        normalized = normalize(char)
        chars.extend(normalized)
        positions.extend([i] * len(normalized))
    return ''.join(chars), positions


def _protected(index, text, resolution):
    normalized, positions = _normalized_offsets(text)
    vocabulary = index.vocabulary
    protected = []
    for start in range(len(normalized)):
        node = vocabulary.literals
        for end in range(start, min(len(normalized), start + _MAX_LITERAL_CHARS)):
            node = node.get(normalized[end])
            if node is None:
                break
            if None in node:
                protected.append((positions[start], positions[end] + 1))
    if isinstance(resolution, dict):
        items = tuple(resolution.get('mentions') or ()) + tuple(resolution.get('ambiguities') or ())
        for item in items:
            if item.get('origin') in ('conversation_reference', 'reference_label', 'page_alias'):
                continue
            word = normalize(item.get('alias', ''))
            if not word or word not in vocabulary.literal_names:
                continue
            for match in re.finditer(re.escape(word), normalized):
                protected.append((positions[match.start()], positions[match.end() - 1] + 1))
    merged = []
    for start, end in sorted(protected):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _letter_distance(left, right, limit=2):
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _matches(index, pronunciation, budget):
    exact = index.exact.get(pronunciation)
    if exact:
        return [(name, 1.0, 'homophone') for name in exact[:_MAX_BUCKET]], len(exact) > _MAX_BUCKET
    if len(pronunciation) < 3 or not budget[0]:
        return [], False
    matches, seen, truncated = {}, set(), False

    def add(alternative, cost, match_type):
        nonlocal truncated
        names = index.exact[alternative]
        truncated |= len(names) > _MAX_BUCKET
        letters = max(sum(map(len, pronunciation)), sum(map(len, alternative)))
        score = round(max(0.0, 1.0 - cost / letters), 4)
        for name in names[:_MAX_BUCKET]:
            old = matches.get(name.entity_id)
            if old is None or (-score, _name_order(name)) < (-old[1], _name_order(old[0])):
                if old is None and len(matches) >= _MAX_BUCKET:
                    truncated = True
                    continue
                matches[name.entity_id] = (name, score, match_type)

    for masked in _masked(pronunciation):
        alternatives = index.near.get(masked, ())
        if alternatives is None:
            truncated = True
            continue
        for alternative in alternatives:
            if alternative in seen:
                continue
            seen.add(alternative)
            if not budget[0]:
                return list(matches.values()), True
            budget[0] -= 1
            allowance = 2 if len(pronunciation) >= 4 else 1
            cost = sum(_letter_distance(a, b, allowance)
                       for a, b in zip(pronunciation, alternative) if a != b)
            if cost <= allowance:
                add(alternative, cost, 'near')

    # One missing/extra syllable is weaker than letter edits. All remaining
    # syllables must match exactly, and both names keep at least three sounds.
    longer = index.deletions.get(pronunciation, ())
    if longer is None:
        truncated = True
        longer = ()
    alternatives = set(longer)
    if len(pronunciation) >= 4:
        alternatives.update(shorter for i in range(len(pronunciation))
                            if (shorter := pronunciation[:i] + pronunciation[i + 1:]) in index.exact)
    for alternative in sorted(alternatives):
        if alternative in seen:
            continue
        seen.add(alternative)
        if not budget[0]:
            return list(matches.values()), True
        budget[0] -= 1
        add(alternative, 3, 'syllable_edit')
    return list(matches.values()), truncated


def candidates(db, text, *, resolution=None, limit=3):
    """Return at most three original spans and three entity alternatives each.

    One/two-character names use homophones only; one-character names require
    confirmation. Longer names allow one letter edit (three syllables), two
    total letter edits (four or more), or one whole-syllable insertion/deletion
    with all remaining sounds exact. Whole-syllable edits require confirmation.
    Omitted alternatives mark the group truncated rather than apparently unique.
    """
    limit = max(0, min(int(limit), _MAX_GROUPS))
    if not isinstance(text, str) or not text or not limit:
        return []
    index = _index(db)
    protected = _protected(index, text[:_MAX_TEXT_CHARS + _MAX_LITERAL_CHARS], resolution)
    groups, budget = [], [_MAX_NEAR_COMPARISONS]
    for run in _HAN_RUN.finditer(text[:_MAX_TEXT_CHARS]):
        positions = [run.start() + i for i, char in enumerate(run.group())
                     if '\u3400' <= char <= '\u9fff']
        word = ''.join(text[i] for i in positions)
        pronunciation = _primary(word)
        if not pronunciation:
            continue
        for width in index.lengths:
            for start in range(len(pronunciation) - width + 1):
                a, b = positions[start], positions[start + width - 1] + 1
                if any(x < b and a < y for x, y in protected):
                    continue
                found, truncated = _matches(index, pronunciation[start:start + width], budget)
                if not found:
                    continue
                found.sort(key=lambda item: (item[2] == 'syllable_edit', -item[1], _name_order(item[0])))
                options = [
                    {'id': name.entity_id, 'name': name.name, 'alias': name.alias, 'score': score,
                     'kind': name.kind, 'origin': name.origin, 'match_type': match_type,
                     'requires_confirmation': bool(name.requires_confirmation or width == 1
                                                   or match_type == 'syllable_edit')}
                    for name, score, match_type in found[:_MAX_CANDIDATES]]
                group = {'heard': text[a:b], 'start': a, 'end': b, 'candidates': options,
                         'match_type': options[0]['match_type'],
                         'requires_confirmation': all(option['requires_confirmation'] for option in options)}
                if truncated or len(found) > _MAX_CANDIDATES:
                    group['truncated'] = True
                groups.append(group)
    # Complete names outrank fragments. Syllable insertion/deletion stays below
    # full windows, so an already matched name does not swallow a final particle.
    groups.sort(key=lambda group: (group['match_type'] == 'syllable_edit',
                                   -len(normalize(group['heard'])),
                                   -group['candidates'][0]['score'], group['start']))
    chosen = []
    for group in groups:
        if any(old['start'] < group['end'] and group['start'] < old['end'] for old in chosen):
            continue
        chosen.append(group)
        if len(chosen) >= limit:
            break
    return sorted(chosen, key=lambda group: group['start'])
