"""Offline source-attributed Wiki relationship graph, build/query/export CLI."""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from friston3.config import KNOWLEDGE_DB, LORE_WIKI_SOURCE
from friston3.knowledge.entity_resolver import resolve_entities, resolve_entity_resolution
from friston3.knowledge.import_lore_wiki import import_lore_wiki
from friston3.knowledge.lore_schema import DDL, DB_VERSION
from friston3.knowledge.retrieve_lore import build_evidence_packet
from friston3.knowledge.lore_search import get_card, ranked_search
from friston3.knowledge.hybrid_search import hybrid_ranked_search
from friston3.knowledge.retrieval_policy import SearchRequest, detect_intent, find_entity
from friston3.npc.identity import IDENTITY

DEFAULT_SOURCE = LORE_WIKI_SOURCE
TABLES = ('entities', 'aliases', 'sources', 'passages', 'relations', 'event_mentions', 'import_issues')


def build_db(*, source=None, db_file=KNOWLEDGE_DB):
    """Validate a complete new snapshot before atomically replacing only lore DB."""
    source = Path(source) if source is not None else DEFAULT_SOURCE
    db_file = Path(db_file).resolve()
    db_file.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix=db_file.stem + '-', suffix='.building.db', dir=db_file.parent)
    os.close(fd)
    temp = Path(filename)
    try:
        with closing(sqlite3.connect(temp)) as db:
            db.executescript(DDL)
            with db:
                metrics = import_lore_wiki(db, source)
                for table in TABLES:
                    metrics[table] = db.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
                db.executemany('INSERT INTO meta VALUES(?,?)', (
                    ('schema', DB_VERSION), ('import_status', 'wiki_import_complete'),
                    ('import_stats', json.dumps(metrics, ensure_ascii=False)),
                ))
                if db.execute('PRAGMA foreign_key_check').fetchall():
                    raise ValueError('图谱包含悬空引用，未替换旧库。')
                if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                    raise ValueError('SQLite 完整性检查未通过，未替换旧库。')
        try:
            os.replace(temp, db_file)
        except PermissionError as exc:
            raise PermissionError(f'请先退出 main.py 或关闭 SQLite 查看器后再构建：{db_file}') from exc
    finally:
        temp.unlink(missing_ok=True)
    return metrics


def connect(db_file=KNOWLEDGE_DB):
    path = Path(db_file).resolve()
    if not path.is_file():
        raise FileNotFoundError('知识库尚未构建，请运行 python -m friston3.knowledge.terra_kb build')
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        version = db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
        if not version or version[0] != DB_VERSION:
            raise RuntimeError('知识库版本已更换，请退出 main.py 后运行 python -m friston3.knowledge.terra_kb build')
    except Exception:
        db.close()
        raise
    return db


def retrieve_packet(question, *, resolution=None, search_query=None, intent=None,
                    db_file=KNOWLEDGE_DB, max_chars=2500):
    """Retrieve evidence for one resolved task, optionally using a new search query.

    Dialogue supplies its SQLite entity resolution. Standalone callers can omit
    it to resolve ``question`` here. Raw user text stays with the dialogue turn.
    """
    if not isinstance(question, str) or not question.strip():
        return {'context': '', 'intent': 'general', 'entity_ids': [], 'subjects': [], 'sources': [],
                'resolution': {'status': 'unknown', 'entity_ids': [], 'mentions': [],
                               'ambiguities': [], 'used_previous': False}}
    with closing(connect(db_file)) as db:
        resolution = dict(resolution) if resolution is not None else resolve_entity_resolution(db, question)
        # Keep only IDs present in this snapshot; a model response is never an ID authority.
        ids = [entity_id for entity_id in dict.fromkeys(resolution['entity_ids'])
               if isinstance(entity_id, str) and find_entity(db, entity_id)][:8]
        resolution['entity_ids'] = ids
        query = search_query if search_query is not None else question
        if resolution['status'] == 'ambiguous' or resolution.get('unresolved_reference'):
            return {'context': '', 'intent': intent or detect_intent(query), 'entity_ids': ids,
                    'subjects': [{key: row[key] for key in ('id', 'name', 'kind', 'scope')}
                                 for entity_id in ids if (row := find_entity(db, entity_id))],
                    'sources': [], 'resolution': resolution}
        packet = build_evidence_packet(db, query, ids, max_chars=max_chars, intent=intent)
        packet['original_question'] = question
        packet['resolved_question'] = question
        packet['entity_ids'] = ids
        packet['resolution'] = {**resolution, 'entity_ids': list(ids)}
        if IDENTITY.entity_id in ids and len(ids) >= 2 and not packet['sources']:
            others = [entity_id for entity_id in ids if entity_id != IDENTITY.entity_id]
            names = [row['name'] for entity_id in others if (row := find_entity(db, entity_id))]
            overview = build_evidence_packet(db, '介绍' + '、'.join(names), others, max_chars=700)
            packet['target_overviews'] = [source for source in overview['sources'] if source['kind'] == 'overview']
        return packet


def reply_entity_mentions(reply, *, db_file=KNOWLEDGE_DB):
    with closing(connect(db_file)) as db:
        return [{'id': entity['id'], 'name': entity['name']}
                for entity_id in resolve_entities(db, reply)
                # Name resolution already checks short aliases in context.
                # Do not discard a resolved one-character character name.
                if (entity := find_entity(db, entity_id))
                and entity['name'] in reply
                and entity['kind'] in ('character', 'machine', 'fan_machine', 'dialogue_speaker')]


def stats(db_file=KNOWLEDGE_DB):
    with closing(connect(db_file)) as db:
        result = {t: db.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in TABLES}
        result['by_kind'] = dict(db.execute('SELECT kind,count(*) FROM entities GROUP BY kind'))
        result['relations_by_status'] = dict(db.execute('SELECT status,count(*) FROM relations GROUP BY status'))
        result['issues_by_reason'] = dict(db.execute('SELECT reason,count(*) FROM import_issues GROUP BY reason'))
        result['import_stats'] = json.loads(db.execute("SELECT value FROM meta WHERE key='import_stats'").fetchone()[0])
        return result


def network(query, *, db_file=KNOWLEDGE_DB, depth=1, limit=30):
    """Bounded neighborhood for inspection, never a new inferred relationship."""
    depth, limit = min(3, max(0, int(depth))), min(500, max(1, int(limit)))
    with closing(connect(db_file)) as db:
        roots = resolve_entities(db, query)[:2]
        visited, edges, queue = set(roots), {}, deque((i, 0) for i in roots)
        truncated = False
        while queue:
            entity_id, distance = queue.popleft()
            if distance >= depth:
                continue
            rows = db.execute('''SELECT r.*,a.name AS src_name,b.name AS dst_name
                FROM relations r JOIN entities a ON a.id=r.src JOIN entities b ON b.id=r.dst
                WHERE r.src=? OR r.dst=?
                ORDER BY CASE WHEN r.src=? THEN 0 ELSE 1 END,r.id''', (entity_id, entity_id, entity_id))
            for row in rows:
                if row['id'] in edges:
                    continue
                if len(edges) >= limit:
                    truncated = True
                    break
                edges[row['id']] = dict(row)
                for target in (row['src'], row['dst']):
                    if target not in visited:
                        visited.add(target)
                        queue.append((target, distance+1))
        return {'roots': roots, 'depth': depth, 'truncated': truncated,
                'note': '箭头表示资料页面主体到关系对象；多跳路径不推导人物关系。Wiki为未人工核验的二次总结。',
                'nodes': [find_entity(db, i) for i in sorted(visited)], 'edges': list(edges.values())}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Friston-3 Wiki 人物关系知识图谱')
    parser.add_argument('operation', nargs='?', default='stats')
    parser.add_argument('question', nargs='*')
    parser.add_argument('--source', help='本地 arknights_lore_wiki 仓库根目录')
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--output', type=Path, help='导出 UTF-8 JSON 文件')
    args = parser.parse_args(argv)
    query = ' '.join(args.question)
    if args.operation == 'build':
        result = build_db(source=args.source)
    elif args.operation == 'stats':
        result = stats()
    elif args.operation == 'network':
        if not query:
            parser.error('network 后请写人物名')
        result = network(query, depth=args.depth, limit=args.limit)
    elif args.operation in ('profile', 'card', 'relations'):
        if not query:
            parser.error('请写人物名')
        with closing(connect()) as db:
            ids = resolve_entities(db, query)
            if not ids:
                result = {'message': '没有明确匹配到人物，请使用完整名称。'}
            elif args.operation == 'relations':
                slots = ','.join('?' for _ in ids)
                where = f'r.src IN ({slots}) AND r.dst IN ({slots})' if len(ids)>1 else 'r.src=? OR r.dst=?'
                result = [dict(row) for row in db.execute(f'''SELECT r.*,a.name AS src_name,b.name AS dst_name
                    FROM relations r JOIN entities a ON a.id=r.src JOIN entities b ON b.id=r.dst
                    WHERE ({where}) ORDER BY r.id LIMIT ?''', (*ids, *ids, min(500, max(1, args.limit))))]
            else:
                result = get_card(db, ids[0], max_sections=min(10, max(1, args.limit)))
    elif args.operation in ('search', 'hybrid'):
        if not query:
            parser.error('search/hybrid 后请写关键词')
        with closing(connect()) as db:
            resolution = resolve_entity_resolution(db, query)
            if resolution['status'] == 'ambiguous':
                result = []
            else:
                request = SearchRequest(db, query, resolution['entity_ids'])
                if args.operation == 'hybrid':
                    result = hybrid_ranked_search(db, request, limit=args.limit,
                                kinds=('profile', 'character', 'event', 'project'))
                else:
                    result = ranked_search(db, request, limit=args.limit)
    else:
        result = retrieve_packet(' '.join([args.operation, *args.question]))
    content = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + '\n', encoding='utf-8')
        print(f'已导出：{args.output.resolve()}')
    else:
        print(content)


if __name__ == '__main__':
    main()
