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
    # 由下方 CLI main 的 build 分支调用；source 是 Wiki 源目录，db_file 是输出数据库文件。
    # * 后的参数必须用 source=... 等名称传入；返回导入数量等 metrics 字典。
    source = Path(source) if source is not None else DEFAULT_SOURCE
    # Path 来自 pathlib，把字符串变成路径对象；resolve 得到绝对路径。
    db_file = Path(db_file).resolve()
    db_file.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp 新建临时文件，返回文件描述符 fd 和文件名；SQLite 后面会自行打开该文件。
    fd, filename = tempfile.mkstemp(prefix=db_file.stem + '-', suffix='.building.db', dir=db_file.parent)
    os.close(fd)
    temp = Path(filename)
    try:
        # sqlite3 是 Python 标准库；connect 返回连接对象，本段把它命名为 db。
        # closing 来自 contextlib，离开 with 时调用 db.close；db 不是全局数据库变量。
        with closing(sqlite3.connect(temp)) as db:
            # DDL 从 lore_schema.py 导入，包含建表语句；executescript 执行多条 SQL。
            db.executescript(DDL)
            # SQLite 连接自身的 with 管理事务：正常结束提交，异常时回滚；它不负责 close。
            with db:
                # import_lore_wiki 来自同名模块，读源资料并向这个临时数据库写入记录。
                metrics = import_lore_wiki(db, source)
                for table in TABLES:
                    # execute 返回游标；fetchone 取一行，[0] 取 count(*) 的唯一数值。
                    # 此处表名只来自上方固定 TABLES，不把用户文字拼接为 SQL 表名。
                    metrics[table] = db.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
                # executemany 重复执行带参数的 SQL；? 由后续各个 (键, 值) 元组绑定。
                db.executemany('INSERT INTO meta VALUES(?,?)', (
                    ('schema', DB_VERSION), ('import_status', 'wiki_import_complete'),
                    ('import_stats', json.dumps(metrics, ensure_ascii=False)),
                ))
                if db.execute('PRAGMA foreign_key_check').fetchall():
                    raise ValueError('图谱包含悬空引用，未替换旧库。')
                if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                    raise ValueError('SQLite 完整性检查未通过，未替换旧库。')
        try:
            # 新库完整建成并关闭连接后才替换目标文件，构建失败不会留下半个正式数据库。
            os.replace(temp, db_file)
        except PermissionError as exc:
            raise PermissionError(f'请先退出 main.py 或关闭 SQLite 查看器后再构建：{db_file}') from exc
    finally:
        temp.unlink(missing_ok=True)
    return metrics


def connect(db_file=KNOWLEDGE_DB):
    # retrieve_packet、reply_entity_mentions 和 NPC 的实体解析调用此公共入口。
    # 参数是文件路径；返回打开的 sqlite3.Connection，调用者用 closing(...) 负责关闭。
    path = Path(db_file).resolve()
    if not path.is_file():
        raise FileNotFoundError('知识库尚未构建，请运行 python -m friston3.knowledge.terra_kb build')
    # as_uri 转成 SQLite 可识别的文件 URI；mode=ro 表示这条连接只读取已有数据库。
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    # Row 让查询结果既能 row[0] 按位置取值，也能 row['name'] 按列名取值。
    db.row_factory = sqlite3.Row
    try:
        # version 是一行记录或 None，比较的是记录中的 value 与项目声明的 DB_VERSION。
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
    # 调用方：NpcTalkSkill._execute、_extend_grounded_evidence，以及下方 CLI。
    # question 是问题文字；resolution 可复用上游已解析的实体；search_query 可换检索措辞。
    # 返回 packet 字典，含 subjects/sources/resolution 等；本函数不生成 NPC 台词。
    if not isinstance(question, str) or not question.strip():
        return {'context': '', 'intent': 'general', 'entity_ids': [], 'subjects': [], 'sources': [],
                'resolution': {'status': 'unknown', 'entity_ids': [], 'mentions': [],
                               'ambiguities': [], 'used_previous': False}}
    with closing(connect(db_file)) as db:
        # dict(...) 是浅拷贝，避免覆盖调用方顶层字段；缺省时调用 entity_resolver 自行解析。
        resolution = dict(resolution) if resolution is not None else resolve_entity_resolution(db, question)
        # Keep only IDs present in this snapshot; a model response is never an ID authority.
        # dict.fromkeys 按第一次出现的顺序去重；列表推导式只保留库中确实存在的字符串 ID。
        # [:8] 是切片，最多取得前 8 项；find_entity 是 retrieval_policy.py 导入的查询函数。
        ids = [entity_id for entity_id in dict.fromkeys(resolution['entity_ids'])
               if isinstance(entity_id, str) and find_entity(db, entity_id)][:8]
        resolution['entity_ids'] = ids
        query = search_query if search_query is not None else question
        if resolution['status'] == 'ambiguous' or resolution.get('unresolved_reference'):
            return {'context': '', 'intent': intent or detect_intent(query), 'entity_ids': ids,
                    'subjects': [{key: row[key] for key in ('id', 'name', 'kind', 'scope')}
                                 for entity_id in ids if (row := find_entity(db, entity_id))],
                    'sources': [], 'resolution': resolution}
        # 实际证据组织在 retrieve_lore.build_evidence_packet：它使用混合检索并保留来源。
        # max_chars 控制交给后续提示词的资料篇幅，不是模型的训练 batch size。
        packet = build_evidence_packet(db, query, ids, max_chars=max_chars, intent=intent)
        packet['original_question'] = question
        packet['resolved_question'] = question
        packet['entity_ids'] = ids
        # **resolution 把旧字典键值展开到新字典；后面的同名 entity_ids 覆盖展开值。
        packet['resolution'] = {**resolution, 'entity_ids': list(ids)}
        if IDENTITY.entity_id in ids and len(ids) >= 2 and not packet['sources']:
            others = [entity_id for entity_id in ids if entity_id != IDENTITY.entity_id]
            names = [row['name'] for entity_id in others if (row := find_entity(db, entity_id))]
            overview = build_evidence_packet(db, '介绍' + '、'.join(names), others, max_chars=700)
            packet['target_overviews'] = [source for source in overview['sources'] if source['kind'] == 'overview']
        return packet


def reply_entity_mentions(reply, *, db_file=KNOWLEDGE_DB):
    # NpcTalkSkill._draft_issues 调用：检查生成文本提到的实体，返回 [{id, name}, ...]。
    # resolve_entities 产生候选 ID；find_entity 回读资料，不能只信模型写出的名字。
    with closing(connect(db_file)) as db:
        return [{'id': entity['id'], 'name': entity['name']}
                for entity_id in resolve_entities(db, reply)
                # Name resolution already checks short aliases in context.
                # Do not discard a resolved one-character character name.
                # := 是赋值表达式：把查询结果保存为 entity，同时用于 if 的真值判断。
                if (entity := find_entity(db, entity_id))
                and entity['name'] in reply
                and entity['kind'] in ('character', 'machine', 'fan_machine', 'dialogue_speaker')]


def stats(db_file=KNOWLEDGE_DB):
    # CLI 的 stats 分支使用；返回统计字典。这里的 SQL 只计数，不改写资料。
    with closing(connect(db_file)) as db:
        result = {t: db.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in TABLES}
        result['by_kind'] = dict(db.execute('SELECT kind,count(*) FROM entities GROUP BY kind'))
        result['relations_by_status'] = dict(db.execute('SELECT status,count(*) FROM relations GROUP BY status'))
        result['issues_by_reason'] = dict(db.execute('SELECT reason,count(*) FROM import_issues GROUP BY reason'))
        result['import_stats'] = json.loads(db.execute("SELECT value FROM meta WHERE key='import_stats'").fetchone()[0])
        return result


def network(query, *, db_file=KNOWLEDGE_DB, depth=1, limit=30):
    """Bounded neighborhood for inspection, never a new inferred relationship."""
    # CLI 的 network 分支使用；query 是人物名，返回 nodes/edges 供查看已有关系图。
    depth, limit = min(3, max(0, int(depth))), min(500, max(1, int(limit)))
    with closing(connect(db_file)) as db:
        roots = resolve_entities(db, query)[:2]
        # visited 是已到达实体的集合；edges 用关系 ID 去重；queue 是待访问的双端队列。
        # 每个队列元素为 (实体 ID, 距起点的跳数)，对应广度优先遍历 BFS。
        visited, edges, queue = set(roots), {}, deque((i, 0) for i in roots)
        truncated = False
        while queue:
            # popleft 从队头取出最早加入的一项；下方 append 把下一层节点放到队尾。
            entity_id, distance = queue.popleft()
            if distance >= depth:
                continue
            # JOIN 用实体 ID 把关系表和实体表连接起来；三个 ? 各绑定一个 entity_id。
            # SQL 参数单独传元组，不把人物名或 ID 直接插进 SQL 文本。
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
    # python -m friston3.knowledge.terra_kb 到达文件末尾后调用此函数。
    # argv 可由 Python 调用者传字符串列表；None 让 argparse 读取命令行参数。
    # argparse 是上方导入的标准库模块，ArgumentParser 是它的类；parser 是新建的解析器对象。
    parser = argparse.ArgumentParser(description='Friston-3 Wiki 人物关系知识图谱')
    # add_argument 是 parser 的方法，只声明如何解析，不执行 build 或查询。
    # operation 是位置参数；nargs='?' 允许省略一个参数，省略时取 stats。
    parser.add_argument('operation', nargs='?', default='stats')
    # nargs='*' 收集零个或多个词，之后 args.question 就是一个字符串列表。
    parser.add_argument('question', nargs='*')
    parser.add_argument('--source', help='本地 arknights_lore_wiki 仓库根目录')
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--output', type=Path, help='导出 UTF-8 JSON 文件')
    # parse_args 真正解析，返回 Namespace 对象；下面通过 args.operation 等属性取结果。
    args = parser.parse_args(argv)
    # str.join 用空格连接词列表，还原为完整查询句。
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
                # SearchRequest 是 retrieval_policy.py 的类，把查询、实体和过滤约定集中保存。
                request = SearchRequest(db, query, resolution['entity_ids'])
                if args.operation == 'hybrid':
                    result = hybrid_ranked_search(db, request, limit=args.limit,
                                kinds=('profile', 'character', 'event', 'project'))
                else:
                    result = ranked_search(db, request, limit=args.limit)
    else:
        result = retrieve_packet(' '.join([args.operation, *args.question]))
    # dumps 把 Python 字典/列表转为 JSON 字符串；ensure_ascii=False 保留可读中文。
    content = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + '\n', encoding='utf-8')
        print(f'已导出：{args.output.resolve()}')
    else:
        print(content)


if __name__ == '__main__':
    main()
