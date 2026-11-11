"""Friston-3 Lesson 02: source-preserving Hybrid RAG (lexical + dense + RRF).

This module only reorders existing passages. It never creates relations,
asserts personal acquaintance, or upgrades LLM-written summaries to canon.

Run manually (no automated test files):
    python -m friston3.knowledge.terra_kb hybrid "赫默为什么反思科研伦理" --limit 5
    python -m friston3.knowledge.terra_kb hybrid "科研理想和伦理争议之间的冲突" --limit 5
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from friston3.knowledge.lore_text import normalize, tokenize
from friston3.knowledge.lore_search import ranked_search
from friston3.knowledge.semantic_index import search as semantic_search
from friston3.npc.identity import IDENTITY

RRF_K = 60
SPARSE_CANDIDATES = 30
DENSE_CANDIDATES = 80
MAX_FROM_ONE_FILE = 2


def _diversified(rows, limit, ids=()):
    """Prioritize different documents, then backfill if the corpus is narrow."""
    # 下方 hybrid_ranked_search 调用；rows 是按优先级排列的段落字典，返回最多 limit 条。
    # 目标是减少同文档/同内容占满结果，不是重新计算向量；ids 用来照顾多个查询主体。
    rows = list(rows)
    if len(ids) >= 2:
        # next(生成器, None) 取首个匹配段落，没找到则返回 None，而不是抛 StopIteration。
        reserved = [next((row for row in rows if row['ref_id'] == entity_id), None) for entity_id in ids]
        reserved = [row for row in reserved if row is not None]
        # 集合推导式用段落 ID 去重；列表 + 列表把保留的主体段落放到最前面。
        reserved_ids = {row['id'] for row in reserved}
        rows = reserved + [row for row in rows if row['id'] not in reserved_ids]
    # selected 已选、deferred 延后补位、seen 已见内容；Counter 来自 collections，缺键计数为 0。
    selected, deferred, seen, per_file = [], [], set(), Counter()
    for row in rows:
        # normalize 来自 lore_text，统一文字形式后再去重，不直接拿整份字典做 set 元素。
        content_key = normalize(row['content'])
        if content_key in seen:
            continue
        seen.add(content_key)
        # get 缺键返回 None；or 从左到右选择第一个非空来源标识。
        source = row.get('source_path') or row.get('source_id') or str(row['id'])
        if per_file[source] >= MAX_FROM_ONE_FILE:
            deferred.append(row)
        else:
            per_file[source] += 1
            selected.append(row)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        # extend 把若干元素依次加到列表；切片只取还缺的数量。
        selected.extend(deferred[:limit - len(selected)])
    return selected[:limit]


def hybrid_ranked_search(db, request, *, limit=8, kinds=None):
    """Fuse lexical ranking and embedding hits while preserving evidence scope.

    The shared request owns subject and scope checks. Semantic similarity
    retrieves context, never creates a relationship edge.
    """
    # 调用方：retrieve_lore.build_evidence_packet 的检索链，以及 terra_kb 的 hybrid CLI。
    # db 是调用方已打开的 SQLite 连接；request 是 retrieval_policy.SearchRequest 实例。
    # 返回段落字典列表；每条带 id、来源、score 等，后续再挑选进提示词，并非直接回复。
    question, ids, intent = request.question, request.ids, request.intent
    # 三个局部变量从 request 属性解包取得；这里不会重新创建或打开数据库。
    limit = max(1, min(30, int(limit)))
    # ranked_search 从 lore_search 导入，取得词法候选；lexical 是候选列表，不是模型对象。
    lexical = ranked_search(db, request, limit=max(SPARSE_CANDIDATES, limit), kinds=kinds)

    # A known person's introduction already has a high-confidence owner card.
    # Only a real profile can short-circuit retrieval. Reference-only faction
    # nodes need mentions in other documents, including leadership questions.
    # all 要求每个实体查询都有结果；(entity_id,) 末尾逗号表示“只有一个元素的元组”。
    # ? 是 SQL 参数占位符；用户值由 execute 的第二个参数传入。
    has_profiles = bool(ids) and all(db.execute(
        "SELECT 1 FROM entities WHERE id=? AND coverage!='reference_only'", (entity_id,)
    ).fetchone() for entity_id in ids)
    categories = request.categories
    direct_facet = (categories[0] == 'ability' and has_profiles and all(
        any(row['ref_id'] == entity_id and row['category'] == 'ability' for row in lexical)
        for entity_id in ids))
    skip_dense = ((intent == 'introduction' and has_profiles) or direct_facet
                  or (ids == [IDENTITY.entity_id] and intent == 'personality'))
    if skip_dense:
        return _diversified(lexical, limit, ids)

    try:
        # Honor the exact SQLite DB supplied by the caller, not a different
        # default DB whose passage IDs might coincidentally overlap.
        # PRAGMA 读取当前连接的信息；一行中的第 3 项（索引 2）是实际数据库文件路径。
        active_db_file = db.execute('PRAGMA database_list').fetchone()[2]
        missing_names = [entity['name'] for entity in request.entities
                         if normalize(entity['name']) not in normalize(question)]
        semantic_query = ('、'.join(missing_names) + '：' if missing_names else '') + question
        # semantic_search 是 semantic_index.search 的导入别名；返回向量命中的 id/score。
        # 完整正文和出处还要在下方从同一 SQLite 库取回，不能只凭相似度当成事实。
        dense_hits = semantic_search(semantic_query, top_k=DENSE_CANDIDATES,
                                     db_file=Path(active_db_file))
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f'[HybridRAG] 语义索引不可用；安全退回原检索：{exc}')
        return _diversified(lexical, limit, ids)

    # kinds 是允许的段落类型；转成 set 后能快速判断 row['kind'] 是否在其中。
    allow = set(kinds) if kinds is not None else None

    # Re-fetch FULL SQLite passage records. A dense result alone has no
    # evidence metadata and must never bypass source, owner and type checks.
    passage_ids = [int(hit['id']) for hit in dense_hits]
    if passage_ids:
        # 若有 3 个 ID，slots 就是 '?,?,?'；这里只拼接占位符，实际 ID 仍作为参数传入。
        slots = ','.join('?' for _ in passage_ids)
        rows = db.execute(f'''SELECT p.*, s.source_type, s.path AS source_path,
                                  s.url AS source_url, s.revision
            FROM passages p LEFT JOIN sources s ON s.id=p.source_id
            WHERE p.id IN ({slots})''', passage_ids)
        # 把 SQLite Row 转为可修改的 dict，并构造 {段落ID: 段落字典}，便于按 ID 查找。
        by_id = {row['id']: dict(row) for row in rows}
    else:
        by_id = {}

    dense = []
    for hit in dense_hits:
        passage_id = int(hit['id'])
        # get 返回 None 表示该向量命中不在当前资料库中；continue 跳过当前循环这一项。
        row = by_id.get(passage_id)
        if row is None:
            continue
        # Relationship passages are eligible dense candidates too. They still
        # pass through the same owner, entity, event-scope and source checks below;
        # a vector hit alone never establishes that a relationship is true.
        if allow is not None and row['kind'] not in allow:
            continue
        # prepare_candidate 属于 SearchRequest，检查主体/资料范围并补全候选字段。
        row = request.prepare_candidate(row)
        if row is None:
            continue
        # 保存原始向量相似度以便诊断；后面的融合 score 会采用另一种定义。
        row['semantic_similarity'] = hit['score']
        dense.append(row)

    # 无明确人物时，过滤只有极少查询关键词重合的词法候选。
    # 只在向量检索成功后执行，不影响原来的词法降级路径。
    if not ids and dense:
        # tokenize 来自 lore_text；split 得到词列表，set 去掉重复词。
        # 下方 query_terms & ... 是集合交集：统计问题与段落共有多少个不同词。
        query_terms = set(tokenize(question).split())

        if len(query_terms) >= 6:
            lexical = [
                row for row in lexical
                if len(
                    query_terms & set(
                        tokenize(
                            (row.get("section") or "") + " " + row["content"]
                        ).split()
                    )
                ) >= 2
            ]
    # Reciprocal Rank Fusion (RRF) uses rank positions, not incomparable
    # BM25/heuristic scores versus cosine similarities.
    # defaultdict 是 collections 的字典类型；首次读取 weights[id] 会用 float() 生成 0.0。
    # combined 保存每个 ID 的一份段落；channels[id] 用 set() 保存命中过的检索通道。
    weights = defaultdict(float)
    combined = {}
    channels = defaultdict(set)

    # 没有明确实体时，适度提高语义检索权重。
    # 有明确实体时，仍采用均衡融合。
    semantic_weight = 1.4 if not ids else 1.0

    # 外层依次取“通道名、该通道候选列表”；内层按该列表的排名累加贡献。
    for channel, hits in (('lexical', lexical), ('semantic', dense)):
        channel_weight = (
            semantic_weight if channel == "semantic" else 1.0
        )

        # enumerate 给列表元素附上序号；start=1 让第一名的 rank 为 1。
        for rank, row in enumerate(hits, start=1):
            passage_id = int(row["id"])

            # RRF 分数 = 各通道 weight/(60+rank) 之和；越靠前贡献越大。
            # 它融合“名次”，避免直接相加不同量纲的词法分数与余弦相似度；不是正确率。
            weights[passage_id] += (
                    channel_weight / (RRF_K + rank)
            )
            # 同一 ID 命中两条通道会累加分数；set.add 自动避免通道名重复。
            channels[passage_id].add(channel)
            # setdefault 仅首次保存正文，后续同 ID 命中不会不断覆盖这份候选字典。
            combined.setdefault(passage_id, dict(row))
    fused = []
    for passage_id, row in combined.items():
        # dict(row) 做浅拷贝后添加融合字段，不把评分直接写回词法候选列表中的原字典。
        item = dict(row)
        item['fusion_score'] = round(weights[passage_id], 7)
        item['retrieval_channels'] = sorted(channels[passage_id])
        # Keep `score` meaningful for callers accustomed to ranked_search.
        item['score'] = item['fusion_score']
        fused.append(item)
    # sort 原地排序；lambda 返回“负分数、ID”元组，使高分在前，同分再按 ID 稳定排序。
    fused.sort(key=lambda row: (-row['fusion_score'], row['id']))
    # Two weak overlapping channels must not bury a strong source-owned
    # lexical answer. Reserve one owner anchor; semantic results add context.
    anchors = [next((row for row in lexical if row['ref_id'] == entity_id), None)
               for entity_id in ids]
    anchors = [row for row in anchors if row is not None]
    output = _diversified(anchors + fused, limit, ids)
    return output
