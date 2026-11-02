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
    rows = list(rows)
    if len(ids) >= 2:
        reserved = [next((row for row in rows if row['ref_id'] == entity_id), None) for entity_id in ids]
        reserved = [row for row in reserved if row is not None]
        reserved_ids = {row['id'] for row in reserved}
        rows = reserved + [row for row in rows if row['id'] not in reserved_ids]
    selected, deferred, seen, per_file = [], [], set(), Counter()
    for row in rows:
        content_key = normalize(row['content'])
        if content_key in seen:
            continue
        seen.add(content_key)
        source = row.get('source_path') or row.get('source_id') or str(row['id'])
        if per_file[source] >= MAX_FROM_ONE_FILE:
            deferred.append(row)
        else:
            per_file[source] += 1
            selected.append(row)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        selected.extend(deferred[:limit - len(selected)])
    return selected[:limit]


def hybrid_ranked_search(db, request, *, limit=8, kinds=None):
    """Fuse lexical ranking and embedding hits while preserving evidence scope.

    The shared request owns subject and scope checks. Semantic similarity
    retrieves context, never creates a relationship edge.
    """
    question, ids, intent = request.question, request.ids, request.intent
    limit = max(1, min(30, int(limit)))
    lexical = ranked_search(db, request, limit=max(SPARSE_CANDIDATES, limit), kinds=kinds)

    # A known person's introduction already has a high-confidence owner card.
    # Only a real profile can short-circuit retrieval. Reference-only faction
    # nodes need mentions in other documents, including leadership questions.
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
        active_db_file = db.execute('PRAGMA database_list').fetchone()[2]
        missing_names = [entity['name'] for entity in request.entities
                         if normalize(entity['name']) not in normalize(question)]
        semantic_query = ('、'.join(missing_names) + '：' if missing_names else '') + question
        dense_hits = semantic_search(semantic_query, top_k=DENSE_CANDIDATES,
                                     db_file=Path(active_db_file))
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f'[HybridRAG] 语义索引不可用；安全退回原检索：{exc}')
        return _diversified(lexical, limit, ids)

    allow = set(kinds) if kinds is not None else None

    # Re-fetch FULL SQLite passage records. A dense result alone has no
    # evidence metadata and must never bypass source, owner and type checks.
    passage_ids = [int(hit['id']) for hit in dense_hits]
    if passage_ids:
        slots = ','.join('?' for _ in passage_ids)
        rows = db.execute(f'''SELECT p.*, s.source_type, s.path AS source_path,
                                  s.url AS source_url, s.revision
            FROM passages p LEFT JOIN sources s ON s.id=p.source_id
            WHERE p.id IN ({slots})''', passage_ids)
        by_id = {row['id']: dict(row) for row in rows}
    else:
        by_id = {}

    dense = []
    for hit in dense_hits:
        passage_id = int(hit['id'])
        row = by_id.get(passage_id)
        if row is None:
            continue
        # Relationship passages are eligible dense candidates too. They still
        # pass through the same owner, entity, event-scope and source checks below;
        # a vector hit alone never establishes that a relationship is true.
        if allow is not None and row['kind'] not in allow:
            continue
        row = request.prepare_candidate(row)
        if row is None:
            continue
        row['semantic_similarity'] = hit['score']
        dense.append(row)

    # 无明确人物时，过滤只有极少查询关键词重合的词法候选。
    # 只在向量检索成功后执行，不影响原来的词法降级路径。
    if not ids and dense:
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
    weights = defaultdict(float)
    combined = {}
    channels = defaultdict(set)

    # 没有明确实体时，适度提高语义检索权重。
    # 有明确实体时，仍采用均衡融合。
    semantic_weight = 1.4 if not ids else 1.0

    for channel, hits in (('lexical', lexical), ('semantic', dense)):
        channel_weight = (
            semantic_weight if channel == "semantic" else 1.0
        )

        for rank, row in enumerate(hits, start=1):
            passage_id = int(row["id"])

            weights[passage_id] += (
                    channel_weight / (RRF_K + rank)
            )
            channels[passage_id].add(channel)
            combined.setdefault(passage_id, dict(row))
    fused = []
    for passage_id, row in combined.items():
        item = dict(row)
        item['fusion_score'] = round(weights[passage_id], 7)
        item['retrieval_channels'] = sorted(channels[passage_id])
        # Keep `score` meaningful for callers accustomed to ranked_search.
        item['score'] = item['fusion_score']
        fused.append(item)
    fused.sort(key=lambda row: (-row['fusion_score'], row['id']))
    # Two weak overlapping channels must not bury a strong source-owned
    # lexical answer. Reserve one owner anchor; semantic results add context.
    anchors = [next((row for row in lexical if row['ref_id'] == entity_id), None)
               for entity_id in ids]
    anchors = [row for row in anchors if row is not None]
    output = _diversified(anchors + fused, limit, ids)
    return output
