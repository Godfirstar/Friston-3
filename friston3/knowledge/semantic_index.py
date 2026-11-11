"""为 NPC 混合检索提供中文语义索引，同时保留独立构建与查询 CLI。

用法：
    python -m friston3.knowledge.semantic_index build
    python -m friston3.knowledge.semantic_index search "为什么凯尔希那么执着" --top-k 5

索引从已构建的 terra_lore.db 读取，不修改原知识库和人物关系。
build 是显式初始化操作，缺失时下载唯一的 BGE 模型；正常对话仅使用本地缓存。
仅索引非关系性文本；检索命中本身不证明人物交情或剧情事实。
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from functools import lru_cache
import os
from pathlib import Path
import sqlite3
import tempfile

import numpy as np

from friston3.config import KNOWLEDGE_DB, PROJECT_ROOT

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
INDEX_PATH = PROJECT_ROOT / "data" / "runtime" / "terra_semantic.npz"
BATCH_SIZE = 32


def _signature(path: Path) -> str:
    """数据库内容指纹：防止旧向量错配新的 passage_id。"""
    # build 与 _snapshot_signature 调用；path 是文件路径，返回 SHA256 十六进制字符串。
    if not path.is_file():
        raise FileNotFoundError(f"知识库不存在：{path}，请先构建 terra_kb。")
    digest = sha256()
    with path.open("rb") as stream:
        # iter(函数, 结束值) 反复调用函数，读到 b'' 才停止；逐块读避免一次把整个库放入内存。
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _connect(db_file: Path):
    # build/search 的内部入口；返回 SQLite 连接，mode=ro 只读取已存在的数据库。
    # row_factory=sqlite3.Row 允许后续用 row['id'] 这样的列名访问结果。
    db = sqlite3.connect(db_file.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def _embedding_class():
    # 延迟导入：未使用语义能力时，不占用模型资源。
    # 返回 fastembed.TextEmbedding 这个类本身；此刻尚未实例化模型，也没有把文本编码成向量。
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError("未安装 FastEmbed，请运行 python -m pip install fastembed") from exc
    return TextEmbedding


@lru_cache(maxsize=1)
def _encoder():
    # search 和 npc.intent_router.prepare 共用；lru_cache 是 functools 的缓存装饰器。
    # 该无参数函数首次构造编码器实例，后续直接复用同一实例，避免每轮重载模型。
    # 对话、意图路由共用这一实例；缺模型不触发联网或改用另一套模型。
    try:
        return _embedding_class()(model_name=MODEL_NAME, threads=2, local_files_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "本地 BGE 语义模型不可用；请运行 python -m friston3.knowledge.semantic_index build "
            f"准备模型和索引。原因为：{exc}"
        ) from exc


def _unit_vectors(vectors: np.ndarray) -> np.ndarray:
    """单位化后，点积即余弦相似度。"""
    # np 是 numpy 的导入别名；输入可以是 [N,D] 或 [D] 数组，返回同形状数组。
    # axis=-1 沿最后的向量维度求长度；keepdims=True 保留末尾长度为 1 的轴以便广播相除。
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    # maximum 逐元素保留不小于 1e-12 的分母，避免零向量除零；没有参数学习或反向传播。
    return vectors / np.maximum(norms, 1e-12)


def build(db_file: Path = KNOWLEDGE_DB, index_path: Path = INDEX_PATH):
    # 仅由本模块 CLI 的 build 分支显式调用：读库→编码全部段落→保存 npz 索引。
    # 参数为库/输出索引路径；结果写文件和终端，没有业务返回值，不会自动训练编码器。
    db_file, index_path = Path(db_file), Path(index_path)
    before = _signature(db_file)
    with _connect(db_file) as db:
        # JOIN 把 passages.ref_id 与 entities.id 对齐，取回“段落 ID、主体名、标题、正文”。
        # fetchall 把游标全部结果收成列表，ORDER BY 保证向量行号与 passage_ids 顺序一致。
        rows = db.execute("""
            SELECT p.id, e.name, p.section, p.content
            FROM passages AS p
            JOIN entities AS e ON e.id = p.ref_id
            WHERE p.kind != 'relationship' AND trim(p.content) != ''
            ORDER BY p.id
        """).fetchall()
    if not rows:
        raise RuntimeError("没有可索引的文本，请先检查 terra_lore.db 的 passages 表。")

    # Only this explicit preparation command may fill the same model cache.
    # 第一个 () 调函数得到类，紧接的 (...) 调类构造器得到 model 实例；允许显式准备时下载缓存。
    # 构造模型与执行 model.embed(texts) 是两步，后者才真正计算这些文本的向量。
    model = _embedding_class()(model_name=MODEL_NAME, threads=2, local_files_only=False)
    # passage_ids 是 [N] 整数数组；第 i 项指明向量矩阵第 i 行属于数据库哪个 passage。
    passage_ids = np.asarray([row["id"] for row in rows], dtype=np.int64)
    batches = []
    print(f"开始为 {len(rows)} 条非关系性文本建立中文向量索引……", flush=True)
    for start in range(0, len(rows), BATCH_SIZE):
        # range 的第三项是步长；[start:start+BATCH_SIZE] 每次切出一个批次，末批可以不足 32 条。
        group = rows[start:start + BATCH_SIZE]
        texts = [f'{row["name"]}｜{row["section"]}：{row["content"]}' for row in group]
        # embed 返回可迭代向量；list 收集它们，asarray 转为 [当前批条数,D] 的 float32 矩阵。
        vectors = np.asarray(list(model.embed(texts, batch_size=16)), dtype=np.float32)
        if len(vectors) != len(group):
            raise RuntimeError("Embedding 返回条目数量与文本不一致。")
        batches.append(_unit_vectors(vectors))
        if (start // BATCH_SIZE + 1) % 20 == 0 or start + len(group) == len(rows):
            print(f"已向量化 {start + len(group)}/{len(rows)} 条", flush=True)

    if _signature(db_file) != before:
        raise RuntimeError("向量生成时原知识库发生变化，拒绝保存过期索引。")
    # batches 是矩阵列表；沿样本轴拼起来，得到完整 [N,D] 矩阵，每行已经单位化。
    matrix = np.concatenate(batches, axis=0)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".npz", prefix="semantic-",
                                         dir=index_path.parent, delete=False) as stream:
            temp_path = Path(stream.name)
            # npz 同时保存命名数组与库指纹：矩阵不能脱离段落 ID 和原库版本单独使用。
            np.savez_compressed(stream, ids=passage_ids, vectors=matrix,
                                db_sha256=before, model=MODEL_NAME)
        os.replace(temp_path, index_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    print(f"索引完成：{index_path}\n文本数：{len(rows)}；向量维度：{matrix.shape[1]}")

@lru_cache(maxsize=2)
def _read_index(path_string: str, mtime_ns: int, file_size: int):
    """缓存向量矩阵；索引文件更新后自动重新加载。"""
    # search 调用；mtime_ns/size 虽不在函数体直接使用，却参与 lru_cache 的键，文件变化会失效。
    # 返回 (段落ID数组, 向量矩阵, 数据库指纹, 编码器名)，仅加载已有数组，不重新编码全部文本。
    with np.load(path_string, allow_pickle=False) as saved:
        return (
            saved["ids"],
            saved["vectors"],
            str(saved["db_sha256"].item()),
            str(saved["model"].item()),
        )


@lru_cache(maxsize=2)
def _snapshot_signature(path_string: str, mtime_ns: int, file_size: int):
    """Hash each immutable database snapshot once, not once per conversation."""
    return _signature(Path(path_string))


def search(question: str, top_k: int = 5, db_file: Path = KNOWLEDGE_DB,
           index_path: Path = INDEX_PATH):
    # hybrid_search.hybrid_ranked_search、NPC 预热和本模块 CLI 调用。
    # question 是查询句，top_k 是结果数；返回带正文/出处/score 的字典列表，不返回台词或实体定论。
    db_file, index_path = Path(db_file), Path(index_path)
    if not index_path.is_file():
        raise FileNotFoundError("语义索引不存在，请先运行 python -m friston3.knowledge.semantic_index build")
    db_stat = db_file.stat()
    current = _snapshot_signature(str(db_file.resolve()), db_stat.st_mtime_ns, db_stat.st_size)
    stat = index_path.stat()

    # 左侧四个变量按顺序接收缓存函数返回的元组；matrix 与 passage_ids 的行号必须保持对应。
    passage_ids, matrix, db_sha256, model_name = _read_index(
        str(index_path.resolve()),
        stat.st_mtime_ns,
        stat.st_size,
    )

    if db_sha256 != current:
        raise RuntimeError(
            "知识库已更新、语义索引过期，请重新运行 semantic_index build。"
        )

    if model_name != MODEL_NAME:
        raise RuntimeError(
            "Embedding 模型与索引不一致，请重新运行 semantic_index build。"
        )
    # 先获取常驻编码器，下面只编码本轮这一个查询；资料段落向量已在 build 阶段生成。
    model = _encoder()
    query = QUERY_PREFIX + question.strip()
    # [query] 包装成单样本批次，[0] 取唯一向量；这与取句子的第一个字符不同。
    query_vector = np.asarray(list(model.embed([query]))[0], dtype=np.float32)
    # [N,D] @ [D] 得到 [N] 分数；@ 是矩阵乘法，两边单位化后每项就是余弦相似度。
    scores = matrix @ _unit_vectors(query_vector)
    top_k = min(max(1, int(top_k)), len(passage_ids))
    # argsort 返回升序的“数组下标”；[-top_k:] 选最大项，再 [::-1] 翻转为高分在前。
    positions = np.argsort(scores)[-top_k:][::-1]
    # positions 是矩阵行号，chosen_ids 才是数据库段落 ID；两者不能混用。
    chosen_ids = [int(passage_ids[i]) for i in positions]

    # 按 ID 数量生成 ?,?,? 占位符；SQL 正文只插入占位符，实际值通过第二参数 chosen_ids 绑定。
    placeholders = ",".join("?" for _ in chosen_ids)
    with _connect(db_file) as db:
        rows = db.execute(f"""
            SELECT p.id, e.name, p.section, p.content, s.path, p.line_start, p.line_end
            FROM passages AS p
            JOIN entities AS e ON e.id=p.ref_id
            JOIN sources AS s ON s.id=p.source_id
            WHERE p.id IN ({placeholders})
        """, chosen_ids).fetchall()
    # SQL 的 IN 查询不承诺按相似度排序，先建 ID 查表，再按 positions 的排名组装输出。
    found = {row["id"]: row for row in rows}
    results = []
    # zip 把相同行号/ID 配对；float 将 NumPy 标量变成普通 Python 数值，便于后续 JSON 输出。
    for position, passage_id in zip(positions, chosen_ids):
        row = found.get(passage_id)
        if row is None:
            continue
        item = dict(row)
        item["score"] = float(scores[position])
        results.append(item)
    return results


def main():
    # python -m friston3.knowledge.semantic_index 执行此入口；被 import 时不会运行 main。
    # parser 是 argparse.ArgumentParser 类的新实例，add_argument 声明参数，parse_args 才读取命令行。
    parser = argparse.ArgumentParser(description="Friston-3 中文语义索引构建与查询")
    parser.add_argument("operation", choices=("build", "search"))
    parser.add_argument("question", nargs="*")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.operation == "build":
        build()
        return
    if not args.question:
        parser.error("search 后面需要写检索问题。")
    results = search(" ".join(args.question), top_k=args.top_k)
    for rank, row in enumerate(results, 1):
        print(f'[{rank}] 语义相似度={row["score"]:.3f} | '
              f'{row["name"]} / {row["section"]} | '
              f'{row["path"]}:{row["line_start"]}-{row["line_end"]}')
        print(row["content"][:160].replace("\n", " "))
    print("提示：以上来自 Wiki 二次总结，语义命中不代表内容已被原作核验。")


if __name__ == "__main__":
    main()
