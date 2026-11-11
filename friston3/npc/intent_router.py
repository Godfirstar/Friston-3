"""Advisory semantic signal for the NPC Agent's evidence-demand decision.

This module does NOT search the lore database and must not be treated as the
final authority for CHAT/RETRIEVE/LIVE decisions. Its editable labelled examples
are encoded with the same FastEmbed model as the lore retriever. The AgenticRAG
model makes the final routing choice when deterministic entity/context handling
does not already require a bounded clarification or retrieval. Similarity and
margin are ranking signals, NOT calibrated probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
import warnings

from friston3.config import PROJECT_ROOT


LABELS = ("CHAT", "KNOWLEDGE", "LIVE", "FOLLOWUP")
# LABELS 是元组：每个标签的位置也作为后续 NumPy 数组里的整数类别编码。
# PROJECT_ROOT 来自项目 config.py；Path 的 / 运算在这里拼接路径，不是数值除法。
EXAMPLES_PATH = PROJECT_ROOT / "data" / "knowledge" / "intent_examples.json"
CACHE_PATH = PROJECT_ROOT / "data" / "runtime" / "intent_semantic.npz"


# dataclass 为下面字段生成构造器；frozen=True 禁止普通属性重新赋值。
# 例如 SemanticGuess(candidate='CHAT') 得到一个携带路由建议的数据对象，不调用模型。
@dataclass(frozen=True)
class SemanticGuess:
    label: str = "UNCERTAIN"
    candidate: str = "UNCERTAIN"
    similarity: float = 0.0
    margin: float = 0.0
    agreement: float = 0.0
    elapsed_ms: float = 0.0

    def trace(self):
        # 调用方 NpcTalkSkill.prepare_route_context；self 就是当前 SemanticGuess 实例。
        # 返回普通字典供 JSON 诊断使用，round 仅改变展示精度，不改变对象原始数值。
        return {"label": self.label, "candidate": self.candidate,
                "similarity": round(self.similarity, 3),
                "margin": round(self.margin, 3),
                "agreement": round(self.agreement, 2),
                "elapsed_ms": round(self.elapsed_ms, 1)}


class SemanticIntentRouter:
    """Selective kNN semantic router; abstains when neighbors disagree.

    Works without sklearn or a second resident language model. The small
    exemplar index lives separately from terra_semantic.npz and never mixes
    dialogue utterances with evidence passages. Index update is atomic.
    """
    def __init__(self):
        # NpcTalkSkill.__init__ 调用类名 SemanticIntentRouter()，Python 随即执行本方法。
        # self 是新实例；self.xxx 是此实例保存的属性，后面的 prepare/classify 共用它们。
        # 这里只记录现有阈值并把模型/矩阵设为 None；真正编码示例发生在 prepare。
        self.min_similarity = 0.78
        self.min_margin = 0.045
        self.min_agreement = 0.8
        self._vectors = None
        self._labels = None
        self._model = None
        self.error = ""

    @staticmethod
    def _examples():
        # prepare 调用；staticmethod 表示这个函数不需要隐式传入 self。
        # 输入来自模块常量 EXAMPLES_PATH；返回 (句子列表, 整数标签列表, 内容摘要)。
        payload = EXAMPLES_PATH.read_bytes()
        # read_bytes 得到 bytes；decode 转成 str；json.loads 再把 JSON 文本转成 Python 字典。
        examples = json.loads(payload.decode("utf-8-sig"))
        if not isinstance(examples, dict) or set(examples) != set(LABELS):
            raise ValueError("意图语料必须包含 CHAT/KNOWLEDGE/LIVE/FOLLOWUP 四类")
        # 一次解包分别创建两个列表和一个集合；seen 用于检测句子重复。
        records, labels, seen = [], [], set()
        for label in LABELS:
            if not isinstance(examples[label], list) or len(examples[label]) < 12:
                raise ValueError(f"意图语料 {label} 条目不足")
            for sentence in examples[label]:
                if not isinstance(sentence, str) or not sentence.strip() or len(sentence) > 500:
                    raise ValueError(f"意图语料 {label} 包含非法文本")
                sentence = sentence.strip()
                if sentence in seen:
                    raise ValueError(f"意图语料重复：{sentence}")
                seen.add(sentence)
                # append 在列表末尾增加元素；两个列表按同一顺序追加，行号因此一一对应。
                records.append(sentence)
                labels.append(LABELS.index(label))
        # SHA256 摘要是缓存版本标记；示例文件内容变化，即使文件名不变也会得到新摘要。
        return records, labels, sha256(payload).hexdigest()

    def prepare(self):
        """Warm at program startup, not halfway through someone's first chat."""
        # 调用方是 NPC 构造器；classify 也会在未准备时调用。无返回值，结果写入 self 属性。
        if self._vectors is not None:
            return
        # Share the existing retrieval encoder instead of loading another model.
        # np 是 numpy 模块的简写别名；_encoder/_unit_vectors 是本项目 semantic_index 的函数。
        import numpy as np
        from friston3.knowledge.semantic_index import MODEL_NAME, _encoder, _unit_vectors
        # 左侧三个局部变量按顺序接收 _examples 返回的三个对象。
        examples, labels, digest = self._examples()
        version = f"{MODEL_NAME}:{digest}:v2"
        matrix = None
        if CACHE_PATH.is_file():
            try:
                # with 在退出这段代码时释放 npz 文件；这里按键名读取已保存的数组。
                with np.load(CACHE_PATH, allow_pickle=False) as cache:
                    if (str(cache["version"].item()) == version
                            and len(cache["vectors"]) == len(examples)
                            and np.array_equal(cache["labels"], np.asarray(labels, dtype=np.uint8))):
                        matrix = cache["vectors"].astype(np.float32)
            except (OSError, KeyError, ValueError, TypeError):
                pass
        # model 是共享的 FastEmbed 编码器实例；它的 embed 方法把句子变成向量。
        model = _encoder()  # shared with lore retriever; never a second embedding model
        if matrix is None:
            # embed 返回可迭代结果，list 收集后，asarray 转成 [示例数 N, 向量维度 D] 的数组。
            matrix = np.asarray(list(model.embed(examples, batch_size=32)), dtype=np.float32)
            if len(matrix) != len(examples):
                raise RuntimeError("意图 Embedding 条数不匹配")
            # 每行除以自身长度，得到单位向量；后续点积才能直接表示余弦相似度。
            matrix = _unit_vectors(matrix)
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            temp = None
            try:
                with tempfile.NamedTemporaryFile(mode="wb", dir=CACHE_PATH.parent,
                         suffix=".npz", prefix="intent-", delete=False) as stream:
                    temp = Path(stream.name)
                    np.savez_compressed(stream, vectors=matrix,
                        labels=np.asarray(labels, dtype=np.uint8), version=version)
                # 先写临时文件再替换最终文件，读者不会看到只写了一半的 npz。
                os.replace(temp, CACHE_PATH)
            except OSError as error:
                # A read-only working tree must not prevent classification.
                warnings.warn(f"无法持久化意图索引，内存索引仍可用：{error}", RuntimeWarning)
            finally:
                if temp is not None:
                    temp.unlink(missing_ok=True)
        # 局部变量离开函数后不再由这个名字访问；保存到 self 后，classify 仍能复用。
        self._model = model
        self._vectors = matrix
        self._labels = np.asarray(labels, dtype=np.uint8)

    def classify(self, text: str) -> SemanticGuess:
        # NpcTalkSkill.prepare_route_context 调用；返回建议，既不查询知识库也不执行技能。
        # text: str / -> SemanticGuess 是输入和返回类型提示，不是运行时自动转换。
        start = perf_counter()
        if self.error:
            return SemanticGuess()
        if self._vectors is None:
            self.prepare()
        import numpy as np
        from friston3.knowledge.semantic_index import _unit_vectors
        value = str(text or "").strip()
        if not value:
            return SemanticGuess()
        # [value] 把一个字符串包装成一条样本的列表；[0] 取其唯一的 D 维编码结果。
        vector = np.asarray(list(self._model.embed([value]))[0], dtype=np.float32)
        # 示例矩阵 [N,D] @ 单位查询向量 [D] -> [N]。示例在 prepare 中也已归一化，
        # 因此点积等于余弦相似度；这里用固定编码器推理，没有 backward 或参数更新。
        similarities = self._vectors @ _unit_vectors(vector)
        neighbor_count = min(5, len(similarities))
        # positions 保存“示例行号”，不是相似度；[-k:] 切片取最后 k 项。
        positions = np.argpartition(similarities, -neighbor_count)[-neighbor_count:]
        # argpartition 只保证取到最大的 k 项，不保证内部顺序，所以再对 k 项排序。
        # 数组[行号数组] 可一次选多项；argsort 返回升序行号，[::-1] 把顺序反转。
        positions = positions[np.argsort(similarities[positions])[::-1]]
        local_labels = self._labels[positions]
        # The top semantic matches vote, instead of interpreting an isolated
        # similar lore passage or an uncalibrated softmax as evidence demand.
        per_label = []
        for code in range(len(LABELS)):
            # 每类取最相似的三个示例求平均；最终类别由这些均值排名决定。
            # 上面的五近邻则用于下方 agreement，不是另训练一个分类网络。
            # self._labels == code 得到布尔数组，以它索引就筛出“当前类别”的示例相似度。
            matching = np.sort(similarities[self._labels == code])[-3:]
            per_label.append(float(matching.mean()))
        # per_label 是四个类别的均值列表；ranking[0]/[1] 分别是第一名/第二名类别编号。
        ranking = np.argsort(per_label)[::-1]
        candidate = LABELS[int(ranking[0])]
        # margin 是两类均值之差；agreement 是五个近邻中支持第一名类别的比例。
        # 它们都是路由线索，不是“这个回答有多大概率正确”。
        margin = per_label[int(ranking[0])] - per_label[int(ranking[1])]
        agreement = float(np.count_nonzero(local_labels == int(ranking[0])) / neighbor_count)
        best_similarity = float(similarities[positions[0]])
        # and 要求三个已有条件同时成立；这是意图建议的置信筛选，不会控制机械臂限位。
        confident = (best_similarity >= self.min_similarity
                     and margin >= self.min_margin and agreement >= self.min_agreement)
        # A two-turn reference only has meaning to the LLM with history.
        # FOLLOWUP is never a direct CHAT/RETRIEVE instruction.
        label = candidate if confident else "UNCERTAIN"
        return SemanticGuess(label=label, candidate=candidate,
            similarity=best_similarity, margin=margin, agreement=agreement,
            elapsed_ms=(perf_counter() - start) * 1000)
