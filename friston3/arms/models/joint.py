"""π₀.₅ 联合后训练：共享 VLM 预测当前子任务和 FAST 动作 token。

两个离散目标都更新最后一个语言层的 K/V 低秩残差。同一层还为动作专家提供
条件 KV。连续动作分支只读取观测和低层指令，在独立前向中计算 Flow Matching。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..config import JOINT_FORMAT
from .weights import fingerprint
from .pi05 import COMPLETE, flow_loss as velocity_mse, make_flow


# OpenPI 使用扩展后的 big_vision SentencePiece 行号。HF 可将 loc/seg 记作 added
# tokens，或另加 <image>；len(HF tokenizer) 和 tokenizer.vocab_size 并不总是此值。
GEMMA_TEXT_VOCAB = 256_000
PALI_VOCAB = GEMMA_TEXT_VOCAB + 1024 + 128
FAST_TOP_ID = PALI_VOCAB - 1 - 128


class FastTokenizer:
    """用官方 FAST+ 发布的两个 JSON 编码动作，无需加载远端 Python 模块。"""

    def __init__(self, directory):
        from tokenizers import Tokenizer

        directory = Path(directory)
        settings = directory / "processor_config.json"
        vocabulary = directory / "tokenizer.json"
        config = json.loads(settings.read_text(encoding="utf-8"))
        if config.get("processor_class") != "UniversalActionProcessor":
            raise ValueError("Expected FAST UniversalActionProcessor in processor_config.json")
        self.scale = float(config["scale"])
        self.min_token = int(config["min_token"])
        self.vocab_size = int(config["vocab_size"])
        self.bpe = Tokenizer.from_file(str(vocabulary))
        if (not math.isfinite(self.scale) or self.scale <= 0 or not 0 < self.vocab_size <= FAST_TOP_ID
                or self.bpe.get_vocab_size() != self.vocab_size):
            raise ValueError("FAST processor and BPE vocabulary disagree")
        self.metadata = {
            "processor_sha256": fingerprint(settings), "bpe_sha256": fingerprint(vocabulary),
            "vocab_size": self.vocab_size, "paligemma_top_id": FAST_TOP_ID,
            "action_normalization": "training_q01_q99_clip_minus1_plus1",
        }

    def encode(self, actions):
        from scipy.fft import dct

        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 6 or not np.isfinite(values).all():
            raise ValueError("Expected a finite [horizon,6] normalized SO-101 action chunk")
        # 每个关节沿时间轴变换，按 [频率,关节] 展平，FAST 和联合 Flow 共用分位数裁剪。
        coeffs = dct(np.clip(values, -1, 1), axis=0, norm="ortho")
        symbols = np.maximum(np.rint(coeffs * self.scale) - self.min_token, 0).astype(np.int64)
        text = "".join(map(chr, symbols.ravel().tolist()))
        ids = self.bpe.encode(text, add_special_tokens=False).ids
        if not ids or min(ids) < 0 or max(ids) >= self.vocab_size:
            raise ValueError("FAST encoded an empty sequence or an invalid BPE ID")
        return ids


class _KVResidual(nn.Module):
    """在原线性层返回后叠加 BAx；冻结层可以由 Accelerate 卸载到 CPU。"""

    def __init__(self, projection, rank, alpha, device):
        super().__init__()
        self.down = nn.Linear(projection.in_features, rank, bias=False, device=device, dtype=torch.float32)
        self.up = nn.Linear(rank, projection.out_features, bias=False, device=device, dtype=torch.float32)
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def add_to_projection(self, projection, args, output):
        update = self.up(self.down(args[0].to(self.down.weight))).mul(self.scale)
        return output + update.to(output)


class SubtaskDecodeError(ValueError):
    """语言解码未得到以 EOS 结束的有效子任务。"""


class JointPolicy:
    """离散预测共用 PaliGemma 词表和语言层，部署动作继续用 LeRobot sample_actions。"""

    def __init__(self, model, tokenizer, rank, alpha, device, *, fast_dir=None):
        self.model, self.tokenizer = model, tokenizer
        self.device = torch.device(device)
        pali = model.paligemma_with_expert.paligemma
        self.language = pali.model.language_model
        self.head = pali.lm_head
        if (self.head.weight.shape != self.language.get_input_embeddings().weight.shape
                or self.head.out_features < PALI_VOCAB or self.head.weight.requires_grad):
            raise ValueError("Joint training needs the frozen, full PaliGemma output and embedding vocabulary")
        self.eos = tokenizer.eos_token_id
        specials = (tokenizer.bos_token_id, self.eos, tokenizer.pad_token_id)
        if len(set(specials)) != 3 or any(i is None or not 0 <= i < GEMMA_TEXT_VOCAB for i in specials):
            raise ValueError("Incompatible PaliGemma BOS/EOS/pad IDs")
        self.specials = set(tokenizer.all_special_ids)
        # Gemma 部分 tokenizer 将普通换行也列入 all_special_ids。提示模板含换行，
        # 标签仍只接受自然文本；分别处理，避免训练在第一个提示上就失败。
        line_break = tokenizer.encode("\n", add_special_tokens=False)
        valid_break = set(line_break).difference(specials, {getattr(tokenizer, "unk_token_id", None)}) if line_break else set()
        self.prompt_specials = self.specials.difference(valid_break)
        self.fast = FastTokenizer(fast_dir) if fast_dir is not None else None
        self.fast_metadata = self.fast.metadata if self.fast is not None else None
        attention = self.language.layers[-1].self_attn
        if not 0 < rank <= attention.k_proj.in_features or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Invalid shared VLM adapter rank or alpha")
        if hasattr(model, "joint_adapter"):
            raise ValueError("The shared VLM adapter was already attached")
        # 模块单独注册在模型根节点，优化器、保存和恢复使用同一组参数名。保持原投影
        # 及其权重装载 hook 完整；在 CPU 卸载层上，原权重静止时可能处于 meta 设备。
        model.joint_adapter = nn.ModuleDict()
        for name in ("k_proj", "v_proj"):
            projection = getattr(attention, name)
            residual = _KVResidual(projection, rank, alpha, self.device)
            model.joint_adapter[name] = residual
            projection.register_forward_hook(residual.add_to_projection)

    def _text(self, text, *, prompt=False):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        forbidden = self.prompt_specials if prompt else self.specials
        if not ids or any(i in forbidden or not 0 <= i < GEMMA_TEXT_VOCAB for i in ids):
            raise ValueError("Expected ordinary PaliGemma text tokens")
        return ids

    def _goal_inputs(self, images, masks, goal, state, views=()):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError("High-level prediction requires six finite normalized joint readings")
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("A nonempty global goal is required")
        bins = np.clip(np.floor((state + 1) * 128), 0, 255).astype(np.int64)
        joints = " ".join(map(str, bins))
        request = "Current subtask:"
        if views:
            request = "Views: " + ", ".join(views) + "\nRelevant boxes by view, then current subtask:"
        prompt = f"Goal: {goal.strip()}\nState: {joints}\n{request}\n"
        tokens = [self.tokenizer.bos_token_id, *self._text(prompt, prompt=True)]
        if len(tokens) > self.model.config.tokenizer_max_length:
            raise ValueError("Global goal plus current state exceeds the model's prompt length")
        ids = torch.tensor([tokens], device=self.device)
        return {"images": images, "img_masks": masks, "tokens": ids, "masks": torch.ones_like(ids, dtype=torch.bool)}

    def _mask(self, allowed):
        dtype = self.language.layers[0].self_attn.q_proj.weight.dtype
        return torch.zeros_like(allowed, dtype=dtype).masked_fill_(~allowed, torch.finfo(dtype).min)[:, None]

    def _observe(self, inputs, answer_prefix, *, use_cache):
        # 忽略缺失相机和右侧文本 padding，最后一个真实提示位置负责预测第一个目标。
        views = [(img, mask) for img, mask in zip(inputs["images"], inputs["img_masks"], strict=True) if mask.item()]
        if not views or inputs["tokens"].shape[0] != 1:
            raise ValueError("Joint training expects one sample and at least one valid camera")
        tokens = inputs["tokens"][:, inputs["masks"][0]]
        if tokens.shape[1] == 0:
            raise ValueError("Empty language prompt")
        if answer_prefix:
            tokens = torch.cat((tokens, tokens.new_tensor([answer_prefix])), dim=1)
        with torch.no_grad():
            embeds, valid, _ = self.model.embed_prefix(
                [img for img, _ in views], [mask for _, mask in views],
                tokens, torch.ones_like(tokens, dtype=torch.bool))
        if embeds.shape[1] > self.language.config.max_position_embeddings:
            raise ValueError("Image, prompt and target tokens exceed the language context")
        group = torch.zeros_like(valid, dtype=torch.long)
        if answer_prefix:
            group[:, -len(answer_prefix):] = 1
        group = group.cumsum(1)
        # 前缀双向且看不到标签；每个 teacher-forced 答案 token 只看自身和左侧。
        allowed = (group[:, None, :] <= group[:, :, None]) & valid[:, :, None] & valid[:, None, :]
        self.language.config._attn_implementation = "sdpa"
        result = self.language(inputs_embeds=embeds, attention_mask=self._mask(allowed),
                               position_ids=valid.long().cumsum(1) - 1, use_cache=use_cache)
        return result.last_hidden_state, result.past_key_values, valid

    def _block_ce(self, hidden, ids):
        logits = self.head(hidden.to(self.head.weight.dtype))[..., :PALI_VOCAB]
        return F.cross_entropy(logits.float(), ids, reduction="sum")

    def _loss(self, inputs, target):
        hidden, _, _ = self._observe(inputs, target[:-1], use_cache=False)
        selected = hidden[0, -len(target):]
        target_ids = torch.tensor(target, device=selected.device)
        # 共享词表有 25 万行；分块重算输出层，避免保存整个动作序列的 float32 logits。
        total = selected.new_zeros((), dtype=torch.float32)
        for features, ids in zip(selected.split(16), target_ids.split(16), strict=True):
            if features.requires_grad and torch.is_grad_enabled():
                total = total + checkpoint(self._block_ce, features, ids, use_reentrant=False)
            else:
                total = total + self._block_ce(features, ids)
        return total / len(target)

    def flow_loss(self, inputs, actions_6, seed=None):
        """联合阶段 α·Flow 的未加权 MSE。干净前缀的 K/V 对共享适配器保持可微。"""
        # 沿用相同的图像、指令、状态前缀。这里不拼入任何真实 FAST token，
        # 不复用上一次更新的缓存。冻结视觉与词嵌入；最后一层 K/V 的残差参与求导。
        _, cache, valid = self._observe(inputs, [], use_cache=True)
        if cache is None:
            raise RuntimeError("Action expert requires the observation prefix KV")
        # 两种动作目标使用同一段在训练集分位数范围内归一化的示教。
        noisy, target, timestep = make_flow(self.model, np.clip(actions_6, -1, 1), self.device, seed)
        # 原生 denoise_step 为本次后缀处理 clone K/V，tensor.clone 保留梯度。
        # 旧版 expert-only 的 velocity 会 deepcopy 冻结缓存，不适用于带计算图的缓存。
        prediction = self.model.denoise_step(prefix_pad_masks=valid, past_key_values=cache,
                                             x_t=noisy, timestep=timestep)
        return velocity_mse(prediction, target)

    def fast_loss(self, inputs, actions_6):
        """低层输入已含标注子任务、状态及 Action 提示。目标为动作 token、终止符、EOS。"""
        if self.fast is None:
            raise ValueError("Set --fast-tokenizer-dir to a local FAST+ snapshot for FAST supervision")
        if np.shape(actions_6) != (self.model.config.chunk_size, 6):
            raise ValueError("FAST and Flow require the same SO-101 action horizon")
        target = [FAST_TOP_ID - i for i in self.fast.encode(actions_6)]
        return self._loss(inputs, target + self._text("|") + [self.eos])

    def _grounding_views(self, camera_keys, masks):
        # LeRobot 将存在的图像按 config.image_features 顺序放在空视角前面。
        if not camera_keys or len(set(camera_keys)) != len(camera_keys):
            raise ValueError("Grounded predictions need the names of the observed cameras")
        views = [key for key in self.model.config.image_features if key in camera_keys]
        if len(views) != len(camera_keys) or len(views) != sum(bool(mask.item()) for mask in masks):
            raise ValueError("Grounding camera names do not match the observed model views")
        # 坐标直接索引预训练 LM head 的 1024 个 loc 行，与 HF 是否为它们注册别名无关。
        if len(self._text("|")) != 1:
            raise ValueError("Grounded output needs a single-token '|' view separator")
        return views

    def _box_target(self, boxes, views):
        # 每个实际视角一组，| 结束该视角；连续两个 | 表示其间视角被标注为无相关目标。
        groups = {view: [] for view in views}
        for box in boxes:
            xmin, ymin, xmax, ymax = box["xyxy"]
            if box["view"] not in groups or not (0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1):
                raise ValueError("Invalid normalized box or unobserved grounding camera")
            groups[box["view"]].append((math.floor(ymin * 1023), math.floor(xmin * 1023),
                                        math.ceil(ymax * 1023), math.ceil(xmax * 1023)))
        target = []
        separator = self._text("|")[0]
        for view in views:
            for coords in sorted(groups[view]):
                target.extend(GEMMA_TEXT_VOCAB + c for c in coords)
            target.append(separator)
        return target

    def semantic_loss(self, images, img_masks, global_goal, subtask, state, boxes=None, camera_keys=None):
        views = self._grounding_views(camera_keys, img_masks) if boxes is not None else []
        inputs = self._goal_inputs(images, img_masks, global_goal, state, views)
        if subtask is not None and (not isinstance(subtask, str) or not subtask.strip()
                                    or subtask.strip().casefold() == COMPLETE.casefold()):
            raise ValueError("Use null for completion and nonempty text for the current subtask")
        target = self._box_target(boxes, views) if boxes is not None else []
        target.extend(self._text(COMPLETE if subtask is None else subtask.strip()) + [self.eos])
        if len(target) > self.model.config.tokenizer_max_length:
            raise ValueError("Subtask target is longer than the inference generation budget")
        return self._loss(inputs, target)

    def predict(self, images, img_masks, global_goal, state):
        return self.predict_result(images, img_masks, global_goal, state)[0]

    @torch.no_grad()
    def predict_result(self, images, img_masks, global_goal, state, camera_keys=None):
        """按观察视角生成可选目标框，随后生成当前子任务。显式完成文本才表示任务结束。"""
        views = self._grounding_views(camera_keys, img_masks) if camera_keys is not None else []
        inputs = self._goal_inputs(images, img_masks, global_goal, state, views)
        hidden, cache, valid = self._observe(inputs, [], use_cache=True)
        if cache is None:
            raise RuntimeError("The semantic language decoder returned no KV cache")
        phrase = []
        boxes = [] if views else None
        view_index, coords = 0, []
        separator = self._text("|")[0] if views else None
        # 自然语言只在 Gemma 文本词表内生成，保留 EOS；不使用示教候选子任务列表。
        excluded = [i for i in self.specials if 0 <= i < GEMMA_TEXT_VOCAB and i != self.eos]
        limit = min(self.model.config.tokenizer_max_length,
                    self.language.config.max_position_embeddings - valid.shape[1] + 1)
        for step in range(limit):
            predicting_box = view_index < len(views)
            vocab_end = GEMMA_TEXT_VOCAB + 1024 if predicting_box else GEMMA_TEXT_VOCAB
            scores = self.head(hidden[:, -1].to(self.head.weight.dtype))[0, :vocab_end].float()
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite semantic token scores")
            if predicting_box:
                # 仅约束输出格式。模型从当前图像学习框数与坐标，分隔符允许零个框。
                k = len(coords)
                lower = 0 if k < 2 else coords[k - 2] + 1
                upper = 1023 if k < 2 else 1024
                locations = scores[GEMMA_TEXT_VOCAB + lower:GEMMA_TEXT_VOCAB + upper]
                best_score, best_index = locations.max(0)
                if k == 0 and scores[separator] >= best_score:
                    token = separator
                    view_index += 1
                else:
                    coordinate = lower + best_index.item()
                    coords.append(coordinate)
                    token = GEMMA_TEXT_VOCAB + coordinate
                    if len(coords) == 4:
                        ymin, xmin, ymax, xmax = coords
                        boxes.append({"view": views[view_index], "xyxy": [xmin/1023, ymin/1023, xmax/1023, ymax/1023]})
                        coords.clear()
            else:
                scores[excluded] = -torch.inf
                token = scores.argmax().item()
                if token == self.eos:
                    text = self.tokenizer.decode(phrase, skip_special_tokens=True).strip()
                    if not text:
                        raise SubtaskDecodeError("Generated an empty subtask")
                    return (None if text == COMPLETE else text), boxes
                phrase.append(token)
            if step + 1 == limit:
                break
            ids = torch.tensor([[token]], device=self.device)
            # 走同一个 embed_prefix 的纯文本分支，复用安装版本对语言 embedding 的缩放。
            embed, _, _ = self.model.embed_prefix([], [], ids, torch.ones_like(ids, dtype=torch.bool))
            valid = F.pad(valid, (0, 1), value=True)
            result = self.language(inputs_embeds=embed, attention_mask=self._mask(valid[:, None, :]),
                                   position_ids=valid.long().sum(1, keepdim=True) - 1,
                                   past_key_values=cache, use_cache=True)
            hidden, cache = result.last_hidden_state, result.past_key_values
        raise SubtaskDecodeError("Generated subtask did not finish with EOS")
