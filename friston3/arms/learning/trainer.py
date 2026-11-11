#!/usr/bin/env python
"""SO-101 π₀.₅ 微调流程，负责采样、梯度累积、优化器更新、验证与断点。

运行 python -m friston3.arms.learning.trainer。run() 调用 data 读取示教样本，
动作阶段用 pi05.make_frozen_prefix、make_flow、velocity 和 flow_loss 计算损失；
语义阶段用 pi05.SemanticPolicy.loss 训练当前子任务文本。joint 阶段的子任务文本 CE、
FAST+ CE 与连续动作 Flow 损失均可更新共享 VLM 的 K/V 适配器。load_training_model
负责训练专用装配，将可训练参数保留在 GPU，冻结的视觉语言骨干可卸载至 CPU。
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import OrderedDict
from importlib import metadata, util
from itertools import groupby
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..models import assets_root, joint, loader, pi05, resolve_asset_path
from ..models.weights import checkpoint_reader
from . import data


class FrozenPrefixCache:
    """LRU of detached prefix storage, bounded in bytes; never serialized.

    The budget bounds retained tensors, including shared-storage views once per
    entry, on the selected storage device. CPU entries retain the original
    dtype and each hit creates an independent execution-device copy. Active
    forward/backward temporaries and CUDA reserved memory are not budgeted.
    """

    def __init__(self, max_bytes, storage_device="cuda"):
        # 构造缓存实例：self是当前对象，self.entries属于此实例，不是全局共享字典。
        self.max_bytes = max_bytes
        self.storage_device = storage_device
        self.entries = OrderedDict()
        # collections.OrderedDict保留可调整的顺序；最早未使用项放前面，方便LRU淘汰。
        self.bytes = self.peak_bytes = self.hits = self.misses = 0

    @staticmethod
    def copy_to(value, device):
        from transformers.cache_utils import DynamicCache

        pad, cache, actions, task = value
        # PI0.5 prefill uses full-attention DynamicLayers. Rebuild them so their
        # device metadata follows the tensors without mutating stored entries.
        if not isinstance(cache, DynamicCache) or any(layer.is_sliding for layer in cache.layers):
            raise ValueError("CPU prefix caching requires a full-attention DynamicCache")
        copied = DynamicCache()
        for index, layer in enumerate(cache.layers):
            copied.update(layer.keys.to(device=device, copy=True),
                          layer.values.to(device=device, copy=True), index)
        return pad.to(device=device, copy=True), copied, actions, task

    @staticmethod
    def storage_bytes(value):
        pad, cache, _, _ = value
        tensors = [pad]
        if hasattr(cache, "layers"):
            for layer in cache.layers:
                tensors.extend(getattr(layer, name, None) for name in ("keys", "values", "cumulative_length"))
        elif hasattr(cache, "key_cache"):
            tensors.extend(cache.key_cache)
            tensors.extend(cache.value_cache)
        else:
            tensors.extend(tensor for pair in cache for tensor in pair)
        seen, total = set(), 0
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.requires_grad or tensor.is_meta:
                raise ValueError("Frozen-prefix cache requires materialized detached tensors")
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
        return total

    def get(self, key, device=None):
        item = self.entries.pop(key, None)
        if item is None:
            self.misses += 1
            return None
        self.entries[key] = item
        self.hits += 1
        return self.copy_to(item[0], device or "cpu") if self.storage_device == "cpu" else item[0]

    def put(self, key, value):
        if not self.max_bytes:
            return
        size = self.storage_bytes(value)
        if size > self.max_bytes:
            return  # An oversized prefix is still usable for this microstep.
        if self.storage_device == "cpu":
            value = self.copy_to(value, "cpu")
            size = self.storage_bytes(value)
            if size > self.max_bytes:
                return
        old = self.entries.pop(key, None)
        if old is not None:
            self.bytes -= old[1]
        while self.entries and self.bytes + size > self.max_bytes:
            _, (_, released) = self.entries.popitem(last=False)
            self.bytes -= released
        self.entries[key] = (value, size)
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)

    def stats(self):
        return {"storage_device": self.storage_device,
                "max_bytes": self.max_bytes, "retained_bytes": self.bytes,
                "peak_retained_bytes": self.peak_bytes, "entries": len(self.entries),
                "hits": self.hits, "misses": self.misses}


def training_prefix(model, device, prefix_cache, root, info, groups, episode_meta,
                    stats, args, episode, start):
    # 【调用方】run中的每个microstep；返回(pad,KV cache,动作标签,task)。
    # prefix_cache是上面的FrozenPrefixCache实例，缓存命中可跳过视频解码与冻结前缀前向。
    # 同一帧条件重复被抽中时，冻结前缀可复用 KV；每次仍重新采样 flow 的噪声和时间。
    # 若未来增加随机图像增强或解冻 PaliGemma，必须同步修改缓存条件/失效规则。
    key = (episode, int(groups[episode][start]["frame_index"]))
    value = prefix_cache.get(key, device)
    if value is None:
        state, actions, images, _, task = data.sample_data(
            root, info, groups, episode_meta, stats, args, episode, start)
        inputs = pi05.prepare_inputs(model, args, device, state, images, task)
        pad, cache = pi05.make_frozen_prefix(model, inputs)
        value = (pad, cache, actions, task)
        prefix_cache.put(key, value)
    return value


def action_sample_loss(model, device, prefix_cache, root, info, groups, episode_meta, stats, args, ep, start):
    pad, cache, actions, task = training_prefix(
        model, device, prefix_cache, root, info, groups, episode_meta, stats, args, ep, start)
    noisy, target, timestep = pi05.make_flow(model, actions, device)
    prediction = pi05.velocity(model, pad, cache, noisy, timestep)
    return pi05.flow_loss(prediction, target), task


def semantic_sample_loss(policy, model, device, root, info, groups, episode_meta, args, ep, start):
    row = groups[ep][start]
    rgb, _ = data.sample_images(root, info, episode_meta[ep], row, args.image_keys)
    images, masks = pi05.prepare_images(model, device, rgb)
    return policy.loss(images, masks, row["global_goal"], row["subtask"]), row["subtask"]


def semantic_frames(groups, episode_ids, *, distinguish_boxes=False):
    """Return every annotated frame, including terminal frames and segments shorter than H."""
    by_label = {}
    for ep in episode_ids:
        for i, row in enumerate(groups[ep]):
            if "subtask" in row:
                label = row["subtask"]
                bucket = (label, "boxes" in row) if distinguish_boxes else label
                by_label.setdefault(bucket, []).append((ep, i))
    if distinguish_boxes:
        # 稀疏框标注独立成组；每一轮都训练少量带框样本，同时保留纯文本样本和完成态。
        by_label = dict(sorted(by_label.items(), key=lambda item: not item[0][1]))
    return by_label


def joint_semantic_inputs(model, device, root, info, groups, episode_meta, stats, args, ep, start):
    state, rgb, goal, subtask = data.sample_subtask_frame(
        root, info, groups, episode_meta, stats, args, ep, start)
    images, masks = pi05.prepare_images(model, device, rgb)
    boxes = groups[ep][start].get("boxes")
    if boxes is not None:
        boxes = [box for box in boxes if box["view"] in args.semantic_image_keys]
    return images, masks, goal, subtask, state, boxes, args.semantic_image_keys if boxes is not None else None


def joint_objectives(policy, model, device, root, info, groups, episode_meta, stats, args, ep, start, sem_ep, sem_i):
    """Yield each graph separately so its backward finishes before constructing the next."""
    semantic = joint_semantic_inputs(model, device, root, info, groups, episode_meta, stats, args, sem_ep, sem_i)
    yield "semantic_ce", policy.semantic_loss(*semantic), 1.0
    del semantic
    state, actions, rgb, _, subtask = data.sample_data(
        root, info, groups, episode_meta, stats, args, ep, start)
    inputs = pi05.prepare_inputs(model, args, device, state, rgb, subtask)
    yield "fast_ce", policy.fast_loss(inputs, actions), 1.0
    if args.flow_weight:
        # 仅用当前观测、状态和标注子任务建立可微前缀。Flow 梯度同时流向动作专家和共享 K/V 适配器。
        yield "flow_mse", policy.flow_loss(inputs, actions), args.flow_weight


def chunk_start_indices(length, horizon, stride):
    # 【调用方】run；纯函数：给定帧数/块长/起点间隔，返回合法起点list，不读文件。
    last = length - horizon
    if last < 0 or min(horizon, stride) < 1:
        raise ValueError("Chunk starts require a positive horizon/stride and a complete window")
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def sample_starts(groups, ids, args, stride):
    """语义 CE 取带标注的当前帧；flow 仅取一个动作标签内能容纳完整 H 的窗口。"""
    pools, short = {}, []
    for ep in ids:
        rows = groups[ep]
        if args.stage == "semantic":
            pools[ep] = list(range(len(rows)))
            continue
        starts = []
        begin = 0
        for label, run in groupby(rows, key=lambda row: data.action_label(row, args.task)):
            length = sum(1 for _ in run)
            if label is not None:
                if length >= args.horizon:
                    starts.extend(begin + i for i in chunk_start_indices(length, args.horizon, stride))
                else:
                    short.append((ep, rows[begin]["frame_index"], length, label))
            begin += length
        if starts:
            pools[ep] = starts
    if short:
        data.event("SKIPPED_SHORT_ACTION_SEGMENTS", {"episode_start_length_label": short, "horizon": args.horizon})
    if not pools:
        raise ValueError(f"No complete action windows in episodes {ids} for horizon {args.horizon}")
    return pools


def configure_training(args, resume_config):
    # 【调用方】run；从旧checkpoint或CLI确定梯度累积/起点间隔，写回args属性。
    # 本函数没有显式return，因此Python返回None；调用方依靠args被更新这一副作用。
    saved = (resume_config or {}).get("training_contract", {})
    if not isinstance(saved, dict) or saved.get("micro_batch_size", 1) != 1:
        raise ValueError("Unsupported saved training micro-batch contract")
    contract = {"micro_batch_size": 1}
    for key in ("gradient_accumulation_steps", "chunk_start_stride"):
        previous = saved.get(key, 1)
        supplied = getattr(args, key, None)
        value = previous if supplied is None else supplied
        if type(previous) is not int or previous < 1 or type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
        if resume_config and value != previous:
            raise ValueError(f"Resume {key} differs; use --init-from for a new training contract")
        setattr(args, key, value)
        contract[key] = value
    if args.stage == "joint":
        if resume_config and args.flow_weight != saved.get("flow_weight"):
            raise ValueError("Resume flow weight differs; use --init-from to change the objective")
        contract.update(flow_weight=args.flow_weight, semantic_weight=1.0, fast_weight=1.0,
                        flow_gradients="action_expert_and_shared_vlm_kv",
                        semantic_sampling="round_robin_label_and_grounding", action_sampling="uniform_episode_then_window")
    args.training_contract = contract


def episode_frame_range(value):
    # 【调用方】argparse的type转换；例如"3=20:90"→("3",[20,90])，右端点不含。
    try:
        episode, bounds = value.split("=", 1)
        start, stop = bounds.split(":", 1)
        episode, start, stop = int(episode), int(start), int(stop)
        if episode < 0 or start < 0 or stop <= start:
            raise ValueError
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected EP=START:STOP with 0 <= START < STOP") from error
    return str(episode), [start, stop]


def select_episode_frames(groups, selected_ids, ranges):
    """Select original row intervals without changing frame IDs or video times."""
    # 【调用方】run；返回选定episode的新字典，里面的row仍保留原frame_index/timestamp。
    if not isinstance(ranges, dict):
        raise ValueError("Episode frame ranges must be an episode-to-[start, stop] mapping")
    for key, bounds in ranges.items():
        if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key:
            raise ValueError(f"Invalid frame-range episode ID: {key!r}")
        episode = int(key)
        if episode not in selected_ids:
            raise ValueError(f"Frame range episode {episode} is outside the selected train/validation split")
        if (not isinstance(bounds, list) or len(bounds) != 2
                or any(type(value) is not int for value in bounds)):
            raise ValueError(f"Invalid frame range for episode {episode}: {bounds!r}")
        start, stop = bounds
        if not 0 <= start < stop <= len(groups[episode]):
            raise ValueError(f"Frame range {episode}={start}:{stop} exceeds its {len(groups[episode])} original rows")
    return {episode: groups[episode][slice(*ranges[str(episode)])] if str(episode) in ranges
            else groups[episode] for episode in selected_ids}


def choose_val_samples(pools, count, seed, sampling="uniform"):
    # 【调用方】configure_validation；返回[(episode,窗口起点),...]，不生成模型动作。
    rng = random.Random(seed)
    choices = []
    valid = list(pools)
    if sampling == "random":
        for i in range(count):
            ep = valid[i % len(valid)]
            choices.append((ep, rng.choice(pools[ep])))
        return choices
    minimum = sum(min(2, len(pools[ep])) for ep in valid)
    if count < minimum:
        raise ValueError(f"Uniform validation needs at least {minimum} samples to cover each episode's endpoints")
    counts = {ep: min(2, len(pools[ep])) for ep in valid}
    for i in range(count - minimum):
        counts[valid[i % len(valid)]] += 1
    for ep in valid:
        indices = np.linspace(0, len(pools[ep]) - 1, counts[ep]).round().astype(int)
        choices.extend((ep, pools[ep][index]) for index in indices)
    return choices


def configure_validation(args, groups, pools, resume_config):
    """Keep resumed validation coordinates fixed, including pre-contract checkpoints."""
    # 【调用方】run；固定验证窗口、seed和生成配置，并保存到args.validation_contract。
    saved = (resume_config or {}).get("validation_contract")
    if saved is None and resume_config:
        state_path = args.resume / "training_state.pt"
        if state_path.is_file():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            previous = next((entry["validation"] for entry in reversed(state.get("history", []))
                             if entry.get("validation", {}).get("samples")), None)
            if previous:
                # Old checkpoints recorded windows but not their CLI noise seed.
                # Accept an explicit seed for that one-time migration; otherwise
                # retain the historical default and record the assumption.
                saved = {"sampling": "random", "eval_seed": 9000 if args.eval_seed is None else args.eval_seed,
                         "samples": [{"episode": row["episode"], "frame_index": row["frame_index"]}
                                     for row in previous["samples"]],
                         "source": "legacy_history",
                         "eval_seed_provenance": ("assumed_legacy_default" if args.eval_seed is None
                                                  else "provided_for_legacy")}
    if saved is not None:
        if args.validation_sampling is not None and args.validation_sampling != saved["sampling"]:
            raise ValueError("Resume validation sampling differs from the saved contract")
        if args.validation_samples is not None and args.validation_samples != len(saved["samples"]):
            raise ValueError("Resume validation sample count differs from the saved contract")
        if args.eval_seed is not None and args.eval_seed != saved["eval_seed"]:
            raise ValueError("Resume validation noise seed differs from the saved contract")
        samples = []
        for item in saved["samples"]:
            ep, frame = item["episode"], item["frame_index"]
            if ep not in pools:
                raise ValueError("Saved validation sample is outside the held-out split")
            positions = {int(row["frame_index"]): i for i, row in enumerate(groups[ep])}
            start = positions.get(frame)
            if start not in pools[ep]:
                raise ValueError("Saved validation sample no longer supplies a valid frame/window")
            samples.append((ep, start))
        if not samples:
            raise ValueError("Saved validation sample list is empty")
        args.validation_sampling, args.eval_seed = saved["sampling"], saved["eval_seed"]
        contract = dict(saved)
    else:
        args.validation_sampling = args.validation_sampling or ("random" if resume_config else "uniform")
        args.eval_seed = 9000 if args.eval_seed is None else args.eval_seed
        count = args.validation_samples or max(10, 2 * len(pools))
        samples = choose_val_samples(pools, count, args.seed + 1000, args.validation_sampling)
        contract = {"sampling": args.validation_sampling, "sample_seed": args.seed + 1000,
                    "eval_seed": args.eval_seed,
                    "samples": [{"episode": ep, "frame_index": int(groups[ep][start]["frame_index"])}
                                for ep, start in samples]}
    args.validation_samples = len(samples)
    args.eval_actions = bool(contract.get("eval_actions", False) if args.eval_actions is None else args.eval_actions)
    args.eval_action_steps = args.eval_action_steps or contract.get("action_num_steps", 10)
    if (saved and saved.get("eval_actions") and args.eval_actions
            and args.eval_action_steps != saved.get("action_num_steps", 10)):
        raise ValueError("Resume generated-action integration steps differ from the saved contract")
    contract.update(eval_actions=args.eval_actions, action_num_steps=args.eval_action_steps)
    args.validation_contract = contract
    return samples


def joint_validation_frames(args, groups, val_ids, resume_config):
    """Keep held-out high-level frame coordinates independent of action horizon and boundaries."""
    saved = (resume_config or {}).get("validation_contract", {}).get("semantic_samples")
    by_label = semantic_frames(groups, val_ids, distinguish_boxes=True)
    if not by_label:
        raise ValueError("Joint validation needs annotated subtask frames in the held-out episodes")
    if saved is None:
        rng = random.Random(args.eval_seed)
        pools = list(by_label.values())
        samples = [rng.choice(pools[i % len(pools)]) for i in range(max(len(pools), args.validation_samples))]
    else:
        positions = {(ep, int(groups[ep][i]["frame_index"])): (ep, i)
                     for frames in by_label.values() for ep, i in frames}
        try:
            samples = [positions[item["episode"], item["frame_index"]] for item in saved]
        except KeyError as error:
            raise ValueError("Saved high-level validation frames differ from the held-out annotations") from error
        if not samples:
            raise ValueError("Saved high-level validation sample list is empty")
    args.validation_contract["semantic_samples"] = [
        {"episode": ep, "frame_index": int(groups[ep][i]["frame_index"])} for ep, i in samples]
    if args.eval_subtasks is None:
        args.eval_subtasks = args.validation_contract.get("eval_subtasks", False)
    args.validation_contract["eval_subtasks"] = args.eval_subtasks
    return samples


def configure_cameras(args, info, model_config, resume):
    available = [key for key in model_config["input_features"]
                 if key.startswith("observation.images.") and key in info["features"]]
    saved = resume or {}
    previous_low = saved.get("image_keys") or ([saved["image_key"]] if "image_key" in saved else None)
    low = args.image_keys or previous_low or available
    selections = [("image_keys", low)]
    if args.stage == "joint":
        previous_high = saved.get("semantic_image_keys")
        if previous_high is None and previous_low is not None:
            previous_high = [key for key in available if key in previous_low]
        high = args.semantic_image_keys or previous_high or low
        if args.image_keys is None and previous_low is None:
            # 只有命名明确的后向视角默认专供高层；任意物理相机可以用显式选项分配。
            low = [key for key in low if not {"rear", "back", "backward"}.intersection(
                key.rsplit(".", 1)[-1].lower().replace("-", "_").split("_"))] or low
        selections = [("image_keys", low), ("semantic_image_keys", high)]
    for field, keys in selections:
        if not keys or len(set(keys)) != len(keys) or not set(keys).issubset(available):
            raise ValueError(f"{field} must select distinct dataset/model cameras from {available}; got {keys}")
        # LeRobot 的图像预处理按模型配置排序；框的视角标号与视觉 token 排列必须一致。
        if field == "semantic_image_keys":
            keys = [key for key in available if key in keys]
        setattr(args, field, keys)
    if resume and previous_low != args.image_keys:
        raise ValueError("Resume camera keys differ; continue with the checkpoint's --image-keys")
    if resume and args.stage == "joint" and previous_high != args.semantic_image_keys:
        raise ValueError("Resume semantic camera keys differ; use --init-from for new camera roles")


def action_metrics(predicted, actual, state, stats):
    """Report absolute action error against commands, plus the hold-current baseline."""
    # evaluate 调用：生成整块后才计算这一指标，与随机 t 上的 flow loss 是两个问题。
    # 分关节输出，避免把五个角度与夹爪百分比混成一个难解释的“总物理误差”。
    predicted = np.asarray(predicted, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    lo, hi = (np.asarray(stats[data.ACTION][key], dtype=np.float32) for key in ("q01", "q99"))
    physical = (predicted + 1) * (hi - lo) / 2 + lo
    hold = np.broadcast_to(np.asarray(state, dtype=np.float32), actual.shape)
    # hold 是 H 步都保持当前实测状态；用于识别模型是否只学会了近似不动。
    errors, hold_errors = np.abs(physical - actual), np.abs(hold - actual)
    return {"mae_per_joint": errors.mean(axis=0).tolist(),
            "hold_mae_per_joint": hold_errors.mean(axis=0).tolist(),
            "first10_mae_per_joint": errors[:10].mean(axis=0).tolist(),
            "first10_hold_mae_per_joint": hold_errors[:10].mean(axis=0).tolist(),
            "normalized_mae_per_joint": np.abs(predicted - data.normalize(actual, stats[data.ACTION])).mean(axis=0).tolist(),
            "normalized_hold_mae_per_joint": np.abs(data.normalize(hold, stats[data.ACTION])
                                                     - data.normalize(actual, stats[data.ACTION])).mean(axis=0).tolist()}


def evaluate(model, loader, args, device, root, info, groups, episode_meta,
             stats, samples, joint_policy=None):
    # 【调用方】run到达eval间隔时；samples是预先固定的(episode,start)列表。
    # 返回flow loss统计字典；可选加入生成动作MAE，不反向传播、不optimizer.step、不控制硬件。
    values, details = [], []
    for index, (episode, start) in enumerate(samples):
        state, actions, images, videos, task = data.sample_data(
            root, info, groups, episode_meta, stats, args, episode, start
        )
        inputs = pi05.prepare_inputs(model, args, device, state, images, task)
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            # fork_rng进入时保存随机状态、退出时还原；验证不会改变之后训练使用的随机流。
            torch.manual_seed(args.eval_seed + index)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.eval_seed + index)
            with torch.no_grad():
                if joint_policy is not None:
                    value = joint_policy.flow_loss(inputs, actions, seed=args.eval_seed + index).item()
                    fast_ce = float(joint_policy.fast_loss(inputs, actions).item())
                else:
                    value = pi05.compute_loss(model, inputs, actions, device, seed=args.eval_seed + index)[0].item()
                generated = None
                if args.eval_actions:
                    # A separate fixed seed prevents flow-noise sampling from changing
                    # generation and fork_rng keeps evaluation out of training RNG state.
                    torch.manual_seed(args.eval_seed + index)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(args.eval_seed + index)
                    prediction = model.sample_actions(**inputs, num_steps=args.eval_action_steps)
                    # **inputs把字典展开为关键字参数，例如images=...、tokens=...。
                    # sample_actions内部多次denoise_step，输出整块[1,H,32]，与单次flow loss不同。
                    generated = action_metrics(
                        prediction[0, :, :6].float().cpu().numpy(),
                        [row[data.ACTION] for row in groups[episode][start:start + args.horizon]],
                        groups[episode][start][data.STATE], stats)
                    del prediction
        values.append(value)
        details.append({
            "episode": episode, "frame_index": int(groups[episode][start]["frame_index"]), "loss": value,
            "task": task,
            "video_time_delta_seconds": {key: value["absolute_difference_s"] for key, value in videos.items()},
        })
        if generated is not None:
            details[-1]["generated_actions"] = generated
        if joint_policy is not None:
            details[-1]["fast_ce"] = fast_ce
        del inputs
    result = {
        "mean_flow_matching_mse_6d": float(np.mean(values)),
        "std_flow_matching_mse_6d": float(np.std(values)),
        "sample_count": len(values), "samples": details,
    }
    if joint_policy is not None:
        result["mean_fast_ce"] = float(np.mean([row["fast_ce"] for row in details]))
    if args.eval_actions:
        result["generated_actions"] = {
            "joint_order": data.JOINTS, "units": ["degrees"] * 5 + ["percent"],
            "num_steps": args.eval_action_steps, "rtc": False,
            **{key: np.mean([row["generated_actions"][key] for row in details], axis=0).tolist()
               for key in details[0]["generated_actions"]}}
    return result


@torch.no_grad()
def evaluate_subtasks(policy, model, device, root, info, groups, episode_meta, stats, args, samples):
    ce_values, predictions = [], []
    for ep, i in samples:
        images, masks, goal, label, state, boxes, camera_keys = joint_semantic_inputs(
            model, device, root, info, groups, episode_meta, stats, args, ep, i)
        ce_values.append(policy.semantic_loss(images, masks, goal, label, state, boxes, camera_keys).item())
        if args.eval_subtasks:
            record = {"episode": ep, "frame_index": int(groups[ep][i]["frame_index"]),
                      "global_goal": goal, "reference": label}
            if boxes is not None:
                record["reference_boxes"] = boxes
            try:
                # 使用与部署相同的提示。验证帧恰好有无人工框只影响监督和比较。
                inference_views = args.semantic_image_keys if args.predicts_boxes else None
                prediction, detected = policy.predict_result(images, masks, goal, state, inference_views)
                record.update(prediction=prediction, exact_match=prediction == label)
                if detected is not None:
                    record["predicted_boxes"] = detected
            except joint.SubtaskDecodeError as error:
                record.update(decode_error=str(error), exact_match=False)
            predictions.append(record)
    metrics = {"mean_semantic_ce": float(np.mean(ce_values)), "semantic_sample_count": len(samples)}
    if predictions:
        metrics.update(subtask_predictions=predictions,
                       subtask_exact_match=float(np.mean([row["exact_match"] for row in predictions])))
    return metrics


def backend_provenance() -> dict:
    """记录断点使用的包版本和可读的 LeRobot 源码指纹，供实验复现。"""
    versions = {}
    for name in ("lerobot", "torch", "transformers", "accelerate"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None

    try:
        spec = util.find_spec("lerobot")
        roots = getattr(spec, "submodule_search_locations", None) or ()
    except (ImportError, ValueError, AttributeError):
        roots = ()
    source_files = (
        "policies/pi05/modeling_pi05.py", "policies/pi05/processor_pi05.py",
        "policies/pi05/configuration_pi05.py", "policies/pi_gemma.py",
        "policies/common/flow_matching.py", "policies/common/vla_utils.py",
    )
    hashes = {}
    for relative in source_files:
        for root in roots:
            path = Path(root) / relative
            if path.is_file():
                try:
                    hashes[relative] = data.fingerprint(path)
                except OSError:
                    pass
                break

    try:
        direct_url = metadata.distribution("lerobot").read_text("direct_url.json")
        origin = json.loads(direct_url) if direct_url else {}
        commit = origin.get("vcs_info", {}).get("commit_id")
    except (metadata.PackageNotFoundError, OSError, ValueError, AttributeError):
        commit = None
    return {
        "status": "available" if source_files[0] in hashes else "unavailable",
        "versions": versions, "recorded_commit": commit, "source_sha256": hashes,
    }


def save_checkpoint(trainable_model, optimizer, args, stats, train_ids, val_ids, step, output, history):
    # 【调用方】run；输入训练对象与元数据，输出新checkpoint目录的字符串路径。
    # 下游是safetensors.save_file、torch.save和data.write_json；会写磁盘，不保存完整base。
    from safetensors.torch import save_file

    directory = output / f"checkpoint_step_{step:06d}"
    # Path的/运算拼路径；f字符串中:06d把整数格式化成6位、不足补0，例如000080。
    config_path = directory / "adapter_config.json"
    adapter_format = {"action": "friston_pi05_expert_lora_v1", "semantic": pi05.SEMANTIC_FORMAT,
                      "joint": joint.JOINT_FORMAT}[args.stage]
    if config_path.is_file() and data.read_json(config_path).get("format") != adapter_format:
        raise ValueError(f"Checkpoint directory already belongs to another training stage: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    # 动作阶段保存 LoRA 和投影层，语义阶段保存文本适配器。
    params = pi05.trainable_dict(trainable_model)
    save_file(
        {k: p.detach().cpu().contiguous().clone() for k, p in params.items()},
        # 每个Parameter p先detach切断图，再cpu移动到内存；contiguous保证连续布局。
        # clone另复制一份，防止保存张量共享存储；k保留参数完整名字供加载时对应。
        str(directory / "trainable.safetensors"),
    )
    # 仅权重可用于推理；完整续训还需要 AdamW 的动量、步数和各随机数生成器状态。
    torch.save(optimizer.state_dict(), directory / "optimizer.pt")
    # optimizer.state_dict()来自PyTorch，导出参数组和Adam动量/步数；它不是模型权重字典。
    training_state = {
        "global_step": step,
        "global_micro_step": step * args.gradient_accumulation_steps,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "history": history,
        "sampler_rng_state": getattr(args, "_sampler_rng_state", None),
    }
    torch.save(training_state, directory / "training_state.pt")
    config = {
        "format": adapter_format,
        "base_model_dir": str(args.model_dir),
        "base_config_sha256": data.fingerprint(args.model_dir / "config.json"),
        # 记录当前依赖源码，辅助复现实验；不作为拒绝旧 checkpoint 的新条件。
        "model_implementation": backend_provenance(),
        **loader.loader_metadata(),
        # ** 将返回字典的键值展开进 config：版本判断兼容性，源码 SHA256 记录实验来源。
        "rank": args.rank, "alpha": args.alpha,
        "initialized_from": args.initialized_from,
        "training_episodes": train_ids, "validation_episodes": val_ids,
        "episode_frames": args.episode_frames,
        "validation_contract": args.validation_contract,
        "training_contract": args.training_contract,
        "micro_steps_completed": step * args.gradient_accumulation_steps,
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "image_key": args.image_keys[0], "image_keys": args.image_keys,
        "dataset_root": str(args.root), "dataset_fps": args.dataset_fps,
        "steps_completed": step, "batch_size": 1,
        "effective_batch_size": args.gradient_accumulation_steps, "learning_rate": args.lr,
        "not_standard_peft_or_lerobot_checkpoint": True,
    }
    if args.subtasks_sha256:
        config["subtasks_sha256"] = args.subtasks_sha256
    observed = {sample["task"] for entry in history for sample in entry.get("micro_samples", [])}
    if args.stage == "semantic":
        config.update(tokenizer_sha256=args.tokenizer_sha256,
                      subtasks=sorted(observed - {None}), has_completion=None in observed)
    else:
        tasks = sorted(observed) if args.subtasks_sha256 else args.tasks
        config.update(training_scope="random_episode_random_chunk",
                      task=tasks[0] if len(tasks) == 1 else "", tasks=tasks,
                      target_modules=args.target_modules, full_training_modules=list(pi05.PROJECTIONS),
                      joint_order=data.JOINTS, model_slots=list(range(6)), supervised_action_dims=6,
                      horizon=args.horizon, max_action_dim=args.max_action_dim, statistics=stats,
                      dataset_statistics=args.dataset_statistics, normalization_source=args.normalization_source,
                      prefix_cache_mib=args.prefix_cache_mib, prefix_cache_device=args.prefix_cache_device)
        if args.stage == "joint":
            # A discrete-only run samples action windows for FAST; these are not expert updates.
            seen = {sample["semantic_subtask"] for entry in history for sample in entry["micro_samples"]}
            seen.update(args.initial_semantic_subtasks)
            if args.initial_has_completion:
                seen.add(None)
            trained_tasks = sorted(set(args.initial_action_tasks) | (observed if args.flow_weight else set()))
            config.update(tasks=trained_tasks, task=trained_tasks[0] if len(trained_tasks) == 1 else "")
            config.update(tokenizer_sha256=args.tokenizer_sha256, fast_metadata=args.fast_metadata,
                          tokenizer_dir=str(args.tokenizer_dir), fast_tokenizer_dir=str(args.fast_tokenizer_dir),
                          subtasks=sorted(seen - {None}),
                          semantic_image_keys=args.semantic_image_keys, grounding_sha256=args.grounding_sha256,
                          predicts_boxes=args.predicts_boxes,
                          has_completion=None in seen, has_action_training=args.has_action_training)
    data.write_json(config_path, config)
    return str(directory)


def run(args):
    # 【调用关系】main → 本函数 → 数据读取/模型加载/训练数值函数/evaluate/save_checkpoint。
    # args由parse_args创建；本函数负责整个训练生命周期，返回None，成果是checkpoint。
    # 读本函数时分四段：小型数据契约、加载模型、恢复或初始化、循环更新/验证/保存。
    # 先验证小型元数据，再加载大模型；训练/验证按 episode 切分，避免相邻帧泄漏。
    torch.set_num_threads(args.cpu_threads)
    # PyTorch单个CPU算子的线程数，不是几个训练worker，也不是GPU数量。
    root = args.root.resolve()
    # args.root已由argparse的type=Path转成路径对象；resolve得到绝对路径。
    info, groups, episode_meta = data.load_rows(root)
    semantic_stage = args.stage == "semantic"
    joint_stage = args.stage == "joint"
    args.subtasks_sha256 = info.get("subtasks_sha256")
    args.grounding_sha256 = info.get("grounding_sha256")
    if semantic_stage or joint_stage:
        groups = {ep: rows for ep, rows in groups.items() if "subtask" in rows[0]}
        if not groups:
            raise ValueError(f"{args.stage} training requires current-frame labels in <root>/meta/subtasks.json")
    # 三元组解包：info为数据说明，groups按episode分帧，episode_meta负责视频定位。
    resume_config = data.read_json(args.resume / "adapter_config.json") if args.resume else None
    # 条件表达式“真分支 if 条件 else 假分支”；没传--resume时不读取该文件。
    source_config = (resume_config if args.resume else
                     data.read_json(args.init_from / "adapter_config.json") if args.init_from else None)
    configure_training(args, resume_config)
    if args.train_episodes is None:
        args.train_episodes = (resume_config or {}).get("training_episodes")
    train_ids = None if args.train_episodes is None else sorted(set(args.train_episodes))
    # set去重、sorted排序得到list；None表示尚未选择，空list表示选择为空，含义不同。
    if train_ids is not None:
        if not train_ids:
            raise ValueError("Training split is empty")
        if not set(train_ids).issubset(groups):
            raise ValueError(f"Missing training episodes {train_ids}; found {sorted(groups)}")
    if args.val_episodes is None:
        args.val_episodes = (resume_config or {}).get("validation_episodes")
    if args.val_episodes is None:
        episode_ids = sorted(set(groups) - set(train_ids or []))
        if not episode_ids:
            raise ValueError("No episodes remain for validation outside the training split")
        args.val_episodes = episode_ids[-max(1, len(episode_ids) // 5):]
    val_ids = sorted(set(args.val_episodes))
    if not val_ids:
        raise ValueError("Validation split is empty")
    if not set(val_ids).issubset(groups):
        raise ValueError(f"Missing validation episodes {val_ids}; found {sorted(groups)}")
    if train_ids is None:
        train_ids = sorted(set(groups) - set(val_ids))
        # set(dict)取得所有键；集合差集减去验证episode，剩下的用于训练。
    if not train_ids:
        raise ValueError("Training split is empty")
    overlap = sorted(set(train_ids) & set(val_ids))
    if overlap:
        raise ValueError(f"Training and validation episodes overlap: {overlap}")
    args.train_episodes, args.val_episodes = train_ids, val_ids
    if args.episode_frames is None:
        args.episode_frames = (resume_config or {}).get("episode_frames", {})
    groups = select_episode_frames(groups, train_ids + val_ids, args.episode_frames)
    if resume_config and args.episode_frames != resume_config.get("episode_frames", {}):
        raise ValueError("Resume checkpoint frame ranges differ")
    stats = None
    if semantic_stage or joint_stage:
        args.tokenizer_sha256 = data.fingerprint(args.tokenizer_dir / "tokenizer.json")
        if source_config and source_config.get("format") in (pi05.SEMANTIC_FORMAT, joint.JOINT_FORMAT):
            loader.validate_loader_compatibility(source_config)
            if (source_config.get("base_config_sha256") != data.fingerprint(args.model_dir / "config.json")
                    or source_config.get("tokenizer_sha256") != args.tokenizer_sha256):
                raise ValueError("Adapter base config or text tokenizer differs")
    if not semantic_stage:
        args.dataset_statistics = data.training_stats(groups, train_ids, require_nonzero=source_config is None)
        stats = (pi05.adapter_statistics(source_config, args.model_dir, args.rank, args.alpha)
                 if source_config is not None else args.dataset_statistics)
    if joint_stage:
        source = source_config or {}
        args.initial_semantic_subtasks = source.get("subtasks", [])
        source_has_expert = (source.get("has_action_training", False)
                             or (source.get("format") == "friston_pi05_expert_lora_v1"
                                 and source.get("steps_completed", 0) > 0))
        args.initial_action_tasks = list(source.get("tasks") or ([source["task"]] if source.get("task") else []))
        if not source_has_expert:
            args.initial_action_tasks = []
        args.initial_has_completion = source.get("has_completion", False)
        args.predicts_boxes = source.get("predicts_boxes", False)
        args.has_action_training = args.flow_weight > 0 or source_has_expert
    args.initialized_from = (str(args.init_from.resolve()) if args.init_from else
                             (resume_config or {}).get("initialized_from"))
    args.normalization_source = ("initialized_adapter" if args.init_from else
                                 (resume_config or {}).get("normalization_source", "training_episodes_only"))
    raw_config = data.read_json(args.model_dir / "config.json")
    args.horizon = int(raw_config["chunk_size"])
    args.max_action_dim = int(raw_config["max_action_dim"])
    # 基础config决定H和D；不是根据SO-101只有6关节就直接把网络宽度改成6。
    start_pools = sample_starts(groups, train_ids, args, args.chunk_start_stride)
    val_pools = sample_starts(groups, val_ids, args, 1)
    configure_cameras(args, info, raw_config, resume_config)
    # 高层轮换监督标签，每个周期都会抽到仅占少量帧的完成态；低层维持原 episode 随机采样。
    if semantic_stage or joint_stage:
        by_label = semantic_frames(groups, train_ids, distinguish_boxes=joint_stage)
        semantic_pools = list(by_label.values())
        semantic_labels = {label for label, _ in by_label} if joint_stage else set(by_label)
        if not any(label is not None for label in semantic_labels):
            raise ValueError("Training split has no annotated action subtask")
    if semantic_stage:
        args.tasks = sorted(label for label in by_label if label is not None)
        if None not in by_label:
            raise ValueError("Semantic training split needs action subtask frames and explicitly observed completion frames")
    else:
        args.tasks = sorted({data.action_label(groups[ep][i], args.task) for ep, starts in start_pools.items() for i in starts})
    args.dataset_fps = float(info["fps"])
    if resume_config:
        if resume_config.get("subtasks_sha256") != args.subtasks_sha256:
            raise ValueError("Resume subtask annotations differ; use --init-from for a new annotation set")
        if joint_stage and resume_config.get("grounding_sha256") != args.grounding_sha256:
            raise ValueError("Resume grounding annotations differ; use --init-from for new box supervision")
        if not semantic_stage and resume_config.get("dataset_statistics", resume_config.get("statistics")) != args.dataset_statistics:
            raise ValueError("Resume dataset statistics differ from the saved checkpoint")
        if not args.subtasks_sha256:
            saved_tasks = sorted(resume_config.get("tasks") or [resume_config.get("task")])
            # 旧断点把验证标签也写进 tasks；继续接受那种元数据。
            all_tasks = sorted(set(args.tasks) | {data.action_label(groups[ep][i], args.task)
                                                  for ep, starts in val_pools.items() for i in starts})
            if saved_tasks not in (args.tasks, all_tasks):
                raise ValueError("Resume dataset tasks differ from the saved checkpoint")
        if resume_config.get("training_episodes") != train_ids or resume_config.get("validation_episodes") != val_ids:
            raise ValueError("Resume checkpoint train/validation split differs")

    val_samples = configure_validation(args, groups, val_pools, resume_config)
    if joint_stage:
        semantic_val_samples = joint_validation_frames(args, groups, val_ids, resume_config)
    data.event("TRAIN_DATA_OK", {
        "train_episodes": train_ids, "validation_episodes": val_ids,
        "episode_frames": args.episode_frames,
        "train_frames": sum(len(groups[e]) for e in train_ids),
        "validation_frames": sum(len(groups[e]) for e in val_ids),
        "stage": args.stage,
        **({} if semantic_stage else {"horizon": args.horizon, "supervised_action_dims": 6,
                                     "statistics_scope": args.normalization_source}),
        "image_keys": args.image_keys, "tasks": args.tasks,
        **({"semantic_image_keys": args.semantic_image_keys} if joint_stage else {}),
        "validation_contract": args.validation_contract,
        "training_contract": args.training_contract,
        "chunk_start_counts": {ep: len(starts) for ep, starts in start_pools.items()},
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
    })

    report = {
        "status": "STARTED", "scope": "multi_episode_random_chunk_lora_finetune",
        "robot_connection_opened": False, "actions_executed": False,
        "base_checkpoint_modified": False, "dataset_modified": False,
        "dataset_root": str(root), "train_episodes": train_ids,
        "validation_episodes": val_ids, "episode_frames": args.episode_frames, "training_stats": stats,
        "steps_requested": args.steps, "history": [],
    }
    model, device = load_training_model(args, loader, report)
    # 返回PI05Pytorch实例及torch.device；到这里才加载大模型，前面的元数据处理不需要GPU。
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    semantic_policy = None
    joint_policy = None
    if semantic_stage:
        semantic_policy = pi05.SemanticPolicy(model, pi05.tokenizer_for(args), args.rank, args.alpha, device)
        semantic_policy.set_candidates(args.tasks)
        trainable_model = semantic_policy.adapter
        targets = []
    else:
        targets = pi05.attach_trainables(model, args.rank, args.alpha)
        trainable_model = model
        if joint_stage:
            # Older action checkpoints contain only expert tensors. Load before registering
            # the shared VLM adapter so the existing strict key check stays meaningful.
            if args.init_from and source_config["format"] == "friston_pi05_expert_lora_v1":
                if targets != source_config.get("target_modules"):
                    raise ValueError("Source action adapter targets differ")
                pi05.load_adapter_weights(pi05.trainable_dict(model), args.init_from)
            joint_policy = joint.JointPolicy(model, pi05.tokenizer_for(args), args.rank, args.alpha,
                                            device, fast_dir=args.fast_tokenizer_dir)
            args.fast_metadata = joint_policy.fast_metadata
            if (source_config and source_config.get("format") == joint.JOINT_FORMAT
                    and source_config.get("fast_metadata") != args.fast_metadata):
                raise ValueError("FAST tokenizer or action-token mapping differs from the joint checkpoint")
    # 原地给model安装LoRA并设requires_grad；返回被替换层名字list，用于checkpoint核对。
    args.target_modules = targets
    params = pi05.trainable_dict(trainable_model)
    # params为{参数名: nn.Parameter}，仅含requires_grad=True的参数，不含全部基础权重。
    if args.prefix_cache_mib and any(p.requires_grad for p in model.paligemma_with_expert.paligemma.parameters()):
        raise ValueError("Prefix caching requires a frozen PaliGemma backbone")
    if source_config is not None:
        if not semantic_stage and targets != source_config.get("target_modules"):
            raise ValueError("Adapter target modules differ from the current model")
        if not (joint_stage and source_config["format"] == "friston_pi05_expert_lora_v1"):
            pi05.load_adapter_weights(params, args.resume or args.init_from)
        if args.init_from:
            data.event("INITIALIZED_FROM_ADAPTER", {
                "checkpoint": args.initialized_from, "optimizer_state_restored": False,
                "rng_state_restored": False, "normalization_source": args.normalization_source,
            })
    data.event("TRAINABLE_READY", {
        "parameters": sum(p.numel() for p in params.values()),
        "tensors": len(params), "lora_modules": len(targets),
        "rank": args.rank, "alpha": args.alpha,
    })
    # optimizer 只接收 requires_grad=True 的参数；冻结权重不创建 Adam 动量。
    optimizer = torch.optim.AdamW(
        list(params.values()), lr=args.lr, weight_decay=0.0, foreach=False
    )
    # values()取得Parameter对象而非名字；list把字典视图变成参数序列。
    # AdamW持有参数引用，在step时维护各参数的动量；当前weight_decay=0。
    # foreach=False选择逐张量更新，避免批量临时缓冲占更多内存。
    start_step = 0
    history = []
    if args.resume is not None:
        resume_dir = args.resume.resolve()
        optimizer_state = torch.load(resume_dir / "optimizer.pt", map_location="cpu", weights_only=False)
        # torch.load恢复保存的Python/Tensor结构；map_location使读入张量先落到CPU。
        optimizer.load_state_dict(optimizer_state)
        # 把Adam历史状态装回optimizer；仅加载模型权重不能恢复这些动量。
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        state_path = resume_dir / "training_state.pt"
        if state_path.is_file():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            # state是训练状态dict，不是每帧observation.state，也不是模型state_dict。
            start_step = int(state["global_step"])
            expected_micro = start_step * args.gradient_accumulation_steps
            if (state.get("global_micro_step", expected_micro) != expected_micro
                    or resume_config.get("micro_steps_completed", expected_micro) != expected_micro):
                raise ValueError("Resume microstep counter differs from the training contract")
            history = list(state.get("history", []))
            random.setstate(state["python_rng_state"])
            np.random.set_state(state["numpy_rng_state"])
            torch.set_rng_state(state["torch_rng_state"])
            if torch.cuda.is_available() and state.get("cuda_rng_states"):
                torch.cuda.set_rng_state_all(state["cuda_rng_states"])
            sampler_state = state.get("sampler_rng_state")
            if sampler_state is None:
                raise ValueError("Resume checkpoint has no sampler RNG state")
            rng = random.Random()
            rng.setstate(sampler_state)
            # rng是独立random.Random实例；setstate恢复随机流位置，继续原窗口抽样序列。
            rng_exact = True
        else:
            # The original Step-20 proof checkpoint predates RNG-state saving.
            # Restore its learned weights and AdamW moments; continue with a new,
            # reproducible sampler stream rather than pretending exact replay.
            start_step = int(resume_config.get("steps_completed", 0))
            history = []
            random.seed(args.seed + start_step)
            np.random.seed(args.seed + start_step)
            torch.manual_seed(args.seed + start_step)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + start_step)
            rng = random.Random(args.seed + start_step)
            rng_exact = False
        data.event("RESUME_OK", {
            "checkpoint": str(resume_dir),
            "global_step": start_step,
            "optimizer_state_restored": True,
            "rng_state_restored": rng_exact,
            "sampler_rng_reinitialized": not rng_exact,
            "global_micro_step": start_step * args.gradient_accumulation_steps,
            "training_contract": args.training_contract,
        })
    if args.steps <= start_step:
        raise ValueError(f"--steps is the target total step; checkpoint already at {start_step}")
    if args.resume is None:
        rng = random.Random(args.seed)
    report["validation_samples"] = [
        {"episode": ep, "frame_index": int(groups[ep][frame]["frame_index"])} for ep, frame in val_samples
    ]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    prefix_cache = FrozenPrefixCache(int(args.prefix_cache_mib * 1024**2), args.prefix_cache_device)
    # MiB转bytes：1024**2是1024的平方；预算0关闭保留缓存，不代表跳过前缀计算。
    started = time.perf_counter()
    eligible_episodes = list(start_pools)

    if semantic_stage:
        def sample_loss(ep, frame):
            return semantic_sample_loss(semantic_policy, model, device, root, info, groups, episode_meta, args, ep, frame)
    else:
        def sample_loss(ep, frame):
            return action_sample_loss(model, device, prefix_cache, root, info, groups, episode_meta, stats, args, ep, frame)

    for step in range(start_step + 1, args.steps + 1):
        # range左闭右开；steps+1使循环包含目标总步数。续训从旧step的下一步开始。
        # 一个 step 表示一次参数更新，不是一次 epoch，也不一定只看一份样本。
        optimizer.zero_grad(set_to_none=True)  # 清掉上次更新的梯度；随后各 microstep 累加。
        micro_samples, loss_value = [], 0.0
        component_means = {}
        # 前一行zero_grad只清参数.grad，不清权重；set_to_none=True把旧梯度置None。
        # 一个list记录本次抽过的窗口；一个float累加供打印的平均loss。
        for micro in range(args.gradient_accumulation_steps):
            # micro从0起；累积次数4表示连续4次前后向，再做一次参数更新。
            # 先等概率选 episode，再等概率选窗口；这不等价于所有帧等概率抽样。
            micro_index = (step - 1) * args.gradient_accumulation_steps + micro
            if semantic_stage:
                episode, start = rng.choice(semantic_pools[micro_index % len(semantic_pools)])
            else:
                episode = rng.choice(eligible_episodes)
                start = rng.choice(start_pools[episode])
            sample = {"micro_step": micro_index + 1, "episode": episode,
                      "frame_index": int(groups[episode][start]["frame_index"])}
            if joint_stage:
                sem_ep, sem_i = rng.choice(semantic_pools[micro_index % len(semantic_pools)])
                sem_row = groups[sem_ep][sem_i]
                if "boxes" in sem_row:
                    sample["grounding_boxes"] = len(sem_row["boxes"])
                    args.predicts_boxes = True
                sample.update(task=groups[episode][start]["subtask"], semantic_episode=sem_ep,
                              semantic_frame_index=int(sem_row["frame_index"]), semantic_subtask=sem_row["subtask"])
                losses = joint_objectives(joint_policy, model, device, root, info, groups, episode_meta,
                                          stats, args, episode, start, sem_ep, sem_i)
            else:
                loss, sample["task"] = sample_loss(episode, start)
                losses = [("semantic_ce" if semantic_stage else "flow_mse", loss, 1.0)]
            for component, loss, weight in losses:
                if not loss.requires_grad or not torch.isfinite(loss).item():
                    raise RuntimeError(f"Step {step}, microstep {micro + 1}: invalid {component} or missing gradient")
                value = loss.item() / args.gradient_accumulation_steps
                component_means[component] = component_means.get(component, 0.0) + value
                loss_value += value * weight
                (loss * (weight / args.gradient_accumulation_steps)).backward()
                del loss
            micro_samples.append(sample)

        if any(not p.requires_grad and p.grad is not None for p in model.paligemma_with_expert.paligemma.parameters()):
            raise RuntimeError("Frozen PaliGemma unexpectedly received gradients")
        if joint_stage and not any(p.grad is not None for p in model.joint_adapter.parameters()):
            raise RuntimeError("Joint objectives did not reach the shared VLM adapter")
        optimizer.step()  # 使用累计梯度更新参数；有效 batch = 1 * 累积次数。
        if any(not torch.isfinite(p).all().item() for p in params.values()):
            # 上一行optimizer.step才改变参数；这里遍历更新后的参数，确认每个数值有限。
            raise RuntimeError(f"Step {step}: nonfinite trainable parameter")
        entry = {
            "step": step, "train_loss": loss_value,
            "episode": micro_samples[0]["episode"], "frame_index": micro_samples[0]["frame_index"],
            "task": micro_samples[0]["task"], "micro_samples": micro_samples,
            "micro_steps_completed": step * args.gradient_accumulation_steps,
        }
        if joint_stage:
            entry["train_components"] = component_means
        history.append(entry)
        report["history"] = history

        if step == 1 or step % args.log_every == 0:
            # 各参数的梯度来自同一次累积更新，保留其原值用于报告总 L2 范数。
            grad_norms = [torch.linalg.vector_norm(p.grad.detach(), dtype=torch.float32)
                          for p in params.values() if p.grad is not None]
            gradient_l2 = torch.linalg.vector_norm(torch.stack(grad_norms)).item() if grad_norms else 0.0
            data.event("TRAIN_STEP", {
                "step": step, "train_loss": loss_value,
                "episode": entry["episode"], "frame_index": entry["frame_index"],
                "micro_steps_completed": entry["micro_steps_completed"],
                "gradient_l2": gradient_l2,
                **({"components": component_means} if joint_stage else {}),
                "prefix_cache": prefix_cache.stats(),
            })

        if step % args.eval_every == 0 or step == args.steps:
            # %是取余：整除间隔时验证；最后一步无论是否整除都验证并保存。
            # 验证使用固定窗口/噪声并隔离 RNG，避免评估过程改掉后续训练采样轨迹。
            if semantic_stage:
                with torch.no_grad():
                    losses = [float(sample_loss(ep, i)[0].item()) for ep, i in val_samples]
                val = {"mean_token_cross_entropy": float(np.mean(losses)), "sample_count": len(losses)}
            else:
                val = evaluate(model, loader, args, device, root, info, groups, episode_meta, stats, val_samples,
                               joint_policy=joint_policy)
                if joint_stage:
                    val.update(evaluate_subtasks(joint_policy, model, device, root, info, groups, episode_meta,
                                                 stats, args, semantic_val_samples))
                    val["mean_joint_loss"] = (val["mean_semantic_ce"] + val["mean_fast_ce"]
                                              + args.flow_weight * val["mean_flow_matching_mse_6d"])
            entry["validation"] = val
            metrics = ({"mean_token_cross_entropy": val["mean_token_cross_entropy"]} if semantic_stage else
                       {"validation_mse": val["mean_flow_matching_mse_6d"], "validation_std": val["std_flow_matching_mse_6d"]})
            if joint_stage:
                metrics.update({name: val[name] for name in ("mean_fast_ce", "mean_semantic_ce",
                                                              "semantic_sample_count", "mean_joint_loss")})
                if args.eval_subtasks:
                    metrics.update({name: val[name] for name in ("subtask_exact_match", "subtask_predictions")})
            data.event("VALIDATION_OK", {"step": step, "train_loss": loss_value, "sample_count": val["sample_count"],
                                   **metrics, **({"generated_actions": val["generated_actions"]} if args.eval_actions else {})})
        if "validation" in entry or (args.save_every > 0 and step % args.save_every == 0):
            args._sampler_rng_state = rng.getstate()
            checkpoint = save_checkpoint(
                trainable_model, optimizer, args, stats, train_ids, val_ids, step, args.output_dir, history
            )
            entry["checkpoint"] = checkpoint
            data.event("CHECKPOINT_SAVED", {"step": step, "path": checkpoint})

    report["elapsed_seconds"] = time.perf_counter() - started
    report["training_peak_memory"] = loader.snapshot(device)
    report["steps_completed"] = args.steps
    report["micro_steps_completed"] = args.steps * args.gradient_accumulation_steps
    report["prefix_cache"] = prefix_cache.stats()
    report["resumed_from_step"] = start_step
    report["final_checkpoint"] = report["history"][-1].get("checkpoint")
    report["status"] = "FINETUNE_OK"
    report["not_validated"] = [
        "real robot policy performance",
        "generalization beyond recorded task/object distribution",
        "pretrained base checkpoint action slot semantics",
        "real-time closed-loop performance",
    ]
    data.event("FINETUNE_OK", {
        "steps_completed": args.steps,
        "micro_steps_completed": report["micro_steps_completed"],
        "elapsed_seconds": report["elapsed_seconds"],
        "final_checkpoint": report["final_checkpoint"],
        "training_peak_memory": report["training_peak_memory"],
        "robot_commands_sent": False,
        "prefix_cache": report["prefix_cache"],
    })


def parse_args(argv=None):
    # 【调用方】main；把命令行字符串转Namespace对象、补默认值，返回args。
    # argv=None时读取实际命令行；传list时可离线解析，例如["--root","data/path"]。
    # --image-keys自动变成args.image_keys：连字符转下划线。
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("action", "semantic", "joint"),
                   help="Action flow (default), legacy subtask text, or joint subtask/FAST+/flow training")
    # ArgumentParser是标准库argparse的类；__doc__是文件顶部模块说明字符串。
    p.add_argument("--root", type=Path, help="Local LeRobot v3 dataset; saved in new checkpoints")
    # add_argument声明选项；type=Path把字符串变成Path对象，不等于检查目录存在。
    p.add_argument("--model-dir", type=Path, help="Base checkpoint (default: assets_root/pi05_base)")
    p.add_argument("--tokenizer-dir", type=Path, help="Text tokenizer (default: assets_root/paligemma_tokenizer)")
    p.add_argument("--fast-tokenizer-dir", type=Path,
                   help="Local snapshot of physical-intelligence/fast for joint training (default: assets_root/fast)")
    p.add_argument("--flow-weight", type=float,
                   help="Joint objective: semantic CE + FAST CE + weight * flow MSE (default 10; 0 for discrete only)")
    p.add_argument("--device-map", type=Path, help="Explicit placement map; overrides --gpu-layers")
    p.add_argument("--image-keys", nargs="+", help="Default: every dataset camera also present in the model")
    p.add_argument("--semantic-image-keys", nargs="+", help="High-level joint cameras (e.g. front, rear, wrist); "
                   "--image-keys sets the low-level cameras. Named rear views default to high-level only")
    # nargs="+"允许一个或多个值：--image-keys key1 key2 → list[str]。
    p.add_argument("--image-key", help="Single-camera compatibility alias")
    p.add_argument("--task", help="Fallback label only for legacy data without task metadata")
    p.add_argument("--train-episodes", type=int, nargs="+",
                   help="Explicit training episode IDs; resume defaults to the checkpoint selection")
    p.add_argument("--val-episodes", type=int, nargs="+")
    p.add_argument("--episode-frames", type=episode_frame_range, action="append", metavar="EP=START:STOP",
                   help="Repeat for selected episodes; original zero-based rows, stop exclusive. Resume inherits saved ranges")
    # type也可传自定义函数；字符串先由episode_frame_range转换，再由append收入列表。
    # 例如--episode-frames 0=10:100 --episode-frames 2=20:200；metavar只改变帮助文本。
    p.add_argument("--steps", type=int, default=200, help="Target total optimizer step, including resumed steps")
    p.add_argument("--gradient-accumulation-steps", type=int,
                   help="Microsteps per optimizer update (default: 1; resume inherits the saved contract)")
    p.add_argument("--chunk-start-stride", type=int,
                   help="Spacing of training observation starts; future actions keep every frame (default: 1)")
    p.add_argument("--prefix-cache-mib", type=float, default=0,
                   help="Maximum retained frozen-prefix tensor storage in MiB; 0 disables the in-memory LRU")
    p.add_argument("--prefix-cache-device", choices=("cpu", "cuda"), default="cuda",
                   help="Frozen-prefix storage location; performance setting may change on resume (default: cuda)")
    p.add_argument("--validation-samples", type=int, help="Default: 10 or enough to cover every episode's endpoints; resume inherits saved samples")
    p.add_argument("--validation-sampling", choices=("uniform", "random"),
                   help="New runs use uniform full-episode coverage; resumed runs preserve their saved windows")
    p.add_argument("--eval-actions", action=argparse.BooleanOptionalAction, default=None,
                   help="Also generate action chunks and compare joint errors with the hold-current baseline")
    p.add_argument("--eval-subtasks", action=argparse.BooleanOptionalAction, default=None,
                   help="Joint validation: freely generate current subtasks on held-out frames and report exact match")
    # BooleanOptionalAction提供--eval-actions和--no-eval-actions，分别设True/False。
    # 实际输入无需“=True”；没传时None允许续训继承旧设置。
    p.add_argument("--eval-action-steps", type=int, help="Euler steps for generated validation actions (default: 10)")
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=5, help="Console progress interval; no log file")
    p.add_argument("--rank", type=int)
    p.add_argument("--alpha", type=float)
    p.add_argument("--lr", type=float)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--gpu-layers", type=int, default=8,
                   help="PaliGemma language layers kept on GPU when --device-map is absent")
    p.add_argument("--cpu-threads", type=int, default=4,
                   help="PyTorch intra-op CPU threads; performance setting may change on resume (default: 4)")
    p.add_argument("--chunk-mib", type=float, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-seed", type=int, help="Fixed validation noise seed (default: 9000; resume inherits)")
    p.add_argument("--output-dir", type=Path)
    source = p.add_mutually_exclusive_group()
    # 互斥组让--resume与--init-from不能同时传：延续训练和新实验初始化是两种路径。
    source.add_argument("--resume", type=Path, help="Continue the same dataset, optimizer and RNG state")
    source.add_argument("--init-from", type=Path,
                        help="Initialize adapter weights/statistics for a new dataset, task or camera setup")
    args = p.parse_args(argv)
    # 到这里才消费输入；--help打印帮助并退出，不继续执行后面的模型加载。
    if args.episode_frames is not None:
        ranges = {}
        for episode, bounds in args.episode_frames:
            if episode in ranges:
                p.error(f"Duplicate --episode-frames for episode {episode}")
            ranges[episode] = bounds
        args.episode_frames = ranges
    # 旧 checkpoint 的路径可能仍指向迁移前目录；只解析已有输入，不重写新输出位置。
    for field in ("model_dir", "tokenizer_dir", "device_map", "resume", "init_from"):
        # (...)是字符串tuple；for逐个取属性名字，不是读这几个路径里的文件。
        value = getattr(args, field)
        # getattr(obj,name)按字符串访问属性；相当于args.model_dir等写法的动态版本。
        if value is not None:
            setattr(args, field, resolve_asset_path(value))
            # setattr按名字写回属性；resolve_asset_path定义在本包__init__.py。
    checkpoint = args.resume or args.init_from
    source_config = data.read_json(checkpoint / "adapter_config.json") if checkpoint else {}
    source_stage = {pi05.SEMANTIC_FORMAT: "semantic", "friston_pi05_expert_lora_v1": "action",
                    joint.JOINT_FORMAT: "joint"}.get(source_config.get("format"))
    args.stage = args.stage or source_stage or "action"
    joint_source = source_config if source_stage == "joint" else {}
    args.model_dir = resolve_asset_path(args.model_dir or joint_source.get("base_model_dir") or assets_root() / "pi05_base")
    args.tokenizer_dir = resolve_asset_path(args.tokenizer_dir or joint_source.get("tokenizer_dir")
                                            or assets_root() / "paligemma_tokenizer")
    warm_start_action = args.init_from and source_stage == "action" and args.stage == "joint"
    if checkpoint and args.stage != source_stage and not warm_start_action:
        p.error("The checkpoint format is incompatible with the selected training stage")
    if args.stage == "joint":
        if args.prefix_cache_mib:
            p.error("Joint training updates prefix K/V; use --prefix-cache-mib 0")
        if args.flow_weight is None:
            args.flow_weight = source_config.get("training_contract", {}).get("flow_weight", 10.0) if args.resume else 10.0
        if not np.isfinite(args.flow_weight) or args.flow_weight < 0:
            p.error("--flow-weight must be finite and nonnegative")
        fast_path = args.fast_tokenizer_dir or source_config.get("fast_tokenizer_dir") or assets_root() / "fast"
        args.fast_tokenizer_dir = resolve_asset_path(fast_path)
        if not args.fast_tokenizer_dir.is_dir():
            p.error(f"Missing FAST+ snapshot directory {args.fast_tokenizer_dir}; supply --fast-tokenizer-dir")
    elif (args.fast_tokenizer_dir is not None or args.flow_weight is not None or args.eval_subtasks
          or args.semantic_image_keys is not None):
        p.error("--fast-tokenizer-dir, --flow-weight, --eval-subtasks and --semantic-image-keys apply to --stage joint")
    if args.stage == "semantic" and (args.eval_actions or args.prefix_cache_mib or args.chunk_start_stride not in (None, 1)):
        p.error("Semantic training uses frame labels; action evaluation, frozen-prefix cache and chunk stride are action-only")
    if args.root is None:
        stored_root = source_config.get("dataset_root") if args.resume else None
        if stored_root:
            args.root = Path(stored_root)
        else:
            p.error("--root is required for a new run or an older checkpoint without dataset_root")
    args.rank = args.rank if args.rank is not None else source_config.get("rank", 8)
    # 优先级：CLI明确值 > 来源adapter值 > 默认8；is not None用于区分是否明确传参。
    args.alpha = args.alpha if args.alpha is not None else source_config.get("alpha", 16.0)
    args.lr = args.lr if args.lr is not None else source_config.get("learning_rate", 1e-4)
    if source_config and (args.rank != source_config["rank"] or args.alpha != source_config["alpha"]):
        p.error("rank and alpha must match the source adapter")
    if args.image_key:
        if args.image_keys:
            p.error("Use --image-key or --image-keys, not both")
        args.image_keys = [args.image_key]
    if (min(args.steps, args.eval_every, args.log_every) < 1
            or (args.validation_samples is not None and args.validation_samples < 1)):
        p.error("steps, validation-samples, eval-every and log-every must be positive")
    if args.eval_action_steps is not None and args.eval_action_steps < 1:
        p.error("--eval-action-steps must be positive")
    if any(value is not None and value < 1 for value in (args.gradient_accumulation_steps, args.chunk_start_stride)):
        p.error("gradient-accumulation-steps and chunk-start-stride must be positive")
    if not np.isfinite(args.prefix_cache_mib) or args.prefix_cache_mib < 0:
        p.error("--prefix-cache-mib must be finite and nonnegative")
    if args.save_every < 0 or args.rank < 1 or min(args.lr, args.alpha, args.chunk_mib) <= 0:
        p.error("Invalid save interval, rank, learning rate, alpha or loading chunk size")
    if args.gpu_layers < 0:
        p.error("--gpu-layers must be nonnegative")
    if args.cpu_threads < 1:
        p.error("--cpu-threads must be positive")
    if not all(np.isfinite(value) for value in (args.lr, args.alpha, args.chunk_mib)):
        p.error("lr, alpha and chunk-mib must be finite")
    args.root = resolve_asset_path(args.root)
    if args.output_dir is None:
        from datetime import datetime
        args.output_dir = assets_root() / "checkpoints" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def load_training_model(args, loader=loader, report: dict | None = None):
    """重用流式加载；仅调度冻结的 PaliGemma 子模型。"""
    # 【调用方】trainer.run；返回(PI05Pytorch,torch.device)，把加载统计写入report字典。
    # 参数默认loader=loader引用上面导入的模块；dict|None表示报告可以省略。
    report = {} if report is None else report
    # 省略时创建新dict，不把{}写为默认参数，避免多个调用共享同一可变默认对象。
    from accelerate import dispatch_model
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    device = torch.device(f'cuda:{args.gpu}')
    # torch.device只是设备描述对象；cuda:0表示第0块GPU，它本身不创建模型张量。
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    model = loader.build_meta_model(args.model_dir)
    mapping_path = args.device_map
    if mapping_path is not None and mapping_path.is_file():
        raw = data.read_json(mapping_path)
        device_map = {k: loader.normalize_device(v) for k, v in raw.get('device_map', raw).items()}
    elif args.device_map:
        raise FileNotFoundError(mapping_path)
    else:
        device_map = loader.default_device_map(model, args.gpu, getattr(args, 'gpu_layers', 8))
    for value in device_map.values():
        target = loader.torch_device(value)
        if target.type not in ('cpu', 'cuda') or (target.type == 'cuda' and target.index != args.gpu):
            raise ValueError('This training path uses one GPU plus CPU storage, not disk/multiple GPUs')
    # Only frozen PaliGemma may use offload. Check before loading huge tensors.
    for name, _ in model.named_parameters():
        if name.startswith(pi05.EXPERT + '.') or name.split('.')[0] in pi05.PROJECTIONS:
            if loader.torch_device(loader.placement(name, device_map)) != device:
                raise ValueError(f'Train graph must remain resident on {device}: {name}')
    proxy = SimpleNamespace(model=model)
    fixer = lambda entries: PI05Policy._fix_pytorch_state_dict_keys(proxy, entries, model.config)
    with checkpoint_reader():
        report['loading'] = loader.stream_load(
            model, loader.checkpoint_sources(args.model_dir), device_map, fixer, args.chunk_mib)
    prefix_map = {key[len(pi05.PALI) + 1:]: value for key, value in device_map.items()
                  if key.startswith(pi05.PALI + '.')}
    if not prefix_map:
        raise ValueError('Device map must explicitly describe PaliGemma submodules')
    pg = model.paligemma_with_expert
    # 冻结前缀允许 CPU 存储 + GPU 执行；有梯度的动作分支保持在 GPU 上。
    # 因而不能直接复用“把整个模型都挂推理卸载钩子”的加载路径来训练。
    pg.paligemma = dispatch_model(pg.paligemma, device_map=prefix_map,
                                  main_device=device, offload_buffers=False, force_hooks=True)
    model.eval().requires_grad_(False)
    model.config.freeze_vision_encoder = True
    model.config.train_expert_only = True
    pg.freeze_vision_encoder = True
    pg.train_expert_only = True
    for name in (pi05.EXPERT, *pi05.PROJECTIONS):
        module = model.get_submodule(name)
        if any(hasattr(child, '_hf_hook') for child in module.modules()):
            raise RuntimeError(f'Unexpected inference hook in trainable branch: {name}')
        if any(p.is_meta or p.device != device for p in module.parameters()):
            raise RuntimeError(f'Nonresident tensor in trainable branch: {name}')
    report['loading'].update(seconds=time.perf_counter()-start,
                             device_map={k: str(v) for k, v in device_map.items()},
                             offload_scope='frozen_paligemma_only', memory=loader.snapshot(device))
    data.event('LOAD_OK', report['loading']['memory'])
    return model, device


def main():
    # 入口只做两步：parse_args解析整理参数，run执行训练；导入模块不会自动执行main。
    run(parse_args())


if __name__ == "__main__":
    # python -m friston3.arms.learning.trainer 时模块名为__main__，才进入此分支。
    main()
