#!/usr/bin/env python
"""SO-101 π₀.₅ 双相机联合微调：从示教抽样，更新模型，验证并保存断点。

高层根据总目标和当前观测预测子任务，低层以子任务为条件预测 FAST 词元与连续动作。
训练专用装配让可训练参数常驻 GPU，冻结的视觉语言骨干可卸载至 CPU。
"""
from __future__ import annotations

import argparse
import random
import time
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from ..config import DATASET_JOINTS, JOINT_FORMAT, POLICY_IMAGE_KEYS
from ..models import assets_root, fast_tokenizer_root, loader, pi05
from ..models.weights import checkpoint_reader
from . import data


def semantic_frames(groups, episode_ids):
    """Return every annotated frame, including terminal frames and segments shorter than H."""
    by_label = {}
    for ep in episode_ids:
        for i, row in enumerate(groups[ep]):
            bucket = (row["subtask"], "boxes" in row)
            by_label.setdefault(bucket, []).append((ep, i))
    # 稀疏框标注独立成组；每轮也训练纯文本样本和完成态。
    return dict(sorted(by_label.items(), key=lambda item: not item[0][1]))


def joint_semantic_inputs(model, device, root, info, groups, episode_meta, stats, args, ep, start):
    state, rgb, goal, subtask = data.sample_subtask_frame(
        root, info, groups, episode_meta, stats, args, ep, start)
    images, masks = pi05.prepare_images(model, device, rgb)
    boxes = groups[ep][start].get("boxes")
    return images, masks, goal, subtask, state, boxes


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
    """同一子任务的连续帧内取能容纳完整动作块的起点。"""
    pools, short = {}, []
    for ep in ids:
        rows = groups[ep]
        starts = []
        begin = 0
        for label, run in groupby(rows, key=itemgetter("subtask")):
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


def configure_validation(args, pools):
    """用固定种子选验证窗口；相同数据和设置会得到相同样本。"""
    args.validation_samples = args.validation_samples or max(10, 2 * len(pools))
    return choose_val_samples(pools, args.validation_samples, args.eval_seed, args.validation_sampling)


def joint_validation_frames(args, groups, val_ids):
    """Keep held-out high-level frame coordinates independent of action horizon and boundaries."""
    by_label = semantic_frames(groups, val_ids)
    if not by_label:
        raise ValueError("Joint validation needs annotated subtask frames in the held-out episodes")
    rng = random.Random(args.eval_seed)
    pools = list(by_label.values())
    return [rng.choice(pools[i % len(pools)]) for i in range(max(len(pools), args.validation_samples))]


def configure_cameras(args, info, model_config):
    """高低层共用正面、腕部视角，按基础模型的输入顺序绑定图像和框标签。"""
    keys = list(POLICY_IMAGE_KEYS)
    model_keys = [key for key in model_config["input_features"] if key in keys]
    if model_keys != keys:
        raise ValueError(f"Base model camera slots must be ordered {keys}, got {model_keys}")
    missing = [key for key in keys if key not in info["features"]]
    if missing:
        raise ValueError(f"Dataset is missing required camera features: {missing}")
    args.image_keys = keys


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


def evaluate(joint_policy, model, args, device, root, info, groups, episode_meta, stats, samples):
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
                value = joint_policy.flow_loss(inputs, actions, seed=args.eval_seed + index).item()
                fast_ce = float(joint_policy.fast_loss(inputs, actions).item())
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
        details[-1]["fast_ce"] = fast_ce
        del inputs
    result = {
        "mean_flow_matching_mse_6d": float(np.mean(values)),
        "std_flow_matching_mse_6d": float(np.std(values)),
        "sample_count": len(values), "samples": details,
    }
    result["mean_fast_ce"] = float(np.mean([row["fast_ce"] for row in details]))
    if args.eval_actions:
        result["generated_actions"] = {
            "joint_order": DATASET_JOINTS, "units": ["degrees"] * 5 + ["percent"],
            "num_steps": args.eval_action_steps, "rtc": False,
            **{key: np.mean([row["generated_actions"][key] for row in details], axis=0).tolist()
               for key in details[0]["generated_actions"]}}
    return result


@torch.no_grad()
def evaluate_subtasks(policy, model, device, root, info, groups, episode_meta, stats, args, samples):
    ce_values, predictions = [], []
    for ep, i in samples:
        images, masks, goal, label, state, boxes = joint_semantic_inputs(
            model, device, root, info, groups, episode_meta, stats, args, ep, i)
        ce_values.append(policy.semantic_loss(images, masks, goal, label, state, boxes).item())
        if args.eval_subtasks:
            record = {"episode": ep, "frame_index": int(groups[ep][i]["frame_index"]),
                      "global_goal": goal, "reference": label}
            if boxes is not None:
                record["reference_boxes"] = boxes
            try:
                # 使用与部署相同的提示。验证帧恰好有无人工框只影响监督和比较。
                prediction, detected = policy.predict_result(images, masks, goal, state,
                                                             predict_boxes=args.predicts_boxes)
                record.update(prediction=prediction, exact_match=prediction == label)
                if detected is not None:
                    record["predicted_boxes"] = detected
            except pi05.SubtaskDecodeError as error:
                record.update(decode_error=str(error), exact_match=False)
            predictions.append(record)
    metrics = {"mean_semantic_ce": float(np.mean(ce_values)), "semantic_sample_count": len(samples)}
    if predictions:
        metrics.update(subtask_predictions=predictions,
                       subtask_exact_match=float(np.mean([row["exact_match"] for row in predictions])))
    return metrics


def save_checkpoint(model, optimizer, args, stats, train_ids, val_ids, step, micro_step, output, history, sampler_state):
    """写完权重、优化器、采样状态与元数据后，一次发布可用的断点目录。"""
    from safetensors.torch import save_file

    target = output / f"checkpoint_step_{step:06d}"
    output.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    config = {
        "format": JOINT_FORMAT,
        "base_model_dir": str(args.model_dir), "tokenizer_dir": str(args.tokenizer_dir),
        "fast_tokenizer_dir": str(args.fast_tokenizer_dir),
        "rank": args.rank, "alpha": args.alpha, "learning_rate": args.lr,
        "joint_order": DATASET_JOINTS, "model_slots": list(range(6)), "supervised_action_dims": 6,
        "horizon": args.horizon, "max_action_dim": args.max_action_dim, "image_keys": args.image_keys,
        "dataset_root": str(args.root), "dataset_fps": args.dataset_fps,
        "training_episodes": train_ids, "validation_episodes": val_ids, "episode_frames": args.episode_frames,
        "statistics": stats,
        "normalization_source": args.normalization_source, "initialized_from": args.initialized_from,
        "training": {key: getattr(args, key) for key in
                     ("gradient_accumulation_steps", "chunk_start_stride", "flow_weight")},
        "validation": {key: getattr(args, key) for key in
                       ("validation_sampling", "validation_samples", "eval_seed",
                        "eval_actions", "eval_action_steps", "eval_subtasks")},
        "steps_completed": step,
        "has_completion": args.has_completion, "has_action_training": args.has_action_training,
        "predicts_boxes": args.predicts_boxes,
    }
    state = {
        "global_step": step, "global_micro_step": micro_step,
        "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "history": history, "sampler_rng_state": sampler_state,
    }
    with TemporaryDirectory(dir=output) as tmp:
        directory = Path(tmp) / target.name
        directory.mkdir()
        params = pi05.trainable_dict(model)
        save_file({k: p.detach().cpu().contiguous().clone() for k, p in params.items()},
                  str(directory / "trainable.safetensors"))
        torch.save(optimizer.state_dict(), directory / "optimizer.pt")
        torch.save(state, directory / "training_state.pt")
        data.write_json(directory / "adapter_config.json", config)
        directory.rename(target)
    return str(target)


def run(args):
    # 【调用关系】main → 本函数 → 数据读取/模型加载/训练数值函数/evaluate/save_checkpoint。
    # args由parse_args创建；本函数负责整个训练生命周期，返回None，成果是checkpoint。
    # 读本函数时分四段：准备数据、加载模型、恢复或初始化、循环更新/验证/保存。
    # 先验证小型元数据，再加载大模型；训练/验证按 episode 切分，避免相邻帧泄漏。
    torch.set_num_threads(args.cpu_threads)
    # PyTorch单个CPU算子的线程数，不是几个训练worker，也不是GPU数量。
    root = args.root.resolve()
    # args.root已由argparse的type=Path转成路径对象；resolve得到绝对路径。
    info, groups, episode_meta = data.load_rows(root)
    source_config = args.source_config
    resume_config = source_config if args.resume else None
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
    unannotated = [ep for ep in train_ids + val_ids if "subtask" not in groups[ep][0]]
    if unannotated:
        raise ValueError(f"Missing subtask annotations for selected episodes {unannotated} in {root / 'meta/subtasks.json'}")
    args.train_episodes, args.val_episodes = train_ids, val_ids
    if args.episode_frames is None:
        args.episode_frames = (resume_config or {}).get("episode_frames", {})
    groups = select_episode_frames(groups, train_ids + val_ids, args.episode_frames)
    # 已学习的权重继续使用原来的归一化坐标；只有全新训练才拟合统计量。
    stats = (data.training_stats(groups, train_ids) if source_config is None
             else pi05.adapter_statistics(source_config))
    args.has_completion = source_config["has_completion"] if source_config else False
    args.predicts_boxes = source_config["predicts_boxes"] if source_config else False
    args.has_action_training = args.flow_weight > 0 or bool(source_config and source_config["has_action_training"])
    args.initialized_from = str(args.init_from) if args.init_from else (resume_config or {}).get("initialized_from")
    args.normalization_source = ("initialized_adapter" if args.init_from else
                                 (resume_config or {}).get("normalization_source", "training_episodes_only"))
    raw_config = data.read_json(args.model_dir / "config.json")
    args.horizon = int(raw_config["chunk_size"])
    args.max_action_dim = int(raw_config["max_action_dim"])
    # 基础config决定H和D；不是根据SO-101只有6关节就直接把网络宽度改成6。
    start_pools = sample_starts(groups, train_ids, args, args.chunk_start_stride)
    val_pools = sample_starts(groups, val_ids, args, 1)
    configure_cameras(args, info, raw_config)
    # 高层轮换子任务与完成态，低层从同一子任务内抽取完整窗口。
    semantic_pools = list(semantic_frames(groups, train_ids).values())
    args.dataset_fps = float(info["fps"])
    if resume_config and resume_config["dataset_fps"] != args.dataset_fps:
        raise ValueError("Resume dataset fps differs; use --init-from for a new action time scale")

    val_samples = configure_validation(args, val_pools)
    semantic_val_samples = joint_validation_frames(args, groups, val_ids)
    data.event("TRAIN_DATA_OK", {
        "train_episodes": train_ids, "validation_episodes": val_ids,
        "episode_frames": args.episode_frames,
        "train_frames": sum(len(groups[e]) for e in train_ids),
        "validation_frames": sum(len(groups[e]) for e in val_ids),
        "horizon": args.horizon, "supervised_action_dims": 6,
        "statistics_scope": args.normalization_source,
        "image_keys": args.image_keys,
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

    args.target_modules = pi05.attach_trainables(model, args.rank, args.alpha)
    joint_policy = pi05.JointPolicy(model, pi05.tokenizer_for(args), args.rank, args.alpha, device,
                                    fast_dir=args.fast_tokenizer_dir)
    params = pi05.trainable_dict(model)
    if source_config is not None:
        pi05.load_adapter_weights(params, args.resume or args.init_from)
        if args.init_from:
            data.event("INITIALIZED_FROM_ADAPTER", {
                "checkpoint": args.initialized_from, "optimizer_state_restored": False,
                "rng_state_restored": False, "normalization_source": args.normalization_source,
            })
    data.event("TRAINABLE_READY", {
        "parameters": sum(p.numel() for p in params.values()),
        "tensors": len(params), "lora_modules": len(args.target_modules),
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
    micro_step = 0
    history = []
    rng = random.Random(args.seed)
    if args.resume is not None:
        resume_dir = args.resume.resolve()
        optimizer_state = torch.load(resume_dir / "optimizer.pt", map_location="cpu", weights_only=False)
        # torch.load恢复保存的Python/Tensor结构；map_location使读入张量先落到CPU。
        optimizer.load_state_dict(optimizer_state)
        # 把Adam历史状态装回optimizer；仅加载模型权重不能恢复这些动量。
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        state = torch.load(resume_dir / "training_state.pt", map_location="cpu", weights_only=False)
        start_step = int(state["global_step"])
        micro_step = int(state["global_micro_step"])
        history = state["history"]
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng_states"])
        rng.setstate(state["sampler_rng_state"])
        data.event("RESUME_OK", {
            "checkpoint": str(resume_dir),
            "global_step": start_step,
            "optimizer_state_restored": True,
            "rng_state_restored": True,
            "global_micro_step": micro_step,
        })
    if args.steps <= start_step:
        raise ValueError(f"--steps is the target total step; checkpoint already at {start_step}")
    report["validation_samples"] = [
        {"episode": ep, "frame_index": int(groups[ep][frame]["frame_index"])} for ep, frame in val_samples
    ]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    eligible_episodes = list(start_pools)

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
            episode = rng.choice(eligible_episodes)
            start = rng.choice(start_pools[episode])
            sample = {"micro_step": micro_step + 1, "episode": episode,
                      "frame_index": int(groups[episode][start]["frame_index"])}
            sem_ep, sem_i = rng.choice(semantic_pools[micro_step % len(semantic_pools)])
            sem_row = groups[sem_ep][sem_i]
            if sem_row["subtask"] is None:
                args.has_completion = True
            if "boxes" in sem_row:
                sample["grounding_boxes"] = len(sem_row["boxes"])
                args.predicts_boxes = True
            sample.update(task=groups[episode][start]["subtask"], semantic_episode=sem_ep,
                          semantic_frame_index=int(sem_row["frame_index"]), semantic_subtask=sem_row["subtask"])
            losses = joint_objectives(joint_policy, model, device, root, info, groups, episode_meta,
                                      stats, args, episode, start, sem_ep, sem_i)
            for component, loss, weight in losses:
                if not loss.requires_grad or not torch.isfinite(loss).item():
                    raise RuntimeError(f"Step {step}, microstep {micro + 1}: invalid {component} or missing gradient")
                value = loss.item() / args.gradient_accumulation_steps
                component_means[component] = component_means.get(component, 0.0) + value
                loss_value += value * weight
                (loss * (weight / args.gradient_accumulation_steps)).backward()
                del loss
            micro_samples.append(sample)
            micro_step += 1  # 独立累计；续训时修改梯度累积次数也不会重算过去的步数。

        optimizer.step()  # 使用累计梯度更新参数；有效 batch = 1 * 累积次数。
        entry = {
            "step": step, "train_loss": loss_value,
            "episode": micro_samples[0]["episode"], "frame_index": micro_samples[0]["frame_index"],
            "task": micro_samples[0]["task"], "micro_samples": micro_samples,
            "micro_steps_completed": micro_step, "train_components": component_means,
        }
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
                "components": component_means,
            })

        if step % args.eval_every == 0 or step == args.steps:
            # %是取余：整除间隔时验证；最后一步无论是否整除都验证并保存。
            # 验证使用固定窗口/噪声并隔离 RNG，避免评估过程改掉后续训练采样轨迹。
            val = evaluate(joint_policy, model, args, device, root, info, groups, episode_meta, stats, val_samples)
            val.update(evaluate_subtasks(joint_policy, model, device, root, info, groups, episode_meta,
                                         stats, args, semantic_val_samples))
            val["mean_joint_loss"] = (val["mean_semantic_ce"] + val["mean_fast_ce"]
                                      + args.flow_weight * val["mean_flow_matching_mse_6d"])
            entry["validation"] = val
            metrics = {"validation_mse": val["mean_flow_matching_mse_6d"],
                       "validation_std": val["std_flow_matching_mse_6d"]}
            metrics.update({name: val[name] for name in ("mean_fast_ce", "mean_semantic_ce",
                                                          "semantic_sample_count", "mean_joint_loss")})
            if args.eval_subtasks:
                metrics.update({name: val[name] for name in ("subtask_exact_match", "subtask_predictions")})
            data.event("VALIDATION_OK", {"step": step, "train_loss": loss_value, "sample_count": val["sample_count"],
                                   **metrics, **({"generated_actions": val["generated_actions"]} if args.eval_actions else {})})
        if "validation" in entry or (args.save_every > 0 and step % args.save_every == 0):
            checkpoint = save_checkpoint(
                model, optimizer, args, stats, train_ids, val_ids, step, micro_step,
                args.output_dir, history, rng.getstate()
            )
            entry["checkpoint"] = checkpoint
            data.event("CHECKPOINT_SAVED", {"step": step, "path": checkpoint})

    report["elapsed_seconds"] = time.perf_counter() - started
    report["training_peak_memory"] = loader.snapshot(device)
    report["steps_completed"] = args.steps
    report["micro_steps_completed"] = micro_step
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
    })


def parse_args(argv=None):
    # 【调用方】main；把命令行字符串转Namespace对象、补默认值，返回args。
    # argv=None时读取实际命令行；传list时可离线解析，例如["--root","data/path"]。
    p = argparse.ArgumentParser(description=__doc__)
    # ArgumentParser是标准库argparse的类；__doc__是文件顶部模块说明字符串。
    p.add_argument("--root", type=Path, help="Local LeRobot v3 dataset; saved in new checkpoints")
    # add_argument声明选项；type=Path把字符串变成Path对象，不等于检查目录存在。
    p.add_argument("--model-dir", type=Path, help="Base checkpoint (default: assets_root/pi05_base)")
    p.add_argument("--tokenizer-dir", type=Path, help="Text tokenizer (default: assets_root/paligemma_tokenizer)")
    p.add_argument("--fast-tokenizer-dir", type=Path,
                   help="Local snapshot of physical-intelligence/fast (default: project FAST asset)")
    p.add_argument("--flow-weight", type=float,
                   help="Joint objective: semantic CE + FAST CE + weight * flow MSE (default 10; 0 for discrete only)")
    p.add_argument("--device-map", type=Path, help="Explicit placement map; overrides --gpu-layers")
    p.add_argument("--train-episodes", type=int, nargs="+",
                   help="Explicit training episode IDs; resume defaults to the checkpoint selection")
    p.add_argument("--val-episodes", type=int, nargs="+")
    p.add_argument("--episode-frames", type=episode_frame_range, action="append", metavar="EP=START:STOP",
                   help="Repeat for selected episodes; original zero-based rows, stop exclusive. Resume inherits saved ranges")
    # type也可传自定义函数；字符串先由episode_frame_range转换，再由append收入列表。
    # 例如--episode-frames 0=10:100 --episode-frames 2=20:200；metavar只改变帮助文本。
    p.add_argument("--steps", type=int, default=200, help="Target total optimizer step, including resumed steps")
    p.add_argument("--gradient-accumulation-steps", type=int,
                   help="Microsteps per optimizer update (default: 1; resume inherits unless overridden)")
    p.add_argument("--chunk-start-stride", type=int,
                   help="Spacing of training observation starts; future actions keep every frame (default: 1)")
    p.add_argument("--validation-samples", type=int, help="Default: 10 or enough to cover every episode's endpoints; resume inherits the count")
    p.add_argument("--validation-sampling", choices=("uniform", "random"),
                   help="Validation window selection (default: uniform; resume inherits unless overridden)")
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
    p.add_argument("--eval-seed", type=int, help="Fixed validation sampling and noise seed (default: 9000; resume inherits)")
    p.add_argument("--output-dir", type=Path)
    source = p.add_mutually_exclusive_group()
    # 互斥组让--resume与--init-from不能同时传：延续训练和新实验初始化是两种路径。
    source.add_argument("--resume", type=Path, help="恢复权重、优化器、步长和随机数生成器；显式训练/验证设置将覆盖保存的值。")
    source.add_argument("--init-from", type=Path,
                        help="初始化新数据集或目标函数的联合适配器权重和归一化")
    args = p.parse_args(argv)
    # 到这里才消费输入；--help打印帮助并退出，不继续执行后面的模型加载。
    if args.episode_frames is not None:
        ranges = {}
        for episode, bounds in args.episode_frames:
            if episode in ranges:
                p.error(f"Duplicate --episode-frames for episode {episode}")
            ranges[episode] = bounds
        args.episode_frames = ranges
    for field in ("resume", "init_from", "device_map"):
        path = getattr(args, field)
        if path is not None:
            setattr(args, field, path.expanduser().resolve())
    checkpoint = args.resume or args.init_from
    args.source_config = data.read_json(checkpoint / "adapter_config.json") if checkpoint else None
    source_config = args.source_config or {}
    if checkpoint and source_config.get("format") != JOINT_FORMAT:
        p.error(f"Expected a dual-camera joint checkpoint ({JOINT_FORMAT}): {checkpoint}")
    assets = assets_root()
    model_dir = args.model_dir or source_config.get("base_model_dir") or assets / "pi05_base"
    tokenizer_dir = args.tokenizer_dir or source_config.get("tokenizer_dir") or assets / "paligemma_tokenizer"
    fast_dir = args.fast_tokenizer_dir or source_config.get("fast_tokenizer_dir") or fast_tokenizer_root()
    args.model_dir = Path(model_dir).expanduser().resolve()
    args.tokenizer_dir = Path(tokenizer_dir).expanduser().resolve()
    args.fast_tokenizer_dir = Path(fast_dir).expanduser().resolve()
    if not args.fast_tokenizer_dir.is_dir():
        p.error(f"Missing FAST+ snapshot directory {args.fast_tokenizer_dir}; supply --fast-tokenizer-dir")
    # 训练/验证参数只在这里填默认值并校验；明确传入的 CLI 参数优先。
    defaults = {
        "gradient_accumulation_steps": 1, "chunk_start_stride": 1, "flow_weight": 10.0,
        "validation_sampling": "uniform", "validation_samples": None, "eval_seed": 9000,
        "eval_actions": False, "eval_action_steps": 10, "eval_subtasks": False,
    }
    saved = {**source_config["training"], **source_config["validation"]} if args.resume else {}
    for key, default in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, saved.get(key, default))
    if not np.isfinite(args.flow_weight) or args.flow_weight < 0:
        p.error("--flow-weight must be finite and nonnegative")
    if args.root is None:
        if not args.resume:
            p.error("--root is required when starting a new training run")
        args.root = Path(source_config["dataset_root"])
    args.rank = args.rank if args.rank is not None else source_config.get("rank", 8)
    # 优先级：CLI明确值 > 来源adapter值 > 默认8；is not None用于区分是否明确传参。
    args.alpha = args.alpha if args.alpha is not None else source_config.get("alpha", 16.0)
    args.lr = args.lr if args.lr is not None else source_config.get("learning_rate", 1e-4)
    if source_config and (args.rank != source_config["rank"] or args.alpha != source_config["alpha"]):
        p.error("rank and alpha must match the source adapter")
    if (min(args.steps, args.eval_every, args.log_every) < 1
            or (args.validation_samples is not None and args.validation_samples < 1)):
        p.error("steps, validation-samples, eval-every and log-every must be positive")
    if args.eval_action_steps < 1:
        p.error("--eval-action-steps must be positive")
    if min(args.gradient_accumulation_steps, args.chunk_start_stride) < 1:
        p.error("gradient-accumulation-steps and chunk-start-stride must be positive")
    if args.save_every < 0 or args.rank < 1 or min(args.lr, args.alpha, args.chunk_mib) <= 0:
        p.error("Invalid save interval, rank, learning rate, alpha or loading chunk size")
    if args.gpu_layers < 0:
        p.error("--gpu-layers must be nonnegative")
    if args.cpu_threads < 1:
        p.error("--cpu-threads must be positive")
    if not all(np.isfinite(value) for value in (args.lr, args.alpha, args.chunk_mib)):
        p.error("lr, alpha and chunk-mib must be finite")
    args.root = args.root.expanduser().resolve()
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
