#!/usr/bin/env python
"""Train or resume the local SO-101 PI0.5 adapter from LeRobot datasets."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from . import assets_root, loader
from . import training as proof


def emit(name, payload):
    print(name, json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)


def sample_data(root, info, groups, episode_meta, stats, args, episode, start):
    rows = groups[episode]
    horizon = args.horizon
    if start < 0 or start + horizon > len(rows):
        raise ValueError("Action chunk crosses an episode boundary")
    row = rows[start]
    task = row.get("task") or args.task
    if not task:
        raise ValueError("Dataset task metadata is missing; provide --task for this legacy dataset")
    if any((item.get("task") or args.task) != task for item in rows[start:start + horizon]):
        raise ValueError("Action chunk crosses a task-label boundary")
    images, videos = {}, {}
    for key in args.image_keys:
        video_path, video_time = proof.video_locator(root, info, episode_meta[episode], key, row)
        decoded = proof.decode_frame(video_path, video_time, float(info["fps"]))
        images[key] = decoded.pop("rgb")
        videos[key] = decoded
    state = proof.normalize(row[proof.STATE], stats[proof.STATE])
    actions = proof.normalize([r[proof.ACTION] for r in rows[start:start + horizon]], stats[proof.ACTION])
    return state, actions, images, videos, task


def prepare_inputs(model, args, device, state, images, task):
    from lerobot.policies.pi05 import processor_pi05 as processor
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from transformers import AutoTokenizer

    if not hasattr(args, "_tokenizer"):
        args._tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), local_files_only=True)
        args._tokenizer.padding_side = "right"
    state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    state = torch.nn.functional.pad(state, (0, model.config.max_state_dim - state.shape[-1]))
    transition = {
        processor.TransitionKey.OBSERVATION: {processor.OBS_STATE: state},
        processor.TransitionKey.COMPLEMENTARY_DATA: {"task": [task]},
    }
    step = processor.Pi05PrepareStateTokenizerProcessorStep(max_state_dim=model.config.max_state_dim)
    prompt = step(transition)[processor.TransitionKey.COMPLEMENTARY_DATA]["task"]
    encoded = args._tokenizer(prompt, padding="max_length", truncation=True,
                             max_length=model.config.tokenizer_max_length, return_tensors="pt")
    batch = {key: torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
             for key, rgb in images.items()}
    proxy = SimpleNamespace(config=model.config, parameters=lambda: iter([torch.empty(0, device=device)]))
    with torch.no_grad():
        processed, masks = PI05Policy._preprocess_images(proxy, batch)
    return {"images": processed, "img_masks": masks,
            "tokens": encoded["input_ids"].to(device),
            "masks": encoded["attention_mask"].to(device=device, dtype=torch.bool)}


def make_flow(model, actions_6, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    actions = torch.zeros(
        (1, model.config.chunk_size, model.config.max_action_dim),
        dtype=torch.float32, device=device,
    )
    actions[..., :6] = torch.as_tensor(actions_6, dtype=torch.float32, device=device)
    noise = model.sample_noise(actions.shape, device)
    timestep = model.sample_time(1, device)
    noisy, target = proof.flow_inputs(actions, noise, timestep)
    return noisy, target, timestep


def compute_loss(model, inputs, actions_6, device, seed=None):
    pad, cache = proof.make_frozen_prefix(model, inputs)
    noisy, target, timestep = make_flow(model, actions_6, device, seed)
    prediction = proof.velocity(model, pad, cache, noisy, timestep)
    loss = proof.flow_loss(prediction, target)
    return loss, (pad, cache, noisy, target, timestep, prediction)


class FrozenPrefixCache:
    """LRU of detached prefix storage, bounded in bytes; never serialized.

    The budget bounds retained tensors, including shared-storage views once per
    entry, on the selected storage device. CPU entries retain the original
    dtype and each hit creates an independent execution-device copy. Active
    forward/backward temporaries and CUDA reserved memory are not budgeted.
    """

    def __init__(self, max_bytes, storage_device="cuda"):
        self.max_bytes = max_bytes
        self.storage_device = storage_device
        self.entries = OrderedDict()
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
    key = (episode, int(groups[episode][start]["frame_index"]))
    value = prefix_cache.get(key, device)
    if value is None:
        state, actions, images, _, task = sample_data(
            root, info, groups, episode_meta, stats, args, episode, start)
        inputs = prepare_inputs(model, args, device, state, images, task)
        pad, cache = proof.make_frozen_prefix(model, inputs)
        value = (pad, cache, actions, task)
        prefix_cache.put(key, value)
    return value


def chunk_start_indices(length, horizon, stride):
    last = length - horizon
    if last < 0 or min(horizon, stride) < 1:
        raise ValueError("Chunk starts require a positive horizon/stride and a complete window")
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def configure_training(args, resume_config):
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
    args.training_contract = contract


def episode_frame_range(value):
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


def choose_val_samples(val_ids, groups, horizon, count, seed, sampling="uniform"):
    rng = random.Random(seed)
    choices = []
    valid = [ep for ep in val_ids if len(groups[ep]) >= horizon]
    if not valid:
        raise ValueError("No held-out episode can supply a complete action chunk")
    if sampling == "random":
        for i in range(count):
            ep = valid[i % len(valid)]
            choices.append((ep, rng.randint(0, len(groups[ep]) - horizon)))
        return choices
    minimum = sum(min(2, len(groups[ep]) - horizon + 1) for ep in valid)
    if count < minimum:
        raise ValueError(f"Uniform validation needs at least {minimum} samples to cover each episode's endpoints")
    counts = {ep: min(2, len(groups[ep]) - horizon + 1) for ep in valid}
    for i in range(count - minimum):
        counts[valid[i % len(valid)]] += 1
    for ep in valid:
        starts = np.linspace(0, len(groups[ep]) - horizon, counts[ep]).round().astype(int)
        choices.extend((ep, int(start)) for start in starts)
    return choices


def configure_validation(args, groups, val_ids, horizon, resume_config):
    """Keep resumed validation coordinates fixed, including pre-contract checkpoints."""
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
            if ep not in val_ids:
                raise ValueError("Saved validation sample is outside the held-out split")
            positions = {int(row["frame_index"]): i for i, row in enumerate(groups[ep])}
            start = positions.get(frame)
            if start is None or start + horizon > len(groups[ep]):
                raise ValueError("Saved validation sample no longer supplies a complete action chunk")
            samples.append((ep, start))
        if not samples:
            raise ValueError("Saved validation sample list is empty")
        args.validation_sampling, args.eval_seed = saved["sampling"], saved["eval_seed"]
        contract = dict(saved)
    else:
        args.validation_sampling = args.validation_sampling or ("random" if resume_config else "uniform")
        args.eval_seed = 9000 if args.eval_seed is None else args.eval_seed
        count = args.validation_samples or max(10, 2 * len(val_ids))
        samples = choose_val_samples(val_ids, groups, horizon, count, args.seed + 1000,
                                     args.validation_sampling)
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


def action_metrics(predicted, actual, state, stats):
    """Report absolute action error against commands, plus the hold-current baseline."""
    predicted = np.asarray(predicted, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    lo, hi = (np.asarray(stats[proof.ACTION][key], dtype=np.float32) for key in ("q01", "q99"))
    physical = (predicted + 1) * (hi - lo) / 2 + lo
    hold = np.broadcast_to(np.asarray(state, dtype=np.float32), actual.shape)
    errors, hold_errors = np.abs(physical - actual), np.abs(hold - actual)
    return {"mae_per_joint": errors.mean(axis=0).tolist(),
            "hold_mae_per_joint": hold_errors.mean(axis=0).tolist(),
            "first10_mae_per_joint": errors[:10].mean(axis=0).tolist(),
            "first10_hold_mae_per_joint": hold_errors[:10].mean(axis=0).tolist(),
            "normalized_mae_per_joint": np.abs(predicted - proof.normalize(actual, stats[proof.ACTION])).mean(axis=0).tolist(),
            "normalized_hold_mae_per_joint": np.abs(proof.normalize(hold, stats[proof.ACTION])
                                                     - proof.normalize(actual, stats[proof.ACTION])).mean(axis=0).tolist()}


def evaluate(model, loader, args, device, root, info, groups, episode_meta,
             stats, samples):
    values, details = [], []
    for index, (episode, start) in enumerate(samples):
        state, actions, images, videos, task = sample_data(
            root, info, groups, episode_meta, stats, args, episode, start
        )
        inputs = prepare_inputs(model, args, device, state, images, task)
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(args.eval_seed + index)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.eval_seed + index)
            with torch.no_grad():
                loss, cache_items = compute_loss(
                    model, inputs, actions, device, seed=args.eval_seed + index
                )
                value = float(loss.item())
                del cache_items, loss
                generated = None
                if args.eval_actions:
                    # A separate fixed seed prevents flow-noise sampling from changing
                    # generation and fork_rng keeps evaluation out of training RNG state.
                    torch.manual_seed(args.eval_seed + index)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(args.eval_seed + index)
                    prediction = model.sample_actions(**inputs, num_steps=args.eval_action_steps)
                    generated = action_metrics(
                        prediction[0, :, :6].float().cpu().numpy(),
                        [row[proof.ACTION] for row in groups[episode][start:start + args.horizon]],
                        groups[episode][start][proof.STATE], stats)
                    del prediction
        values.append(value)
        details.append({
            "episode": episode, "frame_index": int(groups[episode][start]["frame_index"]), "loss": value,
            "task": task,
            "video_time_delta_seconds": {key: value["absolute_difference_s"] for key, value in videos.items()},
        })
        if generated is not None:
            details[-1]["generated_actions"] = generated
        del inputs
    result = {
        "mean_flow_matching_mse_6d": float(np.mean(values)),
        "std_flow_matching_mse_6d": float(np.std(values)),
        "sample_count": len(values), "samples": details,
    }
    if args.eval_actions:
        result["generated_actions"] = {
            "joint_order": proof.JOINTS, "units": ["degrees"] * 5 + ["percent"],
            "num_steps": args.eval_action_steps, "rtc": False,
            **{key: np.mean([row["generated_actions"][key] for row in details], axis=0).tolist()
               for key in details[0]["generated_actions"]}}
    return result


def save_checkpoint(model, optimizer, args, stats, train_ids, val_ids, step, output, history):
    from safetensors.torch import save_file

    directory = output / f"checkpoint_step_{step:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    params = proof.trainable_dict(model)
    save_file(
        {k: p.detach().cpu().contiguous().clone() for k, p in params.items()},
        str(directory / "trainable.safetensors"),
    )
    torch.save(optimizer.state_dict(), directory / "optimizer.pt")
    import random
    import numpy as np
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
        "format": "friston_pi05_expert_lora_v1",
        "base_model_dir": str(args.model_dir),
        "base_config_sha256": proof.fingerprint(args.model_dir / "config.json"),
        "loader_sha256": proof.fingerprint(Path(loader.__file__)),
        "rank": args.rank, "alpha": args.alpha,
        "target_modules": args.target_modules,
        "full_training_modules": list(proof.PROJECTIONS),
        "statistics": stats,
        "dataset_statistics": args.dataset_statistics,
        "normalization_source": args.normalization_source,
        "initialized_from": args.initialized_from,
        "training_episodes": train_ids, "validation_episodes": val_ids,
        "episode_frames": args.episode_frames,
        "validation_contract": args.validation_contract,
        "training_contract": args.training_contract,
        "micro_steps_completed": step * args.gradient_accumulation_steps,
        "prefix_cache_mib": args.prefix_cache_mib,
        "prefix_cache_device": args.prefix_cache_device,
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "task": args.tasks[0] if len(args.tasks) == 1 else "", "tasks": args.tasks,
        "image_key": args.image_keys[0], "image_keys": args.image_keys,
        "dataset_root": str(args.root), "dataset_fps": args.dataset_fps,
        "joint_order": proof.JOINTS, "model_slots": list(range(6)),
        "supervised_action_dims": 6, "horizon": args.horizon,
        "max_action_dim": args.max_action_dim,
        "steps_completed": step, "batch_size": 1,
        "effective_batch_size": args.gradient_accumulation_steps, "learning_rate": args.lr,
        "training_scope": "random_episode_random_chunk",
        "not_standard_peft_or_lerobot_checkpoint": True,
    }
    proof.write_json(directory / "adapter_config.json", config)
    return str(directory)


def run(args):
    torch.set_num_threads(args.cpu_threads)
    root = args.root.resolve()
    info, groups, episode_meta = proof.load_rows(root)
    resume_config = proof.read_json(args.resume / "adapter_config.json") if args.resume else None
    source_config = (resume_config if args.resume else
                     proof.read_json(args.init_from / "adapter_config.json") if args.init_from else None)
    configure_training(args, resume_config)
    if args.train_episodes is None:
        args.train_episodes = (resume_config or {}).get("training_episodes")
    train_ids = None if args.train_episodes is None else sorted(set(args.train_episodes))
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
    args.dataset_statistics = proof.training_stats(groups, train_ids, require_nonzero=source_config is None)
    stats = (proof.adapter_statistics(source_config, args.model_dir, args.rank, args.alpha)
             if source_config is not None else args.dataset_statistics)
    args.initialized_from = (str(args.init_from.resolve()) if args.init_from else
                             (resume_config or {}).get("initialized_from"))
    args.normalization_source = ("initialized_adapter" if args.init_from else
                                 (resume_config or {}).get("normalization_source", "training_episodes_only"))
    raw_config = proof.read_json(args.model_dir / "config.json")
    args.horizon = int(raw_config["chunk_size"])
    args.max_action_dim = int(raw_config["max_action_dim"])
    for ep in train_ids + val_ids:
        if len(groups[ep]) < args.horizon:
            raise ValueError(f"Episode {ep} shorter than horizon {args.horizon}")
    start_pools = {ep: chunk_start_indices(len(groups[ep]), args.horizon, args.chunk_start_stride)
                   for ep in train_ids}
    if args.image_keys is None:
        args.image_keys = [key for key in raw_config["input_features"]
                           if key.startswith("observation.images.") and key in info["features"]]
    for key in args.image_keys:
        if key not in info["features"] or key not in raw_config["input_features"]:
            raise ValueError(f"Camera feature {key} must exist in both dataset and model")
    if not args.image_keys:
        raise ValueError("No camera features shared by dataset and model")
    args.tasks = sorted({row.get("task") or args.task or ""
                         for ep in train_ids + val_ids for row in groups[ep]})
    if "" in args.tasks:
        raise ValueError("Dataset is missing task labels; provide --task only for legacy single-task data")
    args.dataset_fps = float(info["fps"])
    if resume_config:
        saved_images = resume_config.get("image_keys", [resume_config.get("image_key")])
        if saved_images != args.image_keys:
            raise ValueError("Resume camera keys differ; continue with the checkpoint's --image-keys")
        if resume_config.get("dataset_statistics", resume_config.get("statistics")) != args.dataset_statistics:
            raise ValueError("Resume dataset statistics differ from the saved checkpoint")
        saved_tasks = resume_config.get("tasks") or [resume_config.get("task")]
        if sorted(saved_tasks) != args.tasks:
            raise ValueError("Resume dataset tasks differ from the saved checkpoint")
        if resume_config.get("training_episodes") != train_ids or resume_config.get("validation_episodes") != val_ids:
            raise ValueError("Resume checkpoint train/validation split differs")

    val_samples = configure_validation(args, groups, val_ids, args.horizon, resume_config)
    emit("TRAIN_DATA_OK", {
        "train_episodes": train_ids, "validation_episodes": val_ids,
        "episode_frames": args.episode_frames,
        "train_frames": sum(len(groups[e]) for e in train_ids),
        "validation_frames": sum(len(groups[e]) for e in val_ids),
        "horizon": args.horizon, "supervised_action_dims": 6,
        "statistics_scope": args.normalization_source,
        "image_keys": args.image_keys, "tasks": args.tasks,
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
    model, device = proof.load_training_model(args, loader, report)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    targets = proof.attach_trainables(model, args.rank, args.alpha)
    args.target_modules = targets
    params = proof.trainable_dict(model)
    if args.prefix_cache_mib and any(p.requires_grad for p in model.paligemma_with_expert.paligemma.parameters()):
        raise ValueError("Prefix caching requires a frozen PaliGemma backbone")
    if source_config is not None:
        if targets != source_config.get("target_modules"):
            raise ValueError("Adapter target modules differ from the current model")
        proof.load_adapter_weights(params, args.resume or args.init_from)
        if args.init_from:
            emit("INITIALIZED_FROM_ADAPTER", {
                "checkpoint": args.initialized_from, "optimizer_state_restored": False,
                "rng_state_restored": False, "normalization_source": args.normalization_source,
            })
    emit("TRAINABLE_READY", {
        "parameters": sum(p.numel() for p in params.values()),
        "tensors": len(params), "lora_modules": len(targets),
        "rank": args.rank, "alpha": args.alpha,
    })
    optimizer = torch.optim.AdamW(
        list(params.values()), lr=args.lr, weight_decay=0.0, foreach=False
    )
    start_step = 0
    history = []
    if args.resume is not None:
        resume_dir = args.resume.resolve()
        optimizer_state = torch.load(resume_dir / "optimizer.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(optimizer_state)
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        state_path = resume_dir / "training_state.pt"
        if state_path.is_file():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
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
        emit("RESUME_OK", {
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
    torch.cuda.reset_peak_memory_stats(device)
    prefix_cache = FrozenPrefixCache(int(args.prefix_cache_mib * 1024**2), args.prefix_cache_device)
    started = time.perf_counter()

    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        micro_samples, loss_value = [], 0.0
        for micro in range(args.gradient_accumulation_steps):
            episode = rng.choice(train_ids)
            start = rng.choice(start_pools[episode])
            pad, cache, actions, task = training_prefix(
                model, device, prefix_cache, root, info, groups, episode_meta, stats, args, episode, start)
            noisy, target, timestep = make_flow(model, actions, device)
            prediction = proof.velocity(model, pad, cache, noisy, timestep)
            loss = proof.flow_loss(prediction, target)
            if not loss.requires_grad or not torch.isfinite(loss).item():
                raise RuntimeError(f"Step {step}, microstep {micro + 1}: loss is nonfinite or has no grad graph")
            loss_value += float(loss.item()) / args.gradient_accumulation_steps
            (loss / args.gradient_accumulation_steps).backward()
            micro_samples.append({
                "micro_step": (step - 1) * args.gradient_accumulation_steps + micro + 1,
                "episode": episode, "frame_index": int(groups[episode][start]["frame_index"]), "task": task})
            del pad, cache, actions, noisy, target, timestep, prediction, loss
        gradient = proof.check_gradients(params)
        if any(p.grad is not None for p in model.paligemma_with_expert.paligemma.parameters()):
            raise RuntimeError("Frozen PaliGemma unexpectedly received gradients")
        optimizer.step()
        if any(not torch.isfinite(p).all().item() for p in params.values()):
            raise RuntimeError(f"Step {step}: nonfinite trainable parameter")
        entry = {
            "step": step, "train_loss": loss_value,
            "episode": micro_samples[0]["episode"], "frame_index": micro_samples[0]["frame_index"],
            "task": micro_samples[0]["task"], "micro_samples": micro_samples,
            "micro_steps_completed": step * args.gradient_accumulation_steps,
        }
        history.append(entry)
        report["history"] = history

        if step == 1 or step % args.log_every == 0:
            emit("TRAIN_STEP", {
                "step": step, "train_loss": loss_value,
                "episode": entry["episode"], "frame_index": entry["frame_index"],
                "micro_steps_completed": entry["micro_steps_completed"],
                "gradient_l2": gradient["gradient_l2"],
                "prefix_cache": prefix_cache.stats(),
            })

        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(
                model, loader, args, device, root, info, groups, episode_meta,
                stats, val_samples,
            )
            entry["validation"] = val
            emit("VALIDATION_OK", {
                "step": step, "train_loss": loss_value,
                "validation_mse": val["mean_flow_matching_mse_6d"],
                "validation_std": val["std_flow_matching_mse_6d"],
                "sample_count": val["sample_count"],
                **({"generated_actions": val["generated_actions"]} if args.eval_actions else {}),
            })
            args._sampler_rng_state = rng.getstate()
            checkpoint = save_checkpoint(
                model, optimizer, args, stats, train_ids, val_ids, step, args.output_dir, history
            )
            entry["checkpoint"] = checkpoint
            emit("CHECKPOINT_SAVED", {"step": step, "path": checkpoint})
        elif args.save_every > 0 and step % args.save_every == 0:
            args._sampler_rng_state = rng.getstate()
            checkpoint = save_checkpoint(
                model, optimizer, args, stats, train_ids, val_ids, step, args.output_dir, history
            )
            entry["checkpoint"] = checkpoint
            emit("CHECKPOINT_SAVED", {"step": step, "path": checkpoint})

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
    emit("FINETUNE_OK", {
        "steps_completed": args.steps,
        "micro_steps_completed": report["micro_steps_completed"],
        "elapsed_seconds": report["elapsed_seconds"],
        "final_checkpoint": report["final_checkpoint"],
        "training_peak_memory": report["training_peak_memory"],
        "robot_commands_sent": False,
        "prefix_cache": report["prefix_cache"],
    })


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, help="Local LeRobot v3 dataset; saved in new checkpoints")
    p.add_argument("--model-dir", type=Path, default=assets_root() / "pi05_base")
    p.add_argument("--tokenizer-dir", type=Path, default=assets_root() / "paligemma_tokenizer")
    p.add_argument("--device-map", type=Path, help="Explicit placement map; overrides --gpu-layers")
    p.add_argument("--image-keys", nargs="+", help="Default: every dataset camera also present in the model")
    p.add_argument("--image-key", help="Single-camera compatibility alias")
    p.add_argument("--task", help="Fallback label only for legacy data without task metadata")
    p.add_argument("--train-episodes", type=int, nargs="+",
                   help="Explicit training episode IDs; resume defaults to the checkpoint selection")
    p.add_argument("--val-episodes", type=int, nargs="+")
    p.add_argument("--episode-frames", type=episode_frame_range, action="append", metavar="EP=START:STOP",
                   help="Repeat for selected episodes; original zero-based rows, stop exclusive. Resume inherits saved ranges")
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
    source.add_argument("--resume", type=Path, help="Continue the same dataset, optimizer and RNG state")
    source.add_argument("--init-from", type=Path,
                        help="Initialize adapter weights/statistics for a new dataset, task or camera setup")
    args = p.parse_args(argv)
    if args.episode_frames is not None:
        ranges = {}
        for episode, bounds in args.episode_frames:
            if episode in ranges:
                p.error(f"Duplicate --episode-frames for episode {episode}")
            ranges[episode] = bounds
        args.episode_frames = ranges
    checkpoint = args.resume or args.init_from
    source_config = proof.read_json(checkpoint / "adapter_config.json") if checkpoint else {}
    if args.root is None:
        stored_root = source_config.get("dataset_root") if args.resume else None
        if stored_root:
            args.root = Path(stored_root)
        else:
            p.error("--root is required for a new run or an older checkpoint without dataset_root")
    args.rank = args.rank if args.rank is not None else source_config.get("rank", 8)
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
    args.root = args.root.expanduser().resolve()
    if args.output_dir is None:
        from datetime import datetime
        args.output_dir = assets_root() / "checkpoints" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
