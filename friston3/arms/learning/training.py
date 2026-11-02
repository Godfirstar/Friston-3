#!/usr/bin/env python
"""Shared PI0.5 loading, dataset, LoRA and flow-matching operations."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import checkpoint_reader, loader

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
          'wrist_flex', 'wrist_roll', 'gripper']
STATE, ACTION = 'observation.state', 'action'
PALI = 'paligemma_with_expert.paligemma'
EXPERT = 'paligemma_with_expert.gemma_expert.model'
PROJECTIONS = ('action_in_proj', 'action_out_proj', 'time_mlp_in', 'time_mlp_out')
LORA_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
TASK = 'Pick up the cube and place it in the marked area.'


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def event(label: str, data: Any) -> None:
    print(label, json.dumps(data, ensure_ascii=False, allow_nan=False), flush=True)


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def field(row: dict, name: str) -> Any:
    """Accept flattened v3 metadata; also accept genuinely nested equivalents."""
    if name in row:
        return row[name]
    value = row
    for part in name.split('/'):
        value = value[part]
    return value


def names(feature: dict) -> list[str]:
    result = feature.get('names')
    if isinstance(result, dict):
        result = result.get('motors')
    if not isinstance(result, list):
        raise ValueError('State/action features must declare joint names')
    return [str(name).removesuffix('.pos') for name in result]


def load_rows(root: Path) -> tuple[dict, dict[int, list[dict]], dict[int, dict]]:
    """Read local scalar/action Parquet and v3 episode locators, not video pixels."""
    import pyarrow.parquet as pq
    info = read_json(root / 'meta' / 'info.json')
    for key in (STATE, ACTION):
        if names(info['features'][key]) != JOINTS:
            raise ValueError(f'{key} joint order does not match {JOINTS}')
    groups: dict[int, list[dict]] = {}
    task_path = root / 'meta' / 'tasks.parquet'
    tasks = {}
    if task_path.is_file():
        for item in pq.read_table(task_path).to_pylist():
            task_id, task = int(item['task_index']), item['task']
            if task_id in tasks or not isinstance(task, str) or not task.strip():
                raise ValueError('Dataset task labels must be nonempty and uniquely indexed')
            tasks[task_id] = task.strip()
    info = dict(info, tasks=tasks)
    columns = [STATE, ACTION, 'episode_index', 'frame_index', 'timestamp', 'task_index']
    for path in sorted((root / 'data').rglob('*.parquet')):
        available = pq.read_schema(path).names
        selected = [name for name in columns if name in available]
        for row in pq.read_table(path, columns=selected).to_pylist():
            task_id = row.get('task_index')
            if tasks:
                if task_id is None and len(tasks) == 1:
                    task_id = next(iter(tasks))
                if task_id not in tasks:
                    raise ValueError(f'Unknown task_index {task_id} in {path}')
                row['task'] = tasks[int(task_id)]
            groups.setdefault(int(row['episode_index']), []).append(row)
    if not groups:
        raise ValueError(f'No recorded Parquet rows found in {root / "data"}')
    for episode, rows in groups.items():
        rows.sort(key=lambda row: int(row['frame_index']))
        if [int(row['frame_index']) for row in rows] != list(range(len(rows))):
            raise ValueError(f'Episode {episode}: duplicate or noncontiguous frame_index')
        timestamps = np.asarray([row['timestamp'] for row in rows], dtype=float)
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError(f'Episode {episode}: timestamps must be finite and increasing')
        for key in (STATE, ACTION):
            array = np.asarray([row[key] for row in rows], dtype=np.float32)
            if array.shape != (len(rows), 6) or not np.isfinite(array).all():
                raise ValueError(f'Episode {episode}: invalid {key}, shape {array.shape}')
    if sum(map(len, groups.values())) != int(info['total_frames']):
        raise ValueError('Parquet row count differs from meta/info.json total_frames')
    episodes: dict[int, dict] = {}
    for path in sorted((root / 'meta' / 'episodes').rglob('*.parquet')):
        for row in pq.read_table(path).to_pylist():
            eid = int(row['episode_index'])
            if eid in episodes:
                raise ValueError(f'Duplicate episode locator for {eid}')
            episodes[eid] = row
    if set(episodes) != set(groups):
        raise ValueError('This reader requires finalized v3 meta/episodes/*.parquet for every episode')
    for eid, rows in groups.items():
        if 'length' in episodes[eid] and int(episodes[eid]['length']) != len(rows):
            raise ValueError(f'Episode {eid}: metadata length differs from saved rows')
    return info, groups, episodes


def training_stats(groups: dict[int, list[dict]], train_ids: list[int], *, require_nonzero=True) -> dict:
    """Statistics use training episodes ONLY; never the whole-dataset sidecar."""
    stats = {}
    for key in (STATE, ACTION):
        array = np.asarray([row[key] for eid in train_ids for row in groups[eid]], dtype=np.float64)
        low, high = np.quantile(array, [0.01, 0.99], axis=0, method='linear')
        if require_nonzero and np.any(high.astype(np.float32) <= low.astype(np.float32)):
            raise ValueError(f'{key}: zero/inverted quantile span; no bounds will be fabricated')
        stats[key] = {'q01': low.tolist(), 'q99': high.tolist(), 'count': len(array)}
    return stats


def adapter_statistics(config: dict, model_dir: Path, rank: int, alpha: float) -> dict:
    """Validate the adapter's coordinate system before loading model weights."""
    model_config = read_json(model_dir / 'config.json')
    expected = {
        'format': 'friston_pi05_expert_lora_v1',
        'joint_order': JOINTS,
        'model_slots': list(range(6)),
        'supervised_action_dims': 6,
        'horizon': int(model_config['chunk_size']),
        'max_action_dim': int(model_config['max_action_dim']),
        'base_config_sha256': fingerprint(model_dir / 'config.json'),
        'loader_sha256': fingerprint(Path(loader.__file__)),
        'rank': rank, 'alpha': alpha,
        'full_training_modules': list(PROJECTIONS),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Adapter {key} differs from the current training configuration')
    if type(config.get('rank')) is not int or config['rank'] < 1:
        raise ValueError('Adapter LoRA rank must be a positive integer')
    if (type(config.get('alpha')) not in (int, float)
            or not math.isfinite(config['alpha']) or config['alpha'] <= 0):
        raise ValueError('Adapter LoRA alpha must be finite and positive')
    stats = config.get('statistics')
    if not isinstance(stats, dict):
        raise ValueError('Adapter statistics are missing')
    for key in (STATE, ACTION):
        entry = stats.get(key)
        if not isinstance(entry, dict) or not {'q01', 'q99'}.issubset(entry):
            raise ValueError(f'Adapter {key} statistics are missing')
        low, high = (np.asarray(entry[name], dtype=np.float32) for name in ('q01', 'q99'))
        if (low.shape != (6,) or high.shape != (6,)
                or not np.isfinite(low).all() or not np.isfinite(high).all()
                or np.any(high <= low)):
            raise ValueError(f'Adapter {key} statistics need six finite, positive quantile spans')
    return copy.deepcopy(stats)


def load_adapter_weights(parameters: dict[str, nn.Parameter], directory: Path) -> None:
    """Load only learned tensors; optimizer and RNG restoration belong to resume."""
    from safetensors.torch import load_file

    saved = load_file(str(directory / 'trainable.safetensors'), device='cpu', backend='pread')
    if set(saved) != set(parameters):
        raise ValueError('Adapter parameter keys differ from current trainable modules')
    for name, parameter in parameters.items():
        if saved[name].shape != parameter.shape or not torch.isfinite(saved[name]).all().item():
            raise ValueError(f'Invalid adapter tensor: {name}')
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(saved[name].to(device=parameter.device, dtype=parameter.dtype))


def normalize(value: Any, stats: dict) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    lo, hi = np.asarray(stats['q01'], dtype=np.float32), np.asarray(stats['q99'], dtype=np.float32)
    return 2 * (array - lo) / (hi - lo) - 1  # No clipping.


def video_locator(root: Path, info: dict, meta: dict, key: str, row: dict) -> tuple[Path, float]:
    prefix = f'videos/{key}'
    chunk = int(field(meta, prefix + '/chunk_index'))
    file_index = int(field(meta, prefix + '/file_index'))
    start = float(field(meta, prefix + '/from_timestamp'))
    target = start + float(row['timestamp'])
    template = info.get('video_path') or 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'
    relative = template.format(video_key=key, chunk_index=chunk, file_index=file_index,
                               episode_index=int(row['episode_index']))
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, target


def decode_frame(path: Path, timestamp: float, fps: float) -> dict:
    """Find the nearest presentation timestamp; do not equate episode and file IDs."""
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(max(0, int(timestamp / float(stream.time_base))),
                       stream=stream, backward=True, any_frame=False)
        best = None
        best_time = None
        distance = float('inf')
        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            t = float(frame.pts * frame.time_base)
            if abs(t - timestamp) < distance:
                best, best_time, distance = frame, t, abs(t - timestamp)
            if t >= timestamp:
                break
        if best is None:
            raise RuntimeError(f'No decodable frame near {timestamp} in {path}')
        # A time-index integrity check, not a robot motion threshold.
        if distance > 0.5 / fps + 1e-3:
            raise ValueError(f'Video/row mismatch: wanted {timestamp}s, nearest frame {best_time}s')
        rgb = best.to_ndarray(format='rgb24')
    return {'path': str(path), 'requested_time_s': timestamp, 'decoded_time_s': best_time,
            'absolute_difference_s': distance, 'rgb': rgb,
            'hardware_synchronization_verified': False}


class LoRALinear(nn.Module):
    """FP32 trainable low-rank update; original Linear weights remain frozen.

    weight/bias properties preserve Gemma's dtype inspections. No quantization,
    merging, PEFT dependency, offload, or dropout is introduced here.
    """
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.rank, self.alpha, self.scaling = rank, alpha, alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, value):
        original = self.base(value)
        update = F.linear(F.linear(value.float(), self.lora_A), self.lora_B) * self.scaling
        return original + update.to(original.dtype)


def attach_trainables(model: nn.Module, rank: int, alpha: float) -> list[str]:
    model.requires_grad_(False)
    expert = model.get_submodule(EXPERT)
    targets = []
    for relative, module in list(expert.named_modules()):
        if isinstance(module, nn.Linear) and relative.split('.')[-1] in LORA_TARGETS:
            parent_name, _, leaf = relative.rpartition('.')
            parent = expert.get_submodule(parent_name) if parent_name else expert
            setattr(parent, leaf, LoRALinear(module, rank, alpha))
            targets.append(f'{EXPERT}.{relative}')
    expected = len(expert.layers) * len(LORA_TARGETS)
    if len(targets) != expected:
        raise ValueError(f'Expected {expected} expert attention projections, found {len(targets)}')
    for name in PROJECTIONS:
        module = model.get_submodule(name)
        module.to(dtype=torch.float32).requires_grad_(True)
        module.train()
    expert.train()
    # No HF gradient checkpointing: some releases disable/ignore the cache when
    # it is enabled. The short suffix + LoRA is the memory-saving path here.
    expert.gradient_checkpointing = False
    return targets


def trainable_dict(model: nn.Module) -> dict[str, nn.Parameter]:
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def load_training_model(args, loader=loader, report: dict | None = None):
    """Reuse streamed loading; dispatch ONLY the frozen PaliGemma submodel."""
    report = {} if report is None else report
    from accelerate import dispatch_model
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    model = loader.build_meta_model(args.model_dir)
    mapping_path = args.device_map
    if mapping_path is not None and mapping_path.is_file():
        raw = read_json(mapping_path)
        device_map = {k: loader.normalize_device(v) for k, v in raw.get('device_map', raw).items()}
    elif args.device_map:
        raise FileNotFoundError(mapping_path)
    else:
        device_map = loader.default_device_map(model, args.gpu, getattr(args, 'gpu_layers', 8))
    for value in device_map.values():
        target = loader.torch_device(value)
        if target.type not in ('cpu', 'cuda') or (target.type == 'cuda' and target.index != args.gpu):
            raise ValueError('This training proof uses one GPU plus CPU storage, not disk/multiple GPUs')
    # Only frozen PaliGemma may use offload. Check before loading huge tensors.
    for name, _ in model.named_parameters():
        if name.startswith(EXPERT + '.') or name.split('.')[0] in PROJECTIONS:
            if loader.torch_device(loader.placement(name, device_map)) != device:
                raise ValueError(f'Train graph must remain resident on {device}: {name}')
    proxy = SimpleNamespace(model=model)
    fixer = lambda entries: PI05Policy._fix_pytorch_state_dict_keys(proxy, entries, model.config)
    with checkpoint_reader():
        report['loading'] = loader.stream_load(
            model, loader.checkpoint_sources(args.model_dir), device_map, fixer, args.chunk_mib)
    prefix_map = {key[len(PALI) + 1:]: value for key, value in device_map.items()
                  if key.startswith(PALI + '.')}
    if not prefix_map:
        raise ValueError('Device map must explicitly describe PaliGemma submodules')
    pg = model.paligemma_with_expert
    pg.paligemma = dispatch_model(pg.paligemma, device_map=prefix_map,
                                  main_device=device, offload_buffers=False, force_hooks=True)
    model.eval().requires_grad_(False)
    model.config.freeze_vision_encoder = True
    model.config.train_expert_only = True
    pg.freeze_vision_encoder = True
    pg.train_expert_only = True
    for name in (EXPERT, *PROJECTIONS):
        module = model.get_submodule(name)
        if any(hasattr(child, '_hf_hook') for child in module.modules()):
            raise RuntimeError(f'Unexpected inference hook in trainable branch: {name}')
        if any(p.is_meta or p.device != device for p in module.parameters()):
            raise RuntimeError(f'Nonresident tensor in trainable branch: {name}')
    report['loading'].update(seconds=time.perf_counter()-start,
                             device_map={k: str(v) for k, v in device_map.items()},
                             offload_scope='frozen_paligemma_only', memory=loader.snapshot(device))
    event('LOAD_OK', report['loading']['memory'])
    return model, device


def make_frozen_prefix(model, inputs: dict) -> tuple[torch.Tensor, Any]:
    """Mirror the installed inference prefix, but create ordinary detached tensors."""
    impl = importlib.import_module(type(model).__module__)
    make_mask = getattr(impl, 'make_att_2d_masks')
    to4d = getattr(model, '_prepare_attention_masks_4d', None)
    if to4d is None:
        to4d = getattr(impl, 'prepare_attention_masks_4d')
    # Loader's image helper uses inference_mode. Clone outside it so no inference
    # tensor can accidentally be saved for backward inside the expert.
    def ordinary(x):
        if isinstance(x, torch.Tensor):
            return x.detach().clone()
        if isinstance(x, list):
            return [ordinary(v) for v in x]
        return x
    inputs = {k: ordinary(v) for k, v in inputs.items()}
    with torch.no_grad():
        embeds, pad, att = model.embed_prefix(
            inputs['images'], inputs['img_masks'], inputs['tokens'], inputs['masks'])
        mask = to4d(make_mask(pad, att))
        positions = torch.cumsum(pad, dim=1) - 1
        pg = model.paligemma_with_expert
        pg.paligemma.model.language_model.config._attn_implementation = 'eager'
        _, cache = pg.forward(attention_mask=mask, position_ids=positions,
                              past_key_values=None, inputs_embeds=[embeds, None], use_cache=True)
    if cache is None:
        raise RuntimeError('Prefix did not return a KV cache')
    if hasattr(cache, 'layers'):
        tensors = [getattr(layer, attr, None) for layer in cache.layers for attr in ('keys', 'values')]
    elif hasattr(cache, 'key_cache'):
        tensors = list(cache.key_cache) + list(cache.value_cache)
    else:
        tensors = [t for pair in cache for t in pair]
    tensors = [t for t in tensors if isinstance(t, torch.Tensor)]
    if not tensors or any(t.requires_grad or torch.is_inference(t) for t in tensors):
        raise RuntimeError('Prefix KV must be ordinary detached tensors, not inference tensors')
    return pad.detach(), cache


def flow_inputs(actions: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor):
    t = timestep[:, None, None]
    return t * noise + (1 - t) * actions, noise - actions


def velocity(model, pad, cache, noisy, timestep):
    # Releases differ in whether denoise_step clones its cache. Never let a
    # training/evaluation call mutate the stored, frozen prefix cache.
    return model.denoise_step(prefix_pad_masks=pad, past_key_values=copy.deepcopy(cache),
                               x_t=noisy, timestep=timestep)


def flow_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if predicted.shape != target.shape:
        raise ValueError(f'Velocity shape {predicted.shape} != target {target.shape}')
    # The actual embodiment has SIX supervised channels. Padding is not a joint.
    return F.mse_loss(predicted[..., :6].float(), target[..., :6].float())


def check_gradients(parameters: dict[str, nn.Parameter]) -> dict:
    missing, nonfinite, nonzero = [], [], []
    norm2 = 0.0
    for name, p in parameters.items():
        if p.grad is None:
            missing.append(name)
        elif not torch.isfinite(p.grad).all().item():
            nonfinite.append(name)
        else:
            magnitude = p.grad.detach().float().square().sum().item()
            norm2 += magnitude
            if magnitude > 0:
                nonzero.append(name)
    if missing or nonfinite or not nonzero:
        raise RuntimeError(f'Invalid gradients: missing={missing[:6]}, nonfinite={nonfinite[:6]}, '
                           f'nonzero_count={len(nonzero)}')
    return {'gradient_l2': math.sqrt(norm2), 'nonzero_gradient_tensors': len(nonzero),
            'trainable_tensors': len(parameters), 'missing_gradients': 0, 'finite': True,
            'note': 'LoRA A can have a zero first-step gradient because B starts at zero'}
