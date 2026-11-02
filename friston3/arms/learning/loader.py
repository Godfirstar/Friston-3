#!/usr/bin/env python
r"""PI0.5: streamed loading, CPU offload, and real visual-input diagnostics.

Windows / LeRobot / Accelerate 1.15.0. No site-packages changes.

    python load_pi05_lowmem.py --smoke --num-steps 10
    python load_pi05_lowmem.py --camera 1 --state-zero --task "pick up the cube"
    python load_pi05_lowmem.py --image frame.jpg --state-zero

A single unnamed image/camera uses the first configured image feature. For
multiple views repeat --camera KEY=INDEX or --image KEY=PATH. Missing views are
handled by the INSTALLED official PI05Policy._preprocess_images implementation.

--state-zero is an explicitly synthetic, 32-wide placeholder, NOT a measured
robot state. --state-normalized requires values ALREADY normalized with the
checkpoint's training statistics and in its training order. This script does
not infer either from a base checkpoint. All outputs remain raw padded model
actions, WITHOUT robot-specific unnormalization or execution.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
import importlib.metadata
import json
from math import prod
from pathlib import Path
import struct
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any


GIB = 1024**3
ROOT = "paligemma_with_expert"
PALI = f"{ROOT}.paligemma"
LANG = f"{PALI}.model.language_model"
EXPERT = f"{ROOT}.gemma_expert"


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: Any


@dataclass(frozen=True)
class Source:
    file: Path
    key: str
    shape: tuple[int, ...]
    nbytes: int
    offset: int


def strip_model_prefix(key: str) -> str:
    return key.removeprefix("model.")


def checkpoint_sources(checkpoint: Path) -> list[Source]:
    """Inspect only file headers; no tensors or complete state dict are loaded."""
    weight_map = None
    if checkpoint.is_dir():
        index = checkpoint / "model.safetensors.index.json"
        single = checkpoint / "model.safetensors"
        if index.is_file():
            checkpoint = index
        elif single.is_file():
            checkpoint = single
        else:
            indices = sorted(checkpoint.glob("*.safetensors.index.json"))
            if len(indices) != 1:
                raise FileNotFoundError(f"No unambiguous Safetensors checkpoint in {checkpoint}")
            checkpoint = indices[0]
    if checkpoint.name.endswith(".index.json"):
        weight_map = json.loads(checkpoint.read_text(encoding="utf-8"))["weight_map"]
        files = [checkpoint.parent / name for name in sorted(set(weight_map.values()))]
    elif checkpoint.suffix == ".safetensors":
        files = [checkpoint]
    else:
        raise ValueError("This streaming loader requires .safetensors or its .index.json")

    result = []
    seen = set()
    for file in files:
        with file.open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
        for key, entry in header.items():
            if key == "__metadata__":
                continue
            if weight_map is not None and weight_map.get(key) != file.name:
                raise ValueError(f"Checkpoint index disagrees with shard: {file.name}: {key}")
            if key in seen:
                raise ValueError(f"Duplicate checkpoint key: {key}")
            seen.add(key)
            begin, end = entry["data_offsets"]
            result.append(Source(file, key, tuple(entry["shape"]), end - begin, begin))
    if weight_map is not None and set(weight_map) != seen:
        raise ValueError("Checkpoint index contains missing tensor entries")
    return result


def normalize_device(value: Any) -> Any:
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return value


def placement(name: str, device_map: dict[str, Any]) -> Any:
    matches = [key for key in device_map if key == "" or name == key or name.startswith(key + ".")]
    if not matches:
        raise ValueError(f"No device-map entry covers {name}")
    return normalize_device(device_map[max(matches, key=len)])


def torch_device(value: Any) -> Any:
    import torch

    return torch.device(f"cuda:{value}" if isinstance(value, int) else value)


def default_device_map(model: Any, gpu: int, gpu_layers: int = 4) -> dict[str, Any]:
    result = {
        f"{PALI}.model.vision_tower": gpu,
        f"{PALI}.model.multi_modal_projector": gpu,
        f"{LANG}.embed_tokens": gpu,
        f"{LANG}.rotary_emb": gpu,
        f"{PALI}.lm_head": gpu,
        f"{LANG}.norm": "cpu",
        EXPERT: gpu,
        f"{EXPERT}.lm_head": "cpu",
        "action_in_proj": gpu,
        "action_out_proj": gpu,
        "time_mlp_in": gpu,
        "time_mlp_out": gpu,
    }
    layers = model.paligemma_with_expert.paligemma.model.language_model.layers
    result.update({f"{LANG}.layers.{i}": gpu if i < gpu_layers else "cpu" for i in range(len(layers))})
    if getattr(model, "proprio_history_proj", None) is not None:
        result["proprio_history_proj"] = gpu
    return result


def snapshot(device: Any = None) -> dict[str, Any]:
    result = {}
    try:
        import psutil

        result["process_rss_gib"] = round(psutil.Process().memory_info().rss / GIB, 4)
        result["system_available_gib"] = round(psutil.virtual_memory().available / GIB, 4)
    except ImportError:
        result["ram_monitor"] = "Install psutil for RAM measurements"
    if device is not None:
        import torch

        if torch.cuda.is_available() and torch_device(device).type == "cuda":
            free, total = torch.cuda.mem_get_info(device)
            result.update(
                cuda_allocated_gib=round(torch.cuda.memory_allocated(device) / GIB, 4),
                cuda_reserved_gib=round(torch.cuda.memory_reserved(device) / GIB, 4),
                cuda_peak_allocated_gib=round(torch.cuda.max_memory_allocated(device) / GIB, 4),
                cuda_free_gib=round(free / GIB, 4),
                cuda_total_gib=round(total / GIB, 4),
            )
    return result


class RamMonitor:
    """Sample process RSS, including the transient pages touched while loading."""

    def __init__(self):
        self.peak = 0
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        try:
            import psutil
        except ImportError:
            return self
        process = psutil.Process()

        def sample():
            while True:
                self.peak = max(self.peak, process.memory_info().rss)
                if self.stop.wait(0.1):
                    break

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()


def build_meta_model(model_dir: Path) -> Any:
    import torch
    from accelerate import init_empty_weights
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch

    config = PI05Config.from_pretrained(str(model_dir), local_files_only=True)
    config.device = "cpu"
    # Older LeRobot releases use a string; newer ones use torch.dtype.
    config.dtype = "bfloat16" if isinstance(config.dtype, str) else torch.bfloat16
    config.compile_model = False
    config.gradient_checkpointing = False
    # Nonpersistent RoPE/position buffers must retain their initialized values.
    with init_empty_weights(include_buffers=False):
        model = PI05Pytorch(config)
    # The PiGemma constructor can replace the language model after HF ties it.
    model.paligemma_with_expert.paligemma.tie_weights()
    model.eval().requires_grad_(False)
    return model


def plan_sources(model: Any, sources: list[Source], key_fixer: Any = None):
    """Run the installed official key converter on META tensors, not real weights."""
    import torch
    from accelerate.utils import find_tied_parameters

    specs = {name: TensorSpec(tuple(tensor.shape), tensor.dtype) for name, tensor in model.state_dict().items()}
    ties = find_tied_parameters(model)
    canonical = {name: name for name in specs}
    for group in ties:
        for name in group:
            canonical[name] = group[0]
    candidates = {}
    unknown = []
    skipped = []
    for source in sources:
        raw = strip_model_prefix(source.key)
        if key_fixer is None:
            converted = {raw: torch.empty(source.shape, device="meta")}
        else:
            converted = key_fixer({raw: torch.empty(source.shape, device="meta")})
        if not converted:
            skipped.append(source.key)
        for output_key, meta_value in converted.items():
            target = strip_model_prefix(output_key)
            if target not in specs:
                unknown.append((source.key, target))
                continue
            if tuple(meta_value.shape) != specs[target].shape:
                raise ValueError(f"Shape mismatch: {source.key} -> {target}: {source.shape} != {specs[target].shape}")
            name = canonical[target]
            # Prefer the canonical checkpoint entry over an alias synthesized by
            # the official converter; there is only one allocation per tied group.
            priority = (raw == name, raw == target)
            previous = candidates.get(name)
            if previous is None or priority > previous[0]:
                candidates[name] = (priority, source)
    missing = sorted(set(canonical.values()) - candidates.keys())
    if unknown or missing:
        raise ValueError(f"Checkpoint/model key mismatch. Missing ({len(missing)}): {missing[:12]}; "
                         f"unexpected ({len(unknown)}): {unknown[:12]}")
    return specs, ties, {name: pair[1] for name, pair in candidates.items()}, skipped


def stream_load(model: Any, sources: list[Source], device_map: dict[str, Any],
                key_fixer: Any = None, chunk_mib: float = 64.0,
                progress: bool = True) -> dict[str, Any]:
    """Load one parameter at a time, slicing large tensors to bound staging RAM."""
    import torch
    from accelerate.utils import retie_parameters, set_module_tensor_to_device
    from safetensors import safe_open

    specs, ties, chosen, skipped = plan_sources(model, sources, key_fixer)
    for group in ties:
        devices = {str(torch_device(placement(name, device_map))) for name in group}
        if len(devices) != 1:
            raise ValueError(f"Tied parameters have conflicting placements: {group}: {devices}")
    totals = defaultdict(int)
    for name in chosen:
        totals[str(placement(name, device_map))] += prod(specs[name].shape) * torch.empty((), dtype=specs[name].dtype).element_size()
    if progress:
        print("Actual unique-parameter plan (after weight tying):", flush=True)
        print(json.dumps({key: round(value / GIB, 4) for key, value in totals.items()}, indent=2), flush=True)
    # Load GPU-resident weights first, before CPU-resident parameters fill RAM.
    ordered = sorted(chosen, key=lambda name: (
        torch_device(placement(name, device_map)).type == "cpu",
        str(chosen[name].file), chosen[name].offset,
    ))
    with ExitStack() as stack:
        readers = {file: stack.enter_context(safe_open(str(file), framework="pt", device="cpu"))
                   for file in {item.file for item in chosen.values()}}
        for index, name in enumerate(ordered, 1):
            source, spec = chosen[name], specs[name]
            device = torch_device(placement(name, device_map))
            destination = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            reader = readers[source.file]
            if spec.shape and spec.shape[0] and prod(spec.shape):
                row_bytes = source.nbytes // spec.shape[0]
                rows = max(1, int(chunk_mib * 1024**2) // max(row_bytes, 1))
                view = reader.get_slice(source.key)
                for start in range(0, spec.shape[0], rows):
                    end = min(start + rows, spec.shape[0])
                    piece = view[start:end]
                    typed = piece.to(dtype=spec.dtype)
                    destination[start:end].copy_(typed)
                    del piece, typed
                del view
            else:
                piece = reader.get_tensor(source.key)
                destination.copy_(piece)
                del piece
            set_module_tensor_to_device(model, name, device, value=destination, dtype=spec.dtype)
            del destination
            if progress and (index % 100 == 0 or index == len(ordered)):
                print(f"Loaded {index}/{len(ordered)} tensors", flush=True)
    retie_parameters(model, ties)
    # Check BEFORE dispatch: offloaded parameters legitimately become meta later.
    bad = [name for name, tensor in model.state_dict().items()
           if tensor.is_meta or tensor.dtype != specs[name].dtype]
    if bad:
        raise RuntimeError(f"Incomplete load or dtype mismatch: {bad[:12]}")
    return {"loaded_unique_tensors": len(chosen), "tied_groups": [list(group) for group in ties],
            "officially_skipped_keys": skipped,
            "parameter_gib_by_storage": {key: round(value / GIB, 4) for key, value in totals.items()}}


def load_pi05_lowmem(model_dir: str | Path, device_map: dict[str, Any] | None = None,
                     gpu: int = 0, gpu_layers: int = 4, chunk_mib: float = 64.0):
    """Return (dispatched core model, loading report). Do not call model.to()."""
    import torch
    from accelerate import dispatch_model
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    report = {"before_load": snapshot(device)}
    start = time.perf_counter()
    model = build_meta_model(Path(model_dir))
    device_map = default_device_map(model, gpu, gpu_layers) if device_map is None else {
        name: normalize_device(value) for name, value in device_map.items()
    }
    # This loader implements CPU storage + CUDA execution, without a disk tier.
    for value in device_map.values():
        target = torch_device(value)
        if target.type not in {"cpu", "cuda"}:
            raise ValueError(f"Unsupported storage device: {value}")
        if target.type == "cuda" and target.index != gpu:
            raise ValueError("This script's execution path uses one CUDA device")
    proxy = SimpleNamespace(model=model)
    fixer = lambda entries: PI05Policy._fix_pytorch_state_dict_keys(proxy, entries, model.config)
    report.update(stream_load(model, checkpoint_sources(Path(model_dir)), device_map, fixer, chunk_mib))
    report["before_dispatch"] = snapshot(device)
    model = dispatch_model(model, device_map=device_map, main_device=device,
                           offload_buffers=False, force_hooks=True)
    model.eval()
    report.update(load_seconds=round(time.perf_counter() - start, 3),
                  device_map={key: str(value) for key, value in device_map.items()},
                  after_load=snapshot(device))
    print("LOAD_OK", json.dumps(report["after_load"]), flush=True)
    return model, report


def cache_summary(cache: Any) -> dict[str, Any]:
    import torch

    tensors = {}

    def visit(value):
        if isinstance(value, torch.Tensor):
            tensors[id(value)] = value
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)

    if hasattr(cache, "layers"):
        for layer in cache.layers:
            visit(getattr(layer, "keys", None))
            visit(getattr(layer, "values", None))
    elif hasattr(cache, "key_cache"):
        visit(cache.key_cache)
        visit(cache.value_cache)
    else:
        visit(cache)
    return {"class": type(cache).__name__, "tensor_count": len(tensors),
            "devices": sorted({str(tensor.device) for tensor in tensors.values()}),
            "mib": round(sum(tensor.numel() * tensor.element_size() for tensor in tensors.values()) / 1024**2, 3)}


def resolve_visual_sources(items: list[str], image_keys: list[str]) -> list[tuple[str, str]]:
    """Resolve KEY=SOURCE, or one bare source for the first configured camera."""
    if not image_keys:
        raise ValueError("The model config has no image feature keys")
    result = []
    for item in items:
        # Split only a recognized key prefix, so '=' in a filename is harmless.
        key_prefix, separator, value = item.partition("=")
        if separator and (key_prefix in image_keys or key_prefix.startswith("observation.images.")):
            key = key_prefix
        elif len(items) == 1:
            key, value = image_keys[0], item
        else:
            raise ValueError("Multiple views require KEY=SOURCE for every --image/--camera")
        if key not in image_keys:
            raise ValueError(f"Unknown image key {key!r}; expected one of {image_keys}")
        if not value:
            raise ValueError(f"Empty source for {key}")
        if key in {name for name, _ in result}:
            raise ValueError(f"Duplicate image source for {key}")
        result.append((key, value))
    return result


def read_visual_inputs(config: Any, args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read RGB files or camera snapshots. No robot or serial-port connection."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    camera_items = getattr(args, "camera", None)
    items = camera_items or args.image
    if isinstance(items, (str, Path)):
        items = [str(items)]
    sources = resolve_visual_sources(items, list(config.image_features))
    batch, details = {}, []
    for key, value in sources:
        item = {"key": key, "source": value}
        if camera_items:
            import cv2

            try:
                index = int(value)
            except ValueError as error:
                raise ValueError(f"Camera source must be an integer index, got {value!r}") from error
            backends = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW, "any": cv2.CAP_ANY}
            cap = cv2.VideoCapture(index, backends[args.camera_backend])
            try:
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open camera {index} using {args.camera_backend}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
                ok, bgr = False, None
                # Discard startup frames for camera exposure; this is user-configurable.
                for _ in range(args.camera_warmup + 1):
                    ok, bgr = cap.read()
                if not ok or bgr is None:
                    raise RuntimeError(f"Camera {index} did not return a frame")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            finally:
                cap.release()
            frame_dir = args.report.parent / (args.report.stem + "_frames")
            frame_dir.mkdir(parents=True, exist_ok=True)
            filename = "".join(c if c.isalnum() or c in "_-" else "_" for c in key)
            frame_path = frame_dir / f"{filename}.png"
            Image.fromarray(rgb).save(frame_path)
            item.update(kind="camera_snapshot", camera_index=index,
                        backend=args.camera_backend, saved_frame=str(frame_path))
        else:
            with Image.open(value) as image:
                rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"), copy=True)
            item["kind"] = "image_file"
        item["original_rgb_shape"] = list(rgb.shape)
        batch[key] = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        details.append(item)
    return batch, {"visual_sources": details,
                   "expected_image_keys": list(config.image_features),
                   "provided_image_keys": list(batch),
                   "missing_image_keys": [key for key in config.image_features if key not in batch]}


def prepare_inputs(model: Any, args: Any, device: Any,
                   metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    import torch

    config = model.config
    metadata = {} if metadata is None else metadata
    if args.smoke:
        height, width = config.image_resolution
        metadata.update(input_kind="synthetic", state_kind="not_encoded_in_synthetic_tokens",
                        preprocessed_image_shapes=[[1, 3, height, width]], token_shape=[1, 16])
        print("INPUT_OK", json.dumps(metadata, ensure_ascii=False), flush=True)
        return {"images": [torch.zeros(1, 3, height, width, device=device)],
                "img_masks": [torch.ones(1, dtype=torch.bool, device=device)],
                "tokens": torch.ones(1, 16, dtype=torch.long, device=device),
                "masks": torch.ones(1, 16, dtype=torch.bool, device=device)}

    from transformers import AutoTokenizer
    from lerobot.policies.pi05 import processor_pi05 as processor
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    if getattr(args, "state_zero", False):
        state = torch.zeros(1, config.max_state_dim, dtype=torch.float32)
        state_kind = "diagnostic_zero_not_robot_measurement"
        supplied_dim = config.max_state_dim
    else:
        if args.state_normalized is None:
            raise ValueError("Real visual inputs require --state-zero or --state-normalized")
        state = torch.tensor([args.state_normalized], dtype=torch.float32)
        supplied_dim = state.shape[-1]
        state_kind = "user_supplied_normalized_not_verified_against_training_stats"
    if state.shape[-1] > config.max_state_dim or not torch.isfinite(state).all().item():
        raise ValueError(f"State must be finite and at most {config.max_state_dim} values wide")
    state = torch.nn.functional.pad(state, (0, config.max_state_dim - state.shape[-1]))
    # Reuse the INSTALLED official state discretization and prompt formatting.
    transition = {
        processor.TransitionKey.OBSERVATION: {processor.OBS_STATE: state},
        processor.TransitionKey.COMPLEMENTARY_DATA: {"task": [args.task]},
    }
    step = processor.Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim)
    prompt = step(transition)[processor.TransitionKey.COMPLEMENTARY_DATA]["task"]
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), local_files_only=True)
    tokenizer.padding_side = "right"
    encoded = tokenizer(prompt, padding="max_length", truncation=True,
                        max_length=config.tokenizer_max_length, return_tensors="pt")
    if config.input_features is None:
        config.input_features = {}
    if not config.image_features:
        # Retain support for checkpoints with no dataset-specific image keys.
        from lerobot.configs.types import FeatureType, PolicyFeature

        config.input_features["observation.images.camera"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, *config.image_resolution))
    batch, visual_metadata = read_visual_inputs(config, args)
    # Offloaded parameters may be meta: explicitly supply the CUDA execution device.
    proxy = SimpleNamespace(config=config, parameters=lambda: iter([torch.empty(0, device=device)]))
    with torch.inference_mode():
        images, image_masks = PI05Policy._preprocess_images(proxy, batch)
    metadata.update(visual_metadata,
                    input_kind="real_visual_input_with_" + ("diagnostic_zero_state" if args.state_zero else "supplied_normalized_state"),
                    state_kind=state_kind, supplied_state_dim=supplied_dim,
                    padded_state_dim=state.shape[-1], normalization_applied_by_this_script=False,
                    task=args.task, formatted_prompt=prompt[0], tokenizer_dir=str(args.tokenizer_dir),
                    token_shape=list(encoded["input_ids"].shape),
                    active_text_tokens=encoded["attention_mask"].sum(dim=-1).tolist(),
                    preprocessed_image_shapes=[list(image.shape) for image in images],
                    image_validity_masks=[mask.detach().cpu().tolist() for mask in image_masks],
                    output_semantics="raw_model_space_not_robot_commands")
    print("INPUT_OK", json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    return {"images": images, "img_masks": image_masks,
            "tokens": encoded["input_ids"].to(device),
            "masks": encoded["attention_mask"].to(device=device, dtype=torch.bool)}


def infer_once(model: Any, inputs: dict[str, Any], device: Any, num_steps: int | None):
    import torch

    # The loading report retains the loading peak. Measure this forward separately.
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    phases = {"prefix_transformer_seconds": 0.0, "denoise_transformer_seconds": 0.0, "denoise_calls": 0,
              "cuda_peak_scope": "current_inference_only_including_resident_weights",
              "before_inference": snapshot(device)}
    pg = model.paligemma_with_expert
    original = pg.forward

    def measured_forward(*args, **kwargs):
        embeds = kwargs.get("inputs_embeds")
        prefix = embeds is not None and embeds[1] is None
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        output = original(*args, **kwargs)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if prefix:
            phases["prefix_transformer_seconds"] += elapsed
            phases["kv_cache"] = cache_summary(output[1])
            phases["after_prefix"] = snapshot(device)
            print("PREFIX_OK", json.dumps(phases["kv_cache"]), flush=True)
        else:
            phases["denoise_transformer_seconds"] += elapsed
            phases["denoise_calls"] += 1
        return output

    pg.forward = measured_forward
    try:
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            actions = model.sample_actions(**inputs, num_steps=num_steps)
        torch.cuda.synchronize(device)
        phases["total_inference_seconds"] = round(time.perf_counter() - start, 4)
    finally:
        pg.forward = original
    expected = (inputs["tokens"].shape[0], model.config.chunk_size, model.config.max_action_dim)
    if tuple(actions.shape) != expected or not torch.isfinite(actions).all().item():
        raise RuntimeError(f"Invalid model output: shape={tuple(actions.shape)}, expected={expected}")
    phases.update(action_shape=list(actions.shape), action_dtype=str(actions.dtype),
                  action_device=str(actions.device), finite=True, after_inference=snapshot(device))
    phases["prefix_transformer_seconds"] = round(phases["prefix_transformer_seconds"], 4)
    phases["denoise_transformer_seconds"] = round(phases["denoise_transformer_seconds"], 4)
    print("INFERENCE_OK", json.dumps(phases, indent=2), flush=True)
    return actions, phases


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=Path(r"C:\Friston-3-Models\pi05_base"))
    parser.add_argument("--device-map", type=Path, help="Default: adjacent pi05_device_map.json if present")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu-layers", type=int, help="Override prefix GPU layer count; regenerate the map")
    parser.add_argument("--chunk-mib", type=float, default=64, help="Staging chunk size, not a RAM/VRAM limit")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke", action="store_true", help="Synthetic gray image and token IDs; run the actual model")
    group.add_argument("--image", action="append", metavar="[KEY=]PATH",
                       help="RGB image file; repeat with KEY=PATH for multiple views")
    group.add_argument("--camera", action="append", metavar="[KEY=]INDEX",
                       help="Capture a camera snapshot; repeat with KEY=INDEX for multiple views")
    parser.add_argument("--camera-backend", choices=("msmf", "dshow", "any"),
                        default="msmf" if sys.platform == "win32" else "any")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-warmup", type=int, default=8, help="Startup frames discarded before capture")
    parser.add_argument("--task", default="pick up the cube")
    parser.add_argument("--tokenizer-dir", type=Path,
                        default=Path(r"C:\Friston-3-Models\paligemma_tokenizer"))
    state_group = parser.add_mutually_exclusive_group()
    state_group.add_argument("--state-normalized", type=float, nargs="+",
                             help="Already normalized state, using the checkpoint's training statistics/order")
    state_group.add_argument("--state-zero", action="store_true",
                             help="DIAGNOSTIC ONLY: use a zero placeholder, not measured robot state")
    parser.add_argument("--num-steps", type=int, help="Default: checkpoint's num_inference_steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Optional raw action .pt output")
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("pi05_lowmem_report.json"))
    args = parser.parse_args(argv)
    if args.chunk_mib <= 0 or (args.num_steps is not None and args.num_steps <= 0):
        parser.error("--chunk-mib and --num-steps must be positive")
    if (args.image or args.camera) and args.state_normalized is None and not args.state_zero:
        parser.error("--image/--camera requires --state-normalized or explicit --state-zero")
    if args.camera_warmup < 0 or args.camera_width <= 0 or args.camera_height <= 0:
        parser.error("Camera dimensions must be positive; --camera-warmup must be nonnegative")
    if args.device_map is not None and args.gpu_layers is not None:
        parser.error("Choose --device-map or --gpu-layers")
    return args


def main():
    args = parse_args()
    import torch

    versions = {}
    for package in ("torch", "accelerate", "transformers", "lerobot", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    print("Versions:", json.dumps(versions), flush=True)
    torch.manual_seed(args.seed)
    map_file = args.device_map or Path(__file__).with_name("pi05_device_map.json")
    device_map = None
    if args.gpu_layers is None and map_file.is_file():
        data = json.loads(map_file.read_text(encoding="utf-8-sig"))
        device_map = data.get("device_map", data)
        print("Using device map:", map_file, flush=True)
    elif args.device_map is not None:
        raise FileNotFoundError(map_file)
    device = torch.device(f"cuda:{args.gpu}")
    report = {"versions": versions, "status": "STARTED"}
    with RamMonitor() as monitor:
        try:
            model, loading = load_pi05_lowmem(args.model_dir, device_map, args.gpu,
                                              4 if args.gpu_layers is None else args.gpu_layers, args.chunk_mib)
            report.update(loading, status="LOAD_OK")
            if args.smoke or args.image or args.camera:
                input_metadata = {}
                report["inputs"] = input_metadata
                input_start = time.perf_counter()
                inputs = prepare_inputs(model, args, device, metadata=input_metadata)
                torch.cuda.synchronize(device)
                report["input_preparation_seconds"] = round(time.perf_counter() - input_start, 4)
                report["input_kind"] = input_metadata["input_kind"]
                # Seed the action sampler independently of model construction/tokenizer startup.
                torch.manual_seed(args.seed)
                actions, inference = infer_once(model, inputs, device, args.num_steps)
                report.update(inference, status="INFERENCE_OK")
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(actions.detach().cpu(), args.output)
                    report["action_file"] = str(args.output)
        except Exception as error:
            report.update(status="FAILED", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            report["sampled_process_peak_rss_gib"] = round(monitor.peak / GIB, 4) if monitor.peak else None
            report["sampled_process_peak_rss_scope"] = "loading_input_preparation_and_inference"
            report["final_memory"] = snapshot(device)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print("Report:", args.report, flush=True)


if __name__ == "__main__":
    main()
