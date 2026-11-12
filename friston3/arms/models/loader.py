#!/usr/bin/env python
"""π₀.₅ 低显存权重装配：建网络骨架、规划设备、分块读取权重。

部署：pi05.Pi05Policy → load_pi05_lowmem → LeRobot.PI05Pytorch。
训练：trainer.load_training_model 复用 build_meta_model / stream_load，
再单独安排冻结前缀与可训练动作专家的设备调度。

网络的 embed_prefix / embed_suffix / denoise_step / sample_actions 定义在
安装的 lerobot.policies.pi05.modeling_pi05 中，不在本文件重写。
阅读顺序：load_pi05_lowmem → build_meta_model → plan_sources → stream_load。

"""
from __future__ import annotations

import json
import struct
import time
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from math import prod
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# 本模块导入时只加载标准库；Torch/LeRobot 在实际建模或装载时才导入。


GIB = 1024**3
# ** 是乘方：1 GiB = 1024³ 字节，用于把内存数字换成便于阅读的单位。
ROOT = "paligemma_with_expert"
PALI = f"{ROOT}.paligemma"
LANG = f"{PALI}.model.language_model"
EXPERT = f"{ROOT}.gemma_expert"
# ROOT/PALI/LANG/EXPERT 都是模型内部模块的名称字符串，不是模型对象。
# f"{变量}..." 是格式化字符串；这些前缀稍后用来定位 state_dict 中的权重和设备映射。


@dataclass(frozen=True)
class TensorSpec:
    # 【调用方】plan_sources 为模型中的每个参数/持久缓冲区建立一份规格。
    # @ 是装饰器；dataclass 自动生成 __init__ 等方法，frozen=True 禁止字段重新赋值。
    # 此对象只保存形状和 dtype，不持有权重数值；tuple[int, ...] 表示任意长度的整数元组。
    shape: tuple[int, ...]
    dtype: Any


@dataclass(frozen=True)
class Source:
    # 【调用方】checkpoint_sources 读取权重文件头后创建；plan_sources/stream_load 消费它。
    # file 是所在分片的 Path；key 是分片内名称；shape 是原权重形状；nbytes 是磁盘字节数。
    # offset 相对 Safetensors 数据区起点，用于按文件内位置排序；不是整个文件的绝对地址。
    file: Path
    key: str
    shape: tuple[int, ...]
    nbytes: int
    offset: int


def strip_model_prefix(key: str) -> str:
    # 【调用方】plan_sources；去掉检查点名称开头可选的 model.，便于与核心网络的名字对齐。
    # removeprefix 是 str 的方法：只有开头匹配才删；不会删掉字符串中间出现的 model.。
    return key.removeprefix("model.")


def checkpoint_sources(checkpoint: Path) -> list[Source]:
    """Inspect only file headers; no tensors or complete state dict are loaded."""
    # 【输入/输出】接收目录、单个 .safetensors 或分片索引 Path，返回 Source 对象列表。
    # 【调用方】load_pi05_lowmem 与 trainer.load_training_model；此步只查“货物清单”，不搬权重。
    weight_map = None
    # None 表示尚无分片索引。目录输入先寻找标准索引，再找单文件，最后找唯一的其他命名索引。
    if checkpoint.is_dir():
        index = checkpoint / "model.safetensors.index.json"
        single = checkpoint / "model.safetensors"
        # Path 的 / 运算符拼接路径；这里定义路径对象，还没有读取文件。
        if index.is_file():
            checkpoint = index
        elif single.is_file():
            checkpoint = single
        else:
            indices = sorted(checkpoint.glob("*.safetensors.index.json"))
            # glob 按通配符找文件；sorted 把结果转成排序列表，避免不确定地选择多个检查点之一。
            if len(indices) != 1:
                raise FileNotFoundError(f"No unambiguous Safetensors checkpoint in {checkpoint}")
            checkpoint = indices[0]
    if checkpoint.name.endswith(".index.json"):
        weight_map = json.loads(checkpoint.read_text(encoding="utf-8"))["weight_map"]
        # read_text 得到 JSON 字符串，json.loads 解析成 dict；weight_map 是 {权重名称: 分片文件名}。
        files = [checkpoint.parent / name for name in sorted(set(weight_map.values()))]
        # values 取所有分片名，set 去重；列表推导式为每个分片构造完整 Path。
    elif checkpoint.suffix == ".safetensors":
        files = [checkpoint]
    else:
        raise ValueError("This streaming loader requires .safetensors or its .index.json")

    result = []
    seen = set()
    # result 收集 Source；seen 记录见过的权重名，以发现跨分片的重复项。
    for file in files:
        with file.open("rb") as handle:
            # with 管理文件生命周期，离开缩进块即关闭；rb 是二进制只读，handle 是文件对象。
            size = struct.unpack("<Q", handle.read(8))[0]
            # Safetensors 开头8字节保存 JSON 头长度。< 表示小端，Q 是无符号64位整数。
            # unpack 返回元组，取 [0] 得到长度整数；接着只读这么多字节，不读后面的权重区。
            header = json.loads(handle.read(size))
        for key, entry in header.items():
            # items 每次给出 (键,值)，for 用两个变量解包；entry 包含 shape/dtype/data_offsets。
            if key == "__metadata__":
                continue
            if weight_map is not None and weight_map.get(key) != file.name:
                raise ValueError(f"Checkpoint index disagrees with shard: {file.name}: {key}")
            if key in seen:
                raise ValueError(f"Duplicate checkpoint key: {key}")
            seen.add(key)
            begin, end = entry["data_offsets"]
            result.append(Source(file, key, tuple(entry["shape"]), end - begin, begin))
            # 构造 Source 的位置参数依次对应 dataclass 字段；end-begin 是该张量占用的字节数。
    if weight_map is not None and set(weight_map) != seen:
        raise ValueError("Checkpoint index contains missing tensor entries")
    return result


def normalize_device(value: Any) -> Any:
    # 【调用方】placement/load_pi05_lowmem；JSON 的 "0" 转成整数0，"cpu"/"cuda:0" 保持不变。
    # isinstance 检查类型；and 短路求值，只有 str 才会调用 str.isdecimal()。
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return value


def placement(name: str, device_map: dict[str, Any]) -> Any:
    # 【调用方】stream_load；根据参数完整名称，找设备字典中覆盖它的最长模块前缀。
    # 例如 gemma_expert 与 gemma_expert.lm_head 都匹配后者时，较具体的 lm_head 配置优先。
    matches = [key for key in device_map if key == "" or name == key or name.startswith(key + ".")]
    # 空字符串键覆盖整个模型；加 "." 可区分 layers.1 与 layers.10，避免字符串前缀误匹配。
    if not matches:
        raise ValueError(f"No device-map entry covers {name}")
    return normalize_device(device_map[max(matches, key=len)])
    # max(..., key=len) 按字符串长度选最大项；不是按字母顺序找最大值。


def torch_device(value: Any) -> Any:
    # 【调用方】设备规划、内存监控和加载函数；统一返回 torch.device 描述对象。
    import torch

    # 条件表达式 A if 条件 else B：整数0变 "cuda:0"；字符串 "cpu" 等直接传给 Torch。
    # 创建 torch.device 只描述设备，此行没有分配模型权重或显存。
    return torch.device(f"cuda:{value}" if isinstance(value, int) else value)


def default_device_map(model: Any, gpu: int, gpu_layers: int = 4) -> dict[str, Any]:
    # 【调用方】load_pi05_lowmem；输入已建好的网络骨架、GPU编号和常驻GPU的前缀层数。
    # 返回 {模块名称: 存储设备}。CPU存储不代表整层永远在CPU计算；dispatch会按需搬到GPU执行。
    # 视觉塔、动作专家等常驻GPU；较多语言层放CPU，以适应本机显存。
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
    # 连续 . 逐层访问对象属性；layers 是模型中的语言Transformer层集合。
    result.update({f"{LANG}.layers.{i}": gpu if i < gpu_layers else "cpu" for i in range(len(layers))})
    # 字典推导式生成每层的位置，update 合并进 result；i 是从0开始的层号。
    if getattr(model, "proprio_history_proj", None) is not None:
        # getattr 的第三参数是缺属性时的默认值；只有当前模型真的有这层才加入设备表。
        result["proprio_history_proj"] = gpu
    return result


def snapshot(device: Any = None) -> dict[str, Any]:
    # 【调用方】加载/推理/最终报告；采集此刻内存数字并返回 dict，不改变加载策略。
    result = {}
    try:
        import psutil

        result["process_rss_gib"] = round(psutil.Process().memory_info().rss / GIB, 4)
        # RSS 是当前进程驻留物理内存；round(x,4) 保留4位小数。它不等于模型参数体积。
        result["system_available_gib"] = round(psutil.virtual_memory().available / GIB, 4)
    except ImportError:
        result["ram_monitor"] = "Install psutil for RAM measurements"
    if device is not None:
        import torch

        if torch.cuda.is_available() and torch_device(device).type == "cuda":
            free, total = torch.cuda.mem_get_info(device)
            # free/total 是整块GPU的空闲/总显存；allocated/reserved 是本进程Torch的使用/缓存。
            result.update(
                cuda_allocated_gib=round(torch.cuda.memory_allocated(device) / GIB, 4),
                cuda_reserved_gib=round(torch.cuda.memory_reserved(device) / GIB, 4),
                cuda_peak_allocated_gib=round(torch.cuda.max_memory_allocated(device) / GIB, 4),
                cuda_free_gib=round(free / GIB, 4),
                cuda_total_gib=round(total / GIB, 4),
            )
    return result


def build_meta_model(model_dir: Path) -> Any:
    # 【调用方】load_pi05_lowmem、trainer.load_training_model。
    # 返回结构完整、参数主要位于meta设备的 PI05Pytorch；meta有shape/dtype但没有真实权重存储。
    import torch
    from accelerate import init_empty_weights
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch

    config = PI05Config.from_pretrained(str(model_dir), local_files_only=True)
    # from_pretrained 是LeRobot配置类的方法；这里只读本地配置，不等于已加载模型权重。
    config.device = "cpu"
    # Older LeRobot releases use a string; newer ones use torch.dtype.
    config.dtype = "bfloat16" if isinstance(config.dtype, str) else torch.bfloat16
    config.compile_model = False
    config.gradient_checkpointing = False
    # 这两项先关闭编译/梯度检查点；本函数负责建模，训练流程会另外安排训练相关行为。
    # Nonpersistent RoPE/position buffers must retain their initialized values.
    with init_empty_weights(include_buffers=False):
        # Accelerate上下文临时改变参数创建方式，避免先初始化一整份巨大随机权重再覆盖。
        # include_buffers=False 保留缓冲区的正常初始化，避免位置编码等非持久缓冲区丢失数值。
        model = PI05Pytorch(config)
    # The PiGemma constructor can replace the language model after HF ties it.
    model.paligemma_with_expert.paligemma.tie_weights()
    # tie_weights 让应共享的词嵌入/输出权重恢复共享关系；后面按共享组只分配一份存储。
    model.eval().requires_grad_(False)
    # eval 设置评估模式（如关闭dropout），requires_grad_(False) 冻结参数梯度；两者用途不同。
    # 尾部下划线常表示原地修改。训练代码随后会单独启用需要微调的模块。
    return model


def plan_sources(model: Any, sources: list[Source], key_fixer: Any = None):
    """Run the installed official key converter on META tensors, not real weights."""
    # 【调用方】stream_load；把“磁盘权重清单”与“模型需要的参数清单”对齐。
    # 返回四项：目标规格specs、共享权重组ties、每个独立参数的来源chosen、官方跳过项skipped。
    # key_fixer 是可调用对象/函数；传入 {旧名: meta张量}，返回官方转换后的名称和形状。
    import torch
    from accelerate.utils import find_tied_parameters

    specs = {name: TensorSpec(tuple(tensor.shape), tensor.dtype) for name, tensor in model.state_dict().items()}
    # state_dict() 返回参数/持久缓冲区的名称映射；此时meta参数只提供结构信息，不读真实权重。
    ties = find_tied_parameters(model)
    canonical = {name: name for name in specs}
    # find_tied_parameters 来自Accelerate；ties每项是一组共享同一参数的名字。
    # canonical先让每个名字代表自己，再把共享组的名字统一映射到该组第一个名字。
    for group in ties:
        for name in group:
            canonical[name] = group[0]
    candidates = {}
    unknown = []
    skipped = []
    # candidates 保存 {统一名称: (优先级, Source)}；unknown记录无对应目标的键，skipped记录官方删除项。
    for source in sources:
        raw = strip_model_prefix(source.key)
        if key_fixer is None:
            converted = {raw: torch.empty(source.shape, device="meta")}
        else:
            converted = key_fixer({raw: torch.empty(source.shape, device="meta")})
        # torch.empty(...,device="meta") 只造形状占位，不分配真实数据；转换器因此不用处理几十GB数据。
        # 注意本实现随后复制原来源权重；这里复用的是官方名称/共享别名转换，不执行任意数值变换。
        if not converted:
            skipped.append(source.key)
        for output_key, meta_value in converted.items():
            target = strip_model_prefix(output_key)
            if target not in specs:
                unknown.append((source.key, target))
                continue
            if tuple(meta_value.shape) != specs[target].shape:
                # shape必须匹配才能直接复制；这属于装载完整性检查，不是机械臂运动范围限制。
                raise ValueError(f"Shape mismatch: {source.key} -> {target}: {source.shape} != {specs[target].shape}")
            name = canonical[target]
            # Prefer the canonical checkpoint entry over an alias synthesized by
            # the official converter; there is only one allocation per tied group.
            priority = (raw == name, raw == target)
            previous = candidates.get(name)
            # 布尔元组按从左到右比较，True大于False；优先真实规范名，其次直接匹配目标名。
            # get缺键返回None；避免官方补出的共享别名抢占更直接的权重来源。
            if previous is None or priority > previous[0]:
                candidates[name] = (priority, source)
    missing = sorted(set(canonical.values()) - candidates.keys())
    # 集合差集找缺失目标；用统一名比较，所以共享参数不要求每个别名都有独立数据。
    if unknown or missing:
        raise ValueError(f"Checkpoint/model key mismatch. Missing ({len(missing)}): {missing[:12]}; "
                         f"unexpected ({len(unknown)}): {unknown[:12]}")
    return specs, ties, {name: pair[1] for name, pair in candidates.items()}, skipped


def stream_load(model: Any, sources: list[Source], device_map: dict[str, Any],
                key_fixer: Any = None, chunk_mib: float = 64.0,
                progress: bool = True) -> dict[str, Any]:
    """Load one parameter at a time, slicing large tensors to bound staging RAM."""
    # 【调用方】load_pi05_lowmem、trainer.load_training_model；实际把磁盘数值填入model。
    # model按原对象修改，返回值只是统计报告。sources来自checkpoint_sources，设备表来自调用方。
    # chunk_mib控制一次读取的大致字节量，不是总内存/显存上限：目标参数、已有权重、转换副本均另占空间。
    import torch
    from accelerate.utils import retie_parameters, set_module_tensor_to_device
    from safetensors import safe_open
    # 导入发生在函数运行时。外层若使用 checkpoint_reader()，safe_open会是本项目的分块读取包装器。

    specs, ties, chosen, skipped = plan_sources(model, sources, key_fixer)
    for group in ties:
        devices = {str(torch_device(placement(name, device_map))) for name in group}
        # 集合去重后应只剩一个设备；同一块共享权重不能同时常驻两处还保持同一存储。
        if len(devices) != 1:
            raise ValueError(f"Tied parameters have conflicting placements: {group}: {devices}")
    totals = defaultdict(int)
    # 首次访问 totals[设备] 时默认从 int() 即0开始，随后 += 累加独立参数的目标字节数。
    for name in chosen:
        totals[str(placement(name, device_map))] += prod(specs[name].shape) * torch.empty((), dtype=specs[name].dtype).element_size()
        # prod(shape) 是各维长度的乘积；element_size()是目标dtype每元素字节数。
        # empty(()) 是一个标量，用于询问dtype大小，不创建完整参数；共享别名没有重复计数。
    if progress:
        print("Actual unique-parameter plan (after weight tying):", flush=True)
        print(json.dumps({key: round(value / GIB, 4) for key, value in totals.items()}, indent=2), flush=True)
    # Load GPU-resident weights first, before CPU-resident parameters fill RAM.
    ordered = sorted(chosen, key=lambda name: (
        torch_device(placement(name, device_map)).type == "cpu",
        str(chosen[name].file), chosen[name].offset,
    ))
    # lambda 是匿名函数，给sorted返回比较键；False排在True前，所以GPU目标先于CPU目标。
    # 随后按文件名、文件内偏移排序，尽量减少磁盘跳读；这里只安排顺序，尚未读取权重。
    with ExitStack() as stack:
        # 需要打开多少分片运行时才知道；ExitStack登记每个文件，退出with时统一关闭，异常时也关闭。
        readers = {file: stack.enter_context(safe_open(str(file), framework="pt", device="cpu"))
                   for file in {item.file for item in chosen.values()}}
        for index, name in enumerate(ordered, 1):
            # enumerate同时产生(序号,元素)，第二参数1表示进度序号从1开始。
            source, spec = chosen[name], specs[name]
            device = torch_device(placement(name, device_map))
            destination = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            # 真正分配这个目标参数完整的存储；与meta占位不同，CPU占内存，CUDA占显存。
            reader = readers[source.file]
            if spec.shape and spec.shape[0] and prod(spec.shape):
                # 有首维且非空的张量可以沿第0维分段；标量shape=()、空张量转下面else。
                row_bytes = source.nbytes // spec.shape[0]
                rows = max(1, int(chunk_mib * 1024**2) // max(row_bytes, 1))
                # //是整数除法；每“行”指第0维的一项，可能仍包含很多列/更高维元素。
                # 根据磁盘字节数算每块多少行，至少1行；单行过大时仍可能超过chunk_mib。
                view = reader.get_slice(source.key)
                # get_slice得到可切片访问对象；真正选取下方[start:end]时才读取所需数据。
                for start in range(0, spec.shape[0], rows):
                    end = min(start + rows, spec.shape[0])
                    piece = view[start:end]
                    typed = piece.to(dtype=spec.dtype)
                    # Python切片左闭右开，最后一块用min截到张量末尾；to转换为目标dtype。
                    # dtype已相同时typed可能复用piece；需要转换时会有临时副本，计入加载峰值。
                    destination[start:end].copy_(typed)
                    # copy_原地填入目标片段；目标是GPU而来源是CPU时，此处完成主机到显卡复制。
                    del piece, typed
                    # del删除局部引用，让每块临时数据可被释放；不保证进程RSS立即下降，分配器可能缓存。
                del view
            else:
                piece = reader.get_tensor(source.key)
                # 标量/空张量不用沿首维切片，直接取得完整小张量。
                destination.copy_(piece)
                del piece
            set_module_tensor_to_device(model, name, device, value=destination, dtype=spec.dtype)
            # Accelerate工具按name找到模型内部属性，将meta占位替换为刚填好的真实权重。
            del destination
            # 模型现在持有该权重；删除局部变量不等于删除模型里的参数。
            if progress and (index % 100 == 0 or index == len(ordered)):
                print(f"Loaded {index}/{len(ordered)} tensors", flush=True)
    retie_parameters(model, ties)
    # 装载可能替换了Parameter对象；重新绑定共享组，使别名仍指向同一权重而不是额外副本。
    # Check BEFORE dispatch: offloaded parameters legitimately become meta later.
    bad = [name for name, tensor in model.state_dict().items()
           if tensor.is_meta or tensor.dtype != specs[name].dtype]
    # 这里要求所有目标已有真实数据且dtype正确；检查放在dispatch前，因为offload之后meta可能合法。
    if bad:
        raise RuntimeError(f"Incomplete load or dtype mismatch: {bad[:12]}")
    return {"loaded_unique_tensors": len(chosen), "tied_groups": [list(group) for group in ties],
            "officially_skipped_keys": skipped,
            "parameter_gib_by_storage": {key: round(value / GIB, 4) for key, value in totals.items()}}


def load_pi05_lowmem(model_dir: str | Path, device_map: dict[str, Any] | None = None,
                     gpu: int = 0, gpu_layers: int = 4, chunk_mib: float = 64.0):
    """Return (dispatched core model, loading report). Do not call model.to()."""
    # 【调用方】pi05.Pi05Policy这是部署的完整加载组织函数。
    # 输入本地权重目录、可选设备表、GPU编号/层数/分块大小，返回(model, report)二元组。
    # 训练路径不会直接复用这一整套dispatch：trainer.py 需要为冻结前缀和可训练专家分别安排梯度。
    import torch
    from accelerate import dispatch_model
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    # 设置当前GPU并重置Torch显存峰值计数；它不会清空显卡上已有的模型/缓存。
    report = {"before_load": snapshot(device)}
    start = time.perf_counter()
    # perf_counter是单调高精度计时器，用前后差值测耗时；不是日期时间戳。
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
    # SimpleNamespace构造一个能用 .model 访问属性的轻量对象，满足下面官方实例方法所需的self接口。
    fixer = lambda entries: PI05Policy._fix_pytorch_state_dict_keys(proxy, entries, model.config)
    # 官方方法属于PI05Policy类；这里显式传入proxy充当self，不另外实例化/加载一份完整Policy。
    # fixer捕获model/proxy，交给plan_sources；调用fixer(entries)时才执行这个lambda。
    report.update(stream_load(model, checkpoint_sources(Path(model_dir)), device_map, fixer, chunk_mib))
    report["before_dispatch"] = snapshot(device)
    model = dispatch_model(model, device_map=device_map, main_device=device,
                           offload_buffers=False, force_hooks=True)
    # dispatch_model来自Accelerate；安装hook，在执行某层前搬入所需权重/输入，之后按方案卸载。
    # main_device是计算设备，device_map描述存储位置；offload_buffers=False保留缓冲区的既定处理。
    # force_hooks=True确保安装调度钩子；之后不要再对整体model.to(...)，否则会破坏这套放置方案。
    model.eval()
    report.update(load_seconds=round(time.perf_counter() - start, 3),
                  device_map={key: str(value) for key, value in device_map.items()},
                  after_load=snapshot(device))
    print("LOAD_OK", json.dumps(report["after_load"]), flush=True)
    return model, report
