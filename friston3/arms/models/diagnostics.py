#!/usr/bin/env python
r"""π₀.₅ 独立诊断：用合成输入、图片或相机快照检查模型装载与动作采样。

    python -m friston3.arms.models.diagnostics --smoke --num-steps 10
    python -m friston3.arms.models.diagnostics --camera 1 --state-zero --task "pick up the cube"
    python -m friston3.arms.models.diagnostics --image frame.jpg --state-zero

没有相机键的一路输入使用配置中的第一个图像键；多路输入用 KEY=SOURCE。
--state-zero 是合成占位，--state-normalized 需要调用者按训练统计/顺序归一化。
输出是模型空间中的原始动作，不执行机器人专用反归一化，也不连接机械臂。

本文件的 prepare_inputs 只服务诊断。训练输入见 pi05.prepare_inputs，实际部署输入见
pi05.Pi05Policy.predict；模型装配见 loader.py，网络本体在安装的 LeRobot 中。
导入本文件不会加载模型或打开相机；执行 main 后才按参数进行诊断。
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "friston3.arms.models"

from . import loader
# 通过模块访问加载/内存工具，保持 loader 的实现与运行时替换接缝只有一份。


class RamMonitor:
    """Sample process RSS, including the transient pages touched while loading."""
    # 【调用方】main 中 with RamMonitor() as monitor；后台线程每0.1秒抽样一次进程RSS。
    # 抽样峰值可能漏掉短于采样间隔的尖峰，所以报告称 sampled peak，而非精确连续峰值。

    def __init__(self):
        # self 是正在创建的实例；这三个属性之后由 __enter__/sample/__exit__ 共用。
        self.peak = 0
        self.stop = threading.Event()
        # Event 是线程间的停止信号，初始未触发；不是在这里停止线程。
        self.thread = None

    def __enter__(self):
        # with 进入时自动调用；返回 self，所以 as monitor 得到当前实例。
        try:
            import psutil
        except ImportError:
            return self
        process = psutil.Process()

        def sample():
            # 内部函数/闭包能读取外面的 process 和 self；传给 Thread 后在另一线程运行。
            while True:
                self.peak = max(self.peak, process.memory_info().rss)
                if self.stop.wait(0.1):
                    # 最多等待0.1秒；若其他线程 set() 触发信号，就返回True并退出循环。
                    break

        self.thread = threading.Thread(target=sample, daemon=True)
        # target=sample 传函数对象，没有括号，尚未调用；daemon线程不会独自阻止进程退出。
        self.thread.start()
        return self

    def __exit__(self, *args):
        # 离开with时（含异常）自动调用；*args接收with传入的异常类型/值/回溯，本函数不用它们。
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
            # join 等待采样线程结束，避免离开with后它仍在修改峰值。


def cache_summary(cache: Any) -> dict[str, Any]:
    # 【调用方】infer_once的计时包装器；递归统计前缀Transformer输出的KV缓存张量。
    # KV缓存保留已算好的注意力key/value，供后续动作去噪复用；这里只统计，不复制/修改缓存。
    import torch

    tensors = {}

    def visit(value):
        # 内部递归函数：张量直接登记，list/tuple/dict则继续查看其中每个值。
        if isinstance(value, torch.Tensor):
            tensors[id(value)] = value
            # id是Python对象身份；同一个Tensor对象被多处引用时只计一次。
            # 不同Tensor视图即使共享底层storage，也可能分别计数；这是缓存诊断估计值。
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)

    if hasattr(cache, "layers"):
        # hasattr检查属性是否存在；兼容不同Transformers版本的缓存对象结构。
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
    # numel是元素数，element_size是单元素字节数；总和除1024²得到MiB。


def resolve_visual_sources(items: list[str], image_keys: list[str]) -> list[tuple[str, str]]:
    """Resolve KEY=SOURCE, or one bare source for the first configured camera."""
    # 【调用方】read_visual_inputs；把命令行来源转成[(模型图像键,路径/相机编号字符串), ...]。
    # 只有一路输入时允许省略KEY，自动用模型首个图像键；多路需明确绑定，避免相机顺序混淆。
    if not image_keys:
        raise ValueError("The model config has no image feature keys")
    result = []
    for item in items:
        # Split only a recognized key prefix, so '=' in a filename is harmless.
        key_prefix, separator, value = item.partition("=")
        # str.partition按第一个=分成三项；没有=时得到(原字符串,"","")。
        # 只有识别出的图像键前缀才当作KEY=SOURCE，普通文件名中带=不会随意被截断。
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
            # 集合推导式取已登记的键；_表示此处不关心每个二元组的第二项。
            raise ValueError(f"Duplicate image source for {key}")
        result.append((key, value))
    return result


def read_visual_inputs(config: Any, args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read RGB files or camera snapshots. No robot or serial-port connection."""
    # 【调用方】prepare_inputs的真实图像分支；config来自模型，args来自parse_args。
    # 返回(batch, metadata)：batch按模型图像键保存[1,3,H,W]的CPU float32张量，metadata记录来源。
    # 这是独立诊断CLI的读图方式；实际机械臂部署的实时观测由policy.py准备。
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    camera_items = getattr(args, "camera", None)
    items = camera_items or args.image
    # or返回第一个非空值；选择相机列表或文件列表。命令行互斥组保证不能同时指定两者。
    if isinstance(items, (str, Path)):
        items = [str(items)]
    sources = resolve_visual_sources(items, list(config.image_features))
    batch, details = {}, []
    # 多重赋值分别创建dict和list；不是让两个名字指向同一个对象。
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
            # VideoCapture来自OpenCV，此行才真正尝试打开相机；backends把用户选项映射为库常量。
            try:
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open camera {index} using {args.camera_backend}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
                # set向驱动请求分辨率，实际返回帧尺寸还取决于相机/驱动支持，稍后记录rgb.shape。
                ok, bgr = False, None
                # Discard startup frames for camera exposure; this is user-configurable.
                for _ in range(args.camera_warmup + 1):
                    ok, bgr = cap.read()
                # cap.read返回(是否成功, BGR数组)；循环覆盖旧帧，仅保留预热后的最后一帧。
                if not ok or bgr is None:
                    raise RuntimeError(f"Camera {index} did not return a frame")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                # OpenCV默认通道顺序BGR，模型图像路径使用RGB，必须先交换通道含义。
            finally:
                cap.release()
                # finally无论成功/异常都执行，及时释放摄像头给其他程序。
            frame_dir = args.report.parent / (args.report.stem + "_frames")
            frame_dir.mkdir(parents=True, exist_ok=True)
            filename = "".join(c if c.isalnum() or c in "_-" else "_" for c in key)
            # 生成器逐字符处理图像键，join拼回文件名；把点等符号替换为下划线。
            frame_path = frame_dir / f"{filename}.png"
            Image.fromarray(rgb).save(frame_path)
            # Pillow从NumPy数组创建图像并保存，方便对照本次诊断到底看到了什么。
            item.update(kind="camera_snapshot", camera_index=index,
                        backend=args.camera_backend, saved_frame=str(frame_path))
        else:
            with Image.open(value) as image:
                rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"), copy=True)
                # exif_transpose按照片EXIF方向旋转，再转RGB；copy=True让数组在文件关闭后独立存在。
            item["kind"] = "image_file"
        item["original_rgb_shape"] = list(rgb.shape)
        batch[key] = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        # ascontiguousarray保证内存连续；from_numpy创建Tensor；permute把[H,W,3]改为[3,H,W]。
        # unsqueeze(0)加batch维变[1,3,H,W]；float转float32，除255把像素0..255变为0..1。
        # 这里尚未统一模型分辨率，后面的官方_preprocess_images负责其所需预处理与缺图掩码。
        details.append(item)
    return batch, {"visual_sources": details,
                   "expected_image_keys": list(config.image_features),
                   "provided_image_keys": list(batch),
                   "missing_image_keys": [key for key in config.image_features if key not in batch]}


def prepare_inputs(model: Any, args: Any, device: Any,
                   metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    # 【调用方】main；把诊断选项转换为model.sample_actions接收的images/img_masks/tokens/masks。
    # metadata若由调用者提供，会被原地填充，用于报告；返回值只放模型输入张量。
    # 此处不读取机器人关节、不从数据集推算归一化统计；--state-normalized须由调用者预先处理好。
    import torch

    config = model.config
    metadata = {} if metadata is None else metadata
    # 只在没传dict时创建新对象，避免用可变默认参数{}导致不同调用共享历史内容。
    if args.smoke:
        # smoke用合成图像和token走一次真实推理，检查张量通路；不是任务成功率评估。
        height, width = config.image_resolution
        metadata.update(input_kind="synthetic", state_kind="not_encoded_in_synthetic_tokens",
                        preprocessed_image_shapes=[[1, 3, height, width]], token_shape=[1, 16])
        print("INPUT_OK", json.dumps(metadata, ensure_ascii=False), flush=True)
        return {"images": [torch.zeros(1, 3, height, width, device=device)],
                "img_masks": [torch.ones(1, dtype=torch.bool, device=device)],
                "tokens": torch.ones(1, 16, dtype=torch.long, device=device),
                "masks": torch.ones(1, 16, dtype=torch.bool, device=device)}
        # images是视角列表，每项[batch,通道,高,宽]；img_masks每项[batch]表示该视角有效。
        # tokens是整数ID[1,16]，masks是布尔有效位；这些合成token并未编码真实任务/关节状态。

    from transformers import AutoTokenizer
    from lerobot.policies.pi05 import processor_pi05 as processor
    # as只给模块取本地别名；processor中的类与常量仍定义在安装的LeRobot中。
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    if getattr(args, "state_zero", False):
        state = torch.zeros(1, config.max_state_dim, dtype=torch.float32)
        # 全零状态是显式诊断占位，[1,最大状态维]；不能把它误读成串口测得的真实姿态。
        state_kind = "diagnostic_zero_not_robot_measurement"
        supplied_dim = config.max_state_dim
    else:
        if args.state_normalized is None:
            raise ValueError("Real visual inputs require --state-zero or --state-normalized")
        state = torch.tensor([args.state_normalized], dtype=torch.float32)
        # 外层方括号添加batch轴：输入D个关节数值得到[1,D]，例如D=6。
        supplied_dim = state.shape[-1]
        state_kind = "user_supplied_normalized_not_verified_against_training_stats"
    if state.shape[-1] > config.max_state_dim or not torch.isfinite(state).all().item():
        raise ValueError(f"State must be finite and at most {config.max_state_dim} values wide")
    state = torch.nn.functional.pad(state, (0, config.max_state_dim - state.shape[-1]))
    # shape[-1]是最后一维；pad的二元组表示末维左/右补零数，将[1,D]补成[1,max_state_dim]。
    # 这只是补齐模型输入槽位，既不增加真实关节，也不自动把角度转换成训练时的归一化值。
    # Reuse the INSTALLED official state discretization and prompt formatting.
    transition = {
        processor.TransitionKey.OBSERVATION: {processor.OBS_STATE: state},
        processor.TransitionKey.COMPLEMENTARY_DATA: {"task": [args.task]},
    }
    step = processor.Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim)
    prompt = step(transition)[processor.TransitionKey.COMPLEMENTARY_DATA]["task"]
    # step是可调用对象，step(...)会运行其调用逻辑；官方处理器将离散化状态与任务格式化成文本。
    # 连续两个[]从嵌套结果中取出task列表；π₀.₅在这条路径里把状态送进文本token，而非另传state参数。
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), local_files_only=True)
    tokenizer.padding_side = "right"
    # AutoTokenizer来自Transformers；只从本地目录加载分词器。属性控制短序列在哪一侧补齐。
    encoded = tokenizer(prompt, padding="max_length", truncation=True,
                        max_length=config.tokenizer_max_length, return_tensors="pt")
    # 关键字参数分别要求补到最大长度、截断过长文本、返回PyTorch张量。
    # encoded主要包含input_ids与attention_mask：[batch,文本长度]，mask区分有效token和padding。
    if config.input_features is None:
        config.input_features = {}
    if not config.image_features:
        # Retain support for checkpoints with no dataset-specific image keys.
        from lerobot.configs.types import FeatureType, PolicyFeature

        config.input_features["observation.images.camera"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, *config.image_resolution))
        # *解包元组：若image_resolution=(224,224)，则shape=(3,224,224)。
        # 仅当基础模型未声明数据集相机键时提供诊断默认键，不会凭空创建真实相机数据。
    batch, visual_metadata = read_visual_inputs(config, args)
    # Offloaded parameters may be meta: explicitly supply the CUDA execution device.
    proxy = SimpleNamespace(config=config, parameters=lambda: iter([torch.empty(0, device=device)]))
    # 官方图像预处理会询问参数设备。offload模型参数可能显示meta，因此给它一个空CUDA张量作设备提示。
    # parameters是无参lambda，调用时返回迭代器；empty(0)无有效数据，不是模型的新参数。
    with torch.inference_mode():
        images, image_masks = PI05Policy._preprocess_images(proxy, batch)
        # inference_mode禁用自动求导记录；复用已安装官方实现完成图像尺寸/数值与缺失视角处理。
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
    # detach切断梯度引用，cpu移到主内存，tolist转为JSON可记录的Python列表。
    # metadata只记录摘要/形状/有效性，模型真正的像素张量仍在images里。
    print("INPUT_OK", json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    return {"images": images, "img_masks": image_masks,
            "tokens": encoded["input_ids"].to(device),
            "masks": encoded["attention_mask"].to(device=device, dtype=torch.bool)}


def infer_once(model: Any, inputs: dict[str, Any], device: Any, num_steps: int | None):
    # 【调用方】main；执行一次动作块采样并测时间，返回(actions, phases)。
    # actions通常是[batch,chunk_size,max_action_dim]，仍在模型空间，没有转成SO-101目标角度。
    import torch

    # The loading report retains the loading peak. Measure this forward separately.
    torch.cuda.synchronize(device)
    # CUDA操作通常异步提交；同步等待前面的GPU任务完成，避免计时只量到CPU提交命令的时间。
    torch.cuda.reset_peak_memory_stats(device)
    phases = {"prefix_transformer_seconds": 0.0, "denoise_transformer_seconds": 0.0, "denoise_calls": 0,
              "cuda_peak_scope": "current_inference_only_including_resident_weights",
              "before_inference": loader.snapshot(device)}
    pg = model.paligemma_with_expert
    original = pg.forward
    # 保存原forward方法对象，没有括号所以尚未执行；后面临时包装它，最终一定恢复。

    def measured_forward(*args, **kwargs):
        # *args收集位置参数为元组，**kwargs收集关键字参数为dict；可原样转交任意forward调用。
        embeds = kwargs.get("inputs_embeds")
        prefix = embeds is not None and embeds[1] is None
        # 当前模型约定inputs_embeds的第二项是动作专家输入；它为空时，归类为前缀编码调用。
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        output = original(*args, **kwargs)
        # 此处真正执行原Transformer；调用侧*和**分别将元组/字典展开还原成参数。
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if prefix:
            phases["prefix_transformer_seconds"] += elapsed
            phases["kv_cache"] = cache_summary(output[1])
            phases["after_prefix"] = loader.snapshot(device)
            print("PREFIX_OK", json.dumps(phases["kv_cache"]), flush=True)
        else:
            phases["denoise_transformer_seconds"] += elapsed
            phases["denoise_calls"] += 1
        return output

    pg.forward = measured_forward
    # 临时替换实例的方法入口，使sample_actions内部调用也经过计时；不改第三方库文件。
    try:
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            actions = model.sample_actions(**inputs, num_steps=num_steps)
            # **inputs把images等键展开为同名参数；num_steps控制动作采样的去噪步数。
        torch.cuda.synchronize(device)
        phases["total_inference_seconds"] = round(time.perf_counter() - start, 4)
    finally:
        pg.forward = original
        # 即使推理抛异常也恢复原forward，避免后续推理叠加计时包装。
    expected = (inputs["tokens"].shape[0], model.config.chunk_size, model.config.max_action_dim)
    # shape[0]是batch大小；检查输出维度及有限数值，是确认推理结果结构完整。
    if tuple(actions.shape) != expected or not torch.isfinite(actions).all().item():
        raise RuntimeError(f"Invalid model output: shape={tuple(actions.shape)}, expected={expected}")
    phases.update(action_shape=list(actions.shape), action_dtype=str(actions.dtype),
                  action_device=str(actions.device), finite=True, after_inference=loader.snapshot(device))
    phases["prefix_transformer_seconds"] = round(phases["prefix_transformer_seconds"], 4)
    phases["denoise_transformer_seconds"] = round(phases["denoise_transformer_seconds"], 4)
    print("INFERENCE_OK", json.dumps(phases, indent=2), flush=True)
    return actions, phases


def parse_args(argv: list[str] | None = None):
    # 【调用方】main；注册命令行选项并返回argparse.Namespace，之后用args.gpu等属性取值。
    # argv=None时读取当前进程命令行；传字符串列表时解析该列表，便于离线检查参数行为。
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser在此首次定义，是argparse.ArgumentParser类的实例；description/formatter_class是关键字参数。
    # __doc__是文件头三引号字符串；RawDescriptionHelpFormatter保留描述中的原始换行。
    parser.add_argument("--model-dir", type=Path, default=Path(r"C:\Friston-3-Models\pi05_base"))
    # 对象.方法(...)调用parser的方法；第一个字符串是选项名，type把用户输入转为Path。
    # default是在未传选项时采用的值；r字符串保留反斜杠。旧默认路径不一定符合当前资产目录，
    # 诊断时应显式传--model-dir/--tokenizer-dir；项目训练和部署入口有各自的资产路径解析。
    parser.add_argument("--device-map", type=Path, help="Default: adjacent pi05_device_map.json if present")
    parser.add_argument("--gpu", type=int, default=0)
    # --gpu 0会变成args.gpu整数0；选项里的连字符通常在属性名中变为下划线，如args.gpu_layers。
    parser.add_argument("--gpu-layers", type=int, help="Override prefix GPU layer count; regenerate the map")
    parser.add_argument("--chunk-mib", type=float, default=64, help="Staging chunk size, not a RAM/VRAM limit")
    group = parser.add_mutually_exclusive_group()
    # group在此定义，保存同一个parser上的互斥选项组；以下三种输入模式最多选一种，也可全不选。
    # 全不选时main只加载模型，不运行图像/动作推理；group不是新的独立解析器。
    group.add_argument("--smoke", action="store_true", help="Synthetic gray image and token IDs; run the actual model")
    # store_true使无值开关--smoke出现时为True，不出现为False；help是--help显示的说明文字。
    group.add_argument("--image", action="append", metavar="[KEY=]PATH",
                       help="RGB image file; repeat with KEY=PATH for multiple views")
    # append把每次出现的--image收集成列表，支持多视角；metavar只改变帮助中的占位显示。
    # 圆括号内的函数调用允许自然换行；每个逗号分隔一个参数，不需要额外的反斜杠续行。
    group.add_argument("--camera", action="append", metavar="[KEY=]INDEX",
                       help="Capture a camera snapshot; repeat with KEY=INDEX for multiple views")
    parser.add_argument("--camera-backend", choices=("msmf", "dshow", "any"),
                        default="msmf" if sys.platform == "win32" else "any")
    # choices是允许值元组；默认值根据操作系统选择，sys.platform由Python提供。
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-warmup", type=int, default=8, help="Startup frames discarded before capture")
    parser.add_argument("--task", default="pick up the cube")
    parser.add_argument("--tokenizer-dir", type=Path,
                        default=Path(r"C:\Friston-3-Models\paligemma_tokenizer"))
    state_group = parser.add_mutually_exclusive_group()
    # 第二个独立互斥组：实际归一化状态和全零诊断状态不能同时传；与图像模式组互不排斥。
    state_group.add_argument("--state-normalized", type=float, nargs="+",
                             help="Already normalized state, using the checkpoint's training statistics/order")
    # nargs="+"要求该选项后至少一个数值，并收集为列表；type=float逐个转换。
    state_group.add_argument("--state-zero", action="store_true",
                             help="DIAGNOSTIC ONLY: use a zero placeholder, not measured robot state")
    parser.add_argument("--num-steps", type=int, help="Default: checkpoint's num_inference_steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Optional raw action .pt output")
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parents[1] / "learning" / "pi05_lowmem_report.json")
    # 移动诊断代码后保留已公开的默认报告位置，--report 可指定其他位置。
    args = parser.parse_args(argv)
    # 到此才真正解析输入；上面的add_argument只是登记规则。--help会在此打印帮助并退出。
    # 下方是当前已有的选项组合/数值完整性检查；parser.error打印用法并以错误状态结束CLI。
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
    # 【调用方】文件末尾的__main__分支；顺序为解析参数→读取版本→加载→可选推理→写诊断报告。
    # 作为模块被policy/training导入时，不会自动执行main。
    args = parse_args()
    import torch

    versions = {}
    for package in ("torch", "accelerate", "transformers", "lerobot", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
            # 查询已安装发行包版本，不下载依赖；遇到没有元数据的安装方式时在except记unknown。
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    print("Versions:", json.dumps(versions), flush=True)
    torch.manual_seed(args.seed)
    # 固定随机数种子便于诊断复现；不代表不同硬件/库版本必然得到逐位相同的结果。
    map_file = args.device_map or Path(__file__).with_name("pi05_device_map.json")
    # 优先使用显式路径、新位置；兼容已部署项目仍在 learning/ 下的设备表。
    if args.device_map is None and not map_file.is_file():
        map_file = Path(__file__).resolve().parents[1] / "learning" / map_file.name
    device_map = None
    if args.gpu_layers is None and map_file.is_file():
        data = json.loads(map_file.read_text(encoding="utf-8-sig"))
        device_map = data.get("device_map", data)
        # 兼容两种JSON布局：{"device_map": {...}} 或直接{模块名:设备}。
        print("Using device map:", map_file, flush=True)
    elif args.device_map is not None:
        raise FileNotFoundError(map_file)
    device = torch.device(f"cuda:{args.gpu}")
    report = {"versions": versions, "status": "STARTED"}
    with RamMonitor() as monitor:
        # with进入启动后台采样，离开停止线程；monitor是RamMonitor.__enter__返回的实例。
        try:
            model, loading = loader.load_pi05_lowmem(args.model_dir, device_map, args.gpu,
                                              4 if args.gpu_layers is None else args.gpu_layers, args.chunk_mib)
            report.update(loading, status="LOAD_OK")
            # dict.update把加载报告合并，再设置status；不会改变model里的参数。
            if args.smoke or args.image or args.camera:
                input_metadata = {}
                report["inputs"] = input_metadata
                # 报告与局部变量指向同一dict；prepare_inputs填入metadata后，report会看到相同变化。
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
                    # parents=True创建缺失父目录，exist_ok=True允许目录已存在。
                    torch.save(actions.detach().cpu(), args.output)
                    # 保存原始动作张量便于检查；detach移除梯度引用，cpu移到主内存，不会驱动机械臂。
                    report["action_file"] = str(args.output)
        except Exception as error:
            report.update(status="FAILED", error_type=type(error).__name__, error=str(error))
            raise
            # 记录失败原因后裸raise继续抛原异常，调用者/终端仍能看到真实失败，而非伪装成功。
        finally:
            # 成功或异常都尽量写报告，便于定位加载/输入/推理发生在哪一步。
            report["sampled_process_peak_rss_gib"] = round(monitor.peak / loader.GIB, 4) if monitor.peak else None
            report["sampled_process_peak_rss_scope"] = "loading_input_preparation_and_inference"
            report["final_memory"] = loader.snapshot(device)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            # dumps把dict编码成JSON文本，indent=2缩进，ensure_ascii=False保留中文；write_text写入文件。
            print("Report:", args.report, flush=True)


if __name__ == "__main__":
    # 让旧 loader.RamMonitor 等延迟入口也指向当前诊断模块里的同一类/函数对象。
    sys.modules.setdefault(f"{__package__}.diagnostics", sys.modules[__name__])
    main()
