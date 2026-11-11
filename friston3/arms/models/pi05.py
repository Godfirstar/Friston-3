"""Friston-3 使用 LeRobot π₀.₅ 的模型适配层。

训练时，compute_loss 从示教动作构造带噪动作，结合冻结前缀预测速度。
推理时，Pi05Policy.predict 用同一个动作专家的 sample_actions 生成连续动作块。
旧版 SemanticPolicy 在示教词表中选择子任务。联合检查点还包含 joint.JointPolicy
训练的共享语言适配器：高层生成当前子任务文本，低层以该文本为条件生成连续动作。
"""
from __future__ import annotations

import copy
import importlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..config import JOINT_FORMAT, ACTION, DATASET_JOINTS, JOINTS as ROBOT_JOINT_KEYS, STATE
from . import loader
from .weights import checkpoint_reader, fingerprint, read_json

PALI = 'paligemma_with_expert.paligemma'
EXPERT = 'paligemma_with_expert.gemma_expert.model'
PROJECTIONS = ('action_in_proj', 'action_out_proj', 'time_mlp_in', 'time_mlp_out')
LORA_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj')


class LoRALinear(nn.Module):
    # 继承PyTorch nn.Module：获得子模块/参数注册、设备转换、train/eval等机制。
    # 【创建方】attach_trainables用本类替换专家的nn.Linear；后续模型前向会调用forward。
    """在冻结的线性层上学习 FP32 低秩更新，保留 Gemma 检查 dtype 所需的 weight/bias 属性。"""
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        # __init__在LoRALinear(base,rank,alpha)构造实例时执行；self指这一层实例。
        # base是被替换但保留下来的原nn.Linear；rank/alpha来自CLI或adapter配置。
        super().__init__()  # 初始化 nn.Module 注册机制，子层与 Parameter 才会被收集。
        self.base = base
        # 给Module属性赋另一个Module，PyTorch会注册它为子模块；没有复制原权重。
        self.base.requires_grad_(False)  # 冻结预训练 W，不为 W 累积梯度。
        self.in_features, self.out_features = base.in_features, base.out_features
        # 读取原线性层输入/输出宽度；若输入为[B,H,in]，原层输出为[B,H,out]。
        self.rank, self.alpha, self.scaling = rank, alpha, alpha / rank
        # 一次解包赋三个属性；scaling是普通Python数值，A/B才是训练参数。
        # A:[r,in]、B:[out,r]，用 BA 表示低秩增量，参数量从 out*in 降到 r*(in+out)。
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        # torch.empty只分配[r,in]存储、尚无初始化数值；nn.Parameter把tensor登记成可训练参数。
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        # zeros创建[out,r]的全零矩阵；与base放同一设备，但保持FP32的训练增量。
        # B 初始为零，初次前向等于原模型；首步 A 的梯度为零是正常现象。
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # nn.init来自PyTorch；函数末尾_表示原地填充A，不是返回一个替代A的新Parameter。

    @property
    def weight(self):
        # @property让调用方写layer.weight即可执行这个函数；转交原层属性供Gemma检查dtype。
        return self.base.weight

    @property
    def bias(self):
        # 同理暴露原层bias；不额外创建或训练一份偏置。
        return self.base.bias

    def forward(self, value):
        # nn.Module的__call__会调forward；value常为[B,H,in]，函数返回[B,H,out]。
        original = self.base(value)
        # 调用原始nn.Linear：value @ W.T + b；原权重冻结，但前向计算仍执行。
        # F.linear(x,A) 相当于 x @ A.T；两次线性变换得到 x @ A.T @ B.T。
        update = F.linear(F.linear(value.float(), self.lora_A), self.lora_B) * self.scaling
        # 内层输入转FP32，再乘A.T：[B,H,in]→[B,H,r]；外层乘B.T→[B,H,out]。
        # F.linear没有自行登记参数，它使用前面已经登记的A/B，因此autograd能给它们求梯度。
        return original + update.to(original.dtype)


def attach_trainables(model: nn.Module, rank: int, alpha: float) -> list[str]:
    # 【调用方】trainer.run与Pi05Policy.__init__；输入完整模型，原地装同一套LoRA结构。
    # 返回被替换层的完整名称list；模型对象本身没有作为返回值，因为已在原对象中修改。
    # trainer.run 和 Pi05Policy 都调用本函数：训练与加载必须建出相同参数名。
    model.requires_grad_(False)
    # 对所有已注册参数递归冻结；末尾_表示原地修改标志；没有删除计算或替换数值。
    expert = model.get_submodule(EXPERT)
    # get_submodule属于nn.Module；按EXPERT点分路径找到动作专家子模块。
    targets = []
    # 先生成 list 快照，再替换子层；不能边遍历动态模块树边往树上添加 LoRA 子模块。
    for relative, module in list(expert.named_modules()):
        # named_modules返回(相对名字,模块对象)迭代器；list先快照，避免遍历时替换树结构。
        if isinstance(module, nn.Linear) and relative.split('.')[-1] in LORA_TARGETS:
            # isinstance检查类；split按点切开，[-1]取最后段，例如q_proj；只替换目标Linear。
            parent_name, _, leaf = relative.rpartition('.')
            # rpartition从最后一个点切成(父路径,分隔符,叶子名)；_表示本处不使用分隔符。
            parent = expert.get_submodule(parent_name) if parent_name else expert
            # 如果没有父路径，就直接以expert作父对象；否则找到对应嵌套子模块。
            setattr(parent, leaf, LoRALinear(module, rank, alpha))
            # setattr动态写属性，等价于parent.q_proj=新层；新层内部保留原module作为base。
            targets.append(f'{EXPERT}.{relative}')
    expected = len(expert.layers) * len(LORA_TARGETS)
    if len(targets) != expected:
        raise ValueError(f'Expected {expected} expert attention projections, found {len(targets)}')
    for name in PROJECTIONS:
        # 只有四个输入/输出/时间投影整层训练；并非把整个 action expert 全量微调。
        module = model.get_submodule(name)
        module.to(dtype=torch.float32).requires_grad_(True)
        # Module.to会原地转换已注册权重精度并返回自身，因此可链式调用requires_grad_。
        module.train()
        # train()设置层训练模式；与是否求梯度是不同开关，不能用eval替代冻结。
    expert.train()
    # 后缀训练读取冻结前缀的 KV 缓存。这里关闭专家的梯度检查点，保留缓存路径。
    expert.gradient_checkpointing = False
    return targets


def trainable_dict(model: nn.Module) -> dict[str, nn.Parameter]:
    # 【调用方】训练optimizer构造/保存/部署加载；named_parameters给出(名字,Parameter)。
    # dict推导式用if筛选可训练者；返回的是原参数对象引用，优化器据此更新模型本体。
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def adapter_statistics(config: dict, model_dir: Path, rank: int, alpha: float) -> dict:
    """在加载模型权重之前，请验证适配器的坐标系。"""
    # 【调用方】trainer.run的resume/init-from路径；核对映射、加载版本和模型配置后返回统计副本。
    loader.validate_loader_compatibility(config)
    if config.get('format') not in ('friston_pi05_expert_lora_v1', JOINT_FORMAT):
        raise ValueError('Expected an action or joint Pi0.5 adapter')
    # 与部署共用兼容约定；旧检查点按已知历史指纹识别，新检查点按版本识别。
    model_config = read_json(model_dir / 'config.json')
    expected = {
        'joint_order': DATASET_JOINTS,
        'model_slots': list(range(6)),
        'supervised_action_dims': 6,
        'horizon': int(model_config['chunk_size']),
        'max_action_dim': int(model_config['max_action_dim']),
        'base_config_sha256': fingerprint(model_dir / 'config.json'),
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
    """仅加载已学习的张量；优化器和 RNG 恢复属于恢复操作。"""
    # trainer.run 与 Pi05Policy 都按 trainable_dict 的注册键加载同一份增量权重。
    # 副作用是原地改Parameter数值；不创建新优化器，不恢复采样随机流。
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


def tokenizer_for(args):
    from transformers import AutoTokenizer
    if not hasattr(args, "_tokenizer"):
        args._tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), local_files_only=True)
        args._tokenizer.padding_side = "right"
    return args._tokenizer


def _encode_state_task(model, device, state, task, tokenizer, *, padding, state_step=None):
    """SO-101 六维归一化状态 → 补齐模型槽位 → 拼入任务文本 → token 和有效位。"""
    from lerobot.policies.pi05 import processor_pi05 as processor

    # [6] → [1,6] → [1,max_state_dim]。补齐只为适配模型输入，物理关节仍为六个。
    state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    joint_training_format = hasattr(model, "joint_adapter")
    if joint_training_format:
        state = state.clamp(-1, 1)
    state = F.pad(state, (0, model.config.max_state_dim - state.shape[-1]))
    transition = {
        processor.TransitionKey.OBSERVATION: {processor.OBS_STATE: state},
        processor.TransitionKey.COMPLEMENTARY_DATA: {"task": [task]},
    }
    if state_step is None:
        state_step = processor.Pi05PrepareStateTokenizerProcessorStep(max_state_dim=model.config.max_state_dim)
    prompt = state_step(transition)[processor.TransitionKey.COMPLEMENTARY_DATA]["task"]
    if joint_training_format:
        # 论文的控制模式标签区分关节和末端控制。SO-101 示教记录的是关节目标。
        # 在 LeRobot 清理下划线之后加标签，训练和部署共用此函数。
        prompt = [f"<control_mode> joint <control_mode>\n{text}" for text in prompt]
    limit = model.config.tokenizer_max_length if padding == "max_length" or not joint_training_format else None
    encoded = tokenizer(prompt, padding=padding, truncation=not joint_training_format,
                        max_length=limit, return_tensors="pt")
    if joint_training_format and encoded["input_ids"].shape[-1] > model.config.tokenizer_max_length:
        raise ValueError("Low-level subtask and joint state exceed the prompt length")
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device=device, dtype=torch.bool)


def prepare_inputs(model, args, device, state, images, task):
    """训练样本的当前状态、RGB 图像和指令 → LeRobot 条件前缀输入。"""
    tokens, text_mask = _encode_state_task(model, device, state, task, tokenizer_for(args), padding="max_length")
    processed, masks = prepare_images(model, device, images)
    return {"images": processed, "img_masks": masks, "tokens": tokens, "masks": text_mask}


def prepare_images(model, device, images):
    """各路 RGB 数组 → LeRobot 图像张量及相机有效位，保留模型配置的视角顺序。"""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    # RGB [高,宽,3] → [1,3,高,宽]，并从 uint8 像素改成 [0,1] float。
    batch = {key: torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
             for key, rgb in images.items()}
    # 字典推导式保留相机key，对每个rgb做同一变换；items()逐个产生(key,value)。
    # ascontiguousarray确保数组在内存连续；from_numpy通常与CPU数组共享数据存储。
    # permute(2,0,1):[高,宽,3]→[3,高,宽]；unsqueeze(0)→[1,3,高,宽]。
    # float()转float32，除255使uint8像素0..255变成0..1；此时尚未做模型的-1..1归一化。
    proxy = SimpleNamespace(config=model.config, parameters=lambda: iter([torch.empty(0, device=device)]))
    # SimpleNamespace 是Python的轻量属性容器，这里提供官方函数所需的最小对象。
    # lambda 是匿名函数；iter(list)建立迭代器；empty(0)是零元素tensor，只用于报告device。
    # 这样官方预处理读取 next(self.parameters()).device 时拿到执行设备，而非offload的meta设备。
    with torch.no_grad():
        # with 进入上下文，本缩进块不记录autograd图；退出后恢复先前状态。
        processed, masks = PI05Policy._preprocess_images(proxy, batch)
        # 调用官方类上的方法并显式传self=proxy；resize+pad到224平方并归一化到[-1,1]。
        # processed 是各视角[1,3,224,224]的list；masks是各视角[1] bool的list。
    return processed, masks


def make_frozen_prefix(model, inputs: dict) -> tuple[torch.Tensor, Any]:
    """计算图像和任务文本的有效位及各层冻结前缀的 KV 缓存。"""
    # LeRobot 的 mask 辅助函数随版本位于模型实例或其定义模块。
    impl = importlib.import_module(type(model).__module__)
    make_mask = getattr(impl, 'make_att_2d_masks')
    to4d = getattr(model, '_prepare_attention_masks_4d', None)
    if to4d is None:
        to4d = getattr(impl, 'prepare_attention_masks_4d')
    # no_grad 只关闭计算KV这一段建图；返回普通 tensor，后面的可训练分支仍能正常 backward。
    with torch.no_grad():
        # SigLIP 图像嵌入与语言嵌入拼成前缀，一起生成 padding mask 与注意力分组。
        embeds, pad, att = model.embed_prefix(inputs['images'], inputs['img_masks'], inputs['tokens'], inputs['masks'])
        mask = to4d(make_mask(pad, att))
        positions = torch.cumsum(pad, dim=1) - 1
        pg = model.paligemma_with_expert
        # 使用 SDPA 为冻结前缀计算注意力。
        pg.paligemma.model.language_model.config._attn_implementation = 'sdpa'
        # 只传 prefix embeddings，让 PaliGemma 返回各层 KV。
        _, cache = pg.forward(attention_mask=mask, position_ids=positions,
                              past_key_values=None, inputs_embeds=[embeds, None], use_cache=True)
    if cache is None:
        raise RuntimeError("Prefix KV cache is missing")
    return pad.detach(), cache


def flow_inputs(actions: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor):
    """【输入/输出】actions/noise:[B,H,D]，timestep:[B]；返回(x_t,目标速度)两个同shape张量。"""
    # t 是数据与噪声之间的 flow 时间，不是动作块内第几个时刻，也不是训练 step。
    t = timestep[:, None, None]
    # 两个None各插一轴：[B]→[B,1,1]，用于广播到H与D。
    # x_t=(1-t)*a+t*噪声；对 t 求导的目标速度 u=噪声-a。t=0 是动作，t=1 是噪声。
    return t * noise + (1 - t) * actions, noise - actions


def make_flow(model, actions_6, device, seed=None):
    """把一段真实的示教动作，转换成 Flow Matching 训练需要的三个东西：当前位置 x_t、正确速度 u_t、当前时间 t"""
    # 【调用关系】run / compute_loss → 本函数 → model.sample_noise/sample_time、flow_inputs。
    # actions_6:真实示教动作，[H,6]是归一化监督目标；返回 noisy[1,H,D]、target[1,H,D]、timestep[1]。
    # 当前 microbatch 固定为 1，H 与 D 从基础模型配置读取（常见配置 H=50,D=32）。
    if seed is not None:
        torch.manual_seed(seed)
        # 如果传入了随机种子 seed，就固定 PyTorch 的随机数生成序列，让实验可以复现。
    actions = torch.zeros(
        (1, model.config.chunk_size, model.config.max_action_dim),
        dtype=torch.float32, device=device,
    )
    # zeros(shape,dtype,device)创建全0张量，当前shape=(1,50,32)，机械臂只有 6 个控制维度，但这个 π₀.₅ 模型接口支持 32 维动作，所以需要先填充。
    actions[..., :6] = torch.as_tensor(actions_6, dtype=torch.float32, device=device)
    # ...代表前面的所有轴，:6代表最后一轴0..5；[H,6]可广播赋给[1,H,6]。
    # 其余槽位只是模型接口填充；flow_loss 只监督真实六维，不要求学会“控制填充维”。
    noise = model.sample_noise(actions.shape, device)
    # 方法定义在LeRobot PI05Pytorch；产生与actions同shape的高斯噪声，不是示教数据。
    timestep = model.sample_time(1, device)
    # 参数1是batch数量；返回每份样本一个flow时间[1]，不是50个物理动作时刻。
    noisy, target = flow_inputs(actions, noise, timestep)
    # flow_inputs返回x_t及目标速度noise-actions；这里解包两个tensor而不更新模型。
    return noisy, target, timestep


def velocity(model, pad, cache, noisy, timestep):
    """给定带噪动作 x_t 和时间 t，由动作专家预测该点的速度。"""
    # 对照官方 forward 的速度预测：此处沿 denoise_step 的 suffix 分支进入相同动作专家。
    # embed_suffix、time MLP、AdaRMS 和各层 Attention 均由 LeRobot 完成，不在此处另写一份。
    # 【调用方】trainer.run/compute_loss；pad是条件有效位mask，cache是冻结prefix的KV。
    # noisy:[B,H,D]、timestep:[B]输入官方denoise_step；返回速度预测[B,H,D]。
    # denoise_step 是 LeRobot 的可微动作专家前向。
    # 不同版本在 denoise_step 是否会克隆其缓存方面有所不同。切勿让训练/评估调用修改已存储的、冻结的前缀缓存。
    return model.denoise_step(prefix_pad_masks=pad, past_key_values=copy.deepcopy(cache),
                               x_t=noisy, timestep=timestep)


def flow_loss(prediction, target):
    return F.mse_loss(
        prediction[..., :6].float(),
        target[..., :6].float(),
    )


def compute_loss(model, inputs, actions_6, device, seed=None):
    """【调用方】evaluate；将前面的“条件前缀→带噪动作→专家前向→标量loss”函数串成一次计算"""
    # 第二个返回值是中间张量元组，调用方可在统计后释放；loss本身仍是标量Tensor。
    pad, cache = make_frozen_prefix(model, inputs)
    noisy, target, timestep = make_flow(model, actions_6, device, seed)
    prediction = velocity(model, pad, cache, noisy, timestep)
    loss = flow_loss(prediction, target)
    return loss, (pad, cache, noisy, target, timestep, prediction)


SEMANTIC_FORMAT = "friston_pi05_semantic_v1"
COMPLETE = "Goal complete."


class SemanticPolicy:
    """复用冻结的 π₀.₅ 骨干，只在高层读取语言 hidden 时施加残差。"""

    def __init__(self, model, tokenizer, rank, alpha, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        pali = model.paligemma_with_expert.paligemma
        self.language = pali.model.language_model
        self.head = pali.lm_head
        embedding = self.language.get_input_embeddings()
        if (not isinstance(self.head, nn.Linear) or self.head.weight.shape != embedding.weight.shape
                or self.head.out_features != tokenizer.vocab_size or self.head.weight.requires_grad):
            raise ValueError("Semantic prediction requires a frozen PaliGemma vocabulary projection")
        self.vocab_size, width = self.head.weight.shape
        # Some HF conversions flag Gemma newline tokens as special. They separate
        # fields in our prompt and remain valid language input.
        self.special_ids = {i for i in tokenizer.all_special_ids
                            if not tokenizer.convert_ids_to_tokens(i).isspace()}
        special = [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]
        if len(set(special)) != 3 or any(i is None or not 0 <= i < self.vocab_size for i in special):
            raise ValueError("Incompatible text vocabulary or BOS/EOS/padding IDs")
        if not 0 < rank <= width or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Invalid semantic adapter rank or alpha")
        self.scale = alpha / rank
        # adapter 独立注册及存盘；动作前缀从不调用它，也不会把骨干交给语言优化器。
        self.adapter = nn.Sequential(nn.Linear(width, rank, bias=False), nn.Linear(rank, width, bias=False))
        self.adapter.to(device=self.device, dtype=torch.float32)
        nn.init.zeros_(self.adapter[1].weight)
        self.candidates = {}

    def _text_ids(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if (not ids or len(ids) > self.model.config.tokenizer_max_length
                or any(i in self.special_ids or i < 0 or i >= self.vocab_size for i in ids)):
            raise ValueError("Semantic text is empty, too long, or contains special/out-of-vocabulary tokens")
        return ids

    def _target(self, subtask):
        if subtask is not None and (not isinstance(subtask, str) or not subtask.strip()
                                   or subtask != subtask.strip() or subtask.casefold() == COMPLETE.casefold()):
            raise ValueError(f"Invalid subtask label or collision with reserved text {COMPLETE!r}")
        return self._text_ids(COMPLETE if subtask is None else subtask) + [self.tokenizer.eos_token_id]

    def _mask(self, allowed):
        # PiGemma 的 SDPA 要求加性 mask 与 query 同 dtype；0 可见，极小值屏蔽。
        dtype = self.language.layers[0].self_attn.q_proj.weight.dtype
        mask = torch.zeros_like(allowed, dtype=dtype).masked_fill_(~allowed, torch.finfo(dtype).min)
        return mask[:, None]

    @torch.no_grad()
    def _observe(self, images, img_masks, global_goal, answer_prefix, use_cache):
        if not isinstance(global_goal, str) or not global_goal.strip():
            raise ValueError("Semantic prediction requires a global goal")
        if len(images) != len(img_masks) or not images or any(mask.shape != (1,) for mask in img_masks):
            raise ValueError("Semantic prediction requires one current frame per configured camera")
        views = [(image, mask) for image, mask in zip(images, img_masks, strict=True) if mask.item()]
        if not views:
            raise ValueError("Semantic prediction received no observed camera view")
        prompt = self._text_ids(f"Goal: {global_goal.strip()}\nCurrent subtask:\n")
        ids = [self.tokenizer.bos_token_id] + prompt + list(answer_prefix)
        tokens = torch.tensor([ids], device=self.device)
        embeds, valid, _ = self.model.embed_prefix(
            [image for image, _ in views], [mask for _, mask in views],
            tokens, torch.ones_like(tokens, dtype=torch.bool))
        # 图像和提示互见；任何前缀位置都看不到答案，每个答案位置只能看到左侧及自身。
        groups = torch.zeros_like(valid, dtype=torch.long)
        if answer_prefix:
            groups[:, -len(answer_prefix):] = 1
        groups = groups.cumsum(dim=1)
        allowed = (groups[:, None, :] <= groups[:, :, None]) & valid[:, None, :] & valid[:, :, None]
        result = self.language(inputs_embeds=embeds, attention_mask=self._mask(allowed),
                               position_ids=valid.long().cumsum(1) - 1, use_cache=use_cache)
        return result.last_hidden_state, result.past_key_values, valid

    def _logits(self, hidden):
        value = hidden.float()
        adapted = value + self.scale * self.adapter(value)
        return self.head(adapted.to(self.head.weight.dtype)).float()

    def loss(self, images, img_masks, global_goal, subtask):
        """单帧 teacher forcing；只在下一个标签 token 和末尾 EOS 上计算 CE。"""
        target = self._target(subtask)
        hidden, _, _ = self._observe(images, img_masks, global_goal, target[:-1], use_cache=False)
        # 第一项是提示末位置对首 token 的预测；B=1 不填充文本，也不监督视觉或提示。
        logits = self._logits(hidden[0, -len(target):])
        return F.cross_entropy(logits, torch.tensor(target, device=self.device))

    def set_candidates(self, subtasks):
        if not isinstance(subtasks, list) or not subtasks or not all(isinstance(s, str) for s in subtasks):
            raise ValueError("Semantic checkpoint requires a nonempty subtask vocabulary")
        candidates = {tuple(self._target(text)): text for text in [*subtasks, None]}
        if len(candidates) != len(subtasks) + 1:
            raise ValueError("Semantic labels collide after tokenization")
        self.candidates = candidates

    @torch.no_grad()
    def predict(self, images, img_masks, global_goal):
        if not self.candidates:
            raise RuntimeError("Missing semantic checkpoint vocabulary")
        hidden, cache, valid = self._observe(images, img_masks, global_goal, [], use_cache=True)
        if cache is None:
            raise RuntimeError("Semantic language decoder did not return a KV cache")
        remaining = list(self.candidates)
        offset = 0
        while len(remaining) > 1:
            # 受约束的贪心解码；共享开头的标签继续逐词比较，EOS 也可与较长标签竞争。
            options = sorted({ids[offset] for ids in remaining})
            if len(options) > 1:
                scores = self._logits(hidden[:, -1])[:, options]
                if not torch.isfinite(scores).all():
                    raise ValueError("Nonfinite semantic token scores")
                token = options[scores.argmax().item()]
                remaining = [ids for ids in remaining if ids[offset] == token]
            else:
                token = options[0]
            if len(remaining) == 1:
                break  # 余下 token（含 EOS）只剩唯一合法续写，无需重复运行冻结骨干。
            offset += 1
            tokens = torch.tensor([[token]], device=self.device)
            valid = F.pad(valid, (0, 1), value=True)
            output = self.language(inputs_embeds=self.model.paligemma_with_expert.embed_language_tokens(tokens),
                                   attention_mask=self._mask(valid[:, None, :]),
                                   position_ids=valid.long().sum(dim=1, keepdim=True) - 1,
                                   past_key_values=cache, use_cache=True)
            hidden, cache = output.last_hidden_state, output.past_key_values
        return self.candidates[remaining[0]]

    @classmethod
    def from_checkpoint(cls, model, tokenizer, device, directory, model_dir, tokenizer_dir, image_keys, action_tasks):
        """与动作检查点共用骨干和相机，严格加载独立的、已受监督的语言参数。"""
        from safetensors.torch import load_file

        directory = Path(directory)
        cfg = read_json(directory / "adapter_config.json")
        if cfg.get("format") != SEMANTIC_FORMAT or cfg.get("steps_completed", 0) <= 0:
            raise ValueError("Expected a trained semantic checkpoint")
        loader.validate_loader_compatibility(cfg)
        if (cfg.get("base_config_sha256") != fingerprint(Path(model_dir) / "config.json")
                or cfg.get("tokenizer_sha256") != fingerprint(Path(tokenizer_dir) / "tokenizer.json")):
            raise ValueError("Semantic checkpoint base configuration or tokenizer differs")
        if cfg.get("image_keys") != list(image_keys) or cfg.get("has_completion") is not True:
            raise ValueError("Semantic checkpoint needs matching cameras and explicit completed-goal supervision")
        instance = cls(model, tokenizer, cfg["rank"], cfg["alpha"], device)
        instance.set_candidates(cfg["subtasks"])
        unsupported = set(cfg["subtasks"]).difference(action_tasks)
        if unsupported:
            raise ValueError(f"Semantic subtasks absent from action checkpoint: {sorted(unsupported)}")
        saved = load_file(str(directory / "trainable.safetensors"), device="cpu", backend="pread")
        if any(not torch.isfinite(t).all() for t in saved.values()):
            raise ValueError("Nonfinite semantic adapter weights")
        instance.adapter.load_state_dict(saved, strict=True)
        instance.adapter.eval().requires_grad_(False)
        return instance


class Pi05Policy:
    """加载动作检查点及其训练统计，并按配置在同一骨干上进行高低层预测。"""

    def __init__(self, options):
        # PolicySession 的推理线程首次任务时创建；options 来自 policy_options。
        # 本类封装输入/权重适配；真正神经网络类定义在安装的 LeRobot modeling_pi05.py。
        from transformers import AutoTokenizer
        from lerobot.policies.pi05 import processor_pi05
        from .joint import JointPolicy

        self.options = options
        self.device = torch.device(f"cuda:{options['gpu']}")
        # torch.device 是设备描述对象；创建它不等于把整个大模型都放进 GPU。
        # loader 后面会按 gpu_layers 做 CPU/GPU 分布，数据仍须按模型接口放到指定设备。
        if not torch.cuda.is_available():
            raise RuntimeError("Pi0.5 needs the installed CUDA LeRobot environment")
        torch.cuda.set_device(self.device)
        adapter = Path(options["adapter_dir"])
        self.checkpoint = cfg = read_json(adapter / "adapter_config.json")
        # 连续赋值：cfg 和 self.checkpoint 指向同一个 dict；read_json 将配置解析为 Python 对象。
        # cfg 描述训练时的关节、统计、LoRA 配置等，权重张量则在另一个 safetensors 文件。
        if cfg.get("dataset_fps") is not None and float(cfg["dataset_fps"]) != float(options["fps"]):
            raise ValueError("The checkpoint dataset_fps differs from the service control rate")
        joint = cfg.get("format") == JOINT_FORMAT
        self.open_subtasks = joint and options["subtask_mode"] == "trained"
        self.predicts_boxes = joint and bool(cfg.get("predicts_boxes"))
        if not joint and cfg.get("format") != "friston_pi05_expert_lora_v1":
            raise ValueError("Unsupported Pi0.5 adapter format")
        if joint:
            if not cfg.get("has_action_training"):
                raise ValueError("This discrete-only checkpoint needs action post-training before deployment")
            if cfg.get("tokenizer_sha256") != fingerprint(Path(options["tokenizer_dir"]) / "tokenizer.json"):
                raise ValueError("Joint checkpoint text tokenizer differs")
        elif options["subtask_mode"] == "trained" and Path(options["semantic_adapter_dir"]) == adapter:
            raise ValueError("For a legacy action checkpoint, pass its separate --semantic-adapter-dir")
        if cfg.get("joint_order") != [joint.removesuffix(".pos") for joint in ROBOT_JOINT_KEYS]:
            raise ValueError("Adapter joint order differs from the robot")
        if cfg.get("model_slots") != list(range(6)) or cfg.get("supervised_action_dims") != 6:
            raise ValueError("Adapter action mapping differs from the robot")
        loader.validate_loader_compatibility(cfg)
        # 与续训共用相同的 loader 格式约定：检查加载格式版本，源码 SHA256 只用于新格式的追溯。
        # 基础模型配置仍须与训练时一致；解除源码字节绑定不改变其他数据/模型检查。
        path = Path(options["model_dir"]) / "config.json"
        if fingerprint(path) != cfg["base_config_sha256"]:
            raise ValueError(f"Checkpoint dependency differs: {path.name}")
        with checkpoint_reader():
            # checkpoint_reader 是 models/weights.py 提供的权重读取上下文管理器，
            # with 进入时切换权重读取方式，退出时恢复；不是“在这里训练一次”。
            self.model, _ = loader.load_pi05_lowmem(
                options["model_dir"], gpu=options["gpu"], gpu_layers=options["gpu_layers"],
                chunk_mib=options["chunk_mib"])
        targets = attach_trainables(self.model, int(cfg["rank"]), float(cfg["alpha"]))
        # targets 是 attach_trainables 实际替换的动作专家层名。
        # 先重建与训练一致的 LoRA/投影层，再把保存的增量参数放进同名参数槽。
        if targets != cfg["target_modules"]:
            raise ValueError("Adapter target modules differ")
        self.tokenizer = AutoTokenizer.from_pretrained(options["tokenizer_dir"], local_files_only=True)
        self.tokenizer.padding_side = "right"
        if joint:
            # 安装到与训练相同的骨干层；高层解码与 sample_actions 读取同一组适配后的 KV。
            self.semantic = JointPolicy(self.model, self.tokenizer, cfg["rank"], cfg["alpha"], self.device)
        # 按参数名把 LoRA 和四个投影层的增量权重复制到模型，保持设备映射。
        load_adapter_weights(trainable_dict(self.model), adapter)
        self.model.eval().requires_grad_(False)
        # 链式调用：eval 返回模型自己，所以继续调用 requires_grad_ 冻结所有参数。
        # eval 切换模块的训练/推理行为，requires_grad_(False) 冻结参数梯度。
        # 两者与“全局禁止 autograd”不同；下方 guided RTC 仍需要内部求导。
        self.horizon = self.model.config.chunk_size
        # horizon 是一个动作块的时间长度 int，例如50；不是去噪迭代次数。
        if cfg["horizon"] != self.horizon:
            raise ValueError("Adapter action horizon differs")
        self.image_keys = cfg["image_keys"] if "image_keys" in cfg else [cfg["image_key"]]
        self.semantic_image_keys = cfg.get("semantic_image_keys", self.image_keys)
        if options["subtask_mode"] != "direct" and not self.open_subtasks:
            self.subtasks = cfg.get("tasks") or [cfg.get("task")]
            if not isinstance(self.subtasks, list) or not all(isinstance(s, str) and s.strip()
                                                              for s in self.subtasks):
                raise ValueError("High-level selection requires the action checkpoint's trained task labels")
        self.stats = tuple(np.asarray(cfg["statistics"][key][quantile], dtype=np.float32)
                           for key in (STATE, ACTION) for quantile in ("q01", "q99"))
        # 双层生成式按顺序得到四个 [6] 数组：状态q01、状态q99、动作q01、动作q99。
        # np.asarray 将 JSON 里的数值列表转换为 NumPy 数组，便于逐元素计算。
        if any(v.shape != (6,) or not np.isfinite(v).all() for v in self.stats):
            raise ValueError("Adapter statistics must contain six finite values")
        if any(np.any(hi <= lo) for lo, hi in (self.stats[:2], self.stats[2:])):
            raise ValueError("Adapter quantile spans must be positive")
        self.state_step = processor_pi05.Pi05PrepareStateTokenizerProcessorStep(
            max_state_dim=self.model.config.max_state_dim)
        # 这是 LeRobot processor_pi05.py 定义的可调用对象，负责把数值状态整理进 prompt。
        if options["subtask_mode"] == "trained" and not joint:
            self.semantic = SemanticPolicy.from_checkpoint(
                self.model, self.tokenizer, self.device, options["semantic_adapter_dir"],
                options["model_dir"], options["tokenizer_dir"], self.image_keys, self.subtasks)
        if options["rtc"]:
            from lerobot.policies.rtc.configuration_rtc import RTCConfig
            from lerobot.policies.rtc.modeling_rtc import RTCProcessor
            rtc = RTCConfig(enabled=True, mode="guided", execution_horizon=options["blend_steps"])
            self.model.config.rtc_config = rtc
            self.model.rtc_processor = RTCProcessor(rtc)

    def prepare_images(self, frames, image_keys=None):
        """从同一次采集中选出该分支训练用的视角，再执行官方预处理。"""
        import cv2

        keys = self.image_keys if image_keys is None else image_keys
        rgb = {key: cv2.cvtColor(frames[key], cv2.COLOR_BGR2RGB) for key in keys}
        images, masks = prepare_images(self.model, self.device, rgb)
        # 真机只向模型传入有当前帧的视角。训练路径保留官方返回的 padding 视角和掩码。
        valid = [(image, mask) for image, mask in zip(images, masks) if bool(mask.any())]
        return [image for image, _ in valid], [mask for _, mask in valid]

    def _normalized_state(self, observation):
        state = np.asarray([observation[k] for k in ROBOT_JOINT_KEYS], dtype=np.float32)
        if not np.isfinite(state).all():
            raise ValueError("Nonfinite robot observation")
        lo, hi = self.stats[:2]
        return 2 * (state - lo) / (hi - lo) - 1

    def select_subtask(self, frames, global_goal, observation=None, *, return_boxes=False):
        images, masks = self.prepare_images(frames, self.semantic_image_keys)
        if self.open_subtasks:
            if observation is None:
                raise ValueError("Joint subtask prediction requires the current robot observation")
            state = self._normalized_state(observation)
            camera_keys = self.semantic_image_keys if self.predicts_boxes else None
            selected, boxes = self.semantic.predict_result(images, masks, global_goal, state, camera_keys)
            if selected is None and not self.checkpoint["has_completion"]:
                raise ValueError("Generated completion without any completion supervision in this checkpoint")
            return (selected, boxes) if return_boxes else selected
        return self.semantic.predict(images, masks, global_goal)

    def predict(self, frames, observation, task, previous=None, delay=0):
        """由 PolicySession 调用：相机+实测六关节+任务 -> H 步归一化/物理动作。

        H=self.horizon。返回两份动作：raw 的全部模型维度用于下一块 RTC，
        actions 只含 SO-101 的六维实际目标，交给服务按控制频率逐帧执行。
        """

        # 输入：frames 为视角键->BGR图像；observation 为六关节名->实测数值；task 为任务字符串。
        # 默认场景理解路径通常把任务转为英文；关闭 ground_scene 时也可直接传入原始指令。
        # previous 为旧块尚未执行的归一化动作列表；delay 为估计推理期间会执行的帧数。
        normalized = self._normalized_state(observation)
        action_lo, action_hi = self.stats[2:]
        # 单样本推理沿用动态文本长度，训练路径使用固定的 tokenizer_max_length。
        tokens, text_mask = _encode_state_task(
            self.model, self.device, normalized, task, self.tokenizer, padding=True, state_step=self.state_step)
        images, masks = self.prepare_images(frames)
        inputs = {"images": images, "img_masks": masks, "tokens": tokens, "masks": text_mask}
        kwargs = {}
        if self.options["rtc"] and previous:
            # previous 是旧块尚未消费的归一化动作；RTC 在生成新块时参考这段连续性。
            prefix = torch.as_tensor(np.asarray(previous), device=self.device, dtype=torch.float32)
            # prefix 为 [剩余时间步,D_action]；RTC 参考旧块连续性，不直接把它当新示教标签。
            kwargs = {"prev_chunk_left_over": prefix, "inference_delay": min(delay, len(previous)),
                      "execution_horizon": min(len(previous), delay + self.options["blend_steps"])}
        # Guided RTC 内部要求对动作求导，因此这里保留 autograd 的可用性。
        raw = self.model.sample_actions(**inputs, num_steps=self.options["num_steps"], **kwargs)
        # **inputs/**kwargs 将字典展开为关键字参数；sample_actions 定义在 LeRobot 官方模型中。
        # 一次 sample_actions 调用经过 num_steps 轮速度场预测，生成整块 [1,H,D]。
        # num_steps 是去噪迭代数，H 才是未来动作时刻数；不是只做一次网络前向。
        raw = raw.detach().float().cpu().numpy()[0]
        # detach 保留数值但断开求导关系；cpu 搬到主存；numpy 转为数组；[0] 取唯一批次。
        if raw.shape != (self.horizon, self.model.config.max_action_dim) or not np.isfinite(raw).all():
            raise ValueError("Invalid Pi0.5 action chunk")
        actions = (raw[:, :6] + 1) * (action_hi - action_lo) / 2 + action_lo
        # 切片 [:,:6] 保留所有 H 个时刻的前六个动作槽位，shape 从 [H,D_action] 到 [H,6]。
        # 前六维是训练时监督的 SO-101 槽位；反归一化后是绝对关节目标，不是增量。
        # 返回归一化全维动作和物理六维动作，分别供队列用于 RTC 与执行。
        return raw.tolist(), actions.tolist()
