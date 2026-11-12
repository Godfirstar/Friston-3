"""Friston-3 的 LeRobot π₀.₅ 训练与推理。

JointPolicy 在同一骨干上训练子任务文本、FAST 动作 token 和连续动作速度。
Pi05Policy 加载检查点，按当前观测生成子任务，再调用动作专家采样动作块。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from . import loader
from .weights import checkpoint_reader, read_json
from ..config import ACTION, DATASET_JOINTS, JOINT_FORMAT, JOINTS as ROBOT_JOINT_KEYS, POLICY_IMAGE_KEYS, STATE

PALI = 'paligemma_with_expert.paligemma'
EXPERT = 'paligemma_with_expert.gemma_expert.model'
PROJECTIONS = ('action_in_proj', 'action_out_proj', 'time_mlp_in', 'time_mlp_out')
LORA_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj')

COMPLETE = "Goal complete."

# OpenPI 使用扩展后的 big_vision SentencePiece 行号。HF 可将 loc/seg 记作 added
# tokens，或另加 <image>；len(HF tokenizer) 和 tokenizer.vocab_size 并不总是此值。
GEMMA_TEXT_VOCAB = 256_000
PALI_VOCAB = GEMMA_TEXT_VOCAB + 1024 + 128
FAST_TOP_ID = PALI_VOCAB - 1 - 128


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
            parent = expert.get_submodule(parent_name)
            # 有父路径找到对应嵌套子模块。
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
    # 动作专家读取当前观测的 KV；共享 K/V 适配器通过这条路径收到梯度。
    expert.gradient_checkpointing = False
    return targets


def trainable_dict(model: nn.Module) -> dict[str, nn.Parameter]:
    # 【调用方】训练optimizer构造/保存/部署加载；named_parameters给出(名字,Parameter)。
    # dict推导式用if筛选可训练者；返回的是原参数对象引用，优化器据此更新模型本体。
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def adapter_statistics(config: dict) -> dict:
    """读取训练时的归一化统计，确认相机和关节顺序符合当前输入。"""
    if config["format"] != JOINT_FORMAT:
        raise ValueError(f"Expected joint checkpoint {JOINT_FORMAT}")
    if config["image_keys"] != list(POLICY_IMAGE_KEYS) or config["joint_order"] != DATASET_JOINTS:
        raise ValueError("Checkpoint camera views or joint order differ from the current input")
    stats = config["statistics"]
    for key in (STATE, ACTION):
        low, high = (np.asarray(stats[key][name], dtype=np.float32) for name in ("q01", "q99"))
        if (low.shape != (6,) or high.shape != (6,) or not np.isfinite(low).all()
                or not np.isfinite(high).all() or np.any(high <= low)):
            raise ValueError(f"{key} requires six finite, positive quantile spans")
    return stats


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


def _encode_state_task(model, device, state, subtask, tokenizer):
    """低层提示包含关节控制标记、当前子任务和离散化的 SO-101 状态。"""
    from lerobot.policies.pi05 import processor_pi05 as processor

    state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).clamp(-1, 1)
    state = F.pad(state, (0, model.config.max_state_dim - state.shape[-1]))
    transition = {
        processor.TransitionKey.OBSERVATION: {processor.OBS_STATE: state},
        processor.TransitionKey.COMPLEMENTARY_DATA: {"task": [subtask]},
    }
    state_step = processor.Pi05PrepareStateTokenizerProcessorStep(max_state_dim=model.config.max_state_dim)
    [prompt] = state_step(transition)[processor.TransitionKey.COMPLEMENTARY_DATA]["task"]
    # LeRobot 会清理任务中的下划线，控制模式标记需在它格式化提示后添加。
    encoded = tokenizer(f"<control_mode> joint <control_mode>\n{prompt}", return_tensors="pt")
    if encoded["input_ids"].shape[-1] > model.config.tokenizer_max_length:
        raise ValueError("Low-level subtask and joint state exceed the prompt length")
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device=device, dtype=torch.bool)


def prepare_inputs(model, args, device, state, images, task):
    """训练样本的当前状态、RGB 图像和指令 → LeRobot 条件前缀输入。"""
    tokens, text_mask = _encode_state_task(model, device, state, task, tokenizer_for(args))
    processed, masks = prepare_images(model, device, images)
    return {"images": processed, "img_masks": masks, "tokens": tokens, "masks": text_mask}


def prepare_images(model, device, images):
    """各路 RGB 数组 → LeRobot 图像张量及相机有效位，保留模型配置的视角顺序。"""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    # RGB [高,宽,3] → [1,3,高,宽]，并从 uint8 像素改成 [0,1] float。
    batch = {key: torch.from_numpy(np.ascontiguousarray(images[key])).permute(2, 0, 1).unsqueeze(0).float() / 255.0
             for key in POLICY_IMAGE_KEYS}
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
    # LeRobot 先放真实图像，随后补其他视角。训练和推理都只编码两路真实观测。
    return processed[:len(POLICY_IMAGE_KEYS)], masks[:len(POLICY_IMAGE_KEYS)]


def flow_inputs(actions: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor):
    """【输入/输出】actions/noise:[B,H,D]，timestep:[B]；返回(x_t,目标速度)两个同shape张量。"""
    # t 是数据与噪声之间的 flow 时间，不是动作块内第几个时刻，也不是训练 step。
    t = timestep[:, None, None]
    # 两个None各插一轴：[B]→[B,1,1]，用于广播到H与D。
    # x_t=(1-t)*a+t*噪声；对 t 求导的目标速度 u=噪声-a。t=0 是动作，t=1 是噪声。
    return t * noise + (1 - t) * actions, noise - actions


def make_flow(model, actions_6, device, seed=None):
    """把一段真实的示教动作，转换成 Flow Matching 训练需要的三个东西：当前位置 x_t、正确速度 u_t、当前时间 t"""
    # 【调用关系】JointPolicy.flow_loss → 本函数 → model.sample_noise/sample_time、flow_inputs。
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


def flow_loss(prediction, target):
    return F.mse_loss(
        prediction[..., :6].float(),
        target[..., :6].float(),
    )


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
        self.separator_tokens = self._text("|")
        self.fast = FastTokenizer(fast_dir) if fast_dir is not None else None
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
        # 一个训练样本使用两路真实图像，最后一个提示 token 负责预测第一个目标。
        tokens = inputs["tokens"][:, inputs["masks"][0]]
        if tokens.shape[1] == 0:
            raise ValueError("Empty language prompt")
        if answer_prefix:
            tokens = torch.cat((tokens, tokens.new_tensor([answer_prefix])), dim=1)
        with torch.no_grad():
            embeds, valid, _ = self.model.embed_prefix(
                inputs["images"], inputs["img_masks"],
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
        prediction = self.model.denoise_step(prefix_pad_masks=valid, past_key_values=cache,
                                             x_t=noisy, timestep=timestep)
        return flow_loss(prediction, target)

    def fast_loss(self, inputs, actions_6):
        """低层输入已含标注子任务、状态及 Action 提示。目标为动作 token、终止符、EOS。"""
        if self.fast is None:
            raise ValueError("Set --fast-tokenizer-dir to a local FAST+ snapshot for FAST supervision")
        if np.shape(actions_6) != (self.model.config.chunk_size, 6):
            raise ValueError("FAST and Flow require the same SO-101 action horizon")
        target = [FAST_TOP_ID - i for i in self.fast.encode(actions_6)]
        return self._loss(inputs, target + self.separator_tokens + [self.eos])

    def _box_separator(self):
        if len(self.separator_tokens) != 1:
            raise ValueError("Grounded output needs a single-token '|' view separator")
        return self.separator_tokens[0]

    def _box_target(self, boxes):
        # 每个实际视角一组，| 结束该视角；连续两个 | 表示其间视角被标注为无相关目标。
        groups = {view: [] for view in POLICY_IMAGE_KEYS}
        for box in boxes:
            xmin, ymin, xmax, ymax = box["xyxy"]
            if box["view"] not in groups or not (0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1):
                raise ValueError("Invalid normalized box or unobserved grounding camera")
            groups[box["view"]].append((math.floor(ymin * 1023), math.floor(xmin * 1023),
                                        math.ceil(ymax * 1023), math.ceil(xmax * 1023)))
        target = []
        separator = self._box_separator()
        for view in POLICY_IMAGE_KEYS:
            for coords in sorted(groups[view]):
                target.extend(GEMMA_TEXT_VOCAB + c for c in coords)
            target.append(separator)
        return target

    def semantic_loss(self, images, img_masks, global_goal, subtask, state, boxes=None):
        views = POLICY_IMAGE_KEYS if boxes is not None else ()
        inputs = self._goal_inputs(images, img_masks, global_goal, state, views)
        if subtask is not None and (not isinstance(subtask, str) or not subtask.strip()
                                    or subtask.strip().casefold() == COMPLETE.casefold()):
            raise ValueError("Use null for completion and nonempty text for the current subtask")
        target = self._box_target(boxes) if boxes is not None else []
        target.extend(self._text(COMPLETE if subtask is None else subtask.strip()) + [self.eos])
        if len(target) > self.model.config.tokenizer_max_length:
            raise ValueError("Subtask target is longer than the inference generation budget")
        return self._loss(inputs, target)

    @torch.no_grad()
    def predict_result(self, images, img_masks, global_goal, state, *, predict_boxes=False):
        """按观察视角生成可选目标框，随后生成当前子任务。显式完成文本才表示任务结束。"""
        views = POLICY_IMAGE_KEYS if predict_boxes else ()
        inputs = self._goal_inputs(images, img_masks, global_goal, state, views)
        hidden, cache, valid = self._observe(inputs, [], use_cache=True)
        if cache is None:
            raise RuntimeError("The semantic language decoder returned no KV cache")
        phrase = []
        boxes = [] if views else None
        view_index, coords = 0, []
        separator = self._box_separator() if predict_boxes else None
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


class Pi05Policy:
    """加载联合检查点，在同一骨干上生成子任务和连续动作。"""

    def __init__(self, options):
        # PolicySession 的推理线程首次任务时创建；options 来自 policy_options。
        # 本类封装输入/权重适配；真正神经网络类定义在安装的 LeRobot modeling_pi05.py。
        from transformers import AutoTokenizer

        self.options = options
        self.device = torch.device(f"cuda:{options['gpu']}")
        # torch.device 是设备描述对象；创建它不等于把整个大模型都放进 GPU。
        # loader 后面会按 gpu_layers 做 CPU/GPU 分布，数据仍须按模型接口放到指定设备。
        if not torch.cuda.is_available():
            raise RuntimeError("Pi0.5 needs the installed CUDA LeRobot environment")
        torch.cuda.set_device(self.device)
        adapter = Path(options["adapter_dir"])
        cfg = read_json(adapter / "adapter_config.json")
        stats = adapter_statistics(cfg)
        if float(cfg["dataset_fps"]) != float(options["fps"]):
            raise ValueError("The checkpoint dataset_fps differs from the service control rate")
        self.predicts_boxes = cfg["predicts_boxes"]
        self.has_completion = cfg["has_completion"]
        if not cfg["has_action_training"]:
            raise ValueError("This discrete-only checkpoint needs action post-training before deployment")
        with checkpoint_reader():
            # checkpoint_reader 是 models/weights.py 提供的权重读取上下文管理器，
            # with 进入时切换权重读取方式，退出时恢复；不是“在这里训练一次”。
            self.model, _ = loader.load_pi05_lowmem(
                options["model_dir"], gpu=options["gpu"], gpu_layers=options["gpu_layers"],
                chunk_mib=options["chunk_mib"])
        attach_trainables(self.model, int(cfg["rank"]), float(cfg["alpha"]))
        # 先重建与训练一致的 LoRA/投影层，再把保存的增量参数放进同名参数槽。
        self.tokenizer = AutoTokenizer.from_pretrained(options["tokenizer_dir"], local_files_only=True)
        self.tokenizer.padding_side = "right"
        self.joint_policy = JointPolicy(self.model, self.tokenizer, cfg["rank"], cfg["alpha"], self.device)
        load_adapter_weights(trainable_dict(self.model), adapter)
        self.model.eval().requires_grad_(False)
        # 链式调用：eval 返回模型自己，所以继续调用 requires_grad_ 冻结所有参数。
        # eval 切换模块的训练/推理行为，requires_grad_(False) 冻结参数梯度。
        # 两者与“全局禁止 autograd”不同；下方 guided RTC 仍需要内部求导。
        self.horizon = self.model.config.chunk_size
        self.stats = tuple(np.asarray(stats[key][quantile], dtype=np.float32)
                           for key in (STATE, ACTION) for quantile in ("q01", "q99"))

        if options["rtc"]:
            from lerobot.policies.rtc.configuration_rtc import RTCConfig
            from lerobot.policies.rtc.modeling_rtc import RTCProcessor
            rtc = RTCConfig(enabled=True, mode="guided", execution_horizon=options["blend_steps"])
            self.model.config.rtc_config = rtc
            self.model.rtc_processor = RTCProcessor(rtc)

    def prepare_images(self, frames):
        """用训练时相同的顺序预处理工作区和腕部图像。"""
        import cv2

        rgb = {key: cv2.cvtColor(frames[key], cv2.COLOR_BGR2RGB) for key in POLICY_IMAGE_KEYS}
        return prepare_images(self.model, self.device, rgb)

    def _normalized_state(self, observation):
        state = np.asarray([observation[k] for k in ROBOT_JOINT_KEYS], dtype=np.float32)
        if not np.isfinite(state).all():
            raise ValueError("Nonfinite robot observation")
        lo, hi = self.stats[:2]
        return 2 * (state - lo) / (hi - lo) - 1

    def select_subtask(self, frames, global_goal, observation):
        """从最新双路图像、关节状态和总目标生成当前子任务及可选目标框。"""
        images, masks = self.prepare_images(frames)
        state = self._normalized_state(observation)
        selected, boxes = self.joint_policy.predict_result(
            images, masks, global_goal, state, predict_boxes=self.predicts_boxes)
        if selected is None and not self.has_completion:
            raise ValueError("此检查点中，在没有任何完成监督的情况下生成了完成状态。")
        return selected, boxes

    def predict(self, frames, observation, task, previous=None, delay=0):
        """由 PolicySession 调用：相机+实测六关节+任务 -> H 步归一化/物理动作。

        H=self.horizon。返回两份动作：raw 的全部模型维度用于下一块 RTC，
        actions 只含 SO-101 的六维实际目标，交给服务按控制频率逐帧执行。
        """

        # 输入：frames 为视角键->BGR图像；observation 为六关节名->实测数值；task 为任务字符串。
        # task 为高层根据最新观测生成的子任务。
        # previous 为旧块尚未执行的归一化动作列表；delay 为估计推理期间会执行的帧数。
        normalized = self._normalized_state(observation)
        action_lo, action_hi = self.stats[2:]
        tokens, text_mask = _encode_state_task(self.model, self.device, normalized, task, self.tokenizer)
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
