"""π₀.₅ 模型与权重路径；只有调用模型时才导入 PyTorch 和 LeRobot。"""
from __future__ import annotations

import os
from pathlib import Path
# 此处只导入标准库；torch/safetensors等重依赖放在需要它们的函数内部。

DEFAULT_IMAGE_KEY = "observation.images.base_0_rgb"


def assets_root() -> Path:
    """显式设置优先；兼容尚未迁移的项目，默认使用当前 models/pi05 布局。"""
    # 【调用方】record/trainer参数解析、policy_options；无输入，返回资源根Path，不创建目录。
    configured = os.environ.get("FRISTON_MODELS_ROOT", "").strip()
    # os.environ是环境变量映射；get缺键给空串，strip去首尾空白；空串在if中视为False。
    if configured:
        return Path(configured).expanduser().resolve()
        # Path把str变路径对象；expanduser展开~，resolve变成绝对路径。
    project = Path(__file__).resolve().parents[3]
    # __file__是当前模块文件名；parents[0]是models，[1]是arms，[2]是friston3，[3]是项目根。
    legacy = project / "Friston-3-Models"
    return legacy if legacy.is_dir() else project / "models" / "pi05"


def resolve_asset_path(path: str | Path) -> Path:
    """只修复本项目已知资产迁移留下的旧路径，不选择其他 checkpoint 或数据集。

    调用方：policy_options、训练/续训参数解析、采集续写。现存路径始终保留；
    只有旧项目 Friston-3-Models 下的失效路径，且新根内同相对目标存在时才映射。
    """
    # str|Path是类型联合注解，允许两种输入；->Path说明总返回Path，不自动检验类型。
    original = Path(path).expanduser().resolve()
    if original.exists():
        return original
    legacy = Path(__file__).resolve().parents[3] / "Friston-3-Models"
    try:
        relative = original.relative_to(legacy)
        # 相对化得到旧根之下的后半段，例如checkpoints/某次实验；不属于旧根会抛ValueError。
    except ValueError:
        return original
    relocated = assets_root() / relative
    # 把相同相对位置接到新根；只重定位同一资产，不搜索或选择另一个checkpoint。
    return relocated if relocated.exists() else original
