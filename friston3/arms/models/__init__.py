"""本地 π₀.₅ 基座、tokenizer 和微调产物的位置。"""
import os
from pathlib import Path


_MODELS = Path(__file__).resolve().parents[3] / "models"


def assets_root() -> Path:
    configured = os.environ.get("FRISTON_MODELS_ROOT", "").strip()
    return Path(configured).expanduser().resolve() if configured else _MODELS / "pi05"


def fast_tokenizer_root() -> Path:
    if os.environ.get("FRISTON_MODELS_ROOT", "").strip():
        return assets_root() / "fast"
    return _MODELS / "fast"
