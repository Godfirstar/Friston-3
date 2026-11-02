"""LeRobot training and policy support without eager model or hardware imports."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from threading import RLock, get_ident

DEFAULT_IMAGE_KEY = "observation.images.base_0_rgb"


def assets_root() -> Path:
    configured = os.environ.get("FRISTON_MODELS_ROOT", "").strip()
    return Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[3] / "Friston-3-Models"


_checkpoint_reader_lock = RLock()


@contextmanager
def checkpoint_reader():
    """Use bounded CPU file reads without changing the fingerprinted loader.

    Native safe_open can map the whole checkpoint even with backend='pread'.
    Scope the replacement to the loading thread; other frameworks, devices,
    and explicitly requested mmap calls keep their native implementation.
    """
    import safetensors
    from .weights import BoundedSafetensors

    with _checkpoint_reader_lock:
        original = safetensors.safe_open
        owner = get_ident()
        active = True
        unspecified = object()

        def scoped_open(filename, framework, device="cpu", *, backend=unspecified):
            if (active and get_ident() == owner and framework == "pt" and device == "cpu"
                    and (backend is unspecified or backend == "pread")):
                return BoundedSafetensors(filename)
            if backend is unspecified:
                return original(filename, framework, device)
            return original(filename, framework, device, backend=backend)

        safetensors.safe_open = scoped_open
        try:
            yield
        finally:
            active = False
            safetensors.safe_open = original
