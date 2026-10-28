"""Small, persistent user settings shared by speech and the device owner.

The production pipeline has one fixed set of models. This store contains only
choices a person can make during use; model/backend/debug environment switches
are deliberately not configuration inputs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading

from friston3.config import PROJECT_ROOT


DEFAULTS = {
    "speech_enabled": True,
    "volume": 60,
    "last_nonzero_volume": 60,
    "connection_preference": "auto",
    "mic_mode": "touch",
    "gestures_enabled": True,
    "owner_side": "left",
    "placement": "shoulder",
}
_CHOICES = {
    "connection_preference": {"auto", "usb", "wifi"},
    "mic_mode": {"touch", "vad"},
    "owner_side": {"none", "left", "right"},
    "placement": {"shoulder", "desk"},
}


def _validate(key, value):
    if key not in DEFAULTS:
        raise ValueError(f"未知运行设置：{key}")
    if key in _CHOICES:
        if type(value) is not str or value not in _CHOICES[key]:
            raise ValueError(f"{key} 的设置值无效")
    elif key in ("speech_enabled", "gestures_enabled"):
        if type(value) is not bool:
            raise ValueError(f"{key} 必须为开或关")
    else:
        minimum = 1 if key == "last_nonzero_volume" else 0
        if type(value) is not int or not minimum <= value <= 160:
            raise ValueError(f"{key} 超出允许范围")


class RuntimeSettings:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else PROJECT_ROOT / "data" / "runtime" / "settings.json"
        self._lock = threading.RLock()
        self._values = dict(DEFAULTS)
        self._read_error = ""
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
            if not isinstance(raw, dict) or raw.get("schema_version") != 1:
                raise ValueError("设置文件版本或结构无效")
            values = raw.get("settings")
            if not isinstance(values, dict):
                raise ValueError("缺少 settings 对象")
            for key, value in values.items():
                _validate(key, value)
            self._values.update(values)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError) as error:
            self._read_error = str(error)
            print(f"[设置] 无法读取 {self.path.name}，本次使用默认设置：{error}")

    def get(self, key, default=None):
        with self._lock:
            return self._values.get(key, default)

    def snapshot(self):
        with self._lock:
            return dict(self._values)

    def update(self, **changes):
        for key, value in changes.items():
            _validate(key, value)
        with self._lock:
            if self._read_error:
                raise RuntimeError(f"设置文件读取失败，未覆盖原文件：{self.path}；{self._read_error}")
            proposed = {**self._values, **changes}
            if proposed == self._values and self.path.is_file():
                return dict(self._values)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                        prefix=".settings_", suffix=".tmp", dir=self.path.parent, delete=False) as handle:
                    temporary = Path(handle.name)
                    json.dump({"schema_version": 1, "settings": proposed}, handle,
                              ensure_ascii=False, indent=2)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            self._values = proposed
            return dict(proposed)


SETTINGS = RuntimeSettings()
