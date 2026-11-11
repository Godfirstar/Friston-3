"""Small, persistent user settings shared by speech and the device owner.

The production pipeline has one fixed set of models. This store contains only
choices a person can make during use; model/backend/debug environment switches
are deliberately not configuration inputs.
"""
from __future__ import annotations

# 标准库json负责配置编码，pathlib负责路径，tempfile创建临时文件，threading提供进程内线程锁。
# future annotations延迟类型标注求值；它不意味着Path|None会在运行时自动校验实参。
import json
import os
from pathlib import Path
import tempfile
import threading

from friston3.config import PROJECT_ROOT


# DEFAULTS是运行设置默认dict；_CHOICES是部分字段允许值的set，不包含模型后端或调试开关。
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
    # RuntimeSettings.__init__/update共用；输入设置名str及值，通过返回None，失败raise ValueError。
    # in字典检查键存在；type(x)is...要求精确类型，避免True作为int=1进入音量字段。
    if key not in DEFAULTS:
        raise ValueError(f"未知运行设置：{key}")
    if key in _CHOICES:
        if type(value) is not str or value not in _CHOICES[key]:
            raise ValueError(f"{key} 的设置值无效")
    elif key in ("speech_enabled", "gestures_enabled"):
        if type(value) is not bool:
            raise ValueError(f"{key} 必须为开或关")
    else:
        # minimum为当前字段的下界int；非零音量用于从静音恢复，不能保存0为恢复目标。
        minimum = 1 if key == "last_nonzero_volume" else 0
        if type(value) is not int or not minimum <= value <= 160:
            raise ValueError(f"{key} 超出允许范围")


class RuntimeSettings:
    def __init__(self, path: Path | None = None):
        # 文件末尾创建SETTINGS单例时调用；path类型标注表示Path或None，省略则读项目运行配置。
        # Class(...)调用__init__，self为当前实例；这里只读已有配置，不在启动时写默认值覆盖文件。
        self.path = Path(path) if path is not None else PROJECT_ROOT / "data" / "runtime" / "settings.json"
        # RLock为标准库可重入锁：同线程可重复取得；它协调本实例多个线程，不是跨进程文件锁。
        self._lock = threading.RLock()
        # dict(DEFAULTS)复制外层字典，后面修改实例值不会改模块默认值；当前值均为简单标量。
        self._values = dict(DEFAULTS)
        self._read_error = ""
        try:
            # Path.read_text返回str；utf-8-sig兼容带BOM文件；json.loads再解码为Python对象raw。
            raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
            if not isinstance(raw, dict) or raw.get("schema_version") != 1:
                raise ValueError("设置文件版本或结构无效")
            # values为配置中settings子dict；全部条目先校验，之后才统一覆盖实例默认值。
            values = raw.get("settings")
            if not isinstance(values, dict):
                raise ValueError("缺少 settings 对象")
            for key, value in values.items():
                _validate(key, value)
            self._values.update(values)
        # 初次没有配置可正常使用默认值；其他读取/格式错误记录到_read_error，update不会盖掉坏文件。
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError) as error:
            self._read_error = str(error)
            print(f"[设置] 无法读取 {self.path.name}，本次使用默认设置：{error}")

    def get(self, key, default=None):
        # 语音输入/输出、设备表演、DeviceSettingsSkill等调用；返回指定标量，缺键则给default。
        # with取得锁，离开块（包括return/异常）自动释放，防止读取恰好与同实例更新交错。
        with self._lock:
            return self._values.get(key, default)

    def snapshot(self):
        # 返回整个设置的外层副本dict供状态展示；调用方修改它不会直接修改存储中的_values。
        with self._lock:
            return dict(self._values)

    def update(self, **changes):
        # DeviceSettingsSkill、语音/连接管理调用，例如update(mic_mode='touch')。
        # 定义处**changes把命名实参收成dict；这与调用处**字典展开实参正好相反。
        # 先检查所有字段，再持锁完成读旧值->写文件->提交内存值；返回提交后的dict副本。
        for key, value in changes.items():
            _validate(key, value)
        with self._lock:
            if self._read_error:
                raise RuntimeError(f"设置文件读取失败，未覆盖原文件：{self.path}；{self._read_error}")
            # 两个**在字典字面量里展开字段，后面的changes覆盖同名值；旧_values此刻仍未修改。
            proposed = {**self._values, **changes}
            if proposed == self._values and self.path.is_file():
                return dict(self._values)
            # 相同内容且已有文件可跳过写盘；否则mkdir建立配置目录，现有目录不报错。
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # temporary先为None；只有临时文件成功创建后才变成Path，finally据此清理。
            temporary = None
            try:
                # 在目标同目录创建独立临时文件；delete=False让with关闭句柄后文件仍存在以便替换。
                # Windows替换前必须先离开with关闭句柄，不能边保持打开边将其当最终文件使用。
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                        prefix=".settings_", suffix=".tmp", dir=self.path.parent, delete=False) as handle:
                    # handle是文本文件对象，name为实际路径str；Path包装后用于replace/unlink。
                    temporary = Path(handle.name)
                    # json.dump直接写文件，区别于返回字符串的dumps；indent=2使文件便于人工阅读。
                    json.dump({"schema_version": 1, "settings": proposed}, handle,
                              ensure_ascii=False, indent=2)
                    handle.write("\n")
                    # flush把Python缓冲交给系统；fsync请求把该文件描述符的数据同步到底层存储。
                    # 这两个步骤与后面的原子替换职责不同，不能把“写了一半临时文件”暴露为正式配置。
                    handle.flush()
                    os.fsync(handle.fileno())
                # 同目录原子替换：读者看到旧文件或完整新文件，避免直接覆盖写时看到半份JSON。
                # 这里仅说明本地文件替换行为，不把RLock误当成多进程并发更新协调机制。
                os.replace(temporary, self.path)
            finally:
                if temporary is not None:
                    # replace成功后临时路径已不存在；missing_ok=True允许这一情况，也清理失败残留。
                    temporary.unlink(missing_ok=True)
            # 只有文件替换成功后才更新内存；异常会跳过这两句，让调用者知道更新没有提交。
            self._values = proposed
            return dict(proposed)


# 模块级单例：第一次import本模块就会构造并读取设置文件；不要为语法检查而import运行业务模块。
SETTINGS = RuntimeSettings()
