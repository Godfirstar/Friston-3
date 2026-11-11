"""Friston-3 StackChan integration, with hardware imports loaded on demand.

The offline voice installer must work without importing a serial driver, and
running a submodule with ``python -m`` must not preload that same submodule.
"""

__all__ = ["StackChanBridge", "StackChanFaceEvents"]
# __all__ 声明包对外提供的名称；字符串不是对象，单凭这一行不会打开连接或导入驱动。


def __getattr__(name):
    # Python 在访问包内尚不存在的属性时调用它。例如 from stackchan_integration
    # import StackChanFaceEvents 会按需进入对应分支，返回真正的类，而不是创建设备实例。
    if name == "StackChanBridge":
        from .stackchan_bridge import StackChanBridge
        # 开头的点表示从当前包的同级文件导入；类被返回后，调用者才决定何时构造对象。
        return StackChanBridge
    if name == "StackChanFaceEvents":
        from .face_events import StackChanFaceEvents
        return StackChanFaceEvents
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # 未支持的名称用标准 AttributeError 报告，保留正常的 Python 属性访问语义。
