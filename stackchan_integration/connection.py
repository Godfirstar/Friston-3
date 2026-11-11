"""Enumerate plausible StackChan USB devices without opening unrelated ports."""

from dataclasses import dataclass
# dataclass 来自 Python 标准库，自动生成保存字段的 __init__、比较与显示方法。


WIFI_HOST = "192.168.4.1"
WIFI_PORT = 8765
# 供 StackChanBridge 使用的默认离线 Wi-Fi 端点；这些常量本身不发起连接。


@dataclass(frozen=True)
class UsbCandidate:
    # frozen=True 表示创建后不可直接给字段重新赋值；这是枚举结果，不是串口对象。
    port: str
    identity: tuple
    label: str
# port 为 COM 名称；identity 汇总设备身份元数据；label 是用于辨认的描述文字。


def usb_candidates() -> tuple[UsbCandidate, ...]:
    """USB metadata narrows discovery; the bridge must still verify INFO.

    Generic CH340/CP210x/FTDI adapters are deliberately not probed merely for
    being serial ports: those adapters can belong to the SO-101 motor buses.
    CoreS3 native USB normally identifies the Espressif vendor (0x303A).
    """
    try:
        from serial.tools import list_ports
        # serial 是 pyserial 的模块名；list_ports.comports 仅枚举系统元数据，不打开端口。
    except ImportError:
        return ()
    try:
        ports = list_ports.comports()
    except OSError:
        # A transient OS enumeration error must not block the Wi-Fi path.
        # The owner will enumerate again at its next safe boundary.
        return ()
    found = []
    # found 是局部 list；后面逐个加入 UsbCandidate，最终转为不可变 tuple 返回给 Bridge。
    for item in ports:
        label = " ".join(str(getattr(item, field, "") or "") for field in
                         ("description", "manufacturer", "product"))
        # getattr 按字段名字符串读取属性；缺失时用空串。join 把三个描述用空格连起来。
        text = label.casefold()
        branded = any(word in text for word in ("m5stack", "cores3", "core s3", "stackchan"))
        # any 在任意一项为真时为真；casefold 用于不区分大小写地比较设备描述。
        if getattr(item, "vid", None) != 0x303A and not branded:
            continue
        identity = (str(item.device), getattr(item, "vid", None), getattr(item, "pid", None),
                    getattr(item, "serial_number", None), getattr(item, "location", None))
        # VID/PID、序列号与位置共同标识一次枚举对象；不是通过发命令读取机械臂串口。
        found.append(UsbCandidate(str(item.device), identity, label.strip()))
    return tuple(sorted(found, key=lambda item: item.port.casefold()))
    # sorted 返回按 COM 名排序的列表；lambda 是“给每项求排序键”的匿名函数。


def preference_name(value: str) -> str:
    # Bridge 与 FaceEvents 共用：把用户/配置中的连接方式标准化，返回 auto/usb/wifi。
    value = str(value).strip().lower()
    value = {"serial": "usb", "tcp": "wifi"}.get(value, value)
    # dict.get(key, fallback)：别名命中则替换，未命中则沿用原字符串。
    if value not in ("auto", "usb", "wifi"):
        raise ValueError("连接方式只能是 auto、usb 或 wifi")
    return value
