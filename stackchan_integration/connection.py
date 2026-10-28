"""Enumerate plausible StackChan USB devices without opening unrelated ports."""

from dataclasses import dataclass


WIFI_HOST = "192.168.4.1"
WIFI_PORT = 8765


@dataclass(frozen=True)
class UsbCandidate:
    port: str
    identity: tuple
    label: str


def usb_candidates() -> tuple[UsbCandidate, ...]:
    """USB metadata narrows discovery; the bridge must still verify INFO.

    Generic CH340/CP210x/FTDI adapters are deliberately not probed merely for
    being serial ports: those adapters can belong to the SO-101 motor buses.
    CoreS3 native USB normally identifies the Espressif vendor (0x303A).
    """
    try:
        from serial.tools import list_ports
    except ImportError:
        return ()
    try:
        ports = list_ports.comports()
    except OSError:
        # A transient OS enumeration error must not block the Wi-Fi path.
        # The owner will enumerate again at its next safe boundary.
        return ()
    found = []
    for item in ports:
        label = " ".join(str(getattr(item, field, "") or "") for field in
                         ("description", "manufacturer", "product"))
        text = label.casefold()
        branded = any(word in text for word in ("m5stack", "cores3", "core s3", "stackchan"))
        if getattr(item, "vid", None) != 0x303A and not branded:
            continue
        identity = (str(item.device), getattr(item, "vid", None), getattr(item, "pid", None),
                    getattr(item, "serial_number", None), getattr(item, "location", None))
        found.append(UsbCandidate(str(item.device), identity, label.strip()))
    return tuple(sorted(found, key=lambda item: item.port.casefold()))


def preference_name(value: str) -> str:
    value = str(value).strip().lower()
    value = {"serial": "usb", "tcp": "wifi"}.get(value, value)
    if value not in ("auto", "usb", "wifi"):
        raise ValueError("连接方式只能是 auto、usb 或 wifi")
    return value
