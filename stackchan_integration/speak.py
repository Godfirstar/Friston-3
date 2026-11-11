"""Manual speech playback through the same automatic connection and TTS path.

Close main.py before running this utility so one process owns the device.
"""
from __future__ import annotations
# 独立手动朗读入口，复用主程序的TTS/设备路径；它不是另一套合成模型或另一套串口协议。
# 运行会真实连接设备并播放，因此与主程序共用设备时应遵循单一所有者的交接。

import argparse

from .face_events import StackChanFaceEvents
from .voice_output import AgentSpeechOutput


def main() -> None:
    # 【入口】python -m stackchan_integration.speak --text "要朗读的内容"；返回None/异常退出码。
    parser = argparse.ArgumentParser(description="通过自动连接的栈酱朗读一段文字")
    parser.add_argument("--text", required=True, help="需要朗读的中文或英文")
    # required要求该命名选项；有空格的文本用引号传成一个参数。
    args = parser.parse_args()
    # args为Namespace，以args.text读取；--help会在此打印并退出，不进入设备连接。
    voice = None
    try:
        with StackChanFaceEvents() as device:
            # 构造共享设备协调器；with调用其__enter__/__exit__，离开时释放设备资源。
            if not device.wait_ready(timeout=8.0):
                raise ConnectionError("未发现栈酱；连接设备热点或插入 USB 后重试。")
            device.on_request()
            # 通知设备进入请求处理展示状态；这不是TTS合成调用。
            try:
                voice = AgentSpeechOutput(device)
                # 把已经存在的device注入输出器；它会启动/复用独立TTS进程而不重开设备。
                if not voice.enabled or voice.volume == 0:
                    raise RuntimeError("当前已静音，请在主程序说‘恢复声音’或输入 /voice on。")
                if not voice.speak(args.text):
                    # speak在voice_output中：文字→子进程合成→PCM→设备播放，bool反映本轮结果。
                    raise RuntimeError("本次朗读未完成，请查看上方连接或语音提示。")
            finally:
                device.on_reply_ready()
                # 即使本轮播音失败也恢复展示状态；finally负责清理，不表示任务必然成功。
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"朗读未完成：{error}\n")
    finally:
        if voice is not None:
            voice.close()
            # 只在输出器成功创建后关闭它；停止TTS进程、关管道，不依赖垃圾回收时机。


if __name__ == "__main__":
    # import本模块时不会运行；-m运行时__name__为__main__，才执行入口。
    main()
