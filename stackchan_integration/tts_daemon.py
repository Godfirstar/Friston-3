"""Persistent offline TTS producer for Friston-3.

Launched once in the existing friston_tts environment by voice_output.py.
Receives JSONL requests over stdin and streams SEGMENT events over stdout.
Only generates PCM files; never opens a device connection or loads the Agent.

This is a runtime worker, NOT a test script.
"""

from __future__ import annotations
# 运行于voice_output启动的独立Python子进程；stdin接JSONL请求，stdout回JSONL事件。
# TTS模型在子进程中常驻；这里只合成临时PCM文件，不开StackChan连接、不执行Agent推理。

import json
import sys
import time
from pathlib import Path

from .local_tts import OfflineCharacterVoice
from .audio_protocol import MAX_PCM_BYTES, SAMPLE_WIDTH
from .tts_export import split_text, synthesize_part


def emit(obj: dict) -> None:
    # 【调用方】serve；dict→一行JSON→标准输出管道，供父进程reader线程解析。
    sys.stdout.write(json.dumps(obj, ensure_ascii=True, separators=(",", ":")) + "\n")
    # ensure_ascii把中文编码为JSON转义，separators移除额外空格；换行是消息边界。
    sys.stdout.flush()
    # 每个事件立即推出，不能等缓冲区积满才让父进程看到首段音频。


def serve() -> None:
    # 【入口】python -m stackchan_integration.tts_daemon；返回None，持续处理多次请求。
    # 下游：OfflineCharacterVoice加载/合成，tts_export分句，Path写PCM，emit反馈状态。
    start = time.perf_counter()
    try:
        voice = OfflineCharacterVoice()
        # 创建一次引擎并加载模型；所有后续speak请求共用这个voice实例。
        model_load_s = time.perf_counter() - start
        warmup_s = voice.warmup()
        # warmup是真实的短文本合成预热，不播放；与单纯“权重已加载”分别计时。
    except Exception as exc:
        emit({"event": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        return
    emit({"event": "ready", "model_load_s": round(model_load_s, 3),
          "warmup_s": round(warmup_s, 3), "warmed_up": True,
          "backend": voice.backend, "speaker": voice.speaker, "style": voice.style})

    for raw in sys.stdin.buffer:
        # buffer是二进制标准输入；for按换行逐条读取bytes，父进程关闭管道时循环结束。
        if not raw.strip():
            continue
        request = None
        try:
            request = json.loads(raw.decode("utf-8", errors="strict"))
            # 协议步骤：UTF-8 bytes→str→Python dict；错误编码/JSON进入本轮异常处理。
            if request.get("cmd") == "shutdown":
                # 正常退出命令来自父进程_stop_worker；break结束请求循环并退出子进程。
                break
            if request.get("cmd") != "speak":
                emit({"event": "error", "id": request.get("id"), "error": "Unknown command"})
                continue

            request_id = int(request["id"])
            # 请求id由父进程递增分配，所有started/segment/done/error原样带回以匹配轮次。
            request_started = time.perf_counter()
            emit({"event": "started", "id": request_id})
            source = request["text"]
            if not isinstance(source, str) or not 0 < len(source.strip()) <= 500:
                raise ValueError("Reply must contain between 1 and 500 characters")
            out_dir = Path(request["output_dir"])
            # 父进程为本轮创建的临时目录；JSON只传路径字符串，Path负责后续文件操作。
            out_dir.mkdir(parents=True, exist_ok=True)
            # Normalize names using the full language context before chunking.
            source = voice.prepare_text(source)
            # 先用完整上下文规范名字发音，再分句；不改变主程序保存的原始回复。
            parts = split_text(source, limit=32, first_limit=20)
            # split_text定义在tts_export；首段更短以减少首次开口等待，后段可以稍长。
            if not parts:
                raise ValueError("No text after normalization")

            total = 0
            synthesized_s = 0.0
            cancelled = False
            for part in parts:
                # parts是str列表；每轮生成一个短句，首段完成即通知父进程，不等全文合成完。
                if (out_dir / "cancel").exists():
                    # 父进程播音失败后写标记；两段之间检查，保留模型进程供下一轮使用。
                    cancelled = True
                    break
                started = time.perf_counter()
                segments = synthesize_part(voice, part)
                # 返回list[bytes]；通常一段，极长音频会在安静位置分割成多段。
                duration = time.perf_counter() - started
                synthesized_s += duration
                if (out_dir / "cancel").exists():
                    cancelled = True
                    break
                for chunk in segments:
                    # chunk是单声道PCM16 bytes；SAMPLE_WIDTH=2，不能出现半个采样。
                    if not chunk or len(chunk) % SAMPLE_WIDTH or len(chunk) > MAX_PCM_BYTES:
                        raise ValueError("Unexpected PCM buffer size")
                    filename = out_dir / f"segment_{total:03d}.pcm"
                    # f字符串:03d将序号补到3位，例如segment_000.pcm，与父进程预期命名一致。
                    filename.write_bytes(chunk)
                    # 先完整写文件，再发送segment事件；父进程收到事件后才会读取该文件。
                    emit({
                        "event": "segment",
                        "id": request_id,
                        "seq": total,
                        "bytes": len(chunk),
                        "path": str(filename),
                        # 路径转str才能JSON序列化；实际PCM不编码成JSON数组，以免额外复制膨胀。
                        "synth_s": round(duration, 3),
                        "inference_s": round(voice.last_synthesis_timing.get("inference_s", 0.0), 3),
                        "processing_s": round(voice.last_synthesis_timing.get("processing_s", 0.0), 3),
                        "request_elapsed_s": round(time.perf_counter() - request_started, 3),
                    })
                    total += 1
                    # 累加真实PCM片段数，不一定等于短句parts数量。
            emit({
                "event": "done", "id": request_id, "segments": total,
                "synth_total_s": round(synthesized_s, 3),
                "cancelled": cancelled,
            })
            # done只表示生产端处理完；父进程还会播完队列中的音频才返回speak成功。
        except Exception as exc:
            # 本地循环只回传error，不主动重建模型；但voice_output父进程收到error后会停止worker，
            # 后续speak需要时会重新启动它，因此不能据此认定整个系统出错后仍保持模型驻留。
            emit({"event": "error", "id": request.get("id") if isinstance(request, dict) else None,
                  "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    # 直接按模块运行时启动协议循环；被导入时只提供函数定义。
    serve()
