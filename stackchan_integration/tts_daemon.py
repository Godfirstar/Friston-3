"""Persistent offline TTS producer for Friston-3.

Launched once in the existing friston_tts environment by voice_output.py.
Receives JSONL requests over stdin and streams SEGMENT events over stdout.
Only generates PCM files; never opens a device connection or loads the Agent.

This is a runtime worker, NOT a test script.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .local_tts import OfflineCharacterVoice
from .audio_protocol import MAX_PCM_BYTES, SAMPLE_WIDTH
from .tts_export import split_text, synthesize_part


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def serve() -> None:
    start = time.perf_counter()
    try:
        voice = OfflineCharacterVoice()
        model_load_s = time.perf_counter() - start
        warmup_s = voice.warmup()
    except Exception as exc:
        emit({"event": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        return
    emit({"event": "ready", "model_load_s": round(model_load_s, 3),
          "warmup_s": round(warmup_s, 3), "warmed_up": True,
          "backend": voice.backend, "speaker": voice.speaker, "style": voice.style})

    for raw in sys.stdin.buffer:
        if not raw.strip():
            continue
        request = None
        try:
            request = json.loads(raw.decode("utf-8", errors="strict"))
            if request.get("cmd") == "shutdown":
                break
            if request.get("cmd") != "speak":
                emit({"event": "error", "id": request.get("id"), "error": "Unknown command"})
                continue

            request_id = int(request["id"])
            request_started = time.perf_counter()
            emit({"event": "started", "id": request_id})
            source = request["text"]
            if not isinstance(source, str) or not 0 < len(source.strip()) <= 500:
                raise ValueError("Reply must contain between 1 and 500 characters")
            out_dir = Path(request["output_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            # Normalize names using the full language context before chunking.
            source = voice.prepare_text(source)
            parts = split_text(source, limit=32, first_limit=20)
            if not parts:
                raise ValueError("No text after normalization")

            total = 0
            synthesized_s = 0.0
            cancelled = False
            for part in parts:
                if (out_dir / "cancel").exists():
                    cancelled = True
                    break
                started = time.perf_counter()
                segments = synthesize_part(voice, part)
                duration = time.perf_counter() - started
                synthesized_s += duration
                if (out_dir / "cancel").exists():
                    cancelled = True
                    break
                for chunk in segments:
                    if not chunk or len(chunk) % SAMPLE_WIDTH or len(chunk) > MAX_PCM_BYTES:
                        raise ValueError("Unexpected PCM buffer size")
                    filename = out_dir / f"segment_{total:03d}.pcm"
                    filename.write_bytes(chunk)
                    emit({
                        "event": "segment",
                        "id": request_id,
                        "seq": total,
                        "bytes": len(chunk),
                        "path": str(filename),
                        "synth_s": round(duration, 3),
                        "inference_s": round(voice.last_synthesis_timing.get("inference_s", 0.0), 3),
                        "processing_s": round(voice.last_synthesis_timing.get("processing_s", 0.0), 3),
                        "request_elapsed_s": round(time.perf_counter() - request_started, 3),
                    })
                    total += 1
            emit({
                "event": "done", "id": request_id, "segments": total,
                "synth_total_s": round(synthesized_s, 3),
                "cancelled": cancelled,
            })
        except Exception as exc:
            # Keep engine loaded if the current request fails.
            emit({"event": "error", "id": request.get("id") if isinstance(request, dict) else None,
                  "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    serve()
