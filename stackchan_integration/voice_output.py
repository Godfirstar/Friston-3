"""Persistent offline synthesis and bounded audio batching for StackChan.

Keeps the isolated Sherpa-ONNX TTS process alive, plays the first phrase as soon
as it is ready, and coalesces already available later phrases up to 12 seconds.
Uses face_events.py's single USB/TCP owner and the existing device protocol.
"""

from __future__ import annotations

import atexit
import importlib.util
import json
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from .face_events import StackChanFaceEvents
from .audio_protocol import MAX_PCM_BYTES, pcm_seconds
from friston3.runtime_settings import SETTINGS
from friston3.runtime_bootstrap import interpreter_environment


class AgentSpeechOutput:
    """Voice output with asynchronous initialization and sequential playback."""

    def __init__(self, face: StackChanFaceEvents):
        self.face = face
        self.enabled = SETTINGS.get("speech_enabled", True)
        self.volume = SETTINGS.get("volume", 60)
        self._last_nonzero_volume = SETTINGS.get("last_nonzero_volume", 60)

        self._proc: Optional[subprocess.Popen] = None
        self._messages: queue.Queue = queue.Queue()
        self._speech_lock = threading.Lock()
        self._sequence = 0
        self._closed = False
        self.last_speech_timing: dict[str, float] = {}
        atexit.register(self.close)
        if self.enabled and self.face.enabled:
            try:
                self._launch_worker()
                print("[StackChanTTS] 后台预热离线语音引擎（只加载一次模型）。")
            except (OSError, ValueError, RuntimeError) as exc:
                print(f"[StackChanTTS] 后台引擎未启动，仍可正常文字聊天：{exc}")

    @staticmethod
    def _tts_python() -> Path:
        executable = Path(sys.executable).resolve()
        environment = executable.parent if executable.name.lower() == "python.exe" else executable.parent.parent
        for root in (environment.parent / "friston_tts",
                     Path(sys.base_prefix) / "envs" / "friston_tts"):
            candidate = root / ("python.exe" if executable.name.lower() == "python.exe" else "bin/python")
            packages = ([root / "Lib" / "site-packages"] if executable.name.lower() == "python.exe"
                        else list((root / "lib").glob("python*/site-packages")))
            if candidate.is_file() and any(all((path / module).is_dir()
                    for module in ("sherpa_onnx", "numpy", "scipy")) for path in packages):
                return candidate
        if all(importlib.util.find_spec(module) is not None for module in ("sherpa_onnx", "numpy", "scipy")):
            return executable
        raise FileNotFoundError(
            "未找到可用离线语音环境；请安装一次 friston_tts 环境及 sherpa-onnx、numpy、scipy。"
            "程序会自动使用当前环境或相邻 Conda 语音环境。"
        )

    def _launch_worker(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        python_exe = self._tts_python()
        self._messages = queue.Queue()
        self._proc = subprocess.Popen(
            [str(python_exe), "-B", "-u", "-m", "stackchan_integration.tts_daemon"],
            cwd=str(Path(__file__).resolve().parent.parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            env=interpreter_environment(python_exe),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        proc = self._proc
        messages = self._messages

        def reader() -> None:
            try:
                if proc.stdout is None:
                    raise RuntimeError("Worker stdout is not available")
                while True:
                    raw = proc.stdout.readline()
                    if not raw:
                        break
                    try:
                        obj = json.loads(raw.decode("utf-8", errors="strict"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        # Native libraries may print diagnostics. Do not confuse
                        # those with protocol events, or block the producer pipe.
                        continue
                    if isinstance(obj, dict) and "event" in obj:
                        # Timestamp arrival, not later queue consumption while
                        # the main thread may be waiting for a clip to finish.
                        obj["received_at"] = time.perf_counter()
                        if obj["event"] == "ready" and proc is self._proc and not self._closed:
                            print(
                                f"[StackChanTTS] 预热完成：加载 {obj.get('model_load_s')}s，"
                                f"预热 {obj.get('warmup_s', 0)}s；"
                                "Kokoro 中英 int8，男声 58。",
                                flush=True,
                            )
                        messages.put(obj)
            except Exception as exc:
                messages.put({"event": "fatal", "error": str(exc)})
            finally:
                messages.put({"event": "worker_exited", "code": proc.poll()})

        threading.Thread(
            target=reader, daemon=True, name="friston-tts-output"
        ).start()

    def speak(self, reply: str, *, on_start=None) -> bool:
        """Notify once before first PCM dispatch, or on text-only fallback.

        Optional presentation callbacks must not block audio. Callback failure
        is isolated from speech, and multi-clip replies cannot replay it.
        """
        notified = False

        def start_once():
            nonlocal notified
            if notified:
                return
            notified = True
            if callable(on_start):
                try:
                    on_start()
                except Exception as error:
                    print(f"[StackChanTTS] 附带动作未启动，播音继续：{error}")

        try:
            return self._speak(reply, on_start=start_once)
        finally:
            start_once()

    def _speak(self, reply: str, *, on_start) -> bool:
        """Speak NPC text, batching adjacent clips into <=12s PCM commands.

        The TTS daemon synthesizes chunks in a persistent subprocess and emits
        PCM file events. The first phrase plays immediately. Later clips may
        wait for a short prefetch window to join ready neighbors.
        """
        if not self.enabled or self.volume == 0 or not self.face.enabled or self._closed:
            return False
        reply = str(reply).strip()
        if not reply:
            return False
        if len(reply) > 500:
            print("[StackChanTTS] 回复超过 500 字，保留文字，不播音。")
            return False

        start = time.perf_counter()
        with self._speech_lock:
            # A face update can discover a disconnect while the TTS worker is
            # still synthesizing. Late PCM from that reply must not become a
            # new playback request after the device has reconnected.
            playback_epoch = self.face.playback_epoch
            self.last_speech_timing = {}
            try:
                self._launch_worker()
                assert self._proc is not None and self._proc.stdin is not None
                self._sequence += 1
                request_id = self._sequence
                with tempfile.TemporaryDirectory(prefix="friston3_tts_") as output_dir:
                    msg = {
                        "cmd": "speak", "id": request_id,
                        "text": reply, "output_dir": output_dir,
                    }
                    self._proc.stdin.write(
                        (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
                    )
                    self._proc.stdin.flush()

                    max_pcm_bytes = MAX_PCM_BYTES
                    prefetch = 0.08
                    segments_seen = 0
                    clips_played = 0
                    audio_secs = 0.0
                    total_hw_secs = 0.0
                    waiting_pcm: list[bytes] = []
                    waiting_bytes = 0
                    waiting_segments = 0
                    first_ready_at: Optional[float] = None
                    synth_done = False
                    expected_segments = -1
                    first_audio_s: Optional[float] = None
                    first_segment_s: Optional[float] = None
                    engine_wait_s = 0.0
                    playback_cancelled = False

                    def flush_audio() -> None:
                        nonlocal waiting_pcm, waiting_bytes, waiting_segments
                        nonlocal first_ready_at, clips_played, audio_secs, total_hw_secs
                        nonlocal first_audio_s, playback_cancelled
                        if not waiting_pcm:
                            return
                        if playback_cancelled:
                            waiting_pcm, waiting_bytes, waiting_segments = [], 0, 0
                            first_ready_at = None
                            return
                        combined = b"".join(waiting_pcm)
                        seconds = pcm_seconds(len(combined))
                        print(
                            f"[StackChanTTS] 合并 {waiting_segments} 段 → "
                            f"播放第 {clips_played + 1} 条音频（{seconds:.2f}s）；"
                            "正在传输播放..."
                        )
                        began = time.perf_counter()
                        try:
                            on_start()
                            self.face.play_pcm(combined, volume=self.volume,
                                               session_epoch=playback_epoch)
                        except (ConnectionError, TimeoutError) as error:
                            # Keep the request directory until the producer
                            # acknowledges completion. It checks this marker
                            # between phrases, so no writer races directory
                            # removal and the warmed model stays resident.
                            playback_cancelled = True
                            (Path(output_dir) / "cancel").touch()
                            waiting_pcm, waiting_bytes, waiting_segments = [], 0, 0
                            first_ready_at = None
                            print(f"[StackChanTTS] 本轮播音已取消，模型继续驻留：{error}")
                            return
                        elapsed = time.perf_counter() - began
                        if first_audio_s is None:
                            started_at = self.face.last_audio_timing.get("started_at")
                            if isinstance(started_at, (int, float)):
                                first_audio_s = started_at - start
                                print(f"[StackChanTTS] 文本就绪到设备开始播放 {first_audio_s:.2f}s。")
                        print(
                            f"[StackChanTTS] 第 {clips_played + 1} 条音频完成："
                            f"设备传输+播放 {elapsed:.2f}s，其中音频 {seconds:.2f}s。"
                        )
                        clips_played += 1
                        audio_secs += seconds
                        total_hw_secs += elapsed
                        waiting_pcm = []
                        waiting_bytes = 0
                        waiting_segments = 0
                        first_ready_at = None

                    while True:
                        if synth_done:
                            flush_audio()
                            if segments_seen != expected_segments:
                                raise RuntimeError(
                                    f"TTS segment count mismatch: "
                                    f"{segments_seen} != {expected_segments}"
                                )
                            state = "本轮播音取消，已处理" if playback_cancelled else "完成"
                            print(
                                f"[StackChanTTS] {state} {segments_seen} 段合成为 "
                                f"{clips_played} 条音频：实际音频 {audio_secs:.2f}s，"
                                f"设备传输+播放 {total_hw_secs:.2f}s，"
                                f"本轮总耗时 {time.perf_counter()-start:.2f}s。"
                            )
                            self.last_speech_timing = {
                                "text_to_audio_s": first_audio_s if first_audio_s is not None else -1.0,
                                "audio_s": audio_secs,
                                "total_s": time.perf_counter() - start,
                                "segments": segments_seen,
                                "clips": clips_played,
                                "first_segment_s": first_segment_s if first_segment_s is not None else -1.0,
                                "engine_wait_s": engine_wait_s,
                                "cancelled": playback_cancelled,
                            }
                            return clips_played > 0 and not playback_cancelled

                        if waiting_pcm and first_ready_at is not None:
                            remaining = prefetch - (time.perf_counter() - first_ready_at)
                            if remaining <= 0:
                                flush_audio()
                                continue
                            wait_for = max(0.01, remaining)
                        else:
                            wait_for = 90.0
                        try:
                            ev = self._messages.get(timeout=wait_for)
                        except queue.Empty as exc:
                            if waiting_pcm:
                                flush_audio()
                                continue
                            raise TimeoutError(
                                "TTS worker 90 秒未输出；已保留文字回复"
                            ) from exc

                        kind = ev.get("event")
                        if kind == "ready":
                            # The reader announces actual readiness even before
                            # the first conversation. Do not delay it until here.
                            continue
                        if kind in ("fatal", "worker_exited"):
                            raise RuntimeError(f"TTS worker exited: {ev}")
                        if ev.get("id") != request_id:
                            continue
                        if kind == "started":
                            engine_wait_s = max(0.0, float(ev["received_at"]) - start)
                            if engine_wait_s >= 0.1:
                                print(f"[StackChanTTS] 本轮等待引擎/队列 {engine_wait_s:.2f}s。")
                            continue
                        if kind == "error":
                            raise RuntimeError(str(ev.get("error", "TTS worker error")))
                        if kind == "done":
                            synth_done = True
                            expected_segments = int(ev.get("segments", -1))
                            continue
                        if kind != "segment":
                            continue

                        expected = Path(
                            output_dir, f"segment_{segments_seen:03d}.pcm"
                        ).resolve()
                        actual = Path(str(ev.get("path", ""))).resolve()
                        if actual != expected:
                            raise RuntimeError(f"Unexpected PCM segment path: {actual}")
                        raw_pcm = actual.read_bytes()
                        actual.unlink(missing_ok=True)
                        if len(raw_pcm) != int(ev.get("bytes", -1)):
                            raise RuntimeError("PCM size does not match producer event")
                        if not raw_pcm or len(raw_pcm) > max_pcm_bytes or len(raw_pcm) % 2:
                            raise RuntimeError("Invalid 16kHz PCM16 segment")

                        seconds = pcm_seconds(len(raw_pcm))
                        segments_seen += 1
                        ready_s = max(0.0, float(ev.get("received_at", time.perf_counter())) - start)
                        if first_segment_s is None:
                            first_segment_s = ready_s
                        print(
                            f"[StackChanTTS] 第 {segments_seen} 段已合成："
                            f"音频 {seconds:.2f}s，合成 {ev.get('synth_s', '?')}s "
                            f"（推理 {ev.get('inference_s', '?')}s，"
                            f"音频处理 {ev.get('processing_s', '?')}s），"
                            f"收到文本后 {ready_s:.2f}s 就绪。"
                        )
                        if waiting_pcm and waiting_bytes + len(raw_pcm) > max_pcm_bytes:
                            # A single command can play at most 12 seconds.
                            flush_audio()
                        if not waiting_pcm:
                            first_ready_at = time.perf_counter()
                        waiting_pcm.append(raw_pcm)
                        waiting_bytes += len(raw_pcm)
                        waiting_segments += 1
                        if clips_played == 0 or waiting_bytes == max_pcm_bytes:
                            # Never wait to pack the first phrase. The daemon
                            # continues synthesizing while this clip plays.
                            flush_audio()
            except (OSError, ValueError, RuntimeError, TimeoutError,
                    subprocess.SubprocessError, AssertionError) as exc:
                print(f"[StackChanTTS] 语音暂不可用，仍保留文字回复：{exc}")
                self._stop_worker()
                return False

    def _stop_worker(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                try:
                    proc.stdin.write(b'{"cmd":"shutdown"}\n')
                    proc.stdin.flush()
                except (OSError, ValueError):
                    pass
                try:
                    proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            if proc.stdin is not None:
                proc.stdin.close()
            if proc.stdout is not None:
                proc.stdout.close()

    def configure_output(self, *, enabled: bool, volume: int) -> int:
        """Commit one complete audible state before changing its in-memory copy."""
        if type(enabled) is not bool:
            raise ValueError("播音开关必须是布尔值")
        if type(volume) is not int or not 0 <= volume <= 160:
            raise ValueError("Speaker volume must be an integer in 0..160")
        remembered = volume or self._last_nonzero_volume
        SETTINGS.update(speech_enabled=enabled, volume=volume, last_nonzero_volume=remembered)
        self.enabled = enabled
        self.volume = volume
        self._last_nonzero_volume = remembered
        if self.enabled and self.face.enabled and not self._closed:
            try:
                self._launch_worker()
            except (OSError, ValueError, RuntimeError) as exc:
                print(f"[StackChanTTS] 语音预热未启动：{exc}")
        return self.volume

    def unmute(self) -> int:
        return self.configure_output(enabled=True, volume=self._last_nonzero_volume)

    def close(self) -> None:
        with self._speech_lock:
            if self._closed:
                return
            self._closed = True
            self._stop_worker()
