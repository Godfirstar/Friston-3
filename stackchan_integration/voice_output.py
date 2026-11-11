"""Persistent offline synthesis and bounded audio batching for StackChan.

Keeps the isolated Sherpa-ONNX TTS process alive, plays the first phrase as soon
as it is ready, and coalesces already available later phrases up to 12 seconds.
Uses face_events.py's single USB/TCP owner and the existing device protocol.
"""

from __future__ import annotations
# TTS是Text To Speech（文字→语音），与voice_input的ASR（语音→文字）方向相反。
# 主链：主程序/speak工具 → speak → _speak → 子进程tts_daemon → PCM文件 → face.play_pcm。
# 主进程负责组织和播音；独立Python环境中的子进程持有合成模型；设备连接由face_events持有。

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
        # 【创建方】main.py或speak.py；face是已存在的共享设备协调器，不另开串口。
        # self是当前语音输出实例；成员属性跨多次speak保留，所以模型进程可以常驻。
        self.face = face
        self.enabled = SETTINGS.get("speech_enabled", True)
        self.volume = SETTINGS.get("volume", 60)
        self._last_nonzero_volume = SETTINGS.get("last_nonzero_volume", 60)

        self._proc: Optional[subprocess.Popen] = None
        # Optional[Popen]表示子进程对象或None；初始None表示尚未启动合成进程。
        self._messages: queue.Queue = queue.Queue()
        # Queue是线程安全队列：reader线程put子进程事件，_speak线程get并处理。
        self._speech_lock = threading.Lock()
        # 保证同一输出器的一次完整播音事务串行，避免两轮文字/音频事件混在一起。
        self._sequence = 0
        self._closed = False
        self.last_speech_timing: dict[str, float] = {}
        atexit.register(self.close)
        # register传函数对象而不加()，登记解释器正常退出时的兜底清理回调。
        if self.enabled and self.face.enabled:
            try:
                self._launch_worker()
                print("[StackChanTTS] 后台预热离线语音引擎（只加载一次模型）。")
            except (OSError, ValueError, RuntimeError) as exc:
                print(f"[StackChanTTS] 后台引擎未启动，仍可正常文字聊天：{exc}")

    @staticmethod
    def _tts_python() -> Path:
        # 【调用方】_launch_worker；staticmethod无需self，只按当前解释器位置查可用TTS环境。
        # 返回python可执行文件Path；这里只定位已安装环境，不执行pip或下载模型。
        executable = Path(sys.executable).resolve()
        # sys.executable是当前Python路径；resolve得到绝对路径，不是用户shell的工作目录。
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
            # all要求每个依赖都可发现；find_spec查导入位置，区别于实际构造TTS引擎。
            return executable
        raise FileNotFoundError(
            "未找到可用离线语音环境；请安装一次 friston_tts 环境及 sherpa-onnx、numpy、scipy。"
            "程序会自动使用当前环境或相邻 Conda 语音环境。"
        )

    def _launch_worker(self) -> None:
        # 【调用方】构造、_speak、configure_output；副作用是创建/复用子进程与事件读取线程。
        if self._proc is not None and self._proc.poll() is None:
            # Popen.poll()在进程仍运行时返回None；已存在活进程就直接复用，不重复加载模型。
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
        # Popen来自标准库subprocess，参数list避免shell再次解释字符串。
        # -B不写pyc，-u关闭标准流缓冲，-m按模块执行tts_daemon；cwd使项目包可被导入。
        # stdin/stdout=PIPE建立双向管道；stderr丢弃，stdout主要接收JSON事件行。
        # interpreter_environment修正目标环境变量；CREATE_NO_WINDOW避免弹出额外控制台。
        proc = self._proc
        messages = self._messages
        # 捕获本代进程/队列引用，让reader闭包不会因后续重启而改读新进程的管道。

        def reader() -> None:
            # 【运行位置】后台friston-tts-output线程；只搬运事件，不执行设备播放。
            try:
                if proc.stdout is None:
                    raise RuntimeError("Worker stdout is not available")
                while True:
                    raw = proc.stdout.readline()
                    # 从子进程标准输出阻塞读到换行，返回bytes；空bytes意味着管道结束。
                    if not raw:
                        break
                    try:
                        obj = json.loads(raw.decode("utf-8", errors="strict"))
                        # bytes先UTF-8解码成str，再JSON解析为dict；一行对应一个事件(JSONL)。
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        # Native libraries may print diagnostics. Do not confuse
                        # those with protocol events, or block the producer pipe.
                        continue
                    if isinstance(obj, dict) and "event" in obj:
                        # Timestamp arrival, not later queue consumption while
                        # the main thread may be waiting for a clip to finish.
                        obj["received_at"] = time.perf_counter()
                        # 记录真正收到事件的时刻，避免主线程正在播音、晚取队列时把延迟算错。
                        if obj["event"] == "ready" and proc is self._proc and not self._closed:
                            print(
                                f"[StackChanTTS] 预热完成：加载 {obj.get('model_load_s')}s，"
                                f"预热 {obj.get('warmup_s', 0)}s；"
                                "Kokoro 中英 int8，男声 58。",
                                flush=True,
                            )
                        messages.put(obj)
                        # put将dict送入线程安全队列；不需另给这个队列加手写锁。
            except Exception as exc:
                messages.put({"event": "fatal", "error": str(exc)})
            finally:
                messages.put({"event": "worker_exited", "code": proc.poll()})

        threading.Thread(
            target=reader, daemon=True, name="friston-tts-output"
        ).start()
        # target=reader传函数；start才在新线程调用；daemon线程不单独阻止解释器退出。

    def speak(self, reply: str, *, on_start=None) -> bool:
        """Notify once before first PCM dispatch, or on text-only fallback.

        Optional presentation callbacks must not block audio. Callback failure
        is isolated from speech, and multi-clip replies cannot replay it.
        """
        # 【调用方】主程序回复播音/speak工具；reply为文字，返回是否完成有效播音的bool。
        # *后的on_start必须按名字传；通常是机械臂/展示联动回调，不负责生成语音。
        notified = False

        def start_once():
            nonlocal notified
            # 闭包保存本次speak的notified；nonlocal让赋值修改外层变量而非创建新的局部变量。
            if notified:
                return
            notified = True
            if callable(on_start):
                # callable判断传入值能否像函数一样调用；None时只记已通知，不执行动作。
                try:
                    on_start()
                except Exception as error:
                    print(f"[StackChanTTS] 附带动作未启动，播音继续：{error}")

        try:
            return self._speak(reply, on_start=start_once)
        finally:
            start_once()
            # 即使静音/失败导致未分发PCM，也通知一次文本回退；所以它不等于“扬声器已出声”。

    def _speak(self, reply: str, *, on_start) -> bool:
        """Speak NPC text, batching adjacent clips into <=12s PCM commands.

        The TTS daemon synthesizes chunks in a persistent subprocess and emits
        PCM file events. The first phrase plays immediately. Later clips may
        wait for a short prefetch window to join ready neighbors.
        """
        # 【调用方】speak包装器；发送一条JSON请求，按事件消费PCM并同步等设备播放。
        # 事件流可持续产生后续片段，故“当前片段播放”与“子进程合成下一片段”能够重叠。
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
            # with自动获取/释放锁；一次完整reply持有锁，使同一输出器不交叉播放两段回复。
            # A face update can discover a disconnect while the TTS worker is
            # still synthesizing. Late PCM from that reply must not become a
            # new playback request after the device has reconnected.
            playback_epoch = self.face.playback_epoch
            # 保存播放会话代次；断线重连后迟到的旧PCM不能当作新会话音频发出去。
            self.last_speech_timing = {}
            try:
                self._launch_worker()
                assert self._proc is not None and self._proc.stdin is not None
                self._sequence += 1
                request_id = self._sequence
                # 为本次文字生成单调递增id，返回事件必须匹配id才归入本轮。
                with tempfile.TemporaryDirectory(prefix="friston3_tts_") as output_dir:
                    # 每轮创建临时目录，as绑定路径str；退出with后自动清理该目录。
                    # PCM通过这里的文件传递，JSON管道只传事件/路径等元数据，不塞巨大音频数组。
                    msg = {
                        "cmd": "speak", "id": request_id,
                        "text": reply, "output_dir": output_dir,
                    }
                    self._proc.stdin.write(
                        (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
                    )
                    # dict→JSON str→加换行→UTF-8 bytes；子进程逐行读取stdin形成JSONL请求协议。
                    self._proc.stdin.flush()
                    # flush把已写数据交给管道，不让完整请求滞留在父进程缓冲中。

                    max_pcm_bytes = MAX_PCM_BYTES
                    prefetch = 0.08
                    segments_seen = 0
                    clips_played = 0
                    audio_secs = 0.0
                    total_hw_secs = 0.0
                    waiting_pcm: list[bytes] = []
                    # 等待合并的PCM片段list；每项是16kHz、单声道、小端16位的原始采样bytes。
                    waiting_bytes = 0
                    waiting_segments = 0
                    first_ready_at: Optional[float] = None
                    synth_done = False
                    expected_segments = -1
                    first_audio_s: Optional[float] = None
                    first_segment_s: Optional[float] = None
                    engine_wait_s = 0.0
                    playback_cancelled = False
                    # seen统计生产的片段，played统计已播放的合并音频；两者数量可以不同。
                    # synth_done表示合成端done事件已到，不代表设备已经把waiting_pcm播完。

                    def flush_audio() -> None:
                        # 内部闭包，使用当前请求的目录/id/epoch；把暂存片段合成一次设备播放。
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
                        # bytes.join按顺序拼接波形，保持采样时序；不是对波形求和混音。
                        seconds = pcm_seconds(len(combined))
                        # audio_protocol函数按字节数换算时长：bytes/(16000×2)。
                        print(
                            f"[StackChanTTS] 合并 {waiting_segments} 段 → "
                            f"播放第 {clips_played + 1} 条音频（{seconds:.2f}s）；"
                            "正在传输播放..."
                        )
                        began = time.perf_counter()
                        try:
                            on_start()
                            # 首次PCM分发之前调用；外层start_once保证多片段只触发一次。
                            self.face.play_pcm(combined, volume=self.volume,
                                               session_epoch=playback_epoch)
                            # play_pcm定义在face_events，通过共享传输发送并等待播音结果。
                            # 此调用阻塞时，独立tts_daemon仍可继续合成，reader仍可把事件入队。
                        except (ConnectionError, TimeoutError) as error:
                            # Keep the request directory until the producer
                            # acknowledges completion. It checks this marker
                            # between phrases, so no writer races directory
                            # removal and the warmed model stays resident.
                            playback_cancelled = True
                            (Path(output_dir) / "cancel").touch()
                            # touch创建取消标记文件；daemon在片段之间检查它，停止本轮后续合成。
                            # 先等daemon确认done再退出临时目录，避免写音频时目录已被父进程删除。
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
                            # 收到done后仍需播完已到达的缓存片段，再判定整轮完成。
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
                            # bool结果要求至少播放过一条，且本轮没有取消；不是“请求已提交”即成功。

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
                            # get阻塞等事件；有PCM待发时只等短预取窗口，否则允许等待引擎产生结果。
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
                            # 丢弃其他请求/旧进程留下的事件，防止迟到音频串到本轮。
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
                            # continue回到while顶部，先flush剩余PCM；不是立刻结束播音。
                        if kind != "segment":
                            continue

                        expected = Path(
                            output_dir, f"segment_{segments_seen:03d}.pcm"
                        ).resolve()
                        actual = Path(str(ev.get("path", ""))).resolve()
                        if actual != expected:
                            raise RuntimeError(f"Unexpected PCM segment path: {actual}")
                        raw_pcm = actual.read_bytes()
                        # pathlib读原始bytes；必须和segment事件宣称的路径/长度对应。
                        actual.unlink(missing_ok=True)
                        # 数据已读入内存后移除该临时片段；不是删除模型或用户录音。
                        if len(raw_pcm) != int(ev.get("bytes", -1)):
                            raise RuntimeError("PCM size does not match producer event")
                        if not raw_pcm or len(raw_pcm) > max_pcm_bytes or len(raw_pcm) % 2:
                            # PCM16每采样2字节；偶数字节数与单命令长度上限来自音频协议。
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
                        # append把完整片段加到待播队列；waiting_bytes另计总长度以决定能否合并。
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
        # 【调用方】close/播音异常；先发JSON shutdown，未及时退出再终止并关闭管道。
        # 只管理TTS子进程；共享StackChan连接由外层拥有，不在这里关闭。
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                try:
                    proc.stdin.write(b'{"cmd":"shutdown"}\n')
                    # b前缀是bytes字面量，结尾换行满足daemon逐行JSON协议。
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
        # 【调用方】音量/设备设置技能；先持久化设置，再同步本实例，返回新音量整数。
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
        # 恢复最近一次非零音量，而不是把已静音的0再次写回。
        return self.configure_output(enabled=True, volume=self._last_nonzero_volume)

    def close(self) -> None:
        # 【调用方】主程序清理/atexit兜底；可重复调用，_closed使第二次直接返回。
        with self._speech_lock:
            if self._closed:
                return
            self._closed = True
            self._stop_worker()
