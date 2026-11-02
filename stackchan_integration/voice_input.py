"""One offline SeACo recognizer on the shared, reconnecting device transport.

The firmware captures PCM and detects utterance endings. The recognizer stays
loaded across USB/Wi-Fi changes. Recordings are kept only for the current turn.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

from friston3.runtime_settings import SETTINGS

if TYPE_CHECKING:
    from .face_events import StackChanFaceEvents


SEACO_REQUIRED = ("model_quant.onnx", "model_eb_quant.onnx", "config.yaml", "am.mvn", "tokens.json")


def seaco_missing(folder: Path) -> list[str]:
    return [name for name in SEACO_REQUIRED
            if not (folder / name).is_file() or (folder / name).stat().st_size == 0]


def seaco_model_dir() -> Path:
    models = Path(__file__).resolve().parent.parent / "models"
    for folder in (models / "seaco_paraformer", models / "seaco"):
        if not seaco_missing(folder):
            return folder
    return models / "seaco_paraformer"


class StackChanSpeechInput:
    """Read one nonempty transcript without opening another serial connection.

    ``touch`` waits for a touch on the StackChan display before recording;
    ``vad`` listens for an energy threshold while this method is active.
    The main loop should call ``read`` only after its previous response and
    speaker playback finish. This implements half-duplex interaction.
    """

    SAMPLE_RATE = 16000

    def __init__(self, face: StackChanFaceEvents):
        self.face = face
        self.mode = SETTINGS.get("mic_mode", "touch")
        self.max_ms, self.silence_ms, self.threshold = 8000, 700, 100
        self.num_threads = 2
        self.model_dir = seaco_model_dir()
        self._device_epoch = None
        self._recognizer = None
        self._np = None
        self._hotword_text = ""
        self.hotword_count = 0
        self._load_lock = threading.Lock()
        self._closed = threading.Event()
        self.last_input_timing: dict[str, float] = {}
        # Only the accepted utterance of the current read may identify its
        # speaker. A retry, interruption or typed console turn must not reuse it.
        self.last_audio_pcm: bytes = b""
        self.last_audio_rate: int = self.SAMPLE_RATE
        self.last_input_source: str = ""
        # Optional observer only schedules current-audio identity work. It
        # should return immediately so recognition can overlap ASR on the PC.
        self.audio_observer = None

    def preload(self) -> None:
        """Load only the local model; device discovery can finish independently."""
        if self._closed.is_set():
            raise RuntimeError("StackChan 语音输入已关闭")
        with self._load_lock:
            if self._recognizer is not None:
                return
            try:
                import numpy as np
            except (ImportError, OSError) as error:
                raise RuntimeError(
                    "当前 Python 环境未能加载 numpy。请在运行 main.py 的环境执行："
                    "python -m pip install numpy==1.26.4。"
                    f"原始错误：{error}"
                ) from error

            started = time.perf_counter()
            recognizer, model_name = self._load_seaco()
            self._np = np
            self._recognizer = recognizer
            print(
                f"[StackChanASR] 已就绪：SeACo/{model_name}，CPU {self.num_threads} 线程，"
                f"加载 {time.perf_counter() - started:.2f}s。",
                flush=True,
            )

    def _load_seaco(self):
        """Load an already exported model; the wrapper must never prepare one."""
        model_names = ("model_quant.onnx", "model_eb_quant.onnx")
        # Validate before the third-party constructor: absent ONNX files must
        # never trigger its automatic download/export behavior at an exhibit.
        missing = seaco_missing(self.model_dir)
        if missing:
            raise FileNotFoundError(
                f"SeACo 离线模型不完整：{self.model_dir}；缺少或为空：{', '.join(missing)}。"
                "请把已经导出的量化 SeACo 模型及配套配置放回该目录；运行时不下载或导出模型。"
            )
        try:
            import yaml
            from funasr_onnx import SeacoParaformer
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "当前 Python 环境未能加载 SeACo ONNX。请在运行 main.py 的环境执行："
                "python -m pip install funasr-onnx==0.4.3。"
                f"原始错误：{error}"
            ) from error

        try:
            with (self.model_dir / "config.yaml").open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            with (self.model_dir / "tokens.json").open("r", encoding="utf-8") as handle:
                tokens = json.load(handle)
        except (OSError, ValueError, yaml.YAMLError) as error:
            raise ValueError(f"SeACo 配套配置无法读取，请检查模型包：{error}") from error
        if not isinstance(config, dict) or config.get("model") != "SeacoParaformer":
            raise ValueError("config.yaml 必须来自官方 SeacoParaformer 模型，不能混用普通 Paraformer。")
        frontend = config.get("frontend_conf", {})
        if not isinstance(frontend, dict) or frontend.get("fs") != self.SAMPLE_RATE:
            raise ValueError("SeACo 模型必须使用 16000 Hz 前端，才能接收栈酱的 PCM。")
        if not isinstance(tokens, list) or not tokens or not all(isinstance(t, str) for t in tokens):
            raise ValueError("SeACo tokens.json 必须是同一模型包的非空 token 数组。")

        try:
            from friston3.knowledge.speech_entities import hotwords
            from friston3.knowledge.terra_kb import connect

            with closing(connect()) as db:
                words = hotwords(db, max_chars=10)
            if not words:
                raise ValueError("知识库没有可用的人名热词")
            if not isinstance(words, list) or any(
                not isinstance(word, str) or not word or len(word) > 10
                or any(c.isspace() for c in word) for word in words
            ):
                raise ValueError("热词必须是 1..10 个字符且不含空白的名称列表")
        except Exception as error:
            raise RuntimeError(
                "SeACo 热词读取失败。请确认补丁文件完整，并退出主程序后执行 "
                "python -m friston3.knowledge.terra_kb build 重建知识库；"
                f"原始错误：{error}"
            ) from error
        # Keep every returned name. OOV characters follow the official
        # wrapper's <unk> behavior, but report the limitation at startup.
        vocabulary = set(tokens)
        oov_names = [word for word in words if any(c not in vocabulary for c in word)]
        if oov_names:
            print(
                f"[StackChanASR] {len(oov_names)} 个热词含模型字表外字符，仍传入完整词表；"
                "这些字符会按官方接口映射为 <unk>，不能保证原字输出。示例："
                + "、".join(oov_names[:5])
                + ("……" if len(oov_names) > 5 else ""),
                flush=True,
            )
        self._hotword_text = " ".join(words)
        self.hotword_count = len(words)

        class LocalSeaco(SeacoParaformer):
            """Correct the wrapper's empty-word entry, caching tokenization only."""

            def proc_hotword(self, text):
                if getattr(self, "_prepared_text", None) != text:
                    ids, lengths = super().proc_hotword(text)
                    if not text:
                        # 0.4.3 splits "" into an empty word (length -1) plus
                        # the official trailing sos/null row. Keep only null.
                        ids, lengths = ids[-1:], lengths[-1:]
                    self._prepared_text = text
                    self._prepared_words = (ids, lengths)
                return self._prepared_words

        print(
            f"[StackChanASR] 加载本地 SeACo-Paraformer（CPU，只加载一次），"
            f"原生热词：{len(words)} 个。",
            flush=True,
        )
        try:
            recognizer = LocalSeaco(
                model_dir=str(self.model_dir),
                batch_size=1,
                device_id="-1",
                quantize=True,
                intra_op_num_threads=self.num_threads,
            )
            # This only maps text to IDs. Neither ASR nor a model warm-up runs.
            # The wrapper logs each OOV character separately; our bounded
            # startup summary already reports that limitation. Suppress only
            # those duplicate messages during this one-time tokenization.
            def keep_non_oov_logs(record):
                return not record.getMessage().startswith("oov character ")

            wrapper_logger = logging.getLogger("funasr_onnx")
            wrapper_logger.addFilter(keep_non_oov_logs)
            try:
                recognizer.proc_hotword(self._hotword_text)
            finally:
                wrapper_logger.removeFilter(keep_non_oov_logs)
        except Exception as error:
            raise RuntimeError(
                "SeACo 加载失败；请确认 funasr-onnx==0.4.3 已安装，"
                f"主模型、热词编码器和配置来自同一次导出：{error}"
            ) from error
        return recognizer, model_names[0]

    def _transcribe(self, samples) -> str:
        try:
            results = self._recognizer(samples, self._hotword_text)
        except Exception as error:
            raise RuntimeError(f"SeACo 转写失败：{error}") from error
        if not results:
            return ""
        if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
            raise RuntimeError("SeACo 返回了非预期结果；请使用 funasr-onnx==0.4.3。")
        prediction = results[0].get("preds")
        # The official no-timestamp path returns (text, tokens); the timestamp
        # path returns text directly. Never turn the whole tuple into a string.
        if isinstance(prediction, (tuple, list)) and prediction:
            prediction = prediction[0]
        if not isinstance(prediction, str):
            raise RuntimeError("SeACo 结果缺少文字 preds；请检查模型与运行库版本。")
        return prediction.strip()

    def read(self) -> str:
        """Wait for one utterance; keyboard interrupts remain visible to main."""
        self.clear_last_audio()
        self.last_input_timing = {}
        self.preload()
        reconnecting = False
        prompted_mode = None
        while not self._closed.is_set():
            self.mode = SETTINGS.get("mic_mode", "touch")
            if not self.face.wait_ready(timeout=1.0):
                if not reconnecting:
                    print("[StackChanMIC] 等待栈酱连接；连接恢复后继续收音，Ctrl+C 进入电脑控制台。", flush=True)
                reconnecting = True
                self.clear_last_audio()
                continue
            try:
                epoch = self.face.connection_epoch
                if self._device_epoch != epoch:
                    info = self.face.get_info()
                    if "MIC_PCM16_V1" not in info:
                        raise RuntimeError("栈酱固件未提供麦克风接口 MIC_PCM16_V1，请更新当前项目固件。")
                    self._device_epoch = epoch
                if prompted_mode != self.mode or reconnecting:
                    prompt = "轻触栈酱屏幕后说话" if self.mode == "touch" else "正在等待说话"
                    print(f"\n[StackChanMIC] {prompt}，停顿后自动提交；Ctrl+C 进入电脑控制台。", flush=True)
                    prompted_mode = self.mode
                reconnecting = False
                capture_started = time.perf_counter()
                pcm = self.face.record_pcm(
                    mode=self.mode, wait_ms=1000 if self.mode == "touch" else 2000,
                    max_ms=self.max_ms, silence_ms=self.silence_ms, threshold=self.threshold,
                )
            except (ConnectionError, TimeoutError) as error:
                self.clear_last_audio()
                self._device_epoch = None
                if not reconnecting:
                    print(f"[StackChanMIC] 本次录音已结束，等待连接恢复：{error}", flush=True)
                reconnecting = True
                self._closed.wait(0.2)
                continue
            if epoch != self.face.connection_epoch:
                self.clear_last_audio()
                continue
            captured_at = time.perf_counter()
            if self._closed.is_set():
                break
            if not pcm:
                continue
            if not isinstance(pcm, bytes) or len(pcm) % 2:
                raise ValueError("栈酱返回了无效的 PCM16LE 麦克风数据")
            if len(pcm) > self.SAMPLE_RATE * 2 * 12:
                raise ValueError("栈酱录音超过协议允许的 12 秒上限")

            # Explicit little-endian PCM keeps the representation independent
            # of the host CPU. ASR expects mono float32 in [-1, 1).
            samples = self._np.frombuffer(pcm, dtype="<i2").astype(self._np.float32)
            samples *= 1.0 / 32768.0
            if callable(self.audio_observer):
                try:
                    self.audio_observer(pcm, self.SAMPLE_RATE)
                except Exception as error:
                    print(f"[OwnerIdentity] 本轮身份观察未启动：{error}", flush=True)
            text = self._transcribe(samples)
            if epoch != self.face.connection_epoch:
                self.clear_last_audio()
                continue
            asr_seconds = time.perf_counter() - captured_at
            self.last_input_timing = {
                "capture_and_transfer_s": captured_at - capture_started,
                "audio_s": len(pcm) / (self.SAMPLE_RATE * 2),
                "asr_s": asr_seconds,
            }
            if self._closed.is_set():
                break
            if not text or not any(character.isalnum() for character in text):
                print("[StackChanASR] 没有识别到有效文字，请再说一次。", flush=True)
                continue

            print(f"你（栈酱麦克风）: {text}", flush=True)
            print(
                f"[StackChanASR] 音频 {self.last_input_timing['audio_s']:.2f}s，"
                f"转写 {asr_seconds:.2f}s。",
                flush=True,
            )
            # No hard-coded name replacement or inferred speaker identity.
            self.last_audio_pcm = pcm
            self.last_audio_rate = self.SAMPLE_RATE
            self.last_input_source = "speech"
            return text

        raise EOFError("StackChan 语音输入已关闭")

    def set_mode(self, mode: str) -> str:
        if mode not in ("touch", "vad"):
            raise ValueError("收音方式必须是轻触收音或语音检测")
        SETTINGS.update(mic_mode=mode)
        self.mode = mode
        return mode

    def clear_last_audio(self) -> None:
        """Release the in-memory clip, also before a local typed fallback."""
        self.last_audio_pcm = b""
        self.last_audio_rate = self.SAMPLE_RATE
        self.last_input_source = ""

    def close(self) -> None:
        """Stop accepting input; the caller owns and closes the COM scheduler."""
        self._closed.set()
        self.clear_last_audio()
