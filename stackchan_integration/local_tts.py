"""The single offline Friston-3 voice: Kokoro v1.1 Chinese/English int8.

One fixed stock male voice with a light electronic colour. Model installation
is explicit; normal startup and synthesis never download or change backends.
"""
from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from .audio_protocol import SAMPLE_RATE, MAX_PCM_BYTES

BILINGUAL_MODEL_NAME = "kokoro-int8-multi-lang-v1_1"
_NAME = re.compile(r"(?<![A-Za-z0-9_])(?:Friston|Frinston|Firston)[\s\-\u2011\u2013\u2014]*3(?![A-Za-z0-9_])", re.I)
_BARE_NAME = re.compile(r"(?<![A-Za-z0-9_\-\u2011\u2013\u2014])(?:Friston|Frinston|Firston)(?![A-Za-z0-9_\-\u2011\u2013\u2014]|\s*\d)", re.I)
_CJK = re.compile(r"[\u3400-\u9fff]")


def model_root() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    return Path(base) / "Friston3" / "tts" if base else Path.home() / "Friston3Models" / "tts"


def kokoro_missing(folder: Path) -> list[str]:
    required = [folder / name for name in (
        "model.int8.onnx", "voices.bin", "tokens.txt", "lexicon-us-en.txt", "lexicon-zh.txt",
        "phone-zh.fst", "date-zh.fst", "number-zh.fst",
    )]
    missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
    for name in ("espeak-ng-data", "dict"):
        if not (folder / name).is_dir() or not any((folder / name).iterdir()):
            missing.append(str(folder / name))
    return missing


def default_bilingual_dir() -> Path:
    """Locate this one model, including its location before the project rename."""
    canonical = model_root() / BILINGUAL_MODEL_NAME
    base = os.environ.get("LOCALAPPDATA")
    previous_root = Path(base) / "Friston2" / "tts" if base else Path.home() / "Friston2Models" / "tts"
    project_models = Path(__file__).resolve().parent.parent / "models"
    for folder in (canonical, project_models / BILINGUAL_MODEL_NAME,
                   previous_root / BILINGUAL_MODEL_NAME):
        if not kokoro_missing(folder):
            return folder
    return canonical


def prepare_speech_text(text: str) -> str:
    """Pronunciation only: terminal text and dialogue memory remain untouched."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    chinese_context = bool(_CJK.search(text))
    pronunciation = "弗里斯顿三号" if chinese_context else "Friston three"
    text = _NAME.sub(lambda _: pronunciation, text)
    bare = "弗里斯顿" if chinese_context else "Friston"
    return _BARE_NAME.sub(lambda _: bare, text)


class OfflineCharacterVoice:
    """One resident model, returning mono little-endian 16-kHz PCM16."""

    def __init__(self):
        folder = default_bilingual_dir()
        missing = kokoro_missing(folder)
        if missing:
            raise FileNotFoundError(
                "Kokoro 离线模型不完整；运行 python -m stackchan_integration.voice_model install "
                "进行一次安装。缺少：" + ", ".join(missing)
            )
        try:
            import numpy as np
            import sherpa_onnx
            from scipy import signal
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "离线语音依赖未就绪；在 friston_tts 环境安装 sherpa-onnx numpy scipy。"
                f"原始错误：{exc}"
            ) from exc
        self._np, self._sherpa, self._signal = np, sherpa_onnx, signal
        self.model_dir = folder
        self.backend, self.speaker, self.style = "kokoro", 58, "friston"
        self.speed, self.fx_mix = 1.0, 0.12
        threads = min(4, max(1, (os.cpu_count() or 4) // 2))
        voice_config = sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=str(folder / "model.int8.onnx"), voices=str(folder / "voices.bin"),
            tokens=str(folder / "tokens.txt"), data_dir=str(folder / "espeak-ng-data"),
            lexicon=",".join(str(folder / n) for n in ("lexicon-us-en.txt", "lexicon-zh.txt")),
        )
        if hasattr(voice_config, "dict_dir"):
            voice_config.dict_dir = str(folder / "dict")
        model_config = sherpa_onnx.OfflineTtsModelConfig(
            kokoro=voice_config, num_threads=threads, provider="cpu", debug=False,
        )
        config = sherpa_onnx.OfflineTtsConfig(
            model=model_config,
            rule_fsts=",".join(str(folder / n) for n in
                ("phone-zh.fst", "date-zh.fst", "number-zh.fst")),
            max_num_sentences=1,
        )
        if not config.validate():
            raise ValueError("Kokoro 配置无效，请检查完整模型文件和 sherpa-onnx 安装")
        self._tts = sherpa_onnx.OfflineTts(config)
        count = getattr(self._tts, "num_speakers", None)
        count = count() if callable(count) else count
        if isinstance(count, int) and not self.speaker < count:
            raise ValueError("模型不含既定 58 号音色；请安装 Kokoro v1.1 中英 int8 包")
        self._colour_filter = signal.butter(2, (100, 4700), btype="bandpass", fs=SAMPLE_RATE, output="sos")
        self.last_synthesis_timing: dict[str, float] = {}

    def prepare_text(self, text: str) -> str:
        return prepare_speech_text(text)

    def warmup(self) -> float:
        """Run the real text frontend and inference once without playback."""
        started = time.perf_counter()
        self._synthesize_pcm("准备好了。Ready.")
        return time.perf_counter() - started

    def _synthesize_pcm(self, text: str) -> bytes:
        started = time.perf_counter()
        self.last_synthesis_timing = {}
        text = self.prepare_text(text)
        if not text or len(text) > 160:
            raise ValueError("语音片段必须是 1 至 160 个字符")
        options = self._sherpa.GenerationConfig()
        options.sid, options.speed, options.silence_scale = self.speaker, self.speed, 0.2
        inference_started = time.perf_counter()
        audio = self._tts.generate(text, options)
        processing_started = time.perf_counter()
        samples = self._np.array(audio.samples, dtype=self._np.float32, copy=True)
        rate = int(audio.sample_rate)
        if samples.ndim != 1 or samples.size == 0 or rate <= 0:
            raise RuntimeError("TTS did not produce valid mono audio")
        if not self._np.all(self._np.isfinite(samples)):
            raise RuntimeError("TTS produced invalid sample values")
        if rate != SAMPLE_RATE:
            common = math.gcd(rate, SAMPLE_RATE)
            samples = self._signal.resample_poly(samples, SAMPLE_RATE // common, rate // common)
        coloured = self._signal.sosfilt(self._colour_filter, samples)
        carrier = self._np.sin(2.0 * self._np.pi * 46.0 * self._np.arange(len(samples)) / SAMPLE_RATE)
        samples = (1.0 - self.fx_mix) * samples + self.fx_mix * coloured * carrier
        peak = float(self._np.max(self._np.abs(samples)))
        if peak > 1e-4:
            frame_count = len(samples) // 320
            if frame_count:
                frames = samples[:frame_count * 320].reshape(frame_count, 320)
                frame_energy = self._np.mean(frames * frames, axis=1)
                active = frame_energy[frame_energy > max(1e-6, float(frame_energy.max()) * 0.01)]
                if active.size:
                    rms = math.sqrt(float(self._np.mean(active)))
                    gain = min(4.0, 0.10 / rms, 0.94 / peak)
                    samples = samples * gain
                    peak *= gain
        if peak > 0.94:
            samples = samples * (0.94 / peak)
        fade = min(80, len(samples) // 2)
        if fade:
            ramp = self._np.linspace(0.0, 1.0, fade)
            samples[:fade] *= ramp
            samples[-fade:] *= ramp[::-1]
        pcm = (self._np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        self.last_synthesis_timing = {
            "prepare_s": inference_started - started,
            "inference_s": processing_started - inference_started,
            "processing_s": time.perf_counter() - processing_started,
            "total_s": time.perf_counter() - started,
        }
        return pcm

    def pcm16_chunks(self, text: str) -> list[bytes]:
        """Synthesize once, then split exceptional long clips near quiet samples.

        Normal phrase chunks fit one device command. This avoids synthesizing a
        too-long sentence again recursively merely to meet the transport cap.
        """
        pcm = self._synthesize_pcm(text)
        result = []
        while len(pcm) > MAX_PCM_BYTES:
            values = self._np.frombuffer(pcm, dtype="<i2")
            # Find the quietest 10-ms window in the last 1.5 seconds permitted.
            start, end, window = 168000, 191840, 160
            region = values[start:end].astype(self._np.float32)
            blocks = region[:len(region) // window * window].reshape(-1, window)
            energy = self._np.mean(blocks * blocks, axis=1)
            block = int(self._np.argmin(energy))
            low = start + block * window
            # Prefer a zero crossing within that small quiet interval.
            cut = low + int(self._np.argmin(self._np.abs(values[low:low + window].astype(self._np.int32))))
            result.append(pcm[:cut * 2])
            pcm = pcm[cut * 2:]
        if pcm:
            result.append(pcm)
        return result

