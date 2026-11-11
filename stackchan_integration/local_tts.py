"""The single offline Friston-3 voice: Kokoro v1.1 Chinese/English int8.

One fixed stock male voice with a light electronic colour. Model installation
is explicit; normal startup and synthesis never download or change backends.
"""
from __future__ import annotations
# 本文件提供TTS引擎：文字→float波形→16kHz PCM16 bytes；不识别麦克风，也不连接设备。
# tts_daemon持有OfflineCharacterVoice实例；tts_export.synthesize_part调用pcm16_chunks。

import math
import os
import re
import time
from pathlib import Path
from .audio_protocol import SAMPLE_RATE, MAX_PCM_BYTES
# SAMPLE_RATE是最终设备协议的16000Hz，MAX_PCM_BYTES是单次设备音频容量，不是模型参数。

BILINGUAL_MODEL_NAME = "kokoro-int8-multi-lang-v1_1"
_NAME = re.compile(r"(?<![A-Za-z0-9_])(?:Friston|Frinston|Firston)[\s\-\u2011\u2013\u2014]*3(?![A-Za-z0-9_])", re.I)
_BARE_NAME = re.compile(r"(?<![A-Za-z0-9_\-\u2011\u2013\u2014])(?:Friston|Frinston|Firston)(?![A-Za-z0-9_\-\u2011\u2013\u2014]|\s*\d)", re.I)
_CJK = re.compile(r"[\u3400-\u9fff]")


def model_root() -> Path:
    # 【调用方】定位/安装工具；从环境变量计算模型根目录，仅返回Path，不创建或下载文件。
    base = os.environ.get("LOCALAPPDATA")
    return Path(base) / "Friston3" / "tts" if base else Path.home() / "Friston3Models" / "tts"


def kokoro_missing(folder: Path) -> list[str]:
    # 【调用方】定位函数、引擎构造、voice_model安装/status；返回缺失资源路径list。
    required = [folder / name for name in (
        "model.int8.onnx", "voices.bin", "tokens.txt", "lexicon-us-en.txt", "lexicon-zh.txt",
        "phone-zh.fst", "date-zh.fst", "number-zh.fst",
    )]
    # 列表推导式把每个文件名接到目录下；Path的/是路径拼接，不是数值除法。
    missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
    for name in ("espeak-ng-data", "dict"):
        if not (folder / name).is_dir() or not any((folder / name).iterdir()):
            missing.append(str(folder / name))
    return missing


def default_bilingual_dir() -> Path:
    """Locate this one model, including its location before the project rename."""
    # 按候选目录顺序寻找完整模型；兼容旧项目名只改变定位，不迁移/复制权重。
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
    # 【调用方】OfflineCharacterVoice.prepare_text；输入str→发音用str，保持聊天原文不变。
    text = re.sub(r"\s+", " ", str(text)).strip()
    # re.sub是正则替换；\s+匹配连续空白，先合并为空格，再strip去首尾空白。
    chinese_context = bool(_CJK.search(text))
    pronunciation = "弗里斯顿三号" if chinese_context else "Friston three"
    text = _NAME.sub(lambda _: pronunciation, text)
    # _NAME是模块顶部编译的正则；lambda接收匹配对象(这里不用，命名_)并返回发音字符串。
    bare = "弗里斯顿" if chinese_context else "Friston"
    return _BARE_NAME.sub(lambda _: bare, text)


class OfflineCharacterVoice:
    """One resident model, returning mono little-endian 16-kHz PCM16."""

    def __init__(self):
        # 【创建方】tts_daemon.serve，只创建一次；self持有模型与音频处理依赖，供多请求复用。
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
        # 将导入模块保存成实例属性：后文self._np.array就是numpy.array，非自定义算法。
        self.model_dir = folder
        self.backend, self.speaker, self.style = "kokoro", 58, "friston"
        # 多变量赋值；speaker是模型内预置音色编号，不是当前对话说话人的身份。
        self.speed, self.fx_mix = 1.0, 0.12
        threads = min(4, max(1, (os.cpu_count() or 4) // 2))
        # cpu_count可为None，or4提供回退；//是整数除法，外层min/max得到配置线程数。
        voice_config = sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=str(folder / "model.int8.onnx"), voices=str(folder / "voices.bin"),
            tokens=str(folder / "tokens.txt"), data_dir=str(folder / "espeak-ng-data"),
            lexicon=",".join(str(folder / n) for n in ("lexicon-us-en.txt", "lexicon-zh.txt")),
        )
        # 构造官方Kokoro配置对象；文件分别提供ONNX权重、音色、token、发音字典等。
        # 配置只是说明如何运行；后面OfflineTts(config)才实际构造加载引擎。
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
        # 模型在这里加载并驻留CPU；这行尚未对用户某段文本执行generate。
        count = getattr(self._tts, "num_speakers", None)
        count = count() if callable(count) else count
        # 兼容库版本：num_speakers可能是可调用方法，也可能已是整数属性。
        if isinstance(count, int) and not self.speaker < count:
            raise ValueError("模型不含既定 58 号音色；请安装 Kokoro v1.1 中英 int8 包")
        self._colour_filter = signal.butter(2, (100, 4700), btype="bandpass", fs=SAMPLE_RATE, output="sos")
        # scipy.signal构造带通滤波器系数，sos为二阶节形式；保存系数不等于已处理音频。
        self.last_synthesis_timing: dict[str, float] = {}

    def prepare_text(self, text: str) -> str:
        # 实例层入口，转调上面的纯文本函数；daemon可先整段规范发音再拆短句。
        return prepare_speech_text(text)

    def warmup(self) -> float:
        """Run the real text frontend and inference once without playback."""
        # 【调用方】daemon加载后；确实运行一次合成以触发首次开销，返回耗时秒数，不发送设备。
        started = time.perf_counter()
        self._synthesize_pcm("准备好了。Ready.")
        return time.perf_counter() - started

    def _synthesize_pcm(self, text: str) -> bytes:
        # 【调用方】warmup/pcm16_chunks；输入短文本，输出单声道、小端、16kHz PCM16的bytes。
        # 主步骤：文本前处理→模型generate→重采样/音色处理→音量归一→量化编码。
        started = time.perf_counter()
        self.last_synthesis_timing = {}
        text = self.prepare_text(text)
        if not text or len(text) > 160:
            raise ValueError("语音片段必须是 1 至 160 个字符")
        options = self._sherpa.GenerationConfig()
        # 创建本次生成参数对象；不是重建或重新训练模型。
        options.sid, options.speed, options.silence_scale = self.speaker, self.speed, 0.2
        inference_started = time.perf_counter()
        audio = self._tts.generate(text, options)
        # 这里才进行文本到语音模型推理；audio包含浮点波形samples及模型原生sample_rate。
        processing_started = time.perf_counter()
        samples = self._np.array(audio.samples, dtype=self._np.float32, copy=True)
        # NumPy数组shape[N]，每个元素是一瞬间波形幅值；它不是图像/模型权重/文字token。
        # copy=True创建自己的数组，后续音频处理不改动引擎返回对象可能共享的内部缓冲。
        rate = int(audio.sample_rate)
        if samples.ndim != 1 or samples.size == 0 or rate <= 0:
            # ndim是轴数，size是总元素数；一维非空数组表示当前接口需要的单声道音频。
            raise RuntimeError("TTS did not produce valid mono audio")
        if not self._np.all(self._np.isfinite(samples)):
            raise RuntimeError("TTS produced invalid sample values")
        if rate != SAMPLE_RATE:
            common = math.gcd(rate, SAMPLE_RATE)
            # 最大公约数把目标/原采样率化简成整数升采样与降采样比例。
            samples = self._signal.resample_poly(samples, SAMPLE_RATE // common, rate // common)
            # scipy多相重采样改变每秒采样点数，保持实际音频时长；不是简单修改rate标签。
        coloured = self._signal.sosfilt(self._colour_filter, samples)
        # 用前面保存的滤波系数处理波形，返回同长度数组。
        carrier = self._np.sin(2.0 * self._np.pi * 46.0 * self._np.arange(len(samples)) / SAMPLE_RATE)
        # arange(N)产生采样下标；下标/采样率是秒，sin生成46Hz载波，shape同为[N]。
        samples = (1.0 - self.fx_mix) * samples + self.fx_mix * coloured * carrier
        # 逐元素混合原声与经过滤波/调制的波形，fx_mix决定电子音色占比。
        peak = float(self._np.max(self._np.abs(samples)))
        if peak > 1e-4:
            frame_count = len(samples) // 320
            # 16000Hz下320个采样是20ms；只取完整帧来估计活动语音能量。
            if frame_count:
                frames = samples[:frame_count * 320].reshape(frame_count, 320)
                # 切片去掉不足一帧的尾部，reshape变[帧数,每帧320点]，不改变波形元素次序。
                frame_energy = self._np.mean(frames * frames, axis=1)
                # 平方后沿每帧采样轴求均值，得到[帧数]能量；axis=1表示每一行内部。
                active = frame_energy[frame_energy > max(1e-6, float(frame_energy.max()) * 0.01)]
                if active.size:
                    rms = math.sqrt(float(self._np.mean(active)))
                    # 活动帧的均方值再开平方得到RMS幅值，用来计算响度增益。
                    gain = min(4.0, 0.10 / rms, 0.94 / peak)
                    samples = samples * gain
                    peak *= gain
        if peak > 0.94:
            samples = samples * (0.94 / peak)
        fade = min(80, len(samples) // 2)
        if fade:
            ramp = self._np.linspace(0.0, 1.0, fade)
            # linspace生成fade个均匀数值；让音频开头从0渐增，减少直接跳变的噪声。
            samples[:fade] *= ramp
            samples[-fade:] *= ramp[::-1]
            # 负下标取末尾；[::-1]反转ramp，使结尾从1渐降到0。
        pcm = (self._np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        # clip限制波形幅值，乘32767转整数幅度；<i2表示小端有符号16位整数。
        # tobytes按数组存储顺序导出原始采样bytes；不是WAV（没有WAV文件头）。
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
        # 【调用方】tts_export.synthesize_part；返回片段bytes列表，每项可由单条音频命令播放。
        # 一段文本只合成一次，超长时切波形；并未缓存所有历史文本的合成结果。
        pcm = self._synthesize_pcm(text)
        result = []
        while len(pcm) > MAX_PCM_BYTES:
            values = self._np.frombuffer(pcm, dtype="<i2")
            # 将已有PCM bytes零拷贝解释成[N]整数数组，以采样点为单位寻找切割位置。
            # Find the quietest 10-ms window in the last 1.5 seconds permitted.
            start, end, window = 168000, 191840, 160
            region = values[start:end].astype(self._np.float32)
            blocks = region[:len(region) // window * window].reshape(-1, window)
            # -1要求NumPy推断行数；每行window=160点，即16kHz下10ms的小窗口。
            energy = self._np.mean(blocks * blocks, axis=1)
            block = int(self._np.argmin(energy))
            # argmin返回最小能量窗口的下标，尽量把切口放在安静位置。
            low = start + block * window
            # Prefer a zero crossing within that small quiet interval.
            cut = low + int(self._np.argmin(self._np.abs(values[low:low + window].astype(self._np.int32))))
            result.append(pcm[:cut * 2])
            # cut按采样点计，bytes切片需乘2；每个16位采样占两个字节。
            pcm = pcm[cut * 2:]
        if pcm:
            result.append(pcm)
        return result

