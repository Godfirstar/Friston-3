"""One offline SeACo recognizer on the shared, reconnecting device transport.

The firmware captures PCM and detects utterance endings. The recognizer stays
loaded across USB/Wi-Fi changes. Recordings are kept only for the current turn.
"""

from __future__ import annotations
# 本文件是ASR（Automatic Speech Recognition，语音→文字）；TTS（文字→语音）在voice_output。
# 主链：主程序的输入读取 → read → face.record_pcm → _transcribe → 返回文字给Agent。
# 这里复用face_events的设备连接，不重新开串口；识别模型留在电脑CPU，录音在StackChan端。

import json
import logging
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

from friston3.runtime_settings import SETTINGS

if TYPE_CHECKING:
    # TYPE_CHECKING在正常运行时为False；这段只供IDE/类型检查识别注解中的类名。
    from .face_events import StackChanFaceEvents


SEACO_REQUIRED = ("model_quant.onnx", "model_eb_quant.onnx", "config.yaml", "am.mvn", "tokens.json")


def seaco_missing(folder: Path) -> list[str]:
    # 【调用方】seaco_model_dir/_load_seaco；输入模型目录，返回缺失或空文件的名字list。
    # 列表推导式逐个检查固定配套文件；Path的/拼路径，stat().st_size是文件字节数。
    return [name for name in SEACO_REQUIRED
            if not (folder / name).is_file() or (folder / name).stat().st_size == 0]


def seaco_model_dir() -> Path:
    # 【调用方】StackChanSpeechInput.__init__；寻找已有离线模型，返回Path，不下载/加载权重。
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
    # 采样率16000 Hz表示每秒16000个单声道采样；PCM16每个采样2字节，一秒32000字节。

    def __init__(self, face: StackChanFaceEvents):
        # 【创建方】主程序；face是已存在的设备协调器实例，self表示本次语音输入对象。
        # 这里只保存配置/共享对象，真正ASR模型由preload延迟加载；构造不开始一次录音。
        self.face = face
        # 保存同一个对象引用，后续record_pcm走它的共享设备通道。
        self.mode = SETTINGS.get("mic_mode", "touch")
        self.max_ms, self.silence_ms, self.threshold = 8000, 700, 100
        self.num_threads = 2
        self.model_dir = seaco_model_dir()
        self._device_epoch = None
        self._recognizer = None
        # None表示模型尚未加载；成功后保存LocalSeaco实例，多轮read复用它。
        self._np = None
        self._hotword_text = ""
        self.hotword_count = 0
        self._load_lock = threading.Lock()
        # 防止多个调用方并发preload时构造两份大模型；with锁只允许一方进入加载段。
        self._closed = threading.Event()
        # Event是线程间停止信号；set发出停止，is_set查询，wait可被停止信号提前唤醒。
        self.last_input_timing: dict[str, float] = {}
        # Only the accepted utterance of the current read may identify its
        # speaker. A retry, interruption or typed console turn must not reuse it.
        self.last_audio_pcm: bytes = b""
        self.last_audio_rate: int = self.SAMPLE_RATE
        self.last_input_source: str = ""
        # Optional observer only schedules current-audio identity work. It
        # should return immediately so recognition can overlap ASR on the PC.
        self.audio_observer = None
        # 可由外部注入(pcm_bytes,sample_rate)回调，用于当前声音的身份处理。
        # 它在完整录音传输结束之后、ASR之前调用；不是“录音开始”事件。

    def preload(self) -> None:
        """Load only the local model; device discovery can finish independently."""
        # 【调用方】main.py显式预加载或read；输出None，副作用是缓存识别器与NumPy模块引用。
        if self._closed.is_set():
            raise RuntimeError("StackChan 语音输入已关闭")
        with self._load_lock:
            # with进入锁、退出自动释放；即使导入或加载抛异常也不会忘记释放。
            if self._recognizer is not None:
                return
            try:
                import numpy as np
                # np是第三方NumPy模块别名；函数内导入把依赖错误留在真正需要ASR的路径处理。
            except (ImportError, OSError) as error:
                raise RuntimeError(
                    "当前 Python 环境未能加载 numpy。请在运行 main.py 的环境执行："
                    "python -m pip install numpy==1.26.4。"
                    f"原始错误：{error}"
                ) from error

            started = time.perf_counter()
            recognizer, model_name = self._load_seaco()
            # 元组解包：一个可调用识别器对象和用于日志的模型文件名。
            self._np = np
            self._recognizer = recognizer
            print(
                f"[StackChanASR] 已就绪：SeACo/{model_name}，CPU {self.num_threads} 线程，"
                f"加载 {time.perf_counter() - started:.2f}s。",
                flush=True,
            )

    def _load_seaco(self):
        """Load an already exported model; the wrapper must never prepare one."""
        # 【调用方】preload；读配套配置/词表/知识库热词，构造本地SeACo ONNX识别器。
        # 返回(recognizer,model_name)；加载权重与运行一次音频识别是两件事。
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
                # yaml.safe_load把YAML文本解析成Python对象；此处期望dict，不执行任意YAML对象构造。
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
                # connect来自terra_kb，得到SQLite连接；closing保证离开with时调用db.close。
                words = hotwords(db, max_chars=10)
                # speech_entities.hotwords从知识库提取名字列表；这是模型条件，不是识别后硬替换名字。
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
        # set去重并用于快速成员查询；tokens来自模型词表JSON，不是用户本轮识别文本。
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
        # join将名字list按空格拼成官方wrapper需要的单个str；原列表数量另存hotword_count。
        self.hotword_count = len(words)

        class LocalSeaco(SeacoParaformer):
            """Correct the wrapper's empty-word entry, caching tokenization only."""
            # 在函数内部定义子类，继承funasr_onnx的识别能力，只重写热词预处理方法。

            def proc_hotword(self, text):
                # 这里self是LocalSeaco实例，不是外层StackChanSpeechInput实例。
                # 同一text命中缓存直接复用(token IDs,lengths)，不缓存某次音频的转写结果。
                if getattr(self, "_prepared_text", None) != text:
                    ids, lengths = super().proc_hotword(text)
                    # super调用父类原方法；再修正空输入分支，避免把整套识别算法复制进项目。
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
            # ClassName(...)构造识别器并加载本地ONNX；device_id="-1"指定CPU。
            # batch_size=1处理一段音频；quantize=True选量化模型文件；不在这里训练参数。
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
                # 这里只预先把热词转ID；不是读麦克风，也不执行一次ASR暖机。
            finally:
                wrapper_logger.removeFilter(keep_non_oov_logs)
        except Exception as error:
            raise RuntimeError(
                "SeACo 加载失败；请确认 funasr-onnx==0.4.3 已安装，"
                f"主模型、热词编码器和配置来自同一次导出：{error}"
            ) from error
        return recognizer, model_names[0]

    def _transcribe(self, samples) -> str:
        # 【调用方】read；samples是一维NumPy float32[N]，16kHz单声道，幅值约[-1,1)。
        # 返回去首尾空白的识别字符串；下游是已加载识别器的__call__，不负责设备传输。
        try:
            results = self._recognizer(samples, self._hotword_text)
            # 对象后加()调用它的__call__；这里才实际做音频特征处理和ONNX模型推理。
        except Exception as error:
            raise RuntimeError(f"SeACo 转写失败：{error}") from error
        if not results:
            return ""
        if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
            raise RuntimeError("SeACo 返回了非预期结果；请使用 funasr-onnx==0.4.3。")
        prediction = results[0].get("preds")
        # results是单样本list，results[0]是dict；get取预测字段，缺失时返回None。
        # The official no-timestamp path returns (text, tokens); the timestamp
        # path returns text directly. Never turn the whole tuple into a string.
        if isinstance(prediction, (tuple, list)) and prediction:
            prediction = prediction[0]
            # 某版本返回(text,tokens)，只取文字；str(整个tuple)会把结构也念/显示出来。
        if not isinstance(prediction, str):
            raise RuntimeError("SeACo 结果缺少文字 preds；请检查模型与运行库版本。")
        return prediction.strip()

    def read(self) -> str:
        """Wait for one utterance; keyboard interrupts remain visible to main."""
        # 【调用方】主程序输入循环；成功时返回一条非空文字，关闭时抛EOFError。
        # 读取会等待用户说话/设备传输并运行ASR；模型已加载时preload会直接返回。
        self.clear_last_audio()
        # 清除上一轮身份音频，防止本轮重试/键盘输入错误地沿用旧说话人证据。
        self.last_input_timing = {}
        self.preload()
        reconnecting = False
        prompted_mode = None
        while not self._closed.is_set():
            # 每轮可能是等待连接、空录音或无效文字；continue会回到while尝试下一次。
            self.mode = SETTINGS.get("mic_mode", "touch")
            if not self.face.wait_ready(timeout=1.0):
                if not reconnecting:
                    print("[StackChanMIC] 等待栈酱连接；连接恢复后继续收音，Ctrl+C 进入电脑控制台。", flush=True)
                reconnecting = True
                self.clear_last_audio()
                continue
            try:
                epoch = self.face.connection_epoch
                # epoch是连接代次标识；用于识别录制/转写过程中是否发生断线重连。
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
                # 该调用定义在face_events，完成设备录音和MIC数据传输后才返回完整bytes。
                # mode决定触屏或能量VAD触发；这里尚未调用身份观察回调，也尚未ASR。
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
                # len(bytes)是字节数；PCM16每采样2字节，%2非零表示截断/格式不完整。
                raise ValueError("栈酱返回了无效的 PCM16LE 麦克风数据")
            if len(pcm) > self.SAMPLE_RATE * 2 * 12:
                raise ValueError("栈酱录音超过协议允许的 12 秒上限")

            # Explicit little-endian PCM keeps the representation independent
            # of the host CPU. ASR expects mono float32 in [-1, 1).
            samples = self._np.frombuffer(pcm, dtype="<i2").astype(self._np.float32)
            # frombuffer按已有bytes解释为一维整数数组；<是小端，i2是有符号2字节整数。
            # N=len(pcm)//2；astype创建float32数组，后续可做浮点归一化而不修改原bytes。
            samples *= 1.0 / 32768.0
            # 原PCM16范围-32768..32767缩放到[-1,1)；*=是原地逐元素乘，shape仍[N]。
            if callable(self.audio_observer):
                # callable确认回调可调用；此时完整MIC传输已结束，下面ASR还没开始。
                try:
                    self.audio_observer(pcm, self.SAMPLE_RATE)
                    # 传原始bytes与16000采样率；回调应只安排后台身份工作后立即返回。
                    # 这样身份计算可与随后ASR重叠，不意味着录音阶段已并行识别人声。
                except Exception as error:
                    print(f"[OwnerIdentity] 本轮身份观察未启动：{error}", flush=True)
            text = self._transcribe(samples)
            # 将float波形转文字；识别“说了什么”与audio_observer判断“谁在说”是不同任务。
            if epoch != self.face.connection_epoch:
                self.clear_last_audio()
                continue
            asr_seconds = time.perf_counter() - captured_at
            # captured_at设在录音传输返回后；此统计也包含转换数组/调用observer等微小开销。
            self.last_input_timing = {
                "capture_and_transfer_s": captured_at - capture_started,
                "audio_s": len(pcm) / (self.SAMPLE_RATE * 2),
                # 时长=总字节/(每秒采样数×每采样字节数)，无需从识别文字长度猜音频时长。
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
            # 只在本轮有效文字被接受后保留原录音，供本轮身份判别；不会保存历史录音库。
            self.last_audio_rate = self.SAMPLE_RATE
            self.last_input_source = "speech"
            return text

        raise EOFError("StackChan 语音输入已关闭")

    def set_mode(self, mode: str) -> str:
        # 【调用方】设备设置功能；持久化touch/vad选择并返回新模式，不在本函数开始录音。
        if mode not in ("touch", "vad"):
            raise ValueError("收音方式必须是轻触收音或语音检测")
        SETTINGS.update(mic_mode=mode)
        self.mode = mode
        return mode

    def clear_last_audio(self) -> None:
        """Release the in-memory clip, also before a local typed fallback."""
        # 【调用方】read/close及键盘回退；b""是空bytes，与空文本""不同。
        self.last_audio_pcm = b""
        self.last_audio_rate = self.SAMPLE_RATE
        self.last_input_source = ""

    def close(self) -> None:
        """Stop accepting input; the caller owns and closes the COM scheduler."""
        # 发停止信号并释放本轮录音引用；face是外部传入的共享资源，所以这里不关闭它。
        self._closed.set()
        self.clear_last_audio()
