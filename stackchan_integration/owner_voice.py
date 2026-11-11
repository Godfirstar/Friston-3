"""Local CAM++ speaker verification for an enrolled owner and one utterance.

The owner controller persists enrollment vectors. This module keeps audio in
memory, returns no raw audio, and never downloads a model. A cosine score is a
similarity, not a probability; short/noisy/overlapping speech can be inconclusive.
"""

# 延迟求值类型标注，避免仅因注解提前加载某些依赖；此语句不启动模型。
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path


class OwnerVoiceRecognizer:
    """Compare enrolled owner vectors to one mono 16 kHz PCM16LE utterance.

    Constructing this object does not import torch/FunASR or load weights.
    ``preload()`` is suitable for startup after enrollment, or for enrollment
    itself. ``extract()`` can also load lazily. Failed loads are cached so a
    missing dependency does not repeatedly delay dialogue; restart after fixing.
    """

    SAMPLE_RATE = 16000

    def __init__(self):
        # OwnerIdentityManager.__init__ 创建本类；只设置路径、参数和锁，不导入torch或构造模型。
        # __file__ 是当前源码路径；Path.resolve().parent.parent 得到项目根目录。
        root = Path(__file__).resolve().parent.parent
        self.model_dir = root / "models" / "campplus"
        self.threshold = 0.65
        self.reject_threshold = 0.45
        self.min_duration = 1.5
        self.min_active = 1.0
        self.num_threads = 2
        self._model = None
        self._np = None
        self._load_error = ""
        self._model_id = ""
        self._embedding_size = None
        # 两把锁分别串行化“首次加载”和“模型推理”；普通status读取不触发推理。
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self.load_seconds = 0.0

    @property
    def model_id(self):
        # @property 让调用者用 recognizer.model_id 读取，语法像属性，实际执行这个getter方法。
        # 返回缓存的权重/配置摘要str，不在每次读取属性时重新计算哈希。
        return self._model_id

    def _model_files(self):
        # status/preload 内部调用；返回 (config_path,weight_path) 两个Path，只检查本地文件。
        config_path = self.model_dir / "config.yaml"
        weight_name = "campplus_cn_common.bin"
        metadata_path = self.model_dir / "configuration.json"
        if metadata_path.is_file():
            # Path.read_text返回str，json.loads转dict；utf-8-sig兼容带BOM的JSON文件。
            metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
            if not isinstance(metadata, dict):
                raise ValueError("CAM++ configuration.json 格式错误")
            paths = metadata.get("file_path_metas", {})
            if not isinstance(paths, dict):
                raise ValueError("CAM++ file_path_metas 格式错误")
            # or 返回第一个非空值，依次使用配置文件两种字段或默认文件名，不是三个字符串拼接。
            weight_name = paths.get("init_param") or metadata.get("model_file") or weight_name
        if not isinstance(weight_name, str) or not weight_name:
            raise ValueError("CAM++ 权重文件名无效")
        # Path上的/拼接路径，resolve消解相对路径；随后确认最终位置仍属于本地模型目录。
        weight_path = (self.model_dir / weight_name).resolve()
        # A local model package must not redirect loading to a URL or other tree.
        if not weight_path.is_relative_to(self.model_dir.resolve()):
            raise ValueError("CAM++ 权重必须位于本地模型目录中")
        for path in (config_path, weight_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"缺少本地 CAM++ 文件：{path}")
        if weight_path.stat().st_size < 1024:
            raise ValueError("CAM++ 权重文件不完整，可能下载成了 Git LFS 指针")
        return config_path, weight_path

    def status(self):
        """Describe local availability without importing a model or running it."""
        # OwnerIdentityManager.warmup/begin_utterance/status_text读取；返回状态dict，不下载或提取声纹。
        # result是局部字典，ready/loaded依据内存中_model是否已有对象，不是文件是否下载完。
        result = {
            "status": "ready" if self._model is not None else "unloaded",
            "loaded": self._model is not None,
            "model_id": self._model_id,
            "model_dir": str(self.model_dir),
            "threshold": self.threshold,
            "reject_threshold": self.reject_threshold,
            "load_s": round(self.load_seconds, 3),
        }
        if self._load_error:
            result.update(status="unavailable", reason=self._load_error)
        elif self._model is None:
            try:
                self._model_files()
            except (OSError, ValueError, TypeError) as error:
                result.update(status="unavailable", reason=str(error))
        return result

    def preload(self):
        """Load a validated local checkpoint on CPU once; no hub lookup."""
        # Manager.warmup/enroll 与本类extract调用；成功返回status=ok，不代表音频已识别。
        # with Lock 保证同一实例不会由两个线程同时加载；缓存错误后直接返回，避免每轮重复耗时。
        with self._load_lock:
            if self._model is not None:
                return {"status": "ok", "model_id": self._model_id}
            if self._load_error:
                return {"status": "unavailable", "reason": self._load_error, "model_id": ""}
            started = time.perf_counter()
            try:
                config_path, weight_path = self._model_files()
                # yaml(PyYAML)、numpy、funasr是第三方库；直到真正需要加载时才导入。
                import yaml
                import numpy as np
                from funasr import AutoModel

                # 原始配置bytes参与模型ID；safe_load仅解析YAML数据，不把标签当Python对象执行。
                config_bytes = config_path.read_bytes()
                config = yaml.safe_load(config_bytes.decode("utf-8-sig"))
                if not isinstance(config, dict) or config.get("model") != "CAMPPlus":
                    raise ValueError("config.yaml 必须来自 CAM++ / CAMPPlus 模型")
                # model_conf是模型结构参数dict，frontend是采样配置dict，embedding_size是输出维数int。
                model_conf = config.get("model_conf")
                if not isinstance(model_conf, dict):
                    raise ValueError("CAM++ 配置缺少 model_conf")
                if model_conf.get("output_level", "segment") != "segment":
                    raise ValueError("CAM++ 需使用 segment 输出")
                frontend = config.get("frontend_conf") or {}
                if not isinstance(frontend, dict) or frontend.get("fs", 16000) != 16000:
                    raise ValueError("CAM++ 需使用 16000 Hz 模型")
                embedding_size = model_conf.get("embedding_size", 192)
                if not isinstance(embedding_size, int) or not 32 <= embedding_size <= 4096:
                    raise ValueError("CAM++ embedding_size 无效")
                # 摘要同时覆盖配置和权重；分块读文件更新hash，不把整个checkpoint读成Python bytes。
                digest = hashlib.sha256(config_bytes)
                with weight_path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                # Supplying model_conf bypasses FunASR's download_model path.
                # CAMPPlus.inference computes its own fbank; no tokenizer,
                # ASR/VAD/companion speaker model, or remote code is needed.
                # FunASR AutoModel在这里构造网络/加载本地权重；device='cpu'指定执行设备。
                # 这不是训练：没有优化器或backward；后面的generate才处理一段输入音频。
                model = AutoModel(
                    model="CAMPPlus", model_conf=model_conf,
                    init_param=str(weight_path), model_path=str(self.model_dir),
                    device="cpu", ncpu=self.num_threads,
                    disable_update=True, disable_pbar=True, disable_log=True,
                    log_level="ERROR", trust_remote_code=False,
                    ignore_init_mismatch=False,
                    frontend=None, tokenizer=None,
                    vad_model=None, punc_model=None, spk_model=None,
                )
                # 加载成功才提交模型/模块引用与维数；_model_id用于拒绝跨模型混用旧录入模板。
                self._np = np
                self._model = model
                self._embedding_size = embedding_size
                self._model_id = "campplus:" + digest.hexdigest()
            except Exception as error:
                self._load_error = f"{type(error).__name__}: {error}"
                return {"status": "unavailable", "reason": self._load_error, "model_id": ""}
            finally:
                # finally在成功/失败路径均执行，load_seconds专门记录加载阶段耗时。
                self.load_seconds = time.perf_counter() - started
        return {"status": "ok", "model_id": self._model_id}

    def extract(self, pcm_bytes, sample_rate=16000):
        """Return a normalized enrollment vector only for an adequate clip.

        The energy gate is not a speaker-count or overlap detector. Enrollment
        must still be captured with the owner speaking alone.
        """
        # 调用方：OwnerIdentityManager.enroll（录入）及本类observe（本轮比对）。
        # 输入是单声道PCM16LE bytes与采样率int，输出含单位特征列表/质量信息的dict或失败dict。
        started = time.perf_counter()
        result = {"status": "insufficient_audio", "model_id": self._model_id}
        if sample_rate != self.SAMPLE_RATE:
            return {**result, "reason": "声纹输入必须是单声道 16000 Hz PCM16LE"}
        if not isinstance(pcm_bytes, (bytes, bytearray)) or not pcm_bytes or len(pcm_bytes) % 2:
            return {**result, "reason": "没有本轮有效录音"}
        # PCM16每采样点2字节，所以字节数/(2*每秒采样数)得到秒数，不是bytes长度直接等于时长。
        duration = len(pcm_bytes) / (2 * sample_rate)
        result["duration_s"] = round(duration, 3)
        if duration < self.min_duration or duration > 12.0:
            return {**result, "reason": f"录音需为 {self.min_duration:g}..12 秒；短句不用于声纹确认"}
        try:
            import numpy as np
            # frombuffer按小端(<)、有符号2字节整数(i2)解释录音；astype转float32，再缩放到约[-1,1]。
            # samples是一维波形[N]；这一步是信号数值转换，不是声纹神经网络。
            samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
            # clipped是接近满幅样本占比；rms=sqrt(mean(x^2))衡量整段能量，均为float。
            clipped = float(np.mean(np.abs(samples) >= 0.99))
            rms = float(np.sqrt(np.mean(samples * samples)))
            # //整除：16kHz时每帧320点=20ms；裁掉不足一帧的尾部，再reshape成[帧数,320]。
            frame_size = sample_rate // 50
            frames = samples[:len(samples) // frame_size * frame_size].reshape(-1, frame_size)
            # axis=1沿每帧采样点求均值，所以energies为[帧数]，一帧对应一个RMS值。
            energies = np.sqrt(np.mean(frames * frames, axis=1))
            # A modest absolute floor avoids calling silence/background hiss
            # speech. Keeping this explicit makes room calibration possible.
            # active是逐帧bool掩码；这是能量筛选，不判断说话人数，也不能证明真人活体。
            active = energies >= max(0.006, float(np.max(energies)) * 0.10)
            # bool求和得到活跃帧数；每秒50帧，除50得到有效发声秒数。
            active_s = float(np.sum(active)) / 50.0
            result.update(rms=round(rms, 5), active_s=round(active_s, 3),
                          clipped_fraction=round(clipped, 4),
                          quality_gate="energy_only")
            if rms < 0.004 or active_s < self.min_active:
                return {**result, "reason": "有效发声太短或音量过小，请正常说一句完整的话"}
            if clipped > 0.02:
                return {**result, "reason": "录音削波过多，请稍远离麦克风"}
            # 质量通过后才按需加载本地模型；loaded是加载状态dict，而不是声纹预测结果。
            loaded = self.preload()
            if loaded["status"] != "ok":
                return {**result, **loaded}
            # Trim only leading/trailing low-energy frames, keeping natural
            # pauses and a 100 ms margin. Do not concatenate unrelated speech.
            # flatnonzero给出True所在帧索引；首尾各加100ms余量，保留中间自然停顿。
            indices = np.flatnonzero(active)
            begin = max(0, int(indices[0]) * frame_size - sample_rate // 10)
            end = min(len(samples), (int(indices[-1]) + 1) * frame_size + sample_rate // 10)
            # 切片取[begin,end)采样点，copy使输入不再依赖大波形的共享视图。
            speech = samples[begin:end].copy()
            with self._infer_lock:
                # generate是FunASR AutoModel的方法；返回list[dict]，batch_size=1处理这一段音频。
                outputs = self._model.generate(input=speech, fs=sample_rate, batch_size=1)
            if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], dict):
                raise ValueError("CAM++ 未返回单条声纹结果")
            # vector从首条结果的spk_embedding字段取出，可能是PyTorch Tensor或NumPy数组。
            vector = outputs[0].get("spk_embedding")
            if hasattr(vector, "detach"):
                # Tensor.detach脱离梯度图，cpu转CPU内存，numpy转数组；不在这里反向传播。
                vector = vector.detach().cpu().numpy()
            vector = np.asarray(vector, dtype=np.float32)
            if vector.shape != (1, self._embedding_size) or not np.isfinite(vector).all():
                raise ValueError("CAM++ 声纹向量形状或数值无效")
            # 校验[1,D]后取第0项，得到[D]；D来自配置embedding_size，当前CAM++默认为192。
            vector = vector[0]
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-8:
                raise ValueError("CAM++ 返回零向量")
            # 除L2范数将向量归一化到长度1；tolist转换为JSON可保存的普通float列表。
            # inference_s从函数开始计时，包含质量处理和首次可能发生的加载，并非纯网络forward耗时。
            result.update(status="ok", model_id=self._model_id,
                          embedding=(vector / norm).tolist(),
                          inference_s=round(time.perf_counter() - started, 3),
                          reason="本轮录音声纹已提取；不包含活体或多人混说检测")
            return result
        except Exception as error:
            return {**result, "status": "error", "reason": f"{type(error).__name__}: {error}",
                    "inference_s": round(time.perf_counter() - started, 3)}

    def observe(self, pcm_bytes, templates, sample_rate=16000, *, model_id=None):
        """Match only this clip. Never return an enrollment vector in evidence.

        ``templates`` may be vectors or dictionaries with embedding/model_id.
        When passing plain vectors, also supply their shared ``model_id``.
        The mean of the best two references reduces dependence on one outlier.
        """
        # OwnerIdentityManager.begin_utterance的声纹线程调用；返回本轮证据dict，绝不返回新录入特征。
        # templates为向量列表或带model_id/embedding的字典列表；model_id为模板使用的编码器身份。
        if not isinstance(templates, (list, tuple)) or not templates:
            return {"status": "not_enrolled", "matched": False, "reason": "尚未录入主人声纹"}
        extracted = self.extract(pcm_bytes, sample_rate=sample_rate)
        # dict.pop取出并删除embedding；这样随后合并extracted到result时不会泄露本轮特征。
        vector = extracted.pop("embedding", None)
        result = {**extracted, "matched": False, "threshold": self.threshold,
                  "reject_threshold": self.reject_threshold}
        if extracted["status"] != "ok":
            return result
        if model_id is not None and model_id != self._model_id:
            return {**result, "status": "model_mismatch", "reason": "声纹模型已变更，请重新录入主人声纹"}
        np = self._np
        # scores是当前录音与各有效主人模板的相似度list[float]，不是跨多个说话人的分类概率。
        scores = []
        try:
            # probe是extract已归一化的[D]数组；reference稍后也归一化，点积才等于余弦相似度。
            probe = np.asarray(vector, dtype=np.float32)
            for template in templates:
                if isinstance(template, dict):
                    if template.get("model_id") != self._model_id:
                        continue
                    reference = template.get("embedding")
                else:
                    if model_id != self._model_id:
                        continue
                    reference = template
                # dict模板检查各自model_id；普通向量则检查公共model_id，不匹配时continue跳过这一项。
                reference = np.asarray(reference, dtype=np.float32)
                if reference.shape != probe.shape or not np.isfinite(reference).all():
                    continue
                norm = float(np.linalg.norm(reference))
                if norm > 1e-8:
                    # np.dot对两个[D]向量求标量点积，clip限制浮点误差不越过[-1,1]。
                    scores.append(float(np.clip(np.dot(probe, reference / norm), -1.0, 1.0)))
            if not scores:
                return {**result, "status": "model_mismatch", "reason": "没有与当前模型匹配的有效声纹模板"}
            # sorted返回新列表，再[:2]取最高两个分数；只有一个有效模板时就用该分数。
            best = sorted(scores, reverse=True)[:2]
            score = sum(best) / len(best)
            result.update(score=round(score, 5), reference_count=len(scores))
            # 高阈值确认match，低阈值确认mismatch，中间uncertain；后者不会被当成主人身份。
            if score >= self.threshold:
                result.update(status="match", matched=True, reason="本轮声纹与主人模板相符")
            elif score <= self.reject_threshold:
                result.update(status="mismatch", reason="本轮声纹与主人模板不相符")
            else:
                result.update(status="uncertain", reason="声纹相似度位于不确定区间")
            return result
        except (TypeError, ValueError) as error:
            return {**result, "status": "error", "reason": f"声纹模板无效：{error}"}
