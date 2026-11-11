"""Local YuNet/SFace evidence for one enrolled owner.

Only ``extract`` returns an embedding, for an explicit enrollment operation.
``observe`` reports who may be visible, never who is speaking. A camera frame
cannot establish which person produced a microphone recording.
"""
# 以下为标准库：sha256 标识权重文件，RLock 保护共享模型，perf_counter 计算耗时。
from hashlib import sha256
from threading import RLock
from time import perf_counter

from friston3.config import MODELS_DIR, YUNET_MODEL_PATH


SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"


class OwnerFaceRecognizer:
    """Lazy CPU inference; never download models or save camera images.

    Templates are lists of L2-normalized SFace vectors. Pass the ``model_id``
    saved during enrollment when observing, so changing weights requires
    re-enrollment. Thresholds are deployment settings, not calibrated accuracy.
    """

    def __init__(self):
        # OwnerIdentityManager.__init__ 创建本类；self 是当前识别器对象，只保存配置/空模型引用。
        # MODELS_DIR 是 config.py 的 Path；/ 拼接文件名。这里不下载模型，也不构造 OpenCV 网络。
        self.model_path = MODELS_DIR / SFACE_FILENAME
        # 这些是当前实现已有的质量/匹配参数，不是模型训练出来的概率或精度报告。
        self.threshold = .50
        self.ambiguity_margin = .05
        self.min_face_size = 64
        self.min_blur = 25
        self.detector_threshold = .85
        # detector 找脸和关键点，recognizer 把对齐后人脸编码成向量；两者承担不同任务。
        self._detector = None
        self._recognizer = None
        self._cv = None
        self._np = None
        self._load_error = ""
        self._load_attempted = False
        # RLock 是可重入锁：同一线程可再次进入，其他线程要等当前模型操作结束。
        self._lock = RLock()
        self.model_id = ""

    def status(self):
        """Inspect local readiness without starting inference or network I/O."""
        # OwnerIdentityManager.warmup/begin_utterance/status_text 调用；返回本地状态 dict。
        # reason 是已有加载错误 str；否则仅检查路径是否存在，不会读取图片或运行模型。
        reason = self._load_error
        if not reason:
            if not YUNET_MODEL_PATH.is_file():
                reason = f"缺少 YuNet 模型：{YUNET_MODEL_PATH}"
            elif not self.model_path.is_file():
                reason = f"缺少 SFace 模型：{self.model_path}"
        # ready 表示模型对象已加载；configured 只表示文件配置可用，不能据此认定已跑通推理。
        return {
            "status": "unavailable" if reason else "ready" if self._recognizer is not None else "configured",
            "reason": reason, "model_id": self.model_id,
            "threshold": self.threshold, "model_path": str(self.model_path),
        }

    def preload(self):
        """Load local weights once when the enrolled modality is prepared."""
        # OwnerIdentityManager.warmup/enroll 调用；加载一次并返回 status() 字典。
        # with 锁对象在退出代码块时自动释放锁，包括 _ensure_loaded 内发生异常的情况。
        with self._lock:
            self._ensure_loaded()
            return self.status()

    def _ensure_loaded(self):
        # 本类 preload/extract/observe 的内部加载函数；返回 bool，失败原因存 self._load_error。
        if self._recognizer is not None:
            return True
        if self._load_attempted:
            # 缓存失败尝试，避免缺文件时每一轮重复导入/加载；修复环境后需重建实例或重启。
            return False
        self._load_attempted = True
        try:
            # 延迟导入外部库：cv2 是 OpenCV Python 接口，numpy 提供数组与向量计算。
            import cv2
            import numpy as np

            # for 解包每个 (路径,最小字节数) 元组；stat().st_size 读取文件大小，识别明显占位文件。
            for path, minimum in ((YUNET_MODEL_PATH, 200_000), (self.model_path, 1_000_000)):
                if not path.is_file():
                    raise FileNotFoundError(f"缺少本地人脸模型：{path}")
                if path.stat().st_size < minimum:
                    raise ValueError(f"模型文件过小，可能是 Git LFS 指针：{path}")
            # OpenCV 工厂方法构造 YuNet 检测器；最后两项明确使用 OpenCV 后端与 CPU。
            # (320,320) 为初始输入尺寸，实际每帧会在 _prepare 中更新 setInputSize。
            detector = cv2.FaceDetectorYN.create(
                str(YUNET_MODEL_PATH), "", (320, 320),
                self.detector_threshold, .3, 5000,
                cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
            )
            # SFace 编码器同样从本地 ONNX 文件构造；它输出特征，不负责声源判断。
            recognizer = cv2.FaceRecognizerSF.create(
                str(self.model_path), "", cv2.dnn.DNN_BACKEND_OPENCV,
                cv2.dnn.DNN_TARGET_CPU,
            )
            # 分块读取权重计算SHA256；iter(无参数函数,b'') 在函数返回空字节时终止。
            # lambda 只在迭代时读取一块，不会一次把完整权重复制成大字节对象。
            digest = sha256()
            with self.model_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            # 模型ID随权重内容改变，用来防止旧模型录入的特征与新模型特征直接比较。
            self.model_id = "sface-sha256:" + digest.hexdigest()
            # 模块对象也可以保存为属性；后面 self._np / self._cv 直接复用已导入的库。
            self._cv, self._np = cv2, np
            self._detector, self._recognizer = detector, recognizer
            self._load_error = ""
            return True
        except Exception as exc:
            # Unavailable identity evidence must not take the dialogue loop down.
            self._load_error = f"人脸识别不可用（{type(exc).__name__}）：{exc}"
            return False

    def _result(self, status, reason="", **fields):
        # 本类各分支共用的结果构造器；status/reason 为 str，**fields 收集额外命名实参成dict。
        # result.update 原地合并，覆盖默认 face_count/score 等，返回结构一致的证据字典。
        result = {
            "status": status, "reason": reason, "model_id": self.model_id,
            "face_count": 0, "owner_visible": False, "score": None,
            "threshold": self.threshold, "quality": {},
        }
        result.update(fields)
        return result

    def _prepare(self, frame):
        # extract/observe 调用；输入 frame 是 BGR 图像 ndarray[H,W,3]，返回 (准备结果,错误)。
        # 准备成功为 ((图像,检测列表),None)，失败为 (None,证据dict)，由调用方解包判断。
        np, cv = self._np, self._cv
        # ndarray 是 NumPy 数组类；dtype 为元素类型，ndim 为轴数，shape 是各轴长度元组。
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3
                or min(frame.shape[:2]) < 32):
            return None, self._result("invalid_frame", "需要有效的 BGR uint8 摄像头画面。")
        # Bound CPU work for PC cameras; QVGA/VGA StackChan frames stay intact.
        height, width = frame.shape[:2]
        if max(height, width) > 640:
            # scale 是缩放比例float；resize 属于OpenCV，尺寸参数顺序为(width,height)。
            scale = 640 / max(height, width)
            frame = cv.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv.INTER_AREA)
        # 保证像素在内存中连续排布，便于C++接口读取；数据已连续时可能直接复用原数组。
        frame = np.ascontiguousarray(frame)
        height, width = frame.shape[:2]
        self._detector.setInputSize((width, height))
        # YuNet.detect 返回状态和检测矩阵；_ 接住但不使用首项，detections 每行是一张脸。
        _, detections = self._detector.detect(frame)
        if detections is None or not len(detections):
            return None, self._result("no_face", "画面内未检测到人脸。")
        # isfinite 逐元素检查有限数值，all 要求一整行都有效；copy 让结果不依赖原矩阵视图。
        detections = [row.copy() for row in detections if np.isfinite(row).all()]
        if not detections:
            return None, self._result("no_face", "未取得有效的人脸位置。")
        # list.sort 原地按人脸宽×高排序；lambda 是匿名取键函数，reverse=True 最大脸优先。
        detections.sort(key=lambda row: float(row[2] * row[3]), reverse=True)
        return (frame, detections), None

    def _feature(self, frame, detection):
        # extract/observe 对单张脸调用；返回 (128维单位向量或None,质量dict,原因str)。
        cv, np = self._cv, self._np
        height, width = frame.shape[:2]
        # detection 前4项为左上角x/y与宽/高；map(float,...)逐项转成Python浮点数再解包。
        x, y, face_w, face_h = map(float, detection[:4])
        # x0/y0/x1/y1 为落在有效图像范围内的整数裁剪边界，图像索引仍按[y,x]顺序。
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(width, int(x + face_w)), min(height, int(y + face_h))
        quality = {
            "bbox": [round(x), round(y), round(face_w), round(face_h)],
            "detection_score": round(float(detection[14]), 4),
            "face_min_pixels": round(min(face_w, face_h), 1),
        }
        if min(face_w, face_h) < self.min_face_size or x1 <= x0 or y1 <= y0:
            return None, quality, "人脸太小，请靠近摄像头。"
        # visible 是裁剪后面积占检测框面积的比例，不是“主人在场”标志。
        visible = (x1 - x0) * (y1 - y0) / max(face_w * face_h, 1)
        if visible < .95:
            return None, quality, "人脸在画面边缘被截断，请面向摄像头。"
        # 10个关键点坐标重排成[5,2]；[:,0] 取所有x，[:,1] 取所有y，& 逐元素组合条件。
        landmarks = detection[4:14].reshape(5, 2)
        if (not ((landmarks[:, 0] >= 0) & (landmarks[:, 0] < width)).all()
                or not ((landmarks[:, 1] >= 0) & (landmarks[:, 1] < height)).all()):
            return None, quality, "人脸关键点不完整，请面向摄像头。"
        # ndarray双轴切片先行(y)后列(x)；cvtColor 将裁剪区转灰度，Laplacian.var 衡量纹理清晰度。
        crop = cv.cvtColor(frame[y0:y1, x0:x1], cv.COLOR_BGR2GRAY)
        sharpness = float(cv.Laplacian(crop, cv.CV_64F).var())
        quality["sharpness"] = round(sharpness, 1)
        if sharpness < self.min_blur:
            return None, quality, "画面较模糊，请保持稳定并改善照明。"
        # OpenCV SFace.alignCrop 按关键点对齐裁剪；feature 才运行识别网络得到数值特征。
        aligned = self._recognizer.alignCrop(frame, detection)
        # asarray 转float32数组；reshape(-1) 展平为[D]，-1让NumPy推算长度；copy 取独立数据。
        embedding = np.asarray(self._recognizer.feature(aligned), dtype=np.float32).reshape(-1).copy()
        # L2范数 sqrt(sum(x_i^2))；除以它把向量归一到长度1，之后点积可作为余弦相似度。
        norm = float(np.linalg.norm(embedding))
        if len(embedding) != 128 or not np.isfinite(embedding).all() or norm < 1e-8:
            return None, quality, "未取得有效的 128 维 SFace 特征。"
        return embedding / norm, quality, ""

    def extract(self, frame):
        """Extract one owner enrollment sample; reject every multi-face frame."""
        # OwnerIdentityManager.enroll 调用；返回包含 embedding 列表的录入结果dict，无文件保存。
        # started 为float计时；锁覆盖整个检测和编码流程，防止同一个OpenCV模型被并发操作。
        started = perf_counter()
        with self._lock:
            if not self._ensure_loaded():
                return self._result("unavailable", self._load_error)
            try:
                prepared, error = self._prepare(frame)
                if error:
                    return error
                # prepared 是 _prepare 的二元组；录入只接受唯一人脸，避免模板混入不同人。
                image, detections = prepared
                if len(detections) != 1:
                    return self._result("ambiguous", "录入时画面中只能有你一个人。", face_count=len(detections))
                embedding, quality, reason = self._feature(image, detections[0])
                if embedding is None:
                    return self._result("low_quality", reason, face_count=1, quality=quality)
                # ndarray.tolist 转普通Python列表，OwnerProfile.save 才能通过JSON保存它。
                return self._result(
                    "ok", "单人样本已提取；未保存原始照片。", face_count=1,
                    quality=quality, embedding=embedding.tolist(),
                    elapsed_ms=round((perf_counter() - started) * 1000, 1),
                )
            except Exception as exc:
                return self._result("unavailable", f"人脸特征提取失败（{type(exc).__name__}）：{exc}")

    def observe(self, frame, templates, *, model_id=None):
        """Report visible owner evidence; never persist visitor embeddings.

        Scores within the fixed ambiguity margin below the threshold remain
        ambiguous. More than one matching face is also ambiguous. A single
        match among several faces establishes presence only; callers must use
        separate speaker evidence before choosing a personal dialogue context.
        """
        # OwnerIdentityManager.begin_utterance 的人脸线程调用；只返回证据dict，不返回访客特征。
        # frame 为当前BGR帧，templates为已录入向量列表；*要求model_id必须按名称传入。
        started = perf_counter()
        with self._lock:
            if templates is None:
                return self._result("not_enrolled", "尚未录入主人人脸。")
            if not self._ensure_loaded():
                return self._result("unavailable", self._load_error)
            if model_id and model_id != self.model_id:
                return self._result("unavailable", "人脸模型已更换，请重新录入主人人脸。")
            np = self._np
            try:
                # reference 为模板矩阵[N,128]；单条[128]输入会reshape成[1,128]统一处理。
                reference = np.asarray(templates, dtype=np.float32)
                if reference.size == 0:
                    return self._result("not_enrolled", "尚未录入主人人脸。")
                if reference.ndim == 1:
                    reference = reference.reshape(1, -1)
                if (reference.ndim != 2 or reference.shape[1] != 128
                        or not 1 <= len(reference) <= 32 or not np.isfinite(reference).all()):
                    return self._result("unavailable", "主人人脸模板格式异常，请重新录入。")
                # axis=1 按每一行计算范数；keepdims保留[N,1]形状，除法广播到每行128维。
                norms = np.linalg.norm(reference, axis=1, keepdims=True)
                if (norms < 1e-8).any():
                    return self._result("unavailable", "主人人脸模板无效，请重新录入。")
                reference = reference / norms
                prepared, error = self._prepare(frame)
                if error:
                    return error
                image, detections = prepared
                # matches保存(score,人脸索引)元组；qualities/failures分别保存质量字典与失败原因。
                matches = []
                qualities = []
                failures = []
                # Bound inference when a crowd fills the frame; retain the full
                # face count so an unprocessed face cannot imply a lone speaker.
                for index, detection in enumerate(detections[:4]):
                    # enumerate同时给出从0开始的编号；只处理最大4张脸，但结果仍报告完整检测数。
                    embedding, quality, reason = self._feature(image, detection)
                    quality["face_index"] = index
                    qualities.append(quality)
                    if embedding is None:
                        failures.append(reason)
                        continue
                    # [N,128] @ [128] -> [N]，求当前脸与各主人模板的余弦值；max取最佳模板。
                    # clip截回[-1,1]消除浮点误差；score是相似度，不是“属于主人”的概率。
                    score = float(np.clip(np.max(reference @ embedding), -1, 1))
                    matches.append((score, index))
                if not matches:
                    return self._result("low_quality", failures[0] if failures else "没有可比对的人脸。",
                                        face_count=len(detections), quality={"faces": qualities})
                # tuple列表按首元素score倒序排序；第一项给最高分与对应人脸索引。
                matches.sort(reverse=True)
                score, face_index = matches[0]
                # bool求和把True当1，得到达到阈值的人脸数量；不是达到阈值的模板数量。
                qualifying = sum(value >= self.threshold for value, _ in matches)
                if qualifying > 1:
                    status, reason, visible = "ambiguous", "多张人脸均接近主人模板，身份不确定。", False
                elif qualifying == 1:
                    visible = True
                    if len(detections) > 1:
                        status, reason = "ambiguous", "多人画面中有主人；仅能确认在场，不能确定说话人。"
                    else:
                        status, reason = "match", "画面中有一张人脸匹配主人模板。"
                elif score >= self.threshold - self.ambiguity_margin:
                    status, reason, visible = "ambiguous", "相似度接近阈值，暂不确认主人。", False
                else:
                    status, reason, visible = "no_match", "本帧未取得主人人脸匹配证据。", False
                # 多人画面即使有一个匹配，owner_visible可为True而status仍ambiguous，不能确定说话者。
                # 返回只含分数/可见性/质量/耗时，embedding不会流入聊天证据或持久化。
                return self._result(
                    status, reason, face_count=len(detections),
                    owner_visible=visible, score=round(score, 4),
                    matched_face_index=face_index if visible else None,
                    quality={"faces": qualities, "examined_count": len(qualities)},
                    elapsed_ms=round((perf_counter() - started) * 1000, 1),
                )
            except Exception as exc:
                return self._result("unavailable", f"人脸比对失败（{type(exc).__name__}）：{exc}")
