"""Owner enrollment, current-utterance evidence and local exhibit controls.

Biometric matches select a conversational relationship, never administrator
rights. Only an explicit command entered on the local console changes profiles.
There is no inference of real names from visitors and no raw-media persistence.
"""
# 延迟求值类型标注；注解用于表达字段含义，不会自动校验运行时输入。
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
import tempfile
import threading
import time
from uuid import uuid4

from friston3.config import PROJECT_ROOT
# 下游两个项目类：人脸在vision/owner_face.py，声纹在stackchan_integration/owner_voice.py。
# 本文件负责把它们的证据绑定到本轮录音，并选择会话身份；不负责ASR文字转写。
from friston3.vision.owner_face import OwnerFaceRecognizer
from stackchan_integration.owner_voice import OwnerVoiceRecognizer


def _empty_evidence(reason="本轮没有可用观测"):
    # _Utterance默认值与Manager各降级分支调用；返回新证据dict，避免复用上一轮匹配结果。
    return {"status": "not_observed", "reason": reason, "score": None}


def _cosine(left, right):
    # enroll检查新样本是否与首样本一致；输入为已归一化的float列表，返回float。
    # zip逐项配对，生成器算乘积，sum累加；只有输入单位长度时，这个点积才等于余弦值。
    if len(left) != len(right):
        return -1.0
    return sum(a * b for a, b in zip(left, right))


class OwnerProfile:
    """One atomic, local JSON file containing only enrollment vectors."""

    def __init__(self):
        # Manager或main的模板删除入口创建本类；path为Path，data为待读/待写的模板dict。
        # 这里只读JSON模板，不打开摄像头、加载神经网络或识别当前说话者。
        self.path = PROJECT_ROOT / "data" / "runtime" / "owner_identity.json"
        self.data = {"version": 1, "owner_name": "可露希尔"}
        self.error = ""
        try:
            if self.path.exists():
                if self.path.stat().st_size > 1_000_000:
                    raise ValueError("主人模板文件大小异常")
                # Path.read_text读str，json.loads解码为Python对象；utf-8-sig兼容BOM。
                data = json.loads(self.path.read_text(encoding="utf-8-sig"))
                if not isinstance(data, dict) or data.get("version") != 1:
                    raise ValueError("主人模板版本或格式不支持")
                for name in ("face", "voice"):
                    if name in data:
                        self._validate(data[name])
                self.data = data
        except (OSError, ValueError, TypeError) as exc:
            self.error = str(exc)

    @staticmethod
    def _validate(record):
        # __init__/save调用；输入一种模态的dict，合格时隐式返回None，不合格抛ValueError。
        # @staticmethod不自动传self，因为校验只依赖record，不读实例属性。
        if (not isinstance(record, dict) or not isinstance(record.get("model_id"), str)
                or not record["model_id"] or len(record["model_id"]) > 120):
            raise ValueError("模板缺少有效的模型标识")
        # vectors预期为list[list[float]]；每行一个样本，dimension记录第一行维数用于后续比较。
        vectors = record.get("templates")
        if not isinstance(vectors, list) or not 3 <= len(vectors) <= 8:
            raise ValueError("每种主人模板需要 3..8 个有效样本")
        dimension = None
        for vector in vectors:
            if not isinstance(vector, list) or not 32 <= len(vector) <= 4096:
                raise ValueError("模板向量维度异常")
            if dimension is not None and len(vector) != dimension:
                raise ValueError("模板向量维度不一致")
            dimension = len(vector)
            # any只要发现一项异常就为True；type精确比较排除bool，isfinite排除NaN/无穷值。
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in vector):
                raise ValueError("模板向量包含无效数值")
            # sum(v*v)是L2范数的平方，应约等于1；本函数检查而不悄悄重新归一化已保存模板。
            if not 0.98 <= sum(v * v for v in vector) <= 1.02:
                raise ValueError("模板向量未归一化")

    def get(self, modality):
        # Manager读取face/voice模板时调用；返回深拷贝dict或{}，防止调用方改坏self.data的嵌套列表。
        return deepcopy(self.data.get(modality, {})) if not self.error else {}

    def save(self, updates):
        # Manager.enroll收齐样本后调用；updates为{face/voice:模板记录}dict，完成后隐式返回None。
        # A damaged profile is never silently overwritten by a partial enrollment.
        if self.error:
            raise ValueError(f"旧模板损坏：{self.error}；先在电脑端 /owner forget 后重新录入")
        # 先复制现有资料，再逐模态校验/合并，未更新的另一种模板会保留。
        data = deepcopy(self.data)
        for modality, record in updates.items():
            if modality not in ("face", "voice"):
                raise ValueError("未知模板种类")
            self._validate(record)
            data[modality] = record
        # datetime/timezone来自标准库，生成带UTC时区的保存时间文本，而非模型特征的一部分。
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # mkstemp返回整数文件描述符与路径字符串；在目标同目录写临时文件，完备后再原子替换。
        descriptor, temporary = tempfile.mkstemp(prefix=".owner-", suffix=".tmp", dir=self.path.parent)
        try:
            # fdopen把描述符包装成文本流；with负责关闭，json.dump把dict写入流而非返回str。
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
                # flush送出Python缓冲，fsync要求操作系统同步文件；然后replace发布完整的新模板。
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            # 无论成功失败清理仍存在的临时文件；成功replace后原临时路径已不存在。
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.data = data

    def forget(self):
        # main(--forget-owner)/console(/owner forget)明确调用；删除模板JSON并重置内存字段。
        # Path.unlink只删该文件，missing_ok=True允许本来就没录入；不删除模型权重或NPC记忆库。
        self.path.unlink(missing_ok=True)
        self.data = {"version": 1, "owner_name": "可露希尔"}
        self.error = ""


@dataclass
class _Utterance:
    # 一段录音的“票据”：dataclass按字段生成__init__，pcm为bytes，sample_rate为int。
    # default_factory每次创建新对象，不能让所有回合共享同一dict或Event。
    # face/voice保存各线程结果；Event.set发完成信号，主线程用wait/is_set读取进度。
    pcm: bytes
    sample_rate: int
    started: float = field(default_factory=time.monotonic)
    face: dict = field(default_factory=_empty_evidence)
    voice: dict = field(default_factory=_empty_evidence)
    face_done: threading.Event = field(default_factory=threading.Event)
    voice_done: threading.Event = field(default_factory=threading.Event)


class OwnerIdentityManager:
    """Bind new evidence to each utterance; never keep a previous owner match."""

    def __init__(self, camera=None, device=None):
        # main.main创建；camera/device为已共享的设备对象，不在这里再打开一条串口连接。
        # profile读取本地模板；识别器构造只保存配置，warmup或enroll才触发实际模型加载。
        self.camera, self.device = camera, device
        self.profile = OwnerProfile()
        self.face, self.voice = OwnerFaceRecognizer(), OwnerVoiceRecognizer()
        self.wait_budget = 0.65
        self.camera_budget = 2.5
        self.guest_ttl = 45
        self.mode = "auto"
        self._mode_until = 0.0
        self._owner_once = False
        self._once_until = 0.0
        # _ticket指向当前录音票据；_busy是各模态是否仍在工作，_retry_after为重试截止时刻。
        self._ticket = None
        self._busy = {"face": False, "voice": False}
        self._retry_after = {"face": 0.0, "voice": 0.0}
        # Lock保护busy/cooldown与线程结果提交；不是把整个对话循环锁住。
        self._lock = threading.Lock()
        self._closed = False
        self._visitor_id = ""
        self._last_visitor = 0.0
        self._last_role = "unknown"
        self.last_context = {}

    def warmup(self):
        """Only previously enrolled modalities are loaded; no network operations."""
        # main在开始接待前调用；for解包(name,识别器实例)，有本地模板才调用对应preload。
        # 不会因为启动程序就自动替用户下载/录入缺失模态；函数无显式返回值。
        for name, recognizer in (("face", self.face), ("voice", self.voice)):
            if not self.profile.get(name):
                continue
            state = recognizer.preload()
            print(f"[身份] {name}：{state['status']} {state.get('reason', '')}")

    def invalidate(self):
        # Existing workers can finish cleanup, but their ticket will never be used.
        # begin_utterance/set_mode/context_for/close调用；只断开当前票据引用，不强杀后台线程。
        self._ticket = None

    def _start_probe(self, ticket, name, action):
        # begin_utterance为face/voice各调用一次；ticket是_Utterance，name为模态str，action为无参函数。
        # getattr是内置动态取属性，name+'_done'得到face_done或voice_done的Event对象。
        event = getattr(ticket, name + "_done")
        with self._lock:
            # 忙碌/冷却/关闭时为本票据写“不观测”，event.set通知等待者完成，避免沿用旧结果。
            if self._closed or self._busy[name] or time.monotonic() < self._retry_after[name]:
                setattr(ticket, name, _empty_evidence("识别忙碌或设备正在恢复；本轮不沿用旧身份"))
                event.set()
                return
            self._busy[name] = True

        def run():
            # 嵌套函数被Thread调用，闭包保留本次ticket/name/action；它不读取后来新建的_ticket。
            try:
                # action是传入lambda，调用后才真正拍帧/提取声纹；result应为证据dict。
                result = action()
                if not isinstance(result, dict):
                    raise ValueError("身份识别返回格式异常")
            except Exception as exc:
                result = {"status": "error", "reason": f"{type(exc).__name__}: {exc}", "score": None}
            # Commit result and cooldown before another utterance can submit
            # work. Late results still belong only to their abandoned ticket.
            with self._lock:
                # setattr按字符串写本票据face或voice字段；迟到结果仍只归旧票据，不污染新一轮。
                if result.get("status") in ("error", "unavailable"):
                    self._retry_after[name] = time.monotonic() + 30
                setattr(ticket, name, result)
                self._busy[name] = False
            event.set()

        # target=run传函数对象，不写run()；start启动新线程；daemon允许主程序退出时不等该线程。
        # 这里用Thread+Event组合，没有Future对象，也没有阻塞等待模型返回。
        threading.Thread(target=run, name=f"FristonOwner-{name}", daemon=True).start()

    def _frame(self, *, timeout=None):
        # face探测与enroll调用；返回camera.snapshot_for_identity取得的BGR图像ndarray。
        # 该相机方法定义在hardware/camera.py，复用已有设备调度；*令timeout只能按名称传入。
        if self.camera is None:
            raise RuntimeError("没有接入摄像头")
        return self.camera.snapshot_for_identity(timeout=self.camera_budget if timeout is None else timeout)

    def begin_utterance(self, pcm, sample_rate=16000):
        """Called after MIC transfer, before ASR. No audio is written to disk."""
        # main把它赋给StackChanSpeechInput.audio_observer；voice_input.py在ASR前调用此回调。
        # 输入pcm为本段录音bytes，返回None；启动后台识别，使识别与ASR文字转写重叠执行。
        self.invalidate()
        self._expire_mode()
        # 先作废旧票据，再为本段数据创建全新票据；对象引用用于之后核对是否同一段录音。
        ticket = _Utterance(pcm=pcm, sample_rate=sample_rate)
        self._ticket = ticket
        face_record, voice_record = self.profile.get("face"), self.profile.get("voice")
        if self.mode != "auto" or self._owner_once:
            # 显式演示模式不必运行生物识别，两个Event都标为完成后直接返回。
            ticket.face_done.set()
            ticket.voice_done.set()
            return
        if voice_record and self.voice.status().get("loaded"):
            # lambda延迟调用OwnerVoiceRecognizer.observe；传的是已录入模板及其模型ID。
            self._start_probe(ticket, "voice", lambda: self.voice.observe(
                pcm, voice_record["templates"], sample_rate, model_id=voice_record["model_id"]))
        else:
            ticket.voice = _empty_evidence("声纹未录入或未就绪")
            ticket.voice_done.set()
        if (face_record and self.camera is not None
                and self.face.status()["status"] == "ready"):
            # StackChan owns I/O in one scheduler; ASR overlaps this bounded
            # atomic capture/release job without opening another connection.
            self._start_probe(ticket, "face", lambda: self.face.observe(
                self._frame(), face_record["templates"], model_id=face_record["model_id"]))
        else:
            ticket.face = _empty_evidence("人脸未录入或未就绪")
            ticket.face_done.set()

    def new_visitor(self):
        # context_for/set_mode内部调用；uuid4生成新访客会话标识，返回str，不推断访客真实姓名。
        self._visitor_id = "visitor:" + uuid4().hex[:12]
        self._last_visitor = time.monotonic()
        return self._visitor_id

    def handover_visitor(self):
        """An explicit handover also ends any pending owner demonstration."""
        # console.new_visitor调用；保留显式visitor模式，否则回auto，再由set_mode生成新会话ID。
        self.set_mode("visitor" if self.mode == "visitor" else "auto")
        return self._visitor_id

    def set_mode(self, mode):
        # console.handle_identity_command与handover_visitor调用；输入模式str，无显式返回值。
        # 这是本地控制台演示设置；不是把一次识别结果永久保存成主人身份。
        if mode not in ("auto", "owner", "visitor", "once"):
            raise ValueError("模式只能是 auto、owner、visitor、once")
        self.invalidate()
        # ==产生bool；once只影响一轮并带过期时刻，self.mode本身仍设为auto。
        self._owner_once = mode == "once"
        self._once_until = time.monotonic() + 30 if mode == "once" else 0
        self.mode = "auto" if mode == "once" else mode
        # Manual owner mode is an explicit local exhibition session, not a
        # sticky recognition result. The timeout cannot be extended by speech.
        self._mode_until = time.monotonic() + 120 if mode == "owner" else 0
        self.new_visitor()

    def _expire_mode(self):
        # 取上下文/启动识别/展示状态前调用；monotonic是单调时钟，适合算超时，不是日历时间。
        if self.mode == "owner" and time.monotonic() >= self._mode_until:
            self.mode = "auto"
        if self._owner_once and time.monotonic() >= self._once_until:
            self._owner_once = False

    def context_for(self, *, input_source, pcm=None, sample_rate=16000):
        # main在取得一轮输入后调用；input_source为text/speech，pcm为本段录音bytes或None。
        # 返回身份上下文dict给Agent/NpcTalkSkill；人脸在场证据与声音说话者证据分别保存。
        started = time.monotonic()
        self._expire_mode()
        # 多重赋值先读右侧旧值，再写左侧；保存本轮一次演示标志并立即消费它。
        owner_once, self._owner_once = self._owner_once, False
        face, voice = _empty_evidence(), _empty_evidence()
        role, source, reason = "unknown", "unverified", "本轮身份没有确认"
        # local为bool；只从输入来源判断控制台权限，不从声纹match推导管理员权限。
        local = input_source == "text"
        if self.mode == "visitor":
            role, source, reason = "visitor", "local_session", "电脑端指定接待来访者"
        elif local:
            role, source, reason = "owner", "local_console", "主人在自己的电脑控制台输入"
        elif self.mode == "owner" or owner_once:
            role, source, reason = "owner", "local_session", "电脑端明确指定的主人演示回合"
        elif pcm:
            # 身份模式/本地输入都不适用时，才消费这一段录音的后台识别结果。
            ticket = self._ticket
            # is not比较对象身份而非bytes内容：两次独立采集即使字节相同，也不能复用旧票据。
            if ticket is None or ticket.pcm is not pcm or ticket.sample_rate != sample_rate:
                # No reuse across independently captured or rejected recordings.
                self.begin_utterance(pcm, sample_rate)
                ticket = self._ticket
            # 两个Event共用一个deadline，第二个只能等剩余预算，并非各等0.65秒。
            deadline = started + self.wait_budget
            for event in (ticket.voice_done, ticket.face_done):
                event.wait(timeout=max(0, deadline - time.monotonic()))
            # is_set判断线程是否已提交结果；完成才深拷贝证据，超时生成本轮缺失证据。
            face = deepcopy(ticket.face) if ticket.face_done.is_set() else _empty_evidence("本轮人脸观测超时")
            voice = deepcopy(ticket.voice) if ticket.voice_done.is_set() else _empty_evidence("本轮声纹观测超时")
            # 声纹match可选择主人会话；face只能补充是否看见主人，不能单独把声音归给主人。
            if voice.get("status") == "match":
                role, source, reason = "owner", "voice", "本轮声音与主人模板匹配"
                if face.get("owner_visible"):
                    source = "face_voice"
            elif voice.get("status") == "mismatch":
                role, source, reason = "visitor", "voice", "本轮声音与主人模板不匹配"
            elif face.get("owner_visible"):
                reason = "看见主人，但无法确认当前声音是否来自主人"
        if role == "owner":
            # 主人共享固定profile_id；访客超时、从主人切换或尚无ID时另开会话。
            profile_id = "owner"
        else:
            if (not self._visitor_id or started - self._last_visitor > self.guest_ttl
                    or self._last_role == "owner"):
                self.new_visitor()
            profile_id = self._visitor_id
            self._last_visitor = started
        self._last_role = role
        # context只含身份字段、分数和本轮证据，不含录音/图像/新特征向量。
        # wait_ms统计context_for额外等待，后台识别可能早在ASR前启动，二者耗时不能简单相加。
        context = {
            "role": role, "profile_id": profile_id,
            "display_name": "可露希尔" if role == "owner" else "来访者",
            "source": source, "authenticated": role == "owner",
            "local_operator": local and role == "owner",
            "owner_visible": bool(face.get("owner_visible")), "reason": reason,
            "scores": {"face": face.get("score"), "voice": voice.get("score")},
            "observations": {"face": face, "voice": voice},
            "wait_ms": round((time.monotonic() - started) * 1000, 1),
        }
        # 保留只读诊断快照，再作废票据；返回给上层的dict与last_context不共享嵌套对象。
        self.last_context = deepcopy(context)
        self.invalidate()
        return context

    def status_text(self):
        # main(--owner-status)/console身份状态命令调用；返回多行str，不启动新识别或录入。
        # lines是字符串list，append逐条加入；最后用换行join，保持显示与数据返回的区别。
        self._expire_mode()
        lines = [f"身份模式：{self.mode}；声音身份由本轮声纹确认，人脸仅提供可见性",
                 f"主人模板：{self.profile.path}"]
        if self._owner_once:
            lines.append(f"下一轮主人演示待用，{max(0, self._once_until-time.monotonic()):.0f} 秒后失效")
        elif self.mode == "owner":
            lines.append(f"主人演示还剩 {max(0, self._mode_until-time.monotonic()):.0f} 秒")
        if self.profile.error:
            lines.append(f"模板读取失败：{self.profile.error}")
        for name, recognizer in (("face", self.face), ("voice", self.voice)):
            record, state = self.profile.get(name), recognizer.status()
            lines.append(f"{name}：{len(record.get('templates', []))} 个样本；{state['status']} {state.get('reason', '')}")
        if self.last_context:
            lines.append("上一轮身份依据：" + json.dumps(self.last_context, ensure_ascii=False, indent=2))
        return "\n".join(lines)

    def enroll(self, modality="both"):
        # main(--enroll-owner)/console(/owner enroll)显式调用；输入face/voice/both，返回成功bool。
        # 此路径会交互等待input、采样设备并保存模板；它不是运行每轮身份确认时自动执行的操作。
        if modality not in ("face", "voice", "both"):
            raise ValueError("录入种类只能是 face、voice、both")
        if self.profile.error:
            raise ValueError(f"旧模板读取失败：{self.profile.error}；先 /owner forget 再录入")
        # Wait for already submitted work before touching either device/model.
        self.invalidate()
        with self._lock:
            # 当前实现检测busy后抛错让用户稍后重试，不是在这里sleep等待所有识别线程结束。
            if any(self._busy.values()):
                raise RuntimeError("上一轮身份识别尚在释放资源；稍候再录入")
        # updates为本次准备提交的模板dict；先在内存收齐，任一模态失败时旧文件仍不修改。
        updates = {}
        for name in (("face", "voice") if modality == "both" else (modality,)):
            # recognizer按模态指向已有识别器；preload加载模型，extract才运行一帧/一段音频。
            recognizer = self.face if name == "face" else self.voice
            state = recognizer.preload()
            if state.get("status") not in ("ready", "ok"):
                raise RuntimeError(state.get("reason", f"{name} 模型未就绪"))
            count = 5 if name == "face" else 3
            # samples是list[list[float]]，每项一个单位特征；model_id记录这些向量来自哪个权重。
            samples = []
            model_id = ""
            print(f"[主人录入] {name} 需要 {count} 个样本；只让你本人进入镜头/说话。输入 q 可取消。")
            if name == "face":
                print("依次正面、微向左、微向右、稍远、稍近；保持整张脸清晰。建议穿好妆造后录入。")
            else:
                print("每次轻触屏幕后正常说 3–6 秒，三次用不同句子；保持现场安静。")
            attempts = 0
            while len(samples) < count and attempts < count * 4:
                # 内置input阻塞读控制台字符串；strip/lower为str方法，统一取消命令格式。
                if input(f"[{name} {len(samples)+1}/{count}] 就位后按 Enter：").strip().lower() == "q":
                    print("已取消；原模板保留。")
                    return False
                attempts += 1
                if name == "face":
                    frame = self._frame(timeout=8.0)
                    extracted = recognizer.extract(frame)
                    # del移除本地名字引用，不保存原图；不是擦写已有文件或保证物理内存立即清零。
                    del frame
                else:
                    if self.device is None:
                        raise RuntimeError("声纹录入需要栈酱麦克风")
                    # device是StackChanFaceEvents共享对象；record_pcm返回内存bytes，再交声纹extract。
                    pcm = self.device.record_pcm(mode="touch", wait_ms=10000, max_ms=8000,
                                                 silence_ms=900, threshold=350)
                    extracted = recognizer.extract(pcm, sample_rate=16000)
                    del pcm
                if extracted.get("status") != "ok":
                    print("未采用：" + extracted.get("reason", extracted.get("status", "未知错误")))
                    continue
                # extracted是识别器结果dict；embedding为已归一化列表，成功才加入samples。
                vector = extracted["embedding"]
                current_model = extracted["model_id"]
                if model_id and current_model != model_id:
                    raise RuntimeError("录入期间模型发生变化，请重新开始")
                # All samples must resemble the first; cannot enroll several
                # different people by taking a maximum over unrelated templates.
                threshold = self.face.threshold if name == "face" else self.voice.threshold
                if samples and _cosine(samples[0], vector) < threshold:
                    print("未采用：与第一个样本不够一致，请保持同一人、清晰正面/正常音量重试。")
                    continue
                model_id = current_model
                samples.append(vector)
                print(f"已采用 {len(samples)}/{count}")
            if len(samples) < count:
                raise RuntimeError("有效样本不足；旧模板未修改。改善光线/距离/收音后重新录入。")
            updates[name] = {"model_id": model_id, "templates": samples}
        # 所有要求模态收齐后一次调用OwnerProfile.save，原子替换模板文件；只存向量与模型ID。
        self.profile.save(updates)
        print("主人模板已更新，只保存特征向量；原始照片与录音未写入文件。")
        return True

    def close(self):
        # main资源清理调用；禁止再提交探测并作废当前票据，不强杀正在结束的后台线程。
        self._closed = True
        self.invalidate()
