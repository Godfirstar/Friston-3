
# math/time 都是Python标准库；此模块只处理观察dict和时间，不读摄像头、不运行神经网络。
import math
import time


class CompositionController:
    # AutoPhotoSkill.execute每次任务创建本类。它是跨帧状态机：观察相同也可能因稳定时长不同而返回不同阶段。
    def __init__(
        self,
        stable_seconds=1.5,
        countdown_seconds=3.0,
        grace_seconds=0.35,
        alpha=0.35
    ):
        # self为新实例；四个实参分别是连续稳定时长、倒计时、短暂丢脸宽限、EMA新观测权重。
        # 默认数值是当前已有行为，保存在实例属性后由update读取；这里没有启动计时线程。
        self.stable_seconds = stable_seconds
        self.countdown_seconds = countdown_seconds
        self.grace_seconds = grace_seconds
        self.alpha = alpha

        if not 0 < alpha <= 1:
            # Python支持连写比较0<alpha<=1；not取反；不符合约定时抛ValueError给构造调用方。
            raise ValueError("alpha 必须在 (0, 1] 内")

        # 点号找到下方本类reset方法，()调用它，统一初始化时间戳与滤波状态。
        self.reset()

    def reset(self):
        """重置状态机和滤波器。"""
        # __init__与update调用；None表示“尚未开始/没有缓存”，不是时间0。
        # stable_since/countdown_since/missing_since分别记录稳定、倒计时、丢脸的开始时刻。
        self.stable_since = None
        self.countdown_since = None
        self.missing_since = None
        self.smoothed = None

    def _result(self, stage, hint, ready=False):
        # update各分支共用的返回格式：stage/hint为str，ready为bool；调用方据ready决定是否保存。
        return {
            "stage": stage,
            "hint": hint,
            "ready": ready
        }

    def _judge(self, cx, cy, ratio, margin=0.0):
        """根据人脸中心和尺寸判断构图。"""
        # evaluate/update内部调用；cx/cy是0~1的归一化中心，ratio是人脸高度/画面高度。
        # 返回(bool,提示str)二元tuple；margin为当前调用放宽范围的参数，不是时间或像素值。
        # 以下顺序优先返回第一个不满足条件；“Move right”指画面x增大方向，不是机器人右臂。

        if cx < 0.40 - margin:
            return False, "Move face right"

        if cx > 0.60 + margin:
            return False, "Move face left"

        if cy < 0.30 - margin:
            return False, "Move face down"

        if cy > 0.60 + margin:
            return False, "Move face up"

        if ratio < 0.18 - margin:
            return False, "Move closer"

        if ratio > 0.45 + margin:
            return False, "Move back"

        return True, "Composition OK"

    def evaluate(self, observation):
        """保留 v0.4 的单帧构图评价接口。"""
        # 兼容入口：输入YuNetObserver.observe的dict，返回当前单帧(bool,hint)，不更新计时状态。
        # 当前AutoPhotoSkill主路径使用下面update；此方法存在不代表仍被实时任务调用。

        # count来自观察dict的人脸数量int；先确认恰好1人，再访问faces[0]，避免空列表索引。
        count = observation["face_count"]

        if count == 0:
            return False, "No face detected"

        if count != 1:
            return False, "Multiple faces"

        # face是列表中唯一的人脸dict；bbox=[x,y,w,h]，下标3是高，不是右下角y坐标。
        face = observation["faces"][0]

        return self._judge(
            face["center_x"],
            face["center_y"],
            face["bbox"][3] / observation["frame_height"]
        )

    def update(self, observation, now=None):
        # AutoPhotoSkill每帧调用；observation为观测dict，now可传单调时间float，默认读取标准库时钟。
        # 返回stage/hint/ready字典；内部保存各时间戳与滤波状态，下一帧接着使用。
        # 条件表达式只在now为None时调用时钟；monotonic不随系统日期校时倒退，适合算持续时长。
        now = time.monotonic() if now is None else now

        count = observation["face_count"]

        # 1. 暂时丢失人脸：暂停，不立即清零
        if count == 0:
            # 仅已经进入稳定阶段时才允许短暂HOLD；首次丢失时记录开始时间而非每帧重写。
            if self.stable_since is not None:
                if self.missing_since is None:
                    self.missing_since = now

                # lost_time为本次连续丢脸秒数float；HOLD只暂停输出，恢复时再补偿计时起点。
                lost_time = now - self.missing_since

                if lost_time <= self.grace_seconds:
                    return self._result(
                        "HOLD",
                        "Face lost briefly; timer paused"
                    )

            # 超过宽限或尚未稳定时清空状态，下一帧必须重新累计稳定时间。
            self.reset()
            return self._result("ADJUST", "No face detected")

        # 2. 当前只支持单人拍照
        if count != 1:
            self.reset()
            return self._result("ADJUST", "Multiple faces")

        face = observation["faces"][0]

        # raw是三元素tuple(cx,cy,人脸高度比)，全部是比例，不受实际分辨率是320还是640直接影响。
        raw = (
            face["center_x"],
            face["center_y"],
            face["bbox"][3] / observation["frame_height"]
        )

        # 3. 原始观测超出宽松范围，立即重置
        # *raw在调用处展开三个位置实参，等价于_judge(raw[0],raw[1],raw[2],margin=0.06)。
        # raw_ok/hint由返回tuple拆出；先查原始观测，避免平滑把明显越界的当前帧掩盖。
        raw_ok, hint = self._judge(*raw, margin=0.06)

        if not raw_ok:
            self.reset()
            return self._result("ADJUST", hint)

        # 4. 如果刚从短暂丢失中恢复，补偿暂停时间
        if self.missing_since is not None:
            lost_time = now - self.missing_since

            if lost_time > self.grace_seconds:
                self.reset()
            else:
                if self.stable_since is not None:
                    # 把计时起点往后移丢失时长，让(now-起点)排除暂停时间，倒计时同理。
                    self.stable_since += lost_time

                if self.countdown_since is not None:
                    self.countdown_since += lost_time

                self.missing_since = None

        # 5. EMA 平滑人脸中心及尺寸
        if self.smoothed is None:
            # 首帧直接用raw初始化；没有历史值时不做加权。
            self.smoothed = raw
        else:
            # zip把新旧三个数逐项配对；生成器计算EMA=alpha*当前+(1-alpha)*旧值，再收成tuple。
            # 这是固定公式滤波，不是训练神经网络，也没有反向传播或可学习参数。
            self.smoothed = tuple(
                self.alpha * current
                + (1 - self.alpha) * previous
                for current, previous in zip(raw, self.smoothed)
            )

        # 6. 用平滑值进行正式构图判断
        # 正式判断不传margin，采用默认0；失败即reset，防止跳动期间的稳定时间继续累计。
        ok, hint = self._judge(*self.smoothed)

        if not ok:
            self.reset()
            return self._result("ADJUST", hint)

        # 7. 构图连续稳定阶段
        if self.stable_since is None:
            self.stable_since = now

        # elapsed为连续有效构图已持续秒数；尚不足时返回STABILIZING，不开始倒计时。
        elapsed = now - self.stable_since

        if elapsed < self.stable_seconds:
            # remaining为稳定阶段剩余秒数；下面f字符串:.1f只影响显示保留1位小数，不改实际计时。
            remaining = self.stable_seconds - elapsed

            return self._result(
                "STABILIZING",
                f"Keep framing: {remaining:.1f}s"
            )

        # 8. 构图稳定，开始倒计时
        if self.countdown_since is None:
            # 第一次足够稳定才设倒计时起点；之后帧沿用同一个起点，不会每帧重置倒计时。
            self.countdown_since = now

        remaining = (
            self.countdown_seconds
            - (now - self.countdown_since)
        )

        if remaining > 0:
            # math.ceil向上取整，让剩余0.2秒仍显示1秒；真正READY条件看float remaining<=0。
            return self._result(
                "COUNTDOWN",
                f"Photo in {math.ceil(remaining)}s"
            )

        # 9. 最终观测依然有效，允许拍照
        return self._result(
            "READY",
            "Taking photo",
            ready=True
        )
