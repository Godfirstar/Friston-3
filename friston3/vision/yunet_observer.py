
# cv2是外部OpenCV；YuNet只做人脸检测，不给出身份名称，身份识别在owner_face等另外的模块。
import cv2

from friston3.config import YUNET_MODEL_PATH


class YuNetObserver:
    """基于 YuNet 的人脸观测器。"""

    def __init__(self):
        # main._run_session与preview.main创建实例时自动调用；self为当前观察器实例。
        # model_path为config.py提供的pathlib.Path对象，指向本地ONNX权重，不是远程URL。
        model_path = YUNET_MODEL_PATH

        # Path.is_file检查文件存在；stat().st_size读字节数，下面已有检查用于识别误放的LFS指针。
        if not model_path.is_file():
            raise FileNotFoundError(f"模型不存在: {model_path}")

        if model_path.stat().st_size < 200_000:
            raise ValueError("模型文件异常，可能下载到了 LFS 指针")

        # FaceDetectorYN.create为OpenCV提供的工厂，加载模型并返回检测器，保存为self.detector。
        # input_size按(W,H)给；score_threshold为候选阈值，nms_threshold用于重叠框抑制，top_k限制候选数。
        # 这些是当前已设定的检测参数；真正输入尺寸会在每次observe根据实际frame重新设置。
        self.detector = cv2.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=(320, 320),
            score_threshold=0.75,
            nms_threshold=0.3,
            top_k=5000
        )

    def observe(self, frame):
        # preview.main/AutoPhotoSkill.execute调用；frame为OpenCV BGR/HWC图像数组，返回观测dict。
        # 该接口仅推理，不生成动作、不保存照片，也不通过loss.backward训练模型。
        # shape[:2]取(H,W)，多重赋值分别绑定height/width，下面setInputSize需要反过来(W,H)。
        height, width = frame.shape[:2]

        # 让模型输入尺寸与实际画面匹配
        self.detector.setInputSize((width, height))

        # detect是OpenCV检测器方法，返回状态与检测数组；_是普通变量名，约定表示此处不用状态值。
        # detections为None表示没检测到；否则每行包含4个框坐标、5个关键点的10个数、1个得分。
        _, detections = self.detector.detect(frame)

        # faces是本函数新建的list，收集转换成项目格式的人脸dict；不直接把OpenCV数组传给上层。
        faces = []

        if detections is not None:
            for det in detections:
                # YuNet 前四项为 x, y, w, h
                # det为当前行数组；float取Python数，round取整，max/min把像素框裁到有效图像边界。
                # 先算右下x2/y2，再减左上得到裁剪后的w/h，不能把原始det[2]/det[3]当右下坐标。
                x1 = max(0, round(float(det[0])))
                y1 = max(0, round(float(det[1])))
                x2 = min(width, round(float(det[0] + det[2])))
                y2 = min(height, round(float(det[1] + det[3])))

                w = x2 - x1
                h = y2 - y1

                if w <= 0 or h <= 0:
                    # 裁剪后无面积的框不加入结果；continue只跳过当前检测行。
                    continue

                # 中心=(左上+宽高一半)，再除画面尺寸归一化到比例，用于不同分辨率下统一构图判断。
                cx = (x1 + w / 2) / width
                cy = (y1 + h / 2) / height

                # 多层条件表达式生成字符串标签；边界0.4/0.6中间为center，不是机器人关节坐标。
                horizontal = (
                    "left" if cx < 0.4
                    else "right" if cx > 0.6
                    else "center"
                )

                vertical = (
                    "up" if cy < 0.4
                    else "down" if cy > 0.6
                    else "center"
                )

                # append给列表加入一条dict；bbox用像素整数，center用比例，score是det[14]检测置信得分。
                # round(...,3)保留3位小数方便展示与后续使用，不是概率校准步骤。
                faces.append({
                    "bbox": [x1, y1, w, h],
                    "center_x": round(cx, 3),
                    "center_y": round(cy, 3),
                    "horizontal": horizontal,
                    "vertical": vertical,
                    "score": round(float(det[14]), 3)
                })

        # 返回单帧统一观察结构，CompositionController读它判断构图，draw_faces读它绘图。
        return {
            "frame_width": width,
            "frame_height": height,
            "face_count": len(faces),
            "faces": faces
        }
