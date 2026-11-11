# cv2为外部OpenCV库；其余两个import来自项目vision目录，分别负责构图状态与叠画人脸。
import cv2

from friston3.vision.composition import CompositionController
from friston3.vision.visualization import draw_faces


class AutoPhotoSkill:
    """实时分析摄像头画面，构图稳定后倒计时拍照。"""

    # main._run_session创建并注册此实例；Planner读name/description，SkillManager调用execute。
    name = "auto_photo"
    description = "自动构图、倒计时拍照；可选整数countdown_seconds，1至10，默认3。"

    @staticmethod
    def argument_schema():
        # staticmethod不自动传self；Planner._schema_for_routes调用，返回模型参数的JSON Schema dict。
        # countdown_seconds不在required里，因此可省略；minimum/maximum是当前已有参数约定。
        return {"type": "object", "properties": {
            "countdown_seconds": {"type": "integer", "minimum": 1, "maximum": 10}},
            "required": [], "additionalProperties": False}

    @staticmethod
    def normalize_arguments(arguments, *, defaults=False):
        # Planner/SkillManager调用；arguments为dict，返回包含明确倒计时秒数的新dict。
        # set(dict)取键集合；差集非空表示有未知键。type(x)is int可排除bool伪装整数。
        if type(arguments) is not dict or set(arguments) - {"countdown_seconds"}:
            raise ValueError("自动拍照只接受 countdown_seconds")
        # dict.get缺键时默认3；这里的seconds为局部变量，不是已经开始流逝的计时器。
        seconds = arguments.get("countdown_seconds", 3)
        if type(seconds) is not int or not 1 <= seconds <= 10:
            raise ValueError("countdown_seconds 必须为 1 至 10 的整数")
        return {"countdown_seconds": seconds}

    def __init__(self, camera, observer):
        # main传入共享StackChanCamera与YuNetObserver；self.xxx把对象保存到实例，不复制硬件。
        self.camera = camera

        self.observer = observer

    def execute(self, countdown_seconds=3):
        # SkillManager.execute调用；本方法运行到保存/取消/失败后返回结果dict，期间会打开预览窗口。
        # 下游camera负责取帧/保存，observer负责人脸观测，controller决定当前阶段，职责分开。
        # 每次执行任务，都使用新的状态机。
        # controller为vision/composition.py的状态机实例；每次任务新建，避免沿用上次倒计时。
        controller = CompositionController(
            countdown_seconds=countdown_seconds
            )
        if not self.camera.open():
            return {
                "success": False,
                "message": "无法打开摄像头",
            }

        print("[AutoPhoto] 自动摄影已启动，按 Q 取消")
        # result先设默认取消结果；退出循环时若没成功保存，就保留或更新对应失败原因。
        result = {
            "success": False,
            "message": "用户取消自动拍照",
        }

        try:
            while True:
                # frame为HWC/BGR像素数组或None；相机失败只改结果再break，不拿空图继续检测。
                frame = self.camera.read()
                if frame is None:
                    result["message"] = "摄像头读取失败"
                    break

                # observation是YuNetObserver返回dict；state是CompositionController.update返回dict。
                # state含stage/hint/ready，ready才表示可以保存当前帧，不是人脸检测得分。
                observation = self.observer.observe(frame)
                state = controller.update(observation)

                # 在副本上绘制提示，保存照片时仍使用未标注的原始帧。
                # NumPy copy复制像素；后面OpenCV绘图原地修改display，原frame仍是干净照片。
                display = frame.copy()
                height, width = display.shape[:2]
                # rectangle依次接图、左上/右下(x,y)、BGR颜色、线宽；坐标由高宽比例转整数。
                cv2.rectangle(
                    display,
                    (int(width * 0.40), int(height * 0.30)),
                    (int(width * 0.60), int(height * 0.60)),
                    (0, 200, 0),
                    1,
                )

                # 项目visualization.draw_faces原地叠框，show_center为关键字参数，要求再画中心点。
                draw_faces(display, observation, show_center=True)

                # putText属于OpenCV：在像素坐标写文字；下面三次分别写阶段、调整提示、取消按键。
                cv2.putText(
                    display,
                    state["stage"],
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display,
                    state["hint"],
                    (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display,
                    "Q: cancel",
                    (12, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("Friston-3 Auto Photo", display)
                # waitKey处理窗口事件并取得按键；&0xFF保留低8位键码，ord('q')是字母q的整数码。
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                if state["ready"]:
                    # ready后保存的是刚通过最终构图检查的原始frame，不另取可能改变位置的新图。
                    saved = self.camera.save_frame(frame)
                    # saved由camera.save_frame返回，成功时有path，失败时有error；映射成技能统一结果。
                    if saved["success"]:
                        result = {
                            "success": True,
                            "path": saved["path"],
                            "message": f"自动拍照成功：{saved['path']}",
                        }
                    else:
                        result = {
                            "success": False,
                            "message": saved["error"],
                        }
                    break

        except Exception as error:
            result = {
                "success": False,
                "message": f"自动拍照异常：{error}",
            }
        finally:
            # 取消/失败/成功都会收尾；关闭相机会话并销毁窗口，之后仍可继续NPC主程序。
            self.camera.close()
            cv2.destroyAllWindows()

        # 返回路径：SkillManager附executed_skill -> Agent -> main显示并记录真实拍照结果。
        return result
