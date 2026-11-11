# cv2 是外部库 OpenCV 的 Python 模块，负责绘图/显示窗口；不会因 import 自动打开相机。
import cv2

# 这些名字由项目文件提供；下面 main() 才真正创建对象并开始预览。
from friston3.hardware.camera import create_camera
from friston3.vision.yunet_observer import YuNetObserver
from friston3.vision.visualization import draw_faces

from stackchan_integration import StackChanFaceEvents


def main():
    # 独立入口：python preview.py。不是 main.py 的接待循环，不经过 Agent/Planner。
    # 下游：共享StackChan连接 -> camera工厂 -> YuNet观察 -> 可视化；函数没有显式返回业务结果。
    # 预览单独运行，由同一个管理器自动发现连接。
    # StackChanFaceEvents 类由 stackchan_integration/face_events.py 定义；Class() 创建实例。
    face = StackChanFaceEvents()
    # camera 先用 None 占位，使初始化失败时 finally 能判断是否已经有相机需要释放。
    camera = None
    try:
        print("正在自动发现栈酱，Ctrl+C 可退出。")
        # wait_ready 是设备对象方法；每轮等至多1秒，False时继续，Ctrl+C可抛KeyboardInterrupt退出。
        while not face.wait_ready(timeout=1.0):
            pass
        # camera 来自 hardware/camera.py 的工厂函数，内部返回 StackChanCamera(face)。
        camera = create_camera(face)
        # observer 为 vision/yunet_observer.py 的模型观察器；初始化会加载本地YuNet模型。
        observer = YuNetObserver()
        if not camera.open():
            print("无法打开栈酱摄像头")
            return

        print("Friston-3 栈酱视觉预览")
        print("Q: 退出 | S: 拍照 | O: 输出观测")
        while True:
            # frame 是 camera.read() 返回的NumPy数组，通常uint8、HWC排列、BGR颜色顺序；失败为None。
            frame = camera.read()
            if frame is None:
                print("读取视频帧失败")
                break

            # observation 是人脸框/归一化中心/得分等组成的dict，不是图像张量本身。
            observation = observer.observe(frame)
            # copy 复制像素数组，避免后面的原地绘图改变真正要保存的原始frame。
            display = frame.copy()
            # shape 是NumPy数组尺寸tuple(H,W,C)；[:2]切出高宽，再用多重赋值分别绑定局部变量。
            height, width = display.shape[:2]

            # 画出人脸位置分类的参考线。
            for ratio in (0.4, 0.6):
                # OpenCV坐标为(x,y)，左上角原点，x向右/y向下；比例乘像素高后int取整数坐标。
                y = int(height * ratio)
                # line 原地修改display；BGR=(255,255,0)是青色，末尾1是线宽像素。
                cv2.line(display, (0, y), (width, y), (255, 255, 0), 1)

            # 项目vision/visualization.py函数在同一display上叠框；关键字参数打开位置文字。
            draw_faces(display, observation, show_position=True)

            # count 为观察器返回的人脸数量int；putText依次接图像、文字、坐标、字体、比例、颜色、线宽。
            count = observation["face_count"]
            cv2.putText(
                display,
                f"Faces: {count} | S: Save | O: Observe | Q: Quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            # imshow 刷新窗口；waitKey 不仅读按键，也让OpenCV处理窗口事件，否则画面可能不刷新。
            cv2.imshow("Friston-3 Preview", display)

            # waitKey(1) 至少等待约1毫秒；按位与0xFF保留低8位键码，ord('q')取字符的整数编码。
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                # 保存未画框的frame；save_frame返回成功/路径dict，这里仅打印，不走摄影技能。
                print("[Photo]", camera.save_frame(frame))
            elif key == ord("o"):
                print("[Observation]", observation)
    finally:
        # 正常退出、break或异常离开try都会执行finally；先放相机，再关共享设备和OpenCV窗口。
        if camera is not None:
            camera.close()
        face.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 直接运行本文件时条件成立；被别处import时只定义main，不自动启动预览。
    main()
