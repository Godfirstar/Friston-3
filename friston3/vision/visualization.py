# cv2为外部OpenCV库；本文件只把已有观测画到图像，不负责检测人脸。
import cv2


def draw_faces(display, observation, *, show_center=False, show_position=False):
    """在传入的画面副本上绘制人脸框，可选中心点或位置标签。"""
    # preview.main与AutoPhotoSkill.execute调用；display为HWC/BGR NumPy数组，observation为YuNet返回dict。
    # *后两个开关须按名字传；函数原地修改display并返回同一个对象，调用方应先frame.copy()。
    # NumPy的shape排列是(H,W,C)，OpenCV坐标排列却是(x,y)，因此解包高宽后不能互换。
    height, width = display.shape[:2]

    for face in observation["faces"]:
        # face为一个人脸dict；bbox中的x/y是左上角像素，w/h为宽高，不是两个角坐标。
        x, y, w, h = face["bbox"]
        # rectangle接左上与右下坐标；颜色是BGR，(0,255,0)为绿色，2为框线像素宽。
        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

        if show_center:
            # center_x/y为0~1比例，乘画面宽/高还原像素，int转换为OpenCV需要的整数坐标。
            # circle半径4像素，BGR红色(0,0,255)，thickness=-1表示填充圆形。
            cv2.circle(
                display,
                (
                    int(face["center_x"] * width),
                    int(face["center_y"] * height),
                ),
                4,
                (0, 0, 255),
                -1,
            )

        if show_position:
            # f字符串把观察器给出的horizontal/vertical标签拼成str，并非在此重新计算位置。
            position = f"{face['horizontal']}/{face['vertical']}"
            # putText在框上方写字；max(y-8,20)避免文字基线挤到窗口顶部之外。
            cv2.putText(
                display,
                position,
                (x, max(y - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    # 返回值不是另一个副本；与传入display共享同一像素数组，图像内容已被原地修改。
    return display
