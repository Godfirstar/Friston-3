"""StackChan camera on the shared automatically selected device connection."""
# datetime 是标准库时间类型；cv2是外部OpenCV库；PHOTOS_DIR由项目config.py提供Path对象。
from datetime import datetime

import cv2

from friston3.config import PHOTOS_DIR


class StackChanCamera:
    """栈酱 GC0308 摄像头；USB/Wi-Fi 请求交给同一个设备工作线程。

    自动摄影/预览按需取帧，不在闲聊时后台推流。
    实时画面固定 QVGA，普通拍照固定 VGA。
    """

    def __init__(self, device):
        # create_camera 工厂调用；self为新相机实例，device为main/preview共享的StackChanFaceEvents。
        # 这里只保存连接对象，不自己打开另一串口；设备方法在stackchan_integration/face_events.py。
        self.save_dir = PHOTOS_DIR
        self.device = device
        # width 是预览/自动拍照取帧宽度；4:3画面对应320x240，capture单张另指定640。
        self.width = 320
        # epoch记录所用设备连接的代次；_opened为逻辑会话状态；_first_frame暂存打开时取到的帧。
        self._session_epoch = None
        self._opened = False
        self._first_frame = None
        self.last_error = ""

    def save_frame(self, frame):
        """将一帧图像保存为 JPG。"""
        # 调用方：本类capture、AutoPhotoSkill.execute、preview.main；返回success/path或error字典。
        # frame为BGR排列的HWC NumPy数组；本方法直接编码保存，不绘图也不再取新帧。
        if frame is None:
            return {"success": False, "error": "没有可保存的图像"}

        # pathlib.Path.mkdir建立目录；parents=True补父目录，exist_ok=True允许目录已经存在。
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # strftime将当前本地时间格式化为字符串；%f微秒帮助区分短时间内连续保存的照片。
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Path的/运算符拼接路径，而非数字除法；path仍是Path，调用OpenCV时用str转成路径文本。
        path = self.save_dir / f"photo_{timestamp}.jpg"
        # imwrite按.jpg扩展名编码，返回bool；JPEG是压缩文件格式，不等同于内存里的像素数组。
        if not cv2.imwrite(str(path), frame):
            return {"success": False, "error": "保存照片失败"}
        return {"success": True, "path": str(path)}

    def _get_frame(self, width):
        # open/read/capture内部调用；width为目标像素宽int，返回解码后的BGR NumPy数组。
        # numpy放函数内导入，是用到图像解码时才取得该依赖，并非在这里加载神经网络。
        import numpy as np

        # capture_jpeg属于共享设备对象，返回压缩JPEG bytes；USB/Wi-Fi传输由设备层负责。
        jpeg = self.device.capture_jpeg(width=width)
        # np.frombuffer把bytes看作一维uint8序列；cv2.imdecode才把压缩数据解码为HWC像素。
        # IMREAD_COLOR通常得到三个BGR通道，若送RGB模型还需在相应模型输入处转换。
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("栈酱返回的 JPEG 无法解码")
        # //为整数除法，expected按4:3计算(H,W)tuple；shape[:2]只比较高宽，不比较通道。
        expected = (width * 3 // 4, width)
        if frame.shape[:2] != expected:
            raise RuntimeError(f"栈酱图像尺寸异常：{frame.shape[:2]}，预期 {expected}")
        return frame

    def snapshot_for_identity(self, *, timeout=2.5):
        """Fresh low-resolution frame; capture and I2C release are one job."""
        # owner_identity.py的人脸取样调用；*后timeout只能按名称传，返回全新低分辨率BGR帧。
        # 与_get_frame不同，此设备请求把拍摄和I2C释放合成一个工作，适合身份识别按需快照。
        import numpy as np

        jpeg = self.device.capture_identity_jpeg(width=self.width, timeout=timeout)
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[:2] != (self.width * 3 // 4, self.width):
            raise RuntimeError("身份识别收到无效的栈酱画面")
        return frame

    def open(self):
        # preview.main / AutoPhotoSkill.execute调用，返回bool；这是相机会话初始化而非新串口连接。
        # connection_epoch由设备层更新，断线重连后旧缓存不能再代表当前设备会话。
        if self._session_epoch != self.device.connection_epoch:
            self._opened = False
            self._first_frame = None
        if self._opened:
            # 已初始化且连接代次未变化时直接成功，不重复取第一帧。
            return True
        self.last_error = ""
        if not self.device.enabled:
            self.last_error = "栈酱设备尚未就绪，正在自动连接"
            print(f"[StackChanCamera] {self.last_error}")
            return False
        try:
            # 开始时取一帧并缓存，第一轮 read 直接复用，避免重复取图。
            # epoch是取帧前的局部代次；取帧后再比一次，识别获取期间发生的重新连接。
            epoch = self.device.connection_epoch
            self._first_frame = self._get_frame(self.width)
            if epoch != self.device.connection_epoch:
                self._first_frame = None
                raise ConnectionError("取帧期间设备连接已改变，请重新开始拍照")
            # 只有取得有效帧且连接没变化，才提交“已打开”状态与对应epoch。
            self._opened = True
            self._session_epoch = epoch
            return True
        except (OSError, RuntimeError, TimeoutError, ValueError) as error:
            # as error绑定异常对象；错误转文本保存在last_error，释放相机后返回False给调用者。
            self.last_error = str(error)
            print(f"[StackChanCamera] 打开失败：{error}")
            self._release_camera()
            return False

    def read(self):
        # preview/自动拍照循环调用；成功返回帧数组，失败返回None，调用方决定结束这一轮任务。
        if self._session_epoch != self.device.connection_epoch:
            self._opened = False
            self._first_frame = None
        if not self._opened:
            return None
        if self._first_frame is not None:
            # 多重赋值先计算右侧，再取走缓存并清空；第一帧仅消费一次，不再发一次设备请求。
            frame, self._first_frame = self._first_frame, None
            return frame
        try:
            return self._get_frame(self.width)
        except (OSError, RuntimeError, TimeoutError, ValueError) as error:
            self.last_error = str(error)
            print(f"[StackChanCamera] 取帧失败：{error}")
            return None

    def _release_camera(self):
        # open失败、close、capture收尾共用；调用设备close_camera归还I2C资源，不关闭整个设备连接。
        try:
            self.device.close_camera()
        except (OSError, RuntimeError, TimeoutError, ValueError) as error:
            print(f"[StackChanCamera] 释放设备未确认：{error}")

    def close(self):
        # main/preview/AutoPhotoSkill收尾调用，无返回业务数据；保存旧状态后先清本地缓存。
        was_open = self._opened
        self._opened = False
        self._first_frame = None
        if was_open:
            # 归还设备内部 I2C 总线，使触屏、麦克风和扬声器可以继续工作。
            self._release_camera()

    def capture(self):
        # PhotoSkill.execute调用的单张拍照入口；先取640x480原图再save_frame，返回保存结果dict。
        print("[Device] 正在使用栈酱摄像头拍照（640×480）...")
        if not self.device.enabled:
            return {"success": False, "error": "栈酱设备尚未就绪，正在自动连接"}
        try:
            # 一次有效 VGA 帧即可；不照搬电脑摄像头连续抓五帧的等待。
            return self.save_frame(self._get_frame(640))
        except (OSError, RuntimeError, TimeoutError, ValueError) as error:
            self.last_error = str(error)
            return {"success": False, "error": f"栈酱拍照失败：{error}"}
        finally:
            # 即使try中return成功结果，也先执行finally；结束本次相机使用并释放设备I2C。
            self._opened = False
            self._first_frame = None
            self._release_camera()


def create_camera(device):
    """Main and preview share the same StackChan connection owner."""
    # main.py / preview.py调用的工厂函数：统一选择相机实现，调用方只依赖open/read/capture等方法。
    # 输入已创建的StackChanFaceEvents；返回StackChanCamera实例，Class(...)会调用其__init__。
    if device is None:
        raise ValueError("栈酱摄像头需要共享的 StackChanFaceEvents 实例")
    return StackChanCamera(device)
