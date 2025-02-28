import os

import cv2
import threading
import time
from PIL import Image


class BaseCamera:
    def __init__(self, device_index=0, width=None, height=None, fps=None):
        """
        初始化摄像头配置参数
        :param device_index: 摄像头设备索引
        :param width: 期望画面宽度
        :param height: 期望画面高度
        :param fps: 期望帧率
        """
        self.device_index = device_index
        self._requested_width = width
        self._requested_height = height
        self._requested_fps = fps

        # 实际摄像头参数（在start后生效）
        self._actual_width = None
        self._actual_height = None
        self._actual_fps = None
        self.fourcc = cv2.VideoWriter.fourcc(*"MJPG")

        # 摄像头控制相关
        self.cap = None
        self.latest_frame = None
        self.running = False
        self.lock = threading.Lock()
        self.capture_thread = None

    def start(self):
        """
        启动摄像头并开始持续捕获画面
        """
        if self.running:
            return

        # 初始化摄像头
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头设备 {self.device_index}")

        # 设置基础参数
        self._apply_basic_settings()

        # 获取实际参数值
        self._actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        # 启动捕获线程
        self.running = True
        self.capture_thread = threading.Thread(
            target=self._update_frame,
            daemon=True
        )
        self.capture_thread.start()

        # 等待首帧就绪
        while self.latest_frame is None and self.running:
            time.sleep(0.01)

    def _apply_basic_settings(self):
        """应用基础分辨率/FPS设置"""
        if self._requested_width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_width)
        if self._requested_height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)
        if self._requested_fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, self._requested_fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, self.fourcc)

    # ------------------ 新增高级参数控制方法 ------------------
    def set_exposure(self, exposure_value, auto_mode=False):
        """
        设置曝光参数
        :param exposure_value: 曝光值（具体范围取决于硬件）
        :param auto_mode: 是否启用自动曝光
        """
        self._check_camera_active()
        if auto_mode:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 自动模式
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 手动模式
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)

    def set_gain(self, gain_value, auto_mode=False):
        """
        设置增益参数
        :param gain_value: 增益值（具体范围取决于硬件）
        :param auto_mode: 是否启用自动增益
        """
        self._check_camera_active()
        self.cap.set(cv2.CAP_PROP_GAIN, gain_value)
        if auto_mode:
            # 注意：OpenCV没有直接的自动增益控制，需通过其他方式实现
            pass

    def set_white_balance(self, wb_value, auto_mode=False):
        """
        设置白平衡参数
        :param wb_value: 白平衡色温值
        :param auto_mode: 是否启用自动白平衡
        """
        self._check_camera_active()
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1 if auto_mode else 0)
        if not auto_mode:
            self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb_value)

    def set_brightness(self, brightness_value):
        """ 设置亮度值 """
        self._check_camera_active()
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness_value)

    def set_contrast(self, contrast_value):
        """ 设置对比度值 """
        self._check_camera_active()
        self.cap.set(cv2.CAP_PROP_CONTRAST, contrast_value)

    def _check_camera_active(self):
        """验证摄像头是否处于活动状态"""
        if not self.running or self.cap is None:
            raise RuntimeError("摄像头未启动，无法设置参数")

    # ------------------ 基础功能保持不变 ------------------
    def _update_frame(self):
        """独立线程持续捕获最新画面"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            with self.lock:
                self.latest_frame = frame.copy()
        self.cap.release()

    def read(self):
        """获取最新视频帧"""
        with self.lock:
            return self.latest_frame

    def close(self):
        """关闭摄像头"""
        if self.running:
            self.running = False
            self.capture_thread.join()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.latest_frame = None

    @property
    def actual_width(self):
        return self._actual_width

    @property
    def actual_height(self):
        return self._actual_height

    @property
    def actual_fps(self):
        return self._actual_fps


class Camera:
    """改进的摄像头类"""

    def __init__(self, device_id, width, height, fps):
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {device_id}")

        # 设置参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # 实际参数
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        # 视频录制
        self.writer = None
        self.recording = False
        self.lock = threading.Lock()
        self.frame = None

        # 启动采集线程
        self.running = True
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()

    def _update_frame(self):
        """持续采集画面"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            with self.lock:
                self.frame = frame.copy()
        self.cap.release()

    def read(self):
        """获取当前帧"""
        with self.lock:
            return self.frame.copy()

    def save_frame(self, filename, path, img_type="jpg"):
        """保存当前帧"""
        frame = self.read()
        if frame is not None:
            full_path = os.path.join(path, f"{filename}.{img_type}")
            if img_type.lower() == "png":
                cv2.imwrite(full_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            else:
                cv2.imwrite(full_path, frame)

    def start_recording(self, filename:str, save_dir):
        """开始录制"""
        if self.recording:
            return
        os.makedirs(save_dir, exist_ok=True)
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            os.path.join(save_dir, filename),
            fourcc,
            self.actual_fps,
            (self.actual_width, self.actual_height))
        self.recording = True

    def stop_recording(self):
        """停止录制"""
        if self.recording and self.writer:
            self.writer.release()
            self.recording = False

    def close(self):
        """释放资源"""
        self.running = False
        self.stop_recording()
        if self.thread.is_alive():
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()
