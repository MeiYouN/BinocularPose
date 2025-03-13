import os

import cv2
import threading
import time


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


class Camera(BaseCamera):
    def __init__(self, device_index=0, width=None, height=None, fps=None):
        super().__init__(device_index, width, height, fps)
        # 视频录制相关参数
        self.recording_flag = False
        self.video_writer = None
        self.recording_thread = None
        # 自动启动摄像头
        self.start()

    def start_recording(self, filename, save_path):
        """
        开始视频录制
        :param filename: 视频文件名（不含路径）
        :param save_path: 存储目录路径
        """
        # 构建完整文件路径
        full_path = os.path.join(save_path, filename)

        # 自动创建目录
        os.makedirs(save_path, exist_ok=True)

        # 初始化视频编码器（MP4格式）
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')

        # 创建视频写入对象
        self.video_writer = cv2.VideoWriter(
            full_path,
            fourcc,
            self.actual_fps,
            (self.actual_width, self.actual_height))

        if not self.video_writer.isOpened():
            raise RuntimeError("无法创建视频文件")

        # 启动录制线程
        self.recording_flag = True
        self.recording_thread = threading.Thread(
            target=self._recording_loop,
            daemon=True
        )
        self.recording_thread.start()

    def _recording_loop(self):
        """ 定时抓取帧的录制循环 """
        frame_interval = 1 / self.actual_fps  # 计算帧间隔
        next_capture_time = time.time()

        while self.recording_flag:
            # 精确控制帧率
            current_time = time.time()
            if current_time >= next_capture_time:
                frame = self.read()
                if frame is not None:
                    with self.lock:  # 确保线程安全
                        self.video_writer.write(frame)
                # 更新下次采集时间
                next_capture_time += frame_interval
            else:
                # 精确等待剩余时间
                time.sleep(max(0, next_capture_time - current_time - 0.001))

    def stop_recording(self):
        """ 停止视频录制 """
        if self.recording_flag:
            self.recording_flag = False
            if self.recording_thread is not None:
                self.recording_thread.join()
            if self.video_writer is not None:
                self.video_writer.release()
            self.video_writer = None

    def save_frame(self, filename, save_path, img_type="jpg"):
        """
        保存当前帧为图片
        :param filename: 文件名（不含扩展名）
        :param save_path: 存储路径
        :param img_type: 图片格式（默认jpg）
        """
        frame = self.read()
        if frame is not None:
            # 构建完整路径
            full_path = os.path.join(save_path, f"{filename}.{img_type}")
            os.makedirs(save_path, exist_ok=True)

            # 保存图像（设置JPG质量）
            if img_type.lower() == "jpg":
                cv2.imwrite(full_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            else:
                cv2.imwrite(full_path, frame)

    def get_frame(self):
        """ 获取当前帧的numpy数组 """
        return self.read()

    def get_camera_params(self):
        """ 获取摄像头实际参数 """
        return {
            "device_id": self.device_index,
            "width": self.actual_width,
            "height": self.actual_height,
            "fps": self.actual_fps,
            "is_recording": self.recording_flag
        }

    def close(self):
        """ 重写关闭方法 """
        self.stop_recording()
        super().close()

# 使用示例
if __name__ == "__main__":
    # 初始化摄像头（自动启动）
    cam = Camera(
        device_index=0,
        width=1280,
        height=720,
        fps=30
    )

    try:
        # 打印摄像头参数
        print("摄像头参数:", cam.get_camera_params())

        # 开始录制
        cam.start_recording(
            filename="demo_video.mp4",
            save_path="./recordings"
        )

        # 持续运行10秒
        start_time = time.time()
        while time.time() - start_time < 10:
            frame = cam.get_frame()
            if frame is not None:
                cv2.imshow('Preview', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # 保存测试帧
        cam.save_frame(
            filename="snapshot",
            save_path="./screenshots",
            img_type="png"
        )

    finally:
        cam.stop_recording()
        cam.close()
        cv2.destroyAllWindows()