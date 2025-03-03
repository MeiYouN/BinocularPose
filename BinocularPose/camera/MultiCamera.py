import os
import time

import cv2
import threading
import numpy as np
from typing import List, Dict

from BinocularPose.camera.Camera import Camera


class MultiCamera:
    def __init__(self, camera_ids: List[int] = None, width: int = 2048,
                 height: int = 1536, fps: int = 30):
        """
        初始化多摄像头控制器
        :param camera_ids: 摄像头ID列表（默认[0, 1]）
        :param width: 统一设置画面宽度
        :param height: 统一设置画面高度
        :param fps: 统一设置帧率
        """
        self.camera_ids = camera_ids or [0, 1]
        self.base_width = width
        self.base_height = height
        self.base_fps = fps

        # 创建摄像头实例字典
        self.cameras: Dict[int, Camera] = {}
        self.lock = threading.Lock()

        # 初始化所有摄像头
        self._init_cameras()

        # 可视化相关参数
        self.preview_scale = 0.5
        self.grid_layout = (1, len(self.cameras))  # 默认横向排列
        self.preview_thread = None
        self.preview_running = False  # 新增预览状态标志
        self.key_callback = None  # 新增回调函数引用
        self.frames = None

    def _init_cameras(self):
        """初始化所有摄像头设备"""
        for cam_id in self.camera_ids:
            try:
                cam = Camera(
                    device_index=cam_id,
                    width=self.base_width,
                    height=self.base_height,
                    fps=self.base_fps
                )
                self.cameras[cam_id] = cam
                print(f"摄像头 {cam_id} 初始化成功")
            except Exception as e:
                print(f"摄像头 {cam_id} 初始化失败: {str(e)}")

    def start_recording_all(self, folder_name: str, base_path: str):
        """
        开始同步录制所有摄像头
        :param folder_name: 存储文件夹名称
        :param base_path: 根目录路径
        """
        for cam_id, camera in self.cameras.items():
            save_path = os.path.join(base_path, folder_name)
            os.makedirs(save_path, exist_ok=True)
            filename = f"cam_{cam_id}.mp4"
            camera.start_recording(filename, save_path)

    def stop_recording_all(self):
        """停止所有摄像头录制"""
        for camera in self.cameras.values():
            camera.stop_recording()

    def save_frames_all(self, base_name: str, base_path: str, img_type="jpg"):
        """
        保存所有摄像头当前帧
        :param base_name: 图片基础名称
        :param base_path: 根目录路径
        :param img_type: 图片格式
        """
        for cam_id, camera in self.cameras.items():
            cam_path = os.path.join(base_path, f"{cam_id:02d}")
            camera.save_frame(base_name, cam_path, img_type)

    def get_all_status(self) -> Dict[int, Dict]:
        """获取所有摄像头状态信息"""
        return {
            cam_id: {
                **camera.get_camera_params(),
                "is_alive": camera.running
            }
            for cam_id, camera in self.cameras.items()
        }

    def visualize(self, layout: tuple = None, scale: float = None):
        """
        实时显示多摄像头画面
        :param layout: 排列布局 (rows, cols)
        :param scale: 显示缩放比例
        """
        self.start_preview(layout, scale)

    def close_all(self):
        """关闭所有摄像头"""
        for camera in self.cameras.values():
            camera.close()
        print("所有摄像头已关闭")

    def set_display_layout(self, rows: int, cols: int):
        """设置画面排列布局"""
        self.grid_layout = (rows, cols)

    def set_display_scale(self, scale: float):
        """设置显示缩放比例"""
        self.preview_scale = scale

    def start_preview(self, layout: tuple = None, scale: float = None, key_callback=None):
        """
        启动非阻塞预览线程
        """
        """ 新增key_callback参数 """
        self.key_callback = key_callback

        if self.preview_running:
            return

        self.preview_running = True
        self.preview_thread = threading.Thread(
            target=self._preview_loop,
            args=(layout, scale),
            daemon=True
        )
        self.preview_thread.start()

    def _get_frames(self):
        self.frames = []
        # 获取所有摄像头帧（线程安全）
        for cam_id in sorted(self.cameras.keys()):
            with self.cameras[cam_id].lock:
                frame = self.cameras[cam_id].latest_frame
                self.frames.append(frame)

        return self.frames

    def get_frames(self):
        return self.frames

    def _preview_loop(self, layout, scale):
        """
        独立线程运行的可视化循环
        """
        cv2.namedWindow('Multi-Camera Preview')
        fps_counter = 0
        last_time = 0
        while self.preview_running:
            frames = self._get_frames()
            resized_frames = []
            for frame in frames:
                if frame is not None:
                    h, w = frame.shape[:2]
                    new_size = (int(w * scale), int(h * scale))
                    resized = cv2.resize(frame.copy(), new_size)
                    resized_frames.append(resized)

            if len(resized_frames) == 0:
                continue

            # 动态计算布局
            rows, cols = self._calculate_layout(layout, len(resized_frames))
            combined = self._arrange_frames(resized_frames, rows, cols)

            current_time = time.time()
            fps_counter += 1
            if current_time - last_time >= 1.0:
                fps = fps_counter / (current_time - last_time)
                # cv2.putText(combined, f"FPS: {fps:.1f}", ...)
                fps_counter = 0
                last_time = current_time

            # 显示画面
            cv2.imshow('Multi-Camera Preview', combined)
            # 非阻塞等待按键（1ms）
            key = cv2.waitKey(1)
            # 触发回调函数
            if key != -1 and self.key_callback:
                self.key_callback(key)

            # 退出检测保留
            if not self.preview_running:
                break

        cv2.destroyWindow('Multi-Camera Preview')

    def stop_preview(self):
        """停止预览"""
        self.preview_running = False
        if self.preview_thread is not None:
            self.preview_thread.join(timeout=1)
        self.preview_thread = None

    def _calculate_layout(self, user_layout, frame_count):
        """智能计算最佳布局"""
        if user_layout:
            return user_layout
        # 自动计算接近正方形的布局
        sqrt = int(np.sqrt(frame_count))
        rows = sqrt
        cols = sqrt if sqrt * sqrt == frame_count else sqrt + 1
        return (rows, cols)

    def _arrange_frames(self, frames, rows, cols):
        """排列拼接画面"""
        grid = []
        for i in range(rows):
            row_start = i * cols
            row_end = min((i + 1) * cols, len(frames))
            row_frames = frames[row_start:row_end]

            # 填充空白保持布局完整
            while len(row_frames) < cols:
                row_frames.append(np.zeros_like(row_frames[0]))

            grid.append(np.hstack(row_frames))
        return np.vstack(grid)


# 使用示例
if __name__ == "__main__":
    # 初始化双摄像头控制器
    controller = MultiCamera(
        camera_ids=[0, 1],
        width=1280,
        height=720,
        fps=30
    )

    try:
        # 查看状态
        print("摄像头状态:", controller.get_all_status())

        # 开始录制
        controller.start_recording_all(
            folder_name="experiment_1",
            base_path="./recordings"
        )

        # 设置显示布局
        controller.set_display_layout(2, 1)  # 2行1列
        controller.set_display_scale(0.3)

        # 启动可视化
        controller.visualize()

    finally:
        # 保存测试帧
        controller.save_frames_all(
            base_name="snapshot",
            base_path="./snapshots",
            img_type="png"
        )
        controller.close_all()