import os
import time
from datetime import datetime
from queue import Queue

import cv2
import threading
import numpy as np
from typing import List, Dict, Tuple

from BinocularPose.camera.Camera import Camera


class MultiCamera:
    def __init__(self,
                 camera_ids: List[int] = [0, 1],
                 resolution: Tuple[int, int] = (1280, 720),
                 fps: int = 30,
                 work_dir: str = "workspace"):
        """
        多摄像头控制器
        :param camera_ids: 摄像头ID列表
        :param resolution: 分辨率 (宽, 高)
        :param fps: 帧率
        :param work_dir: 工作根目录
        """
        # 硬件控制部分
        self.cameras = self._init_cameras(camera_ids, resolution, fps)
        self.running = True

        # 文件管理
        self.work_dir = os.path.abspath(work_dir)
        self._create_dirs()
        self.save_counter = 0
        self.record_counter = 0
        self.is_recording = False

        # 可视化控制
        self.event_queue = Queue()
        self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self.preview_thread.start()

    def _init_cameras(self, ids, res, fps) -> Dict[int, Camera]:
        """初始化摄像头实例"""
        cams = {}
        for cam_id in ids:
            try:
                cam = Camera(cam_id, res[0], res[1], fps)
                cams[cam_id] = cam
                print(f"摄像头 {cam_id} 初始化成功")
            except Exception as e:
                print(f"摄像头 {cam_id} 初始化失败: {str(e)}")
        return cams

    def _create_dirs(self):
        """创建标准目录结构"""
        self.image_dir = os.path.join(self.work_dir, "images")
        self.video_dir = os.path.join(self.work_dir, "videos")
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

    def _preview_loop(self):
        """可视化与事件处理主循环"""
        cv2.namedWindow("Camera Controller", cv2.WINDOW_NORMAL)

        while self.running:
            # 获取所有摄像头画面
            frames = []
            for cam_id in sorted(self.cameras.keys()):
                frame = self.cameras[cam_id].read()
                if frame is not None:
                    frames.append(cv2.resize(frame, (640, 480)))  # 统一缩放尺寸

            # 拼接画面（2x1布局）
            if len(frames) >= 2:
                combined = np.vstack(frames)
                self._draw_hud(combined)
                cv2.imshow("Camera Controller", combined)

            # 按键检测
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC/q退出
                self.shutdown()
            elif key == ord('s'):  # 保存快照
                self._save_snapshots()
            elif key == ord('r'):  # 录制控制
                self._toggle_recording()

        cv2.destroyAllWindows()

    def _draw_hud(self, canvas):
        """绘制状态信息"""
        status = [
            f"Record Counter: {self.record_counter}",
            f"Save Counter: {self.save_counter}",
            f"Recording: {'ON' if self.is_recording else 'OFF'}"
        ]
        y = 30
        for text in status:
            cv2.putText(canvas, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y += 40

    def _save_snapshots(self):
        """保存当前帧到images目录"""
        self.save_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for cam_id, camera in self.cameras.items():
            path = os.path.join(self.image_dir, f"cam{cam_id}")
            os.makedirs(path, exist_ok=True)
            camera.save_frame(
                f"snap_{self.save_counter:03d}_{timestamp}",
                path,
                "png"
            )
        print(f"已保存快照 #{self.save_counter}")

    def _toggle_recording(self):
        """切换录制状态"""
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        """开始录制视频到videos目录"""
        self.record_counter += 1
        folder = os.path.join(self.video_dir, f"recording_{self.record_counter:03d}")
        for cam_id, camera in self.cameras.items():
            camera.start_recording(f"cam{cam_id}.mp4", folder)
        self.is_recording = True
        print(f"开始录制 #{self.record_counter}")

    def _stop_recording(self):
        """停止录制"""
        for camera in self.cameras.values():
            camera.stop_recording()
        self.is_recording = False
        print(f"停止录制 #{self.record_counter}")

    def shutdown(self):
        """安全关闭系统"""
        self.running = False
        self._stop_recording()
        for camera in self.cameras.values():
            camera.close()
        print("系统资源已释放")

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