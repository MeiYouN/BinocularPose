import os
import queue

import cv2
import time
import threading
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple

from BinocularPose.camera.MultiCamera import MultiCamera


class LiveVideo:
    def __init__(self,
                 camera_ids: List[int] = [0, 1],
                 resolution: Tuple[int, int] = (2048, 1536),
                 fps: int = 30,
                 work_dir: str = "recordings"):
        """
        初始化多摄像头控制器
        :param camera_ids: 摄像头ID列表
        :param resolution: 视频分辨率 (width, height)
        :param fps: 帧率
        :param work_dir: 工作目录路径
        """
        # 初始化摄像头系统
        self.controller = MultiCamera(
            camera_ids=camera_ids,
            width=resolution[0],
            height=resolution[1],
            fps=fps
        )

        # 状态管理变量
        self.is_recording = False
        self.record_counter = 0
        self.video_dir = work_dir
        key_events = queue.Queue()  # 按键事件队列

        # 定义按键回调函数
        def key_handler(key):
            key_events.put(key)

        # 启动可视化预览（左右布局）
        self.controller.start_preview(
            layout=(1, 2),  # 1行2列
            scale=0.5,
            key_callback=key_handler
        )

    def start_recording(self,pose_name = None):
        """开始同步录制所有摄像头"""
        if self.is_recording:
            print("录制正在进行中")
            return

        if pose_name is None:
            self.record_counter += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"recording_{self.record_counter:03d}_{timestamp}"
        else:
            folder_name = pose_name

        self.controller.start_recording_all(
            folder_name=folder_name,
            base_path=self.video_dir
        )
        self.is_recording = True
        print(f"开始录制 #{self.record_counter}")

    def stop_recording(self):
        """停止所有摄像头录制"""
        if not self.is_recording:
            print("没有正在进行的录制")
            return

        self.controller.stop_recording_all()
        self.is_recording = False
        print(f"已停止录制 #{self.record_counter}")

    def close(self):
        """安全关闭系统"""
        self.stop_recording()
        self.controller.stop_preview()
        self.controller.close_all()
        cv2.destroyAllWindows()
        print("系统资源已释放")


live_video = LiveVideo(camera_ids=[0, 1],
        resolution=(2048, 1536),
        fps=30,
        work_dir="./demo")

def start_recording(pose_name = None, live_pose = False):
    live_video.start_recording(pose_name, live_pose)

def stop_recording(pose_name = None, live_pose = False):
    live_video.stop_recording()

def close():
    live_video.close()

# 使用示例
if __name__ == "__main__":
    # 初始化双摄像头控制器
    controller = LiveVideo(
        camera_ids=[0, 1],
        resolution=(1280, 720),
        fps=30,
        work_dir="my_experiment"
    )

    try:
        # 开始录制
        controller.start_recording()

        # 保持程序运行
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        # 安全关闭
        controller.close()

    print("程序已正常退出")