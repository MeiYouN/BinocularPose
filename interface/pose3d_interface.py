
import numpy as np
from typing import List

from BinocularPose.camera.MultiCamera import MultiCamera
from BinocularPose.models.mymmpose.mymmpose import MyMMP
from BinocularPose.models.yolo.yolo_det import Yolo_Det
from BinocularPose.mytools.load_para import load_yml
from BinocularPose.triangulate import SimpleTriangulate
from app_demo import Timer
from interface.live_video_interface import LiveVideo


class ThreeDPoseProcess:
    def __init__(self, yolo_model, pose_model):
        self.frame_index = 0
        # self.video_model = LiveVideo(camera_ids=[0, 1],
        #                        resolution=(2048, 1536),
        #                        fps=30,
        #                        work_dir="./demo")

        # 绑定模型
        self.yolo = yolo_model
        self.pose_model = pose_model
        self.triangulate = SimpleTriangulate()

    def capture_frames(self):
        """双流捕获线程，保证帧同步"""
        pass



    def detection(self, frames):
        """批量目标检测线程"""
        bboxs = self.yolo(frames)
        return bboxs

    def pose_estimation(self, frames: List[np.ndarray], bboxes: List[np.ndarray])->np.ndarray:
        keypoints = []
        for frame, bbox in zip(frames, bboxes):
            keypoints2d = self.pose_model(frame, bbox)

            keypoints.append(keypoints2d)

        return np.array(keypoints)

    def count_empty_arrays(self, arr_list):
        return sum(1 for arr in arr_list if isinstance(arr, np.ndarray) and arr.size == 0)

    def run_process(self, frames, cameras):

        self.frame_index += 1
        bboxs = self.detection(frames)
        print(bboxs)
        # if len(bboxs) <= 2:
        #     print('当前有效视角不足')
        #     return np.array([])
        keypoints = self.pose_estimation(frames, bboxs)
        print(keypoints)
        keypoints3d = self.triangulate(keypoints, cameras)
        # print(keypoints3d)

        return keypoints3d




if __name__ == '__main__':
    # cameras = load_yml('demo_data/p2/')
    # cameras = load_yml('demo_data/demo2')
    folder_path = "../demo_data/demo2"
    save_path = folder_path
    left_video = folder_path + "/videos/01.mp4"
    right_video = folder_path + "/videos/02.mp4"

    # 初始化处理系统
    # processor = ThreeDPoseProcess(
    #     yolo_model=Yolo_Det('BinocularPose/models/mymmpose/weights/yolo11n.pt'),
    #     pose_model=MyMMP('BinocularPose/models/mymmpose')
    # )

    # controller = MultiCamera(
    #     camera_ids=[left_video, right_video],
    #     width=2048,
    #     height=1536,
    #     fps=30,
    # )
    #
    # controller.visualize()

    # processor.run_process(frames, cameras)




