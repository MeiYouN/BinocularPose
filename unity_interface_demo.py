import os

from BinocularPose.models.mymmpose.mymmpose import MyMMP
from BinocularPose.models.yolo.yolo_det import Yolo_Det
from interface.live_video_interface import LiveVideo
from interface.pose3d_interface import ThreeDPoseProcess
from interface.cal_interface import cal_interface as cal_in


class UnityInterfaceDemo(object):
    def __init__(self,intri:str, cam_id_list:list[int]=[0,1], work_dri:str='./demo/' ):

        self.cam_id_list = cam_id_list
        self.work_dri = work_dri
        self.cal_dir = os.path.join(self.work_dri, 'calibration')
        self.intri = intri

        self.video_model = LiveVideo(camera_ids=self.cam_id_list,
                               resolution=(2048, 1536),
                               fps=30,
                               work_dir=self.work_dri)

        self.processor = ThreeDPoseProcess(
            yolo_model=Yolo_Det('BinocularPose/models/mymmpose/weights/yolo11n.pt'),
            pose_model=MyMMP('BinocularPose/models/mymmpose')
        )

    def cal_interface(self):
        cal_in(self.cal_dir,self.intri,self.video_model.controller)


    def start_recording(self,pose_name=None, live_pose=False):
        self.video_model.start_recording(pose_name)


    def stop_recording(self,pose_name=None, live_pose=False):
        self.video_model.stop_recording()


    def close_video(self):
        self.video_model.close()

    def offline_pose(self):
        pass

    def online_pose(self):
        pass
