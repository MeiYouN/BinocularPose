from pathlib import Path
from dataclasses import dataclass



from BinocularPose.models.hrnet import SimpleHRNet
from BinocularPose.models.mymmpose.mymmpose import MyMMP
from BinocularPose.models.yolo.yolo_det import Yolo_Det
from interface.pose3d_interface import ThreeDPoseProcess


class AppConfig:
    workdir: Path = Path.cwd()
    calib_dir = workdir / "calibration"


    # 摄像头参数
    camera_ids = [0, 1]  # 使用两个摄像头
    resolution = (1280, 720)  # 分辨率设置,,(2048,1536)
    fps = 30  # 帧率

    # 摄像头内参文件
    path_intri = 'D:\desktop\MouseWithoutBorders\BinocularPose\gui\calibration\intri.yml'

    yolo_path = 'D:\Desktop\EveryThing\WorkProject\BinocularPose\BinocularPose\models\mymmpose\weights\yolo11n.pt'
    mmpose_path = 'D:\Desktop\EveryThing\WorkProject\BinocularPose\BinocularPose\models\mymmpose'
    # def load_model(self):
    #     self.yolo_model = Yolo_Det('BinocularPose/models/mymmpose/weights/yolo11n.pt')
    #     # pose_model = SimpleHRNet('BinocularPose/models/hrnet/weights/pose_hrnet_w48_384x288.pth')
    #     self.pose_model = MyMMP('BinocularPose/models/mymmpose')

    processor = ThreeDPoseProcess(
        yolo_model=Yolo_Det(yolo_path),
        pose_model=MyMMP(mmpose_path)
    )

class GlobalConfig:
    _instance = None
    config = AppConfig()

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance



# 全局访问入口
CONFIG = GlobalConfig().config