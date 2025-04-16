import os
import threading
import traceback
from datetime import datetime

import cv2
from typing import List, Union

from BinocularPose.models.mymmpose.mymmpose import MyMMP
from BinocularPose.models.hrnet.hrnet_api import SimpleHRNet
from BinocularPose.models.yolo.yolo_det import Yolo_Det
from BinocularPose.mytools.csv_file import CsvFile
from BinocularPose.mytools.json_file import JsonFile
from BinocularPose.mytools.load_para import load_yml
from BinocularPose.visualize.plot3d import vis_plot
from interface.live_video_interface import LiveVideo
from interface.pose3d_interface import ThreeDPoseProcess
from interface.cal_interface import cal_interface as cal_in


def get_files_by_extension(path, extension='.mp4'):
    """
    获取指定路径下特定后缀的文件名（含路径）
    :param path: 目标路径
    :param extension: 文件后缀（如 ".txt"）
    :return: 符合条件的文件路径列表
    """

    # 遍历目录，筛选符合条件的文件
    matched_files = []
    for filename in os.listdir(path):
        full_path = os.path.join(path, filename)
        if os.path.isfile(full_path) and filename.endswith(extension) and filename.startswith('cam'):
            matched_files.append(full_path)

    return matched_files


class UnityInterfaceDemo(object):
    def __init__(self, intri:str = None, cam_id_list = [0, 1], work_dri:str= './demo/'):

        self.cameras = None
        self.Onlineing = False
        self.cam_id_list = cam_id_list
        self.work_dri = work_dri
        self.cal_dir = os.path.join(self.work_dri, 'calibration')
        if intri is None:
            self.intri = intri
        else:
            self.intri = self.cal_dir

        # if cam_id_list is not None:
        #     self.video_model = LiveVideo(camera_ids=self.cam_id_list,
        #                            resolution=(2048, 1536),
        #                            fps=30,
        #                            work_dir=self.work_dri)

        self.processor = ThreeDPoseProcess(
            yolo_model=Yolo_Det('BinocularPose/models/mymmpose/weights/yolo11n.pt'),
            pose_model=MyMMP('BinocularPose/models/mymmpose')
        )
        # self.processor = ThreeDPoseProcess(
        #     yolo_model=Yolo_Det('BinocularPose/models/mymmpose/weights/yolo11n.pt'),
        #     pose_model=SimpleHRNet('BinocularPose/models/hrnet/weights/pose_hrnet_w48_384x288.pth')
        # )

        self.online_thread = None

    def cal_interface(self):
        cal_in(self.cal_dir,self.intri,self.video_model.controller)


    def start_recording(self,pose_name=None, live_pose=False):
        self.video_model.start_recording(pose_name)
        if live_pose:
            self.online_thread = threading.Thread(
                target=self.online_pose,
                args=pose_name,
                daemon=True
            )
            self.Onlineing = True
            self.online_thread.start()


    def stop_recording(self):
        self.Onlineing = False
        if self.online_thread is not None:
            self.online_thread.join()
        self.online_thread = None
        self.video_model.stop_recording()

    def load_yml(self):
        self.cameras = load_yml(self.cal_dir)
        return self.cameras

    def close_video(self):
        self.video_model.close()

    def offline_pose(self, pose_name):
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        save_data_name = f"{pose_name}.json"
        videos_path = os.path.join(self.work_dri, pose_name)
        video_names = get_files_by_extension(videos_path)

        plot = vis_plot()
        cameras = self.load_yml()
        caplist = []
        for video_name in video_names:
            video_path = os.path.join(videos_path, video_name)
            cap = cv2.VideoCapture(video_path)
            caplist.append(cap)
        jf = JsonFile(videos_path, savedir+'/zsdata/'+save_data_name)

        cap_nums = len(caplist)
        try:
            while True:

                frames = []
                for cap in caplist:
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)

                if not cap_nums==len(frames):
                    print(f'视频已结束，共{jf.index}帧')
                    break

                keypoints3d = self.processor.run_process(frames, cameras)
                # plot.show(keypoints3d)
                jf.update(keypoints3d)
        except Exception:
            print(traceback.format_exc())
            jf.save()
        finally:
            jf.save()



    def online_pose(self, pose_name):
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        save_data_name = f"{pose_name}_{timestamp}.json"
        videos_path = os.path.join(self.work_dri, pose_name)
        cameras = self.load_yml()

        jf = JsonFile(videos_path, videos_path + '/run/' + save_data_name)

        while self.Onlineing:

            frames = self.video_model.controller.get_frames()

            if not len(self.cam_id_list) == len(frames):
                print(f'视频已结束，共{jf.index}帧')
                break

            keypoints3d = self.processor.run_process(frames, cameras)

            jf.update(keypoints3d)

        jf.save()


savedir = 'D:\Desktop\EveryThing\WorkProject\BinocularPose\demo_data\datasetdemo\dataset'

if __name__ == '__main__':

    intri = 'D:\Desktop\EveryThing\WorkProject\Data\s1-videos'
    work_dri = 'D:\Desktop\EveryThing\WorkProject\Data\s1-videos'

    unity = UnityInterfaceDemo(work_dri=work_dri)

    posenames = ['acting1']
    for pose_name in posenames:
        print(pose_name)
        unity.offline_pose(pose_name)

