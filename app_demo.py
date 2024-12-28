import time

import cv2
import numpy as np
from BinocularPose.models.yolo.yolo_det import Yolo_Det
from BinocularPose.mytools.json_file import JsonFile
from BinocularPose.triangulate import SimpleTriangulate
from BinocularPose.mytools.load_para import load_cameras, load_yml
from BinocularPose.visualize.plot3d import vis_plot
from BinocularPose.models.mymmpose.mymmpose import MyMMP



class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """开始计时"""
        self.start_time = time.time()
        self.end_time = None  # 重置结束时间

    def stop(self):
        """停止计时"""
        if self.start_time is None:
            print("请先调用 start() 方法以开始计时。")
            return
        self.end_time = time.time()
        print(f"计时结束，总计时长为: {self.elapsed:.4f} 秒")

    @property
    def elapsed(self):
        """返回已经过去的时间，如果计时器仍在运行，则返回当前已过去的时间"""
        if self.start_time is None:
            return 0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


def main():
    cameras = load_yml('.\demo_data\demo2')
    folder_path = "./demo_data/demo2/v1080"
    save_path = './run/demo2'
    left_video = folder_path + "/01.mp4"
    right_video = folder_path + "/02.mp4"
    # left_video = 0
    # right_video = 1
    capL = cv2.VideoCapture(left_video)
    capR = cv2.VideoCapture(right_video)


    # model = YoloV11("./yolo_model/yolo11x-pose.pt")
    model = MyMMP('BinocularPose/models/mymmpose')
    triangulate = SimpleTriangulate()
    vis = vis_plot()
    yolo = Yolo_Det('BinocularPose/models/mymmpose/weights/yolo11n.pt')
    timer = Timer()

    jf = JsonFile(folder_path, save_path)

    runing = True
    while runing:
        timer.start()
        retl, framel = capL.read()
        retr, framer = capR.read()
        if not retl or not retr or not runing:
            print(f'视频已结束，共{jf.index}帧')
            break
        # bbox1 = yolo(framel)
        # bbox2 = yolo(framer)
        keypoints3d = None
        posekeypointsl = model(framel, None)
        posekeypointsr = model(framer, None)
        print(posekeypointsl)
        print(posekeypointsr)

        if posekeypointsl is not None and posekeypointsr is not None:
            # print(bbox)


            keypoints = np.concatenate([[posekeypointsl], [posekeypointsr]])
            # print(keypoints)
            keypoints3d = triangulate(keypoints, cameras)
            vis.show(keypoints3d)
            keypoints3d = keypoints3d.tolist()
        else:
            print('视角不完整')
        jf.update(keypoints3d)
        timer.stop()

    jf.save()

if __name__ == '__main__':

    main()






