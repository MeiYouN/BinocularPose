
import cv2
import numpy as np
from BinocularPose.models.yolo.yolo_det import Yolo_Det
from BinocularPose.mytools.json_file import JsonFile
from BinocularPose.triangulate import SimpleTriangulate
from BinocularPose.mytools.load_para import load_cameras, load_yml
from BinocularPose.visualize.plot3d import vis_plot
from BinocularPose.models.mymmpose.mymmpose import MyMMP



# class
#
# def process_threading():


def main():
    cameras = load_yml('.\BinocularPose\config')
    folder_path = "C:/Users/hu/Desktop/Z55/ThreeD_demo/data10/video"
    save_path = './run'
    # left_video = folder_path + "/01.mp4"
    left_video = 0
    right_video = 1
    # right_video = folder_path + "/02.mp4"
    capL = cv2.VideoCapture(left_video)
    capR = cv2.VideoCapture(right_video)


    # model = YoloV11("./yolo_model/yolo11x-pose.pt")
    model = MyMMP('BinocularPose/models/mymmpose')
    triangulate = SimpleTriangulate()
    vis = vis_plot()
    yolo = Yolo_Det()

    jf = JsonFile(folder_path, save_path)

    while True:
        retl, framel = capL.read()
        retr, framer = capR.read()
        if not retl or not retr:
            print(f'视频已结束，共{jf.index}帧')
            break
        bbox1 = yolo(framel)
        bbox2 = yolo(framer)
        keypoints3d = None

        if bbox1 is not None and bbox2 is not None:
            # print(bbox)
            posekeypointsl = model(framel, bbox1)
            posekeypointsr = model(framer, bbox2)

            keypoints = np.concatenate([[posekeypointsl], [posekeypointsr]])
            print(keypoints)
            keypoints3d = triangulate(keypoints, cameras)
            vis.show(keypoints3d)
            cv2.waitKey()
            keypoints3d = keypoints3d.tolist()
        jf.update(keypoints3d)
    jf.save()

if __name__ == '__main__':

    main()






