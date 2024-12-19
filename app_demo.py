
import cv2
import numpy as np
from BinocularPose.models.yolo.yolo_det import Yolo_Det
from BinocularPose.triangulate import SimpleTriangulate
from BinocularPose.mytools.load_para import load_cameras, load_yml
from BinocularPose.visualize.plot3d import vis_plot
from BinocularPose.models.mymmpose.mymmpose import MyMMP



def get_video_info(video_cap):
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    numFrames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    return width, height, numFrames, fps

if __name__ == '__main__':

    cameras = load_yml('D:\Desktop\EveryThing\EasyMocap\mydata\data10\\video')

    left_video = "D:\Desktop\EveryThing\EasyMocap\mydata\data10\\video\\01.mp4"
    right_video = "D:\Desktop\EveryThing\EasyMocap\mydata\data10\\video\\02.mp4"
    capL = cv2.VideoCapture(left_video)
    capR = cv2.VideoCapture(right_video)

    # model = YoloV11("./yolo_model/yolo11x-pose.pt")
    model = MyMMP('BinocularPose/models/mymmpose')
    triangulate = SimpleTriangulate('naive')
    vis = vis_plot()
    yolo = Yolo_Det()

    retr, framer = capR.read()
    retl, framel = capL.read()

    while True:
        retl, framel = capL.read()
        retr, framer = capR.read()
        bbox1 = yolo(framel)
        bbox2 = yolo(framer)
        if bbox1 is not None and bbox2 is not None:
            # print(bbox)
            posekeypointsl = model(framel, bbox1)
            posekeypointsr = model(framer, bbox2)

            # try:
            keypoints = np.concatenate([[posekeypointsl], [posekeypointsr]])
            print(keypoints)
            keypoints3d = triangulate(keypoints, cameras)['keypoints3d']
            vis.show(keypoints3d)
            cv2.waitKey()
            # print(keypoints3d)
            # except Exception as e:
            #     print(posekeypointsl)
            #     print(posekeypointsr)
            #     print(keypoints)
            #     break






