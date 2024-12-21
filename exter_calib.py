import os
import queue
import threading

import cv2
from alfred.modules.data.split_voc import save_dir

from BinocularPose.calibration.calib_extri import calib_extri
from BinocularPose.calibration.extract_video import extract_video
from BinocularPose.calibration.detect_chessboard import det_board


class VideoCapture:
    """Customized VideoCapture, always read latest frame """

    def __init__(self, camera_id, width=640, height=480, fps=30):
        # "camera_id" is a int type id or string name
        self.cap = cv2.VideoCapture(camera_id)
        self.set_cap_info(width, height, fps)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
        self.q = queue.Queue(maxsize=3)
        self.stop_threads = False  # to gracefully close sub-thread
        self.cap.grab()
        th = threading.Thread(target=self._reader)
        th.daemon = True  # 设置工作线程为后台运行
        th.start()

    # 实时读帧，只保存最后一帧
    def _reader(self):
        while not self.stop_threads:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def terminate(self):
        self.stop_threads = True
        self.cap.release()

    def get_cap_info(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        return width, height, fps

    def set_cap_info(self,width, height, fps):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)


class StereoCamera:

    def __init__(self,save_dir ,cam_id_l=1, cam_id_r=0 ,wight=640, height=480, fps=30):

        self.capL = VideoCapture(cam_id_l, wight, height, fps)
        self.capR = VideoCapture(cam_id_r, wight, height, fps)
        print("cap info: ", wight, height, fps)
        self.create_file(save_dir)
        self.save_videoL = self.create_file(save_dir, "videos", "01.mp4")
        self.save_videoR = self.create_file(save_dir, "videos", "02.mp4")
        self.writerL = self.get_video_writer(self.save_videoL, wight, height, fps)
        self.writerR = self.get_video_writer(self.save_videoR, wight, height, fps)


    def capture(self):

        frameL = self.capL.read()
        frameR = self.capR.read()
        self.writerL.write(frameL)
        self.writerR.write(frameR)

        return frameL, frameR

    @staticmethod
    def get_video_writer(save_path, width, height, fps):
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frameSize = (int(width), int(height))
        video_writer = cv2.VideoWriter(save_path, fourcc, fps, frameSize)
        print("video:width:{},height:{},fps:{}".format(width, height, fps))
        return video_writer

    @staticmethod
    def create_file(parent_dir, dir1=None, filename=None):
        out_path = parent_dir
        if dir1:
            out_path = os.path.join(parent_dir, dir1)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if filename:
            out_path = os.path.join(out_path, filename)
        return out_path

    def close(self):
        self.capL.terminate()
        self.capR.terminate()


def capture(cam_id_l, cam_id_r, save_dir):

    cams = StereoCamera(save_dir, cam_id_l, cam_id_r, fps=20)
    i = 0
    while True:
        frameL, frameR = cams.capture()
        l = frameL.copy()
        r = frameR.copy()
        cv2.namedWindow('left', 0)
        cv2.namedWindow('right', 0)
        cv2.resizeWindow('left', 640, 480)
        cv2.resizeWindow('right', 640, 480)
        cv2.imshow('left', l)
        cv2.imshow('right', r)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        elif key == ord('c') or key == ord('s'):
            print("save image:{:0=3d}".format(i))
            cv2.imwrite(os.path.join(save_dir + "/images/01", "{:0=6d}.png".format(i)), frameL)
            cv2.imwrite(os.path.join(save_dir + "/images/02", "{:0=6d}.png".format(i)), frameR)
            i += 1
    cams.close()
    print(f'已完成图片采集，共{i}张')
    cv2.destroyAllWindows()

def main():
    # cam_id_l = 1
    # cam_id_r = 0
    cam_id_l = './demo_data/demo2/Camera Roll/01.mp4'
    cam_id_r = './demo_data/demo2/Camera Roll/02.mp4'
    dir_path = './demo_data/demo2/'

    capture(cam_id_l,cam_id_r,dir_path)

    # extract_video(dir_path, 4)
    det_board(dir_path, (7,5), 0.1)
    calib_extri(dir_path, 'demo_data/demo2/intri.yml',3)


if __name__ == '__main__':
    main()
