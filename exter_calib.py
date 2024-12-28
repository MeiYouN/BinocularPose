import os
import cv2

from BinocularPose.calibration.calib_extri import calib_extri
from BinocularPose.calibration.extract_video import extract_video
from BinocularPose.calibration.detect_chessboard import det_board
from BinocularPose.camera.VideoCapture import StereoCamera




def capture(cam_id_l, cam_id_r, save_dir, frameSize=(2048, 1536), fps=20):

    dir01 = save_dir + '/images/01'
    dir02 = save_dir + '/images/02'
    cams = StereoCamera(save_dir, cam_id_l, cam_id_r,frameSize[0] ,frameSize[1] ,fps=fps)
    cams.create_file(dir01)
    cams.create_file(dir02)


    i = 0
    while True:
        frameL, frameR = cams.capture()
        if frameL is None or frameR is None:
            break
        l = frameL.copy()
        r = frameR.copy()
        cv2.namedWindow('left', 0)
        cv2.namedWindow('right', 0)
        cv2.resizeWindow('left', 640, 480)
        cv2.resizeWindow('right', 640, 480)
        cv2.imshow('left', l)
        cv2.imshow('right', r)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break
        elif key == ord('c') or key == ord('s'):
            print("save image:{:0=3d}".format(i))
            cv2.imwrite(os.path.join(dir01, "{:0=6d}.jpg".format(i)), frameL)
            cv2.imwrite(os.path.join(dir02, "{:0=6d}.jpg".format(i)), frameR)
            i += 1
    # cams.close()
    print(f'已完成图片采集，共保存{i}张图片')
    cv2.destroyAllWindows()

def main():
    # cam_id_l = 1
    # cam_id_r = 0
    cam_id_l = './demo_data/demo2/v1080/01.mp4'
    cam_id_r = './demo_data/demo2/v1080/02.mp4'
    dir_path = './demo_data/demo2/'
    print(1)
    # capture(cam_id_l,cam_id_r,dir_path)
    print(2)
    # extract_video(dir_path, 4)s
    det_board(dir_path, (6,4), 0.1)
    print(3)
    calib_extri(dir_path, 'demo_data/demo2/intri.yml',1)
    print(4)

if __name__ == '__main__':
    main()
