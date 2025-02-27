import os
from typing import List,Tuple

import cv2

from BinocularPose.calibration.calib_extri import calib_extri
from BinocularPose.calibration.extract_video import extract_video
from BinocularPose.calibration.detect_chessboard import det_board
from BinocularPose.camera.MultiCamera import MultiCamera
from BinocularPose.camera.VideoCapture import StereoCamera




def sync_frame_capture(
    camera_ids: List[int] = None,
    save_path: str = "./demo/images",
    image_size: Tuple[int, int] = (2048, 1536)
) -> bool:
    """
    多摄像头同步拍摄功能
    :param camera_ids: 摄像头ID列表
    :param save_path: 图片保存根目录
    :param image_size: 图像分辨率 (width, height)
    :return: (运行状态, 保存数量) 或 (False, 错误信息)
    """
    counter = 0
    multi_cam = None
    try:
        # 参数校验
        if not camera_ids:
            camera_ids = [0, 1]
        if not isinstance(image_size, tuple) or len(image_size) != 2:
            raise ValueError("图像尺寸参数格式错误，应为(width, height)元组")

        # 创建多摄像头控制器
        multi_cam = MultiCamera(
            camera_ids=camera_ids,
            width=image_size[0],
            height=image_size[1]
        )

        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)

        # 启动可视化预览
        multi_cam.start_preview(scale=0.5)
        print("操作指南：\n[s] 保存当前帧\n[q/ESC] 退出")

        while True:
            # 非阻塞按键检测
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC或q键
                break
            elif key == ord('s'):
                # 保存所有摄像头画面
                base_name = f"frame_{counter:04d}"
                multi_cam.save_frames_all(
                    base_name=base_name,
                    base_path=save_path,
                    img_type="jpg"
                )
                print(f"已保存第 {counter} 组画面")
                counter += 1
        print(f"共保存{counter}张图片")

        return True

    except Exception as e:
        print(f"运行时发生错误：{str(e)}")
        return False
    finally:
        # 清理资源
        if multi_cam is not None:
            multi_cam.stop_preview()
            multi_cam.close_all()
        cv2.destroyAllWindows()


def main():
    cam_id_l = 1
    cam_id_r = 0
    dir_path = './demo/images/'
    print(1)
    sync_frame_capture([cam_id_l,cam_id_r],dir_path)
    print(2)
    # extract_video(dir_path, 4)
    # det_board(dir_path, (6,4), 0.1)
    # print(3)
    # calib_extri(dir_path, 'demo_data/demo2/intri.yml',1)
    # print(4)

if __name__ == '__main__':
    main()
