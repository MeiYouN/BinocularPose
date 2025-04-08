import os
import queue
import time
from typing import List,Tuple

import cv2

from BinocularPose.calibration.calib_extri import calib_extri
from BinocularPose.calibration.detect_chessboard import det_board
from BinocularPose.camera.MultiCamera import MultiCamera




def sync_frame_capture(
    camera_model: MultiCamera = None,
    save: str = "./demo/cal/",
    image_size: Tuple[int, int] = (2048, 1536)
) -> bool:
    """
    多摄像头同步拍摄功能
    :param camera_model: 摄像头模块
    :param save_path: 图片保存根目录
    :param image_size: 图像分辨率 (width, height)
    :return: (运行状态, 保存数量) 或 (False, 错误信息)
    """
    counter = 0
    multi_cam = None
    key_events = queue.Queue()  # 新增事件队列

    try:
        # 参数校验
        if not camera_model:
            camera_ids = [0, 1]
            multi_cam = MultiCamera(
                camera_ids=camera_ids,
                width=image_size[0],
                height=image_size[1]
            )
        else:
            multi_cam = camera_model

        # 创建保存目录
        save_path = os.path.join(save, 'images')
        os.makedirs(save_path, exist_ok=True)

        # 启动带事件回调的可视化预览
        def preview_callback(key):
            """ 将按键事件放入队列 """
            key_events.put(key)

        multi_cam.start_preview(
            scale=0.5,
            key_callback=preview_callback  # 新增回调参数
        )

        print("操作指南：\n[s] 保存当前帧\n[q/ESC] 退出")

        while True:
            try:
                # 从队列获取按键事件（非阻塞）
                key = key_events.get_nowait()
                if key in (27, ord('q')):
                    break
                elif key == ord('s'):
                    # 保存所有摄像头画面
                    base_name = f"{counter:04d}"
                    multi_cam.save_frames_all(
                        base_name=base_name,
                        base_path=save_path,
                        img_type="jpg"
                    )
                    print(f"已保存第 {counter} 组画面")
                    counter += 1
            except queue.Empty:
                time.sleep(0.01)  # 避免CPU空转
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


def cal_interface(dir_path, intri_path, cams=None, pattern=(7,5), grid=0.1):

    print("图像获取开始")
    sync_frame_capture(cams,dir_path)
    print("开始检测角点")
    det_board(dir_path, pattern, grid)
    print("开始计算")
    calib_extri(dir_path, intri_path, 1)
    print("标定结束")

if __name__ == '__main__':

    dir_path = 'D:\Desktop\EveryThing\WorkProject\BinocularPose\demo_data\cal_test'
    intri_path = 'D:\Desktop\EveryThing\WorkProject\BinocularPose\demo_data\cal_test/intri.yml'

    cal_interface(dir_path, intri_path)
