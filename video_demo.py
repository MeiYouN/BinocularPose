import os
import cv2
import time
import queue
import threading
from typing import List, Tuple

from BinocularPose.camera.MultiCamera import MultiCamera


def dual_camera_recording(path):
    # 初始化参数
    camera_ids = [0, 1]          # 使用两个摄像头
    resolution = (1280, 720)     # 分辨率设置
    fps = 30                     # 帧率
    save_folder = path  # 存储目录
    video_dir = path+"/videos"       # 视频保存路径
    img_dir = path+"/images"
    key_events = queue.Queue()   # 按键事件队列
    video_flag = False

    try:
        # 创建多摄像头控制器
        controller = MultiCamera(
            camera_ids=camera_ids,
            width=resolution[0],
            height=resolution[1],
            fps=fps
        )

        # 定义按键回调函数
        def key_handler(key):
            key_events.put(key)

        # 启动可视化预览（左右布局）
        controller.start_preview(
            layout=(1, 2),  # 1行2列
            scale=0.5,
            key_callback=key_handler
        )

        print("""
        ===== 操作说明 =====
        [s] 保存当前帧截图
        [q] 停止录制并退出
        [r] 开始或停止视频录制
        ===================
        """)

        img_count = 0
        video_count = 0

        # 主控制循环
        running = True
        while running:
            try:
                # 非阻塞获取按键事件
                key = key_events.get_nowait()
                if key in (27, ord('q')):  # ESC或q键
                    running = False
                elif key == ord('s'):
                    # 保存当前帧（带时间戳）
                    controller.save_frames_all(
                        base_name=f"{img_count}",
                        base_path=img_dir,
                        img_type="jpg"
                    )
                    img_count += 1
                    print(f"已保存 {img_count}.jpg")
                elif key == ord('r'):
                    if video_flag:
                        video_flag = False
                        controller.stop_recording_all()
                        print(f"视频{video_count}录制结束")
                        video_count += 1
                    else:
                        video_flag = True
                        # 启动视频录制
                        controller.start_recording_all(
                            folder_name=f"video_{video_count}",
                            base_path=video_dir
                        )
                        print(f"视频{video_count}录制开始")

            except queue.Empty:
                time.sleep(0.01)  # 避免CPU空转

        return True

    except Exception as e:
        print(f"程序运行异常: {str(e)}")
        return False
    finally:
        # 清理资源
        if 'controller' in locals():
            if video_flag:
                controller.stop_recording_all()
            print("正在关闭摄像头...")
            controller.stop_preview()
            controller.close_all()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 创建必要目录
    path = "./demo_date/new_test"
    # 运行主程序
    if dual_camera_recording(path):
        print("程序正常退出")
    else:
        print("程序异常终止")

    # 展示保存内容
    print(f"\n录制内容已保存至：{path}")
    # print(f"视频文件：{os.path.abspath(video_dir)}/{save_folder}/")
    # print(f"截图文件：{os.path.abspath('./screenshots')}/")