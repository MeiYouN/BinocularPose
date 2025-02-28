import os
import cv2
import time
import queue
import threading
from typing import List, Tuple

def dual_camera_recording():
    # 初始化参数
    camera_ids = [0, 1]          # 使用两个摄像头
    resolution = (1280, 720)     # 分辨率设置
    fps = 30                     # 帧率
    save_folder = "dual_record"  # 存储目录
    video_dir = "./videos"       # 视频保存路径
    key_events = queue.Queue()   # 按键事件队列

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

        # 启动视频录制
        controller.start_recording_all(
            folder_name=save_folder,
            base_path=video_dir
        )

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
        ===================
        """)

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
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    controller.save_frames_all(
                        base_name=f"snapshot_{timestamp}",
                        base_path="./screenshots",
                        img_type="png"
                    )
                    print(f"已保存时间点 {timestamp} 的截图")
            except queue.Empty:
                time.sleep(0.01)  # 避免CPU空转

        return True

    except Exception as e:
        print(f"程序运行异常: {str(e)}")
        return False
    finally:
        # 清理资源
        if 'controller' in locals():
            print("正在停止录制...")
            controller.stop_recording_all()
            print("正在关闭摄像头...")
            controller.close_all()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 创建必要目录
    os.makedirs("./videos", exist_ok=True)
    os.makedirs("./screenshots", exist_ok=True)

    # 运行主程序
    if dual_camera_recording():
        print("程序正常退出")
    else:
        print("程序异常终止")

    # 展示保存内容
    print("\n录制内容已保存至：")
    # print(f"视频文件：{os.path.abspath(video_dir)}/{save_folder}/")
    # print(f"截图文件：{os.path.abspath('./screenshots')}/")