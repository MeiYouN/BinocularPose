import time

from BinocularPose.camera.MultiCamera import MultiCamera

if __name__ == "__main__":
    # 使用示例
    controller = MultiCamera(
        camera_ids=[0, 1],
        resolution=(2048, 1536),
        fps=30,
        work_dir="./experiment_data"
    )

    try:
        while controller.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        controller.shutdown()

    print("程序已退出")