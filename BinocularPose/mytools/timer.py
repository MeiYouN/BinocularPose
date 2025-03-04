import time


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
