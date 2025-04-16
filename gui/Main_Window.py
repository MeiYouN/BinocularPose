import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget
from home_tab import HomeTab
from record_tab import RecordTab
from data_tab import DataTab
from realtime_tab import RealTimeTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MeiYouMocap动作记录分析系统")
        self.setGeometry(100, 100, 800, 600)

        # 创建标签容器
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.home_tab = HomeTab()
        self.record_tab = RecordTab()
        self.data_tab = DataTab()
        self.realtime_tab = RealTimeTab()

        self.tabs.addTab(self.home_tab, "主页")
        self.tabs.addTab(self.record_tab, "视频录制")
        self.tabs.addTab(self.data_tab, "数据分析")
        self.tabs.addTab(self.realtime_tab, "实时姿态")

        # 连接跳转信号
        self.home_tab.switch_to_record.connect(
            lambda: self.tabs.setCurrentIndex(1)
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())