import cv2
import socket
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QCheckBox, QLineEdit, QSplitter, QMessageBox, QGridLayout
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from config.Config import CONFIG


class RealTimeTab(QWidget):
    update_frame = Signal(QImage)
    update_3d = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.camera = None
        self.is_running = False
        self.sock = None
        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        main_splitter = QSplitter(Qt.Horizontal)

        # 左侧视频预览区 -------------------------------------------
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: black;
            min-height: 400px;
            border: 2px solid #666;
        """)
        self.set_video_overlay("摄像头未开启")
        main_splitter.addWidget(self.video_label)

        # 右侧3D可视化区 ------------------------------------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 3D画布
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.setup_3d_axes()
        right_layout.addWidget(self.canvas)

        main_splitter.addWidget(right_widget)

        # 顶部控制栏 ----------------------------------------------
        top_control = QHBoxLayout()
        # top_control = QGridLayout()
        self.cb_forward = QCheckBox("实时转发姿态数据")
        self.txt_host = QLineEdit("127.0.0.1")
        self.txt_host.setFixedWidth(120)
        self.txt_port = QLineEdit("12345")
        self.txt_port.setFixedWidth(80)
        self.btn_test = QPushButton("测试连接")
        self.btn_test.setFixedWidth(80)
        self.lbl_status = QLabel("就绪")

        top_control.addWidget(self.cb_forward)
        top_control.addWidget(QLabel("目标地址:"))
        top_control.addWidget(self.txt_host)
        top_control.addWidget(QLabel(":"))
        top_control.addWidget(self.txt_port)
        top_control.addWidget(self.btn_test)
        top_control.addWidget(self.lbl_status)
        top_control.addStretch()

        # 底部控制栏 ----------------------------------------------
        bottom_control = QHBoxLayout()
        self.btn_camera = QPushButton("打开摄像头")
        self.btn_camera.setFixedWidth(120)
        self.btn_control = QPushButton("开始运行")
        self.btn_control.setFixedWidth(120)
        self.btn_control.setEnabled(False)

        bottom_control.addStretch()
        bottom_control.addWidget(self.btn_camera)
        bottom_control.addSpacing(20)
        bottom_control.addWidget(self.btn_control)
        bottom_control.addStretch()

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_control)
        main_layout.addWidget(main_splitter)
        main_layout.addLayout(bottom_control)
        self.setLayout(main_layout)

        # 定时器
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video)

    def setup_connections(self):
        """初始化信号连接"""
        self.btn_camera.clicked.connect(self.toggle_camera)
        self.btn_control.clicked.connect(self.toggle_operation)
        self.btn_test.clicked.connect(self.test_connection)
        self.update_frame.connect(self.show_video_frame)
        self.update_3d.connect(self.update_3d_plot)

    def setup_3d_axes(self):
        """初始化3D坐标系"""
        self.ax.clear()
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.canvas.draw()

    def set_video_overlay(self, text):
        """设置视频覆盖文本"""
        self.video_label.setText(f"""
            <div style='color:white; font-size:24px;'>
                {text}
            </div>
        """)

    def toggle_camera(self):
        """开关摄像头"""
        if self.camera and self.camera.isOpened():
            self.close_camera()
        else:
            self.open_camera()

    def open_camera(self):
        """打开摄像头"""
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头")
            return

        self.btn_camera.setText("关闭摄像头")
        self.btn_control.setEnabled(True)
        self.set_video_overlay("")
        self.video_timer.start(30)

    def close_camera(self):
        """关闭摄像头"""
        if self.camera:
            self.camera.release()
        self.video_timer.stop()
        self.btn_camera.setText("打开摄像头")
        self.btn_control.setEnabled(False)
        self.set_video_overlay("摄像头已关闭")
        self.video_label.setPixmap(QPixmap())

    def update_video(self):
        """更新视频帧"""
        ret, frame = self.camera.read()
        if ret:
            # 视频显示
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.update_frame.emit(q_img)

            # TODO: 在此处添加姿态估计算法
            # 示例数据，替换为实际姿态数据
            dummy_data = np.random.rand(17, 3)  # 17个关节点
            self.update_3d.emit(dummy_data)

            # 转发数据
            if self.cb_forward.isChecked():
                self.send_data(dummy_data)

    def show_video_frame(self, q_img):
        """显示视频帧"""
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def update_3d_plot(self, joints):
        """更新3D姿态显示"""
        self.ax.clear()
        self.ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='r', s=50)
        self.setup_3d_axes()
        self.canvas.draw()

    def toggle_operation(self):
        """切换运行状态"""
        self.is_running = not self.is_running
        self.btn_control.setText("停止运行" if self.is_running else "开始运行")
        self.lbl_status.setText("运行中" if self.is_running else "已停止")

    def test_connection(self):
        """测试网络连接"""
        # 实现同第三个界面
        pass

    def send_data(self, data):
        """发送姿态数据"""
        # 实现同第三个界面
        pass

    def closeEvent(self, event):
        """关闭时释放资源"""
        self.close_camera()
        super().closeEvent(event)