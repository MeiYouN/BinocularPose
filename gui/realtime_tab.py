import json
import os
import threading

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

from BinocularPose.camera.MultiCamera import MultiCamera
from BinocularPose.mytools.json_file import JsonFile
from BinocularPose.mytools.load_para import load_yml
from config.Config import CONFIG
connnection = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
               [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4]]

class RealTimeTab(QWidget):
    update_frame = Signal(QImage)
    update_3d = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.connection_status = None
        self.frames = None
        self.camera = None
        self.is_running = False
        self.sock = None
        self.init_ui()
        self.setup_connections()

        # 实时预测线程
        self.online_thread = None

    def init_ui(self):
        main_splitter = QSplitter(Qt.Horizontal)

        # 左侧视频预览区 -------------------------------------------
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: rgba(0, 0, 0, 50);
            min-weight: 600px;
            max-weight: 800px;
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

        self.txt_action = QLineEdit()
        self.txt_action.setPlaceholderText("输入动作名称...")

        bottom_control.addStretch()
        bottom_control.addWidget(self.btn_camera)
        bottom_control.addSpacing(20)
        bottom_control.addWidget(self.txt_action)
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
        self.ax.set_xlim3d(-1, 1)
        self.ax.set_ylim3d(-1, 1.1)
        self.ax.set_zlim3d(0, 1.8)
        self.canvas.draw()

    def set_video_overlay(self, text):
        """设置视频覆盖文本"""
        self.video_label.setText(f"""
            <div style='color:white; font-size:20px;'>
                {text}
            </div>
        """)

    def toggle_camera(self):
        """开关摄像头"""
        if self.camera:
            self.close_camera()
        else:
            self.open_camera()

    def open_camera(self):
        """打开摄像头"""

        self.camera = MultiCamera(
            camera_ids=CONFIG.camera_ids,
            width=CONFIG.resolution[0],
            height=CONFIG.resolution[1],
            fps=CONFIG.fps
        )

        self.btn_camera.setText("关闭摄像头")
        self.btn_control.setEnabled(True)
        self.set_video_overlay("")
        self.video_timer.start(33)

    def close_camera(self):
        """关闭摄像头"""
        if self.camera:
            self.camera.close_all()
            self.camera = None
        self.video_timer.stop()
        self.btn_camera.setText("打开摄像头")
        self.btn_control.setEnabled(False)
        self.set_video_overlay("摄像头已关闭")
        self.video_label.setPixmap(QPixmap())

    def update_video(self):
        """更新视频帧"""
        self.frames = self.camera.get_frames()

        frame = self.camera.arrange_frames(self.frames, 2, 1)
        h, w, ch = frame.shape
        # w, h = w//3, h//3
        # frame = cv2.resize(frame, (w, h))
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.update_frame.emit(q_img)



    def show_video_frame(self, q_img):
        """显示视频帧"""
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def update_3d_plot(self, keypoints3d):
        """更新3D姿态显示"""
        self.ax.clear()
        self.ax.set_xlim3d(-1, 1)
        self.ax.set_ylim3d(-1, 1.1)
        self.ax.set_zlim3d(0, 1.8)
        self.ax.scatter(keypoints3d[:, 0], keypoints3d[:, 1], keypoints3d[:, 2], s=10)
        for _c in connnection:
            self.ax.plot([keypoints3d[_c[0], 0], keypoints3d[_c[1], 0]],
                         [keypoints3d[_c[0], 1], keypoints3d[_c[1], 1]],
                         [keypoints3d[_c[0], 2], keypoints3d[_c[1], 2]], 'g')

        self.canvas.draw()

    def toggle_operation(self):
        """切换运行状态"""
        self.is_running = not self.is_running
        self.btn_control.setText("停止运行" if self.is_running else "开始运行")
        self.lbl_status.setText("运行中" if self.is_running else "已停止")
        if self.is_running:
            self.stop_recording()
        else:
            self.start_recording()


    def start_recording(self, live_pose=True):

        if not self.txt_action.text().strip():
            QMessageBox.warning(self, "提示", "请先输入动作名称")
            return
        action_name = self.txt_action.text()

        self.camera.start_recording_all(
            folder_name=f"{action_name}",
            base_path=CONFIG.workdir
        )
        if live_pose:
            self.online_thread = threading.Thread(
                target=self.online_pose,
                args=action_name,
                daemon=True
            )
            self.is_running = True
            self.online_thread.start()

    def stop_recording(self):
        self.is_running = False
        if self.online_thread is not None:
            self.online_thread.join()
        self.online_thread = None
        self.camera.stop_recording_all()

    def online_pose(self, pose_name):
        videos_path = os.path.join(str(CONFIG.workdir), pose_name)
        cameras = load_yml(str(CONFIG.calib_dir))

        jf = JsonFile(videos_path, videos_path  + f'run_{pose_name}.json')

        frame_count = 0
        while self.is_running:
            frames = self.frames
            if not 2 == len(frames):
                print(f'视频已结束，共{jf.index}帧')
                break

            keypoints3d = CONFIG.processor.run_process(frames, cameras)
            self.update_3d.emit(keypoints3d)
            # 转发数据
            if self.cb_forward.isChecked():
                current_data = {
                    "frame": frame_count,
                    "position": keypoints3d.tolist(),
                }
                self.send_data(current_data)
            jf.update(keypoints3d)
            frame_count += 1

        jf.save()

    def init_socket(self):
        """初始化网络连接"""
        if not self.cb_forward.isChecked():
            return

        host = self.txt_host.text()
        port = self.txt_port.text()
        if not host or not port.isdigit():
            QMessageBox.warning(self, "错误", "请输入有效的主机地址和端口号")
            return False

        try:
            self.close_connection()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(3)
            self.sock.connect((host, int(port)))
            self.connection_status = True
            return True
        except Exception as e:
            self.connection_status = False
            QMessageBox.critical(self, "连接失败", f"无法连接到服务器: {str(e)}")
            return False



    def close_connection(self):
        """关闭网络连接"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            finally:
                self.sock = None
        self.connection_status = False

    def test_connection(self):
        """测试网络连接"""
        # 实现同第三个界面
        if self.init_socket():
            QMessageBox.information(self, "成功", "连接测试成功")

    def send_data(self, data):
        """发送姿态数据"""
        if not self.connection_status or not self.sock:
            return
        try:
            # 转换为JSON格式
            json_data = json.dumps(data)
            self.sock.sendall(json_data.encode('utf-8') + b'\n')
        except Exception as e:
            self.connection_status = False
            print(f"数据发送失败: {str(e)}")

    def closeEvent(self, event):
        """关闭时释放资源"""
        self.close_camera()
        super().closeEvent(event)