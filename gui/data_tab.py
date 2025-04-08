import json
import socket
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QLabel, QFileDialog, QSplitter,
    QTreeView, QSizePolicy, QMessageBox, QFileSystemModel, QProgressBar, QCheckBox
)
from PySide6.QtCore import Qt, QDir, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from config.Config import CONFIG


class DataTab(QWidget):
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.play_timer = QTimer(self)
        self.current_frame = 0
        self.is_playing = False
        self.init_ui()
        self.setup_file_tree()
        self.setup_connections()

        self.check_state = False
        self.sock = None
        self.connection_status = False

    def init_ui(self):
        main_splitter = QSplitter(Qt.Horizontal)

        # 左侧文件树 -----------------------------------------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # 刷新按钮
        self.btn_refresh = QPushButton("刷新目录")
        self.btn_refresh.setFixedWidth(80)
        left_layout.addWidget(self.btn_refresh, 0, Qt.AlignLeft)

        self.file_tree = QTreeView()
        left_layout.addWidget(self.file_tree)
        main_splitter.addWidget(left_widget)

        # 右侧主区域 -----------------------------------------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 控制区域
        control_layout = QVBoxLayout()

        # 第一行控制
        row1 = QHBoxLayout()
        self.btn_offline = QPushButton("离线姿态估计")
        self.btn_offline.setFixedWidth(100)
        self.txt_action_folder = QLineEdit()
        self.txt_action_folder.setPlaceholderText("选择动作名称文件夹...")


        row1.addWidget(self.btn_offline)
        row1.addWidget(self.txt_action_folder)

        # 第二行控制
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("加载现有动作文件:"))
        self.btn_load = QPushButton("加载数据")
        self.btn_load.setFixedWidth(100)
        self.txt_file_path = QLineEdit()
        self.txt_file_path.setReadOnly(True)
        self.btn_select_file = QPushButton("...")
        self.btn_select_file.setFixedWidth(40)

        row2.addWidget(self.btn_load)
        row2.addWidget(self.txt_file_path)
        row2.addWidget(self.btn_select_file)

        # 第三行控制
        row3 = QHBoxLayout()
        self.cb_forward = QCheckBox("转发姿态数据")
        self.txt_host = QLineEdit()
        self.txt_host.setPlaceholderText("主机地址")
        self.txt_host.setFixedWidth(120)
        self.txt_port = QLineEdit()
        self.txt_port.setPlaceholderText("端口")
        self.txt_port.setFixedWidth(80)
        self.btn_test_conn = QPushButton("测试连接")
        self.btn_test_conn.setFixedWidth(80)
        self.lbl_conn_status = QLabel()
        self.lbl_conn_status.setFixedWidth(60)
        self.txt_host.setEnabled(False)
        self.txt_port.setEnabled(False)
        self.btn_test_conn.setEnabled(False)

        row3.addWidget(self.cb_forward)
        row3.addWidget(QLabel("目标地址:"))
        row3.addWidget(self.txt_host)
        row3.addWidget(QLabel(":"))
        row3.addWidget(self.txt_port)
        row3.addWidget(self.btn_test_conn)
        row3.addWidget(self.lbl_conn_status)
        row3.addStretch()


        control_layout.addLayout(row1)
        control_layout.addLayout(row2)
        control_layout.addLayout(row3)

        # 图表区域
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.plot_placeholder()

        # 播放控制区域
        playback_layout = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)

        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(60)
        self.btn_restart = QPushButton("↻")
        self.btn_restart.setFixedWidth(60)

        playback_layout.addWidget(self.btn_restart)
        playback_layout.addWidget(self.btn_play)
        playback_layout.addWidget(self.progress)

        # 组合布局
        right_layout.addLayout(control_layout)
        right_layout.addWidget(self.canvas)
        right_layout.addLayout(playback_layout)

        main_splitter.addWidget(right_widget)
        main_splitter.setStretchFactor(1, 3)

        # 初始状态
        self.set_playback_enabled(False)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(main_splitter)
        self.setLayout(main_layout)



    def setup_connections(self):
        """初始化信号连接"""
        self.btn_select_file.clicked.connect(self.select_action_file)
        self.btn_load.clicked.connect(self.load_data)
        self.file_tree.doubleClicked.connect(self.on_tree_double_click)
        self.btn_offline.clicked.connect(self.run_offline_estimation)
        self.btn_refresh.clicked.connect(self.refresh_file_tree)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_restart.clicked.connect(self.restart_playback)
        self.play_timer.timeout.connect(self.update_playback)
        # 连接新信号
        self.cb_forward.stateChanged.connect(self.toggle_forwarding)
        self.btn_test_conn.clicked.connect(self.test_connection)

    def toggle_forwarding(self):
        """切换转发功能状态"""
        self.check_state = not self.check_state

        self.txt_host.setEnabled(self.check_state)
        self.txt_port.setEnabled(self.check_state)
        self.btn_test_conn.setEnabled(self.check_state)
        if not self.check_state:
            self.close_connection()
            self.update_conn_status(False)

    def update_conn_status(self, connected):
        """更新连接状态显示"""
        self.connection_status = connected
        self.lbl_conn_status.setText("已连接" if connected else "未连接")
        color = "green" if connected else "red"
        self.lbl_conn_status.setStyleSheet(f"color: {color};")

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
            self.update_conn_status(True)
            return True
        except Exception as e:
            self.update_conn_status(False)
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
        self.update_conn_status(False)

    def test_connection(self):
        """测试连接按钮点击事件"""
        if self.init_socket():
            QMessageBox.information(self, "成功", "连接测试成功")

    def send_frame_data(self, frame_data):
        """发送当前帧数据"""
        if not self.connection_status or not self.sock:
            return
        try:
            # 转换为JSON格式
            json_data = json.dumps(frame_data)
            self.sock.sendall(json_data.encode('utf-8') + b'\n')
        except Exception as e:
            self.update_conn_status(False)
            print(f"数据发送失败: {str(e)}")

    def setup_file_tree(self):
        """初始化文件树"""
        self.model = QFileSystemModel()
        self.model.setRootPath(QDir.rootPath())
        self.file_tree.setModel(self.model)
        self.file_tree.setSortingEnabled(True)
        self.file_tree.setColumnWidth(0, 200)
        for col in range(1, 4):
            self.file_tree.hideColumn(col)
        self.update_file_tree_root()

    def update_file_tree_root(self):
        """更新文件树根目录"""
        if CONFIG.workdir:
            root_index = self.model.index(str(CONFIG.workdir))
            self.file_tree.setRootIndex(root_index)

    def refresh_file_tree(self):
        """刷新文件树"""
        self.update_file_tree_root()

    def on_tree_double_click(self, index):
        """处理文件树双击事件"""
        path = self.model.filePath(index)
        if path.endswith('.json'):
            self.txt_file_path.setText(path)
            self.current_file = Path(path)

    def select_action_folder(self):
        """选择动作文件夹"""
        folder = QFileDialog.getExistingDirectory(
            self, "选择动作文件夹",
            str(CONFIG.workdir) if CONFIG.workdir else ""
        )
        if folder:
            self.txt_action_folder.setText(folder)

    def select_action_file(self):
        """选择动作文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择动作文件",
            str(CONFIG.workdir) if CONFIG.workdir else "",
            "JSON Files (*.json)"
        )
        if file_path:
            self.txt_file_path.setText(file_path)
            self.current_file = Path(file_path)

    def plot_placeholder(self):
        """绘制空图表"""
        self.ax.clear()
        self.canvas.draw()

    def validate_json(self, file_path):
        """验证JSON文件格式"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            required = ['positions', 'rotations']
            return all(key in data for key in required)
        except:
            return False

    def set_playback_enabled(self, enabled):
        """设置播放控件状态"""
        self.btn_play.setEnabled(enabled)
        self.btn_restart.setEnabled(enabled)
        self.progress.setEnabled(enabled)

    def toggle_play(self):
        """切换播放状态"""
        self.is_playing = not self.is_playing
        self.btn_play.setText("⏸" if self.is_playing else "▶")
        if self.is_playing:
            self.play_timer.start(33)  # 20fps
        else:
            self.play_timer.stop()

    def restart_playback(self):
        """重置播放进度"""
        self.current_frame = 0
        self.progress.setValue(0)
        if self.is_playing:
            self.toggle_play()

    def update_playback(self):
        """更新播放进度"""
        if self.current_frame >= 100:
            self.restart_playback()
            return
        self.current_frame += 1
        self.progress.setValue(self.current_frame)

        # 获取当前帧数据（示例）
        current_data = {
            "frame": self.current_frame,
            "position": [0.1, 0.2, 0.3],  # 替换为真实数据
            "timestamp": np.datetime64('now')
        }
        # 发送数据
        self.send_frame_data(current_data)

    def load_data(self):
        """加载数据并可视化"""
        if not self.current_file or not self.current_file.exists():
            QMessageBox.warning(self, "错误", "请先选择有效的动作文件")
            return

        if not self.validate_json(self.current_file):
            QMessageBox.critical(self, "错误", "无效的动作数据格式")
            return

        try:
            with open(self.current_file, 'r') as f:
                motion_data = json.load(f)

            # 示例可视化逻辑
            positions = np.array(motion_data['positions'])
            x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

            self.ax.clear()
            self.ax.scatter(x, y, z, c='r', marker='o')
            self.ax.set_title(f"动作数据可视化: {self.current_file.stem}")
            self.canvas.draw()

            self.set_playback_enabled(True)
            self.progress.setMaximum(100)  # 假设总帧数为100
            self.progress.setValue(0)

            # 初始化网络连接
            if self.cb_forward.isChecked():
                if not self.init_socket():
                    self.cb_forward.setChecked(False)

            # 加载成功后启用播放控件
            self.set_playback_enabled(True)

        except Exception as e:
            self.set_playback_enabled(False)
            QMessageBox.critical(self, "错误", f"数据加载失败: {str(e)}")
            self.plot_placeholder()

    def run_offline_estimation(self):
        """离线姿态估计"""
        folder = self.txt_action_folder.text()
        if not folder:
            QMessageBox.warning(self, "提示", "请先选择动作文件夹")
            return

        # 这里添加实际处理逻辑
        print(f"开始处理文件夹: {folder}")
        QMessageBox.information(self, "提示", "离线姿态估计功能需要具体实现")
        self.set_playback_enabled(False)
        try:
        # ...原有处理逻辑...
            pass
        finally:
            # 处理完成后恢复状态
            self.set_playback_enabled(True)