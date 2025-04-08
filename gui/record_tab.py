import shutil
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QPushButton, QLineEdit,
    QLabel, QVBoxLayout, QFileDialog, QMessageBox,
    QHBoxLayout
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
import cv2

from BinocularPose.camera.MultiCamera import MultiCamera
from BinocularPose.models.hrnet.hrnet_api import config_
from config.Config import CONFIG
from interface.cal_interface import cal_interface


class RecordTab(QWidget):
    update_frame = Signal(QImage)  # 视频帧更新信号

    def __init__(self):
        super().__init__()
        self.camera = None
        self.is_recording = False
        self.calib_files_loaded = False
        self.init_ui()
        self.setup_camera_check()
        self.update_frame.connect(self.show_frame)

    def init_ui(self):
        main_layout = QVBoxLayout()

        # 控制面板 -------------------------------------------------
        control_layout = QGridLayout()

        # 第一行控件
        self.btn_camera = QPushButton("打开摄像头")
        self.btn_calibrate = QPushButton("校准摄像头")
        self.btn_import = QPushButton("导入校准")
        self.lbl_calib = QLineEdit()
        self.lbl_calib.setReadOnly(True)
        self.lbl_calib.setPlaceholderText("请选择和创建校准文件")

        # 第二行控件
        self.btn_record = QPushButton("开始录制")
        self.txt_action = QLineEdit()
        self.txt_action.setPlaceholderText("输入动作名称...")

        # 棋盘参数
        hbox1 = QHBoxLayout()
        hbox1.addWidget(QLabel("棋盘大小(mm):"))
        self.txt_chessboard = QLineEdit()
        self.txt_chessboard.setPlaceholderText("20")
        hbox1.addWidget(self.txt_chessboard)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(QLabel("角点格式:"))
        self.txt_corner_w = QLineEdit()
        self.txt_corner_w.setPlaceholderText("5")
        hbox2.addWidget(self.txt_corner_w)
        hbox2.addWidget(QLabel("x"))
        self.txt_corner_h = QLineEdit()
        self.txt_corner_h.setPlaceholderText("7")
        hbox2.addWidget(self.txt_corner_h)

        # 布局管理
        control_layout.addWidget(self.btn_camera, 0, 0)
        control_layout.addWidget(self.btn_calibrate, 0, 1)
        control_layout.addWidget(self.btn_import, 0, 2)
        control_layout.addWidget(self.lbl_calib, 0, 3)

        control_layout.addWidget(self.btn_record, 1, 0)
        control_layout.addWidget(self.txt_action, 1, 1)
        control_layout.addLayout(hbox1, 1, 2)
        control_layout.addLayout(hbox2, 1, 3)

        # 视频显示区域 ---------------------------------------------
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: rgba(0, 0, 0, 50);
            min-height: 200px;
            border: 2px dashed #666;
        """)
        self.set_camera_overlay("点击打开摄像头")

        # 组合布局
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.video_label)

        # 连接信号
        self.btn_camera.clicked.connect(self.toggle_camera)
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_calibrate.clicked.connect(self.calibrate)
        self.btn_import.clicked.connect(self.import_calibration)

        # 定时器初始化
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video_frame)

        self.setLayout(main_layout)

    def setup_camera_check(self):
        """检查校准文件是否存在"""
        if CONFIG.workdir:
            calib_dir = CONFIG.calib_dir
            if (calib_dir / "intri.yml").exists() and (calib_dir / "extri.yml").exists():
                self.lbl_calib.setText(str(calib_dir))
                self.calib_files_loaded = True

    def set_camera_overlay(self, text):
        """设置摄像头预览覆盖文本"""
        self.video_label.setText(f"""
            <div style='color:white; font-size:24px;'>
                {text}
            </div>
        """)

    def import_calibration(self):
        """导入校准文件"""
        selected_dir = QFileDialog.getExistingDirectory(
            self, "选择校准文件目录", str(Path.cwd()))
        if not selected_dir:
            return

        src_dir = Path(selected_dir)
        required_files = ["intri.yml", "extri.yml"]
        missing = [f for f in required_files if not (src_dir / f).exists()]

        if missing:
            QMessageBox.critical(self, "错误", f"缺少必要文件: {', '.join(missing)}")
            return

        # 创建目标目录
        calib_dir = CONFIG.calib_dir
        calib_dir.mkdir(parents=True, exist_ok=True)

        try:
            for f in required_files:
                shutil.copy(src_dir / f, calib_dir / f)
            self.lbl_calib.setText(str(calib_dir))
            self.calib_files_loaded = True
            QMessageBox.information(self, "成功", "校准文件导入成功")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"文件复制失败: {str(e)}")

    def toggle_camera(self):
        """开关摄像头"""
        if self.camera:
            self.close_camera()
        else:
            self.open_camera()

    def open_camera(self):
        """打开摄像头"""
        if not self.calib_files_loaded:
            QMessageBox.warning(self, "警告", "请先导入校准文件")
            return

        # self.camera = cv2.VideoCapture(0)
        self.camera = MultiCamera(
            camera_ids=CONFIG.camera_ids,
            width=CONFIG.resolution[0],
            height=CONFIG.resolution[1],
            fps=CONFIG.fps
        )

        self.btn_camera.setText("关闭摄像头")
        self.set_camera_overlay("")
        self.timer.start(30)  # 30ms更新一帧

    def close_camera(self):
        """关闭摄像头"""
        if self.camera:
            self.camera.close_all()
            self.camera = None
        self.timer.stop()
        self.btn_camera.setText("打开摄像头")
        self.set_camera_overlay("摄像头已关闭")
        self.video_label.setPixmap(QPixmap())

    def update_video_frame(self):
        """更新视频帧"""
        frames = self.camera.get_frames()

        frame = self.camera.arrange_frames(frames,1,2)
        h, w, ch = frame.shape
        # w, h = w//3, h//3
        # frame = cv2.resize(frame, (w, h))
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.update_frame.emit(q_img)

    def show_frame(self, image):
        """显示视频帧"""
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.width(),  # 使用固定宽度
            400,
            Qt.KeepAspectRatio,
        ))

    def get_calibration_params(self):
        """获取校准参数"""
        try:
            pattern_size = (
                int(self.txt_corner_w.text() or 7),
                int(self.txt_corner_h.text() or 5)
            )
            square_size = int(self.txt_chessboard.text() or 100)
            return pattern_size, square_size
        except ValueError:
            QMessageBox.critical(self, "错误", "无效的校准参数")
            return None

    def calibrate(self):
        """执行校准"""
        params = self.get_calibration_params()
        if not params:
            return

        calib_dir = CONFIG.calib_dir
        calib_dir.mkdir(parents=True, exist_ok=True)

        pattern_size, square_size = params
        # 这里添加实际校准逻辑
        print(f"开始校准 - 角点格式: {pattern_size}, 棋盘格尺寸: {square_size}mm")
        cal_interface(calib_dir.__str__(), CONFIG.path_intri.__str__(),pattern=pattern_size,grid=square_size/1000)
        # QMessageBox.information(self, "校准", "校准功能需要具体实现")

    def toggle_recording(self):
        """切换录制状态"""
        if not self.txt_action.text().strip():
            QMessageBox.warning(self, "提示", "请先输入动作名称")
            return

        self.is_recording = not self.is_recording
        self.btn_record.setText("停止录制" if self.is_recording else "开始录制")
        action_name = self.txt_action.text()

        if self.is_recording:
            self.camera.start_recording_all(
                folder_name=f"{action_name}",
                base_path=CONFIG.workdir
            )
            print(f"开始录制动作: {action_name}")
        else:
            self.camera.stop_recording_all()
            print(f"停止录制动作: {action_name}")
            # 这里添加实际停止录制逻辑