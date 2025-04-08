from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSpacerItem, QFileDialog, QInputDialog, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from pathlib import Path
from config.Config import CONFIG


class HomeTab(QWidget):
    switch_to_record = Signal()

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        # Logo
        logo = QLabel("MotionRecorder Pro")
        logo.setFont(QFont("Arial", 24, QFont.Bold))
        layout.addWidget(logo)

        # 欢迎文本
        welcome = QLabel("欢迎使用动作记录分析系统")
        welcome.setFont(QFont("微软雅黑", 14))
        layout.addWidget(welcome)

        # 按钮区域
        btn_new = QPushButton("创建新记录")
        btn_new.setFixedSize(200, 40)
        btn_load = QPushButton("加载记录")
        btn_load.setFixedSize(200, 40)

        # 添加垂直间距
        layout.addSpacerItem(QSpacerItem(20, 40))
        layout.addWidget(btn_new, 0, Qt.AlignCenter)
        layout.addSpacerItem(QSpacerItem(20, 20))
        layout.addWidget(btn_load, 0, Qt.AlignCenter)
        layout.addSpacerItem(QSpacerItem(20, 40))

        # 连接信号
        btn_new.clicked.connect(self.new_record)
        btn_load.clicked.connect(self.load_record)

        self.setLayout(layout)

    def new_record(self):
        """创建新记录"""
        # 选择父目录
        parent_dir = QFileDialog.getExistingDirectory(
            self, "选择保存目录", str(Path.home()))
        if not parent_dir:
            return

        # 获取文件夹名称
        folder_name, ok = QInputDialog.getText(
            self, '新建记录', '输入记录名称:',
            text="MyMotionRecord"
        )
        if not ok or not folder_name:
            return

        # 创建完整路径
        target_path = Path(parent_dir) / folder_name
        try:
            if target_path.exists():
                raise FileExistsError("记录已存在")
            target_path.mkdir(parents=True)
            CONFIG.workdir = target_path
            CONFIG.calib_dir = CONFIG.workdir / 'calibration'
            self.switch_to_record.emit()
            # QMessageBox.information(self, "成功", f"记录已创建在:\n{target_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建失败: {str(e)}")

    def load_record(self):
        """加载现有记录"""
        selected_dir = QFileDialog.getExistingDirectory(
            self, "选择记录目录", str(Path.home()))
        if not selected_dir:
            return

        target_path = Path(selected_dir)
        if not target_path.is_dir():
            QMessageBox.warning(self, "错误", "无效的目录")
            return

        CONFIG.workdir = target_path
        CONFIG.calib_dir = CONFIG.workdir / 'calibration'
        self.switch_to_record.emit()
        QMessageBox.information(self, "成功", f"已加载记录:\n{target_path}")