import os
import subprocess
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MainDashboard(QMainWindow):
    """PU-Learning 主界面，集中入口管理四大功能模块 (PyQt5 版本)。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PU-Learning 综合分析平台")
        self.resize(1360, 820)
        self.setMinimumSize(1100, 700)

        self.modules = [
            {
                "name": "数据准备",
                "desc": "DAT/H5 数据预处理、合并与切割",
                "path": os.path.join(BASE_DIR, "数据准备", "main.py"),
                "script": "main.py",
                "candidates": ["数据准备"],
            },
            {
                "name": "模型训练",
                "desc": "PU 学习训练与日志监控",
                "path": os.path.join(BASE_DIR, "模型训练", "main.py"),
                "script": "main.py",
                "candidates": ["模型训练"],
            },
            {
                "name": "模型预测",
                "desc": "批量预测与结果可视化",
                "path": os.path.join(BASE_DIR, "模型预测", "main.py"),
                "script": "main.py",
                "candidates": ["模型预测"],
            },
            {
                "name": "模型评估",
                "desc": "GDR/DEC/SHAP 综合评估",
                "path": os.path.join(BASE_DIR, "模型评估", "main.py"),
                "script": "main.py",
                "candidates": ["模型评估"],
            },
        ]

        self.modules.append(
            {
                "name": "模型对比",
                "desc": "导入 model_comparison 模块",
                "path": os.path.join(BASE_DIR, "model_comparison", "main.py"),
                "script": "main.py",
                "candidates": ["model_comparison"],
            }
        )

        self._init_fonts()
        self._build_layout()

    def _init_fonts(self):
        self.header_font = QFont("Microsoft YaHei", 18, QFont.Bold)
        self.subheader_font = QFont("Microsoft YaHei", 11)
        self.button_font = QFont("Microsoft YaHei", 12, QFont.Bold)

    def _build_layout(self):
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(12)

        header = QFrame()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(10, 10, 10, 10)
        header_layout.setSpacing(4)

        title = QLabel("矿产资源大语言模型分析系统")
        title.setFont(self.header_font)
        title.setStyleSheet("color: #1b365d;")
        header_layout.addWidget(title)

        subtitle = QLabel("统一入口 · 一键切换 · 提升多环节协同效率")
        subtitle.setFont(self.subheader_font)
        subtitle.setStyleSheet("color: #4b5563;")
        header_layout.addWidget(subtitle)

        root_layout.addWidget(header)

        body = QFrame()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(18)

        sidebar = self._build_sidebar()
        body_layout.addWidget(sidebar, 0)

        showcase = self._build_showcase()
        body_layout.addWidget(showcase, 1)

        root_layout.addWidget(body, 1)

        status_bar = QLabel("提示：每个功能会在单独窗口中打开，可并行操作。")
        status_bar.setFont(self.subheader_font)
        status_bar.setStyleSheet("color: #4b5563;")
        status_bar.setContentsMargins(4, 8, 4, 0)
        root_layout.addWidget(status_bar)

    def _build_sidebar(self) -> QWidget:
        group = QGroupBox("功能导航")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        for module in self.modules:
            button = QPushButton(module["name"])
            button.setFont(self.button_font)
            button.setCursor(Qt.PointingHandCursor)
            button.setStyleSheet(
                "QPushButton {background-color: #1b365d; color: white; padding: 10px; border-radius: 6px;}"
                "QPushButton:hover {background-color: #243f75;}"
            )
            button.clicked.connect(lambda _, m=module: self._launch_module(m))
            layout.addWidget(button)

            desc = QLabel(module["desc"])
            desc.setWordWrap(True)
            desc.setStyleSheet("color: #6b7280;")
            desc.setFont(self.subheader_font)
            layout.addWidget(desc)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        layout.addWidget(divider)

        workflow_hint = QLabel("工作流建议：先完成数据准备，再开展训练、预测与评估。")
        workflow_hint.setWordWrap(True)
        workflow_hint.setStyleSheet("color: #475467;")
        workflow_hint.setFont(self.subheader_font)
        layout.addWidget(workflow_hint)

        layout.addStretch(1)
        return group

    def _build_showcase(self) -> QWidget:
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame {background-color: white; border: 2px solid #cbd5f5; border-radius: 12px;}"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(16)

        welcome = QLabel("欢迎使用 PU-Learning 综合分析平台")
        welcome.setFont(self.header_font)
        welcome.setStyleSheet("color: #1b365d;")
        layout.addWidget(welcome)

        tips = [
            "▶ 数据准备：原始 DAT/H5 数据预处理、合并与切割",
            "▶ 模型训练：支持多模型/多折交叉验证",
            "▶ 模型预测：批量预测并输出可视化结果",
            "▶ 模型评估：一键生成 GDR / DEC / SHAP",
        ]

        for tip in tips:
            label = QLabel(tip)
            label.setFont(self.subheader_font)
            label.setStyleSheet("color: #4b5563;")
            layout.addWidget(label)

        layout.addStretch(1)
        return frame

    def _resolve_script_path(self, module_meta):
        # 优先使用显式路径
        script_path = module_meta.get("path")
        if script_path and os.path.exists(script_path):
            return script_path

        script_name = module_meta.get("script", "main.py")
        for rel_dir in module_meta.get("candidates", []):
            candidate = os.path.join(BASE_DIR, rel_dir, script_name)
            if os.path.exists(candidate):
                return candidate

        # 兜底：在同级目录中搜索含有目标脚本的子目录
        for entry in os.scandir(BASE_DIR):
            if not entry.is_dir():
                continue
            candidate = os.path.join(entry.path, script_name)
            if os.path.exists(candidate):
                return candidate
        return script_path

    def _launch_module(self, module_meta):
        script_path = self._resolve_script_path(module_meta)
        if not os.path.exists(script_path):
            QMessageBox.critical(self, "文件不存在", f"未找到 {module_meta['name']} 脚本:\n{script_path}")
            return

        try:
            subprocess.Popen([sys.executable, script_path], cwd=os.path.dirname(script_path))
        except Exception as exc:
            QMessageBox.critical(self, "启动失败", f"无法打开 {module_meta['name']}：{exc}")


def main():
    qt_app = QApplication(sys.argv)
    window = MainDashboard()
    window.show()
    sys.exit(qt_app.exec_())


if __name__ == "__main__":
    main()
