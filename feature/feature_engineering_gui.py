# -*- coding: utf-8 -*-
"""GRD 数据融合到 H5 的数据准备界面。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from .grid_to_h5 import combine_grid_files_to_h5
except ImportError:
    from grid_to_h5 import combine_grid_files_to_h5


class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass


class Worker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    log_message = pyqtSignal(str)

    def __init__(self, input_files, output_file, normalize_method="none", params_path=None):
        super().__init__()
        self.input_files = input_files
        self.output_file = output_file
        self.normalize_method = normalize_method
        self.params_path = params_path

    def run(self):
        try:
            self.log_message.emit("=" * 50)
            self.log_message.emit("开始 GRD -> H5 数据融合任务")
            self.log_message.emit("=" * 50)
            self.log_message.emit(f"输入文件数量: {len(self.input_files)}")
            self.log_message.emit(f"输出文件: {os.path.basename(self.output_file)}")
            self.log_message.emit(f"标准化方式: {self.normalize_method if self.normalize_method != 'none' else '无'}")
            if self.params_path:
                self.log_message.emit(f"外部参数文件: {os.path.basename(self.params_path)}")
            self.log_message.emit("")

            combine_grid_files_to_h5(
                self.input_files,
                self.output_file,
                normalize_method=self.normalize_method,
                params_path=self.params_path,
                log_func=self.log_message.emit,
            )

            self.log_message.emit("")
            self.log_message.emit("=" * 50)
            self.log_message.emit("数据融合任务全部完成！")
            self.log_message.emit("=" * 50)
        except Exception as exc:
            self.log_message.emit("")
            self.log_message.emit("=" * 50)
            self.log_message.emit(f"数据融合任务失败: {exc}")
            self.log_message.emit("=" * 50)
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


class DataPreparationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GRD 数据融合")
        self.resize(1120, 840)
        self.thread = None
        self.worker = None
        self._build_ui()

        self.stream = EmittingStream()
        self.stream.textWritten.connect(self.log_message)
        sys.stdout = self.stream

    def _build_ui(self):
        self.setStyleSheet(
            """
            QMainWindow { background: #f5f7fb; }
            QLabel { color: #1f2937; font-size: 13px; }
            QGroupBox {
                border: 1px solid #dbe3f0;
                border-radius: 10px;
                margin-top: 12px;
                background: white;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QLineEdit, QTextEdit, QListWidget, QComboBox {
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 6px;
                background: white;
            }
            QPushButton {
                background: #1b365d;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 14px;
                font-weight: 600;
            }
            QPushButton:hover { background: #243f75; }
            QPushButton:disabled { background: #94a3b8; }
            """
        )

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        header = QFrame()
        header.setStyleSheet("QFrame {background: white; border: 1px solid #dbe3f0; border-radius: 12px;}")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(18, 14, 18, 14)
        header_layout.setSpacing(4)

        title = QLabel("GRD -> H5 数据融合")
        title.setStyleSheet("font-size: 22px; font-weight: 700; color: #1b365d;")
        header_layout.addWidget(title)

        subtitle = QLabel("选择一个或多个 GRD 文件，按顺序融合为旧系统风格的 fused_features + metadata H5。")
        subtitle.setStyleSheet("color: #64748b;")
        header_layout.addWidget(subtitle)
        root.addWidget(header)

        input_group = QGroupBox("输入设置")
        input_layout = QVBoxLayout(input_group)
        input_layout.setSpacing(10)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("GRD 文件目录:"))
        self.dir_edit = QLineEdit()
        self.dir_edit.setReadOnly(True)
        dir_row.addWidget(self.dir_edit, 1)
        browse_dir_btn = QPushButton("浏览...")
        browse_dir_btn.clicked.connect(self.select_directory)
        dir_row.addWidget(browse_dir_btn)
        input_layout.addLayout(dir_row)
        root.addWidget(input_group)

        list_group = QGroupBox("文件选择与排序")
        list_row = QHBoxLayout(list_group)
        list_row.setSpacing(12)

        left = QVBoxLayout()
        left.addWidget(QLabel("可用 GRD 文件"))
        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        left.addWidget(self.available_list)
        list_row.addLayout(left, 1)

        mid = QVBoxLayout()
        mid.addStretch(1)
        btn_add = QPushButton(">")
        btn_add.clicked.connect(self.add_selected)
        btn_add_all = QPushButton(">>")
        btn_add_all.clicked.connect(self.add_all)
        btn_remove = QPushButton("<")
        btn_remove.clicked.connect(self.remove_selected)
        btn_remove_all = QPushButton("<<")
        btn_remove_all.clicked.connect(self.remove_all)
        for btn in (btn_add, btn_add_all, btn_remove, btn_remove_all):
            btn.setMinimumWidth(54)
            mid.addWidget(btn)
        mid.addStretch(1)
        list_row.addLayout(mid)

        right = QVBoxLayout()
        right.addWidget(QLabel("待融合文件（可拖拽排序）"))
        self.selected_list = QListWidget()
        self.selected_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.selected_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.selected_list.setDefaultDropAction(Qt.MoveAction)
        right.addWidget(self.selected_list)
        list_row.addLayout(right, 1)
        root.addWidget(list_group, 1)

        output_group = QGroupBox("输出设置")
        output_layout = QVBoxLayout(output_group)
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("输出 H5 文件:"))
        self.output_edit = QLineEdit()
        output_row.addWidget(self.output_edit, 1)
        browse_out_btn = QPushButton("另存为...")
        browse_out_btn.clicked.connect(self.select_output_file)
        output_row.addWidget(browse_out_btn)
        output_layout.addLayout(output_row)
        self.mode_label = QLabel("提示：当前版本会输出旧系统兼容的 fused_features + metadata 结构。")
        self.mode_label.setStyleSheet("color: #64748b;")
        output_layout.addWidget(self.mode_label)
        root.addWidget(output_group)

        norm_group = QGroupBox("标准化设置")
        norm_layout = QVBoxLayout(norm_group)
        norm_layout.setSpacing(8)

        norm_row = QHBoxLayout()
        norm_row.addWidget(QLabel("标准化方式:"))
        self.normalization_combo = QComboBox()
        self.normalization_combo.addItems(["Z-score 标准化", "0-1 标准化", "不处理"])
        self.normalization_combo.setCurrentIndex(2)
        self.normalization_combo.setToolTip("选择对每个特征通道应用的预处理方法")
        norm_row.addWidget(self.normalization_combo, 1)
        norm_layout.addLayout(norm_row)

        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("标准化参数文件(可选):"))
        self.params_path_edit = QLineEdit()
        self.params_path_edit.setPlaceholderText("如果选择，将使用此 JSON 参数文件进行标准化")
        params_row.addWidget(self.params_path_edit, 1)
        browse_params_btn = QPushButton("浏览...")
        browse_params_btn.clicked.connect(self.select_params_file)
        params_row.addWidget(browse_params_btn)
        norm_layout.addLayout(params_row)

        norm_hint = QLabel("提示：若不提供参数文件且选择标准化，系统会自动生成同名 JSON 参数文件。")
        norm_hint.setStyleSheet("color: #64748b;")
        norm_layout.addWidget(norm_hint)
        root.addWidget(norm_group)

        action_row = QHBoxLayout()
        action_row.addStretch(1)
        self.run_btn = QPushButton("开始融合")
        self.run_btn.clicked.connect(self.start_fusion)
        action_row.addWidget(self.run_btn)
        root.addLayout(action_row)

        log_group = QGroupBox("日志输出")
        log_layout = QVBoxLayout(log_group)
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMinimumHeight(260)
        self.log_console.setStyleSheet("background: #f8fafc;")
        log_layout.addWidget(self.log_console)
        root.addWidget(log_group, 1)

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "选择 GRD 文件所在目录")
        if directory:
            self.dir_edit.setText(directory)
            self.populate_file_list(directory)

    def populate_file_list(self, directory):
        self.available_list.clear()
        self.selected_list.clear()
        grd_files = sorted(Path(directory).glob("*.grd"))
        if not grd_files:
            self.log_message("警告：该目录下没有找到 .grd 文件。")
            return
        for file_path in grd_files:
            item = QListWidgetItem(os.path.basename(file_path))
            item.setData(Qt.UserRole, str(file_path))
            self.available_list.addItem(item)

    def add_selected(self):
        items = list(self.available_list.selectedItems())
        for item in items:
            self.selected_list.addItem(self.available_list.takeItem(self.available_list.row(item)))

    def add_all(self):
        while self.available_list.count() > 0:
            self.selected_list.addItem(self.available_list.takeItem(0))

    def remove_selected(self):
        items = list(self.selected_list.selectedItems())
        for item in items:
            self.available_list.addItem(self.selected_list.takeItem(self.selected_list.row(item)))
        self.available_list.sortItems()

    def remove_all(self):
        while self.selected_list.count() > 0:
            self.available_list.addItem(self.selected_list.takeItem(0))
        self.available_list.sortItems()

    def select_output_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "保存 H5 文件", "", "HDF5 Files (*.h5);;All Files (*)")
        if file_path:
            if not file_path.lower().endswith(".h5"):
                file_path += ".h5"
            self.output_edit.setText(file_path)

    def select_params_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择标准化参数文件", "", "JSON Files (*.json);;All Files (*)")
        if file_path:
            self.params_path_edit.setText(file_path)

    def start_fusion(self):
        if self.selected_list.count() == 0:
            self.show_error("请至少选择一个 GRD 文件。")
            return

        output_path = self.output_edit.text().strip()
        if not output_path:
            self.show_error("请指定输出 H5 文件路径。")
            return

        normalize_map = {
            "Z-score 标准化": "zscore",
            "0-1 标准化": "normalize",
            "不处理": "none",
        }
        normalize_method = normalize_map.get(self.normalization_combo.currentText(), "none")
        params_path = self.params_path_edit.text().strip()
        if params_path and normalize_method == "none":
            reply = QMessageBox.question(
                self,
                "确认操作",
                "你选择了一个参数文件，但标准化方式为“不处理”。参数文件将被忽略，是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

        input_files = [self.selected_list.item(i).data(Qt.UserRole) for i in range(self.selected_list.count())]

        self.run_btn.setEnabled(False)
        self.log_console.clear()
        self.log_message("开始数据融合，请稍候...")

        self.thread = QThread()
        self.worker = Worker(
            input_files,
            output_path,
            normalize_method=normalize_method,
            params_path=params_path or None,
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.log_message.connect(self.log_message)
        self.thread.start()

    def log_message(self, message):
        text = str(message).rstrip()
        if text:
            self.log_console.append(text)
            bar = self.log_console.verticalScrollBar()
            bar.setValue(bar.maximum())

    def on_finished(self):
        self.run_btn.setEnabled(True)
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        QMessageBox.information(self, "完成", "数据融合任务已完成。")

    def on_error(self, message):
        self.run_btn.setEnabled(True)
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        self.show_error(message)

    def show_error(self, message):
        QMessageBox.critical(self, "错误", message)

    def closeEvent(self, event):
        sys.stdout = sys.__stdout__
        super().closeEvent(event)


def show_data_preparation_app():
    app = QApplication.instance() or QApplication(sys.argv)
    window = DataPreparationApp()
    window.show()
    return window


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataPreparationApp()
    window.show()
    sys.exit(app.exec_())
