import os
import csv
import json
import queue
import shutil
import sys
import threading
import gc
import hashlib
from functools import partial

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(CURRENT_DIR)
COMMON_DIR = os.path.join(CODE_ROOT, "common")
for path in (CODE_ROOT, COMMON_DIR):
    if path not in sys.path:
        sys.path.append(path)

from feature_channel_utils import parse_selected_channels
from model_name_utils import WORKFLOW_MODEL_KEYS, get_model_display_name, get_model_display_names, normalize_model_key

import numpy as np
import torch
from PyQt5.QtCore import QEvent, QObject, Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QPlainTextEdit,
)
from torch.utils.data import DataLoader

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

try:
    from PIL import Image
except ImportError:
    Image = None

from predict import PredictionDataset, WindowDataset, load_model, predict_batch, save_predictions
from plot_labels import (
    _fused_probability_columns,
    _positions_to_geo,
    load_data as load_plot_data,
    save_predictions_to_dat,
    save_predictions_to_csv,
    plot_labels_comparison,
)


class _ResizeEventFilter(QObject):
    """Helper to re-render images when scroll viewport changes size."""

    def __init__(self, callback):
        super().__init__()
        self._callback = callback

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Resize:
            self._callback()
        return super().eventFilter(obj, event)


class PredictionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("模型预测界面")
        self.resize(1440, 960)

        self.model_type_combo = None
        self.model_path_edit = None
        self.data_path_edit = None
        self.label_path_edit = None
        self.output_dir_edit = None
        self.batch_size_edit = None
        self.patch_stride_edit = None
        self.device_combo = None
        self.grid_spacing_edit = None
        self.map_generation_combo = None
        self.aggregation_combo = None
        self.run_button = None
        self.status_label = None
        self.preview_label = None
        self.view_button = None
        self.log_view = None
        self.loss_source_table = None
        self.loss_title_edit = None
        self.loss_status_label = None
        self.loss_fig = None
        self.loss_canvas = None
        self.par_threshold_edit = None
        self.par_hit_distance_edit = None
        self.par_feedback_table = None
        self.par_feedback_status_label = None

        self.plot_paths = {}
        self.last_plot_path = None
        self.last_predictions_file = None
        self.last_dat_path = None
        self.last_map_generation_mode = None
        self._preview_pixmap = None
        self.loss_last_summaries = []

        self.log_queue = queue.Queue()
        self.worker_thread = None
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._process_queue)
        self.poll_timer.start(80)
        self.par_feedback_timer = QTimer(self)
        self.par_feedback_timer.setSingleShot(True)
        self.par_feedback_timer.timeout.connect(self._update_par_feedback)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI Construction
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(12)

        tabs = QTabWidget()
        root_layout.addWidget(tabs, 1)

        prediction_tab = QWidget()
        prediction_layout = QVBoxLayout(prediction_tab)
        prediction_layout.setContentsMargins(0, 0, 0, 0)
        prediction_layout.setSpacing(12)
        prediction_layout.addWidget(self._build_form_section())
        prediction_layout.addWidget(self._build_preview_section())
        prediction_layout.addWidget(self._build_log_section(), 1)
        tabs.addTab(prediction_tab, "模型预测")
        tabs.addTab(self._build_loss_curve_tab(), "损失曲线")

    def _build_form_section(self) -> QGroupBox:
        group = QGroupBox("模型预测")
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignRight)

        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(get_model_display_names(WORKFLOW_MODEL_KEYS))
        self.model_type_combo.setCurrentText(get_model_display_name("cnnt"))
        form.addRow("模型类型", self.model_type_combo)

        self.model_path_edit = self._create_path_row(form, "模型文件(可多选)", self._browse_model)
        self.data_path_edit = self._create_path_row(form, "数据文件", self._browse_data)
        self.label_path_edit = self._create_path_row(form, "标签文件", self._browse_label)
        self.output_dir_edit = self._create_path_row(form, "输出目录", self._browse_output, directory=True)
        self.output_dir_edit.setText("predictions")

        self.batch_size_edit = QLineEdit("32")
        self.batch_size_edit.setPlaceholderText("请输入正整数")
        form.addRow("Batch Size", self.batch_size_edit)

        self.patch_stride_edit = QLineEdit("")
        self.patch_stride_edit.setPlaceholderText("留空则沿用训练步长，可填 1 提高预测分辨率")
        self.patch_stride_edit.setToolTip("预测时重新切窗的步长；窗口大小仍沿用训练参数，避免模型输入尺寸不匹配。")
        form.addRow("预测步长", self.patch_stride_edit)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu"])
        self.device_combo.setCurrentText("cuda")
        form.addRow("计算设备", self.device_combo)

        grid_layout = QHBoxLayout()
        self.grid_spacing_edit = QLineEdit("10.0")
        grid_layout.addWidget(self.grid_spacing_edit)
        grid_notice = QLabel("单位: 坐标系长度")
        grid_notice.setStyleSheet("color: #6b7280;")
        grid_layout.addWidget(grid_notice)
        grid_layout.addStretch(1)
        form.addRow("网格间距", grid_layout)

        self.map_generation_combo = QComboBox()
        self.map_generation_combo.addItem("中心点原值", "center")
        self.map_generation_combo.addItem("窗口平均融合", "window_average")
        self.map_generation_combo.addItem("窗口最大融合", "window_max")
        self.map_generation_combo.setCurrentIndex(1)
        form.addRow("成图方式", self.map_generation_combo)

        self.aggregation_combo = QComboBox()
        self.aggregation_combo.addItem("算术平均", "mean")
        self.aggregation_combo.addItem("中位数", "median")
        self.aggregation_combo.setCurrentIndex(0)
        self.aggregation_combo.setToolTip("多选模型时，对每个网格的正类预测概率进行聚合。")
        form.addRow("预测值聚合", self.aggregation_combo)

        btn_layout = QHBoxLayout()
        self.run_button = QPushButton("开始预测")
        self.run_button.setCursor(Qt.PointingHandCursor)
        self.run_button.clicked.connect(self._start_prediction)
        btn_layout.addWidget(self.run_button)

        self.status_label = QLabel("等待开始预测")
        self.status_label.setStyleSheet("color: #2563eb;")
        btn_layout.addWidget(self.status_label)

        form.addRow(btn_layout)
        return group

    def _build_preview_section(self) -> QGroupBox:
        group = QGroupBox("成矿潜力概率图预览")
        v_layout = QVBoxLayout(group)

        self.preview_label = QLabel("尚未生成置信度插值图，请先运行预测")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(260)
        self.preview_label.setStyleSheet("color: #4b5563; border: 1px dashed #cbd5f5; padding: 12px;")
        v_layout.addWidget(self.preview_label, 1)
        v_layout.addWidget(self._build_par_feedback_section())

        btn_row = QHBoxLayout()
        self.view_button = QPushButton("查看大图")
        self.view_button.setEnabled(False)
        self.view_button.clicked.connect(self._open_full_image)
        btn_row.addWidget(self.view_button)

        save_button = QPushButton("另存为图片")
        save_button.clicked.connect(self._save_current_preview)
        btn_row.addWidget(save_button)

        export_csv_button = QPushButton("导出CSV")
        export_csv_button.clicked.connect(self._export_current_probability_csv)
        btn_row.addWidget(export_csv_button)

        v_layout.addLayout(btn_row)
        return group

    def _build_par_feedback_section(self) -> QGroupBox:
        group = QGroupBox("PAR阈值反馈")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("面积阈值(PAR)"))
        self.par_threshold_edit = QLineEdit("10,30")
        self.par_threshold_edit.setPlaceholderText("例如: 10 或 10,30；支持百分比或 0.1,0.3")
        self.par_threshold_edit.setToolTip("按成矿潜力概率从高到低排序，统计累计前 PAR 面积对应的概率阈值和矿点击中率。")
        self.par_threshold_edit.textChanged.connect(self._schedule_par_feedback_update)
        input_row.addWidget(self.par_threshold_edit, 1)

        input_row.addWidget(QLabel("矿点命中距离"))
        self.par_hit_distance_edit = QLineEdit("1")
        self.par_hit_distance_edit.setPlaceholderText("默认 1")
        self.par_hit_distance_edit.setToolTip("单位为坐标系长度；矿点到任一入选高潜力预测单元的距离不超过该值时计为命中。")
        self.par_hit_distance_edit.textChanged.connect(self._schedule_par_feedback_update)
        input_row.addWidget(self.par_hit_distance_edit)

        refresh_button = QPushButton("刷新反馈")
        refresh_button.clicked.connect(self._update_par_feedback)
        input_row.addWidget(refresh_button)

        self.par_feedback_status_label = QLabel("预测完成后可实时调整 PAR 阈值")
        self.par_feedback_status_label.setStyleSheet("color: #6b7280;")
        input_row.addWidget(self.par_feedback_status_label)
        layout.addLayout(input_row)

        self.par_feedback_table = QTableWidget(0, 8)
        self.par_feedback_table.setHorizontalHeaderLabels(
            [
                "PAR阈值",
                "实际PAR",
                "概率阈值",
                "选中单元",
                "选区平均概率",
                "矿点命中",
                "矿点击中率",
                "正样本窗口命中率",
            ]
        )
        self.par_feedback_table.verticalHeader().setVisible(False)
        self.par_feedback_table.horizontalHeader().setStretchLastSection(True)
        self.par_feedback_table.setMinimumHeight(112)
        layout.addWidget(self.par_feedback_table)
        return group

    def _build_log_section(self) -> QGroupBox:
        group = QGroupBox("日志")
        v_layout = QVBoxLayout(group)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        v_layout.addWidget(self.log_view)
        return group

    def _build_loss_curve_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        source_group = QGroupBox("损失曲线数据源")
        source_layout = QHBoxLayout(source_group)

        self.loss_source_table = QTableWidget(0, 4)
        self.loss_source_table.setHorizontalHeaderLabels(["图例名称", "训练图例", "验证图例", "曲线文件/训练目录"])
        self.loss_source_table.horizontalHeader().setStretchLastSection(True)
        self.loss_source_table.verticalHeader().setVisible(False)
        source_layout.addWidget(self.loss_source_table, 1)

        source_buttons = QVBoxLayout()
        add_dir_btn = QPushButton("添加目录...")
        add_dir_btn.clicked.connect(self._add_loss_curve_dir)
        source_buttons.addWidget(add_dir_btn)
        add_file_btn = QPushButton("添加文件...")
        add_file_btn.clicked.connect(self._add_loss_curve_files)
        source_buttons.addWidget(add_file_btn)
        remove_btn = QPushButton("移除选中")
        remove_btn.clicked.connect(self._remove_selected_loss_curve_sources)
        source_buttons.addWidget(remove_btn)
        clear_btn = QPushButton("清空列表")
        clear_btn.clicked.connect(self._clear_loss_curve_sources)
        source_buttons.addWidget(clear_btn)
        source_buttons.addStretch(1)
        source_layout.addLayout(source_buttons)
        layout.addWidget(source_group, 0)

        option_group = QGroupBox("绘图参数")
        option_layout = QFormLayout(option_group)
        option_layout.setLabelAlignment(Qt.AlignRight)
        self.loss_title_edit = QLineEdit("五折空间交叉验证 Loss 曲线")
        option_layout.addRow("图标题", self.loss_title_edit)

        action_row = QHBoxLayout()
        plot_btn = QPushButton("绘制曲线")
        plot_btn.clicked.connect(self._plot_loss_curves)
        action_row.addWidget(plot_btn)
        view_btn = QPushButton("查看大图")
        view_btn.clicked.connect(self._open_loss_curve_fullscreen)
        action_row.addWidget(view_btn)
        save_btn = QPushButton("保存图片")
        save_btn.clicked.connect(self._save_loss_curve_image)
        action_row.addWidget(save_btn)
        export_btn = QPushButton("导出CSV")
        export_btn.clicked.connect(self._export_loss_curve_summary_csv)
        action_row.addWidget(export_btn)
        self.loss_status_label = QLabel("尚未绘制")
        self.loss_status_label.setStyleSheet("color: #2563eb;")
        action_row.addWidget(self.loss_status_label)
        action_row.addStretch(1)
        option_layout.addRow(action_row)
        layout.addWidget(option_group, 0)

        plot_group = QGroupBox("Loss曲线预览")
        plot_layout = QVBoxLayout(plot_group)
        self.loss_fig = Figure(figsize=(8, 4.8), dpi=100)
        self.loss_canvas = FigureCanvas(self.loss_fig)
        plot_layout.addWidget(NavigationToolbar(self.loss_canvas, tab))
        plot_layout.addWidget(self.loss_canvas, 1)
        layout.addWidget(plot_group, 1)
        self._clear_loss_curve_plot()
        return tab

    def _create_path_row(self, form_layout: QFormLayout, label: str, callback, directory: bool = False) -> QLineEdit:
        container = QWidget()
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        line_edit = QLineEdit()
        browse_btn = QPushButton("浏览" if directory else "选择")
        browse_btn.setCursor(Qt.PointingHandCursor)
        browse_btn.clicked.connect(callback)
        h_layout.addWidget(line_edit)
        h_layout.addWidget(browse_btn)
        form_layout.addRow(label, container)
        return line_edit

    # ------------------------------------------------------------------
    # File dialogs
    def _browse_model(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择模型文件",
            filter="模型文件 (*.pth *.pkl);;PyTorch 模型 (*.pth);;Pickle 模型 (*.pkl);;所有文件 (*.*)",
        )
        if paths:
            self.model_path_edit.setText("; ".join(paths))
            self._apply_training_metadata(paths[0])

    def _browse_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", filter="H5 文件 (*.h5);;所有文件 (*.*)")
        if path:
            self.data_path_edit.setText(path)

    def _browse_label(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择标签文件",
            filter="标签/矿点文件 (*.h5 *.txt *.csv *.tsv *.dat);;所有文件 (*.*)",
        )
        if path:
            self.label_path_edit.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_dir_edit.setText(path)

    def _load_training_metadata(self, model_path):
        model_dir = os.path.dirname(model_path) or "."
        params_path = os.path.join(model_dir, "params.json")
        metadata = {}
        resolved_model_dir = model_dir
        if os.path.exists(params_path):
            try:
                with open(params_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as exc:  # noqa: BLE001
                self._queue_log(f"读取训练参数失败: {exc}")
        else:
            repeat_model_dir = self._resolve_repeat_export_model_dir(model_path)
            if repeat_model_dir:
                repeat_params_path = os.path.join(repeat_model_dir, "params.json")
                try:
                    with open(repeat_params_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    resolved_model_dir = repeat_model_dir
                except Exception as exc:  # noqa: BLE001
                    self._queue_log(f"读取重复训练参数失败: {exc}")
        return resolved_model_dir, metadata

    def _resolve_repeat_export_model_dir(self, model_path):
        """Map repeat-summary exports like CNN/model1.pth back to run_001/.../params.json."""
        model_dir = os.path.dirname(model_path) or "."
        base_name = os.path.basename(model_path)
        stem, _ = os.path.splitext(base_name)
        lower_stem = stem.lower()

        if lower_stem.startswith("model"):
            suffix = stem[5:]
            if suffix.isdigit():
                run_dir = os.path.join(model_dir, f"run_{int(suffix):03d}")
                match = self._find_params_dir_under(run_dir)
                if match:
                    return match

        # Some repeat exports can be manually renamed or mistyped; match by file hash.
        return self._find_repeat_params_dir_by_model_hash(model_path)

    def _find_params_dir_under(self, run_dir):
        if not run_dir or not os.path.isdir(run_dir):
            return ""
        candidates = []
        for root, _, files in os.walk(run_dir):
            if "params.json" in files:
                candidates.append(root)
        if not candidates:
            return ""
        candidates.sort(key=lambda path: (len(path), path))
        return candidates[0]

    def _file_md5(self, path):
        digest = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _find_repeat_params_dir_by_model_hash(self, model_path):
        model_dir = os.path.dirname(model_path) or "."
        try:
            target_hash = self._file_md5(model_path)
        except Exception:
            return ""

        for run_name in sorted(os.listdir(model_dir)):
            run_dir = os.path.join(model_dir, run_name)
            if not run_name.lower().startswith("run_") or not os.path.isdir(run_dir):
                continue
            for root, _, files in os.walk(run_dir):
                for candidate_name in ("best_model.pth", "model.pth", "best_model.pkl", "model.pkl"):
                    if candidate_name not in files:
                        continue
                    candidate_path = os.path.join(root, candidate_name)
                    try:
                        if self._file_md5(candidate_path) == target_hash:
                            return root if os.path.exists(os.path.join(root, "params.json")) else ""
                    except Exception:
                        continue
        return ""

    def _apply_training_metadata(self, model_path):
        _, metadata = self._load_training_metadata(model_path)
        if not metadata:
            return

        model_name = normalize_model_key(metadata.get("model"))
        if model_name in WORKFLOW_MODEL_KEYS:
            self.model_type_combo.setCurrentText(get_model_display_name(model_name))

        batch_size = metadata.get("batchsize")
        if batch_size is not None:
            try:
                self.batch_size_edit.setText(str(int(batch_size)))
            except Exception:
                pass
        patch_stride = metadata.get("patch_stride")
        if patch_stride is not None:
            try:
                self.patch_stride_edit.setText(str(int(patch_stride)))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Prediction flow
    def _parse_model_paths(self, text: str):
        return [part.strip().strip('"') for part in text.replace("\n", ";").split(";") if part.strip()]

    def _start_prediction(self):
        if self.worker_thread and self.worker_thread.is_alive():
            QMessageBox.information(self, "提示", "正在执行预测，请稍候")
            return

        config = self._collect_config()
        if not config:
            return

        self.log_view.clear()
        self.plot_paths = {}
        self.last_plot_path = None
        self.last_predictions_file = None
        self.last_dat_path = None
        self.last_map_generation_mode = None
        self._clear_par_feedback("预测运行中...")
        self._set_running_state(True)

        self.worker_thread = threading.Thread(target=self._run_prediction_thread, args=(config,), daemon=True)
        self.worker_thread.start()

    def _collect_config(self):
        model_paths = self._parse_model_paths(self.model_path_edit.text())
        data_path = self.data_path_edit.text().strip()
        label_path = self.label_path_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip() or "predictions"

        missing = []
        if not model_paths:
            missing.append("模型文件")
        if not data_path:
            missing.append("数据文件")
        if not label_path:
            missing.append("标签文件")
        if missing:
            QMessageBox.warning(self, "缺少信息", f"请填写: {', '.join(missing)}")
            return None

        missing_files = [path for path in model_paths if not os.path.exists(path)]
        if missing_files:
            QMessageBox.critical(self, "文件不存在", "以下模型文件不存在:\n" + "\n".join(missing_files[:5]))
            return None

        try:
            batch_size = int(self.batch_size_edit.text().strip())
            if batch_size <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.critical(self, "参数错误", "Batch Size 必须为正整数")
            return None

        patch_stride_text = self.patch_stride_edit.text().strip()
        prediction_patch_stride = None
        if patch_stride_text:
            try:
                prediction_patch_stride = int(patch_stride_text)
                if prediction_patch_stride <= 0:
                    raise ValueError
            except ValueError:
                QMessageBox.critical(self, "参数错误", "预测步长必须为正整数")
                return None

        try:
            grid_spacing = float(self.grid_spacing_edit.text().strip() or "10")
        except ValueError:
            QMessageBox.critical(self, "参数错误", "网格间距必须为数值")
            return None

        return {
            "model_type": normalize_model_key(self.model_type_combo.currentText()),
            "model_paths": model_paths,
            "model_path": model_paths[0],
            "data_path": data_path,
            "label_path": label_path,
            "output_dir": output_dir,
            "batch_size": batch_size,
            "prediction_patch_stride": prediction_patch_stride,
            "device_pref": self.device_combo.currentText(),
            "grid_spacing": grid_spacing,
            "map_generation_mode": self.map_generation_combo.currentData(),
            "aggregation_method": self.aggregation_combo.currentData(),
            "aggregation_label": self.aggregation_combo.currentText(),
        }

    def _set_running_state(self, running: bool):
        self.run_button.setEnabled(not running)
        self.status_label.setText("正在运行预测..." if running else "等待开始预测")

    def _resolve_model_context(self, model_path, fallback_model_type, fallback_prior):
        resolved_model_dir, training_meta = self._load_training_metadata(model_path)
        model_type = fallback_model_type
        metadata_model_type = normalize_model_key(training_meta.get("model"))
        if metadata_model_type in WORKFLOW_MODEL_KEYS:
            model_type = metadata_model_type

        prior = fallback_prior
        if training_meta.get("resolved_prior") is not None:
            prior = float(training_meta["resolved_prior"])
        elif training_meta.get("calculated_prior") is not None:
            prior = float(training_meta["calculated_prior"])
        elif training_meta.get("manual_prior") is not None and not training_meta.get("auto_prior", False):
            prior = float(training_meta["manual_prior"])

        return {
            "path": model_path,
            "dir": resolved_model_dir or os.path.dirname(model_path) or ".",
            "meta": training_meta,
            "type": model_type,
            "display_name": get_model_display_name(model_type),
            "prior": prior,
        }

    def _check_model_metadata_compatibility(self, model_items, base_meta):
        base_patch_size = base_meta.get("patch_size") or base_meta.get("img_size")
        base_channels = parse_selected_channels(base_meta.get("selected_channels"))
        warnings = []
        for item in model_items[1:]:
            meta = item["meta"]
            patch_size = meta.get("patch_size") or meta.get("img_size")
            selected_channels = parse_selected_channels(meta.get("selected_channels"))
            if base_patch_size and patch_size and int(base_patch_size) != int(patch_size):
                raise ValueError(
                    f"模型窗口大小不一致: {os.path.basename(item['path'])} 为 {patch_size}，首个模型为 {base_patch_size}"
                )
            if base_channels != selected_channels:
                warnings.append(os.path.basename(item["path"]))

        if warnings:
            raise ValueError(
                "模型 selected_channels 与首个模型不同，不能对不同输入图层的预测值做平均/中位数聚合: "
                + ", ".join(warnings[:5])
            )

    def _run_prediction_thread(self, config):
        self._queue_log("开始加载数据与模型...")
        try:
            device = self._resolve_device(config["device_pref"])
            self._queue_log(f"使用设备: {device}")

            model_items = [
                self._resolve_model_context(path, config["model_type"], config.get("prior", 0.2))
                for path in config["model_paths"]
            ]
            training_meta = model_items[0]["meta"]
            model_dir = model_items[0]["dir"]
            model_type = model_items[0]["type"]
            model_count = len(model_items)
            aggregation_method = config.get("aggregation_method") or "mean"
            aggregation_label = config.get("aggregation_label") or ("中位数" if aggregation_method == "median" else "算术平均")
            self._check_model_metadata_compatibility(model_items, training_meta)
            if model_count == 1:
                model_display_name = model_items[0]["display_name"]
            else:
                model_display_name = f"{model_items[0]['display_name']}_{aggregation_label}{model_count}模型"
            self._queue_log(f"当前模型名称: {model_items[0]['display_name']}")
            self._queue_log(f"导入模型数量: {model_count}")
            self._queue_log(f"预测值聚合方式: {aggregation_label}")
            for idx, item in enumerate(model_items, start=1):
                self._queue_log(f"模型 {idx}: {item['path']}")
                if item.get("meta"):
                    self._queue_log(f"模型 {idx} 训练参数目录: {item['dir']}")
                else:
                    self._queue_log(f"模型 {idx} 未找到训练参数，使用界面/默认参数。")
            self._queue_log(f"从训练参数读取先验概率: {model_items[0]['prior']}")

            patch_size = int(training_meta.get("patch_size") or training_meta.get("img_size") or 16)
            training_patch_stride = int(training_meta.get("patch_stride") or patch_size)
            prediction_patch_stride = int(config.get("prediction_patch_stride") or training_patch_stride)
            self._queue_log(f"训练窗口大小: {patch_size} | 训练步长: {training_patch_stride} | 本次预测步长: {prediction_patch_stride}")

            dataset = PredictionDataset(
                config["data_path"],
                config["label_path"],
                model_dir,
                use_custom_norm=True,
                norm_params_path=training_meta.get("normalization_params_path"),
                patch_size=patch_size,
                patch_stride=prediction_patch_stride,
                use_reflect_padding=True,
                selected_channels=parse_selected_channels(training_meta.get("selected_channels")),
            )
            if getattr(dataset, "feature_mode", "") == "windows":
                self._queue_log("当前H5已包含预切好的 windows，预测步长不会重新切割该文件。")
            selected_channel_names = list(getattr(dataset, "metadata", {}).get("selected_channel_names", []) or [])
            self._queue_log(
                "参与预测图层: "
                + (", ".join(selected_channel_names[:6]) + ("..." if len(selected_channel_names) > 6 else "") if selected_channel_names else "全部")
            )

            if len(dataset) == 0:
                raise RuntimeError("数据集中没有任何样本，请检查输入数据与标签文件是否正确")

            sample_data = dataset[0][0]
            input_channels, input_height, input_width = sample_data.shape
            input_dim = input_channels * input_height * input_width
            coord_offset = getattr(dataset, "coord_offset", (input_width / 2.0, input_height / 2.0))
            self._queue_log(
                f"检测到输入维度: {input_dim} ({input_channels}x{input_height}x{input_width})"
            )

            dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
            output_dir = config["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            method_suffix = "median" if aggregation_method == "median" else "mean"
            output_suffix = f"{model_type}_{method_suffix}{model_count}" if model_count > 1 else model_type
            output_file = os.path.join(output_dir, f"predictions_{output_suffix}.h5")

            all_positions = []
            all_labels = []
            confidence_stack = []

            self._queue_log("开始批量预测...")
            for model_idx, item in enumerate(model_items, start=1):
                self._queue_log(f"加载第 {model_idx}/{model_count} 个模型: {os.path.basename(item['path'])}")
                model, enforced_shape = load_model(
                    item["type"],
                    item["path"],
                    prior=item["prior"],
                    input_dim=input_dim,
                    device=device,
                    data_shape=(input_channels, input_height, input_width),
                )
                if enforced_shape and all(val is not None for val in enforced_shape):
                    self._queue_log("模型要求输入形状: " + "x".join(str(val) for val in enforced_shape))

                model_confidences = []
                for batch_idx, (data, positions, labels) in enumerate(dataloader, start=1):
                    _, confidences = predict_batch(model, data, device)
                    model_confidences.extend(confidences.tolist())

                    if model_idx == 1:
                        cloned_positions = [pos.detach().cpu().clone() for pos in positions]
                        all_positions.extend(cloned_positions)
                        all_labels.append(labels.detach().cpu().clone())

                    if batch_idx % 20 == 0:
                        self._queue_log(f"模型 {model_idx}/{model_count} 已处理 {batch_idx} 个批次")

                model_confidences = torch.as_tensor(model_confidences, dtype=torch.float32).numpy()
                if confidence_stack:
                    if confidence_stack[0].shape != model_confidences.shape:
                        raise RuntimeError(
                            f"模型 {model_idx} 输出数量与首个模型不一致: {len(model_confidences)} != {len(confidence_stack[0])}"
                        )
                confidence_stack.append(model_confidences)

                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            if not all_positions:
                raise RuntimeError("未从数据集中读取到任何样本")

            if not confidence_stack:
                raise RuntimeError("未生成有效预测概率")

            confidence_array = np.vstack(confidence_stack).astype(np.float32, copy=False)
            if aggregation_method == "median":
                aggregated_confidences = np.median(confidence_array, axis=0)
            else:
                aggregated_confidences = np.mean(confidence_array, axis=0)
            all_confidences = aggregated_confidences.tolist()
            all_predictions = ["Positive" if conf >= 0.5 else "Negative" for conf in all_confidences]
            labels_tensor = torch.cat(all_labels)
            labels_np = labels_tensor.cpu().numpy()
            prediction_metadata = dict(getattr(dataset, "metadata", None) or {})
            prediction_metadata.update(
                {
                    "prediction_model_type": model_type,
                    "prediction_model_display_name": model_items[0]["display_name"],
                    "prediction_model_count": int(model_count),
                    "prediction_aggregation_method": aggregation_method,
                    "prediction_aggregation_label": aggregation_label,
                    "prediction_model_paths": json.dumps([item["path"] for item in model_items], ensure_ascii=False),
                }
            )
            save_predictions(
                all_positions,
                all_predictions,
                all_confidences,
                labels_np,
                output_file,
                coord_offset=coord_offset,
                metadata=prediction_metadata,
                mineral_points=getattr(dataset, "mineral_points", None),
            )

            summary = (
                f"预测完成，结果保存至: {output_file}\n"
                f"模型数量: {model_count} | 聚合方式: {aggregation_label} | 总样本数: {len(all_positions)} | 正类预测: {sum(1 for p in all_predictions if p == 'Positive')}"
            )
            self._queue_log(summary)

            self._queue_log("正在生成成矿潜力概率图...")
            viz_result = self._generate_visualizations(
                output_file,
                model_display_name,
                output_dir,
                config["grid_spacing"],
                config["map_generation_mode"],
            )
            if viz_result:
                self.plot_paths = viz_result.get("plots", {})
                self.last_predictions_file = viz_result.get("predictions_file") or output_file
                self.last_dat_path = viz_result.get("dat")
                self.log_queue.put(("preview", None))

            self.log_queue.put(("info_box", ("预测成功", summary)))
        except Exception as exc:  # noqa: BLE001
            self._queue_log(f"预测失败: {exc}")
            if "list index out of range" in str(exc).lower():
                hint = "请确认标签文件与数据文件的样本数量一致，并确保 positions/labels 维度正确。"
                self._queue_log(hint)
            self.log_queue.put(("error_box", ("预测失败", str(exc))))
        finally:
            self.log_queue.put(("running", False))

    def _generate_visualizations(
        self,
        predictions_file: str,
        model_name: str,
        base_output_dir: str,
        grid_spacing: float,
        map_generation_mode: str = "center",
    ):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        viz_dir = os.path.join(base_output_dir, "label_plots")
        os.makedirs(viz_dir, exist_ok=True)
        positions, predictions, confidences, labels, metadata, mineral_points = load_plot_data(predictions_file)
        self._queue_log(f"读取 {len(positions)} 条预测记录用于可视化")
        self.last_map_generation_mode = map_generation_mode

        dat_path = save_predictions_to_dat(positions, predictions, confidences, viz_dir, model_name, metadata=metadata)
        plots = plot_labels_comparison(
            positions,
            predictions,
            labels,
            viz_dir,
            model_name,
            confidences=confidences,
            grid_spacing=grid_spacing,
            metadata=metadata,
            mineral_points=mineral_points,
            map_generation_mode=map_generation_mode,
        )
        self._queue_log(f"可视化输出已生成，目录: {viz_dir}")
        return {"dat": dat_path, "plots": plots, "predictions_file": predictions_file}

    # ------------------------------------------------------------------
    # Preview helpers
    def _schedule_par_feedback_update(self):
        if hasattr(self, "par_feedback_timer") and self.par_feedback_timer is not None:
            self.par_feedback_timer.start(250)

    def _clear_par_feedback(self, message="预测完成后可实时调整 PAR 阈值"):
        if self.par_feedback_table is not None:
            self.par_feedback_table.setRowCount(0)
        if self.par_feedback_status_label is not None:
            self.par_feedback_status_label.setText(message)
            self.par_feedback_status_label.setStyleSheet("color: #6b7280;")

    def _parse_par_thresholds(self, text: str):
        normalized = str(text or "").replace("％", "%")
        for separator in ("，", "、", ";", "；", "\n", "\t"):
            normalized = normalized.replace(separator, ",")
        parts = []
        for chunk in normalized.split(","):
            for sub_chunk in chunk.split():
                value_text = sub_chunk.strip()
                if value_text:
                    parts.append(value_text)

        thresholds = []
        for value_text in parts:
            is_percent = value_text.endswith("%")
            value_text = value_text.rstrip("%").strip()
            if not value_text:
                continue
            try:
                value = float(value_text)
            except ValueError as exc:
                raise ValueError(f"PAR阈值不是有效数字: {value_text}") from exc
            if is_percent or value > 1.0:
                value = value / 100.0
            if not 0.0 < value <= 1.0:
                raise ValueError("PAR阈值需在 0-100% 或 0-1 之间")
            thresholds.append(float(value))

        if not thresholds:
            raise ValueError("请至少输入一个 PAR 阈值")

        unique_thresholds = []
        seen = set()
        for value in thresholds:
            key = round(value, 8)
            if key not in seen:
                unique_thresholds.append(value)
                seen.add(key)
        return unique_thresholds

    def _parse_par_hit_distance(self):
        if self.par_hit_distance_edit is None:
            return 1.0
        value_text = self.par_hit_distance_edit.text().strip()
        if not value_text:
            return 1.0
        try:
            distance = float(value_text)
        except ValueError as exc:
            raise ValueError("矿点命中距离需要是数字") from exc
        if distance < 0:
            raise ValueError("矿点命中距离不能小于 0")
        return float(distance)

    def _probability_values_for_map_mode(self, positions, confidences, metadata):
        mode = str(self.last_map_generation_mode or self.map_generation_combo.currentData() or "center")
        confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
        adjusted_positions = _positions_to_geo(positions, metadata)

        if mode in {"window_average", "window_max"}:
            average_values, max_values = _fused_probability_columns(adjusted_positions, confidences, metadata)
            candidate = average_values if mode == "window_average" else max_values
            candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
            if len(candidate) == len(confidences) and np.any(np.isfinite(candidate)):
                return adjusted_positions, candidate, mode

        return adjusted_positions, confidences, "center"

    def _mineral_point_hit_stats(self, positions, selected_mask, mineral_points, hit_distance):
        mineral_points = np.asarray(mineral_points if mineral_points is not None else [], dtype=np.float64)
        if mineral_points.ndim != 2 or mineral_points.shape[1] < 2 or len(mineral_points) == 0:
            return None

        positions = np.asarray(positions, dtype=np.float64)
        selected_mask = np.asarray(selected_mask, dtype=bool).reshape(-1)
        if positions.ndim != 2 or positions.shape[1] < 2 or len(positions) != len(selected_mask):
            return None
        valid_position_mask = np.isfinite(positions[:, 0]) & np.isfinite(positions[:, 1])
        if not np.any(valid_position_mask):
            return None

        valid_positions = positions[valid_position_mask, :2]
        valid_selected_positions = valid_positions[selected_mask[valid_position_mask]]
        hit_distance = float(hit_distance)
        hit_distance_sq = hit_distance * hit_distance
        hit_count = 0
        total_count = 0
        for mineral in mineral_points[:, :2]:
            if not np.all(np.isfinite(mineral)):
                continue
            total_count += 1
            if len(valid_selected_positions) == 0:
                continue
            distances = np.sum((valid_selected_positions - mineral) ** 2, axis=1)
            hit_count += int(bool(np.any(distances <= hit_distance_sq)))

        if total_count == 0:
            return None
        return hit_count, total_count

    def _set_par_table_item(self, row, column, value):
        item = QTableWidgetItem(str(value))
        item.setTextAlignment(Qt.AlignCenter)
        self.par_feedback_table.setItem(row, column, item)

    def _update_par_feedback(self):
        if self.par_feedback_table is None or self.par_threshold_edit is None:
            return
        if not self.last_predictions_file or not os.path.exists(self.last_predictions_file):
            self._clear_par_feedback("暂无预测结果")
            return

        try:
            thresholds = self._parse_par_thresholds(self.par_threshold_edit.text())
            hit_distance = self._parse_par_hit_distance()
            positions, _predictions, confidences, labels, metadata, mineral_points = load_plot_data(self.last_predictions_file)
            positions, scores, mode = self._probability_values_for_map_mode(positions, confidences, metadata)

            labels = np.asarray(labels).reshape(-1) if labels is not None else np.full(len(scores), np.nan)
            scores = np.asarray(scores, dtype=np.float64).reshape(-1)
            valid_mask = np.isfinite(scores)
            if len(labels) != len(scores):
                labels = np.full(len(scores), np.nan)
            if not np.any(valid_mask):
                raise ValueError("预测概率中没有有效数值")

            valid_scores = scores[valid_mask]
            valid_indices = np.where(valid_mask)[0]
            order = np.argsort(-valid_scores, kind="mergesort")
            total_count = len(valid_scores)

            self.par_feedback_table.setRowCount(len(thresholds))
            for row, par in enumerate(thresholds):
                selected_count = int(np.ceil(par * total_count))
                selected_count = max(1, min(selected_count, total_count))
                selected_valid_indices = valid_indices[order[:selected_count]]
                selected_mask = np.zeros(len(scores), dtype=bool)
                selected_mask[selected_valid_indices] = True

                threshold_value = float(valid_scores[order[selected_count - 1]])
                selected_scores = scores[selected_mask]
                mean_probability = float(np.nanmean(selected_scores)) if len(selected_scores) else float("nan")
                actual_par = selected_count / total_count if total_count else 0.0

                mineral_stats = self._mineral_point_hit_stats(positions, selected_mask, mineral_points, hit_distance)
                if mineral_stats is not None:
                    mineral_hit, mineral_total = mineral_stats
                    mineral_text = f"{mineral_hit}/{mineral_total}"
                    mineral_rate_text = f"{(mineral_hit / mineral_total * 100.0):.2f}%"
                else:
                    mineral_text = "-"
                    mineral_rate_text = "-"

                positive_mask = labels > 0
                positive_total = int(np.sum(positive_mask))
                if positive_total > 0:
                    positive_hit = int(np.sum(selected_mask & positive_mask))
                    positive_rate_text = f"{positive_hit}/{positive_total} ({positive_hit / positive_total * 100.0:.2f}%)"
                else:
                    positive_rate_text = "-"

                self._set_par_table_item(row, 0, f"{par * 100.0:.2f}%")
                self._set_par_table_item(row, 1, f"{actual_par * 100.0:.2f}%")
                self._set_par_table_item(row, 2, f"{threshold_value:.6f}")
                self._set_par_table_item(row, 3, f"{selected_count}/{total_count}")
                self._set_par_table_item(row, 4, f"{mean_probability:.6f}")
                self._set_par_table_item(row, 5, mineral_text)
                self._set_par_table_item(row, 6, mineral_rate_text)
                self._set_par_table_item(row, 7, positive_rate_text)

            mode_label = {
                "center": "中心点原值",
                "window_average": "窗口平均融合",
                "window_max": "窗口最大融合",
            }.get(mode, mode)
            self.par_feedback_status_label.setText(f"已按{mode_label}概率刷新，命中距离={hit_distance:g}")
            self.par_feedback_status_label.setStyleSheet("color: #047857;")
        except Exception as exc:  # noqa: BLE001
            self.par_feedback_table.setRowCount(0)
            self.par_feedback_status_label.setText(str(exc))
            self.par_feedback_status_label.setStyleSheet("color: #b91c1c;")

    def _refresh_confidence_preview(self):
        plot_path = self.plot_paths.get("confidence")
        if not plot_path or not os.path.exists(plot_path):
            self.preview_label.setText("暂无可用的置信度插值图，请检查预测流程")
            self.preview_label.setPixmap(QPixmap())
            self.view_button.setEnabled(False)
            self._clear_par_feedback("暂无可用的概率图")
            return
        self._update_preview(plot_path)
        self._update_par_feedback()

    def _update_preview(self, plot_path: str):
        pixmap = QPixmap(plot_path)
        if pixmap.isNull():
            self.preview_label.setText(f"无法加载图像: {plot_path}")
            self.view_button.setEnabled(False)
            return
        self.last_plot_path = plot_path
        self._preview_pixmap = pixmap
        scaled = pixmap.scaled(780, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)
        self.view_button.setEnabled(True)

    def _open_full_image(self):
        if not self.last_plot_path or not os.path.exists(self.last_plot_path):
            QMessageBox.warning(self, "未找到图像", "请先完成预测并生成图像")
            return

        try:
            image_data = mpimg.imread(self.last_plot_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "加载失败", f"无法读取图像: {exc}")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("成矿潜力概率图 - Matplotlib 预览")
        dialog.resize(1280, 860)

        v_layout = QVBoxLayout(dialog)
        control_row = QHBoxLayout()
        save_btn = QPushButton("另存为")
        save_btn.clicked.connect(partial(self._save_image_as, self.last_plot_path))
        control_row.addWidget(save_btn)
        export_csv_btn = QPushButton("导出CSV")
        export_csv_btn.clicked.connect(self._export_current_probability_csv)
        control_row.addWidget(export_csv_btn)
        control_row.addStretch(1)
        v_layout.addLayout(control_row)

        figure = Figure(figsize=(10, 6), dpi=100)
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, dialog)
        v_layout.addWidget(toolbar)
        v_layout.addWidget(canvas, 1)

        ax = figure.add_subplot(111)
        ax.imshow(image_data)
        ax.axis('off')
        ax.set_title("成矿潜力概率图", fontsize=14, weight="bold")
        figure.tight_layout()
        canvas.draw()

        dialog.exec_()

    def _save_current_preview(self):
        if not self.last_plot_path:
            QMessageBox.information(self, "提示", "暂无可保存的图像")
            return
        self._save_image_as(self.last_plot_path)

    def _save_image_as(self, source_path: str):
        if not os.path.exists(source_path):
            QMessageBox.critical(self, "保存失败", "找不到源图像文件")
            return
        target, _ = QFileDialog.getSaveFileName(
            self,
            "另存为",
            os.path.basename(source_path),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;所有文件 (*.*)",
        )
        if not target:
            return
        try:
            shutil.copy2(source_path, target)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "保存失败", str(exc))
        else:
            QMessageBox.information(self, "保存成功", f"图像已保存至: {target}")

    def _export_current_probability_csv(self):
        if not self.last_predictions_file or not os.path.exists(self.last_predictions_file):
            QMessageBox.information(self, "提示", "暂无可导出的预测结果，请先完成预测")
            return
        default_name = os.path.splitext(os.path.basename(self.last_predictions_file))[0] + "_probability.csv"
        default_dir = os.path.dirname(self.last_predictions_file) or os.getcwd()
        target, _ = QFileDialog.getSaveFileName(
            self,
            "导出成矿潜力概率CSV",
            os.path.join(default_dir, default_name),
            "CSV 文件 (*.csv);;所有文件 (*.*)",
        )
        if not target:
            return
        if not os.path.splitext(target)[1]:
            target += ".csv"
        try:
            positions, predictions, confidences, labels, metadata, _ = load_plot_data(self.last_predictions_file)
            save_predictions_to_csv(positions, predictions, confidences, labels, target, metadata=metadata)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败", str(exc))
        else:
            QMessageBox.information(self, "导出成功", f"CSV 已保存至: {target}")

    # ------------------------------------------------------------------
    # Loss curve helpers
    def _clear_loss_curve_plot(self):
        if self.loss_fig is None or self.loss_canvas is None:
            return
        self.loss_fig.clear()
        ax = self.loss_fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "尚未绘制Loss曲线", ha="center", va="center", color="#6b7280")
        self.loss_fig.tight_layout()
        self.loss_canvas.draw()

    def _add_loss_curve_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择训练结果目录", CODE_ROOT)
        if path:
            self._append_loss_curve_source(path)

    def _add_loss_curve_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择Loss曲线数据文件",
            CODE_ROOT,
            "Loss曲线数据 (*.csv *.json);;CSV 文件 (*.csv);;JSON 文件 (*.json);;所有文件 (*.*)",
        )
        for path in paths or []:
            self._append_loss_curve_source(path)

    def _append_loss_curve_source(self, path: str):
        resolved_files = self._find_loss_curve_files(path)
        if not resolved_files:
            QMessageBox.warning(
                self,
                "未找到曲线数据",
                "所选路径中未找到可用的 CSV/JSON Loss 曲线数据。\n"
                "CSV 需包含 epoch 以及 train_loss、val_loss 或 test_loss 列；"
                "JSON 需包含 train_losses、val_losses 或 test_losses 字段。",
            )
            return

        existing_paths = {
            self.loss_source_table.item(row, 3).text()
            for row in range(self.loss_source_table.rowCount())
            if self.loss_source_table.item(row, 3)
        }
        added = 0
        skipped = 0
        for resolved in resolved_files:
            if resolved in existing_paths:
                skipped += 1
                continue
            self._append_loss_curve_file_row(
                resolved,
                path if len(resolved_files) == 1 else resolved,
            )
            existing_paths.add(resolved)
            added += 1

        if added == 0 and skipped:
            QMessageBox.information(self, "提示", "该曲线数据源已经在列表中。")
        elif added > 1 and self.loss_status_label is not None:
            self.loss_status_label.setText(f"已添加 {added} 个曲线文件")

    def _append_loss_curve_file_row(self, resolved: str, label_source: str):
        row = self.loss_source_table.rowCount()
        self.loss_source_table.insertRow(row)
        label = self._default_loss_curve_label(label_source, resolved)
        self.loss_source_table.setItem(row, 0, QTableWidgetItem(label))
        self.loss_source_table.setItem(row, 1, QTableWidgetItem(f"{label} 训练"))
        self.loss_source_table.setItem(row, 2, QTableWidgetItem(f"{label} 验证"))
        path_item = QTableWidgetItem(resolved)
        path_item.setFlags(path_item.flags() & ~Qt.ItemIsEditable)
        self.loss_source_table.setItem(row, 3, path_item)
        self.loss_source_table.resizeColumnsToContents()
        self.loss_source_table.horizontalHeader().setStretchLastSection(True)

    def _remove_selected_loss_curve_sources(self):
        rows = sorted({index.row() for index in self.loss_source_table.selectedIndexes()}, reverse=True)
        if not rows:
            QMessageBox.information(self, "提示", "请先选择要移除的数据源。")
            return
        for row in rows:
            self.loss_source_table.removeRow(row)

    def _clear_loss_curve_sources(self):
        self.loss_source_table.setRowCount(0)
        self.loss_last_summaries = []
        self._clear_loss_curve_plot()
        if self.loss_status_label is not None:
            self.loss_status_label.setText("尚未绘制")

    def _resolve_loss_curve_file(self, path: str) -> str:
        matches = self._find_loss_curve_files(path)
        return matches[0] if matches else ""

    def _find_loss_curve_files(self, path: str) -> list[str]:
        path = str(path or "").strip().strip('"')
        if not path:
            return []
        if os.path.isfile(path):
            suffix = os.path.splitext(path)[1].lower()
            return [path] if suffix in {".csv", ".json"} else []
        if not os.path.isdir(path):
            return []

        matches = []
        seen = set()

        def add_if_match(candidate: str):
            normalized = os.path.abspath(candidate)
            if normalized in seen or not os.path.isfile(candidate):
                return
            suffix = os.path.splitext(candidate)[1].lower()
            if suffix not in {".csv", ".json"}:
                return
            if self._looks_like_loss_curve_file(candidate):
                matches.append(candidate)
                seen.add(normalized)

        candidates = (
            "cv_loss_curve_data.csv",
            "cv_training_curves_data.csv",
            "cv_training_curves_data.json",
            "cv_results.json",
        )
        for name in candidates:
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                add_if_match(candidate)
        if matches:
            return matches

        try:
            immediate_names = sorted(os.listdir(path))
        except OSError:
            return []
        for name in immediate_names:
            add_if_match(os.path.join(path, name))
        if matches:
            return matches

        for root, _, files in os.walk(path):
            if os.path.abspath(root) == os.path.abspath(path):
                continue
            for name in sorted(files):
                add_if_match(os.path.join(root, name))
            if len(matches) >= 100:
                break
        return matches

    def _looks_like_loss_curve_file(self, path: str) -> bool:
        suffix = os.path.splitext(path)[1].lower()
        try:
            if suffix == ".csv":
                with open(path, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, [])
                columns = {str(item or "").strip().lower() for item in header}
                return bool(
                    "epoch" in columns
                    and {"train_loss", "val_loss", "test_loss"}.intersection(columns)
                )
            if suffix == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if not isinstance(payload, dict):
                    return False
                return any(key in payload for key in ("train_losses", "val_losses", "test_losses"))
        except Exception:
            return False
        return False

    def _default_loss_curve_label(self, original_path: str, source_file: str) -> str:
        result_dir = original_path if os.path.isdir(original_path) else os.path.dirname(source_file)
        params_path = os.path.join(result_dir, "params.json")
        if os.path.exists(params_path):
            try:
                with open(params_path, "r", encoding="utf-8") as f:
                    params = json.load(f)
                loss_type = str(params.get("loss_type") or "").lower()
                if loss_type == "standard":
                    return "固定 nnPU loss"
                if loss_type == "adaptive":
                    return "自适应 nn-loss"
                model_key = normalize_model_key(params.get("model"))
                if model_key:
                    return get_model_display_name(model_key)
            except Exception:
                pass
        stem = os.path.splitext(os.path.basename(source_file))[0]
        if "自适应" in stem or "adaptive" in stem.lower():
            return "自适应 nn-loss"
        if "标准" in stem or "standard" in stem.lower():
            return "固定 nnPU loss"
        cleaned = stem.replace("cv_loss_curve_data", "").replace("cv_training_curves_data", "")
        cleaned = cleaned.strip("-_ ")
        return cleaned or stem or os.path.basename(os.path.dirname(source_file))

    def _loss_curve_sources(self):
        sources = []
        for row in range(self.loss_source_table.rowCount()):
            name_item = self.loss_source_table.item(row, 0)
            train_item = self.loss_source_table.item(row, 1)
            val_item = self.loss_source_table.item(row, 2)
            path_item = self.loss_source_table.item(row, 3)
            path = str(path_item.text() if path_item else "").strip()
            if not path:
                continue
            label = str(name_item.text() if name_item else "").strip()
            label = label or os.path.basename(os.path.dirname(path)) or os.path.basename(path)
            train_label = str(train_item.text() if train_item else "").strip() or f"{label} 训练"
            val_label = str(val_item.text() if val_item else "").strip() or f"{label} 验证"
            sources.append({"label": label, "train_label": train_label, "val_label": val_label, "path": path})
        return sources

    def _as_float(self, value):
        try:
            if value is None or value == "":
                return None
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if np.isfinite(result) else None

    def _read_loss_curve_csv(self, path: str):
        records = []
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epoch = self._as_float(row.get("epoch"))
                if epoch is None:
                    continue
                records.append(
                    {
                        "fold": int(self._as_float(row.get("fold")) or 1),
                        "epoch": int(epoch),
                        "train_loss": self._as_float(row.get("train_loss")),
                        "val_loss": self._as_float(row.get("val_loss")),
                    }
                )
        return records

    def _as_fold_lists(self, value):
        if not isinstance(value, list):
            return []
        if not value:
            return []
        if all(not isinstance(item, list) for item in value):
            return [value]
        return [item for item in value if isinstance(item, list)]

    def _read_loss_curve_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        train_lists = self._as_fold_lists(payload.get("train_losses"))
        val_lists = self._as_fold_lists(payload.get("val_losses") or payload.get("test_losses"))
        fold_count = max(len(train_lists), len(val_lists))
        records = []
        for fold_index in range(fold_count):
            train_values = train_lists[fold_index] if fold_index < len(train_lists) else []
            val_values = val_lists[fold_index] if fold_index < len(val_lists) else []
            epoch_count = max(len(train_values), len(val_values))
            for epoch_index in range(epoch_count):
                train_loss = self._as_float(train_values[epoch_index]) if epoch_index < len(train_values) else None
                val_loss = self._as_float(val_values[epoch_index]) if epoch_index < len(val_values) else None
                records.append(
                    {
                        "fold": fold_index + 1,
                        "epoch": epoch_index + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    }
                )
        return records

    def _summarize_loss_records(self, label: str, source_file: str, records, train_label: str = "", val_label: str = ""):
        by_epoch = {}
        for record in records:
            epoch = int(record.get("epoch") or 0)
            if epoch <= 0:
                continue
            bucket = by_epoch.setdefault(epoch, {"train": [], "val": []})
            train_loss = record.get("train_loss")
            val_loss = record.get("val_loss")
            if train_loss is not None:
                bucket["train"].append(float(train_loss))
            if val_loss is not None:
                bucket["val"].append(float(val_loss))

        epochs = sorted(by_epoch)
        if not epochs:
            raise ValueError("曲线数据中没有可用 epoch")

        def mean_std(values):
            arr = np.asarray(values, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                return np.nan, np.nan, 0
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            return float(np.mean(arr)), std, int(len(arr))

        train_mean, train_std, train_n = [], [], []
        val_mean, val_std, val_n = [], [], []
        for epoch in epochs:
            train_m, train_s, train_count = mean_std(by_epoch[epoch]["train"])
            val_m, val_s, val_count = mean_std(by_epoch[epoch]["val"])
            train_mean.append(train_m)
            train_std.append(train_s)
            train_n.append(train_count)
            val_mean.append(val_m)
            val_std.append(val_s)
            val_n.append(val_count)

        if not np.any(np.isfinite(train_mean)) and not np.any(np.isfinite(val_mean)):
            raise ValueError("曲线数据中没有可用 train_loss 或 val_loss")

        return {
            "label": label,
            "train_label": train_label or f"{label} 训练",
            "val_label": val_label or f"{label} 验证",
            "source_file": source_file,
            "epochs": np.asarray(epochs, dtype=np.int32),
            "train_mean": np.asarray(train_mean, dtype=np.float64),
            "train_std": np.asarray(train_std, dtype=np.float64),
            "train_n": train_n,
            "val_mean": np.asarray(val_mean, dtype=np.float64),
            "val_std": np.asarray(val_std, dtype=np.float64),
            "val_n": val_n,
        }

    def _load_loss_curve_summary(self, label: str, path: str, train_label: str = "", val_label: str = ""):
        suffix = os.path.splitext(path)[1].lower()
        if suffix == ".csv":
            records = self._read_loss_curve_csv(path)
        elif suffix == ".json":
            records = self._read_loss_curve_json(path)
        else:
            raise ValueError(f"不支持的曲线数据格式: {path}")
        return self._summarize_loss_records(label, path, records, train_label=train_label, val_label=val_label)

    def _draw_loss_curve_figure(self, figure, summaries):
        figure.clear()
        ax = figure.add_subplot(111)
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if not colors:
            colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

        for index, summary in enumerate(summaries):
            color = colors[index % len(colors)]
            epochs = summary["epochs"]
            train_mean = summary["train_mean"]
            train_std = summary["train_std"]
            val_mean = summary["val_mean"]
            val_std = summary["val_std"]
            label = summary["label"]
            train_label = str(summary.get("train_label") or f"{label} 训练")
            val_label = str(summary.get("val_label") or f"{label} 验证")

            train_mask = np.isfinite(train_mean)
            if np.any(train_mask):
                ax.plot(epochs[train_mask], train_mean[train_mask], color=color, linewidth=2.0, label=train_label)
                ax.fill_between(
                    epochs[train_mask],
                    train_mean[train_mask] - np.nan_to_num(train_std[train_mask]),
                    train_mean[train_mask] + np.nan_to_num(train_std[train_mask]),
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                )

            val_mask = np.isfinite(val_mean)
            if np.any(val_mask):
                ax.plot(
                    epochs[val_mask],
                    val_mean[val_mask],
                    color=color,
                    linewidth=2.0,
                    linestyle="--",
                    label=val_label,
                )
                ax.fill_between(
                    epochs[val_mask],
                    val_mean[val_mask] - np.nan_to_num(val_std[val_mask]),
                    val_mean[val_mask] + np.nan_to_num(val_std[val_mask]),
                    color=color,
                    alpha=0.08,
                    linewidth=0,
                )

        title = self.loss_title_edit.text().strip() if self.loss_title_edit is not None else ""
        ax.set_title(title or "五折空间交叉验证 Loss 曲线")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        ax.legend(loc="best", fontsize=9)
        figure.tight_layout()
        return ax

    def _plot_loss_curves(self):
        sources = self._loss_curve_sources()
        if not sources:
            QMessageBox.information(self, "提示", "请先添加至少一个训练目录或曲线数据文件。")
            return

        summaries = []
        errors = []
        for source in sources:
            try:
                summaries.append(
                    self._load_loss_curve_summary(
                        source["label"],
                        source["path"],
                        train_label=source.get("train_label", ""),
                        val_label=source.get("val_label", ""),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{source['label']}: {exc}")

        if errors:
            self._queue_log("[Loss曲线] 部分数据源读取失败:\n" + "\n".join(errors))
        if not summaries:
            QMessageBox.critical(self, "绘制失败", "没有可用的Loss曲线数据。\n" + "\n".join(errors[:5]))
            return

        self.loss_last_summaries = summaries
        self._draw_loss_curve_figure(self.loss_fig, summaries)
        self.loss_canvas.draw()
        self.loss_status_label.setText(f"已绘制 {len(summaries)} 组曲线")
        self._queue_log(
            "[Loss曲线] 已绘制: "
            + "；".join(f"{item['label']} ({os.path.basename(item['source_file'])})" for item in summaries)
        )

    def _open_loss_curve_fullscreen(self):
        if not self.loss_last_summaries:
            QMessageBox.information(self, "提示", "请先绘制Loss曲线。")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Loss曲线 - Matplotlib 预览")
        dialog.resize(1180, 760)
        layout = QVBoxLayout(dialog)
        figure = Figure(figsize=(10, 6), dpi=100)
        canvas = FigureCanvas(figure)
        layout.addWidget(NavigationToolbar(canvas, dialog))
        layout.addWidget(canvas, 1)
        self._draw_loss_curve_figure(figure, self.loss_last_summaries)
        canvas.draw()
        dialog.exec_()

    def _save_loss_curve_image(self):
        if not self.loss_last_summaries:
            QMessageBox.information(self, "提示", "请先绘制Loss曲线。")
            return
        target, _ = QFileDialog.getSaveFileName(
            self,
            "保存Loss曲线",
            os.path.join(CODE_ROOT, "loss_curve_comparison.png"),
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;所有文件 (*.*)",
        )
        if not target:
            return
        if not os.path.splitext(target)[1]:
            target += ".png"
        try:
            self.loss_fig.savefig(target, dpi=300, bbox_inches="tight")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "保存失败", str(exc))
        else:
            QMessageBox.information(self, "保存成功", f"图像已保存至: {target}")

    def _export_loss_curve_summary_csv(self):
        if not self.loss_last_summaries:
            QMessageBox.information(self, "提示", "请先绘制Loss曲线。")
            return
        target, _ = QFileDialog.getSaveFileName(
            self,
            "导出Loss曲线均值数据",
            os.path.join(CODE_ROOT, "loss_curve_summary.csv"),
            "CSV 文件 (*.csv);;所有文件 (*.*)",
        )
        if not target:
            return
        if not os.path.splitext(target)[1]:
            target += ".csv"

        rows = []
        for summary in self.loss_last_summaries:
            for idx, epoch in enumerate(summary["epochs"]):
                rows.append(
                    {
                        "source": summary["label"],
                        "train_legend": summary.get("train_label", ""),
                        "val_legend": summary.get("val_label", ""),
                        "source_file": summary["source_file"],
                        "epoch": int(epoch),
                        "train_loss_mean": summary["train_mean"][idx],
                        "train_loss_std": summary["train_std"][idx],
                        "train_fold_count": summary["train_n"][idx],
                        "val_loss_mean": summary["val_mean"][idx],
                        "val_loss_std": summary["val_std"][idx],
                        "val_fold_count": summary["val_n"][idx],
                    }
                )

        try:
            with open(target, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "source",
                        "train_legend",
                        "val_legend",
                        "source_file",
                        "epoch",
                        "train_loss_mean",
                        "train_loss_std",
                        "train_fold_count",
                        "val_loss_mean",
                        "val_loss_std",
                        "val_fold_count",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败", str(exc))
        else:
            QMessageBox.information(self, "导出成功", f"CSV 已保存至: {target}")

    # ------------------------------------------------------------------
    # Logging helpers
    def _queue_log(self, message: str):
        self.log_queue.put(("log", message))

    def _process_queue(self):
        updated = False
        while True:
            try:
                kind, payload = self.log_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "log":
                self.log_view.appendPlainText(payload.rstrip())
                updated = True
            elif kind == "info_box":
                QMessageBox.information(self, *payload)
            elif kind == "error_box":
                QMessageBox.critical(self, *payload)
            elif kind == "preview":
                self._refresh_confidence_preview()
            elif kind == "running":
                self._set_running_state(payload)

        if updated:
            self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    # ------------------------------------------------------------------
    # Backend helpers
    def _resolve_device(self, preference: str) -> torch.device:
        pref = preference.lower()
        if pref == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if pref == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("当前环境未检测到可用的 CUDA，请改用 CPU 或检查驱动/显卡")
            return torch.device("cuda")
        return torch.device("cpu")


def launch_gui():
    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication(sys.argv)
        owns_app = True
    window = PredictionWindow()
    window.show()
    if owns_app:
        app.exec_()


def main():
    launch_gui()


if __name__ == "__main__":
    main()
