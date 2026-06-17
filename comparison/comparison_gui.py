"""GUI for model comparison and AutoML workflows."""

from __future__ import annotations

import math
import os
import sys

import torch
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHeaderView,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QInputDialog,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt
from sklearn.model_selection import train_test_split
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(CURRENT_DIR)
COMMON_DIR = os.path.join(CODE_ROOT, "common")
for path in (CODE_ROOT, COMMON_DIR):
    if path not in sys.path:
        sys.path.append(path)

from .analysis_utils import format_params
from feature_channel_utils import describe_selected_channels, infer_h5_channel_names
from .artifact_manager import ArtifactManager
from .automl_engine import AutoMLEngine
from .comparison_engine import ComparisonEngine
from .data_loader_thread import (
    AutoMLRunThread,
    ComparisonRunThread,
    DataLoaderThread,
    ManifestRebuildThread,
    WorkflowBatchRunThread,
    WorkflowRunThread,
)
from .dataset_builder import ComparisonDataBuilder
from .model_wrappers import DeepLearningWrapper, DecisionTreeWrapper, RandomForestWrapper, SVMWrapper
from .result_analyzer import ResultAnalyzer
from .spatial_region_splitter import serialize_region_split_config, split_dataframe_by_regions
from .visualization import Visualization
from .workflow_orchestrator import WorkflowOrchestrator
from ..cnn.cnn import CNNModel
from ..resnet.resnet import ResNet18, ResNet34, ResNet50


class MineralSelectionDialog(QDialog):
    """Let the user manually choose reserved test mineral points."""

    def __init__(self, mineral_df, parent=None, reserved_rows=None):
        super().__init__(parent)
        self.mineral_df = mineral_df.reset_index(drop=True).copy()
        self.reserved_rows = set(reserved_rows or [])
        self.setWindowTitle("手工划分矿点")
        self.resize(1100, 700)
        self._build_ui()
        self._update_summary()

    @staticmethod
    def _find_column_alias(frame, aliases):
        normalized_columns = {str(column).strip().lower(): column for column in frame.columns}
        for alias in aliases:
            matched = normalized_columns.get(alias.strip().lower())
            if matched is not None:
                return matched
        return None

    @staticmethod
    def _format_value(value):
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except TypeError:
            pass
        return str(value)

    def _build_display_columns(self):
        columns = ["x", "y"]
        label_column = self._find_column_alias(
            self.mineral_df,
            ["label", "label_id", "class", "category", "标签", "类别", "type", "成因类型"],
        )
        if label_column and label_column not in columns:
            columns.append(label_column)

        x_alias = self._find_column_alias(
            self.mineral_df,
            ["x", "coord_x", "point_x", "east", "easting", "东坐标", "横坐标"],
        )
        y_alias = self._find_column_alias(
            self.mineral_df,
            ["y", "coord_y", "point_y", "north", "northing", "北坐标", "纵坐标"],
        )
        excluded = {alias for alias in [x_alias, y_alias] if alias is not None}
        for column in self.mineral_df.columns:
            if column in columns or column in excluded or str(column).startswith("Unnamed"):
                continue
            columns.append(column)
        return columns

    def _build_ui(self):
        layout = QVBoxLayout(self)

        info_label = QLabel(
            "勾选“预留测试”列中的矿点作为测试集，未勾选的自动作为训练集。"
            "你也可以先选中若干行，再用下方按钮批量设置。"
            "内部验证仍由主界面下方的比例自动切分。",
            self,
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self.table = QTableWidget(self)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(False)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(self.table, 1)

        self.summary_label = QLabel(self)
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        action_row = QHBoxLayout()
        self.reserve_selected_btn = QPushButton("设为预留", self)
        self.reserve_selected_btn.clicked.connect(lambda: self._set_selected_rows_state(True))
        self.train_selected_btn = QPushButton("设为训练", self)
        self.train_selected_btn.clicked.connect(lambda: self._set_selected_rows_state(False))
        self.reserve_all_btn = QPushButton("全部预留", self)
        self.reserve_all_btn.clicked.connect(lambda: self._set_all_rows_state(True))
        self.train_all_btn = QPushButton("全部训练", self)
        self.train_all_btn.clicked.connect(lambda: self._set_all_rows_state(False))
        for button in [
            self.reserve_selected_btn,
            self.train_selected_btn,
            self.reserve_all_btn,
            self.train_all_btn,
        ]:
            action_row.addWidget(button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._populate_table()
        self.table.itemChanged.connect(self._update_summary)

    def _populate_table(self):
        columns = self._build_display_columns()
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.mineral_df))
        self.table.setColumnCount(len(columns) + 2)
        self.table.setHorizontalHeaderLabels(["行号", "预留测试"] + [str(column) for column in columns])

        for row_index, (_, row) in enumerate(self.mineral_df.iterrows()):
            row_item = QTableWidgetItem(str(row_index + 1))
            row_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(row_index, 0, row_item)

            reserve_item = QTableWidgetItem("")
            reserve_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            reserve_item.setCheckState(Qt.Checked if row_index in self.reserved_rows else Qt.Unchecked)
            self.table.setItem(row_index, 1, reserve_item)

            for column_index, column in enumerate(columns, start=2):
                value = row[column] if column in row.index else ""
                cell_item = QTableWidgetItem(self._format_value(value))
                cell_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.table.setItem(row_index, column_index, cell_item)

        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()

    def _current_state(self):
        reserved_rows = set()
        for row_index in range(self.table.rowCount()):
            item = self.table.item(row_index, 1)
            if item is not None and item.checkState() == Qt.Checked:
                reserved_rows.add(row_index)
        train_count = self.table.rowCount() - len(reserved_rows)
        return train_count, len(reserved_rows), reserved_rows

    def _update_summary(self, *_):
        train_count, test_count, _ = self._current_state()
        if train_count <= 0 or test_count <= 0:
            self.summary_label.setStyleSheet("color: #b00020;")
            self.summary_label.setText(
                f"当前选择: 训练矿点 {train_count} 个，预留测试矿点 {test_count} 个。"
                "请至少保留 1 个训练矿点和 1 个预留测试矿点。"
            )
        else:
            self.summary_label.setStyleSheet("")
            self.summary_label.setText(
                f"当前选择: 训练矿点 {train_count} 个，预留测试矿点 {test_count} 个。"
                "未勾选的矿点将作为训练矿点。"
            )

    def _set_selected_rows_state(self, checked):
        selected_rows = {index.row() for index in self.table.selectionModel().selectedRows()}
        if not selected_rows:
            return
        self.table.blockSignals(True)
        for row_index in selected_rows:
            item = self.table.item(row_index, 1)
            if item is not None:
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.table.blockSignals(False)
        self._update_summary()

    def _set_all_rows_state(self, checked):
        self.table.blockSignals(True)
        for row_index in range(self.table.rowCount()):
            item = self.table.item(row_index, 1)
            if item is not None:
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.table.blockSignals(False)
        self._update_summary()

    def accept(self):
        train_count, test_count, _ = self._current_state()
        if train_count <= 0 or test_count <= 0:
            QMessageBox.warning(
                self,
                "提示",
                "请至少保留 1 个训练矿点和 1 个预留测试矿点。",
            )
            return
        super().accept()

    def get_split(self):
        _, _, reserved_rows = self._current_state()
        reserved_mask = [row_index in reserved_rows for row_index in range(self.table.rowCount())]
        mask_series = pd.Series(reserved_mask, index=self.mineral_df.index)
        train_minerals = self.mineral_df.loc[~mask_series].reset_index(drop=True)
        test_minerals = self.mineral_df.loc[mask_series].reset_index(drop=True)
        return train_minerals, test_minerals, reserved_rows


class ModelComparisonGUI(QWidget):
    MODEL_FACTORIES = {
        "CNN": CNNModel,
        "ResNet-18": ResNet18,
        "ResNet-34": ResNet34,
        "ResNet-50": ResNet50,
    }
    AUTOML_CLASSIC_MODELS = ["Random Forest", "SVM", "Decision Tree"]
    MODE_SEPARATE = "训练/补充训练/测试"
    MODE_SINGLE = "单文件自动分割"
    MODE_MANUAL = "单文件手工划分"
    MODE_REGION = "坐标区域切分"
    CHART_TEST_ACC = "测试召回率对比"
    CHART_TRAIN_TIME = "训练时间对比"
    CHART_GENERALIZATION = "验证召回率对比"
    CHART_VAL_MINERAL = "验证集预测面积占比对比"
    CHART_TEST_MINERAL = "测试集预测面积占比对比"
    CHART_VAL_EI = "验证集EI对比"
    CHART_TEST_EI = "预测集EI对比"
    CHART_COMPOSITE = "综合验证得分对比"
    RUNTIME_MODE_SCHEME_CAPS = {
        "快速筛查": 8,
        "实用批量": 16,
        "深度搜索": 24,
    }
    AUTO_ML_BATCH_SIZE_HINT = 48
    DEEP_MODEL_COMPLEXITY = {
        "CNN": 1.0,
        "ResNet-18": 1.25,
        "ResNet-34": 1.65,
        "ResNet-50": 2.2,
    }
    CLASSIC_MODEL_BASE_SECONDS = {
        "Random Forest": 0.8,
        "SVM": 1.3,
        "Decision Tree": 0.35,
    }

    def __init__(self):
        super().__init__()
        self.engine = ComparisonEngine()
        self.data_builder = ComparisonDataBuilder(negative_ratio=3)
        self.dataset_bundle = None
        self.data_loader_thread = None
        self.comparison_thread = None
        self.automl_thread = None
        self.automl_engine = None
        self.automl_results = []
        self.workflow_thread = None
        self.workflow_orchestrator = None
        self.workflow_summary = None
        self.workflow_current_stage = None
        self.workflow_batch_thread = None
        self.workflow_batch_summary = None
        self.workflow_batch_active = False
        self.workflow_batch_total_runs = 0
        self.workflow_batch_current_run = 0
        self.workflow_batch_base_seed = None
        self.rebuild_thread = None
        self.manual_mineral_selection = None
        self.artifact_manager = ArtifactManager(self._create_automl_wrapper)
        self._build_ui()
        self._connect_signals()
        self._toggle_mineral_mode()
        self._set_dataset_summary_text("数据体检: 尚未加载。")
        self._refresh_action_buttons()
        self._update_workload_estimate()

    def _build_ui(self):
        self.setWindowTitle("模型对比和微调工具")
        self.setMinimumSize(1280, 860)
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        self.tabs.addTab(self._create_comparison_tab(), "模型对比")
        self.tabs.addTab(self._create_results_tab(), "结果分析")
        self.tabs.addTab(self._create_automl_tab(), "自动优化")
        self.tabs.addTab(self._create_artifact_tab(), "模型包与重建")
        layout.addWidget(self.tabs)

    def _create_comparison_tab(self):
        page = QWidget(self)
        page_layout = QVBoxLayout(page)

        data_group = QGroupBox("数据配置", page)
        self.data_group_widget = data_group
        data_layout = QVBoxLayout(data_group)

        self.full_h5_edit = QLineEdit(data_group)
        full_h5_btn = QPushButton("浏览...", data_group)
        full_h5_btn.clicked.connect(lambda: self._browse_file(self.full_h5_edit, "完整数据集"))
        row = QHBoxLayout()
        row.addWidget(QLabel("完整数据集 H5:", data_group))
        row.addWidget(self.full_h5_edit)
        row.addWidget(full_h5_btn)
        data_layout.addLayout(row)
        self.available_feature_channels = []
        self.selected_feature_channels = None
        feature_row = QHBoxLayout()
        self.feature_channel_btn = QPushButton("选择参与图层", data_group)
        self.feature_channel_btn.clicked.connect(self._choose_feature_channels)
        self.feature_channel_summary_label = QLabel("图层: 默认全选", data_group)
        self.feature_channel_summary_label.setWordWrap(True)
        self.feature_channel_summary_label.setStyleSheet("color: #666666;")
        feature_row.addWidget(self.feature_channel_btn)
        feature_row.addWidget(self.feature_channel_summary_label, 1)
        data_layout.addLayout(feature_row)

        row = QHBoxLayout()
        row.addWidget(QLabel("矿点文件模式:", data_group))
        self.mineral_mode_combo = QComboBox(data_group)
        self.mineral_mode_combo.addItems([self.MODE_SEPARATE, self.MODE_SINGLE, self.MODE_MANUAL, self.MODE_REGION])
        row.addWidget(self.mineral_mode_combo)
        data_layout.addLayout(row)

        self.separate_mineral_widget = QWidget(data_group)
        separate_layout = QVBoxLayout(self.separate_mineral_widget)
        separate_layout.setContentsMargins(0, 0, 0, 0)
        self.train_mineral_edit = QLineEdit(self.separate_mineral_widget)
        self.val_mineral_edit = QLineEdit(self.separate_mineral_widget)
        self.test_mineral_edit = QLineEdit(self.separate_mineral_widget)
        file_rows = [
            ("训练矿点:", self.train_mineral_edit, "训练矿点"),
            ("补充训练矿点:", self.val_mineral_edit, "补充训练矿点"),
            ("预留测试矿点:", self.test_mineral_edit, "预留测试矿点"),
        ]
        for label_text, edit, title in file_rows:
            browse_btn = QPushButton("浏览...", self.separate_mineral_widget)
            browse_btn.clicked.connect(
                lambda _=False, line_edit=edit, dialog_title=title: self._browse_mineral_file(line_edit, dialog_title)
            )
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text, self.separate_mineral_widget))
            row.addWidget(edit)
            row.addWidget(browse_btn)
            separate_layout.addLayout(row)
        data_layout.addWidget(self.separate_mineral_widget)

        self.single_mineral_widget = QWidget(data_group)
        single_layout = QVBoxLayout(self.single_mineral_widget)
        single_layout.setContentsMargins(0, 0, 0, 0)
        self.all_mineral_edit = QLineEdit(self.single_mineral_widget)
        all_btn = QPushButton("浏览...", self.single_mineral_widget)
        all_btn.clicked.connect(lambda: self._browse_mineral_file(self.all_mineral_edit, "全部矿点"))
        row = QHBoxLayout()
        row.addWidget(QLabel("全部矿点:", self.single_mineral_widget))
        row.addWidget(self.all_mineral_edit)
        row.addWidget(all_btn)
        single_layout.addLayout(row)

        self.test_split_ratio_row = QWidget(self.single_mineral_widget)
        ratio_row = QHBoxLayout(self.test_split_ratio_row)
        ratio_row.setContentsMargins(0, 0, 0, 0)
        ratio_row.addWidget(QLabel("测试集比例:", self.test_split_ratio_row))
        self.test_split_ratio_spin = QDoubleSpinBox(self.single_mineral_widget)
        self.test_split_ratio_spin.setRange(0.1, 0.5)
        self.test_split_ratio_spin.setValue(0.2)
        self.test_split_ratio_spin.setSingleStep(0.05)
        ratio_row.addWidget(self.test_split_ratio_spin)
        single_layout.addWidget(self.test_split_ratio_row)

        self.manual_split_row = QWidget(self.single_mineral_widget)
        manual_row_layout = QVBoxLayout(self.manual_split_row)
        manual_row_layout.setContentsMargins(0, 0, 0, 0)
        manual_button_row = QHBoxLayout()
        manual_button_row.addWidget(QLabel("手工划分:", self.manual_split_row))
        self.manual_split_button = QPushButton("选择训练/预留矿点...", self.manual_split_row)
        self.manual_split_button.setToolTip("打开矿点选择窗口，勾选预留测试矿点。未勾选的矿点将作为训练矿点。")
        self.manual_split_button.clicked.connect(self._open_manual_mineral_split_dialog)
        manual_button_row.addWidget(self.manual_split_button)
        manual_button_row.addStretch(1)
        manual_row_layout.addLayout(manual_button_row)
        self.manual_split_summary_label = QLabel(
            "尚未手工划分矿点。未勾选的矿点将作为训练矿点，内部验证仍按下方比例自动切分。",
            self.manual_split_row,
        )
        self.manual_split_summary_label.setWordWrap(True)
        self.manual_split_summary_label.setStyleSheet("color: #666666;")
        manual_row_layout.addWidget(self.manual_split_summary_label)
        single_layout.addWidget(self.manual_split_row)

        self.region_split_row = QWidget(self.single_mineral_widget)
        region_layout = QVBoxLayout(self.region_split_row)
        region_layout.setContentsMargins(0, 0, 0, 0)
        region_hint = QLabel(
            "按矩形坐标框定义训练区和预测区，格式为 xmin,xmax,ymin,ymax。边界缓冲带内的样本会被剔除，避免空间泄漏。",
            self.region_split_row,
        )
        region_hint.setWordWrap(True)
        region_hint.setStyleSheet("color: #666666;")
        region_layout.addWidget(region_hint)

        region_grid = QGridLayout()
        region_grid.setContentsMargins(0, 0, 0, 0)
        region_grid.setHorizontalSpacing(8)
        region_grid.setVerticalSpacing(6)

        self.train_region_edit = QLineEdit(self.region_split_row)
        self.train_region_edit.setPlaceholderText("例如: 1000,2000,3000,4000")
        self.train_region_edit.setToolTip("训练区域坐标范围，格式: xmin,xmax,ymin,ymax")
        self.test_region_edit = QLineEdit(self.region_split_row)
        self.test_region_edit.setPlaceholderText("例如: 2200,3200,3000,4000")
        self.test_region_edit.setToolTip("预测区域坐标范围，格式: xmin,xmax,ymin,ymax")
        self.region_buffer_spin = QDoubleSpinBox(self.region_split_row)
        self.region_buffer_spin.setRange(0.0, 20000.0)
        self.region_buffer_spin.setDecimals(1)
        self.region_buffer_spin.setSingleStep(50.0)
        self.region_buffer_spin.setValue(0.0)
        self.region_buffer_spin.setToolTip("训练区和预测区边界附近需要剔除的灰区宽度。")

        region_grid.addWidget(QLabel("训练区域坐标:", self.region_split_row), 0, 0)
        region_grid.addWidget(self.train_region_edit, 0, 1)
        region_grid.addWidget(QLabel("预测区域坐标:", self.region_split_row), 1, 0)
        region_grid.addWidget(self.test_region_edit, 1, 1)
        region_grid.addWidget(QLabel("边界缓冲带(米):", self.region_split_row), 2, 0)
        region_grid.addWidget(self.region_buffer_spin, 2, 1)
        region_layout.addLayout(region_grid)
        single_layout.addWidget(self.region_split_row)

        self.region_preview_group = QGroupBox("区域预览", self.single_mineral_widget)
        preview_layout = QVBoxLayout(self.region_preview_group)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_toolbar = QHBoxLayout()
        self.region_preview_summary_label = QLabel(
            "输入训练区/预测区坐标后，可实时查看训练区、预测区、灰区与无矿锚点。",
            self.region_preview_group,
        )
        self.region_preview_summary_label.setWordWrap(True)
        self.region_preview_summary_label.setStyleSheet("color: #666666;")
        self.region_preview_refresh_btn = QPushButton("刷新预览", self.region_preview_group)
        self.region_preview_refresh_btn.clicked.connect(self._update_region_preview)
        preview_toolbar.addWidget(self.region_preview_summary_label)
        preview_toolbar.addStretch(1)
        preview_toolbar.addWidget(self.region_preview_refresh_btn)
        preview_layout.addLayout(preview_toolbar)

        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.region_preview_figure = Figure(figsize=(6.0, 4.2))
        self.region_preview_canvas = FigureCanvasQTAgg(self.region_preview_figure)
        self.region_preview_canvas.setMinimumHeight(280)
        self.region_preview_ax = self.region_preview_figure.add_subplot(111)
        preview_layout.addWidget(self.region_preview_canvas)
        single_layout.addWidget(self.region_preview_group)
        data_layout.addWidget(self.single_mineral_widget)

        self.no_ore_mineral_edit = QLineEdit(data_group)
        no_ore_btn = QPushButton("浏览...", data_group)
        no_ore_btn.clicked.connect(lambda: self._browse_mineral_file(self.no_ore_mineral_edit, "无矿钻孔/坐标"))
        row = QHBoxLayout()
        row.addWidget(QLabel("无矿钻孔/坐标（可选）:", data_group))
        row.addWidget(self.no_ore_mineral_edit)
        row.addWidget(no_ore_btn)
        data_layout.addLayout(row)

        self.mineral_stats_label = QLabel("矿点统计: 尚未选择文件。", data_group)
        data_layout.addWidget(self.mineral_stats_label)

        row = QHBoxLayout()
        self.val_ratio_label = QLabel("内部验证比例:", data_group)
        self.val_ratio_spin = QDoubleSpinBox(data_group)
        self.val_ratio_spin.setRange(0.1, 0.5)
        self.val_ratio_spin.setValue(0.2)
        self.val_ratio_spin.setSingleStep(0.05)
        self.buffer_radius_spin = QSpinBox(data_group)
        self.buffer_radius_spin.setRange(0, 5000)
        self.buffer_radius_spin.setValue(500)
        self.buffer_radius_spin.setSingleStep(100)
        self.patch_size_spin = QSpinBox(data_group)
        self.patch_size_spin.setRange(8, 256)
        self.patch_size_spin.setValue(64)
        self.patch_size_spin.setSingleStep(8)
        self.patch_stride_spin = QSpinBox(data_group)
        self.patch_stride_spin.setRange(1, 256)
        self.patch_stride_spin.setValue(32)
        self.reflect_padding_check = QCheckBox("切窗 reflect 补边", data_group)
        self.reflect_padding_check.setChecked(True)
        self.reflect_padding_check.setToolTip("仅对完整特征 H5 重建窗口时生效；默认启用以覆盖边缘区域。")
        self.n_blocks_spin = QSpinBox(data_group)
        self.n_blocks_spin.setRange(2, 10)
        self.n_blocks_spin.setValue(3)
        for label_widget, input_widget in [
            (self.val_ratio_label, self.val_ratio_spin),
            (QLabel("缓冲区半径(米):", data_group), self.buffer_radius_spin),
            (QLabel("窗口大小:", data_group), self.patch_size_spin),
            (QLabel("步长:", data_group), self.patch_stride_spin),
            (QLabel("空间分块数:", data_group), self.n_blocks_spin),
        ]:
            row.addWidget(label_widget)
            row.addWidget(input_widget)
        row.addWidget(self.reflect_padding_check)
        data_layout.addLayout(row)
        negative_row = QHBoxLayout()
        self.negative_sampling_mode_btn = QPushButton("启用远区负样本模式", data_group)
        self.negative_sampling_mode_btn.setCheckable(True)
        self.negative_sampling_mode_btn.setToolTip(
            "开启后，负样本只从距离所有已知矿点都足够远的样本中抽取；中间缓冲带样本将被剔除。"
        )
        self.negative_sampling_mode_btn.toggled.connect(self._update_negative_sampling_controls)
        self.negative_distance_multiplier_spin = QDoubleSpinBox(data_group)
        self.negative_distance_multiplier_spin.setRange(1.1, 10.0)
        self.negative_distance_multiplier_spin.setDecimals(2)
        self.negative_distance_multiplier_spin.setValue(2.0)
        self.negative_distance_multiplier_spin.setSingleStep(0.1)
        self.negative_distance_multiplier_spin.setToolTip("远区负样本的起始距离倍率，实际阈值 = 缓冲半径 × 倍率。")
        negative_row.addWidget(QLabel("负样本模式:", data_group))
        negative_row.addWidget(self.negative_sampling_mode_btn)
        negative_row.addWidget(QLabel("远区起始倍数:", data_group))
        negative_row.addWidget(self.negative_distance_multiplier_spin)
        negative_row.addStretch(1)
        data_layout.addLayout(negative_row)
        self._update_negative_sampling_controls(self.negative_sampling_mode_btn.isChecked())
        page_layout.addWidget(data_group)

        summary_group = QGroupBox("数据体检", page)
        summary_layout = QVBoxLayout(summary_group)
        self.dataset_summary_text = QTextEdit(summary_group)
        self.dataset_summary_text.setReadOnly(True)
        self.dataset_summary_text.setMaximumHeight(180)
        summary_layout.addWidget(self.dataset_summary_text)
        page_layout.addWidget(summary_group)

        model_group = QGroupBox("模型选择", page)
        model_layout = QVBoxLayout(model_group)
        self.model_checks = {}
        for index, model_name in enumerate(self.MODEL_FACTORIES):
            checkbox = QCheckBox(model_name, model_group)
            checkbox.setChecked(index == 0)
            self.model_checks[model_name] = checkbox
            model_layout.addWidget(checkbox)
        page_layout.addWidget(model_group)

        train_group = QGroupBox("训练配置", page)
        train_layout = QVBoxLayout(train_group)
        self.epochs_spin = QSpinBox(train_group)
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(50)
        self.batch_spin = QSpinBox(train_group)
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(32)
        self.lr_spin = QDoubleSpinBox(train_group)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setValue(0.001)
        self.optimizer_combo = QComboBox(train_group)
        self.optimizer_combo.addItems(["Adam", "AdamW", "SGD", "RMSprop"])
        for label_text, widget in [
            ("训练轮次:", self.epochs_spin),
            ("批量大小:", self.batch_spin),
            ("学习率:", self.lr_spin),
            ("优化器:", self.optimizer_combo),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text, train_group))
            row.addWidget(widget)
            train_layout.addLayout(row)
        page_layout.addWidget(train_group)

        row = QHBoxLayout()
        self.load_data_btn = QPushButton("加载数据", page)
        self.load_data_btn.clicked.connect(self.load_data)
        self.start_btn = QPushButton("开始对比", page)
        self.start_btn.clicked.connect(self.start_comparison)
        self.stop_btn = QPushButton("停止", page)
        self.stop_btn.clicked.connect(self.stop_comparison)
        row.addWidget(self.load_data_btn)
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        page_layout.addLayout(row)

        self.progress_bar = QProgressBar(page)
        page_layout.addWidget(self.progress_bar)
        self.log_text = QTextEdit(page)
        self.log_text.setReadOnly(True)
        page_layout.addWidget(self.log_text)
        return page

    def _create_results_tab(self):
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        page = QWidget(self)
        layout = QVBoxLayout(page)
        self.chart_figure = Figure(figsize=(8, 4))
        self.chart_canvas = FigureCanvasQTAgg(self.chart_figure)
        layout.addWidget(self.chart_canvas)

        self.results_table = QTableWidget(page)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.results_table)

        row = QHBoxLayout()
        row.addWidget(QLabel("图表类型:", page))
        self.chart_type_combo = QComboBox(page)
        self.chart_type_combo.addItems(
            [
                self.CHART_TEST_ACC,
                self.CHART_GENERALIZATION,
                self.CHART_VAL_MINERAL,
                self.CHART_TEST_MINERAL,
                self.CHART_VAL_EI,
                self.CHART_TEST_EI,
                self.CHART_COMPOSITE,
                self.CHART_TRAIN_TIME,
            ]
        )
        self.chart_type_combo.currentTextChanged.connect(self._update_chart)
        export_btn = QPushButton("导出 CSV", page)
        export_btn.clicked.connect(self.export_results)
        row.addWidget(self.chart_type_combo)
        row.addWidget(export_btn)
        layout.addLayout(row)
        return page

    def _create_automl_tab(self):
        page = QWidget(self)
        layout = QVBoxLayout(page)

        model_group = QGroupBox("选择模型", page)
        model_layout = QVBoxLayout(model_group)
        self.automl_model_checks = {}
        for model_name in list(self.MODEL_FACTORIES.keys()) + self.AUTOML_CLASSIC_MODELS:
            checkbox = QCheckBox(model_name, model_group)
            checkbox.setChecked(True)
            self.automl_model_checks[model_name] = checkbox
            model_layout.addWidget(checkbox)
        layout.addWidget(model_group)

        search_group = QGroupBox("搜索空间", page)
        search_layout = QVBoxLayout(search_group)
        self.patch_size_candidates_edit = QLineEdit(search_group)
        self.patch_size_candidates_edit.setText(str(self.patch_size_spin.value()))
        self.patch_size_candidates_edit.setToolTip("输入多个候选窗口大小，逗号分隔，例如 64, 96, 128。")
        self.patch_stride_candidates_edit = QLineEdit(search_group)
        self.patch_stride_candidates_edit.setText(str(self.patch_stride_spin.value()))
        self.patch_stride_candidates_edit.setToolTip("输入多个候选步长，逗号分隔，例如 16, 32。")
        self.buffer_radius_candidates_edit = QLineEdit(search_group)
        self.buffer_radius_candidates_edit.setText(str(self.buffer_radius_spin.value()))
        self.buffer_radius_candidates_edit.setToolTip("输入多个候选缓冲半径，逗号分隔，例如 300, 500, 800。")
        self.sampling_percentage_candidates_edit = QLineEdit(search_group)
        self.sampling_percentage_candidates_edit.setText("100")
        self.sampling_percentage_candidates_edit.setToolTip("输入 1-100 的百分比，或 0-1 的小数；例如 25, 50, 100。")
        self.balance_ratio_candidates_edit = QLineEdit(search_group)
        self.balance_ratio_candidates_edit.setPlaceholderText("例如: 1.0, 2.0")
        self.balance_ratio_candidates_edit.setToolTip("正/负样本比例；留空则使用当前默认负样本采样方式。")
        for label_text, widget in [
            ("窗口大小候选:", self.patch_size_candidates_edit),
            ("步长候选:", self.patch_stride_candidates_edit),
            ("缓冲半径候选:", self.buffer_radius_candidates_edit),
            ("总样本抽样比例候选:", self.sampling_percentage_candidates_edit),
            ("正/负样本比例候选:", self.balance_ratio_candidates_edit),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text, search_group))
            row.addWidget(widget)
            search_layout.addLayout(row)
        layout.addWidget(search_group)

        trials_group = QGroupBox("优化参数", page)
        trials_layout = QVBoxLayout(trials_group)
        self.stage1_trials_spin = QSpinBox(trials_group)
        self.stage1_trials_spin.setRange(1, 100)
        self.stage1_trials_spin.setValue(12)
        self.stage1_trials_spin.setToolTip("第 1 阶段试验次数：用于粗筛候选数据方案，数值越小越快。")
        self.stage1_epochs_spin = QSpinBox(trials_group)
        self.stage1_epochs_spin.setRange(1, 200)
        self.stage1_epochs_spin.setValue(12)
        self.stage1_epochs_spin.setToolTip("第 1 阶段训练轮次：每个 trial 内部的 epoch 数，决定粗筛时单次训练有多充分。")
        self.stage2_trials_spin = QSpinBox(trials_group)
        self.stage2_trials_spin.setRange(1, 300)
        self.stage2_trials_spin.setValue(50)
        self.stage2_trials_spin.setToolTip("第 2 阶段试验次数：用于对筛选后的方案进行正式搜索，数值越大越充分。")
        self.stage2_epochs_spin = QSpinBox(trials_group)
        self.stage2_epochs_spin.setRange(1, 500)
        self.stage2_epochs_spin.setValue(50)
        self.stage2_epochs_spin.setToolTip("第 2 阶段训练轮次：每个 trial 内部的 epoch 数，决定精筛时单次训练有多充分。")
        self.runtime_mode_combo = QComboBox(trials_group)
        self.runtime_mode_combo.addItems(["实用批量", "快速筛查", "深度搜索"])
        self.runtime_mode_combo.setToolTip(
            "快速筛查：减少候选数据方案，优先看结果是否可用。\n"
            "实用批量：默认平衡，适合日常批量运行。\n"
            "深度搜索：扩大候选数据方案范围，适合最终定稿。"
        )
        self.runtime_mode_hint_label = QLabel(trials_group)
        self.runtime_mode_hint_label.setWordWrap(True)
        self.runtime_mode_hint_label.setStyleSheet("color: #666666;")
        self.runtime_mode_combo.currentTextChanged.connect(self._update_runtime_mode_hint)
        self.workflow_output_dir_pick = QLineEdit(trials_group)
        self.workflow_output_dir_pick.setPlaceholderText("默认输出到 ./outputs/model_comparison/")
        row = QHBoxLayout()
        row.addWidget(QLabel("第1阶段试验次数（粗筛）:", trials_group))
        row.addWidget(self.stage1_trials_spin)
        trials_layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("第1阶段训练轮次（粗筛）:", trials_group))
        row.addWidget(self.stage1_epochs_spin)
        trials_layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("运行模式:", trials_group))
        row.addWidget(self.runtime_mode_combo)
        trials_layout.addLayout(row)
        trials_layout.addWidget(self.runtime_mode_hint_label)
        self._update_runtime_mode_hint(self.runtime_mode_combo.currentText())
        row = QHBoxLayout()
        row.addWidget(QLabel("第2阶段试验次数（精筛）:", trials_group))
        row.addWidget(self.stage2_trials_spin)
        trials_layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("第2阶段训练轮次（精筛）:", trials_group))
        row.addWidget(self.stage2_epochs_spin)
        trials_layout.addLayout(row)
        browse_output_btn = QPushButton("输出目录...", trials_group)
        browse_output_btn.clicked.connect(lambda: self._browse_directory(self.workflow_output_dir_pick, "选择自动试验输出目录"))
        row = QHBoxLayout()
        row.addWidget(QLabel("输出目录:", trials_group))
        row.addWidget(self.workflow_output_dir_pick)
        row.addWidget(browse_output_btn)
        trials_layout.addLayout(row)
        layout.addWidget(trials_group)

        workload_group = QGroupBox("工作量预估", page)
        workload_layout = QVBoxLayout(workload_group)
        self.workload_summary_text = QTextEdit(workload_group)
        self.workload_summary_text.setReadOnly(True)
        self.workload_summary_text.setMinimumHeight(170)
        self.workload_summary_text.setPlainText(
            "工作量预估：请先选择模型并输入搜索空间。数据加载后会自动补全实际 trial 数和耗时估计。"
        )
        workload_layout.addWidget(self.workload_summary_text)
        layout.addWidget(workload_group)

        progress_group = QGroupBox("工作流进度", page)
        progress_layout = QVBoxLayout(progress_group)
        self.workflow_stage_label = QLabel("尚未开始自动试验", progress_group)
        self.workflow_progress_bar = QProgressBar(progress_group)
        progress_layout.addWidget(self.workflow_stage_label)
        progress_layout.addWidget(self.workflow_progress_bar)
        layout.addWidget(progress_group)

        row = QHBoxLayout()
        self.automl_start_btn = QPushButton("开始自动优化", page)
        self.automl_start_btn.clicked.connect(self.start_automl_workflow)
        self.export_report_btn = QPushButton("导出 PDF 报告", page)
        self.export_report_btn.clicked.connect(self.export_automl_report)
        row.addWidget(self.automl_start_btn)
        row.addWidget(self.export_report_btn)
        layout.addLayout(row)

        self.automl_results_table = QTableWidget(page)
        self.automl_results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.automl_results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.automl_results_table.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.automl_results_table)

        details_group = QGroupBox("结果详情", page)
        details_layout = QVBoxLayout(details_group)
        self.automl_details_text = QTextEdit(details_group)
        self.automl_details_text.setReadOnly(True)
        details_layout.addWidget(self.automl_details_text)
        layout.addWidget(details_group)

        log_group = QGroupBox("优化日志", page)
        log_layout = QVBoxLayout(log_group)
        self.automl_log = QTextEdit(page)
        self.automl_log.setReadOnly(True)
        log_layout.addWidget(self.automl_log)
        layout.addWidget(log_group)
        return page

    def _create_artifact_tab(self):
        page = QWidget(self)
        layout = QVBoxLayout(page)

        summary_group = QGroupBox("当前模型包", page)
        summary_layout = QVBoxLayout(summary_group)
        self.workflow_output_path_edit = QLineEdit(summary_group)
        self.workflow_output_path_edit.setReadOnly(True)
        self.best_model_path_edit = QLineEdit(summary_group)
        self.best_model_path_edit.setReadOnly(True)
        self.best_manifest_path_edit = QLineEdit(summary_group)
        self.best_manifest_path_edit.setReadOnly(True)
        for label_text, widget in [
            ("输出目录:", self.workflow_output_path_edit),
            ("最佳模型文件:", self.best_model_path_edit),
            ("最佳清单文件:", self.best_manifest_path_edit),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text, summary_group))
            row.addWidget(widget)
            summary_layout.addLayout(row)
        self.artifact_details_text = QTextEdit(summary_group)
        self.artifact_details_text.setReadOnly(True)
        summary_layout.addWidget(self.artifact_details_text)
        layout.addWidget(summary_group)

        rebuild_group = QGroupBox("从模型包重建预测区", page)
        rebuild_layout = QVBoxLayout(rebuild_group)
        self.rebuild_manifest_edit = QLineEdit(rebuild_group)
        browse_manifest_btn = QPushButton("浏览清单...", rebuild_group)
        browse_manifest_btn.clicked.connect(self._browse_rebuild_manifest)
        row = QHBoxLayout()
        row.addWidget(QLabel("Manifest:", rebuild_group))
        row.addWidget(self.rebuild_manifest_edit)
        row.addWidget(browse_manifest_btn)
        rebuild_layout.addLayout(row)

        self.rebuild_output_dir_edit = QLineEdit(rebuild_group)
        browse_output_btn = QPushButton("输出目录...", rebuild_group)
        browse_output_btn.clicked.connect(lambda: self._browse_directory(self.rebuild_output_dir_edit, "选择重建输出目录"))
        row = QHBoxLayout()
        row.addWidget(QLabel("重建目录:", rebuild_group))
        row.addWidget(self.rebuild_output_dir_edit)
        row.addWidget(browse_output_btn)
        rebuild_layout.addLayout(row)

        row = QHBoxLayout()
        self.use_best_manifest_btn = QPushButton("使用当前最佳清单", rebuild_group)
        self.use_best_manifest_btn.clicked.connect(self._use_best_manifest_for_rebuild)
        self.rebuild_from_manifest_btn = QPushButton("重建预测区", rebuild_group)
        self.rebuild_from_manifest_btn.clicked.connect(self.rebuild_from_manifest)
        row.addWidget(self.use_best_manifest_btn)
        row.addWidget(self.rebuild_from_manifest_btn)
        rebuild_layout.addLayout(row)

        self.rebuild_log_text = QTextEdit(rebuild_group)
        self.rebuild_log_text.setReadOnly(True)
        rebuild_layout.addWidget(self.rebuild_log_text)
        layout.addWidget(rebuild_group)
        return page

    def _connect_signals(self):
        self.engine.log_message.connect(self.log_text.append)
        self.engine.task_progress.connect(self._update_progress)
        self.engine.experiment_completed.connect(self._show_results)
        self.engine.error_occurred.connect(self.log_text.append)
        self.mineral_mode_combo.currentTextChanged.connect(self._toggle_mineral_mode)
        self.all_mineral_edit.textChanged.connect(self._clear_manual_mineral_selection)

        invalidation_signals = [
            self.full_h5_edit.textChanged,
            self.train_mineral_edit.textChanged,
            self.val_mineral_edit.textChanged,
            self.test_mineral_edit.textChanged,
            self.no_ore_mineral_edit.textChanged,
            self.all_mineral_edit.textChanged,
            self.test_split_ratio_spin.valueChanged,
            self.train_region_edit.textChanged,
            self.test_region_edit.textChanged,
            self.region_buffer_spin.valueChanged,
            self.val_ratio_spin.valueChanged,
            self.buffer_radius_spin.valueChanged,
            self.negative_sampling_mode_btn.toggled,
            self.negative_distance_multiplier_spin.valueChanged,
            self.patch_size_spin.valueChanged,
            self.patch_stride_spin.valueChanged,
            self.n_blocks_spin.valueChanged,
            self.spatial_cluster_count_spin.valueChanged,
            self.spatial_train_ratio_spin.valueChanged,
            self.spatial_cv_folds_spin.valueChanged,
            self.batch_spin.valueChanged,
            self.patch_size_candidates_edit.textChanged,
            self.patch_stride_candidates_edit.textChanged,
            self.buffer_radius_candidates_edit.textChanged,
        ]
        for signal in invalidation_signals:
            signal.connect(self._invalidate_loaded_data)

        mineral_stat_signals = [
            self.full_h5_edit.textChanged,
            self.train_mineral_edit.textChanged,
            self.val_mineral_edit.textChanged,
            self.test_mineral_edit.textChanged,
            self.no_ore_mineral_edit.textChanged,
            self.all_mineral_edit.textChanged,
            self.test_split_ratio_spin.valueChanged,
            self.train_region_edit.textChanged,
            self.test_region_edit.textChanged,
            self.region_buffer_spin.valueChanged,
            self.val_ratio_spin.valueChanged,
            self.n_blocks_spin.valueChanged,
            self.spatial_cluster_count_spin.valueChanged,
            self.spatial_train_ratio_spin.valueChanged,
            self.spatial_cv_folds_spin.valueChanged,
        ]
        for signal in mineral_stat_signals:
            signal.connect(lambda *_: self._load_mineral_stats())

        self.automl_results_table.itemSelectionChanged.connect(self._update_automl_details_panel)
        self.rebuild_manifest_edit.textChanged.connect(lambda *_: self._refresh_action_buttons())
        self.rebuild_output_dir_edit.textChanged.connect(lambda *_: self._refresh_action_buttons())

        workload_signals = [
            self.buffer_radius_candidates_edit.textChanged,
            self.patch_size_candidates_edit.textChanged,
            self.patch_stride_candidates_edit.textChanged,
            self.sampling_percentage_candidates_edit.textChanged,
            self.balance_ratio_candidates_edit.textChanged,
            self.stage1_trials_spin.valueChanged,
            self.stage1_epochs_spin.valueChanged,
            self.stage2_trials_spin.valueChanged,
            self.stage2_epochs_spin.valueChanged,
        ]
        for signal in workload_signals:
            signal.connect(lambda *_: self._update_workload_estimate())
        for checkbox in self.automl_model_checks.values():
            checkbox.stateChanged.connect(lambda *_: self._update_workload_estimate())
        self.runtime_mode_combo.currentTextChanged.connect(lambda *_: self._update_workload_estimate())

    def _browse_file(self, line_edit, title):
        path, _ = QFileDialog.getOpenFileName(self, f"选择{title}", "", "H5 Files (*.h5)")
        if path:
            line_edit.setText(path)
            if line_edit is self.full_h5_edit:
                self._refresh_feature_channel_options(reset_selection=True)

    def _refresh_feature_channel_options(self, reset_selection=False):
        h5_path = self.full_h5_edit.text().strip()
        if not h5_path:
            self.available_feature_channels = []
            self.selected_feature_channels = None
            self.feature_channel_summary_label.setText("图层: 未选择特征 H5")
            return
        try:
            self.available_feature_channels = infer_h5_channel_names(h5_path)
        except Exception as exc:  # noqa: BLE001
            self.available_feature_channels = []
            if reset_selection:
                self.selected_feature_channels = None
            self.feature_channel_summary_label.setText(f"图层: 读取失败 ({exc})")
            return
        if reset_selection:
            self.selected_feature_channels = None
        if self.selected_feature_channels:
            valid = [index for index in self.selected_feature_channels if 0 <= index < len(self.available_feature_channels)]
            self.selected_feature_channels = valid if valid and len(valid) < len(self.available_feature_channels) else None
        self.feature_channel_summary_label.setText(
            describe_selected_channels(self.selected_feature_channels, self.available_feature_channels)
        )

    def _choose_feature_channels(self):
        self._refresh_feature_channel_options(reset_selection=False)
        if not self.available_feature_channels:
            QMessageBox.information(self, "图层选择", "当前 H5 中未识别到可选图层。")
            return
        prompt_lines = ["输入要参与建模的图层编号，使用逗号分隔。", "留空表示全选。", ""]
        prompt_lines.extend(f"{index}. {name}" for index, name in enumerate(self.available_feature_channels, start=1))
        current = ",".join(str(index + 1) for index in (self.selected_feature_channels or []))
        text, ok = QInputDialog.getText(self, "选择参与图层", "\n".join(prompt_lines), text=current)
        if not ok:
            return
        raw = str(text).strip()
        if not raw:
            self.selected_feature_channels = None
        else:
            try:
                chosen = sorted({int(part.strip()) - 1 for part in raw.split(",") if part.strip()})
            except ValueError:
                QMessageBox.warning(self, "输入错误", "图层编号必须是整数，并使用逗号分隔。")
                return
            if not chosen or min(chosen) < 0 or max(chosen) >= len(self.available_feature_channels):
                QMessageBox.warning(self, "输入错误", "存在超出范围的图层编号。")
                return
            self.selected_feature_channels = None if len(chosen) >= len(self.available_feature_channels) else chosen
        self.feature_channel_summary_label.setText(
            describe_selected_channels(self.selected_feature_channels, self.available_feature_channels)
        )

    def _browse_mineral_file(self, line_edit, title):
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"选择{title}",
            "",
            "支持的文件 (*.csv *.xlsx *.xls);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls)",
        )
        if path:
            line_edit.setText(path)

    def _browse_directory(self, line_edit, title):
        path = QFileDialog.getExistingDirectory(self, title, line_edit.text().strip() or "")
        if path:
            line_edit.setText(path)

    def _browse_rebuild_manifest(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择清单文件", "", "JSON Files (*.json)")
        if path:
            self.rebuild_manifest_edit.setText(path)

    def _clear_manual_mineral_selection(self, *_):
        self.manual_mineral_selection = None
        if hasattr(self, "manual_split_summary_label"):
            self.manual_split_summary_label.setStyleSheet("color: #666666;")
            self.manual_split_summary_label.setText(
                "尚未手工划分矿点。未勾选的矿点将作为训练矿点，内部验证仍按下方比例自动切分。"
            )

    def _set_manual_split_summary(self, train_count, test_count):
        if not hasattr(self, "manual_split_summary_label"):
            return
        if train_count <= 0 or test_count <= 0:
            self.manual_split_summary_label.setStyleSheet("color: #b00020;")
            self.manual_split_summary_label.setText(
                f"当前选择: 训练矿点 {train_count} 个，预留测试矿点 {test_count} 个。"
                "请至少保留 1 个训练矿点和 1 个预留测试矿点。"
            )
        else:
            self.manual_split_summary_label.setStyleSheet("color: #666666;")
            self.manual_split_summary_label.setText(
                f"已手工划分: 训练矿点 {train_count} 个，预留测试矿点 {test_count} 个。"
                "未勾选的矿点将作为训练矿点，内部验证仍按下方比例自动切分。"
            )

    def _manual_selection_for_path(self, mineral_path):
        if not self.manual_mineral_selection:
            return None
        if self.manual_mineral_selection.get("source_path") != mineral_path:
            return None
        return self.manual_mineral_selection

    def _prompt_manual_mineral_split(self, mineral_path, mineral_df):
        current_selection = self._manual_selection_for_path(mineral_path)
        reserved_rows = None if current_selection is None else current_selection.get("reserved_rows")
        dialog = MineralSelectionDialog(mineral_df, parent=self, reserved_rows=reserved_rows)
        if dialog.exec_() != QDialog.Accepted:
            return None

        train_minerals, test_minerals, reserved_rows = dialog.get_split()
        val_minerals = train_minerals.iloc[0:0].copy()
        self.manual_mineral_selection = {
            "source_path": mineral_path,
            "reserved_rows": set(reserved_rows),
            "train_minerals": train_minerals.reset_index(drop=True),
            "val_minerals": val_minerals.reset_index(drop=True),
            "test_minerals": test_minerals.reset_index(drop=True),
        }
        self._invalidate_loaded_data()
        self._set_manual_split_summary(len(train_minerals), len(test_minerals))
        self._load_mineral_stats()
        return self.manual_mineral_selection

    def _open_manual_mineral_split_dialog(self):
        mineral_path = self.all_mineral_edit.text().strip()
        if not mineral_path:
            QMessageBox.warning(self, "提示", "请先选择全部矿点文件。")
            return

        try:
            mineral_df = self._normalize_mineral_columns(
                self._load_table_file(mineral_path),
                "矿点文件",
            )
        except Exception as exc:
            QMessageBox.critical(self, "手工划分失败", f"矿点文件读取失败: {exc}")
            return

        self._prompt_manual_mineral_split(mineral_path, mineral_df)

    def _toggle_mineral_mode(self):
        mode = self.mineral_mode_combo.currentText()
        is_separate = mode == self.MODE_SEPARATE
        is_single = mode == self.MODE_SINGLE
        is_manual = mode == self.MODE_MANUAL
        is_region = mode == self.MODE_REGION
        self.separate_mineral_widget.setVisible(is_separate)
        self.single_mineral_widget.setVisible(is_single or is_manual or is_region)
        self.test_split_ratio_row.setVisible(is_single)
        if hasattr(self, "split_repeat_row"):
            self.split_repeat_row.setVisible(is_single)
        if hasattr(self, "batch_seed_row"):
            self.batch_seed_row.setVisible(is_single)
        if hasattr(self, "batch_split_btn"):
            self.batch_split_btn.setVisible(is_single)
        self.manual_split_row.setVisible(is_manual)
        self.region_split_row.setVisible(is_region)
        if hasattr(self, "region_preview_group"):
            self.region_preview_group.setVisible(is_region)
        self._load_mineral_stats()
        self._invalidate_loaded_data()

    def _update_negative_sampling_controls(self, enabled):
        if hasattr(self, "negative_sampling_mode_btn"):
            self.negative_sampling_mode_btn.setText("已启用远区负样本模式" if enabled else "启用远区负样本模式")
        if hasattr(self, "negative_distance_multiplier_spin"):
            self.negative_distance_multiplier_spin.setEnabled(bool(enabled) and not self._is_busy())

    def _set_dataset_summary_text(self, text):
        self.dataset_summary_text.setPlainText(text)

    def _is_busy(self):
        return any(
            thread is not None and thread.isRunning()
            for thread in [
                self.data_loader_thread,
                self.comparison_thread,
                self.automl_thread,
                self.workflow_thread,
                self.workflow_batch_thread,
                self.rebuild_thread,
            ]
        )

    def _build_dataset_bundle(self, *args, **kwargs):
        return self.data_builder.build_bundle(*args, **kwargs)

    def _refresh_action_buttons(self):
        busy = self._is_busy()
        is_single = self.mineral_mode_combo.currentText() == self.MODE_SINGLE
        bundle_ready = self.dataset_bundle is not None
        comparison_running = self.comparison_thread is not None and self.comparison_thread.isRunning()
        editable_widgets = [
            self.full_h5_edit,
            self.mineral_mode_combo,
            self.train_mineral_edit,
            self.val_mineral_edit,
            self.test_mineral_edit,
            self.no_ore_mineral_edit,
            self.all_mineral_edit,
            self.test_split_ratio_spin,
            self.manual_split_button,
            self.buffer_radius_spin,
            self.patch_size_spin,
            self.patch_stride_spin,
            self.n_blocks_spin,
            self.train_region_edit,
            self.test_region_edit,
            self.region_buffer_spin,
            self.region_preview_refresh_btn,
            self.batch_spin,
            self.negative_sampling_mode_btn,
            self.patch_size_candidates_edit,
            self.patch_stride_candidates_edit,
            self.buffer_radius_candidates_edit,
            self.sampling_percentage_candidates_edit,
            self.balance_ratio_candidates_edit,
            self.stage1_trials_spin,
            self.stage2_trials_spin,
            self.runtime_mode_combo,
            self.workflow_output_dir_pick,
            self.split_repeat_row,
            self.split_repeat_spin,
            self.batch_seed_row,
            self.batch_seed_spin,
            self.rebuild_manifest_edit,
            self.rebuild_output_dir_edit,
        ]
        for widget in editable_widgets:
            widget.setEnabled(not busy)
        self.val_ratio_spin.setEnabled(not busy)
        self.val_ratio_label.setEnabled(not busy)

        self.load_data_btn.setEnabled(not busy)
        self.start_btn.setEnabled(bundle_ready and not busy)
        self.stop_btn.setEnabled(comparison_running)
        self.automl_start_btn.setEnabled(not busy)
        if hasattr(self, "batch_split_btn"):
            self.batch_split_btn.setEnabled(is_single and not busy)
        self.export_report_btn.setEnabled(bool(self.automl_results) and not busy)
        self.use_best_manifest_btn.setEnabled(bool(self.best_manifest_path_edit.text().strip()) and not busy)
        self.rebuild_from_manifest_btn.setEnabled(bool(self.rebuild_manifest_edit.text().strip()) and not busy)
        if hasattr(self, "negative_sampling_mode_btn"):
            self._update_negative_sampling_controls(self.negative_sampling_mode_btn.isChecked())

    def _invalidate_loaded_data(self, *_):
        if self._is_busy() or self.dataset_bundle is None:
            self._refresh_action_buttons()
            return

        self.data_builder.release_bundle(self.dataset_bundle)
        self.dataset_bundle = None
        self._clear_automl_results_ui()
        self._clear_artifact_summary()
        self.progress_bar.setValue(0)
        self._set_dataset_summary_text("数据体检: 当前配置已变更，请重新加载数据。")
        self._refresh_action_buttons()
        self._update_workload_estimate()

    def _load_table_file(self, file_path):
        if file_path.lower().endswith(".csv"):
            last_error = None
            for encoding in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
                try:
                    return pd.read_csv(file_path, encoding=encoding)
                except UnicodeDecodeError as exc:
                    last_error = exc
            if last_error is not None:
                raise last_error
            return pd.read_csv(file_path)
        return pd.read_excel(file_path)

    def _find_column_alias(self, mineral_df, aliases):
        normalized_columns = {str(column).strip().lower(): column for column in mineral_df.columns}
        for alias in aliases:
            matched = normalized_columns.get(alias.strip().lower())
            if matched is not None:
                return matched
        return None

    def _normalize_mineral_columns(self, mineral_df, title):
        x_column = self._find_column_alias(
            mineral_df,
            ["x", "coord_x", "point_x", "east", "easting", "东坐标", "横坐标"],
        )
        y_column = self._find_column_alias(
            mineral_df,
            ["y", "coord_y", "point_y", "north", "northing", "北坐标", "纵坐标"],
        )
        if x_column is None or y_column is None:
            raise ValueError(f"{title} 缺少必要的 x/y 坐标列。")
        normalized_df = mineral_df.copy()
        normalized_df["x"] = pd.to_numeric(normalized_df[x_column], errors="coerce")
        normalized_df["y"] = pd.to_numeric(normalized_df[y_column], errors="coerce")
        normalized_df = normalized_df.dropna(subset=["x", "y"]).reset_index(drop=True)
        if normalized_df.empty:
            raise ValueError(f"{title} 未找到可用的 X/Y 坐标数据。")
        return normalized_df

    def _load_optional_no_ore_minerals(self):
        no_ore_widget = getattr(self, "no_ore_mineral_edit", None)
        if no_ore_widget is None:
            return None
        path = no_ore_widget.text().strip()
        if not path:
            return None
        return self._normalize_mineral_columns(
            self._load_table_file(path),
            "无矿钻孔/坐标文件",
        )

    def _parse_region_text(self, text, title):
        cleaned = str(text or "").replace("，", ",").replace(";", ",").strip()
        parts = [chunk.strip() for chunk in cleaned.split(",") if chunk.strip()]
        if len(parts) != 4:
            raise ValueError(f"{title} 需填写 xmin,xmax,ymin,ymax 四个坐标值。")
        try:
            xmin, xmax, ymin, ymax = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"{title} 包含无法识别的数字。") from exc
        return {
            "xmin": min(xmin, xmax),
            "xmax": max(xmin, xmax),
            "ymin": min(ymin, ymax),
            "ymax": max(ymin, ymax),
        }

    def _collect_spatial_region_config(self):
        if self.mineral_mode_combo.currentText() != self.MODE_REGION:
            return None
        return serialize_region_split_config(
            {
                "train_region": self._parse_region_text(self.train_region_edit.text(), "训练区域"),
                "test_region": self._parse_region_text(self.test_region_edit.text(), "预测区域"),
                "buffer_distance": float(self.region_buffer_spin.value()),
            }
        )

    def _split_frame_for_region_mode(self, frame, title, *, require_train_test=True):
        region_config = self._collect_spatial_region_config()
        if region_config is None:
            raise ValueError("当前未启用坐标区域切分模式。")
        split_result = split_dataframe_by_regions(
            frame,
            region_config["train_region"],
            region_config["test_region"],
            buffer_distance=float(region_config["buffer_distance"]),
            x_column="x",
            y_column="y",
        )
        if require_train_test:
            train_frame = split_result["train"]
            test_frame = split_result["test"]
            if len(train_frame) == 0:
                raise ValueError(f"{title} 在训练区域内为空，请检查坐标范围。")
            if len(test_frame) == 0:
                raise ValueError(f"{title} 在预测区域内为空，请检查坐标范围。")
        return split_result

    def _draw_region_preview_rectangle(self, ax, region, *, color, label):
        if not region:
            return
        try:
            xmin = float(region.get("xmin"))
            xmax = float(region.get("xmax"))
            ymin = float(region.get("ymin"))
            ymax = float(region.get("ymax"))
        except (TypeError, ValueError, AttributeError):
            return
        from matplotlib.colors import to_rgba
        from matplotlib.patches import Rectangle

        width = xmax - xmin
        height = ymax - ymin
        if width <= 0 or height <= 0:
            return
        rect = Rectangle(
            (xmin, ymin),
            width,
            height,
            facecolor=to_rgba(color, 0.12),
            edgecolor=color,
            linewidth=1.8,
            linestyle="-",
            label=label,
            zorder=1,
        )
        ax.add_patch(rect)
        xs = [xmin, xmax, xmax, xmin, xmin]
        ys = [ymin, ymin, ymax, ymax, ymin]
        ax.plot(xs, ys, color=color, linewidth=1.8, linestyle="--", label="_nolegend_", zorder=2)

    def _update_region_preview(self, *_):
        if not hasattr(self, "region_preview_canvas") or not hasattr(self, "region_preview_ax"):
            return

        ax = self.region_preview_ax
        ax.clear()
        ax.set_title("区域预览")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.18)
        ax.set_axisbelow(True)

        if self.mineral_mode_combo.currentText() != self.MODE_REGION:
            ax.text(
                0.5,
                0.5,
                "只有在“坐标区域切分”模式下才显示预览",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="#666666",
            )
            self.region_preview_summary_label.setText("切换到坐标区域切分模式后，可实时查看训练区、预测区与灰区。")
            self.region_preview_canvas.draw_idle()
            return

        mineral_path = self.all_mineral_edit.text().strip()
        if not mineral_path:
            ax.text(
                0.5,
                0.5,
                "请先选择全部矿点文件",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="#666666",
            )
            self.region_preview_summary_label.setText("尚未选择全部矿点文件。")
            self.region_preview_canvas.draw_idle()
            return

        try:
            mineral_df = self._normalize_mineral_columns(
                self._load_table_file(mineral_path),
                "矿点文件",
            )
            region_split = self._split_frame_for_region_mode(mineral_df, "矿点文件", require_train_test=False)
            region_config = self._collect_spatial_region_config()
            if region_config is None:
                raise ValueError("当前未启用坐标区域切分模式。")
            no_ore_df = None
            if getattr(self, "no_ore_mineral_edit", None) is not None and self.no_ore_mineral_edit.text().strip():
                no_ore_df = self._load_optional_no_ore_minerals()
        except Exception as exc:
            ax.text(
                0.5,
                0.5,
                f"预览失败: {exc}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="#b00020",
                wrap=True,
            )
            self.region_preview_summary_label.setText(f"预览失败: {exc}")
            self.region_preview_canvas.draw_idle()
            return

        categories = [
            (region_split["train"], "训练区", "#2e8b57", "o", 28, 0.85),
            (region_split["test"], "预测区", "#1f77b4", "o", 28, 0.85),
            (region_split["gray"], "灰区", "#f39c12", "o", 24, 0.65),
            (region_split["outside"], "区域外", "#95a5a6", "o", 20, 0.45),
        ]
        for frame, label, color, marker, size, alpha in categories:
            if frame is None or len(frame) == 0:
                continue
            ax.scatter(
                frame["x"],
                frame["y"],
                s=size,
                c=color,
                marker=marker,
                alpha=alpha,
                edgecolors="none",
                label=label,
                zorder=4 if label in {"训练区", "预测区"} else 3,
            )

        if no_ore_df is not None and len(no_ore_df) > 0:
            ax.scatter(
                no_ore_df["x"],
                no_ore_df["y"],
                s=44,
                c="#111111",
                marker="x",
                linewidths=1.5,
                label="无矿锚点",
                zorder=5,
            )

        self._draw_region_preview_rectangle(
            ax,
            region_config.get("train_region"),
            color="#2e8b57",
            label="训练区范围",
        )
        self._draw_region_preview_rectangle(
            ax,
            region_config.get("test_region"),
            color="#1f77b4",
            label="预测区范围",
        )

        all_x = list(mineral_df["x"].astype(float).tolist())
        all_y = list(mineral_df["y"].astype(float).tolist())
        if no_ore_df is not None and len(no_ore_df) > 0:
            all_x.extend(no_ore_df["x"].astype(float).tolist())
            all_y.extend(no_ore_df["y"].astype(float).tolist())
        for region in [region_config.get("train_region"), region_config.get("test_region")]:
            if region:
                try:
                    all_x.extend([float(region["xmin"]), float(region["xmax"])])
                    all_y.extend([float(region["ymin"]), float(region["ymax"])])
                except (TypeError, ValueError, KeyError):
                    pass
        if all_x and all_y:
            x_min = min(all_x)
            x_max = max(all_x)
            y_min = min(all_y)
            y_max = max(all_y)
            x_span = max(x_max - x_min, 0.0)
            y_span = max(y_max - y_min, 0.0)
            padding_x = max(x_span * 0.04, 0.5)
            padding_y = max(y_span * 0.04, 0.5)
            ax.set_xlim(x_min - padding_x, x_max + padding_x)
            ax.set_ylim(y_min - padding_y, y_max + padding_y)

        ax.set_aspect("equal", adjustable="box")
        handles, labels = ax.get_legend_handles_labels()
        legend_items = []
        seen_labels = set()
        for handle, label in zip(handles, labels):
            if not label or label == "_nolegend_" or label in seen_labels:
                continue
            seen_labels.add(label)
            legend_items.append((handle, label))
        if legend_items:
            legend_handles, legend_labels = zip(*legend_items)
            legend_cols = 2 if len(legend_labels) <= 4 else 3
            ax.legend(
                legend_handles,
                legend_labels,
                loc="best",
                fontsize=7,
                frameon=True,
                ncol=legend_cols,
                framealpha=0.9,
                borderpad=0.35,
                handletextpad=0.45,
                columnspacing=0.9,
                labelspacing=0.25,
            )

        summary_text = (
            f"训练区 {len(region_split['train'])} 个 | "
            f"预测区 {len(region_split['test'])} 个 | "
            f"灰区 {len(region_split['gray'])} 个 | "
            f"区域外 {len(region_split['outside'])} 个"
        )
        if no_ore_df is not None and len(no_ore_df) > 0:
            summary_text += f" | 无矿点 {len(no_ore_df)} 个"
        self.region_preview_summary_label.setText(summary_text)
        self.region_preview_canvas.draw_idle()

    def _get_stratify_values(self, mineral_df):
        label_column = self._find_column_alias(
            mineral_df,
            ["标签", "label", "label_id", "class", "类别", "type", "成因类型"],
        )
        if label_column is None or mineral_df[label_column].isna().any():
            return None
        counts = mineral_df[label_column].value_counts()
        return None if (counts < 2).any() else mineral_df[label_column]

    def _safe_split_minerals(self, mineral_df, test_size, random_state=42):
        stratify = self._get_stratify_values(mineral_df)
        try:
            return train_test_split(
                mineral_df,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )
        except ValueError:
            return train_test_split(
                mineral_df,
                test_size=test_size,
                random_state=random_state,
                shuffle=True,
            )

    def _split_single_mode_minerals(self, all_minerals, split_seed=42):
        if len(all_minerals) < 2:
            raise ValueError("单文件模式至少需要 2 个矿点，才能拆分开发区和测试区。")

        dev_minerals, test_minerals = self._safe_split_minerals(
            all_minerals,
            self.test_split_ratio_spin.value(),
            random_state=split_seed,
        )
        dev_minerals = dev_minerals.reset_index(drop=True)
        test_minerals = test_minerals.reset_index(drop=True)
        if len(dev_minerals) == 0:
            raise ValueError("划分测试矿点后开发区矿点为空。")
        val_minerals = dev_minerals.iloc[0:0].copy()
        return (
            dev_minerals.reset_index(drop=True),
            val_minerals.reset_index(drop=True),
            test_minerals,
        )

    def _load_mineral_stats(self):
        train_count = 0
        val_count = 0
        test_count = 0
        errors = []
        no_ore_count = 0
        no_ore_minerals = None
        try:
            no_ore_minerals = self._load_optional_no_ore_minerals()
            if no_ore_minerals is not None:
                no_ore_count = len(no_ore_minerals)
        except Exception as exc:
            errors.append(f"无矿钻孔/坐标读取失败: {exc}")
        mode = self.mineral_mode_combo.currentText()
        if mode == self.MODE_SINGLE and self.all_mineral_edit.text().strip():
            try:
                all_minerals = self._normalize_mineral_columns(
                    self._load_table_file(self.all_mineral_edit.text().strip()),
                    "矿点文件",
                )
                train_minerals, val_minerals, test_minerals = self._split_single_mode_minerals(all_minerals)
                train_count = len(train_minerals)
                val_count = len(val_minerals)
                test_count = len(test_minerals)
            except Exception as exc:
                errors.append(f"矿点文件读取失败: {exc}")
        elif mode == self.MODE_MANUAL and self.all_mineral_edit.text().strip():
            try:
                mineral_path = self.all_mineral_edit.text().strip()
                selection = self._manual_selection_for_path(mineral_path)
                if selection is None:
                    all_minerals = self._normalize_mineral_columns(
                        self._load_table_file(mineral_path),
                        "矿点文件",
                    )
                    text = f"矿点统计: 全部矿点 {len(all_minerals)} 个，尚未手工划分。"
                    if hasattr(self, "manual_split_summary_label"):
                        self.manual_split_summary_label.setStyleSheet("color: #666666;")
                        self.manual_split_summary_label.setText(
                            "尚未手工划分矿点。未勾选的矿点将作为训练矿点，内部验证仍按下方比例自动切分。"
                        )
                else:
                    train_count = len(selection["train_minerals"])
                    test_count = len(selection["test_minerals"])
                    self._set_manual_split_summary(train_count, test_count)
                    text = f"矿点统计: 训练矿点 {train_count} 个，预留测试矿点 {test_count} 个。"
                if errors:
                    text = text + (f" | 无矿钻孔/坐标 {no_ore_count} 个" if no_ore_count else " | 无矿钻孔/坐标 未提供")
                    text = text + " 错误: " + "; ".join(errors)
                self.mineral_stats_label.setText(text)
                self._update_region_preview()
                return
            except Exception as exc:
                errors.append(f"矿点文件读取失败: {exc}")
        elif mode == self.MODE_REGION and self.all_mineral_edit.text().strip():
            try:
                all_minerals = self._normalize_mineral_columns(
                    self._load_table_file(self.all_mineral_edit.text().strip()),
                    "矿点文件",
                )
                region_split = self._split_frame_for_region_mode(all_minerals, "矿点文件")
                train_count = len(region_split["train"])
                test_count = len(region_split["test"])
                gray_count = len(region_split["gray"])
                outside_count = len(region_split["outside"])
                text = (
                    f"矿点统计: 训练区域矿点 {train_count} 个，预测区域矿点 {test_count} 个，"
                    f"灰区剔除 {gray_count} 个，区域外 {outside_count} 个。"
                )
                if no_ore_count:
                    try:
                        no_ore_split = (
                            self._split_frame_for_region_mode(
                                no_ore_minerals,
                                "无矿钻孔/坐标文件",
                                require_train_test=False,
                            )
                            if no_ore_minerals is not None
                            else None
                        )
                    except Exception as exc:
                        errors.append(f"无矿钻孔/坐标区域划分失败: {exc}")
                        no_ore_split = None
                    if no_ore_split is not None:
                        text += (
                            f" | 无矿钻孔/坐标: 训练区 {len(no_ore_split['train'])} 个，"
                            f"预测区 {len(no_ore_split['test'])} 个，灰区 {len(no_ore_split['gray'])} 个。"
                        )
                    else:
                        text += f" | 无矿钻孔/坐标 {no_ore_count} 个。"
                else:
                    text += " | 无矿钻孔/坐标 未提供。"
                if errors:
                    text = text + " 错误: " + "; ".join(errors)
                self.mineral_stats_label.setText(text)
                return
            except Exception as exc:
                errors.append(f"坐标区域划分失败: {exc}")
        elif mode == self.MODE_SEPARATE:
            path_specs = [
                ("训练矿点读取失败", self.train_mineral_edit.text().strip(), "训练矿点文件"),
                ("补充训练矿点读取失败", self.val_mineral_edit.text().strip(), "补充训练矿点文件"),
                ("测试矿点读取失败", self.test_mineral_edit.text().strip(), "测试矿点文件"),
            ]
            counts = []
            for prefix, file_path, title in path_specs:
                if not file_path:
                    counts.append(0)
                    continue
                try:
                    counts.append(
                        len(
                            self._normalize_mineral_columns(
                                self._load_table_file(file_path),
                                title,
                            )
                        )
                    )
                except Exception as exc:
                    counts.append(0)
                    errors.append(f"{prefix}: {exc}")
            train_count, val_count, test_count = counts
            train_total = train_count + val_count
        else:
            train_total = train_count

        if mode == self.MODE_SEPARATE:
            text = (
                f"矿点统计: 训练矿点 {train_total} 个（其中补充训练 {val_count} 个），"
                f"预留测试矿点 {test_count} 个。"
            )
        elif mode == self.MODE_MANUAL:
            text = f"矿点统计: 训练矿点 {train_count} 个，预留测试矿点 {test_count} 个。"
        elif mode == self.MODE_REGION:
            text = f"矿点统计: 训练区域矿点 {train_count} 个，预测区域矿点 {test_count} 个。"
        else:
            text = f"矿点统计: 开发矿点 {train_count} 个，预留测试矿点 {test_count} 个。"
        text = text + (f" | 无矿钻孔/坐标 {no_ore_count} 个" if no_ore_count else " | 无矿钻孔/坐标 未提供")
        if errors:
            text = text + " 错误: " + "; ".join(errors)
        self.mineral_stats_label.setText(text)
        self._update_region_preview()

    def _collect_dataset_request(self, *, all_minerals=None, split_seed=42):
        h5_path = self.full_h5_edit.text().strip()
        if not h5_path:
            raise ValueError("请选择完整数据集 H5 文件。")

        if self.mineral_mode_combo.currentText() == self.MODE_SINGLE:
            mineral_path = self.all_mineral_edit.text().strip()
            if not mineral_path:
                raise ValueError("请选择矿点文件。")
            if all_minerals is None:
                all_minerals = self._normalize_mineral_columns(
                    self._load_table_file(mineral_path),
                    "矿点文件",
                )
            train_minerals, val_minerals, test_minerals = self._split_single_mode_minerals(
                all_minerals,
                split_seed=split_seed,
            )
        elif self.mineral_mode_combo.currentText() == self.MODE_MANUAL:
            mineral_path = self.all_mineral_edit.text().strip()
            if not mineral_path:
                raise ValueError("请选择矿点文件。")
            selection = self._manual_selection_for_path(mineral_path)
            if selection is None:
                all_minerals = self._normalize_mineral_columns(
                    self._load_table_file(mineral_path),
                    "矿点文件",
                )
                selection = self._prompt_manual_mineral_split(mineral_path, all_minerals)
            if selection is None:
                return None
            train_minerals = selection["train_minerals"]
            val_minerals = selection["val_minerals"]
            test_minerals = selection["test_minerals"]
        elif self.mineral_mode_combo.currentText() == self.MODE_REGION:
            mineral_path = self.all_mineral_edit.text().strip()
            if not mineral_path:
                raise ValueError("请选择矿点文件。")
            if all_minerals is None:
                all_minerals = self._normalize_mineral_columns(
                    self._load_table_file(mineral_path),
                    "矿点文件",
                )
            region_split = self._split_frame_for_region_mode(all_minerals, "矿点文件")
            train_minerals = region_split["train"]
            val_minerals = train_minerals.iloc[0:0].copy()
            test_minerals = region_split["test"]
        else:
            train_path = self.train_mineral_edit.text().strip()
            val_path = self.val_mineral_edit.text().strip()
            test_path = self.test_mineral_edit.text().strip()
            if not train_path or not val_path or not test_path:
                raise ValueError("请分别选择训练、补充训练和测试矿点文件。")
            train_primary = self._normalize_mineral_columns(
                self._load_table_file(train_path),
                "训练矿点文件",
            )
            train_supplement = self._normalize_mineral_columns(
                self._load_table_file(val_path),
                "补充训练矿点文件",
            )
            train_minerals = pd.concat([train_primary, train_supplement], ignore_index=True)
            val_minerals = train_minerals.iloc[0:0].copy()
            test_minerals = self._normalize_mineral_columns(
                self._load_table_file(test_path),
                "测试矿点文件",
            )

        no_ore_minerals = self._load_optional_no_ore_minerals()

        build_config = {
            "batch_size": int(self.batch_spin.value()),
            "val_ratio": float(self.val_ratio_spin.value()),
            "buffer_radius": float(self.buffer_radius_spin.value()),
            "negative_sampling_mode": "far_distance" if self.negative_sampling_mode_btn.isChecked() else "default",
            "negative_distance_multiplier": float(self.negative_distance_multiplier_spin.value()),
            "patch_size": int(self.patch_size_spin.value()),
            "patch_stride": int(self.patch_stride_spin.value()),
            "use_reflect_padding": bool(self.reflect_padding_check.isChecked()),
            "selected_channels": None if self.selected_feature_channels is None else list(self.selected_feature_channels),
            "n_blocks": int(self.n_blocks_spin.value()),
            "spatial_region_split": self._collect_spatial_region_config(),
            "split_mode": (
                "spatial_region"
                if self.mineral_mode_combo.currentText() == self.MODE_REGION
                else "manual"
                if self.mineral_mode_combo.currentText() == self.MODE_MANUAL
                else "single"
                if self.mineral_mode_combo.currentText() == self.MODE_SINGLE
                else "separate"
            ),
        }
        return (
            h5_path,
            train_minerals.reset_index(drop=True),
            val_minerals.reset_index(drop=True),
            test_minerals.reset_index(drop=True),
            None if no_ore_minerals is None else no_ore_minerals.reset_index(drop=True),
            self.data_builder.detect_h5_mode(h5_path),
            build_config,
        )

    def load_data(self):
        if self._is_busy():
            return

        try:
            dataset_request = self._collect_dataset_request()
            if dataset_request is None:
                return
            h5_path, train_minerals, val_minerals, test_minerals, no_ore_minerals, h5_mode, build_config = dataset_request
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", f"数据校验失败: {exc}")
            return

        self.log_text.append("开始加载数据...")
        self.progress_bar.setValue(0)
        self.data_loader_thread = DataLoaderThread(
            self,
            h5_path,
            train_minerals,
            val_minerals,
            test_minerals,
            h5_mode,
            build_config,
            no_ore_minerals=no_ore_minerals,
        )
        self.data_loader_thread.progress.connect(self.log_text.append)
        self.data_loader_thread.finished.connect(self._on_data_loaded)
        self.data_loader_thread.error.connect(self._on_data_load_error)
        self.data_loader_thread.finished.connect(self._on_worker_finished)
        self.data_loader_thread.error.connect(self._on_worker_finished)
        self._refresh_action_buttons()
        self.data_loader_thread.start()

    def _format_dataset_summary(self, summary):
        train_positive_count = int(summary.get("train_positive_count", 0) or 0)
        train_negative_count = int(summary.get("train_negative_count", 0) or 0)
        supervised_train_negative_count = min(train_positive_count, train_negative_count)
        supervised_train_total_count = train_positive_count + supervised_train_negative_count
        pu_train_unlabeled_count = train_negative_count
        pu_train_total_count = train_positive_count + pu_train_unlabeled_count

        lines = [
            f"H5 模式: {summary.get('h5_mode', '-')}",
            f"输入通道: {summary.get('input_channels', '-')}, 窗口大小: {summary.get('image_size', '-')}, 类别数: {summary.get('num_classes', '-')}",
            f"样本数: 内部训练={summary.get('train_sample_count', 0)}, 内部验证={summary.get('val_sample_count', 0)}, 测试={summary.get('test_sample_count', 0)}",
            f"矿点划分模式: {self.mineral_mode_combo.currentText()}",
            f"切窗补边: {'启用 reflect 补边' if summary.get('reflect_padding') else '未启用'}",
            f"参与图层: {', '.join((summary.get('selected_channel_names') or [])[:6]) or '全部'}",
            f"抽样比例: {self._format_sampling_percentage_display(summary.get('sampling_percentage'))}",
            f"正/负样本比例: {self._format_balance_ratio_display(summary.get('balance_ratio')) or '-'}",
            f"负样本策略: {self._format_negative_sampling_mode_display(summary)}",
            f"负/正样本倍数: {summary.get('negative_ratio', '-') or '-'}",
            (
                "训练视图样本数(监督1:1): "
                f"正样本={train_positive_count}, 负样本={supervised_train_negative_count}, 总计={supervised_train_total_count}"
            ),
            (
                "训练视图样本数(PU): "
                f"正样本={train_positive_count}, 未标记样本={pu_train_unlabeled_count}, 总计={pu_train_total_count}"
            ),
            f"正样本: 内部训练={summary.get('train_positive_count', 0)}, 内部验证={summary.get('val_positive_count', 0)}, 测试={summary.get('test_positive_count', 0)}",
            f"负样本: 内部训练={summary.get('train_negative_count', 0)}, 内部验证={summary.get('val_negative_count', 0)}, 测试={summary.get('test_negative_count', 0)}",
            f"冲突样本数: {summary.get('conflict_sample_count', 0)}",
            f"无矿钻孔/坐标: {'启用' if summary.get('no_ore_active') else '未提供'}",
            f"无矿锚点: 文件点数={summary.get('no_ore_point_count', 0)}, 覆盖样本={summary.get('no_ore_sample_count', 0)}, 冲突覆盖={summary.get('no_ore_conflict_count', 0)}",
            f"矿点数: 内部训练={summary.get('train_mineral_count', 0)}, 内部验证={summary.get('val_mineral_count', 0)}, 测试={summary.get('test_mineral_count', 0)}",
        ]
        if summary.get("spatial_region_active"):
            train_region = summary.get("spatial_region_train_bounds") or {}
            test_region = summary.get("spatial_region_test_bounds") or {}
            lines.extend(
                [
                    (
                        "坐标区域切分: "
                        f"训练区=({train_region.get('xmin', '-')}, {train_region.get('xmax', '-')}, "
                        f"{train_region.get('ymin', '-')}, {train_region.get('ymax', '-')})"
                    ),
                    (
                        "预测区域: "
                        f"({test_region.get('xmin', '-')}, {test_region.get('xmax', '-')}, "
                        f"{test_region.get('ymin', '-')}, {test_region.get('ymax', '-')})"
                    ),
                    (
                        f"区域样本: 训练区原始={summary.get('spatial_region_train_sample_count', 0)}, "
                        f"预测区原始={summary.get('spatial_region_test_sample_count', 0)}, "
                        f"灰区剔除={summary.get('spatial_region_gray_sample_count', 0)}, "
                        f"区域外={summary.get('spatial_region_outside_sample_count', 0)}"
                    ),
                    (
                        f"区域无矿负样本: 训练区={summary.get('spatial_region_train_no_ore_sample_count', 0)}, "
                        f"预测区={summary.get('spatial_region_test_no_ore_sample_count', 0)}, "
                        f"边界缓冲带={summary.get('spatial_region_buffer_distance', 0.0)} 米"
                    ),
                ]
            )
        split_info = summary.get("split_info") or {}
        if split_info:
            lines.extend(
                [
                    f"空间验证: requested={split_info.get('requested_strategy', '-')}, effective={split_info.get('effective_strategy', '-')}",
                    f"空间分块: n_blocks={summary.get('n_blocks', split_info.get('n_blocks', '-'))}, 内部验证比例={summary.get('validation_split', split_info.get('validation_split', '-'))}",
                    f"缓冲剔除: enabled={split_info.get('buffer_requested', False)}, applied={split_info.get('buffer_applied', False)}, removed={split_info.get('buffer_removed_count', 0)}, distance={split_info.get('buffer_distance', 0.0)}",
                    f"回退原因: {split_info.get('fallback_reason', '-')}",
                ]
            )
        if "total_patch_count" in summary:
            lines.append(f"总切片数: {summary.get('total_patch_count', 0)}")
        if summary.get("label_mapping"):
            lines.append(f"标签映射: {summary['label_mapping']}")
        return "\n".join(lines)

    def _on_data_loaded(self, dataset_bundle):
        self.data_builder.release_bundle(self.dataset_bundle)
        self.dataset_bundle = dataset_bundle
        self._clear_automl_results_ui()
        self._clear_artifact_summary()
        self._set_dataset_summary_text(self._format_dataset_summary(dataset_bundle.dataset_summary))
        self._update_workload_estimate()
        self.log_text.append("数据加载完成。")
        QMessageBox.information(self, "成功", "数据加载完成，可以开始模型对比和自动优化。")

    def _on_data_load_error(self, error_msg):
        self.log_text.append(f"数据加载失败: {error_msg}")
        self._set_dataset_summary_text("数据体检: 加载失败，请检查配置。")
        self._update_workload_estimate()
        QMessageBox.critical(self, "错误", f"数据加载失败: {error_msg}")

    def _on_worker_finished(self, *_):
        if self.data_loader_thread is not None and not self.data_loader_thread.isRunning():
            self.data_loader_thread = None
        if self.comparison_thread is not None and not self.comparison_thread.isRunning():
            self.comparison_thread = None
        if self.automl_thread is not None and not self.automl_thread.isRunning():
            self.automl_thread = None
            self.automl_engine = None
        if self.workflow_thread is not None and not self.workflow_thread.isRunning():
            self.workflow_thread = None
            self.workflow_orchestrator = None
        if self.workflow_batch_thread is not None and not self.workflow_batch_thread.isRunning():
            self.workflow_batch_thread = None
            self.workflow_orchestrator = None
            self.workflow_batch_active = False
            self.workflow_batch_total_runs = 0
            self.workflow_batch_current_run = 0
        if self.rebuild_thread is not None and not self.rebuild_thread.isRunning():
            self.rebuild_thread = None
        self._refresh_action_buttons()

    def _update_progress(self, task_id, current, total):
        del task_id
        self.progress_bar.setValue(0 if total <= 0 else int((current / total) * 100))

    def start_comparison(self):
        if self._is_busy():
            return
        if self.dataset_bundle is None:
            QMessageBox.warning(self, "提示", "请先加载数据。")
            return

        selected_models = [name for name, checkbox in self.model_checks.items() if checkbox.isChecked()]
        if not selected_models:
            QMessageBox.warning(self, "提示", "请至少选择一个模型。")
            return

        self.engine.clear_experiments()
        config = {
            "num_classes": self.dataset_bundle.dataset_meta["num_classes"],
            "input_channels": self.dataset_bundle.dataset_meta["input_channels"],
            "image_size": self.dataset_bundle.dataset_meta["image_size"],
            "epochs": int(self.epochs_spin.value()),
            "learning_rate": float(self.lr_spin.value()),
            "batch_size": int(self.batch_spin.value()),
            "optimizer": self.optimizer_combo.currentText(),
            "device": "auto",
            "prediction_patch_size": int(
                self.dataset_bundle.dataset_summary.get(
                    "patch_size",
                    self.dataset_bundle.dataset_meta["image_size"],
                )
            ),
            "prediction_batch_size": max(64, int(self.batch_spin.value())),
        }
        for model_name in selected_models:
            self.engine.add_experiment(model_name, self.MODEL_FACTORIES[model_name], config)

        self.log_text.append("开始执行模型对比...")
        self.progress_bar.setValue(0)
        self.comparison_thread = ComparisonRunThread(self.engine, self.dataset_bundle)
        self.comparison_thread.error.connect(
            lambda message: QMessageBox.critical(self, "模型对比失败", message)
        )
        self.comparison_thread.finished.connect(self._on_worker_finished)
        self._refresh_action_buttons()
        self.comparison_thread.start()

    def stop_comparison(self):
        if self.comparison_thread is None or not self.comparison_thread.isRunning():
            return
        self.log_text.append("正在请求停止当前训练...")
        self.comparison_thread.stop()
        self.stop_btn.setEnabled(False)

    def _create_automl_wrapper(self, model_name, dataset_meta=None, profile="stage2"):
        meta = dataset_meta or self.dataset_bundle.dataset_meta
        fixed_epochs = None
        if profile == "stage1" and model_name in self.MODEL_FACTORIES:
            fixed_epochs = int(self.stage1_epochs_spin.value())
        elif profile == "stage2" and model_name in self.MODEL_FACTORIES:
            fixed_epochs = int(self.stage2_epochs_spin.value())
        if model_name == "CNN":
            return DeepLearningWrapper(
                CNNModel,
                meta["num_classes"],
                meta["input_channels"],
                meta["image_size"],
                model_name=model_name,
                fixed_epochs=fixed_epochs,
                persist_artifacts=False,
            )
        if model_name == "ResNet-18":
            return DeepLearningWrapper(
                ResNet18,
                meta["num_classes"],
                meta["input_channels"],
                meta["image_size"],
                model_name=model_name,
                fixed_epochs=fixed_epochs,
                persist_artifacts=False,
            )
        if model_name == "ResNet-34":
            return DeepLearningWrapper(
                ResNet34,
                meta["num_classes"],
                meta["input_channels"],
                meta["image_size"],
                model_name=model_name,
                fixed_epochs=fixed_epochs,
                persist_artifacts=False,
            )
        if model_name == "ResNet-50":
            return DeepLearningWrapper(
                ResNet50,
                meta["num_classes"],
                meta["input_channels"],
                meta["image_size"],
                model_name=model_name,
                fixed_epochs=fixed_epochs,
                persist_artifacts=False,
            )
        if model_name == "Random Forest":
            return RandomForestWrapper()
        if model_name == "SVM":
            return SVMWrapper()
        if model_name == "Decision Tree":
            return DecisionTreeWrapper()
        raise ValueError(f"不支持的 AutoML 模型: {model_name}")

    def start_automl(self):
        if self._is_busy():
            return
        if self.dataset_bundle is None:
            QMessageBox.warning(self, "提示", "请先加载数据。")
            return

        selected_models = [name for name, checkbox in self.automl_model_checks.items() if checkbox.isChecked()]
        if not selected_models:
            QMessageBox.warning(self, "提示", "请至少选择一个 AutoML 模型。")
            return

        self._clear_automl_results_ui()
        self.export_report_btn.setEnabled(False)
        self.automl_log.clear()
        self.automl_log.append("开始自动优化...")
        self.automl_engine = AutoMLEngine()
        for model_name in selected_models:
            self.automl_engine.register_model(model_name, self._create_automl_wrapper(model_name))

        self.automl_thread = AutoMLRunThread(
            self.automl_engine,
            self.dataset_bundle,
            int(self.stage2_trials_spin.value()),
        )
        self.automl_thread.model_started.connect(
            lambda name: self.automl_log.append(f"\n开始优化 {name}")
        )
        self.automl_thread.model_progress.connect(
            lambda name, current, total: self.automl_log.append(f"{name}: {current}/{total}")
        )
        self.automl_thread.model_completed.connect(self._append_automl_model_summary)
        self.automl_thread.all_completed.connect(self._on_automl_completed)
        self.automl_thread.log_message.connect(self.automl_log.append)
        self.automl_thread.error.connect(
            lambda message: QMessageBox.critical(self, "自动优化失败", message)
        )
        self.automl_thread.finished.connect(self._on_worker_finished)
        self._refresh_action_buttons()
        self.automl_thread.start()

    def start_automl_workflow(self):
        if self._is_busy():
            return

        selected_models = [name for name, checkbox in self.automl_model_checks.items() if checkbox.isChecked()]
        if not selected_models:
            QMessageBox.warning(self, "提示", "请至少选择一个自动试验模型。")
            return

        try:
            dataset_request = self._collect_workflow_request()
            if dataset_request is None:
                return
            candidate_config = self._collect_candidate_config()
        except Exception as exc:
            QMessageBox.critical(self, "自动试验启动失败", str(exc))
            return

        self._clear_automl_results_ui()
        self._clear_artifact_summary()
        self.export_report_btn.setEnabled(False)
        self.automl_log.clear()
        self.automl_log.append("开始两阶段自动试验...")
        self.workflow_stage_label.setText("准备开始第1阶段（粗筛）")
        self.workflow_progress_bar.setValue(0)
        self.workflow_current_stage = None

        self.workflow_orchestrator = WorkflowOrchestrator(self.data_builder, self._create_automl_wrapper)
        runtime_mode = self.runtime_mode_combo.currentText()
        stage1_trials = int(self.stage1_trials_spin.value())
        stage2_trials = int(self.stage2_trials_spin.value())
        stage1_epochs = int(self.stage1_epochs_spin.value())
        stage2_epochs = int(self.stage2_epochs_spin.value())
        self.workflow_orchestrator.STAGE1_TRIALS = stage1_trials
        self.workflow_orchestrator.STAGE2_TRIALS = stage2_trials
        self.workflow_orchestrator.STAGE1_DEEP_EPOCHS = stage1_epochs
        self.workflow_orchestrator.STAGE2_DEEP_EPOCHS = stage2_epochs
        self.workflow_orchestrator.STAGE1_MAX_SCHEMES = self._get_stage1_scheme_cap(runtime_mode)
        workload_snapshot = self._build_workload_snapshot(candidate_config, selected_models, runtime_mode)
        estimated_text = self._format_duration(workload_snapshot["estimated_seconds"])
        self.automl_log.append(
            f"工作量预估: 候选 {workload_snapshot['candidate_total']} 组 | Stage1 {workload_snapshot['stage1_scheme_count']} 组 | Stage2 {workload_snapshot['stage2_scheme_count']} 组 | 总 trial {workload_snapshot['total_trial_count']} | 预计 {estimated_text}"
        )
        self.automl_log.append(
            f"运行模式: {runtime_mode} | 第1阶段试验次数: {stage1_trials} | 第1阶段训练轮次: {stage1_epochs} | 第2阶段试验次数: {stage2_trials} | 第2阶段训练轮次: {stage2_epochs}"
        )
        self.workflow_thread = WorkflowRunThread(
            self.workflow_orchestrator,
            dataset_request,
            selected_models,
            candidate_config,
            runtime_mode,
            self.workflow_output_dir_pick.text().strip() or None,
        )
        self.workflow_thread.stage_changed.connect(self._on_workflow_stage_changed)
        self.workflow_thread.scheme_progress.connect(self._on_workflow_scheme_progress)
        self.workflow_thread.model_progress.connect(self._on_workflow_model_progress)
        self.workflow_thread.log_message.connect(self.automl_log.append)
        self.workflow_thread.completed.connect(self._on_workflow_completed)
        self.workflow_thread.error.connect(lambda message: QMessageBox.critical(self, "自动试验失败", message))
        self.workflow_thread.finished.connect(self._on_worker_finished)
        self._refresh_action_buttons()
        self.workflow_thread.start()

    def _build_batch_workflow_requests(self, repeat_count, base_seed):
        if self.mineral_mode_combo.currentText() != self.MODE_SINGLE:
            raise ValueError("连续划分优化仅在单文件自动分割模式下可用。")

        mineral_path = self.all_mineral_edit.text().strip()
        if not mineral_path:
            raise ValueError("请选择矿点文件。")

        all_minerals = self._normalize_mineral_columns(
            self._load_table_file(mineral_path),
            "矿点文件",
        )
        requests = []
        for index in range(int(repeat_count)):
            split_seed = int(base_seed) + index
            request = self._collect_workflow_request(all_minerals=all_minerals, split_seed=split_seed)
            if request is None:
                return None
            request["split_seed"] = split_seed
            request["split_round"] = index + 1
            requests.append(request)
        return requests

    def start_batch_automl_workflow(self):
        if self._is_busy():
            return

        if self.mineral_mode_combo.currentText() != self.MODE_SINGLE:
            QMessageBox.warning(self, "提示", "连续划分并优化仅在单文件自动分割模式下可用。")
            return

        selected_models = [name for name, checkbox in self.automl_model_checks.items() if checkbox.isChecked()]
        if not selected_models:
            QMessageBox.warning(self, "提示", "请至少选择一个自动试验模型。")
            return

        repeat_count = int(self.split_repeat_spin.value())
        if repeat_count < 2:
            QMessageBox.warning(self, "提示", "连续划分次数至少为 2。")
            return

        try:
            candidate_config = self._collect_candidate_config()
            base_seed = int(self.batch_seed_spin.value())
            run_requests = self._build_batch_workflow_requests(repeat_count, base_seed)
            if not run_requests:
                return
        except Exception as exc:
            QMessageBox.critical(self, "连续划分启动失败", str(exc))
            return

        self._clear_automl_results_ui()
        self._clear_artifact_summary()
        self.export_report_btn.setEnabled(False)
        self.automl_log.clear()
        self.automl_log.append(f"开始连续划分优化... 共 {repeat_count} 轮")
        self.automl_log.append(f"基础随机种子: {base_seed}")
        self.workflow_stage_label.setText("准备开始连续划分优化")
        self.workflow_progress_bar.setValue(0)
        self.workflow_current_stage = None
        self.workflow_batch_active = True
        self.workflow_batch_total_runs = repeat_count
        self.workflow_batch_current_run = 0
        self.workflow_batch_base_seed = base_seed
        self.workflow_batch_summary = None

        self.workflow_orchestrator = WorkflowOrchestrator(self.data_builder, self._create_automl_wrapper)
        runtime_mode = self.runtime_mode_combo.currentText()
        stage1_trials = int(self.stage1_trials_spin.value())
        stage2_trials = int(self.stage2_trials_spin.value())
        stage1_epochs = int(self.stage1_epochs_spin.value())
        stage2_epochs = int(self.stage2_epochs_spin.value())
        self.workflow_orchestrator.STAGE1_TRIALS = stage1_trials
        self.workflow_orchestrator.STAGE2_TRIALS = stage2_trials
        self.workflow_orchestrator.STAGE1_DEEP_EPOCHS = stage1_epochs
        self.workflow_orchestrator.STAGE2_DEEP_EPOCHS = stage2_epochs
        self.workflow_orchestrator.STAGE1_MAX_SCHEMES = self._get_stage1_scheme_cap(runtime_mode)

        workload_snapshot = self._build_workload_snapshot(candidate_config, selected_models, runtime_mode)
        estimated_text = self._format_duration(workload_snapshot["estimated_seconds"])
        self.automl_log.append(
            f"工作量预估: 单轮 trial {workload_snapshot['total_trial_count']} | 连续 {repeat_count} 轮总 trial {workload_snapshot['batch_total_trial_count']} | 预计 {estimated_text}"
        )
        self.automl_log.append(
            f"运行模式: {runtime_mode} | 连续划分次数: {repeat_count} | 第1阶段试验次数: {stage1_trials} | 第1阶段训练轮次: {stage1_epochs} | 第2阶段试验次数: {stage2_trials} | 第2阶段训练轮次: {stage2_epochs}"
        )

        self.workflow_batch_thread = WorkflowBatchRunThread(
            self.workflow_orchestrator,
            run_requests,
            selected_models,
            candidate_config,
            runtime_mode,
            self.workflow_output_dir_pick.text().strip() or None,
            base_seed,
            float(self.test_split_ratio_spin.value()),
        )
        self.workflow_batch_thread.run_started.connect(self._on_batch_run_started)
        self.workflow_batch_thread.run_completed.connect(self._on_batch_run_completed)
        self.workflow_batch_thread.stage_changed.connect(self._on_workflow_stage_changed)
        self.workflow_batch_thread.scheme_progress.connect(self._on_workflow_scheme_progress)
        self.workflow_batch_thread.model_progress.connect(self._on_workflow_model_progress)
        self.workflow_batch_thread.log_message.connect(self.automl_log.append)
        self.workflow_batch_thread.completed.connect(self._on_batch_workflow_completed)
        self.workflow_batch_thread.error.connect(lambda message: QMessageBox.critical(self, "连续划分优化失败", message))
        self.workflow_batch_thread.finished.connect(self._on_worker_finished)
        self._refresh_action_buttons()
        self.workflow_batch_thread.start()

    def _collect_candidate_config(self):
        candidate_config = {
            "patch_size_candidates": self._parse_candidate_values(self.patch_size_candidates_edit.text(), int),
            "patch_stride_candidates": self._parse_candidate_values(self.patch_stride_candidates_edit.text(), int),
            "buffer_radius_candidates": self._parse_candidate_values(self.buffer_radius_candidates_edit.text(), float),
        }
        sampling_text = self.sampling_percentage_candidates_edit.text().strip()
        sampling_percentage_candidates = self._parse_sampling_percentage_candidates(sampling_text)
        if sampling_text and not sampling_percentage_candidates:
            raise ValueError("抽样比例候选必须大于 0。")
        if sampling_percentage_candidates:
            candidate_config["sampling_percentage_candidates"] = sampling_percentage_candidates
        balance_text = self.balance_ratio_candidates_edit.text().strip()
        balance_ratio_candidates = self._parse_optional_float_candidates(balance_text)
        if balance_text and not balance_ratio_candidates:
            raise ValueError("正/负样本比例候选必须大于 0。")
        if balance_ratio_candidates:
            candidate_config["balance_ratio_candidates"] = balance_ratio_candidates
        return candidate_config

    def _collect_workflow_request(self, *, all_minerals=None, split_seed=42):
        dataset_request = self._collect_dataset_request(all_minerals=all_minerals, split_seed=split_seed)
        if dataset_request is None:
            return None
        h5_path, train_minerals, val_minerals, test_minerals, no_ore_minerals, h5_mode, build_config = dataset_request
        return {
            "h5_path": h5_path,
            "train_minerals": train_minerals,
            "val_minerals": val_minerals,
            "test_minerals": test_minerals,
            "no_ore_minerals": no_ore_minerals,
            "h5_mode": h5_mode,
            "build_config": build_config,
            "split_mode": (
                "spatial_region"
                if self.mineral_mode_combo.currentText() == self.MODE_REGION
                else
                "manual"
                if self.mineral_mode_combo.currentText() == self.MODE_MANUAL
                else "single"
                if self.mineral_mode_combo.currentText() == self.MODE_SINGLE
                else "separate"
            ),
        }

    def _parse_candidate_values(self, text, caster):
        raw_items = []
        for chunk in str(text or "").replace("，", ",").replace(";", ",").split(","):
            item = chunk.strip()
            if item:
                raw_items.append(caster(item))
        if not raw_items:
            raise ValueError("候选参数列表不能为空。")
        return sorted(set(raw_items))

    def _on_workflow_stage_changed(self, stage_name):
        label_map = {
            "stage1": "第1阶段（粗筛）：扫描候选数据方案",
            "stage2": "第2阶段（精筛）：正式 AutoML 搜索",
        }
        self.workflow_current_stage = stage_name
        if self.workflow_batch_active and self.workflow_batch_total_runs > 0 and self.workflow_batch_current_run > 0:
            prefix = f"批次 {self.workflow_batch_current_run}/{self.workflow_batch_total_runs} | "
        else:
            prefix = ""
        self.workflow_stage_label.setText(prefix + label_map.get(stage_name, stage_name))
        if self.workflow_batch_active and self.workflow_batch_total_runs > 0 and self.workflow_batch_current_run > 0:
            slot = 100.0 / float(self.workflow_batch_total_runs)
            stage_offset = 0.0 if stage_name == "stage1" else 0.5 if stage_name == "stage2" else 0.0
            progress = ((self.workflow_batch_current_run - 1) + stage_offset) * slot
            self.workflow_progress_bar.setValue(int(progress))
        elif stage_name == "stage1":
            self.workflow_progress_bar.setValue(0)
        elif stage_name == "stage2":
            self.workflow_progress_bar.setValue(50)

    def _on_workflow_scheme_progress(self, current, total, cache_key):
        if total > 0:
            if self.workflow_batch_active and self.workflow_batch_total_runs > 0 and self.workflow_batch_current_run > 0:
                stage_fraction = (current / total) * 0.5
                if self.workflow_current_stage == "stage2":
                    stage_fraction = 0.5 + stage_fraction
                slot = 100.0 / float(self.workflow_batch_total_runs)
                overall = ((self.workflow_batch_current_run - 1) + stage_fraction) * slot
                self.workflow_progress_bar.setValue(int(overall))
            else:
                baseline = 50 if self.workflow_current_stage == "stage2" else 0
                span = 50
                self.workflow_progress_bar.setValue(baseline + int((current / total) * span))
        self.automl_log.append(f"[Scheme {current}/{total}] {cache_key}")

    def _on_workflow_model_progress(self, stage_name, model_name, current, total):
        self.automl_log.append(f"{stage_name} | {model_name}: {current}/{total}")

    def _on_workflow_completed(self, summary):
        self.workflow_summary = summary
        self.workflow_stage_label.setText("自动试验完成")
        self.workflow_progress_bar.setValue(100)
        self.automl_log.append("\n=== 自动试验完成 ===")
        self.automl_log.append(f"输出目录: {summary.get('output_dir', '')}")
        best_result = summary.get("best_result")
        if best_result:
            self.automl_log.append(f"最佳模型: {best_result.get('model_name')} | 综合评分 {best_result.get('composite_score', 0.0):.4f}")
        self._show_workflow_results(summary, update_workflow_summary=False, update_artifact_summary=True)

    def _show_workflow_results(
        self,
        summary,
        *,
        update_workflow_summary=False,
        update_artifact_summary=False,
        activate_tab=True,
    ):
        if update_workflow_summary:
            self.workflow_summary = summary
        self.automl_results = list(summary.get("final_results") or [])
        self._populate_automl_results_table(self.automl_results, activate_tab=activate_tab)
        self.export_report_btn.setEnabled(bool(self.automl_results))
        if self.automl_results:
            self.automl_results_table.selectRow(0)
            self._update_automl_details_panel()
        else:
            self.automl_details_text.clear()
        if update_artifact_summary:
            self._update_artifact_summary(summary)

    def _on_batch_run_started(self, run_index, total_runs, split_seed):
        self.workflow_batch_current_run = int(run_index)
        self.workflow_batch_total_runs = int(total_runs)
        self.workflow_stage_label.setText(f"批次 {run_index}/{total_runs}：准备生成新划分（seed={split_seed}）")
        if total_runs > 0:
            self.workflow_progress_bar.setValue(int(((run_index - 1) / total_runs) * 100))
        self.automl_log.append(f"\n--- 第 {run_index}/{total_runs} 轮 | split_seed={split_seed} ---")

    def _on_batch_run_completed(self, run_index, summary):
        best_result = summary.get("best_result") or {}
        if best_result:
            self.automl_log.append(
                f"第 {run_index} 轮完成: {best_result.get('model_name', '-')} | 综合评分 {best_result.get('composite_score', 0.0):.4f}"
            )
        else:
            self.automl_log.append(f"第 {run_index} 轮完成")
        if self.workflow_batch_total_runs > 0:
            self.workflow_progress_bar.setValue(int((run_index / self.workflow_batch_total_runs) * 100))

    def _format_batch_summary_text(self, batch_summary):
        best_run = batch_summary.get("best_run") or {}
        best_result = batch_summary.get("best_result") or {}
        stable_model = batch_summary.get("best_model") or {}
        lines = [
            "连续划分总结",
            f"- 连续划分次数: {batch_summary.get('split_count', 0)}",
            f"- 基础随机种子: {batch_summary.get('base_seed', '-')}",
            f"- 运行模式: {batch_summary.get('runtime_mode', '-')}",
            f"- 负样本策略: {self._format_negative_sampling_mode_display(batch_summary)}",
            f"- 输出目录: {batch_summary.get('output_dir', '')}",
            f"- 最佳划分轮次: 第 {batch_summary.get('best_run_index', '-')} 轮",
            f"- 最佳划分模型: {best_result.get('model_name', '-')}",
            f"- 最佳划分综合分: {self._format_optional_metric(best_result.get('composite_score'))}",
            f"- 最稳模型: {stable_model.get('model_name', '-')}",
            f"- 最稳分数: {self._format_optional_metric(stable_model.get('stability_score'))}",
            f"- 最稳模型 Top1 次数: {self._format_optional_metric(stable_model.get('top1_count'))}",
            f"- 最佳划分输出: {best_run.get('output_dir', '')}",
        ]
        if batch_summary.get("batch_report_path"):
            lines.append(f"- 批次报告: {batch_summary.get('batch_report_path')}")
        if batch_summary.get("batch_summary_xlsx_path"):
            lines.append(f"- Excel 汇总: {batch_summary.get('batch_summary_xlsx_path')}")
        if batch_summary.get("batch_excel_error"):
            lines.append(f"- Excel 生成提示: {batch_summary.get('batch_excel_error')}")
        lines.append("- 提示: 点击“每轮结果”表中的某一行，可在下方主结果区查看该轮的完整模型明细。")
        return "\n".join(lines)

    def _populate_batch_summary_tables(self, batch_summary):
        run_rows = list(batch_summary.get("run_rows") or [])
        run_columns = [
            "轮次",
            "随机种子",
            "最佳标记",
            "最佳模型",
            "最佳综合分",
            "验证准确率",
            "测试检出率",
            "训练矿点数",
            "测试矿点数",
            "输出目录",
        ]
        self.batch_runs_table.setRowCount(len(run_rows))
        self.batch_runs_table.setColumnCount(len(run_columns))
        self.batch_runs_table.setHorizontalHeaderLabels(run_columns)
        for row_index, row in enumerate(run_rows):
            values = [
                row.get("run_index", ""),
                row.get("split_seed", ""),
                "是" if row.get("is_best_run") else "",
                row.get("best_model_name", ""),
                self._format_optional_metric(row.get("best_composite_score")),
                self._format_optional_metric(row.get("best_val_accuracy")),
                self._format_optional_metric(row.get("best_test_mineral_detection_rate")),
                row.get("train_mineral_count", ""),
                row.get("test_mineral_count", ""),
                row.get("output_dir", ""),
            ]
            for column_index, value in enumerate(values):
                self.batch_runs_table.setItem(row_index, column_index, QTableWidgetItem(str(value)))
        self.batch_runs_table.resizeColumnsToContents()

        model_rows = list(batch_summary.get("model_rows") or [])
        model_columns = [
            "模型",
            "出现次数",
            "Top1次数",
            "平均综合分",
            "标准差",
            "平均验证准确率",
            "平均测试检出率",
            "稳定分",
        ]
        self.batch_models_table.setRowCount(len(model_rows))
        self.batch_models_table.setColumnCount(len(model_columns))
        self.batch_models_table.setHorizontalHeaderLabels(model_columns)
        for row_index, row in enumerate(model_rows):
            values = [
                row.get("model_name", ""),
                row.get("run_count", ""),
                row.get("top1_count", ""),
                self._format_optional_metric(row.get("mean_composite_score")),
                self._format_optional_metric(row.get("std_composite_score")),
                self._format_optional_metric(row.get("mean_val_accuracy")),
                self._format_optional_metric(row.get("mean_test_mineral_detection_rate")),
                self._format_optional_metric(row.get("stability_score")),
            ]
            for column_index, value in enumerate(values):
                self.batch_models_table.setItem(row_index, column_index, QTableWidgetItem(str(value)))
        self.batch_models_table.resizeColumnsToContents()

        self.batch_summary_text.setPlainText(self._format_batch_summary_text(batch_summary))

    def _on_batch_workflow_completed(self, batch_summary):
        self.workflow_batch_summary = batch_summary
        self.workflow_batch_active = False
        self.workflow_stage_label.setText("连续划分优化完成")
        self.workflow_progress_bar.setValue(100)
        self.automl_log.append("\n=== 连续划分优化完成 ===")
        self.automl_log.append(f"批次输出目录: {batch_summary.get('output_dir', '')}")
        if batch_summary.get("batch_report_path"):
            self.automl_log.append(f"批次报告: {batch_summary.get('batch_report_path')}")
        if batch_summary.get("batch_summary_xlsx_path"):
            self.automl_log.append(f"Excel汇总: {batch_summary.get('batch_summary_xlsx_path')}")
        if batch_summary.get("batch_excel_error"):
            self.automl_log.append(f"Excel生成提示: {batch_summary.get('batch_excel_error')}")
        best_result = batch_summary.get("best_result") or {}
        if best_result:
            self.automl_log.append(
                f"最佳划分模型: {best_result.get('model_name', '-')} | 综合评分 {best_result.get('composite_score', 0.0):.4f}"
            )
        stable_model = batch_summary.get("best_model") or {}
        if stable_model:
            self.automl_log.append(
                f"最稳模型: {stable_model.get('model_name', '-')} | 稳定分 {self._format_optional_metric(stable_model.get('stability_score'))}"
            )
        self._populate_batch_summary_tables(batch_summary)
        self.workflow_summary = batch_summary
        best_run = batch_summary.get("best_run") or {}
        best_run_index = int(batch_summary.get("best_run_index") or 0)
        if best_run_index > 0 and hasattr(self, "batch_runs_table"):
            self.batch_runs_table.selectRow(best_run_index - 1)
        self._show_workflow_results(best_run, update_workflow_summary=False, update_artifact_summary=False, activate_tab=False)
        if self.automl_results:
            self.automl_results_table.selectRow(0)
            self._update_automl_details_panel()
        if hasattr(self, "automl_output_tabs") and hasattr(self, "batch_summary_tab"):
            self.automl_output_tabs.setCurrentWidget(self.batch_summary_tab)
        self._update_artifact_summary(batch_summary)
        self.export_report_btn.setEnabled(bool(batch_summary.get("report_path")))

    def _update_batch_run_selection(self):
        if not self.workflow_batch_summary:
            return
        selected_rows = self.batch_runs_table.selectionModel().selectedRows() if self.batch_runs_table.selectionModel() else []
        if not selected_rows:
            return
        row_index = selected_rows[0].row()
        run_rows = list(self.workflow_batch_summary.get("run_rows") or [])
        run_summaries = list(self.workflow_batch_summary.get("run_summaries") or [])
        if row_index < 0 or row_index >= len(run_summaries):
            return
        run_summary = run_summaries[row_index]
        self.automl_results = list(run_summary.get("final_results") or [])
        self._populate_automl_results_table(self.automl_results, activate_tab=False)
        if self.automl_results:
            self.automl_results_table.selectRow(0)
            self._update_automl_details_panel()
        else:
            self.automl_details_text.clear()

    def _append_automl_model_summary(self, name, result):
        lines = [
            f"{name} 完成",
            f"  综合评分: {result.get('composite_score', 0.0):.4f}",
            f"  验证准确率: {result.get('val_accuracy', 0.0):.4f}",
            f"  内部验证矿点检出率: {self._format_optional_metric(result.get('val_mineral_detection_rate'))}",
            f"  测试矿点检出率: {self._format_optional_metric(result.get('test_mineral_detection_rate'))}",
            f"  最优 Trial 分数: {result.get('best_score', 0.0):.4f}",
            "  最优参数:",
            format_params(result.get("best_params")),
        ]
        self.automl_log.append("\n".join(lines))

    def _on_automl_completed(self, results):
        self.automl_results = results
        self.automl_log.append("\n=== 自动优化完成 ===")
        for item in results:
            self.automl_log.append(
                f"{item['model_name']}: 综合评分 {item.get('composite_score', 0.0):.4f}"
            )
        self._populate_automl_results_table(results)
        self.export_report_btn.setEnabled(bool(results))
        if results:
            self.automl_results_table.selectRow(0)
            self._update_automl_details_panel()

    def _populate_automl_results_table(self, results, activate_tab=True):
        columns = [
            "模型",
            "综合评分",
            "验证准确率",
            "验证F1",
            "内部验证矿点检出率",
            "测试矿点检出率",
            "最优Trial分数",
            "训练时间(秒)",
        ]
        self.automl_results_table.setRowCount(len(results))
        self.automl_results_table.setColumnCount(len(columns))
        self.automl_results_table.setHorizontalHeaderLabels(columns)
        for row_index, item in enumerate(results):
            values = [
                item.get("model_name", ""),
                f"{item.get('composite_score', 0.0):.4f}",
                f"{item.get('val_accuracy', 0.0):.4f}",
                f"{item.get('val_f1', 0.0):.4f}",
                self._format_optional_metric(item.get("val_mineral_detection_rate")),
                self._format_optional_metric(item.get("test_mineral_detection_rate")),
                f"{item.get('best_score', 0.0):.4f}",
                f"{item.get('training_time', 0.0):.1f}",
            ]
            for column_index, value in enumerate(values):
                self.automl_results_table.setItem(row_index, column_index, QTableWidgetItem(str(value)))
        self.automl_results_table.resizeColumnsToContents()

    def _update_automl_details_panel(self):
        selected_rows = self.automl_results_table.selectionModel().selectedRows()
        if not selected_rows or not self.automl_results:
            self.automl_details_text.clear()
            return

        row_index = selected_rows[0].row()
        if row_index < 0 or row_index >= len(self.automl_results):
            self.automl_details_text.clear()
            return

        result = self.automl_results[row_index]
        detail_lines = [
            f"模型: {result.get('model_name', '-')}",
            f"综合评分: {result.get('composite_score', 0.0):.4f}",
            f"验证准确率: {result.get('val_accuracy', 0.0):.4f}",
            f"验证F1: {result.get('val_f1', 0.0):.4f}",
            f"内部验证矿点检出率: {self._format_optional_metric(result.get('val_mineral_detection_rate'))}",
            f"测试准确率: {self._format_optional_metric(result.get('test_accuracy'))}",
            f"测试F1: {self._format_optional_metric(result.get('test_f1'))}",
            f"测试矿点检出率: {self._format_optional_metric(result.get('test_mineral_detection_rate'))}",
            f"最优 Trial 分数: {result.get('best_score', 0.0):.4f}",
            "",
            "最优超参数:",
            format_params(result.get("best_params")),
            "",
            "搜索边界命中:",
            "\n".join(result.get("search_boundary_hits", [])) or "无",
            "",
            "改进建议:",
            "\n".join(result.get("improvement_advice", [])) or "暂无",
        ]
        self.automl_details_text.setPlainText("\n".join(detail_lines))

    def _clear_automl_results_ui(self):
        self.automl_results = []
        self.automl_results_table.clear()
        self.automl_results_table.setRowCount(0)
        self.automl_results_table.setColumnCount(0)
        self.automl_details_text.clear()

    def export_automl_report(self):
        if not self.automl_results:
            QMessageBox.warning(self, "提示", "没有可导出的 AutoML 结果。")
            return

        output_path, _ = QFileDialog.getSaveFileName(self, "保存报告", "", "PDF Files (*.pdf)")
        if not output_path:
            return

        try:
            from .report_generator import ReportGenerator

            ReportGenerator(self.automl_results).generate_pdf_report(output_path)
            QMessageBox.information(self, "成功", f"报告已保存到: {output_path}")
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"导出失败: {exc}")

    def _parse_optional_candidate_values(self, text, caster):
        raw_items = []
        for chunk in str(text or "").replace("，", ",").replace(";", ",").split(","):
            item = chunk.strip()
            if item:
                raw_items.append(caster(item))
        return sorted(set(raw_items))

    def _parse_optional_float_candidates(self, text):
        return [float(value) for value in self._parse_optional_candidate_values(text, float) if float(value) > 0]

    def _parse_sampling_percentage_candidates(self, text):
        values = self._parse_optional_candidate_values(text, float)
        normalized = []
        for value in values:
            numeric = float(value)
            if numeric <= 0:
                continue
            fraction = numeric / 100.0 if numeric > 1.0 else numeric
            normalized.append(min(fraction, 1.0))
        return sorted(set(normalized))

    def _update_runtime_mode_hint(self, mode):
        mode_messages = {
            "快速筛查": "快速筛查：候选数据方案数量更少，优先保证速度，适合先看结果是否可用。",
            "实用批量": "实用批量：候选数据方案数量适中，速度和效果更均衡，适合日常批量运行。",
            "深度搜索": "深度搜索：候选数据方案数量更多，搜索更充分，适合最终定稿，但耗时会明显增加。",
        }
        base_note = "说明：运行模式只影响候选数据方案的数量上限；第1阶段 / 第2阶段的试验次数和训练轮次都由下方输入框分别控制。"
        self.runtime_mode_hint_label.setText(f"{mode_messages.get(mode, '运行模式说明：请选择一个模式。')}\n{base_note}")

    def _get_selected_automl_models(self):
        return [name for name, checkbox in self.automl_model_checks.items() if checkbox.isChecked()]

    def _get_stage1_scheme_cap(self, runtime_mode):
        return int(self.RUNTIME_MODE_SCHEME_CAPS.get(runtime_mode, 16))

    def _estimate_model_trial_seconds(self, model_name, stage_epochs, dataset_summary):
        train_samples = int(dataset_summary.get("train_sample_count") or dataset_summary.get("dev_pool_sample_count") or 0)
        image_size = max(int(dataset_summary.get("image_size") or 64), 1)
        input_channels = max(int(dataset_summary.get("input_channels") or 3), 1)
        has_cuda = torch.cuda.is_available()

        if model_name in self.MODEL_FACTORIES:
            batches_per_epoch = max(1, math.ceil(max(train_samples, 1) / max(self.AUTO_ML_BATCH_SIZE_HINT, 1)))
            complexity = float(self.DEEP_MODEL_COMPLEXITY.get(model_name, 1.0))
            device_factor = 0.55 if has_cuda else 1.0
            size_factor = (image_size / 64.0) ** 1.1
            channel_factor = (input_channels / 3.0) ** 0.2
            return max(
                1.0,
                batches_per_epoch
                * max(int(stage_epochs), 1)
                * 0.012
                * complexity
                * size_factor
                * channel_factor
                * device_factor,
            )

        train_scale = max(train_samples, 1) / 1000.0
        model_factor = float(self.CLASSIC_MODEL_BASE_SECONDS.get(model_name, 1.0))
        return max(0.5, train_scale * model_factor)

    def _estimate_scheme_round_seconds(self, selected_models, stage_epochs, dataset_summary):
        return sum(
            self._estimate_model_trial_seconds(model_name, stage_epochs, dataset_summary)
            for model_name in selected_models
        )

    def _build_workload_snapshot(self, candidate_config, selected_models, runtime_mode):
        candidate_items = [
            ("窗口大小候选", candidate_config.get("patch_size_candidates") or []),
            ("步长候选", candidate_config.get("patch_stride_candidates") or []),
            ("缓冲半径候选", candidate_config.get("buffer_radius_candidates") or []),
            ("总样本抽样比例候选", candidate_config.get("sampling_percentage_candidates") or [1.0]),
            ("正/负样本比例候选", candidate_config.get("balance_ratio_candidates") or [None]),
        ]
        candidate_details = [(name, max(len(values), 1)) for name, values in candidate_items]
        candidate_total = math.prod(count for _, count in candidate_details) if candidate_details else 0

        stage1_cap = self._get_stage1_scheme_cap(runtime_mode)
        stage1_scheme_count = min(candidate_total, stage1_cap)
        stage2_scheme_count = min(stage1_scheme_count, WorkflowOrchestrator.STAGE1_TOP_K)
        selected_models = list(selected_models or [])
        deep_model_count = sum(1 for model_name in selected_models if model_name in self.MODEL_FACTORIES)
        selected_model_count = len(selected_models)
        stage1_epochs_setting = int(self.stage1_epochs_spin.value())
        stage2_epochs_setting = int(self.stage2_epochs_spin.value())

        stage1_trials_setting = int(self.stage1_trials_spin.value())
        stage2_trials_setting = int(self.stage2_trials_spin.value())
        stage1_trial_count = stage1_scheme_count * selected_model_count * stage1_trials_setting
        stage2_trial_count = stage2_scheme_count * selected_model_count * stage2_trials_setting
        total_trial_count = stage1_trial_count + stage2_trial_count
        repeat_count = int(self.split_repeat_spin.value()) if hasattr(self, "split_repeat_spin") and self.mineral_mode_combo.currentText() == self.MODE_SINGLE else 1
        batch_base_seed = (
            int(self.batch_seed_spin.value())
            if repeat_count > 1 and hasattr(self, "batch_seed_spin") and self.mineral_mode_combo.currentText() == self.MODE_SINGLE
            else None
        )
        batch_stage1_trial_count = stage1_trial_count * repeat_count
        batch_stage2_trial_count = stage2_trial_count * repeat_count
        batch_total_trial_count = total_trial_count * repeat_count

        dataset_summary = dict(self.dataset_bundle.dataset_summary or {}) if self.dataset_bundle is not None else {}
        stage1_seconds = None
        stage2_seconds = None
        estimated_seconds = None
        if dataset_summary and selected_model_count > 0 and total_trial_count > 0:
            stage1_scheme_seconds = self._estimate_scheme_round_seconds(
                selected_models,
                stage1_epochs_setting,
                dataset_summary,
            )
            stage2_scheme_seconds = self._estimate_scheme_round_seconds(
                selected_models,
                stage2_epochs_setting,
                dataset_summary,
            )
            stage1_seconds = stage1_scheme_seconds * stage1_scheme_count * stage1_trials_setting
            stage2_seconds = stage2_scheme_seconds * stage2_scheme_count * stage2_trials_setting
            estimated_seconds = stage1_seconds + stage2_seconds
            if repeat_count > 1:
                stage1_seconds *= repeat_count
                stage2_seconds *= repeat_count
                estimated_seconds *= repeat_count

        recommendations = []
        if selected_model_count == 0:
            recommendations.append("请至少勾选一个模型。")
        else:
            recommendations.append(
                f"第1阶段试验次数每减少 1 次，可少跑 {stage1_scheme_count * selected_model_count} 个 trial。"
            )
            recommendations.append(
                f"第2阶段试验次数每减少 1 次，可少跑 {stage2_scheme_count * selected_model_count} 个 trial。"
            )
            recommendations.append(
                f"少勾选 1 个模型，可少跑 {stage1_scheme_count * stage1_trials_setting + stage2_scheme_count * stage2_trials_setting} 个 trial。"
            )
            if deep_model_count > 0:
                recommendations.append(
                    f"第1/2阶段训练轮次只影响 {deep_model_count} 个深度模型，CNN / ResNet 越多，减轮次越能明显省时。"
                )
            if candidate_total > stage1_cap:
                sorted_candidates = sorted(
                    ((name, count) for name, count in candidate_details if count > 1),
                    key=lambda item: item[1],
                    reverse=True,
                )
                if sorted_candidates:
                    readable = "、".join(f"{name}({count})" for name, count in sorted_candidates[:3])
                    recommendations.append(
                        f"当前候选组合数 {candidate_total} 已超过 Stage 1 上限 {stage1_cap}；若只想缩小搜索面，优先收窄 {readable}，但实际 trial 数不会立刻下降。"
                    )
                else:
                    recommendations.append(
                        f"当前候选组合数 {candidate_total} 已超过 Stage 1 上限 {stage1_cap}，继续缩小候选列表不会立刻减少 trial；先减试验次数或轮次更有效。"
                    )
            else:
                sorted_candidates = sorted(
                    ((name, count) for name, count in candidate_details if count > 1),
                    key=lambda item: item[1],
                    reverse=True,
                )
                if sorted_candidates:
                    readable = "、".join(f"{name}({count})" for name, count in sorted_candidates[:3])
                    recommendations.append(f"如果还想缩小搜索范围，优先收窄: {readable}。")
            if repeat_count > 1:
                recommendations.append(f"当前连续划分次数为 {repeat_count}，总 trial 和耗时会按轮次近似放大 {repeat_count} 倍。")

        return {
            "runtime_mode": runtime_mode,
            "candidate_details": candidate_details,
            "candidate_total": candidate_total,
            "stage1_cap": stage1_cap,
            "stage1_scheme_count": stage1_scheme_count,
            "stage2_scheme_count": stage2_scheme_count,
            "selected_models": selected_models,
            "selected_model_count": selected_model_count,
            "deep_model_count": deep_model_count,
            "stage1_epochs_setting": stage1_epochs_setting,
            "stage2_epochs_setting": stage2_epochs_setting,
            "stage1_trials_setting": stage1_trials_setting,
            "stage2_trials_setting": stage2_trials_setting,
            "stage1_trial_count": stage1_trial_count,
            "stage2_trial_count": stage2_trial_count,
            "total_trial_count": total_trial_count,
            "repeat_count": repeat_count,
            "batch_base_seed": batch_base_seed,
            "batch_stage1_trial_count": batch_stage1_trial_count,
            "batch_stage2_trial_count": batch_stage2_trial_count,
            "batch_total_trial_count": batch_total_trial_count,
            "stage1_seconds": stage1_seconds,
            "stage2_seconds": stage2_seconds,
            "estimated_seconds": estimated_seconds,
            "dataset_loaded": bool(dataset_summary),
            "device_label": "GPU" if torch.cuda.is_available() else "CPU",
            "recommendations": recommendations,
        }

    def _format_duration(self, seconds):
        if seconds is None:
            return "数据未加载，暂无法估时"

        total_seconds = max(int(round(float(seconds))), 0)
        if total_seconds < 60:
            return f"{total_seconds} 秒"

        minutes, sec = divmod(total_seconds, 60)
        if minutes < 60:
            return f"{minutes} 分 {sec} 秒"

        hours, minutes = divmod(minutes, 60)
        if hours < 24:
            return f"{hours} 小时 {minutes} 分"

        days, hours = divmod(hours, 24)
        return f"{days} 天 {hours} 小时"

    def _build_workload_summary_text(self, snapshot):
        lines = [
            "工作量预估",
            f"- 运行模式: {snapshot['runtime_mode']}",
            f"- 候选方案总数: {snapshot['candidate_total']}",
            f"- Stage 1 实际保留: {snapshot['stage1_scheme_count']} / 上限 {snapshot['stage1_cap']}",
            f"- Stage 2 实际进入: {snapshot['stage2_scheme_count']}",
            f"- 选中模型: {snapshot['selected_model_count']}（深度模型 {snapshot['deep_model_count']}）",
            f"- 连续划分次数: {snapshot.get('repeat_count', 1)}",
            f"- 第1阶段试验次数设置: {snapshot['stage1_trials_setting']}",
            f"- 第1阶段训练轮次设置: {snapshot['stage1_epochs_setting']}",
            f"- 第2阶段试验次数设置: {snapshot['stage2_trials_setting']}",
            f"- 第2阶段训练轮次设置: {snapshot['stage2_epochs_setting']}",
            f"- Stage 1 trial: {snapshot['stage1_trial_count']}",
            f"- Stage 2 trial: {snapshot['stage2_trial_count']}",
            f"- 总 trial: {snapshot['total_trial_count']}",
            f"- 连续总 trial: {snapshot.get('batch_total_trial_count', snapshot['total_trial_count'])}",
        ]
        if snapshot.get("batch_base_seed") is not None:
            lines.append(f"- 基础随机种子: {snapshot['batch_base_seed']}")
        if snapshot["estimated_seconds"] is not None:
            lines.append(f"- 预估耗时: 约 {self._format_duration(snapshot['estimated_seconds'])}（按当前 {snapshot['device_label']} 与已加载数据粗估）")
            if snapshot["stage1_seconds"] is not None and snapshot["stage2_seconds"] is not None:
                lines.append(f"- Stage 1 约: {self._format_duration(snapshot['stage1_seconds'])}")
                lines.append(f"- Stage 2 约: {self._format_duration(snapshot['stage2_seconds'])}")
        else:
            lines.append("- 预估耗时: 请先加载数据，系统会根据当前样本规模给出粗估")
        lines.append("- 优先减的参数:")
        for index, item in enumerate(snapshot["recommendations"], start=1):
            lines.append(f"  {index}. {item}")
        return "\n".join(lines)

    def _update_workload_estimate(self, *_):
        if not hasattr(self, "workload_summary_text"):
            return

        try:
            candidate_config = self._collect_candidate_config()
            selected_models = self._get_selected_automl_models()
            runtime_mode = self.runtime_mode_combo.currentText()
            snapshot = self._build_workload_snapshot(candidate_config, selected_models, runtime_mode)
            self.workload_summary_text.setPlainText(self._build_workload_summary_text(snapshot))
        except Exception as exc:
            self.workload_summary_text.setPlainText(
                "工作量预估\n"
                f"- 当前输入还不完整: {exc}\n"
                "- 请检查候选参数是否为空，或稍后再刷新。"
            )

    def _show_empty_chart(self, message):
        self.chart_figure.clear()
        ax = self.chart_figure.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center")
        self.chart_canvas.draw()

    def _update_chart(self):
        if not self.engine.results:
            self._show_empty_chart("当前没有可显示的结果。")
            return

        self.chart_figure.clear()
        chart_type = self.chart_type_combo.currentText()
        if chart_type == self.CHART_TEST_ACC:
            results = [item for item in self.engine.results if item.get("test_metrics_available")]
            if not results:
                self._show_empty_chart("当前结果没有测试召回率。")
                return
            Visualization(results).plot_metric_comparison(
                "test_recall",
                fig=self.chart_figure,
                ylabel="测试召回率",
                title="测试召回率对比",
                color="steelblue",
            )
        elif chart_type == self.CHART_TRAIN_TIME:
            Visualization(self.engine.results).plot_metric_comparison(
                "training_time_seconds",
                fig=self.chart_figure,
                ylabel="训练时间(秒)",
                title="训练时间对比",
                color="coral",
            )
        elif chart_type == self.CHART_GENERALIZATION:
            results = [item for item in self.engine.results if item.get("results", {}).get("val_recall") is not None]
            if not results:
                self._show_empty_chart("当前结果没有验证召回率。")
                return
            Visualization(results).plot_metric_comparison(
                "val_recall",
                fig=self.chart_figure,
                ylabel="验证召回率",
                title="验证召回率对比",
                color="slategray",
            )
        elif chart_type == self.CHART_VAL_MINERAL:
            results = [item for item in self.engine.results if item.get("val_prediction_area_ratio") is not None]
            if not results:
                self._show_empty_chart("当前结果没有验证集预测面积占比。")
                return
            Visualization(results).plot_metric_comparison(
                "val_prediction_area_ratio",
                fig=self.chart_figure,
                ylabel="验证集预测面积占比",
                title="验证集预测面积占比对比",
                color="seagreen",
            )
        elif chart_type == self.CHART_TEST_MINERAL:
            results = [item for item in self.engine.results if item.get("test_prediction_area_ratio") is not None]
            if not results:
                self._show_empty_chart("当前结果没有测试集预测面积占比。")
                return
            Visualization(results).plot_metric_comparison(
                "test_prediction_area_ratio",
                fig=self.chart_figure,
                ylabel="测试集预测面积占比",
                title="测试集预测面积占比对比",
                color="teal",
            )
        elif chart_type == self.CHART_VAL_EI:
            results = [item for item in self.engine.results if item.get("val_ei") is not None]
            if not results:
                self._show_empty_chart("当前结果没有验证集EI。")
                return
            Visualization(results).plot_metric_comparison(
                "val_ei",
                fig=self.chart_figure,
                ylabel="验证集EI",
                title="验证集EI对比",
                color="darkorange",
            )
        elif chart_type == self.CHART_TEST_EI:
            results = [item for item in self.engine.results if item.get("test_ei") is not None]
            if not results:
                self._show_empty_chart("当前结果没有预测集EI。")
                return
            Visualization(results).plot_metric_comparison(
                "test_ei",
                fig=self.chart_figure,
                ylabel="预测集EI",
                title="预测集EI对比",
                color="mediumpurple",
            )
        elif chart_type == self.CHART_COMPOSITE:
            results = [item for item in self.engine.results if item.get("validation_score") is not None]
            if not results:
                self._show_empty_chart("当前结果没有综合验证得分。")
                return
            Visualization(results).plot_metric_comparison(
                "validation_score",
                fig=self.chart_figure,
                ylabel="综合验证得分",
                title="综合验证得分对比",
                color="darkcyan",
            )
        self.chart_canvas.draw()
    def _show_results(self, results):
        if not results:
            self.log_text.append("没有可展示的对比结果。")
            return

        result_df = ResultAnalyzer(results).generate_comparison_table()
        self.results_table.setRowCount(len(result_df))
        self.results_table.setColumnCount(len(result_df.columns))
        self.results_table.setHorizontalHeaderLabels(result_df.columns.tolist())
        for row_index, (_, row_series) in enumerate(result_df.iterrows()):
            for column_index, value in enumerate(row_series):
                self.results_table.setItem(row_index, column_index, QTableWidgetItem(str(value)))
        self.results_table.resizeColumnsToContents()
        self.tabs.setCurrentIndex(1)
        self._update_chart()
        QMessageBox.information(self, "完成", "模型对比已完成。")

    def export_results(self):
        if not self.engine.results:
            QMessageBox.warning(self, "提示", "没有可导出的结果。")
            return
        output_path, _ = QFileDialog.getSaveFileName(self, "保存结果", "", "CSV Files (*.csv)")
        if not output_path:
            return
        ResultAnalyzer(self.engine.results).export_to_csv(output_path)
        QMessageBox.information(self, "成功", f"结果已导出到: {output_path}")

    def _format_optional_metric(self, value):
        if value is None:
            return ""
        return f"{float(value):.4f}"

    def _format_sampling_percentage_display(self, value):
        if value is None or value == "":
            return ""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if numeric <= 1.0:
            return f"{numeric * 100:.0f}%"
        return f"{numeric:.0f}%"

    def _format_balance_ratio_display(self, value):
        if value is None or value == "":
            return ""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        return f"{numeric:.2f}:1"

    def _format_negative_sampling_mode_display(self, summary):
        summary = summary or {}
        mode = str(summary.get("negative_sampling_mode", "") or "").strip().lower()
        multiplier = summary.get("negative_distance_multiplier")
        distance_radius = summary.get("negative_distance_radius")
        applied = bool(summary.get("negative_sampling_applied", True))
        if mode == "far_distance":
            text = "远区确认负样本"
            if distance_radius not in (None, ""):
                try:
                    text += f"（起始半径约 {float(distance_radius):.1f} 米）"
                except (TypeError, ValueError):
                    pass
            elif multiplier not in (None, ""):
                try:
                    text += f"（起始倍数 {float(multiplier):.2f}）"
                except (TypeError, ValueError):
                    pass
        elif mode == "default":
            text = "默认背景随机负样本"
        elif mode:
            text = mode
        else:
            text = "-"
        if not applied:
            text += "（当前数据源未重建负样本）"
        return text

    def _format_dataset_params_display(self, dataset_params):
        params = dict(dataset_params or {})
        if not params:
            return "暂无"

        def _format_distance(value):
            try:
                return f"{float(value):.1f} 米"
            except (TypeError, ValueError):
                return str(value)

        def _format_value(key, value):
            if key == "negative_sampling_mode":
                return self._format_negative_sampling_mode_display(params)
            if key == "sampling_percentage":
                return self._format_sampling_percentage_display(value)
            if key == "balance_ratio":
                return self._format_balance_ratio_display(value)
            if key in {"no_ore_active", "spatial_region_active", "reflect_padding"}:
                return "是" if bool(value) else "否"
            if key == "selected_channel_names":
                values = list(value or [])
                return "、".join(values[:6]) + ("..." if len(values) > 6 else "")
            if key in {"spatial_region_train_bounds", "spatial_region_test_bounds"}:
                bounds = dict(value or {})
                return (
                    f"x=[{bounds.get('xmin', '-')}, {bounds.get('xmax', '-')}] "
                    f"y=[{bounds.get('ymin', '-')}, {bounds.get('ymax', '-')}]"
                )
            if key == "spatial_region_buffer_distance":
                return _format_distance(value)
            if key in {"patch_size", "patch_stride", "buffer_radius", "n_blocks", "train_mineral_count", "val_mineral_count", "test_mineral_count"}:
                return str(value)
            return str(value)

        buffer_radius = params.get("buffer_radius")
        multiplier = params.get("negative_distance_multiplier")
        derived_radius = None
        if params.get("negative_sampling_mode") == "far_distance":
            try:
                derived_radius = float(buffer_radius) * float(multiplier)
            except (TypeError, ValueError):
                derived_radius = params.get("negative_distance_radius")

        rows = []
        ordered_keys = [
            ("patch_size", "窗口大小"),
            ("patch_stride", "步长"),
            ("reflect_padding", "切窗补边"),
            ("selected_channel_names", "参与图层"),
            ("buffer_radius", "缓冲半径"),
            ("negative_sampling_mode", "负样本策略"),
            ("negative_distance_multiplier", "远区起始倍数"),
            ("sampling_percentage", "抽样比例"),
            ("balance_ratio", "正/负样本比例"),
            ("split_mode", "矿点划分模式"),
            ("spatial_region_active", "坐标区域切分"),
            ("spatial_region_train_bounds", "训练区域范围"),
            ("spatial_region_test_bounds", "预测区域范围"),
            ("spatial_region_buffer_distance", "边界缓冲带"),
            ("spatial_region_train_sample_count", "训练区原始样本"),
            ("spatial_region_test_sample_count", "预测区原始样本"),
            ("spatial_region_gray_sample_count", "灰区剔除样本"),
            ("spatial_region_outside_sample_count", "区域外样本"),
            ("no_ore_active", "无矿钻孔启用"),
            ("no_ore_point_count", "无矿钻孔点数"),
            ("no_ore_sample_count", "无矿锚点覆盖样本"),
            ("no_ore_conflict_count", "无矿冲突覆盖"),
            ("spatial_region_train_no_ore_sample_count", "训练区无矿负样本"),
            ("spatial_region_test_no_ore_sample_count", "预测区无矿负样本"),
            ("train_mineral_count", "训练矿点数"),
            ("val_mineral_count", "验证矿点数"),
            ("test_mineral_count", "测试矿点数"),
            ("cache_key", "缓存键"),
        ]
        for key, label in ordered_keys:
            if key not in params:
                continue
            value = _format_value(key, params.get(key))
            if key == "negative_distance_multiplier" and derived_radius is not None:
                value = f"{float(multiplier):.2f}（约{_format_distance(derived_radius)}）"
            elif key == "split_mode":
                value = {
                    "single": "单文件自动分割",
                    "manual": "手工划分",
                    "separate": "独立文件划分",
                    "spatial_region": "坐标区域切分",
                }.get(str(value).strip().lower(), str(value))
            elif key == "balance_ratio":
                value = value or ""
            rows.append(f"{label}: {value}")

        return "\n".join(rows) if rows else "暂无"

    def _populate_automl_results_table(self, results):
        columns = [
            "模型",
            "综合评分",
            "验证准确率",
            "验证F1",
            "内部验证矿点检出率",
            "测试矿点检出率",
            "最佳Trial分数",
            "训练时间(秒)",
            "窗口大小",
            "步长",
            "切窗补边",
            "缓冲半径",
            "抽样比例",
            "正负样本比例",
            "模型文件",
        ]
        self.automl_results_table.setRowCount(len(results))
        self.automl_results_table.setColumnCount(len(columns))
        self.automl_results_table.setHorizontalHeaderLabels(columns)
        for row_index, item in enumerate(results):
            dataset_params = item.get("dataset_params", {})
            values = [
                item.get("model_name", ""),
                f"{item.get('composite_score', 0.0):.4f}",
                self._format_optional_metric(item.get("val_accuracy")),
                self._format_optional_metric(item.get("val_f1")),
                self._format_optional_metric(item.get("val_mineral_detection_rate")),
                self._format_optional_metric(item.get("test_mineral_detection_rate")),
                self._format_optional_metric(item.get("best_score")),
                self._format_optional_metric(item.get("training_time")),
                str(dataset_params.get("patch_size", "")),
                str(dataset_params.get("patch_stride", "")),
                "是" if dataset_params.get("reflect_padding") else "否",
                str(dataset_params.get("buffer_radius", "")),
                self._format_sampling_percentage_display(dataset_params.get("sampling_percentage")),
                self._format_balance_ratio_display(dataset_params.get("balance_ratio")),
                item.get("model_artifact_path", ""),
            ]
            for column_index, value in enumerate(values):
                self.automl_results_table.setItem(row_index, column_index, QTableWidgetItem(str(value)))
        self.automl_results_table.resizeColumnsToContents()

    def _update_automl_details_panel(self):
        selected_rows = self.automl_results_table.selectionModel().selectedRows()
        if not selected_rows or not self.automl_results:
            self.automl_details_text.clear()
            return

        row_index = selected_rows[0].row()
        if row_index < 0 or row_index >= len(self.automl_results):
            self.automl_details_text.clear()
            return

        result = self.automl_results[row_index]
        dataset_params = result.get("dataset_params", {})
        zone_exports = (result.get("prediction_artifact") or {}).get("zone_exports", [])
        zone_lines = [
            f"threshold={item.get('threshold')} | zone_count={item.get('zone_count')} | area_ratio={float(item.get('area_ratio', 0.0)):.4f}"
            for item in zone_exports
        ]
        detail_lines = [
            f"模型: {result.get('model_name', '-')}",
            f"综合评分: {float(result.get('composite_score', 0.0)):.4f}",
            f"验证准确率: {self._format_optional_metric(result.get('val_accuracy'))}",
            f"验证F1: {self._format_optional_metric(result.get('val_f1'))}",
            f"内部验证矿点检出率: {self._format_optional_metric(result.get('val_mineral_detection_rate'))}",
            f"测试准确率: {self._format_optional_metric(result.get('test_accuracy'))}",
            f"测试F1: {self._format_optional_metric(result.get('test_f1'))}",
            f"测试矿点检出率: {self._format_optional_metric(result.get('test_mineral_detection_rate'))}",
            f"最佳Trial分数: {self._format_optional_metric(result.get('best_score'))}",
            "",
            "数据参数:",
            self._format_dataset_params_display(dataset_params),
            "",
            "最优超参数:",
            format_params(result.get("best_params")),
            "",
            f"模型文件: {result.get('model_artifact_path', '')}",
            f"重建清单: {result.get('rebuild_manifest_path', '')}",
            "",
            "预测区统计:",
            "\n".join(zone_lines) or "暂无",
            "",
            "搜索边界命中:",
            "\n".join(result.get("search_boundary_hits", [])) or "无",
            "",
            "改进建议:",
            "\n".join(result.get("improvement_advice", [])) or "暂无",
        ]
        self.automl_details_text.setPlainText("\n".join(detail_lines))

    def _clear_automl_results_ui(self):
        self.automl_results = []
        self.workflow_summary = None
        self.automl_results_table.clear()
        self.automl_results_table.setRowCount(0)
        self.automl_results_table.setColumnCount(0)
        self.automl_details_text.clear()
        self.workflow_stage_label.setText("尚未开始自动试验")
        self.workflow_progress_bar.setValue(0)

    def _update_artifact_summary(self, summary):
        best_result = summary.get("best_result") or {}
        self.workflow_output_path_edit.setText(summary.get("output_dir", ""))
        self.best_model_path_edit.setText(best_result.get("model_artifact_path", ""))
        self.best_manifest_path_edit.setText(best_result.get("rebuild_manifest_path", ""))
        self.rebuild_manifest_edit.setText(best_result.get("rebuild_manifest_path", ""))
        if summary.get("output_dir") and not self.rebuild_output_dir_edit.text().strip():
            self.rebuild_output_dir_edit.setText(str(summary["output_dir"]))
        detail_lines = [
            f"输出目录: {summary.get('output_dir', '')}",
            f"最佳模型: {best_result.get('model_name', '')}",
            f"综合评分: {self._format_optional_metric(best_result.get('composite_score'))}",
            f"模型文件: {best_result.get('model_artifact_path', '')}",
            f"重建清单: {best_result.get('rebuild_manifest_path', '')}",
            f"报告文件: {summary.get('report_path', '')}",
            f"排行榜: {summary.get('leaderboard_path', '')}",
            f"Trial汇总: {summary.get('all_trials_path', '')}",
        ]
        if summary.get("batch_summary_xlsx_path"):
            detail_lines.append(f"Excel汇总: {summary.get('batch_summary_xlsx_path', '')}")
        if summary.get("batch_excel_error"):
            detail_lines.append(f"Excel生成提示: {summary.get('batch_excel_error', '')}")
        if summary.get("batch_mode"):
            best_model = summary.get("best_model") or {}
            detail_lines.extend(
                [
                    f"连续划分次数: {summary.get('split_count', 0)}",
                    f"基础随机种子: {summary.get('base_seed', '')}",
                    f"最佳划分轮次: 第 {summary.get('best_run_index', '')} 轮",
                    f"最稳模型: {best_model.get('model_name', '')}",
                    f"批次报告: {summary.get('batch_report_path', '')}",
                    f"批次Excel汇总: {summary.get('batch_summary_xlsx_path', '')}",
                    f"批次轮次明细: {summary.get('batch_runs_path', '')}",
                    f"模型稳定性表: {summary.get('batch_models_path', '')}",
                ]
            )
        self.artifact_details_text.setPlainText("\n".join(detail_lines))
        self._refresh_action_buttons()

    def _clear_artifact_summary(self):
        for widget in [
            self.workflow_output_path_edit,
            self.best_model_path_edit,
            self.best_manifest_path_edit,
            self.rebuild_manifest_edit,
        ]:
            widget.clear()
        self.artifact_details_text.clear()
        self._refresh_action_buttons()

    def _use_best_manifest_for_rebuild(self):
        manifest_path = self.best_manifest_path_edit.text().strip()
        if manifest_path:
            self.rebuild_manifest_edit.setText(manifest_path)
            if not self.rebuild_output_dir_edit.text().strip():
                self.rebuild_output_dir_edit.setText(self.workflow_output_path_edit.text().strip())
        self._refresh_action_buttons()

    def rebuild_from_manifest(self):
        if self._is_busy():
            return

        manifest_path = self.rebuild_manifest_edit.text().strip()
        if not manifest_path:
            QMessageBox.warning(self, "提示", "请先选择重建清单。")
            return

        output_dir = self.rebuild_output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "提示", "请先选择重建输出目录。")
            return

        self.rebuild_log_text.clear()
        self.rebuild_log_text.append("开始从清单重建预测区...")
        self.rebuild_thread = ManifestRebuildThread(self.artifact_manager, manifest_path, output_dir)
        self.rebuild_thread.log_message.connect(self.rebuild_log_text.append)
        self.rebuild_thread.completed.connect(self._on_rebuild_completed)
        self.rebuild_thread.error.connect(lambda message: QMessageBox.critical(self, "重建失败", message))
        self.rebuild_thread.finished.connect(self._on_worker_finished)
        self._refresh_action_buttons()
        self.rebuild_thread.start()

    def _on_rebuild_completed(self, result):
        self.rebuild_log_text.append("重建完成。")
        self.rebuild_log_text.append(f"概率图: {result.get('probability_map_path', '')}")
        self.rebuild_log_text.append(f"统计表: {result.get('zone_statistics_path', '')}")
        self.artifact_details_text.setPlainText(
            "\n".join(
                [
                    "最近一次重建结果:",
                    f"输出目录: {result.get('output_dir', '')}",
                    f"概率图: {result.get('probability_map_path', '')}",
                    f"统计表: {result.get('zone_statistics_path', '')}",
                ]
            )
        )
        self._refresh_action_buttons()

    def export_automl_report(self):
        if self.workflow_summary and self.workflow_summary.get("batch_mode"):
            message_lines = []
            if self.workflow_summary.get("batch_report_path"):
                message_lines.append(f"PDF 报告: {self.workflow_summary['batch_report_path']}")
            if self.workflow_summary.get("batch_summary_xlsx_path"):
                message_lines.append(f"Excel 汇总: {self.workflow_summary['batch_summary_xlsx_path']}")
            if self.workflow_summary.get("batch_excel_error"):
                message_lines.append(f"Excel 生成提示: {self.workflow_summary['batch_excel_error']}")
            if message_lines:
                QMessageBox.information(self, "报告位置", "最新批次结果已生成:\n" + "\n".join(message_lines))
                return
        if self.workflow_summary and self.workflow_summary.get("report_path"):
            QMessageBox.information(self, "报告位置", f"最新工作流报告已生成:\n{self.workflow_summary['report_path']}")
            return
        if not self.automl_results:
            QMessageBox.warning(self, "提示", "没有可导出的 AutoML 结果。")
            return
        output_path, _ = QFileDialog.getSaveFileName(self, "保存报告", "", "PDF Files (*.pdf)")
        if not output_path:
            return
        try:
            from .report_generator import ReportGenerator

            ReportGenerator(self.automl_results).generate_pdf_report(output_path)
            QMessageBox.information(self, "成功", f"报告已保存到: {output_path}")
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"导出失败: {exc}")

    def _create_automl_tab(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        content = QWidget(scroll)
        scroll.setWidget(content)

        main_layout = QVBoxLayout(content)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        def add_row(grid, row, label_text, widget, *, label_alignment=Qt.AlignRight | Qt.AlignVCenter, col_span=1):
            label = QLabel(label_text, content)
            label.setAlignment(label_alignment)
            grid.addWidget(label, row, 0)
            grid.addWidget(widget, row, 1, 1, col_span)
            return label

        model_group = QGroupBox("选择模型", content)
        model_layout = QGridLayout(model_group)
        model_layout.setContentsMargins(8, 8, 8, 8)
        model_layout.setHorizontalSpacing(18)
        model_layout.setVerticalSpacing(8)
        self.automl_model_checks = {}
        model_names = list(self.MODEL_FACTORIES.keys()) + self.AUTOML_CLASSIC_MODELS
        for index, model_name in enumerate(model_names):
            checkbox = QCheckBox(model_name, model_group)
            checkbox.setChecked(True)
            self.automl_model_checks[model_name] = checkbox
            model_layout.addWidget(checkbox, index // 2, index % 2)
        main_layout.addWidget(model_group)

        search_group = QGroupBox("搜索空间", content)
        search_layout = QGridLayout(search_group)
        search_layout.setContentsMargins(8, 8, 8, 8)
        search_layout.setHorizontalSpacing(12)
        search_layout.setVerticalSpacing(8)
        self.patch_size_candidates_edit = QLineEdit(search_group)
        self.patch_size_candidates_edit.setText(str(self.patch_size_spin.value()))
        self.patch_size_candidates_edit.setToolTip("输入多个候选窗大小，使用逗号分隔，例如：64, 96, 128")
        self.patch_stride_candidates_edit = QLineEdit(search_group)
        self.patch_stride_candidates_edit.setText(str(self.patch_stride_spin.value()))
        self.patch_stride_candidates_edit.setToolTip("输入多个候选步长，使用逗号分隔，例如：16, 32")
        self.buffer_radius_candidates_edit = QLineEdit(search_group)
        self.buffer_radius_candidates_edit.setText(str(self.buffer_radius_spin.value()))
        self.buffer_radius_candidates_edit.setToolTip("输入多个候选缓冲半径，使用逗号分隔，例如：300, 500, 800")
        self.sampling_percentage_candidates_edit = QLineEdit(search_group)
        self.sampling_percentage_candidates_edit.setText("100")
        self.sampling_percentage_candidates_edit.setToolTip("输入 1-100 的百分比，或 0-1 的小数，例如：25, 50, 100")
        self.balance_ratio_candidates_edit = QLineEdit(search_group)
        self.balance_ratio_candidates_edit.setPlaceholderText("例如: 1.0, 2.0")
        self.balance_ratio_candidates_edit.setToolTip("正负样本比例；留空则使用当前默认负样本采样方式。")

        for row, (label_text, widget) in enumerate(
            [
                ("窗大小候选:", self.patch_size_candidates_edit),
                ("步长候选:", self.patch_stride_candidates_edit),
                ("缓冲半径候选:", self.buffer_radius_candidates_edit),
                ("采样比例候选:", self.sampling_percentage_candidates_edit),
                ("正负样本比候选:", self.balance_ratio_candidates_edit),
            ]
        ):
            add_row(search_layout, row, label_text, widget)
        main_layout.addWidget(search_group)

        trials_group = QGroupBox("优化参数", content)
        trials_layout = QGridLayout(trials_group)
        trials_layout.setContentsMargins(8, 8, 8, 8)
        trials_layout.setHorizontalSpacing(12)
        trials_layout.setVerticalSpacing(8)

        self.stage1_trials_spin = QSpinBox(trials_group)
        self.stage1_trials_spin.setRange(1, 100)
        self.stage1_trials_spin.setValue(12)
        self.stage1_trials_spin.setToolTip("第1阶段试验次数：用于粗筛候选数据方案。")
        self.stage1_epochs_spin = QSpinBox(trials_group)
        self.stage1_epochs_spin.setRange(1, 200)
        self.stage1_epochs_spin.setValue(12)
        self.stage1_epochs_spin.setToolTip("第1阶段训练轮次：决定粗筛阶段单次试验的训练强度。")
        self.stage2_trials_spin = QSpinBox(trials_group)
        self.stage2_trials_spin.setRange(1, 300)
        self.stage2_trials_spin.setValue(50)
        self.stage2_trials_spin.setToolTip("第2阶段试验次数：用于对筛选后的方案进行正式搜索。")
        self.stage2_epochs_spin = QSpinBox(trials_group)
        self.stage2_epochs_spin.setRange(1, 500)
        self.stage2_epochs_spin.setValue(50)
        self.stage2_epochs_spin.setToolTip("第2阶段训练轮次：决定精筛阶段单次试验的训练强度。")

        self.runtime_mode_combo = QComboBox(trials_group)
        self.runtime_mode_combo.addItems(["实用批量", "快速筛查", "深度搜索"])
        self.runtime_mode_combo.setToolTip(
            "快速筛查：减少候选数据方案，优先看结果是否可用。\n"
            "实用批量：默认平衡，适合日常批量运行。\n"
            "深度搜索：扩大候选方案范围，适合最终定稿。"
        )
        self.runtime_mode_combo.currentTextChanged.connect(self._update_runtime_mode_hint)

        self.runtime_mode_hint_label = QLabel(trials_group)
        self.runtime_mode_hint_label.setWordWrap(True)
        self.runtime_mode_hint_label.setStyleSheet("color: #666666;")

        self.workflow_output_dir_pick = QLineEdit(trials_group)
        self.workflow_output_dir_pick.setPlaceholderText("默认输出到 ./outputs/model_comparison/")
        browse_output_btn = QPushButton("输出目录...", trials_group)
        browse_output_btn.clicked.connect(
            lambda: self._browse_directory(self.workflow_output_dir_pick, "选择自动试验输出目录")
        )

        add_row(trials_layout, 0, "Stage 1 trial:", self.stage1_trials_spin)
        add_row(trials_layout, 1, "Stage 1 epoch:", self.stage1_epochs_spin)
        add_row(trials_layout, 2, "Stage 2 trial:", self.stage2_trials_spin)
        add_row(trials_layout, 3, "Stage 2 epoch:", self.stage2_epochs_spin)
        add_row(trials_layout, 4, "运行模式:", self.runtime_mode_combo)

        output_dir_row = QWidget(trials_group)
        output_dir_layout = QHBoxLayout(output_dir_row)
        output_dir_layout.setContentsMargins(0, 0, 0, 0)
        output_dir_layout.addWidget(self.workflow_output_dir_pick)
        output_dir_layout.addWidget(browse_output_btn)
        add_row(trials_layout, 5, "输出目录:", output_dir_row)
        self.split_repeat_row = QWidget(trials_group)
        split_repeat_layout = QHBoxLayout(self.split_repeat_row)
        split_repeat_layout.setContentsMargins(0, 0, 0, 0)
        split_repeat_layout.addWidget(QLabel("连续划分次数:", self.split_repeat_row))
        self.split_repeat_spin = QSpinBox(self.split_repeat_row)
        self.split_repeat_spin.setRange(2, 50)
        self.split_repeat_spin.setValue(5)
        self.split_repeat_spin.setToolTip("仅在单文件自动分割时有效。一次点击后连续重新划分并完整优化的次数。")
        split_repeat_layout.addWidget(self.split_repeat_spin)
        split_repeat_layout.addStretch(1)
        trials_layout.addWidget(self.split_repeat_row, 6, 0, 1, 2)
        self.batch_seed_row = QWidget(trials_group)
        batch_seed_layout = QHBoxLayout(self.batch_seed_row)
        batch_seed_layout.setContentsMargins(0, 0, 0, 0)
        batch_seed_layout.addWidget(QLabel("基础随机种子:", self.batch_seed_row))
        self.batch_seed_spin = QSpinBox(self.batch_seed_row)
        self.batch_seed_spin.setRange(0, 2147483647)
        self.batch_seed_spin.setValue(42)
        self.batch_seed_spin.setToolTip("仅在单文件自动分割批次模式下有效。相同种子会复现相同的连续划分顺序。")
        batch_seed_layout.addWidget(self.batch_seed_spin)
        batch_seed_layout.addStretch(1)
        trials_layout.addWidget(self.batch_seed_row, 7, 0, 1, 2)
        trials_layout.addWidget(self.runtime_mode_hint_label, 8, 0, 1, 2)
        self._update_runtime_mode_hint(self.runtime_mode_combo.currentText())
        main_layout.addWidget(trials_group)

        workload_group = QGroupBox("工作量预估", content)
        workload_layout = QVBoxLayout(workload_group)
        workload_layout.setContentsMargins(8, 8, 8, 8)
        self.workload_summary_text = QTextEdit(workload_group)
        self.workload_summary_text.setReadOnly(True)
        self.workload_summary_text.setMinimumHeight(130)
        self.workload_summary_text.setMaximumHeight(160)
        self.workload_summary_text.setPlainText(
            "工作量预估：请先选择模型并输入搜索空间。数据加载后会自动补全实际 trial 数和耗时估计。"
        )
        workload_layout.addWidget(self.workload_summary_text)
        main_layout.addWidget(workload_group)

        progress_group = QGroupBox("工作流进度", content)
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setContentsMargins(8, 8, 8, 8)
        self.workflow_stage_label = QLabel("尚未开始自动试验", progress_group)
        self.workflow_stage_label.setWordWrap(True)
        self.workflow_progress_bar = QProgressBar(progress_group)
        progress_layout.addWidget(self.workflow_stage_label)
        progress_layout.addWidget(self.workflow_progress_bar)
        main_layout.addWidget(progress_group)

        action_row = QWidget(content)
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        self.automl_start_btn = QPushButton("开始自动优化", action_row)
        self.automl_start_btn.clicked.connect(self.start_automl_workflow)
        self.batch_split_btn = QPushButton("连续划分并优化", action_row)
        self.batch_split_btn.clicked.connect(self.start_batch_automl_workflow)
        self.export_report_btn = QPushButton("导出 PDF 报告", action_row)
        self.export_report_btn.clicked.connect(self.export_automl_report)
        action_layout.addWidget(self.automl_start_btn, 1)
        action_layout.addWidget(self.batch_split_btn, 1)
        action_layout.addWidget(self.export_report_btn, 1)
        main_layout.addWidget(action_row)

        self.automl_output_tabs = QTabWidget(content)

        results_tab = QWidget(self.automl_output_tabs)
        results_layout = QVBoxLayout(results_tab)
        results_layout.setContentsMargins(8, 8, 8, 8)
        self.automl_results_table = QTableWidget(results_tab)
        self.automl_results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.automl_results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.automl_results_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.automl_results_table.setAlternatingRowColors(True)
        results_layout.addWidget(self.automl_results_table)
        self.automl_output_tabs.addTab(results_tab, "结果表")

        self.batch_summary_tab = QWidget(self.automl_output_tabs)
        batch_layout = QVBoxLayout(self.batch_summary_tab)
        batch_layout.setContentsMargins(8, 8, 8, 8)
        self.batch_summary_text = QTextEdit(self.batch_summary_tab)
        self.batch_summary_text.setReadOnly(True)
        self.batch_summary_text.setMaximumHeight(110)
        self.batch_summary_text.setPlaceholderText("连续划分优化完成后，这里会显示全局汇总和最稳定模型。")
        batch_layout.addWidget(self.batch_summary_text)

        batch_runs_group = QGroupBox("每轮结果", self.batch_summary_tab)
        batch_runs_layout = QVBoxLayout(batch_runs_group)
        batch_runs_layout.setContentsMargins(8, 8, 8, 8)
        self.batch_runs_table = QTableWidget(batch_runs_group)
        self.batch_runs_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.batch_runs_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.batch_runs_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.batch_runs_table.setAlternatingRowColors(True)
        self.batch_runs_table.itemSelectionChanged.connect(self._update_batch_run_selection)
        batch_runs_layout.addWidget(self.batch_runs_table)
        batch_layout.addWidget(batch_runs_group, 1)

        batch_models_group = QGroupBox("模型稳定性", self.batch_summary_tab)
        batch_models_layout = QVBoxLayout(batch_models_group)
        batch_models_layout.setContentsMargins(8, 8, 8, 8)
        self.batch_models_table = QTableWidget(batch_models_group)
        self.batch_models_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.batch_models_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.batch_models_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.batch_models_table.setAlternatingRowColors(True)
        batch_models_layout.addWidget(self.batch_models_table)
        batch_layout.addWidget(batch_models_group, 1)

        self.automl_output_tabs.addTab(self.batch_summary_tab, "连续划分")

        details_tab = QWidget(self.automl_output_tabs)
        details_layout = QVBoxLayout(details_tab)
        details_layout.setContentsMargins(8, 8, 8, 8)
        self.automl_details_text = QTextEdit(details_tab)
        self.automl_details_text.setReadOnly(True)
        self.automl_details_text.setPlaceholderText("从结果表中选择一行后，这里会显示完整详情。")
        details_layout.addWidget(self.automl_details_text)
        self.automl_output_tabs.addTab(details_tab, "结果详情")

        log_tab = QWidget(self.automl_output_tabs)
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(8, 8, 8, 8)
        self.automl_log = QTextEdit(log_tab)
        self.automl_log.setReadOnly(True)
        self.automl_log.setPlaceholderText("自动优化日志会按时间顺序显示在这里。")
        log_layout.addWidget(self.automl_log)
        self.automl_output_tabs.addTab(log_tab, "优化日志")

        self.automl_output_tabs.setMinimumHeight(300)
        main_layout.addWidget(self.automl_output_tabs, 1)

        return scroll

    def _build_workload_summary_text(self, snapshot):
        lines = [
            "工作量预估",
            f"- 运行模式: {snapshot['runtime_mode']}",
            f"- 候选方案总数: {snapshot['candidate_total']}",
            f"- Stage 1 保留: {snapshot['stage1_scheme_count']} / 上限 {snapshot['stage1_cap']}",
            f"- Stage 2 参与评估: {snapshot['stage2_scheme_count']}",
            f"- 已选模型: {snapshot['selected_model_count']}（深度模型 {snapshot['deep_model_count']}）",
            f"- 总 trial 数: {snapshot['total_trial_count']}",
        ]
        if snapshot["estimated_seconds"] is not None:
            lines.append(
                f"- 预计耗时: 约 {self._format_duration(snapshot['estimated_seconds'])} "
                f"（按当前 {snapshot['device_label']} 与已加载数据粗估）"
            )
        else:
            lines.append("- 预计耗时: 请先加载数据后自动补全")
        lines.append("- 提示: 具体 trial/epoch 设置已集中在上方“优化参数”区域。")
        if snapshot["recommendations"]:
            lines.append("- 优先收缩的参数:")
            for index, item in enumerate(snapshot["recommendations"], start=1):
                lines.append(f"  {index}. {item}")
        return "\n".join(lines)

    def _populate_automl_results_table(self, results, activate_tab=True):
        columns = [
            "模型",
            "综合评分",
            "验证准确率",
            "验证F1",
            "内部验证矿点检出率",
            "测试矿点检出率",
            "最佳Trial分数",
            "训练时间(秒)",
            "窗口大小",
            "步长",
            "缓冲半径",
            "抽样比例",
            "正负样本比例",
            "模型文件",
        ]
        self.automl_results_table.setRowCount(len(results))
        self.automl_results_table.setColumnCount(len(columns))
        self.automl_results_table.setHorizontalHeaderLabels(columns)
        for row_index, item in enumerate(results):
            dataset_params = item.get("dataset_params", {})
            values = [
                item.get("model_name", ""),
                f"{item.get('composite_score', 0.0):.4f}",
                self._format_optional_metric(item.get("val_accuracy")),
                self._format_optional_metric(item.get("val_f1")),
                self._format_optional_metric(item.get("val_mineral_detection_rate")),
                self._format_optional_metric(item.get("test_mineral_detection_rate")),
                self._format_optional_metric(item.get("best_score")),
                self._format_optional_metric(item.get("training_time")),
                str(dataset_params.get("patch_size", "")),
                str(dataset_params.get("patch_stride", "")),
                str(dataset_params.get("buffer_radius", "")),
                self._format_sampling_percentage_display(dataset_params.get("sampling_percentage")),
                self._format_balance_ratio_display(dataset_params.get("balance_ratio")),
                item.get("model_artifact_path", ""),
            ]
            for column_index, value in enumerate(values):
                self.automl_results_table.setItem(row_index, column_index, QTableWidgetItem(str(value)))
        self.automl_results_table.resizeColumnsToContents()
        if activate_tab and hasattr(self, "automl_output_tabs"):
            self.automl_output_tabs.setCurrentIndex(0)

    def _clear_automl_results_ui(self):
        self.automl_results = []
        self.workflow_summary = None
        self.workflow_batch_summary = None
        self.workflow_batch_active = False
        self.workflow_batch_total_runs = 0
        self.workflow_batch_current_run = 0
        self.workflow_batch_base_seed = None
        self.automl_results_table.clear()
        self.automl_results_table.setRowCount(0)
        self.automl_results_table.setColumnCount(0)
        self.automl_details_text.clear()
        self.workflow_stage_label.setText("尚未开始自动试验")
        self.workflow_progress_bar.setValue(0)
        if hasattr(self, "batch_summary_text"):
            self.batch_summary_text.clear()
        if hasattr(self, "batch_runs_table"):
            self.batch_runs_table.clear()
            self.batch_runs_table.setRowCount(0)
            self.batch_runs_table.setColumnCount(0)
        if hasattr(self, "batch_models_table"):
            self.batch_models_table.clear()
            self.batch_models_table.setRowCount(0)
            self.batch_models_table.setColumnCount(0)
        if hasattr(self, "automl_output_tabs"):
            self.automl_output_tabs.setCurrentIndex(0)

    def closeEvent(self, event):
        if self.data_loader_thread is not None and self.data_loader_thread.isRunning():
            self.data_loader_thread.wait(1000)
        if self.comparison_thread is not None and self.comparison_thread.isRunning():
            self.comparison_thread.stop()
            self.comparison_thread.wait(1000)
        if self.automl_thread is not None and self.automl_thread.isRunning():
            self.automl_thread.stop()
            self.automl_thread.wait(1000)
        if self.workflow_thread is not None and self.workflow_thread.isRunning():
            self.workflow_thread.stop()
            self.workflow_thread.wait(1000)
        if self.workflow_batch_thread is not None and self.workflow_batch_thread.isRunning():
            self.workflow_batch_thread.stop()
            self.workflow_batch_thread.wait(1000)
        if self.rebuild_thread is not None and self.rebuild_thread.isRunning():
            self.rebuild_thread.wait(1000)
        self.data_builder.release_bundle(self.dataset_bundle)
        self.dataset_bundle = None
        super().closeEvent(event)
