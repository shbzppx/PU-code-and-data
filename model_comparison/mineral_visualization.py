"""矿床预测可视化面板"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
from matplotlib.figure import Figure


class MineralPredictionViewer(QWidget):
    """矿床预测可视化面板"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.prediction_map = None
        self.train_minerals = None
        self.test_minerals = None
        self.metadata = None
        self._build_ui()

    def _build_ui(self):
        """构建界面"""
        layout = QVBoxLayout(self)

        # 控制区域
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("模型:"))
        self.model_combo = QComboBox()
        control_layout.addWidget(self.model_combo)
        control_layout.addStretch()
        layout.addLayout(control_layout)

        # 图表区域
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)

        # 统计信息
        self.stats_label = QLabel("统计信息: 未加载")
        layout.addWidget(self.stats_label)

    def set_data(self, prediction_map, train_minerals, test_minerals, metadata):
        """设置数据"""
        self.prediction_map = prediction_map
        self.train_minerals = train_minerals
        self.test_minerals = test_minerals
        self.metadata = metadata
        self.plot()

    def plot(self):
        """绘制预测图和矿点"""
        if self.prediction_map is None:
            return

        self.ax.clear()

        # 显示预测概率图
        extent = [self.metadata['x_min'], self.metadata['x_max'],
                  self.metadata['y_min'], self.metadata['y_max']]
        im = self.ax.imshow(self.prediction_map, cmap='viridis',
                           extent=extent, origin='lower', alpha=0.8)

        # 叠加训练矿点（绿色）
        if self.train_minerals is not None and len(self.train_minerals) > 0:
            self.ax.scatter(self.train_minerals['x'], self.train_minerals['y'],
                          marker='*', color='green', s=150, edgecolor='black',
                          linewidths=1.5, zorder=5, label='训练矿点')

        # 叠加测试矿点（红色）
        if self.test_minerals is not None and len(self.test_minerals) > 0:
            self.ax.scatter(self.test_minerals['x'], self.test_minerals['y'],
                          marker='*', color='red', s=150, edgecolor='black',
                          linewidths=1.5, zorder=5, label='测试矿点')

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('矿床预测图')
        self.ax.legend()
        self.figure.colorbar(im, ax=self.ax, label='预测概率')
        self.canvas.draw()

