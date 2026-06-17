"""Visualization helpers for comparison results."""

from __future__ import annotations

import matplotlib.pyplot as plt


class Visualization:
    """Render comparison charts."""

    def __init__(self, results):
        self.results = results
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

    def plot_metric_comparison(
        self,
        metric="test_acc",
        fig=None,
        ylabel=None,
        title=None,
        color="steelblue",
    ):
        if not self.results:
            return None

        models = [item.get("model_type", "") for item in self.results]
        values = [self._extract_metric(item, metric) for item in self.results]

        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            ax = fig.add_subplot(111)

        ax.bar(models, values, color=color)
        ax.set_ylabel(ylabel or metric)
        ax.set_title(title or f"{metric} 对比")
        max_value = max(values) if values else 0.0
        ax.set_ylim([0, max(max_value * 1.1, 1.0)])
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return fig

    def plot_mineral_detection_comparison(
        self,
        metric="val_mineral_detection_rate",
        fig=None,
        title="各模型内部验证矿点检出率对比",
    ):
        if not self.results:
            return None

        models = [item.get("model_type", "") for item in self.results]
        detection_rates = [self._extract_metric(item, metric) for item in self.results]

        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            ax = fig.add_subplot(111)

        ax.bar(models, detection_rates, color="seagreen")
        ax.set_ylabel("内部验证矿点检出率")
        ax.set_title(title)
        ax.set_ylim([0, 1])
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return fig

    def _extract_metric(self, item, metric):
        if metric in item:
            value = item.get(metric)
        else:
            value = item.get("results", {}).get(metric, 0.0)
        return float(value or 0.0)
