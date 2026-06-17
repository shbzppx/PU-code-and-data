"""Result analysis helpers for model comparison."""

from __future__ import annotations

import pandas as pd

from .analysis_utils import flatten_advice


class ResultAnalyzer:
    """Build tables and summaries from comparison results."""

    def __init__(self, results):
        self.results = results

    def generate_comparison_table(self):
        if not self.results:
            return pd.DataFrame()

        rows = []
        for result in self.results:
            metrics = result.get("results", {})
            row = {
                "模型": result.get("model_type", ""),
                "训练准确率": f"{metrics.get('train_acc', 0.0):.4f}",
                "验证准确率": f"{metrics.get('val_acc', 0.0):.4f}",
                "验证召回率": self._format_optional(metrics.get("val_recall")),
                "测试召回率": self._format_optional(metrics.get("test_recall") if result.get("test_metrics_available") else None),
                "验证集预测面积占比": self._format_optional(result.get("val_prediction_area_ratio")),
                "测试集预测面积占比": self._format_optional(result.get("test_prediction_area_ratio")),
                "验证集EI": self._format_optional(result.get("val_ei")),
                "预测集EI": self._format_optional(result.get("test_ei")),
                "综合验证得分": self._format_optional(result.get("validation_score")),
                "训练时间(秒)": f"{metrics.get('training_time_seconds', 0.0):.1f}",
                "改进建议": flatten_advice(result.get("improvement_advice")),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def get_best_model(self, metric="composite_score"):
        if not self.results:
            return None

        def value_getter(item):
            if metric in item:
                return item.get(metric, 0.0) or 0.0
            return item.get("results", {}).get(metric, 0.0) or 0.0

        return max(self.results, key=value_getter)

    def export_to_csv(self, output_path):
        self.generate_comparison_table().to_csv(output_path, index=False, encoding="utf-8-sig")

    def _format_optional(self, value):
        if value is None:
            return ""
        return f"{float(value):.4f}"
