"""Result analysis helpers for model comparison."""

from __future__ import annotations

import math

import pandas as pd

from .analysis_utils import flatten_advice, primary_test_ei_score


class ResultAnalyzer:
    """Build tables and summaries from comparison results."""

    def __init__(self, results):
        self.results = results

    def generate_comparison_table(self):
        if not self.results:
            return pd.DataFrame()

        rows = []
        for result in self.results:
            dataset_summary = dict(result.get("dataset_summary", {}) or {})
            row = {
                "图层方案": result.get("layer_scheme_name", ""),
                "图层数": result.get("layer_scheme_channel_count", ""),
                "模型": result.get("model_type", ""),
                "验证样本数": self._format_integer(dataset_summary.get("val_sample_count")),
                "验证正样本数": self._format_integer(dataset_summary.get("val_positive_count")),
                "测试样本数": self._format_integer(dataset_summary.get("test_sample_count")),
                "测试正样本数": self._format_integer(dataset_summary.get("test_positive_count")),
                "验证召回率": self._format_optional(
                    self._metric(result, "val_recall", fallback=result.get("val_mineral_detection_rate"))
                ),
                "验证损失": self._format_optional(self._metric(result, "val_loss")),
                "内部验证矿点检出率": self._format_optional(result.get("val_mineral_detection_rate")),
                "验证集预测面积占比(PAF)": self._format_optional(
                    self._metric(result, "val_paf", "val_prediction_area_ratio")
                ),
                "验证集EI": self._format_optional(self._metric(result, "val_ei")),
                "测试准确率": self._format_optional(
                    self._metric(result, "test_acc") if result.get("test_metrics_available") else None
                ),
                "测试精确率": self._format_optional(
                    self._metric(result, "test_precision") if result.get("test_metrics_available") else None
                ),
                "测试召回率": self._format_optional(
                    self._metric(result, "test_recall", fallback=result.get("test_mineral_detection_rate"))
                ),
                "测试F1": self._format_optional(
                    self._metric(result, "test_f1") if result.get("test_metrics_available") else None
                ),
                "测试矿点检出率": self._format_optional(result.get("test_mineral_detection_rate")),
                "测试集预测面积占比(PAF)": self._format_optional(
                    self._metric(result, "test_paf", "test_prediction_area_ratio")
                ),
                "测试集EI": self._format_optional(self._metric(result, "test_ei")),
                "CV折数": self._format_integer(self._metric(result, "cv_fold_count")),
                "CV SR均值": self._format_optional(self._metric(result, "cv_sr_mean", "val_sr")),
                "CV SR标准差": self._format_optional(self._metric(result, "cv_sr_std")),
                "CV SR(mean±std)": self._format_mean_std(
                    self._metric(result, "cv_sr_mean", "val_sr"),
                    self._metric(result, "cv_sr_std"),
                ),
                "CV PAF均值": self._format_optional(self._metric(result, "cv_paf_mean", "val_paf")),
                "CV PAF标准差": self._format_optional(self._metric(result, "cv_paf_std")),
                "CV PAF(mean±std)": self._format_mean_std(
                    self._metric(result, "cv_paf_mean", "val_paf"),
                    self._metric(result, "cv_paf_std"),
                ),
                "CV EI均值": self._format_optional(self._metric(result, "cv_ei_mean", "val_ei")),
                "CV EI标准差": self._format_optional(self._metric(result, "cv_ei_std")),
                "CV EI(mean±std)": self._format_mean_std(
                    self._metric(result, "cv_ei_mean", "val_ei"),
                    self._metric(result, "cv_ei_std"),
                ),
                "泛化分数": self._format_optional(self._metric(result, "generalization_score")),
                "综合验证得分": self._format_optional(
                    self._metric(result, "validation_score", "composite_score")
                ),
                "训练时间(秒)": self._format_optional(self._metric(result, "training_time_seconds"), digits=1),
                "改进建议": flatten_advice(result.get("improvement_advice")),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def get_best_model(self, metric="best_test_ei"):
        if not self.results:
            return None
        if metric in {"best_test_ei", "test_ei", "primary_score"}:
            return max(self.results, key=primary_test_ei_score)

        def value_getter(item):
            if metric in item:
                return item.get(metric, 0.0) or 0.0
            return item.get("results", {}).get(metric, 0.0) or 0.0

        return max(self.results, key=value_getter)

    def export_to_csv(self, output_path):
        self.generate_comparison_table().to_csv(output_path, index=False, encoding="utf-8-sig")

    def _metric(self, result, *keys, fallback=None):
        metrics = dict(result.get("results", {}) or {})
        for key in keys:
            if key in result and result.get(key) is not None:
                return result.get(key)
            if key in metrics and metrics.get(key) is not None:
                return metrics.get(key)
        return fallback

    def _format_optional(self, value, digits=4):
        if value is None:
            return ""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not math.isfinite(number):
            return ""
        return f"{number:.{int(digits)}f}"

    def _format_mean_std(self, mean_value, std_value, digits=4):
        mean_text = self._format_optional(mean_value, digits=digits)
        if not mean_text:
            return ""
        std_text = self._format_optional(0.0 if std_value is None else std_value, digits=digits)
        return f"{mean_text} ± {std_text}"

    def _format_integer(self, value):
        if value is None:
            return ""
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return ""
