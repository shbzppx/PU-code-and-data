"""PDF reporting for repeated split AutoML comparison results."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from .report_generator import ReportGenerator


class BatchReportGenerator(ReportGenerator):
    """Generate a PDF report for repeated split AutoML experiments."""

    def __init__(self, batch_summary):
        super().__init__([])
        self.batch_summary = dict(batch_summary or {})

    def _chunk_rows(self, rows, chunk_size):
        rows = list(rows or [])
        if chunk_size <= 0:
            return [rows]
        return [rows[index : index + chunk_size] for index in range(0, len(rows), chunk_size)] or [[]]

    def _overview_rows(self):
        summary = self.batch_summary
        best_result = summary.get("best_result") or {}
        best_model = summary.get("best_model") or {}
        best_run = summary.get("best_run") or {}
        selected_models = summary.get("selected_models") or []
        test_split_ratio = summary.get("test_split_ratio", "")
        if test_split_ratio not in ("", None):
            try:
                test_split_ratio = f"{float(test_split_ratio) * 100:.0f}%"
            except (TypeError, ValueError):
                test_split_ratio = str(test_split_ratio)
        multiplier = summary.get("negative_distance_multiplier", "")
        try:
            multiplier = f"{float(multiplier):.2f}" if multiplier not in ("", None) else ""
        except (TypeError, ValueError):
            multiplier = str(multiplier)
        radius = summary.get("negative_distance_radius", "")
        try:
            radius = f"{float(radius):.1f} 米" if radius not in ("", None) else ""
        except (TypeError, ValueError):
            radius = str(radius)

        def _format_bounds(bounds):
            bounds = dict(bounds or {})
            if not bounds:
                return ""
            return (
                f"x=[{bounds.get('xmin', '-')}, {bounds.get('xmax', '-')}]"
                f"\ny=[{bounds.get('ymin', '-')}, {bounds.get('ymax', '-')}]"
            )

        rows = [
            ["连续划分次数", summary.get("split_count", 0)],
            ["基础随机种子", summary.get("base_seed", "")],
            ["运行模式", summary.get("runtime_mode", "")],
            ["负样本策略", self._format_negative_sampling_mode_display(summary.get("negative_sampling_mode"))],
            ["负样本是否生效", "是" if summary.get("negative_sampling_applied") else "否"],
            ["远区起始倍数", multiplier],
            ["远区起始半径", radius],
            ["无矿钻孔/坐标", "启用" if summary.get("no_ore_active") else "未提供"],
            ["无矿锚点点数", summary.get("no_ore_point_count", 0)],
            ["无矿锚点覆盖样本", summary.get("no_ore_sample_count", 0)],
            ["无矿锚点冲突覆盖", summary.get("no_ore_conflict_count", 0)],
            ["坐标区域切分", "启用" if summary.get("spatial_region_active") else "未启用"],
            ["训练区域范围", _format_bounds(summary.get("spatial_region_train_bounds"))],
            ["预测区域范围", _format_bounds(summary.get("spatial_region_test_bounds"))],
            ["边界缓冲带", f"{float(summary.get('spatial_region_buffer_distance', 0.0) or 0.0):.1f} 米"],
            ["训练区域样本", summary.get("spatial_region_train_sample_count", 0)],
            ["预测区域样本", summary.get("spatial_region_test_sample_count", 0)],
            ["灰区剔除样本", summary.get("spatial_region_gray_sample_count", 0)],
            ["区域外样本", summary.get("spatial_region_outside_sample_count", 0)],
            ["训练区无矿负样本", summary.get("spatial_region_train_no_ore_sample_count", 0)],
            ["预测区无矿负样本", summary.get("spatial_region_test_no_ore_sample_count", 0)],
            ["测试集比例", test_split_ratio],
            ["选中模型", "、".join(str(item) for item in selected_models)],
            ["最佳划分轮次", summary.get("best_run_index", "")],
            ["最佳划分模型", best_result.get("model_name", "")],
            ["最佳划分测试集EI", f"{float(best_result.get('best_test_ei') or best_result.get('test_ei') or 0.0):.4f}" if best_result else ""],
            ["最稳模型", best_model.get("model_name", "")],
            ["最稳分数", f"{float(best_model.get('stability_score') or 0.0):.4f}" if best_model else ""],
            ["最稳模型Top1次数", best_model.get("top1_count", "")],
            ["输出目录", summary.get("output_dir", "")],
            ["最佳划分输出", best_run.get("output_dir", "")],
        ]
        if summary.get("batch_report_path"):
            rows.append(["批次报告", summary.get("batch_report_path", "")])
        if summary.get("batch_summary_xlsx_path"):
            rows.append(["Excel 汇总", summary.get("batch_summary_xlsx_path", "")])
        return rows
        best_result = summary.get("best_result") or {}
        best_model = summary.get("best_model") or {}
        best_run = summary.get("best_run") or {}
        selected_models = summary.get("selected_models") or []
        test_split_ratio = summary.get("test_split_ratio", "")
        if test_split_ratio not in ("", None):
            try:
                test_split_ratio = f"{float(test_split_ratio) * 100:.0f}%"
            except (TypeError, ValueError):
                test_split_ratio = str(test_split_ratio)
        multiplier = summary.get("negative_distance_multiplier", "")
        try:
            multiplier = f"{float(multiplier):.2f}" if multiplier not in ("", None) else ""
        except (TypeError, ValueError):
            multiplier = str(multiplier)
        radius = summary.get("negative_distance_radius", "")
        try:
            radius = f"{float(radius):.1f} 米" if radius not in ("", None) else ""
        except (TypeError, ValueError):
            radius = str(radius)
        rows = [
            ["连续划分次数", summary.get("split_count", 0)],
            ["基础随机种子", summary.get("base_seed", "")],
            ["运行模式", summary.get("runtime_mode", "")],
            ["负样本策略", self._format_negative_sampling_mode_display(summary.get("negative_sampling_mode"))],
            ["负样本是否生效", "是" if summary.get("negative_sampling_applied") else "否"],
            ["远区起始倍数", multiplier],
            ["远区起始半径", radius],
            ["无矿钻孔/坐标", "启用" if summary.get("no_ore_active") else "未提供"],
            ["无矿锚点点数", summary.get("no_ore_point_count", 0)],
            ["无矿锚点覆盖样本", summary.get("no_ore_sample_count", 0)],
            ["无矿锚点冲突覆盖", summary.get("no_ore_conflict_count", 0)],
            ["测试集比例", test_split_ratio],
            ["选中模型", "、".join(str(item) for item in selected_models)],
            ["最佳划分轮次", summary.get("best_run_index", "")],
            ["最佳划分模型", best_result.get("model_name", "")],
            ["最佳划分测试集EI", f"{float(best_result.get('best_test_ei') or best_result.get('test_ei') or 0.0):.4f}" if best_result else ""],
            ["最稳模型", best_model.get("model_name", "")],
            ["最稳分数", f"{float(best_model.get('stability_score') or 0.0):.4f}" if best_model else ""],
            ["最稳模型 Top1 次数", best_model.get("top1_count", "")],
            ["输出目录", summary.get("output_dir", "")],
            ["最佳划分输出", best_run.get("output_dir", "")],
        ]
        if summary.get("batch_report_path"):
            rows.append(["批次报告", summary.get("batch_report_path", "")])
        if summary.get("batch_summary_xlsx_path"):
            rows.append(["Excel 汇总", summary.get("batch_summary_xlsx_path", "")])
        return rows

    def _run_rows(self):
        rows = []
        for item in self.batch_summary.get("run_rows") or []:
            rows.append(
                [
                    item.get("run_index", ""),
                    item.get("split_seed", ""),
                    "是" if item.get("is_best_run") else "",
                    item.get("best_model_name", ""),
                    f"{float(item.get('best_test_ei') or 0.0):.4f}" if item.get("best_test_ei") is not None else "",
                    f"{float(item.get('best_val_accuracy') or 0.0):.4f}" if item.get("best_val_accuracy") is not None else "",
                    f"{float(item.get('best_test_mineral_detection_rate') or 0.0):.4f}" if item.get("best_test_mineral_detection_rate") is not None else "",
                    item.get("train_mineral_count", ""),
                    item.get("test_mineral_count", ""),
                    item.get("output_name", item.get("output_dir", "")),
                ]
            )
        return rows

    def _model_rows(self):
        rows = []
        for item in self.batch_summary.get("model_rows") or []:
            rows.append(
                [
                    item.get("model_name", ""),
                    item.get("run_count", ""),
                    item.get("top1_count", ""),
                    f"{float(item.get('mean_test_ei') or 0.0):.4f}" if item.get("mean_test_ei") is not None else "",
                    f"{float(item.get('std_test_ei') or 0.0):.4f}" if item.get("std_test_ei") is not None else "",
                    f"{float(item.get('mean_val_accuracy') or 0.0):.4f}" if item.get("mean_val_accuracy") is not None else "",
                    f"{float(item.get('mean_test_mineral_detection_rate') or 0.0):.4f}" if item.get("mean_test_mineral_detection_rate") is not None else "",
                    f"{float(item.get('stability_score') or 0.0):.4f}" if item.get("stability_score") is not None else "",
                ]
            )
        return rows

    def _overview_frame(self):
        return pd.DataFrame(self._overview_rows(), columns=["字段", "内容"])

    def _run_frame(self):
        return pd.DataFrame(
            self._run_rows(),
            columns=["轮次", "seed", "最佳", "模型", "测试集EI", "验证Acc", "测试检出", "训练矿点", "测试矿点", "输出目录"],
        )

    def _model_frame(self):
        return pd.DataFrame(
            self._model_rows(),
            columns=["模型", "出现次数", "Top1次数", "平均测试集EI", "测试集EI标准差", "平均验证Acc", "平均测试检出", "稳定分"],
        )

    def _split_rows(self):
        summary = self.batch_summary.get("mineral_split_summary") or {}
        split_mode = str(summary.get("split_mode") or self.batch_summary.get("split_mode") or "").strip().lower()
        specs = [("train", "训练矿点"), ("test", "测试矿点")]
        if split_mode and split_mode not in {"single", "single_file", "single-file", "singlefile", "spatial_region"}:
            specs = [("train", "训练矿点"), ("val", "验证矿点"), ("test", "测试矿点")]
        if summary.get("no_ore"):
            specs.append(("no_ore", "无矿钻孔/坐标"))

        rows = []
        for split_key, split_label in specs:
            block = summary.get(split_key) or {}
            items = [str(item) for item in (block.get("items") or []) if str(item).strip()]
            if not items:
                rows.append([split_label, "", "暂无"])
                continue
            for index, item in enumerate(items, start=1):
                rows.append([split_label, index, item])
        return rows

    def _split_frame(self):
        return pd.DataFrame(self._split_rows(), columns=["分组", "序号", "矿点"])

    def _get_excel_engine(self):
        for engine in ("openpyxl", "xlsxwriter"):
            if importlib.util.find_spec(engine) is not None:
                return engine
        return None

    def _render_overview_page(self):
        fig = plt.figure(figsize=(11.69, 8.27))
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[0.18, 0.82],
            width_ratios=[1.05, 0.95],
            hspace=0.24,
            wspace=0.16,
        )

        ax_title = fig.add_subplot(gs[0, :])
        best_result = self.batch_summary.get("best_result") or {}
        best_model = self.batch_summary.get("best_model") or {}
        subtitle = (
            f"最佳划分: {best_result.get('model_name', '-')} | "
            f"最稳模型: {best_model.get('model_name', '-')} | "
            f"总轮次: {self.batch_summary.get('split_count', 0)}"
        )
        self._render_page_header(ax_title, "连续划分结果总览", subtitle)

        ax_left = fig.add_subplot(gs[1, 0])
        self._render_table_section(
            ax_left,
            "批次摘要",
            self._overview_rows(),
            col_labels=("字段", "内容"),
            font_size=8.2,
            bbox=(0.0, 0.0, 1.0, 0.90),
            col_widths=(0.34, 0.62),
            wrap_text=True,
        )

        ax_right = fig.add_subplot(gs[1, 1])
        top_models = list(self.batch_summary.get("model_rows") or [])[:5]
        model_rows = [
            [
                item.get("model_name", ""),
                item.get("run_count", ""),
                item.get("top1_count", ""),
                item.get("stability_score", ""),
            ]
            for item in top_models
        ] or [["暂无", "", "", ""]]
        self._render_table_section(
            ax_right,
            "Top 5 稳定模型",
            model_rows,
            col_labels=("模型", "出现", "Top1", "稳定分"),
            font_size=8.1,
            bbox=(0.0, 0.0, 1.0, 0.90),
            col_widths=(0.42, 0.14, 0.14, 0.20),
            left_aligned=False,
        )

        fig.subplots_adjust(left=0.04, right=0.96, top=0.95, bottom=0.04)
        return fig

    def _render_run_page(self, run_rows, page_index, total_pages):
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_subplot(111)
        subtitle = f"第 {page_index}/{total_pages} 页 | 每轮结果仅显示摘要，完整矿点清单保存在各轮输出目录中"
        self._render_page_header(ax, "每轮划分与优化结果", subtitle)
        rows = run_rows or [["暂无", "", "", "", "", "", "", "", "", ""]]
        self._render_table_section(
            ax,
            "轮次对比",
            rows,
            col_labels=("轮次", "seed", "最佳", "模型", "综合分", "验证Acc", "测试检出", "训练矿点", "测试矿点", "输出目录"),
            font_size=7.5,
            bbox=(0.0, 0.0, 1.0, 0.86),
            col_widths=(0.06, 0.08, 0.05, 0.14, 0.10, 0.10, 0.10, 0.08, 0.08, 0.21),
            left_aligned=False,
        )
        fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.03)
        return fig

    def _render_model_page(self, model_rows):
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_subplot(111)
        self._render_page_header(
            ax,
            "模型稳定性汇总",
            "稳定分越高、标准差越小，模型在不同划分下越稳。",
        )
        self._render_table_section(
            ax,
            "模型对比",
            model_rows,
            col_labels=("模型", "出现次数", "Top1次数", "平均综合分", "标准差", "平均验证Acc", "平均测试检出", "稳定分"),
            font_size=7.8,
            bbox=(0.0, 0.0, 1.0, 0.86),
            col_widths=(0.14, 0.09, 0.09, 0.10, 0.10, 0.12, 0.13, 0.10),
            left_aligned=False,
        )
        fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.03)
        return fig

    def generate_pdf_report(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        run_rows = self._run_rows()
        model_rows = self._model_rows()
        run_pages = self._chunk_rows(run_rows, 12)
        if not run_pages:
            run_pages = [[]]

        with PdfPages(str(output_path)) as pdf:
            pdf.savefig(self._render_overview_page(), bbox_inches="tight")
            plt.close("all")

            negative_mode_context = self.batch_summary.get("best_result") or self.batch_summary.get("best_run") or self.batch_summary
            pdf.savefig(self._build_negative_sampling_mode_page(negative_mode_context), bbox_inches="tight")
            plt.close("all")

            for page_index, page_rows in enumerate(run_pages, start=1):
                pdf.savefig(self._render_run_page(page_rows, page_index, len(run_pages)), bbox_inches="tight")
                plt.close("all")

            pdf.savefig(self._render_model_page(model_rows or [["暂无", "", "", "", "", "", "", ""]]), bbox_inches="tight")
            plt.close("all")

        return str(output_path)

    def generate_excel_report(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        engine = self._get_excel_engine()
        if engine is None:
            raise ImportError("需要安装 openpyxl 或 xlsxwriter 才能导出 Excel 汇总。")

        with pd.ExcelWriter(str(output_path), engine=engine) as writer:
            self._overview_frame().to_excel(writer, sheet_name="总览", index=False)
            negative_mode_context = self.batch_summary.get("best_result") or self.batch_summary.get("best_run") or self.batch_summary
            self._negative_sampling_mode_frame(negative_mode_context).to_excel(writer, sheet_name="负样本模式", index=False)
            self._run_frame().to_excel(writer, sheet_name="每轮结果", index=False)
            self._model_frame().to_excel(writer, sheet_name="模型稳定性", index=False)
            self._split_frame().to_excel(writer, sheet_name="最佳划分矿点", index=False)

        return str(output_path)
