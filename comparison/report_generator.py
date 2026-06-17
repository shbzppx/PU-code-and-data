"""PDF reporting for AutoML comparison results."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import textwrap

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from .analysis_utils import (
    DETECTION_THRESHOLD,
    flatten_advice,
    resolve_composite_formula,
    to_jsonable,
)


class ReportGenerator:
    """Generate PDF reports for AutoML results."""

    def __init__(self, results):
        self.results = list(results or [])
        self.font_family = self._resolve_cjk_font_family()
        plt.rcParams["font.family"] = self.font_family
        plt.rcParams["font.sans-serif"] = [
            self.font_family,
            "Microsoft YaHei",
            "SimHei",
            "STSong",
            "SimSun",
            "KaiTi",
        ]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42

    @staticmethod
    def _resolve_cjk_font_family() -> str:
        """Choose a Chinese-capable font that exists on this machine."""
        available_fonts = {font.name for font in fm.fontManager.ttflist}
        for font_name in ["Microsoft YaHei", "SimHei", "STSong", "SimSun", "KaiTi"]:
            if font_name in available_fonts:
                return font_name
        return "DejaVu Sans"

    def _apply_table_font(self, rendered_table) -> None:
        """Apply the selected font to every cell in a Matplotlib table."""
        for cell in rendered_table.get_celld().values():
            cell.get_text().set_fontfamily(self.font_family)

    def _compact_value(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, Mapping):
            return json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(", ", ": "))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            parts = []
            for item in value:
                text = self._compact_value(item)
                if text:
                    parts.append(text)
            return "；".join(parts)
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def _wrap_display_text(self, value, width: int) -> str:
        text = self._compact_value(value)
        if not text:
            return ""
        width = max(int(width or 0), 8)
        wrapped_lines = []
        for paragraph in str(text).splitlines():
            if not paragraph:
                wrapped_lines.append("")
                continue
            wrapped_lines.append(
                textwrap.fill(
                    paragraph,
                    width=width,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
            )
        return "\n".join(wrapped_lines)

    def _kv_rows(self, mapping) -> list[list[str]]:
        if not mapping:
            return [["暂无", ""]]
        label_map = {
            "cache_key": "缓存键",
            "train_mineral_count": "内部训练矿点数",
            "val_mineral_count": "内部验证矿点数",
            "test_mineral_count": "测试矿点数",
            "train_sample_count": "内部训练样本数",
            "val_sample_count": "内部验证样本数",
            "test_sample_count": "测试样本数",
            "train_positive_count": "内部训练正样本数",
            "val_positive_count": "内部验证正样本数",
            "test_positive_count": "测试正样本数",
            "train_negative_count": "内部训练负样本数",
            "val_negative_count": "内部验证负样本数",
            "test_negative_count": "测试负样本数",
            "sampling_percentage": "抽样比例",
            "balance_ratio": "正/负样本比例",
            "negative_ratio": "负/正样本倍数",
            "negative_sampling_mode": "负样本策略",
            "negative_sampling_applied": "负样本策略是否生效",
            "negative_distance_multiplier": "远区起始倍数",
            "negative_distance_radius": "远区起始半径",
            "no_ore_active": "无矿钻孔/坐标是否启用",
            "no_ore_point_count": "无矿钻孔/坐标点数",
            "no_ore_sample_count": "无矿锚点覆盖样本数",
            "no_ore_conflict_count": "无矿锚点冲突覆盖数",
            "spatial_region_active": "坐标区域切分是否启用",
            "spatial_region_train_bounds": "训练区域范围",
            "spatial_region_test_bounds": "预测区域范围",
            "spatial_region_buffer_distance": "边界缓冲带",
            "spatial_region_train_sample_count": "训练区原始样本数",
            "spatial_region_test_sample_count": "预测区原始样本数",
            "spatial_region_gray_sample_count": "灰区剔除样本数",
            "spatial_region_outside_sample_count": "区域外样本数",
            "spatial_region_overlap_sample_count": "区域重叠样本数",
            "spatial_region_train_no_ore_sample_count": "训练区无矿负样本数",
            "spatial_region_test_no_ore_sample_count": "预测区无矿负样本数",
            "val_ratio": "内部验证比例",
            "validation_split": "内部验证比例",
        }
        rows = []
        for key, value in sorted(mapping.items(), key=lambda item: str(item[0])):
            display_value = self._compact_value(value)
            if key == "negative_sampling_mode":
                display_value = self._format_negative_sampling_mode_display(value)
            elif key == "negative_sampling_applied":
                display_value = "是" if bool(value) else "否"
            elif key == "cache_key":
                cache_key_text = self._compact_value(value)
                parts = [part for part in cache_key_text.split("|") if part]
                display_value = "\n".join(parts) if len(parts) > 1 else cache_key_text
            elif key in {"no_ore_active", "spatial_region_active"}:
                display_value = "是" if bool(value) else "否"
            elif key in {"spatial_region_train_bounds", "spatial_region_test_bounds"}:
                bounds = dict(value or {}) if isinstance(value, Mapping) else {}
                display_value = (
                    f"x=[{bounds.get('xmin', '-')}, {bounds.get('xmax', '-')}]"
                    f"\ny=[{bounds.get('ymin', '-')}, {bounds.get('ymax', '-')}]"
                )
            rows.append([label_map.get(str(key), str(key)), str(display_value)])
        return rows

    def _workflow_config_rows(self, mapping) -> list[list[str]]:
        if not mapping:
            return []
        label_map = {
            "runtime_mode": "运行模式",
            "stage1_trials": "第1阶段试验次数",
            "stage1_epochs": "第1阶段训练轮次",
            "stage2_trials": "第2阶段试验次数",
            "stage2_epochs": "第2阶段训练轮次",
        }
        row_order = ["runtime_mode", "stage1_trials", "stage1_epochs", "stage2_trials", "stage2_epochs"]
        rows = []
        for key in row_order:
            if key in mapping:
                rows.append([label_map[key], self._compact_value(mapping.get(key))])
        return rows

    def _format_negative_sampling_mode_display(self, value) -> str:
        mode = str(value or "").strip().lower()
        if mode == "far_distance":
            return "远区确认负样本"
        if mode == "default":
            return "默认背景随机负样本"
        if mode:
            return mode
        return ""

    def _format_trial_value(self, value) -> str:
        if value is None:
            return "default"
        if isinstance(value, bool):
            return "true" if value else "false"
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return str(value)

        if abs(numeric_value - round(numeric_value)) < 1e-9:
            return str(int(round(numeric_value)))
        return f"{numeric_value:.4g}"

    def _format_trial_scheme_brief(self, mapping) -> str:
        if not mapping:
            return "暂无"

        if isinstance(mapping, Mapping):
            preferred_keys = [
                ("patch_size", "ps"),
                ("patch_stride", "st"),
                ("buffer_radius", "br"),
                ("negative_sampling_mode", "ns"),
                ("negative_distance_multiplier", "ndm"),
                ("sampling_percentage", "sp"),
                ("balance_ratio", "bal"),
            ]
            if any(key in mapping for key, _ in preferred_keys):
                parts = []
                for key, label in preferred_keys:
                    if key in mapping:
                        if key == "negative_sampling_mode":
                            parts.append(f"{label}={self._format_negative_sampling_mode_display(mapping.get(key))}")
                        else:
                            parts.append(f"{label}={self._format_trial_value(mapping.get(key))}")
                return " | ".join(parts) if parts else "暂无"

            preview_items = list(mapping.items())[:5]
            parts = [f"{str(key)}={self._format_trial_value(value)}" for key, value in preview_items]
            if len(mapping) > 5:
                parts.append("...")
            return " | ".join(parts) if parts else "暂无"

        return self._format_trial_value(mapping)

    def _trial_scheme_lines(self, result) -> list[str]:
        summary = result.get("trial_scheme_summary") or {}
        stage1_trials = list(summary.get("stage1_trials") or [])

        if stage1_trials:
            stage1_total = int(summary.get("stage1_total", len(stage1_trials)) or len(stage1_trials))
            stage2_total = int(summary.get("stage2_total", 0) or 0)
            lines = [
                f"Stage 1 扫描 {stage1_total} 个候选方案；Stage 2 复评 {stage2_total} 个方案",
            ]
            for item in stage1_trials[:2]:
                rank = item.get("rank")
                score = item.get("scheme_score")
                scheme_text = self._format_trial_scheme_brief(item.get("scheme") or {})
                prefix = f"{rank}. " if rank is not None else "- "
                suffix = f" | 方案评分 {float(score):.4f}" if score is not None else ""
                lines.append(f"{prefix}{scheme_text}{suffix}")
            if len(stage1_trials) > 2:
                lines.append(f"... 其余 {len(stage1_trials) - 2} 个候选方案未展开")
            return lines

        history = result.get("optimization_history") or []
        if history:
            lines = [f"本次共进行了 {len(history)} 次参数 trial 搜索"]
            for index, item in enumerate(history[:2], start=1):
                lines.append(f"{index}. {self._format_trial_scheme_brief(item.get('params') or {})}")
            if len(history) > 2:
                lines.append(f"... 其余 {len(history) - 2} 次 trial 未展开")
            return lines

        return ["暂无参数试验方案记录"]

    def _get_mineral_split_mode(self, result) -> str:
        summary = result.get("mineral_split_summary") or {}
        split_mode = summary.get("split_mode") or (result.get("dataset_params") or {}).get("split_mode")
        return str(split_mode or "").strip().lower()

    def _mineral_section_specs(self, result) -> list[tuple[str, str]]:
        if self._get_mineral_split_mode(result) in {"single", "spatial_region"}:
            return [("train", "训练矿点"), ("test", "测试矿点")]
        return [("train", "训练矿点"), ("val", "验证矿点")]

    def _mineral_split_lines(self, result) -> list[str]:
        summary = result.get("mineral_split_summary") or {}
        lines = []
        for key, title in self._mineral_section_specs(result):
            block = summary.get(key) or {}
            count = int(block.get("count", 0) or 0)
            items = list(block.get("items") or [])
            lines.append(f"{title} {count} 个")
            if items:
                preview_items = items[:1]
                lines.extend(preview_items)
                if block.get("truncated") or len(items) > len(preview_items):
                    lines.append(f"... 其余 {count - len(preview_items)} 个未展开")
            else:
                lines.append("暂无")
        no_ore_block = summary.get("no_ore") or {}
        if no_ore_block:
            count = int(no_ore_block.get("count", 0) or 0)
            items = list(no_ore_block.get("items") or [])
            lines.append(f"无矿钻孔/坐标 {count} 个")
            if items:
                preview_items = items[:1]
                lines.extend(preview_items)
                if no_ore_block.get("truncated") or len(items) > len(preview_items):
                    lines.append(f"... 其余 {count - len(preview_items)} 个未展开")
            else:
                lines.append("暂无")
        return lines

    def _trial_scheme_table_rows(self, result) -> tuple[list[list[str]], int, int]:
        summary = result.get("trial_scheme_summary") or {}
        stage1_trials = list(summary.get("stage1_trials") or [])

        rows = []
        for item in stage1_trials:
            scheme = item.get("scheme") or {}
            score = item.get("scheme_score")
            try:
                score_text = f"{float(score):.4f}" if score is not None else ""
            except (TypeError, ValueError):
                score_text = self._compact_value(score)

            rows.append(
                [
                    str(item.get("rank") or len(rows) + 1),
                    self._format_trial_value(scheme.get("patch_size")),
                    self._format_trial_value(scheme.get("patch_stride")),
                    self._format_trial_value(scheme.get("buffer_radius")),
                    self._format_trial_value(scheme.get("sampling_percentage")),
                    self._format_trial_value(scheme.get("balance_ratio")),
                    score_text,
                ]
            )

        if not rows:
            rows = [["暂无", "", "", "", "", "", ""]]

        stage1_total = int(summary.get("stage1_total", len(stage1_trials) or 0) or 0)
        stage2_total = int(summary.get("stage2_total", 0) or 0)
        return rows, stage1_total, stage2_total

    def _mineral_preview_table_rows(self, result, split_key: str) -> tuple[list[list[str]], int]:
        summary = result.get("mineral_split_summary") or {}
        block = summary.get(split_key) or {}
        count = int(block.get("count", 0) or 0)
        items = [str(item) for item in (block.get("items") or []) if str(item).strip()]

        rows = [[item] for item in items]
        if block.get("truncated"):
            rows.append([f"... 其余 {max(count - len(items), 0)} 个未展开"])

        if not rows:
            rows = [["暂无"]]

        return rows, count

    def _mineral_table_font_size(self, row_count: int) -> float:
        if row_count <= 8:
            return 8.2
        if row_count <= 12:
            return 7.8
        if row_count <= 18:
            return 7.0
        if row_count <= 24:
            return 6.2
        return 5.6

    def _build_trial_summary_page(self, result):
        trial_rows, stage1_total, stage2_total = self._trial_scheme_table_rows(result)
        split_mode = self._get_mineral_split_mode(result)
        mineral_sections = self._mineral_section_specs(result)
        train_key, train_title = mineral_sections[0]
        second_key, second_title = mineral_sections[1]
        train_rows, train_count = self._mineral_preview_table_rows(result, train_key)
        second_rows, second_count = self._mineral_preview_table_rows(result, second_key)
        mineral_table_font_size = self._mineral_table_font_size(max(train_count, second_count))

        fig = plt.figure(figsize=(11.69, 8.27))
        gs = fig.add_gridspec(
            3,
            2,
            height_ratios=[0.18, 0.41, 0.41],
            width_ratios=[1.35, 0.95],
            hspace=0.30,
            wspace=0.18,
        )

        ax_title = fig.add_subplot(gs[0, :])
        if split_mode == "single":
            subtitle = (
                f"Stage 1 扫描 {stage1_total} 个候选方案；Stage 2 复评 {stage2_total} 个方案 | "
                f"训练矿点 {train_count} 个 | 测试矿点 {second_count} 个 | "
                "单文件自动分割（完整清单）"
            )
        elif split_mode == "spatial_region":
            subtitle = (
                f"Stage 1 扫描 {stage1_total} 个候选方案；Stage 2 复评 {stage2_total} 个方案 | "
                f"训练矿点 {train_count} 个 | 测试矿点 {second_count} 个 | "
                "坐标区域切分（完整清单）"
            )
        else:
            subtitle = (
                f"Stage 1 扫描 {stage1_total} 个候选方案；Stage 2 复评 {stage2_total} 个方案 | "
                f"训练矿点 {train_count} 个 | 验证矿点 {second_count} 个"
            )
        self._render_page_header(ax_title, "参数试验与矿点信息", subtitle)

        ax_trials = fig.add_subplot(gs[1:, 0])
        self._render_table_section(
            ax_trials,
            "参数试验方案",
            trial_rows,
            col_labels=("排名", "ps", "st", "br", "sp", "bal", "评分"),
            font_size=7.9,
            bbox=(0.0, 0.0, 1.0, 0.90),
            col_widths=(0.08, 0.10, 0.10, 0.10, 0.12, 0.12, 0.12),
            left_aligned=False,
        )

        ax_train = fig.add_subplot(gs[1, 1])
        self._render_table_section(
            ax_train,
            f"{train_title}完整清单（{train_count} 个）" if split_mode in {"single", "spatial_region"} else f"{train_title}（{train_count} 个）",
            train_rows,
            col_labels=("预览",),
            font_size=min(mineral_table_font_size, 8.2),
            bbox=(0.0, 0.0, 1.0, 0.90),
            col_widths=(0.95,),
            wrap_text=True,
        )

        ax_second = fig.add_subplot(gs[2, 1])
        self._render_table_section(
            ax_second,
            f"{second_title}完整清单（{second_count} 个）" if split_mode in {"single", "spatial_region"} else f"{second_title}（{second_count} 个）",
            second_rows,
            col_labels=("预览",),
            font_size=min(mineral_table_font_size, 8.2),
            bbox=(0.0, 0.0, 1.0, 0.90),
            col_widths=(0.95,),
            wrap_text=True,
        )

        fig.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
        return fig

    def _negative_sampling_mode_frame(self, context):
        return pd.DataFrame(self._negative_sampling_mode_rows(context), columns=["模式", "说明"])

    def _build_negative_sampling_mode_page(self, context):
        rows = self._negative_sampling_mode_rows(context)
        current_summary = rows[-1][1] if rows else "暂无"
        fig = plt.figure(figsize=(11.69, 4.9))
        gs = fig.add_gridspec(2, 1, height_ratios=[0.22, 0.78], hspace=0.18)

        ax_title = fig.add_subplot(gs[0, 0])
        self._render_page_header(ax_title, "负样本模式说明", f"当前采用: {current_summary}")

        ax_table = fig.add_subplot(gs[1, 0])
        self._render_table_section(
            ax_table,
            "模式对照",
            rows,
            col_labels=("模式", "说明"),
            font_size=9.2,
            bbox=(0.0, 0.0, 1.0, 0.88),
            col_widths=(0.26, 0.68),
            wrap_text=True,
        )

        fig.subplots_adjust(left=0.04, right=0.96, top=0.94, bottom=0.06)
        return fig

    def _render_page_header(self, ax, title: str, subtitle: str | None = None) -> None:
        ax.axis("off")
        ax.text(
            0.0,
            0.66,
            title,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=18,
            fontweight="bold",
            color="#17324d",
            fontfamily=self.font_family,
        )
        if subtitle:
            ax.text(
                0.0,
                0.12,
                subtitle,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=10,
                color="#607080",
                fontfamily=self.font_family,
            )

    def _render_table_section(
        self,
        ax,
        title: str,
        rows,
        *,
        col_labels=("字段", "内容"),
        font_size: float = 9.4,
        bbox=(0.0, 0.0, 1.0, 0.92),
        col_widths=(0.32, 0.64),
        left_aligned: bool = True,
        wrap_text: bool = False,
    ):
        ax.axis("off")
        ax.text(
            0.0,
            1.03,
            title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#17324d",
            fontfamily=self.font_family,
        )
        raw_rows = [list(row) for row in (rows or [])]
        if not raw_rows:
            raw_rows = [["暂无", ""] if len(col_labels) > 1 else ["暂无"]]
        column_count = max(len(col_labels), max((len(row) for row in raw_rows), default=0))
        rendered_labels = list(col_labels) + [""] * max(0, column_count - len(col_labels))
        rendered_rows = []
        row_line_counts = []
        if wrap_text:
            if column_count <= 1:
                wrap_widths = [80]
            elif column_count == 2:
                wrap_widths = [18, 56]
            elif column_count == 3:
                wrap_widths = [12, 10, 44]
            else:
                wrap_widths = [max(10, int(64 / max(column_count, 1)))] * column_count
        else:
            wrap_widths = []

        for row in raw_rows:
            padded_row = list(row) + [""] * max(0, column_count - len(row))
            if wrap_text:
                wrapped_row = [
                    self._wrap_display_text(padded_row[idx], wrap_widths[min(idx, len(wrap_widths) - 1)])
                    for idx in range(column_count)
                ]
                rendered_rows.append(wrapped_row)
                row_line_counts.append(max((cell.count("\n") + 1 for cell in wrapped_row), default=1))
            else:
                rendered_rows.append([self._compact_value(value) for value in padded_row[:column_count]])
                row_line_counts.append(1)

        rendered_col_widths = list(col_widths)
        if len(rendered_col_widths) < column_count:
            fallback_width = rendered_col_widths[-1] if rendered_col_widths else 1.0
            rendered_col_widths.extend([fallback_width] * (column_count - len(rendered_col_widths)))
        table = ax.table(
            cellText=rendered_rows,
            colLabels=rendered_labels,
            cellLoc="left" if left_aligned else "center",
            loc="center",
            bbox=bbox,
            colWidths=rendered_col_widths,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        for (row_idx, col_idx), cell in table.get_celld().items():
            cell.set_edgecolor("#c3ced8")
            cell.set_linewidth(0.6)
            cell.get_text().set_fontfamily(self.font_family)
            if wrap_text:
                cell.get_text().set_wrap(True)
            if row_idx == 0:
                cell.set_facecolor("#e7eef7")
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_color("#17324d")
                cell.get_text().set_ha("center")
            else:
                cell.set_facecolor("#fbfdff" if row_idx % 2 else "#f3f8fd")
                if left_aligned:
                    cell.get_text().set_ha("left")
        if wrap_text and column_count > 0:
            header_units = 1.25
            body_units = [max(1.0, float(line_count)) for line_count in row_line_counts]
            total_units = header_units + sum(body_units)
            available_height = float(bbox[3])
            unit_height = available_height / total_units if total_units > 0 else available_height
            header_height = unit_height * header_units
            for col_idx in range(column_count):
                header_cell = table[(0, col_idx)]
                header_cell.set_height(header_height)
                header_cell.get_text().set_va("center")
            for row_offset, line_count in enumerate(row_line_counts, start=1):
                row_height = unit_height * max(1.0, float(line_count))
                for col_idx in range(column_count):
                    cell = table[(row_offset, col_idx)]
                    cell.set_height(row_height)
                    cell.get_text().set_va("center")
                    if left_aligned:
                        cell.get_text().set_ha("left")
        return table

    def _render_text_section(
        self,
        ax,
        title: str,
        lines,
        *,
        fontsize: float = 9.4,
        empty_text: str = "暂无",
    ) -> None:
        ax.axis("off")
        ax.text(
            0.0,
            1.03,
            title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#17324d",
            fontfamily=self.font_family,
        )
        body = "\n".join(lines or []) if lines else empty_text
        ax.text(
            0.02,
            0.96,
            body,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=fontsize,
            fontfamily=self.font_family,
            linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fbfdff", edgecolor="#d3dde6"),
        )

    def _negative_sampling_mode_rows(self, context) -> list[list[str]]:
        context = dict(context or {})
        params = dict(context.get("dataset_params") or context.get("dataset_summary") or {})
        mode = str(
            params.get("negative_sampling_mode")
            or context.get("negative_sampling_mode")
            or ""
        ).strip().lower()
        multiplier = params.get("negative_distance_multiplier", context.get("negative_distance_multiplier"))
        buffer_radius = params.get("buffer_radius", context.get("buffer_radius"))
        distance_radius = params.get("negative_distance_radius", context.get("negative_distance_radius"))
        applied = bool(params.get("negative_sampling_applied", context.get("negative_sampling_applied", True)))
        no_ore_active = bool(params.get("no_ore_active", context.get("no_ore_active", False)))
        no_ore_point_count = int(params.get("no_ore_point_count", context.get("no_ore_point_count", 0)) or 0)
        spatial_region_active = bool(params.get("spatial_region_active", context.get("spatial_region_active", False)))

        if mode == "far_distance":
            current_label = "远区确认负样本"
            current_desc = "只从距离所有已知矿点都足够远的样本中抽取，灰区直接剔除。"
            threshold_text = ""
            try:
                if buffer_radius not in (None, "") and multiplier not in (None, ""):
                    threshold = float(buffer_radius) * float(multiplier)
                    threshold_text = f"阈值约 {threshold:.1f} 米，缓冲半径 x {float(multiplier):.2f}"
                elif distance_radius not in (None, ""):
                    threshold_text = f"阈值约 {float(distance_radius):.1f} 米"
            except (TypeError, ValueError):
                threshold_text = ""
            if threshold_text:
                current_desc = f"{current_desc} {threshold_text}"
        elif mode == "default":
            current_label = "默认背景随机负样本"
            current_desc = "在缓冲区外的背景样本池中随机抽取，不再按远近分层。"
        elif mode:
            current_label = self._format_negative_sampling_mode_display(mode)
            current_desc = "当前模式的说明暂未定义。"
        else:
            current_label = "暂无"
            current_desc = "暂无"

        if not applied and current_label != "暂无":
            current_desc = f"{current_desc}（当前数据源未重建负样本）"
        if no_ore_active:
            if spatial_region_active:
                current_desc = f"{current_desc} 无矿钻孔/坐标启用，按所在区域分别进入训练区或预测区负样本，重叠区域优先按无矿处理。"
            else:
                current_desc = f"{current_desc} 无矿钻孔/坐标启用，重叠区域优先按无矿处理。"

        return [
            ["默认背景随机负样本", "在缓冲区外的背景样本池中随机抽取，不再按远近分层。"],
            ["远区确认负样本", "只从距离所有已知矿点都足够远的样本中抽取，灰区直接剔除。"],
            [
                "无矿钻孔/坐标",
                (
                    f"启用（{no_ore_point_count} 个点）；按所在区域进入训练区或预测区负样本。"
                    if no_ore_active and spatial_region_active
                    else f"启用（{no_ore_point_count} 个点）"
                    if no_ore_active
                    else "未提供；如有则优先保留为硬负样本。"
                ),
            ],
            ["当前采用模式", f"{current_label}: {current_desc}" if current_label != "暂无" else current_desc],
        ]

    def generate_pdf_report(self, output_path):
        if not self.results:
            raise ValueError("没有可导出的 AutoML 结果。")

        best_result = max(self.results, key=lambda item: item.get("composite_score", 0.0))

        with PdfPages(output_path) as pdf:
            overview_fig = self._build_overview_page()
            pdf.savefig(overview_fig)
            plt.close(overview_fig)

            trial_summary_fig = self._build_trial_summary_page(best_result)
            pdf.savefig(trial_summary_fig)
            plt.close(trial_summary_fig)

            negative_mode_fig = self._build_negative_sampling_mode_page(best_result)
            pdf.savefig(negative_mode_fig)
            plt.close(negative_mode_fig)

            ranking_chart_fig = self._build_ranking_chart_page()
            pdf.savefig(ranking_chart_fig)
            plt.close(ranking_chart_fig)

            ranking_table_fig = self._build_ranking_table_page()
            pdf.savefig(ranking_table_fig)
            plt.close(ranking_table_fig)

            for result in self.results:
                detail_fig = self._build_model_detail_page(result)
                pdf.savefig(detail_fig)
                plt.close(detail_fig)

                history_figure = self._build_optimization_history_page(result)
                if history_figure is not None:
                    pdf.savefig(history_figure)
                    plt.close(history_figure)

    def _build_overview_page(self):
        best_result = max(self.results, key=lambda item: item.get("composite_score", 0.0))
        fig = plt.figure(figsize=(11.69, 8.27))
        gs = fig.add_gridspec(
            5,
            2,
            height_ratios=[0.18, 0.88, 0.82, 1.00, 0.70],
            width_ratios=[1.18, 0.82],
            hspace=0.28,
            wspace=0.18,
        )

        ax_title = fig.add_subplot(gs[0, :])
        subtitle = (
            f"最佳模型: {best_result.get('model_name', '-')} | "
            f"综合评分: {best_result.get('composite_score', 0.0):.4f} | "
            f"模型文件: {Path(best_result.get('model_artifact_path', '')).name or '暂无'}"
        )
        self._render_page_header(ax_title, "自动优化报告概览", subtitle)

        summary_rows = [
            ["最优模型", best_result.get("model_name", "-")],
            ["综合评分", f"{best_result.get('composite_score', 0.0):.4f}"],
            ["验证准确率", self._format_optional(best_result.get("val_accuracy"))],
            ["内部验证矿点检出率", self._format_optional(best_result.get("val_mineral_detection_rate"))],
            ["测试矿点检出率", self._format_optional(best_result.get("test_mineral_detection_rate"))],
            ["最优 Trial 分数", f"{best_result.get('best_score', 0.0):.4f}"],
            [
                "综合评分公式",
                resolve_composite_formula(
                    best_result.get("val_mineral_detection_rate"),
                    best_result.get("val_f1"),
                ),
            ],
            ["矿点检出阈值", f"{DETECTION_THRESHOLD:.2f}"],
        ]
        summary_rows.extend(self._workflow_config_rows(best_result.get("workflow_config")))
        ax_summary = fig.add_subplot(gs[1, :])
        self._render_table_section(
            ax_summary,
            "关键指标",
            summary_rows,
            col_labels=("指标", "数值"),
            font_size=9.5,
            bbox=(0.0, 0.0, 1.0, 0.92),
            col_widths=(0.30, 0.66),
            wrap_text=True,
        )

        ax_trials = fig.add_subplot(gs[2, :])
        self._render_text_section(
            ax_trials,
            "参数试验与矿点信息",
            self._trial_scheme_lines(best_result) + [""] + self._mineral_split_lines(best_result),
            fontsize=8.2,
        )

        ax_dataset = fig.add_subplot(gs[3, 0])
        self._render_table_section(
            ax_dataset,
            "最佳数据方案",
            self._kv_rows(best_result.get("dataset_params")),
            col_labels=("参数", "取值"),
            font_size=8.3,
            bbox=(0.0, 0.0, 1.0, 0.92),
            col_widths=(0.30, 0.66),
            wrap_text=True,
        )

        ax_params = fig.add_subplot(gs[3, 1])
        self._render_table_section(
            ax_params,
            "最佳超参数",
            self._kv_rows(best_result.get("best_params")),
            col_labels=("参数", "取值"),
            font_size=8.3,
            bbox=(0.0, 0.0, 1.0, 0.92),
            col_widths=(0.30, 0.66),
            wrap_text=True,
        )

        advice_lines = [f"• {item}" for item in (flatten_advice(best_result.get("improvement_advice")) or "暂无").split("；") if item.strip()]
        ax_advice = fig.add_subplot(gs[4, :])
        self._render_text_section(ax_advice, "改进建议", advice_lines, fontsize=9.3)

        fig.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
        return fig

    def _build_ranking_chart_page(self):
        models = [item.get("model_name", "") for item in self.results]
        scores = [item.get("composite_score", 0.0) for item in self.results]
        val_accuracy = [item.get("val_accuracy", 0.0) for item in self.results]
        val_detection = [item.get("val_mineral_detection_rate") or 0.0 for item in self.results]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle("模型性能对比", fontsize=15, fontweight="bold", y=0.98)
        chart_specs = [
            ("综合评分", scores, "steelblue"),
            ("验证准确率", val_accuracy, "darkorange"),
            ("内部验证矿点检出率", val_detection, "seagreen"),
        ]
        for axis, (title, values, color) in zip(axes, chart_specs):
            bars = axis.bar(models, values, color=color)
            axis.set_title(title)
            upper = max(1.0, max(values or [0.0]) * 1.15)
            axis.set_ylim(0, upper)
            axis.grid(axis="y", alpha=0.25)
            axis.tick_params(axis="x", rotation=45)
            if hasattr(axis, "bar_label"):
                axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    def _build_ranking_table_page(self):
        table = pd.DataFrame(
            [
                {
                    "模型": item.get("model_name", ""),
                    "综合评分": f"{item.get('composite_score', 0.0):.4f}",
                    "验证准确率": self._format_optional(item.get("val_accuracy")),
                    "内部验证矿点检出率": self._format_optional(item.get("val_mineral_detection_rate")),
                    "测试矿点检出率": self._format_optional(item.get("test_mineral_detection_rate")),
                    "patch_size": item.get("dataset_params", {}).get("patch_size", ""),
                    "patch_stride": item.get("dataset_params", {}).get("patch_stride", ""),
                    "buffer_radius": item.get("dataset_params", {}).get("buffer_radius", ""),
                    "sampling_percentage": item.get("dataset_params", {}).get("sampling_percentage", ""),
                    "balance_ratio": item.get("dataset_params", {}).get("balance_ratio", ""),
                    "模型文件": item.get("model_artifact_path", ""),
                }
                for item in self.results
            ]
        )

        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(2, 1, height_ratios=[0.14, 0.86], hspace=0.1)
        ax_title = fig.add_subplot(gs[0, 0])
        self._render_page_header(ax_title, "模型排名表", f"共 {len(self.results)} 个结果")

        ax = fig.add_subplot(gs[1, 0])
        ax.axis("off")
        columns = table.columns.tolist()
        col_widths = [0.12, 0.09, 0.10, 0.11, 0.11, 0.08, 0.08, 0.08, 0.09, 0.09, 0.11]
        rendered_table = ax.table(
            cellText=table.values.tolist(),
            colLabels=columns,
            cellLoc="center",
            loc="center",
            bbox=(0.01, 0.03, 0.98, 0.90),
            colWidths=col_widths[: len(columns)],
        )
        rendered_table.auto_set_font_size(False)
        rendered_table.set_fontsize(8.8)
        self._apply_table_font(rendered_table)
        for (row_idx, col_idx), cell in rendered_table.get_celld().items():
            cell.set_edgecolor("#c3ced8")
            cell.set_linewidth(0.6)
            if row_idx == 0:
                cell.set_facecolor("#e7eef7")
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_color("#17324d")
            else:
                cell.set_facecolor("#fbfdff" if row_idx % 2 else "#f3f8fd")
        fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.03)
        return fig

    def _build_model_detail_page(self, result):
        zone_exports = (result.get("prediction_artifact") or {}).get("zone_exports", [])
        zone_rows = [
            [
                f"{float(item.get('threshold', 0.0)):.4f}",
                str(item.get("zone_count", "")),
                f"{float(item.get('area_ratio', 0.0)):.4f}",
            ]
            for item in zone_exports
        ] or [["暂无", "", ""]]

        fig = plt.figure(figsize=(11.69, 8.27))
        gs = fig.add_gridspec(
            5,
            2,
            height_ratios=[0.18, 0.88, 1.02, 0.92, 0.88],
            hspace=0.34,
            wspace=0.18,
        )

        subtitle = (
            f"综合评分: {result.get('composite_score', 0.0):.4f} | "
            f"验证准确率: {self._format_optional(result.get('val_accuracy'))} | "
            f"测试准确率: {self._format_optional(result.get('test_accuracy'))}"
        )
        ax_title = fig.add_subplot(gs[0, :])
        self._render_page_header(ax_title, f"模型详情：{result.get('model_name', '-')}", subtitle)

        summary_rows = [
            ["综合评分", f"{result.get('composite_score', 0.0):.4f}"],
            ["验证准确率", self._format_optional(result.get("val_accuracy"))],
            ["验证F1", self._format_optional(result.get("val_f1"))],
            ["内部验证矿点检出率", self._format_optional(result.get("val_mineral_detection_rate"))],
            ["测试准确率", self._format_optional(result.get("test_accuracy"))],
            ["测试F1", self._format_optional(result.get("test_f1"))],
            ["测试矿点检出率", self._format_optional(result.get("test_mineral_detection_rate"))],
            ["最优 Trial 分数", f"{result.get('best_score', 0.0):.4f}"],
            ["训练时间(秒)", f"{result.get('training_time', 0.0):.1f}"],
        ]
        summary_rows.extend(self._workflow_config_rows(result.get("workflow_config")))
        ax_summary = fig.add_subplot(gs[1, :])
        self._render_table_section(
            ax_summary,
            "关键指标",
            summary_rows,
            col_labels=("指标", "数值"),
            font_size=9.3,
            bbox=(0.0, 0.0, 1.0, 0.92),
            col_widths=(0.30, 0.66),
        )

        ax_dataset = fig.add_subplot(gs[2, 0])
        self._render_table_section(
            ax_dataset,
            "数据方案",
            self._kv_rows(result.get("dataset_params")),
            col_labels=("参数", "取值"),
            font_size=8.9,
            bbox=(0.0, 0.0, 1.0, 0.92),
            col_widths=(0.34, 0.62),
        )

        ax_params = fig.add_subplot(gs[2, 1])
        self._render_table_section(
            ax_params,
            "最佳超参数",
            self._kv_rows(result.get("best_params")),
            col_labels=("参数", "取值"),
            font_size=8.9,
            bbox=(0.0, 0.0, 1.0, 0.92),
            col_widths=(0.34, 0.62),
        )

        prediction_rows = zone_rows
        ax_prediction = fig.add_subplot(gs[3, 0])
        self._render_table_section(
            ax_prediction,
            "预测区统计",
            prediction_rows,
            col_labels=("阈值", "区块数", "面积比例"),
            font_size=8.9,
            bbox=(0.0, 0.0, 1.0, 0.92),
            col_widths=(0.28, 0.28, 0.34),
            left_aligned=False,
        )

        boundary_lines = [f"• {item}" for item in result.get("search_boundary_hits", [])] or ["暂无"]
        ax_boundary = fig.add_subplot(gs[3, 1])
        self._render_text_section(ax_boundary, "搜索边界命中", boundary_lines, fontsize=9.0)

        advice_lines = [
            "结果文件:",
            f"• 模型文件: {result.get('model_artifact_path', '') or '暂无'}",
            f"• 重建清单: {result.get('rebuild_manifest_path', '') or '暂无'}",
            "",
            "改进建议:",
        ]
        advice_items = [item for item in (result.get("improvement_advice") or []) if str(item).strip()]
        advice_lines.extend([f"• {item}" for item in advice_items] or ["• 暂无"])
        ax_footer = fig.add_subplot(gs[4, :])
        self._render_text_section(ax_footer, "结果与建议", advice_lines, fontsize=9.0)

        fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.03)
        return fig

    def _build_optimization_history_page(self, result):
        history = result.get("optimization_history") or []
        if not history:
            return None

        trials = [item.get("trial", 0) + 1 for item in history]
        composite_scores = [item.get("composite_score", item.get("value", 0.0)) for item in history]
        val_accuracy = [item.get("val_accuracy", 0.0) for item in history]
        val_detection = [item.get("val_mineral_detection_rate", 0.0) for item in history]

        fig, ax = plt.subplots(figsize=(11.69, 6.5))
        ax.set_title(f"{result.get('model_name', '-')} 优化历史", fontsize=14, fontweight="bold", pad=12)
        ax.plot(trials, composite_scores, marker="o", label="综合评分")
        ax.plot(trials, val_accuracy, marker="s", label="验证准确率")
        ax.plot(trials, val_detection, marker="^", label="内部验证矿点检出率")
        ax.set_xlabel("Trial")
        ax.set_ylabel("分数")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig

    def _format_optional(self, value):
        if value is None:
            return ""
        return f"{float(value):.4f}"
