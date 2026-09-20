import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import argparse
import os
import sys
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(CURRENT_DIR)
COMMON_DIR = os.path.join(CODE_ROOT, "common")
for path in (CODE_ROOT, COMMON_DIR):
    if path not in sys.path:
        sys.path.append(path)

from model_name_utils import get_model_display_name, normalize_model_key

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def load_data(predictions_file):
    """加载预测结果数据"""
    print(f"加载预测结果: {predictions_file}")
    with h5py.File(predictions_file, 'r') as f:
        positions = f['positions'][:]  # 位置信息
        predictions = f['predictions'][:]  # 预测标签
        confidences = f['confidences'][:]  # 预测置信度
        labels = f['labels'][:]  # 真实标签
        mineral_points = f['mineral_points'][:] if 'mineral_points' in f else None
        metadata = {key: f.attrs[key] for key in f.attrs.keys()}
    
    print(f"数据加载完成，共 {len(positions)} 个点")
    print(f"真实正样本数量: {np.sum(labels == 1)}")
    print(f"预测正样本数量: {np.sum(predictions == 1)}")
    
    return positions, predictions, confidences, labels, metadata, mineral_points


def _positions_to_geo(positions, metadata):
    if positions is None:
        return positions
    arr = np.asarray(positions, dtype=np.float64)
    if arr.size == 0:
        return arr
    if arr.ndim == 1:
        if arr.shape[0] < 2:
            return arr
        arr = arr.reshape(-1, 2)
    meta = metadata or {}
    try:
        x_min = float(meta["x_min"])
        x_max = float(meta["x_max"])
        y_min = float(meta["y_min"])
        y_max = float(meta["y_max"])
        nx = int(meta.get("nx", 0) or 0)
        ny = int(meta.get("ny", 0) or 0)
    except (KeyError, TypeError, ValueError):
        return arr
    if nx <= 1 or ny <= 1:
        return arr
    x_bound = max(nx - 1, 1) * 1.5
    y_bound = max(ny - 1, 1) * 1.5
    if np.nanmax(np.abs(arr[:, 0])) > x_bound or np.nanmax(np.abs(arr[:, 1])) > y_bound:
        return arr
    x_step = (x_max - x_min) / max(nx - 1, 1)
    y_step = (y_max - y_min) / max(ny - 1, 1)
    geo_x = x_min + arr[:, 0] * x_step
    geo_y = y_max - arr[:, 1] * y_step
    return np.column_stack([geo_x, geo_y]).astype(np.float64, copy=False)


def _axis_edges(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        half_step = 0.5
        return np.array([values[0] - half_step, values[0] + half_step], dtype=np.float64)

    mids = (values[:-1] + values[1:]) / 2.0
    first = values[0] - (values[1] - values[0]) / 2.0
    last = values[-1] + (values[-1] - values[-2]) / 2.0
    return np.concatenate([[first], mids, [last]]).astype(np.float64, copy=False)


def _confidence_grid_from_cut_positions(positions, confidences):
    """Rebuild a probability raster from regular cut-window center coordinates."""
    coords = np.asarray(positions, dtype=np.float64)
    values = np.asarray(confidences, dtype=np.float64).flatten()
    if coords.ndim != 2 or coords.shape[1] < 2 or len(coords) != len(values):
        return None

    rounded_x = np.round(coords[:, 0], 8)
    rounded_y = np.round(coords[:, 1], 8)
    x_keys, x_inverse = np.unique(rounded_x, return_inverse=True)
    y_keys, y_inverse = np.unique(rounded_y, return_inverse=True)
    if len(x_keys) * len(y_keys) != len(coords):
        return None

    x_values = np.array([coords[rounded_x == key, 0].mean() for key in x_keys], dtype=np.float64)
    y_values = np.array([coords[rounded_y == key, 1].mean() for key in y_keys], dtype=np.float64)
    x_order = np.argsort(x_values)
    y_order = np.argsort(y_values)
    x_rank = np.empty_like(x_order)
    y_rank = np.empty_like(y_order)
    x_rank[x_order] = np.arange(len(x_order))
    y_rank[y_order] = np.arange(len(y_order))

    grid = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)
    for idx, confidence in enumerate(values):
        row = y_rank[y_inverse[idx]]
        col = x_rank[x_inverse[idx]]
        if np.isfinite(grid[row, col]):
            return None
        grid[row, col] = confidence

    if np.isnan(grid).any():
        return None

    return x_values[x_order], y_values[y_order], grid


MAP_SPATIAL_MODES = ("center", "window_average", "window_max")
MAP_GENERATION_MODES = (
    "center",
    "window_average",
    "window_max",
    "center_rank",
    "window_average_rank",
    "window_max_rank",
)
MAP_GENERATION_MODE_LABELS = {
    "center": "中心点原值",
    "window_average": "窗口平均融合",
    "window_max": "窗口最大融合",
    "center_rank": "中心点百分位秩",
    "window_average_rank": "窗口平均+百分位秩",
    "window_max_rank": "窗口最大+百分位秩",
}


def resolve_map_generation_mode(mode):
    """Parse map mode into (spatial_mode, apply_percentile_rank)."""
    raw = str(mode or "center").strip().lower().replace("-", "_")
    aliases = {
        "rank": ("center", True),
        "percentile": ("center", True),
        "percentile_rank": ("center", True),
        "center_percentile": ("center", True),
        "window_average_percentile": ("window_average", True),
        "window_max_percentile": ("window_max", True),
    }
    if raw in aliases:
        return aliases[raw]
    if raw.endswith("_rank"):
        spatial = raw[: -len("_rank")]
        if spatial in MAP_SPATIAL_MODES:
            return spatial, True
    if raw in MAP_SPATIAL_MODES:
        return raw, False
    return "center", False


def percentile_rank_transform(values):
    """Map finite scores to empirical percentile ranks in [0, 1] (average-rank ties)."""
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    flat = arr.reshape(-1)
    mask = np.isfinite(flat)
    n = int(np.sum(mask))
    if n == 0:
        return out
    x = flat[mask]
    if n == 1:
        ranks01 = np.array([0.5], dtype=np.float64)
    else:
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        sorted_x = x[order]
        i = 0
        while i < n:
            j = i + 1
            while j < n and sorted_x[j] == sorted_x[i]:
                j += 1
            # 1-based average rank over the tied block
            avg_rank = 0.5 * ((i + 1) + j)
            ranks[order[i:j]] = avg_rank
            i = j
        ranks01 = (ranks - 1.0) / float(n - 1)
    flat_out = out.reshape(-1)
    flat_out[mask] = ranks01
    return out


def apply_map_generation_to_grid(grid, metadata, mode):
    """Apply spatial window fusion and optional percentile-rank display transform."""
    spatial, use_rank = resolve_map_generation_mode(mode)
    out = _window_fused_grid_from_regular_grid(grid, metadata, spatial)
    if use_rank:
        out = percentile_rank_transform(out)
    return out


def apply_map_generation_to_values(values, mode):
    """Optional percentile-rank transform for a 1-D score vector (spatial fusion already done)."""
    _spatial, use_rank = resolve_map_generation_mode(mode)
    arr = np.asarray(values, dtype=np.float64)
    if use_rank:
        return percentile_rank_transform(arr)
    return arr


def _window_fused_grid_from_regular_grid(grid, metadata, mode):
    """Fuse window-level scores back to raster cells by averaging or taking max.

    ``mode`` may be a full map-generation mode (including ``*_rank``); rank is applied
    after spatial fusion. Prefer ``apply_map_generation_to_grid`` for new callers.
    """
    spatial, use_rank = resolve_map_generation_mode(mode)
    if spatial not in {"window_average", "window_max"}:
        out = np.asarray(grid, dtype=np.float64)
        return percentile_rank_transform(out) if use_rank else out

    meta = metadata or {}
    try:
        window_width = int(meta.get("window_width", 1) or 1)
        window_height = int(meta.get("window_height", 1) or 1)
    except (TypeError, ValueError):
        out = np.asarray(grid, dtype=np.float64)
        return percentile_rank_transform(out) if use_rank else out

    if window_width <= 1 and window_height <= 1:
        out = np.asarray(grid, dtype=np.float64)
        return percentile_rank_transform(out) if use_rank else out

    values = np.asarray(grid, dtype=np.float64)
    ny, nx = values.shape
    if nx == 0 or ny == 0:
        return values

    # Prediction positions are saved as window centers; fill a centered window.
    left = max(window_width // 2, 0)
    right = max(window_width - left, 1)
    lower = max(window_height // 2, 0)
    upper = max(window_height - lower, 1)

    if spatial == "window_average":
        fused = np.zeros_like(values, dtype=np.float64)
        counts = np.zeros_like(values, dtype=np.float64)
    else:
        fused = np.full_like(values, -np.inf, dtype=np.float64)

    for row in range(ny):
        row_start = max(0, row - lower)
        row_end = min(ny, row + upper)
        for col in range(nx):
            score = values[row, col]
            if not np.isfinite(score):
                continue
            col_start = max(0, col - left)
            col_end = min(nx, col + right)
            if spatial == "window_average":
                fused[row_start:row_end, col_start:col_end] += score
                counts[row_start:row_end, col_start:col_end] += 1.0
            else:
                current = fused[row_start:row_end, col_start:col_end]
                fused[row_start:row_end, col_start:col_end] = np.maximum(current, score)

    if spatial == "window_average":
        valid = counts > 0
        fused[valid] /= counts[valid]
        fused[~valid] = np.nan
    else:
        fused[~np.isfinite(fused)] = np.nan

    fused = np.clip(fused, 0.0, 1.0)
    if use_rank:
        return percentile_rank_transform(fused)
    return fused


def _grid_values_for_positions(positions, x_values, y_values, grid):
    """Read grid values back in the same order as positions."""
    coords = np.asarray(positions, dtype=np.float64)
    result = np.full(len(coords), np.nan, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        return result

    x_lookup = {round(float(value), 8): idx for idx, value in enumerate(x_values)}
    y_lookup = {round(float(value), 8): idx for idx, value in enumerate(y_values)}
    for idx, pos in enumerate(coords):
        col = x_lookup.get(round(float(pos[0]), 8))
        row = y_lookup.get(round(float(pos[1]), 8))
        if row is not None and col is not None:
            result[idx] = grid[row, col]
    return result


def _fused_probability_columns(positions, confidences, metadata):
    """Return window-average and window-max probabilities aligned to CSV rows."""
    empty = np.full(len(confidences), np.nan, dtype=np.float64)
    rebuilt = _confidence_grid_from_cut_positions(positions, confidences)
    if rebuilt is None:
        return empty, empty

    x_values, y_values, grid = rebuilt
    average_grid = _window_fused_grid_from_regular_grid(grid, metadata, "window_average")
    max_grid = _window_fused_grid_from_regular_grid(grid, metadata, "window_max")
    average_values = _grid_values_for_positions(positions, x_values, y_values, average_grid)
    max_values = _grid_values_for_positions(positions, x_values, y_values, max_grid)
    return average_values, max_values


def save_predictions_to_dat(positions, predictions, confidences, output_dir, model_name, metadata=None):
    """将预测结果保存为dat文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    display_name = get_model_display_name(normalize_model_key(model_name))
    safe_model_name = display_name.replace("/", "-").replace("\\", "-").strip()
    output_file = os.path.join(output_dir, f'predictions_{safe_model_name}_{timestamp}.dat')
    
    # 创建表头
    header = "X坐标\tY坐标\t预测标签\t置信度\n"
    
    positions = _positions_to_geo(positions, metadata)

    # 保存数据（直接使用1和-1作为标签，坐标保持地理坐标）
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header)
        for pos, pred, conf in zip(positions, predictions, confidences):
            f.write(f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pred}\t{conf:.6f}\n")
    
    print(f"预测结果已保存到: {output_file}")
    return output_file


def save_predictions_to_csv(positions, predictions, confidences, labels, output_file, metadata=None):
    """将成矿潜力概率图对应的点位概率导出为 CSV。

    除中心/窗口平均/窗口最大概率外，同步写出三种百分位秩列，便于与成图方式对照。
    """
    positions = _positions_to_geo(positions, metadata)
    predictions = np.asarray(predictions).reshape(-1)
    confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1) if labels is not None else np.full(len(confidences), np.nan)
    if len(positions) != len(confidences) or len(predictions) != len(confidences):
        raise ValueError("positions、predictions、confidences 数量不一致，无法导出 CSV")
    if len(labels) != len(confidences):
        labels = np.full(len(confidences), np.nan)

    average_probs, max_probs = _fused_probability_columns(positions, confidences, metadata)
    center_rank = percentile_rank_transform(confidences).reshape(-1)
    average_rank = percentile_rank_transform(average_probs).reshape(-1)
    max_rank = percentile_rank_transform(max_probs).reshape(-1)

    import csv

    def _fmt_score(value):
        return "" if not np.isfinite(value) else f"{float(value):.8f}"

    with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "X坐标",
            "Y坐标",
            "预测标签",
            "成矿潜力概率",
            "窗口平均融合概率",
            "窗口最大融合概率",
            "中心点百分位秩",
            "窗口平均百分位秩",
            "窗口最大百分位秩",
            "真实标签",
        ])
        for (
            pos,
            pred,
            conf,
            avg_prob,
            max_prob,
            c_rank,
            a_rank,
            m_rank,
            label,
        ) in zip(
            positions,
            predictions,
            confidences,
            average_probs,
            max_probs,
            center_rank,
            average_rank,
            max_rank,
            labels,
        ):
            label_value = "" if isinstance(label, float) and np.isnan(label) else int(label)
            writer.writerow([
                f"{float(pos[0]):.8f}",
                f"{float(pos[1]):.8f}",
                int(pred),
                _fmt_score(conf),
                _fmt_score(avg_prob),
                _fmt_score(max_prob),
                _fmt_score(c_rank),
                _fmt_score(a_rank),
                _fmt_score(m_rank),
                label_value,
            ])
    print(f"成矿潜力概率 CSV 已保存到: {output_file}")
    return output_file


def build_prediction_score_grid(positions, confidences, metadata, mode="center"):
    """从规则切窗中心点重建栅格，并按成图方式做窗口融合/百分位秩。

    Returns
    -------
    (x_values, y_values, grid) or None
        无法构成规则网格时返回 None。
    """
    geo_positions = _positions_to_geo(positions, metadata)
    built = _confidence_grid_from_cut_positions(geo_positions, confidences)
    if built is None:
        return None
    x_values, y_values, grid = built
    fused = apply_map_generation_to_grid(grid, metadata, mode)
    return x_values, y_values, fused


def export_prediction_surfer_grids(
    positions,
    confidences,
    metadata,
    output_dir,
    model_name,
    modes=None,
):
    """导出选定成图方式对应的 Surfer DSAA .grd。

    Parameters
    ----------
    modes : sequence of str, optional
        默认导出中心 / 窗口平均 / 窗口最大。也可传入 ``*_rank`` 模式。

    Returns
    -------
    list of str
        成功写出的文件路径。
    """
    selected = list(modes) if modes else list(MAP_SPATIAL_MODES)
    if not selected:
        raise ValueError("未选择任何导出模式")

    data_prep_dir = os.path.join(CODE_ROOT, "数据准备")
    if data_prep_dir not in sys.path:
        sys.path.append(data_prep_dir)

    try:
        from geochem_kriging import write_surfer_dsaa
    except ImportError:
        from 数据准备.geochem_kriging import write_surfer_dsaa  # type: ignore

    os.makedirs(output_dir, exist_ok=True)
    stem = str(model_name or "prediction").strip() or "prediction"
    written = []
    for mode in selected:
        spatial, _ = resolve_map_generation_mode(mode)
        built = build_prediction_score_grid(positions, confidences, metadata, mode=mode)
        if built is None:
            raise ValueError(
                "预测点无法构成规则网格，无法导出 Surfer GRD。"
                "请确认预测坐标来自规则切窗中心点。"
            )
        x_values, y_values, grid = built
        safe_mode = str(mode).strip().lower().replace("-", "_")
        out_path = os.path.join(output_dir, f"{stem}_{safe_mode}.grd")
        write_surfer_dsaa(
            out_path,
            np.asarray(grid, dtype=np.float64),
            x_min=float(np.min(x_values)),
            x_max=float(np.max(x_values)),
            y_min=float(np.min(y_values)),
            y_max=float(np.max(y_values)),
        )
        written.append(out_path)
        label = MAP_GENERATION_MODE_LABELS.get(safe_mode) or MAP_GENERATION_MODE_LABELS.get(spatial, safe_mode)
        print(f"Surfer GRD 已保存 ({label}): {out_path}")
    return written


def plot_confidence_interpolation(
    adjusted_positions,
    confidences,
    output_dir,
    model_name,
    grid_spacing=10.0,
    mineral_points=None,
    metadata=None,
    map_generation_mode="center",
):
    """基于切割后的窗口中心坐标生成成矿潜力概率图。"""
    if confidences is None or len(confidences) == 0:
        raise ValueError("confidences 不能为空")

    adjusted_positions = np.asarray(adjusted_positions, dtype=np.float64)
    confidences = np.asarray(confidences).flatten()
    if len(confidences) != len(adjusted_positions):
        raise ValueError("confidences 与 positions 数量不一致，无法插值")

    x = adjusted_positions[:, 0]
    y = adjusted_positions[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    display_name = get_model_display_name(normalize_model_key(model_name))
    safe_model_name = display_name.replace("/", "-").replace("\\", "-").strip()
    spatial_mode, use_rank = resolve_map_generation_mode(map_generation_mode)
    map_generation_mode = f"{spatial_mode}_rank" if use_rank else spatial_mode
    mode_label = MAP_GENERATION_MODE_LABELS.get(map_generation_mode, map_generation_mode)
    confidence_path = os.path.join(
        output_dir,
        f'confidence_map_{safe_model_name}_{map_generation_mode}_{timestamp}.png',
    )

    if x_range == 0:
        x_range = grid_spacing
    if y_range == 0:
        y_range = grid_spacing

    aspect_ratio = x_range / y_range if y_range else 1.0
    base_size = 6.0
    fig_width = np.clip(base_size * aspect_ratio ** 0.5, 4.0, 12.0)
    fig_height = np.clip(base_size / (aspect_ratio ** 0.5 if aspect_ratio else 1.0), 4.0, 12.0)

    plt.figure(figsize=(fig_width, fig_height))
    rebuilt_grid = _confidence_grid_from_cut_positions(adjusted_positions, confidences)
    if rebuilt_grid is not None:
        x_values, y_values, grid_Z = rebuilt_grid
        grid_Z = apply_map_generation_to_grid(grid_Z, metadata, map_generation_mode)
        x_edges = _axis_edges(x_values)
        y_edges = _axis_edges(y_values)
        mesh_x, mesh_y = np.meshgrid(x_edges, y_edges)
        pcm = plt.pcolormesh(
            mesh_x,
            mesh_y,
            grid_Z,
            shading='auto',
            cmap='jet',
            vmin=0.0,
            vmax=1.0,
        )
        x_min, x_max = x_edges[0], x_edges[-1]
        y_min, y_max = y_edges[0], y_edges[-1]
        print(
            f"使用成图方式重建概率栅格: {len(x_values)}×{len(y_values)}, "
            f"方式={mode_label}"
        )
    else:
        grid_spacing = float(grid_spacing) if grid_spacing else 10.0
        if grid_spacing <= 0:
            print(f"警告: 网格间距 {grid_spacing} 无效，将回退到默认 10.0")
            grid_spacing = 10.0
        grid_res_x = max(2, int(np.ceil(x_range / grid_spacing)) + 1)
        grid_res_y = max(2, int(np.ceil(y_range / grid_spacing)) + 1)
        grid_x = np.linspace(x_min, x_max, grid_res_x)
        grid_y = np.linspace(y_min, y_max, grid_res_y)
        grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
        points = np.column_stack((x, y))
        display_confidences = apply_map_generation_to_values(confidences, map_generation_mode)
        grid_Z = griddata(points, display_confidences, (grid_X, grid_Y), method="cubic")
        if np.all(np.isnan(grid_Z)):
            raise RuntimeError("插值结果全部为无效值，请检查输入数据分布是否覆盖足够区域")
        nan_mask = np.isnan(grid_Z)
        if np.any(nan_mask):
            nearest_Z = griddata(points, display_confidences, (grid_X, grid_Y), method="nearest")
            grid_Z[nan_mask] = nearest_Z[nan_mask]
        pcm = plt.pcolormesh(
            grid_X,
            grid_Y,
            grid_Z,
            shading='auto',
            cmap='jet',
            vmin=0.0,
            vmax=1.0,
        )
        print(
            f"切割坐标不是完整规则网格，使用插值兜底: Δ={grid_spacing:.2f}, "
            f"分辨率=({grid_res_x}×{grid_res_y}), 方式={mode_label}"
        )

    if mineral_points is not None and len(mineral_points) > 0:
        mineral_points = np.asarray(mineral_points, dtype=np.float64)
        prob_x_min, prob_x_max = x_min, x_max
        prob_y_min, prob_y_max = y_min, y_max
        outside_mask = (
            (mineral_points[:, 0] < prob_x_min)
            | (mineral_points[:, 0] > prob_x_max)
            | (mineral_points[:, 1] < prob_y_min)
            | (mineral_points[:, 1] > prob_y_max)
        )
        if np.any(outside_mask):
            print(f"提示: {int(np.sum(outside_mask))} 个真实矿点位于预测栅格覆盖范围之外，已扩展坐标轴显示。")
        x_min = min(x_min, float(np.min(mineral_points[:, 0])))
        x_max = max(x_max, float(np.max(mineral_points[:, 0])))
        y_min = min(y_min, float(np.min(mineral_points[:, 1])))
        y_max = max(y_max, float(np.max(mineral_points[:, 1])))

        plt.scatter(
            mineral_points[:, 0],
            mineral_points[:, 1],
            facecolors='#ff2d55',
            edgecolors='white',
            s=58,
            linewidths=1.2,
            marker='o',
            label='真实矿点',
            zorder=5,
        )
        plt.legend(loc='best', fontsize=9)

    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    title = '成矿潜力百分位秩图' if use_rank else '成矿潜力概率图'
    plt.title(f'{title}\n({mode_label})', fontsize=16, weight='bold')
    plt.grid(True, linestyle='--', linewidth=0.5, color='white', alpha=0.6)
    plt.gca().set_facecolor('#0b1d51')

    cbar = plt.colorbar(pcm, fraction=0.046, pad=0.04)
    cbar.set_label('成矿潜力百分位秩' if use_rank else '成矿潜力概率', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    plt.savefig(confidence_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"置信度插值图已保存到: {confidence_path}")
    return confidence_path

def plot_labels_comparison(
    positions,
    predictions,
    labels,
    output_dir,
    model_name,
    confidences=None,
    include_confidence_map=True,
    grid_spacing=10.0,
    metadata=None,
    mineral_points=None,
    map_generation_mode="center",
):
    """绘制插值置信度热力图"""
    adjusted_positions = _positions_to_geo(positions, metadata)

    plot_paths = {}

    if include_confidence_map and confidences is not None:
        try:
            confidence_path = plot_confidence_interpolation(
                adjusted_positions,
                confidences,
                output_dir,
                model_name,
                grid_spacing=grid_spacing,
                mineral_points=mineral_points,
                metadata=metadata,
                map_generation_mode=map_generation_mode,
            )
        except Exception as exc:
            print(f"置信度插值图生成失败: {exc}")
        else:
            plot_paths["confidence"] = confidence_path
    else:
        print("警告: 未提供置信度数据，无法绘制插值图")

    # 打印预测统计信息方便排查
    true_positives = np.sum((predictions == 1) & (labels == 1))
    false_positives = np.sum((predictions == 1) & (labels == -1))
    true_negatives = np.sum((predictions == -1) & (labels == -1))
    false_negatives = np.sum((predictions == -1) & (labels == 1))

    print("\n预测统计信息:")
    print(f"真正例 (TP): {true_positives}")
    print(f"假正例 (FP): {false_positives}")
    print(f"真负例 (TN): {true_negatives}")
    print(f"假负例 (FN): {false_negatives}")

    accuracy = (true_positives + true_negatives) / len(labels)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n模型性能指标:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

    return plot_paths

def main():
    parser = argparse.ArgumentParser(description='绘制真实标签和预测标签的对比图')
    parser.add_argument('--predictions', type=str, default='predictions/predictions_cnnt.h5',
                      help='预测结果文件路径 (H5格式)')
    parser.add_argument('--model-name', type=str, default='cnnt',
                      help='模型名称')
    parser.add_argument('--output-dir', type=str, default='label_plots',
                      help='输出目录')
    parser.add_argument('--map-generation-mode', type=str, default='center',
                      choices=list(MAP_GENERATION_MODES),
                      help='成矿潜力图生成方式（含百分位秩变换）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 加载数据
        positions, predictions, confidences, labels, metadata, mineral_points = load_data(args.predictions)
        
        # 保存预测结果到dat文件
        save_predictions_to_dat(positions, predictions, confidences, args.output_dir, args.model_name, metadata=metadata)
        
        # 绘制对比图
        plot_labels_comparison(
            positions,
            predictions,
            labels,
            args.output_dir,
            args.model_name,
            confidences=confidences,
            metadata=metadata,
            mineral_points=mineral_points,
            map_generation_mode=args.map_generation_mode,
        )
        
        print("\n程序执行完成!")
        
    except Exception as e:
        print(f"\n程序执行出错：")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        import traceback
        print("\n详细错误信息:")
        traceback.print_exc()

if __name__ == '__main__':
    main() 
