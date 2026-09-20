from __future__ import annotations

import os
import sys
import hashlib
from typing import Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(CURRENT_DIR)
COMMON_DIR = os.path.join(CODE_ROOT, "common")
for path in (CODE_ROOT, COMMON_DIR):
    if path not in sys.path:
        sys.path.append(path)

from feature_channel_utils import infer_h5_channel_names, subset_samples_by_channels
from reviewer_protocol import merge_mineralization_units

try:
    from feature.patch_creator import PatchCreator
except Exception:  # pragma: no cover
    try:
        from ..feature.patch_creator import PatchCreator  # type: ignore
    except Exception:  # pragma: no cover
        PatchCreator = None


SPATIAL_MINERAL_SPLIT_MODES = {
    "spatial_cluster",
    "spatial_hard",
    "spatial_stratified",
    "spatial_cluster_holdout_cv",
    "leave_one_camp",
    "leave_one_fault",
    "variogram_block_cv",
}

try:
    from spatial_validation_splits import (
        assign_coords_to_nearest_fault,
        assign_coords_to_variogram_blocks,
        auto_assign_faults_from_lines,
        load_deposit_camp_assignment,
        load_deposit_fault_assignment,
        load_fault_lines,
        map_assignment_to_minerals,
        apply_whole_basin_split,
        read_basin_grd,
        sample_basin_ids_at_coords,
        split_leave_one_camp,
        split_leave_one_fault,
        split_variogram_block_cv,
    )
except Exception:  # pragma: no cover
    try:
        from .spatial_validation_splits import (  # type: ignore
            assign_coords_to_nearest_fault,
            assign_coords_to_variogram_blocks,
            auto_assign_faults_from_lines,
            apply_whole_basin_split,
            load_deposit_camp_assignment,
            load_deposit_fault_assignment,
            load_fault_lines,
            map_assignment_to_minerals,
            read_basin_grd,
            sample_basin_ids_at_coords,
            split_leave_one_camp,
            split_leave_one_fault,
            split_variogram_block_cv,
        )
    except Exception:  # pragma: no cover
        split_leave_one_camp = None
        split_leave_one_fault = None
        split_variogram_block_cv = None
        load_deposit_fault_assignment = None
        load_deposit_camp_assignment = None
        load_fault_lines = None
        map_assignment_to_minerals = None
        auto_assign_faults_from_lines = None
        assign_coords_to_nearest_fault = None
        assign_coords_to_variogram_blocks = None
        apply_whole_basin_split = None
        read_basin_grd = None
        sample_basin_ids_at_coords = None


def _apply_basin_split_if_available(mineral_split: dict, ogr_options: Optional[dict]) -> dict:
    """Apply whole-catchment train/test reassignment when basin GRD is configured."""
    if mineral_split is None:
        return mineral_split
    opts = ogr_options or {}
    basin_path = str(opts.get("basin_grd_path", "") or "").strip()
    if not basin_path or not os.path.exists(basin_path) or read_basin_grd is None:
        return mineral_split
    try:
        basin_grid = read_basin_grd(basin_path)
        out = _enrich_mineral_split_basins(mineral_split, basin_grid)
        train_df = out.get("train")
        test_df = out.get("test")
        n_train = int(len(train_df)) if train_df is not None else 0
        n_test = int(len(test_df)) if test_df is not None else 0
        print(
            f"汇水域整盆划分(训练优先-未标记): test_basins={out.get('test_basin_ids')} "
            f"train_basins={len(out.get('train_basin_ids') or [])} "
            f"train_minerals={n_train} test_minerals={n_test}"
        )
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: 汇水域整盆划分失败，沿用 leave-one 矿点划分: {exc}")
        return mineral_split


def _indices_sha256(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _patch_footprint_embargo_mask(
    coordinates: np.ndarray,
    candidate_indices: np.ndarray,
    held_out_indices: np.ndarray,
    reference_patch_size: int,
) -> np.ndarray:
    """Flag candidate centers whose patches can share a raster cell with held-out centers."""
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    held_out_indices = np.asarray(held_out_indices, dtype=np.int64).reshape(-1)
    if len(candidate_indices) == 0 or len(held_out_indices) == 0:
        return np.zeros(len(candidate_indices), dtype=bool)
    radius_cells = max(int(reference_patch_size) - 1, 0)
    if radius_cells <= 0:
        return np.zeros(len(candidate_indices), dtype=bool)
    raw = np.asarray(coordinates, dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(raw[held_out_indices, :2])
        distances, _ = tree.query(raw[candidate_indices, :2], k=1, p=np.inf)
        return np.asarray(distances <= float(radius_cells) + 1e-9, dtype=bool)
    except Exception:
        held = raw[held_out_indices, :2]
        out = np.zeros(len(candidate_indices), dtype=bool)
        for offset, dataset_index in enumerate(candidate_indices.tolist()):
            delta = np.max(np.abs(held - raw[int(dataset_index), :2]), axis=1)
            out[offset] = bool(np.any(delta <= float(radius_cells) + 1e-9))
        return out


def _grid_chebyshev_proximity_mask(
    candidate_coords: np.ndarray,
    held_out_coords: np.ndarray,
    radius_grid_intervals: int,
) -> np.ndarray:
    candidates = np.asarray(candidate_coords, dtype=np.float64)[:, :2]
    held_out = np.asarray(held_out_coords, dtype=np.float64)[:, :2]
    radius = max(int(radius_grid_intervals), 0)
    if len(candidates) == 0 or len(held_out) == 0 or radius <= 0:
        return np.zeros(len(candidates), dtype=bool)

    def _axis_step(values):
        unique = np.unique(np.asarray(values, dtype=np.float64))
        diffs = np.diff(unique)
        diffs = diffs[np.isfinite(diffs) & (diffs > 1e-12)]
        return float(np.min(diffs)) if len(diffs) else 1.0

    scale = np.asarray(
        [_axis_step(np.concatenate((candidates[:, axis], held_out[:, axis]))) for axis in (0, 1)],
        dtype=np.float64,
    )
    scaled_candidates = candidates / scale[None, :]
    scaled_held_out = held_out / scale[None, :]
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(scaled_held_out).query(
            scaled_candidates,
            k=1,
            p=np.inf,
        )
        return np.asarray(distances <= float(radius) + 1e-9, dtype=bool)
    except Exception:
        out = np.zeros(len(candidates), dtype=bool)
        for index, xy in enumerate(scaled_candidates):
            out[index] = bool(
                np.any(np.max(np.abs(scaled_held_out - xy), axis=1) <= float(radius) + 1e-9)
            )
        return out


def _sample_to_fixed_area_indices(
    sampled_dataset_indices: np.ndarray,
    area_dataset_indices: np.ndarray,
) -> np.ndarray:
    lookup = {
        int(dataset_index): int(area_index)
        for area_index, dataset_index in enumerate(
            np.asarray(area_dataset_indices, dtype=np.int64).reshape(-1).tolist()
        )
    }
    mapped = np.asarray(
        [lookup.get(int(dataset_index), -1) for dataset_index in sampled_dataset_indices],
        dtype=np.int64,
    )
    if np.any(mapped < 0):
        raise ValueError("Sampled training centers are not contained in the fixed inner-evaluation area.")
    return mapped


def _area_mineral_ids(
    area_coordinates: np.ndarray,
    minerals: pd.DataFrame,
    metadata: dict,
) -> np.ndarray:
    ids = np.full(len(area_coordinates), -1, dtype=np.int64)
    if minerals is None or len(minerals) == 0 or len(area_coordinates) == 0:
        return ids
    center_indices, _ = _assign_minerals_to_center_windows(
        area_coordinates,
        minerals,
        metadata,
    )
    for mineral_id, center_index in enumerate(
        np.asarray(center_indices, dtype=np.int64).reshape(-1).tolist()
    ):
        if 0 <= int(center_index) < len(ids):
            ids[int(center_index)] = int(mineral_id)
    return ids


def _select_one_window_per_unit(
    positions: np.ndarray,
    labels: np.ndarray,
    mineral_ids: np.ndarray,
    minerals: pd.DataFrame,
) -> np.ndarray:
    """Legacy helper: keep unlabeled + one nearest containing window per unit.

    Prefer ``_assign_minerals_to_center_windows`` for the cleaner mineral-centered
    positive construction used by ``one_window_per_unit``.
    """
    positions = np.asarray(positions, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    mineral_ids = np.asarray(mineral_ids, dtype=np.int64).reshape(-1)
    keep = np.zeros(len(labels), dtype=bool)
    unlabeled = labels != 1
    keep[unlabeled] = True
    if minerals is None or len(minerals) == 0:
        return np.where(keep)[0].astype(np.int64)
    mineral_coords = minerals[["x", "y"]].to_numpy(dtype=np.float64)
    for mid in np.unique(mineral_ids[(mineral_ids >= 0) & (labels == 1)]):
        mid = int(mid)
        idxs = np.where((mineral_ids == mid) & (labels == 1))[0]
        if len(idxs) == 0:
            continue
        if mid < len(mineral_coords):
            center = mineral_coords[mid]
            d = np.sqrt(np.sum((positions[idxs, :2] - center) ** 2, axis=1))
            best = int(idxs[int(np.argmin(d))])
        else:
            best = int(idxs[0])
        keep[best] = True
    return np.where(keep)[0].astype(np.int64)


def _grid_geo_extent(metadata: Optional[dict]) -> Optional[dict]:
    """Return geographic grid extent from H5 metadata, or None if unusable."""
    meta = metadata or {}
    try:
        x_min = float(meta["x_min"])
        x_max = float(meta["x_max"])
        y_min = float(meta["y_min"])
        y_max = float(meta["y_max"])
        nx = int(meta.get("nx", meta.get("image_width", meta.get("width", 0))))
        ny = int(meta.get("ny", meta.get("image_height", meta.get("height", 0))))
    except (KeyError, TypeError, ValueError):
        return None
    if nx <= 1 or ny <= 1 or (x_max - x_min) == 0 or (y_max - y_min) == 0:
        return None
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "nx": nx,
        "ny": ny,
        "x_step": (x_max - x_min) / max(nx - 1, 1),
        "y_step": (y_max - y_min) / max(ny - 1, 1),
    }


def _assert_mineral_window_geo_alignment(
    *,
    window_geo: np.ndarray,
    mineral_coords: np.ndarray,
    window_indices: np.ndarray,
    distances: np.ndarray,
    metadata: Optional[dict],
) -> None:
    """Fail fast when window centers are still row/col indices (missing H5 geo metadata)."""
    minerals = np.asarray(mineral_coords, dtype=np.float64)
    geo = np.asarray(window_geo, dtype=np.float64)
    if len(minerals) == 0 or len(geo) == 0:
        return
    indices = np.asarray(window_indices).reshape(-1)
    dist = np.asarray(distances, dtype=np.float64).reshape(-1)
    n_minerals = int(len(minerals))
    unique_windows = int(len(np.unique(indices))) if len(indices) else 0
    finite = dist[np.isfinite(dist)]
    mean_snap = float(np.mean(finite)) if len(finite) else float("nan")
    mineral_span = float(np.linalg.norm(np.ptp(minerals, axis=0)))
    win_abs_max = float(np.nanmax(np.abs(geo)))
    min_abs_max = float(np.nanmax(np.abs(minerals)))
    index_like_windows = win_abs_max <= 5000.0 and min_abs_max > 10.0 * max(win_abs_max, 1.0)
    collapsed = n_minerals >= 5 and unique_windows <= 1
    huge_snap = np.isfinite(mean_snap) and mineral_span > 0 and mean_snap > max(50.0, 0.5 * mineral_span)
    extent = _grid_geo_extent(metadata)
    if extent is None and (index_like_windows or collapsed or huge_snap):
        raise ValueError(
            "矿点中心窗无法对齐到地理坐标：H5 缺少 metadata.x_min/x_max/y_min/y_max/nx/ny。"
            f"窗坐标看起来像行列号(max={win_abs_max:.1f})，矿点是地图坐标(max={min_abs_max:.1f})，"
            f"unique_windows={unique_windows}/{n_minerals}，mean_snap={mean_snap:.1f}。"
            "请从带 metadata 的基座包复制地理范围后再训练。"
        )
    if collapsed and huge_snap:
        raise ValueError(
            "矿点中心窗对齐失败：空间上分散的矿点全部吸附到同一离散窗，"
            f"mean_snap={mean_snap:.1f}，unique_windows={unique_windows}/{n_minerals}。"
            "通常是窗坐标仍为行列号、H5 地理元数据缺失或坐标系不一致。"
        )


def _assign_minerals_to_center_windows(
    sample_coords: np.ndarray,
    minerals: pd.DataFrame,
    metadata: Optional[dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """Map each mineral to the sliding-window patch whose geo-center is nearest.

    With reflect-padding + stride-1 centered patches, this is the discrete-grid
    mineral-centered window (not \"any window that contains the point\").
    Returns (window_indices, distances_m) aligned with ``minerals`` rows.
    """
    if minerals is None or len(minerals) == 0:
        empty_i = np.array([], dtype=np.int64)
        empty_d = np.array([], dtype=np.float64)
        return empty_i, empty_d
    coords = np.asarray(sample_coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2 or len(coords) == 0:
        raise ValueError("sample_coords must be an Nx2 array for mineral-centered windows.")
    geo = _patch_indices_to_geo(coords, metadata or {})
    if geo is None or len(geo) == 0:
        geo = coords[:, :2]
    geo = np.asarray(geo, dtype=np.float64)[:, :2]
    mineral_coords = minerals[["x", "y"]].to_numpy(dtype=np.float64)
    from scipy.spatial import cKDTree

    tree = cKDTree(geo)
    distances, indices = tree.query(mineral_coords, k=1)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    _assert_mineral_window_geo_alignment(
        window_geo=geo,
        mineral_coords=mineral_coords,
        window_indices=indices,
        distances=distances,
        metadata=metadata,
    )
    return indices, distances


DEFAULT_MULTI_WINDOW_MAX_PER_UNIT = 5
# Manuscript main-text positives: 3 windows per deposit, equal loss 1/3.
THREE_WINDOWS_EQUAL_WEIGHT_MODE = "three_windows_equal_weight"
DEFAULT_THREE_WINDOWS_PER_UNIT = 3
DEFAULT_THREE_WINDOWS_SAMPLE_HALFWIDTH = 1
DEFAULT_THREE_WINDOWS_SAMPLE_REF_WINDOW = 3
# Anchor the random-positive neighborhood to the smallest study window (5→halfwidth 2),
# so 5/7/9/11/13/15/17 comparisons do not dilate positive support with W.
DEFAULT_MULTI_WINDOW_SAMPLE_REF_WINDOW = 5
DEFAULT_MULTI_WINDOW_SAMPLE_HALFWIDTH = DEFAULT_MULTI_WINDOW_SAMPLE_REF_WINDOW // 2
# Wider fixed neighborhood (±5 cells); refW=11 → halfwidth 5 when halfwidth unset.
DEFAULT_MULTI_WINDOW_SAMPLE_HALFWIDTH_HW5 = 5
DEFAULT_MULTI_WINDOW_SAMPLE_REF_WINDOW_HW5 = 11
DEFAULT_MULTI_WINDOW_DISTANCE_SIGMA = 1.0
# Keep relative decay profile similar to ±2/σ=1 when neighborhood expands to ±5.
DEFAULT_MULTI_WINDOW_DISTANCE_SIGMA_HW5 = 2.5
HW5_MULTI_WINDOW_MODES = frozenset(
    {
        "multi_window_weighted_hw5",
        "multi_window_distance_weighted_hw5",
    }
)
CAPPED_MULTI_WINDOW_MODES = frozenset(
    {
        THREE_WINDOWS_EQUAL_WEIGHT_MODE,
        "multi_window_weighted",
        "multi_window_distance_weighted",
        "multi_window_weighted_hw5",
        "multi_window_distance_weighted_hw5",
        "multi_window_deposit_weighted",  # alias → equal 1/n (±2)
    }
)
DISTANCE_WEIGHTED_MULTI_WINDOW_MODES = frozenset(
    {
        "multi_window_distance_weighted",
        "multi_window_distance_weighted_hw5",
    }
)
EQUAL_WEIGHT_CAPPED_MULTI_WINDOW_MODES = frozenset(
    {
        THREE_WINDOWS_EQUAL_WEIGHT_MODE,
        "multi_window_weighted",
        "multi_window_weighted_hw5",
        "multi_window_deposit_weighted",
    }
)


def _patch_extent_arrays(
    sample_coords: np.ndarray,
    metadata: Optional[dict],
    patch_size,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return (x_left, x_right, y_top, y_bottom) for each patch, or None if unavailable."""
    coords = np.asarray(sample_coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2 or len(coords) == 0:
        return None
    meta = metadata or {}
    try:
        x_min = float(meta["x_min"])
        x_max = float(meta["x_max"])
        y_min = float(meta["y_min"])
        y_max = float(meta["y_max"])
        nx = int(meta.get("nx", meta.get("image_width", meta.get("width", 0))))
        ny = int(meta.get("ny", meta.get("image_height", meta.get("height", 0))))
    except (KeyError, TypeError, ValueError):
        return None
    if nx <= 1 or ny <= 1 or (x_max - x_min) == 0 or (y_max - y_min) == 0:
        return None
    x_step = (x_max - x_min) / max(nx - 1, 1)
    y_step = (y_max - y_min) / max(ny - 1, 1)
    window_width = int(patch_size or meta.get("window_width", 1) or 1)
    window_height = int(patch_size or meta.get("window_height", 1) or 1)
    rows = coords[:, 0]
    cols = coords[:, 1]
    coordinates_are_centers = bool(meta.get("coordinates_are_centers", False))
    if coordinates_are_centers:
        center_x = x_min + cols * x_step
        center_y = y_max - rows * y_step
        half_width = max(float(window_width) * x_step / 2.0, 0.0)
        half_height = max(float(window_height) * y_step / 2.0, 0.0)
        x_left = center_x - half_width
        x_right = center_x + half_width
        y_top = center_y + half_height
        y_bottom = center_y - half_height
    else:
        x_left = x_min + cols * x_step
        x_right = x_left + window_width * x_step
        y_top = y_max - rows * y_step
        y_bottom = y_top - window_height * y_step
    return x_left, x_right, y_top, y_bottom


def _resolve_multi_window_sample_halfwidth(
    *,
    sample_halfwidth=None,
    sample_ref_window=None,
    patch_size=None,
) -> int:
    """Fixed Chebyshev half-width (grid cells) for multi_window_weighted sampling."""
    if sample_halfwidth is not None:
        try:
            hw = int(sample_halfwidth)
            if hw >= 0:
                return hw
        except (TypeError, ValueError):
            pass
    ref = sample_ref_window
    if ref is None:
        ref = DEFAULT_MULTI_WINDOW_SAMPLE_REF_WINDOW
    try:
        ref_i = int(ref)
    except (TypeError, ValueError):
        ref_i = DEFAULT_MULTI_WINDOW_SAMPLE_REF_WINDOW
    if ref_i < 1:
        ref_i = DEFAULT_MULTI_WINDOW_SAMPLE_REF_WINDOW
    # Prefer odd reference windows; fall back to current patch only if explicitly tiny.
    return max(0, ref_i // 2)


def _select_capped_multi_windows_per_unit(
    sample_coords: np.ndarray,
    minerals: pd.DataFrame,
    metadata: Optional[dict],
    patch_size,
    *,
    max_windows_per_unit: int = DEFAULT_MULTI_WINDOW_MAX_PER_UNIT,
    sample_halfwidth: Optional[int] = None,
    sample_ref_window: Optional[int] = None,
    random_state: int = 42,
    neighbor_selection: str = "random",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per mineral: 1 center window + up to (max-1) neighbor windows in a fixed neighborhood.

    The neighbor pool is defined by Chebyshev distance on patch-center grid indices,
    anchored to ``sample_ref_window`` (default 5 → halfwidth 2), **not** by the
    training patch size ``W``. This keeps positive spatial support comparable across
    W∈{5,7,9,11,13,15,17} while each sample still uses WxW features.

    ``neighbor_selection``:
    - ``nearest``: deterministic 2 (or fewer) nearest Chebyshev neighbors — manuscript
      three-window equal-weight protocol.
    - ``random``: sample up to (max-1) neighbors (legacy ±2 / ±5 modes).

    Returns
    ``(window_indices, mineral_ids, center_snap_distances, chebyshev_offsets)``
    aligned lists. ``chebyshev_offsets`` is 0 for the center window and the
    Chebyshev grid distance from that center for neighbors (for distance-decay loss).
    """
    empty_i = np.array([], dtype=np.int64)
    empty_d = np.array([], dtype=np.float64)
    if minerals is None or len(minerals) == 0:
        return empty_i, empty_i.copy(), empty_d, empty_d.copy()

    coords = np.asarray(sample_coords, dtype=np.float64)
    n_windows = int(len(coords))
    if n_windows <= 0:
        return empty_i, empty_i.copy(), empty_d, empty_d.copy()

    max_n = max(1, int(max_windows_per_unit or DEFAULT_MULTI_WINDOW_MAX_PER_UNIT))
    halfwidth = _resolve_multi_window_sample_halfwidth(
        sample_halfwidth=sample_halfwidth,
        sample_ref_window=sample_ref_window,
        patch_size=patch_size,
    )
    center_idx, center_dist = _assign_minerals_to_center_windows(coords, minerals, metadata)
    mineral_coords = minerals[["x", "y"]].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(int(random_state))

    rows = coords[:, 0].astype(np.float64)
    cols = coords[:, 1].astype(np.float64)
    geo = _patch_indices_to_geo(coords, metadata or {})
    if geo is None or len(geo) == 0:
        geo = coords[:, :2]
    geo = np.asarray(geo, dtype=np.float64)[:, :2]

    out_windows: list = []
    out_minerals: list = []
    out_dists: list = []
    out_cheby: list = []

    for mid, (mineral_x, mineral_y) in enumerate(mineral_coords):
        c_idx = int(center_idx[mid]) if mid < len(center_idx) else -1
        c_dist = float(center_dist[mid]) if mid < len(center_dist) else float("nan")
        if not (0 <= c_idx < n_windows):
            # Fallback: nearest geo window as center.
            d = np.sqrt((geo[:, 0] - mineral_x) ** 2 + (geo[:, 1] - mineral_y) ** 2)
            c_idx = int(np.argmin(d))
            c_dist = float(d[c_idx])

        c_row = float(rows[c_idx])
        c_col = float(cols[c_idx])
        # Fixed neighborhood in index space (independent of training W).
        chebyshev = np.maximum(np.abs(rows - c_row), np.abs(cols - c_col))
        neighborhood = np.where(chebyshev <= float(halfwidth) + 1e-9)[0].astype(np.int64)

        selected: list = [c_idx]
        others = [int(i) for i in neighborhood.tolist() if int(i) != c_idx]
        need = max_n - len(selected)
        if need > 0 and others:
            if len(others) <= need:
                selected.extend(others)
            elif str(neighbor_selection or "random").strip().lower() == "nearest":
                other_idx = np.asarray(others, dtype=np.int64)
                cheby_d = chebyshev[other_idx]
                geo_d = np.sqrt(
                    (geo[other_idx, 0] - mineral_x) ** 2
                    + (geo[other_idx, 1] - mineral_y) ** 2
                )
                order = np.lexsort((geo_d, cheby_d))
                selected.extend(int(other_idx[i]) for i in order[:need])
            else:
                picks = rng.choice(np.asarray(others, dtype=np.int64), size=need, replace=False)
                selected.extend(int(x) for x in np.asarray(picks).tolist())

        for widx in selected:
            widx_i = int(widx)
            out_windows.append(widx_i)
            out_minerals.append(int(mid))
            is_center = widx_i == int(c_idx)
            out_dists.append(c_dist if is_center else float("nan"))
            out_cheby.append(0.0 if is_center else float(chebyshev[widx_i]))

    return (
        np.asarray(out_windows, dtype=np.int64),
        np.asarray(out_minerals, dtype=np.int64),
        np.asarray(out_dists, dtype=np.float64),
        np.asarray(out_cheby, dtype=np.float64),
    )


def _empirical_positive_rate(labels, unlabeled_subsample_ratio=None) -> float:
    """Window-level P(Y=1) on the original unlabeled population.

    When unlabeled rows were kept at rate ``r`` while all positives were kept,
    the raw ``n_pos / (n_pos + n_u)`` is inflated. Restore with ``n_u / r``.
    If both classes were subsampled equally, pass ``unlabeled_subsample_ratio=None``.
    """
    y = np.asarray(labels).reshape(-1)
    n_pos = int(np.sum(y == 1))
    n_unlabeled = int(np.sum(y != 1))
    if n_pos + n_unlabeled <= 0:
        return 0.0
    try:
        ratio = float(unlabeled_subsample_ratio) if unlabeled_subsample_ratio is not None else None
    except (TypeError, ValueError):
        ratio = None
    if ratio is not None and 0.0 < ratio < 1.0:
        restored_unlabeled = float(n_unlabeled) / ratio
        denom = float(n_pos) + restored_unlabeled
        if denom <= 0:
            return 0.0
        return float(n_pos / denom)
    return float(n_pos / (n_pos + n_unlabeled))


def _sample_unlabeled_keep_all_positives(
    y_arr: np.ndarray,
    sample_ratio: float,
    random_state: int,
) -> np.ndarray:
    """Keep every positive sample; subsample only unlabeled (negative) rows."""
    y_flat = np.asarray(y_arr).reshape(-1)
    if sample_ratio >= 1.0:
        return np.arange(len(y_flat), dtype=np.int64)
    rng = np.random.default_rng(int(random_state))
    pos_indices = np.where(y_flat == 1)[0]
    neg_indices = np.where(y_flat == -1)[0]
    if len(neg_indices) == 0:
        return np.asarray(pos_indices, dtype=np.int64)
    neg_keep = min(len(neg_indices), max(1, int(round(len(neg_indices) * float(sample_ratio)))))
    neg_selected = rng.choice(neg_indices, neg_keep, replace=False)
    selected = np.concatenate((np.asarray(pos_indices, dtype=np.int64), np.asarray(neg_selected, dtype=np.int64)))
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def _deposit_loss_weights_from_mineral_ids(mineral_ids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights so each deposit unit contributes equally among its windows."""
    mineral_ids = np.asarray(mineral_ids, dtype=np.int64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    weights = np.ones(len(labels), dtype=np.float64)
    pos = labels == 1
    for mid in np.unique(mineral_ids[pos & (mineral_ids >= 0)]):
        mask = pos & (mineral_ids == int(mid))
        count = int(np.sum(mask))
        if count > 0:
            weights[mask] = 1.0 / float(count)
    return weights


def _deposit_loss_weights_gaussian_distance(
    mineral_ids: np.ndarray,
    labels: np.ndarray,
    chebyshev_offsets: np.ndarray,
    *,
    sigma: float = DEFAULT_MULTI_WINDOW_DISTANCE_SIGMA,
) -> np.ndarray:
    """Gaussian distance-decay loss weights, renormalized to sum=1 per deposit unit.

    ``w_raw = exp(-d^2 / (2 σ^2))`` with Chebyshev grid offset ``d`` from the mineral
    center window; then per-mineral positive weights are scaled so they sum to 1
    (same deposit-level loss budget as equal 1/n weighting).
    """
    mineral_ids = np.asarray(mineral_ids, dtype=np.int64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    offsets = np.asarray(chebyshev_offsets, dtype=np.float64).reshape(-1)
    n = len(labels)
    weights = np.ones(n, dtype=np.float64)
    if len(offsets) != n:
        offsets = np.zeros(n, dtype=np.float64)
    try:
        sigma_f = float(sigma)
    except (TypeError, ValueError):
        sigma_f = float(DEFAULT_MULTI_WINDOW_DISTANCE_SIGMA)
    if not np.isfinite(sigma_f) or sigma_f <= 0:
        sigma_f = float(DEFAULT_MULTI_WINDOW_DISTANCE_SIGMA)

    pos = labels == 1
    for mid in np.unique(mineral_ids[pos & (mineral_ids >= 0)]):
        mask = pos & (mineral_ids == int(mid))
        d = offsets[mask].copy()
        d[~np.isfinite(d)] = 0.0
        d = np.maximum(d, 0.0)
        raw = np.exp(-(d * d) / (2.0 * sigma_f * sigma_f))
        raw = np.maximum(raw, 0.0)
        total = float(np.sum(raw))
        if total > 0:
            weights[mask] = raw / total
        else:
            count = int(np.sum(mask))
            weights[mask] = 1.0 / float(max(count, 1))
    return weights


def _read_mineral_points(label_path: str) -> pd.DataFrame:
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"鏍囩鏂囦欢涓嶅瓨鍦? {label_path}")

    ext = os.path.splitext(label_path)[1].lower()
    if ext in {".csv", ".txt", ".tsv"}:
        last_error = None
        for kwargs in (
            {"sep": None, "engine": "python"},
            {"sep": "\t"},
            {"sep": ","},
            {},
        ):
            try:
                frame = pd.read_csv(label_path, **kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        else:  # pragma: no cover
            raise last_error
    else:
        raise ValueError("Label file must be TXT/CSV/TSV.")

    if frame.empty:
        raise ValueError("Label file is empty.")

    column_map = {str(col).strip().lower(): col for col in frame.columns}
    x_column = next((column_map[key] for key in ("x", "coord_x", "point_x", "east", "easting") if key in column_map), None)
    y_column = next((column_map[key] for key in ("y", "coord_y", "point_y", "north", "northing") if key in column_map), None)
    if x_column is None or y_column is None:
        if frame.shape[1] >= 2:
            x_column, y_column = frame.columns[:2].tolist()
        else:
            raise KeyError("Mineral coordinate file must contain X/Y columns.")

    result = frame[[x_column, y_column]].copy()
    result.columns = ["x", "y"]
    for extra in frame.columns:
        if extra not in {x_column, y_column}:
            result[extra] = frame[extra]
    result = result.dropna(subset=["x", "y"]).reset_index(drop=True)
    result["x"] = pd.to_numeric(result["x"], errors="coerce")
    result["y"] = pd.to_numeric(result["y"], errors="coerce")
    result = result.dropna(subset=["x", "y"]).reset_index(drop=True)
    if result.empty:
        raise ValueError("No valid X/Y coordinates were found in the mineral file.")
    return result


def _read_label_h5(label_path: str) -> np.ndarray:
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"鏍囩鏂囦欢涓嶅瓨鍦? {label_path}")

    with h5py.File(label_path, "r") as handle:
        for key in ("labels", "label", "y", "targets", "target"):
            if key in handle:
                labels = np.asarray(handle[key][:]).reshape(-1)
                break
        else:
            labels = None

        if labels is None:
            windows = None
            if "windows" in handle:
                windows = np.asarray(handle["windows"][:])
            elif "label_windows" in handle:
                windows = np.asarray(handle["label_windows"][:])

            if windows is None:
                raise ValueError("H5 label file must contain labels/y/windows dataset.")

            sample_count = None
            for key in ("index_positions", "positions"):
                if key in handle:
                    sample_count = len(handle[key])
                    break

            if sample_count is not None:
                sample_axis = next((idx for idx, dim in enumerate(windows.shape) if dim == sample_count), None)
            else:
                sample_axis = None

            if sample_axis is None:
                sample_axis = windows.ndim - 1

            windows = np.moveaxis(windows, sample_axis, 0)
            if windows.ndim == 1:
                labels = windows
            else:
                positive_mask = np.any(windows == 1, axis=tuple(range(1, windows.ndim)))
                labels = np.where(positive_mask, 1, -1)

    labels = np.asarray(labels).reshape(-1)
    if labels.size == 0:
        raise ValueError("No usable samples found in H5 labels.")
    labels = np.where(labels > 0, 1, -1).astype(np.int32)
    return labels


def _extract_metadata(handle: h5py.File) -> dict:
    metadata = {}
    if "metadata" in handle:
        metadata.update(dict(handle["metadata"].attrs))
    for key in (
        "x_min", "x_max", "y_min", "y_max", "nx", "ny", "window_width", "window_height",
        "meters_per_coordinate_unit", "metres_per_coordinate_unit", "m_per_coordinate_unit",
        "coordinate_unit_to_meters", "cell_size_m", "pixel_size_m", "grid_resolution_m",
        "crs", "crs_wkt", "epsg", "affine", "transform",
    ):
        if key in handle.attrs:
            metadata[key] = handle.attrs[key]
    return metadata


def _patch_indices_to_geo(coords, metadata):
    if coords is None:
        return None

    arr = np.asarray(coords)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] < 2:
        return arr.astype(np.float64, copy=False)

    arr = arr[:, :2].astype(np.float64, copy=False)
    meta = metadata or {}
    try:
        x_min = float(meta["x_min"])
        x_max = float(meta["x_max"])
        y_min = float(meta["y_min"])
        y_max = float(meta["y_max"])
        nx = int(meta.get("nx", meta.get("image_width", meta.get("width", 0))))
        ny = int(meta.get("ny", meta.get("image_height", meta.get("height", 0))))
    except (KeyError, TypeError, ValueError):
        return arr

    if nx <= 1 or ny <= 1:
        return arr

    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span == 0 or y_span == 0:
        return arr

    direct_x = arr[:, 0]
    direct_y = arr[:, 1]
    x_bound = max(nx - 1, 1) * 1.5
    y_bound = max(ny - 1, 1) * 1.5
    direct_ok = np.nanmax(np.abs(direct_x)) <= x_bound and np.nanmax(np.abs(direct_y)) <= y_bound

    swapped_x = arr[:, 1]
    swapped_y = arr[:, 0]
    swapped_ok = np.nanmax(np.abs(swapped_x)) <= x_bound and np.nanmax(np.abs(swapped_y)) <= y_bound

    if not direct_ok and swapped_ok:
        x_values = swapped_x
        y_values = swapped_y
    else:
        x_values = direct_x
        y_values = direct_y

    if np.nanmax(np.abs(x_values)) <= x_bound and np.nanmax(np.abs(y_values)) <= y_bound:
        x_step = x_span / max(nx - 1, 1)
        y_step = y_span / max(ny - 1, 1)
        window_width = int(meta.get("window_width", 1) or 1)
        window_height = int(meta.get("window_height", 1) or 1)
        coordinates_are_centers = bool(meta.get("coordinates_are_centers", False))
        x_offset = 0.0 if coordinates_are_centers else (window_width - 1) / 2.0 if window_width > 1 else 0.0
        y_offset = 0.0 if coordinates_are_centers else (window_height - 1) / 2.0 if window_height > 1 else 0.0

        geo_x = x_min + (x_values + x_offset) * x_step
        geo_y = y_max - (y_values + y_offset) * y_step
        return np.column_stack([geo_x, geo_y]).astype(np.float64, copy=False)

    return arr


def _load_feature_tensor(
    data_path: str,
    patch_size: int,
    patch_stride: int,
    use_reflect_padding: bool = False,
    selected_channels=None,
):
    with h5py.File(data_path, "r") as handle:
        metadata = _extract_metadata(handle)
        metadata["available_channel_names"] = infer_h5_channel_names(data_path)
        if "windows" in handle:
            windows = np.asarray(handle["windows"][:], dtype=np.float32)
            coordinates = np.asarray(handle["coordinates"][:], dtype=np.float64) if "coordinates" in handle else None
            if windows.ndim < 4:
                raise ValueError(f"windows 鏁版嵁鑷冲皯搴斾负 4 缁达紝褰撳墠涓?{windows.shape}")
            samples = np.transpose(windows, (3, 2, 0, 1))
            samples, metadata = subset_samples_by_channels(samples, metadata, selected_channels)
            return samples, coordinates, metadata, "windows"

        if PatchCreator is None:
            raise ImportError("PatchCreator is unavailable; cannot generate patches from raw H5.")
        creator = PatchCreator(data_path)
        try:
            samples, coordinates = creator.generate_patches(
                int(patch_size),
                int(patch_stride),
                enable_padding=bool(use_reflect_padding),
                padding_mode="reflect",
            )
        finally:
            creator.close()
        metadata["window_width"] = int(patch_size)
        metadata["window_height"] = int(patch_size)
        metadata["patch_stride"] = int(patch_stride)
        metadata["reflect_padding"] = bool(use_reflect_padding)
        metadata["coordinates_are_centers"] = bool(use_reflect_padding)
        if getattr(creator, "source_kind", "") == "fused_features" and _grid_geo_extent(metadata) is None:
            raise ValueError(
                "特征 H5 缺少地理范围 metadata（x_min/x_max/y_min/y_max/nx/ny）: "
                f"{data_path}。矿点中心窗与汇水域整盆隔离都需要该信息；"
                "请从带 metadata 的基座包复制后再训练。"
            )
        samples, metadata = subset_samples_by_channels(np.asarray(samples, dtype=np.float32), metadata, selected_channels)
        return samples, np.asarray(coordinates, dtype=np.float64), metadata, "patches"


def _labels_from_minerals(sample_coords: np.ndarray, minerals: pd.DataFrame, buffer_radius: float) -> np.ndarray:
    labels = np.zeros(len(sample_coords), dtype=np.int32)
    if minerals is None or len(minerals) == 0:
        return labels
    mineral_coords = minerals[["x", "y"]].to_numpy(dtype=np.float64)
    for mineral_x, mineral_y in mineral_coords:
        distances = np.sqrt((sample_coords[:, 0] - mineral_x) ** 2 + (sample_coords[:, 1] - mineral_y) ** 2)
        labels[distances <= float(buffer_radius)] = 1
    return labels


def _window_contains_minerals(sample_coords: np.ndarray, minerals: pd.DataFrame, metadata: dict, patch_size) -> np.ndarray:
    mineral_ids = _window_primary_mineral_ids(sample_coords, minerals, metadata, patch_size)
    return (mineral_ids >= 0).astype(np.int32)


def _window_primary_mineral_ids(
    sample_coords: np.ndarray,
    minerals: pd.DataFrame,
    metadata: dict,
    patch_size,
) -> np.ndarray:
    assigned_ids = np.full(len(sample_coords), -1, dtype=np.int64)
    if minerals is None or len(minerals) == 0:
        return assigned_ids

    coords = np.asarray(sample_coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        return assigned_ids

    geo_coords = _patch_indices_to_geo(coords, metadata)
    if geo_coords is None or len(geo_coords) == 0:
        geo_coords = coords[:, :2]
    mineral_coords = minerals[["x", "y"]].to_numpy(dtype=np.float64)
    best_distances = np.full(len(coords), np.inf, dtype=np.float64)

    meta = metadata or {}
    try:
        x_min = float(meta["x_min"])
        x_max = float(meta["x_max"])
        y_min = float(meta["y_min"])
        y_max = float(meta["y_max"])
        nx = int(meta.get("nx", meta.get("image_width", meta.get("width", 0))))
        ny = int(meta.get("ny", meta.get("image_height", meta.get("height", 0))))
    except (KeyError, TypeError, ValueError):
        x_min = x_max = y_min = y_max = 0.0
        nx = ny = 0

    if nx > 1 and ny > 1 and (x_max - x_min) != 0 and (y_max - y_min) != 0:
        x_step = (x_max - x_min) / max(nx - 1, 1)
        y_step = (y_max - y_min) / max(ny - 1, 1)
        window_width = int(patch_size or meta.get("window_width", 1) or 1)
        window_height = int(patch_size or meta.get("window_height", 1) or 1)
        rows = coords[:, 0]
        cols = coords[:, 1]
        coordinates_are_centers = bool(meta.get("coordinates_are_centers", False))
        if coordinates_are_centers:
            center_x = x_min + cols * x_step
            center_y = y_max - rows * y_step
            half_width = max(float(window_width) * x_step / 2.0, 0.0)
            half_height = max(float(window_height) * y_step / 2.0, 0.0)
            x_left = center_x - half_width
            x_right = center_x + half_width
            y_top = center_y + half_height
            y_bottom = center_y - half_height
        else:
            x_left = x_min + cols * x_step
            x_right = x_left + window_width * x_step
            y_top = y_max - rows * y_step
            y_bottom = y_top - window_height * y_step

        for mineral_idx, (mineral_x, mineral_y) in enumerate(mineral_coords):
            inside = (mineral_x >= x_left) & (mineral_x < x_right) & (mineral_y <= y_top) & (mineral_y > y_bottom)
            if not np.any(inside):
                continue
            distances = np.sqrt((geo_coords[:, 0] - mineral_x) ** 2 + (geo_coords[:, 1] - mineral_y) ** 2)
            update_mask = inside & (distances < best_distances)
            assigned_ids[update_mask] = int(mineral_idx)
            best_distances[update_mask] = distances[update_mask]
        return assigned_ids

    radius = max(float(patch_size or 1) / 2.0, 0.5)
    for mineral_idx, (mineral_x, mineral_y) in enumerate(mineral_coords):
        distances = np.sqrt((geo_coords[:, 0] - mineral_x) ** 2 + (geo_coords[:, 1] - mineral_y) ** 2)
        update_mask = (distances <= radius) & (distances < best_distances)
        assigned_ids[update_mask] = int(mineral_idx)
        best_distances[update_mask] = distances[update_mask]
    return assigned_ids


def _buffer_exclusion_mask(sample_coords: np.ndarray, minerals: pd.DataFrame, metadata: dict, buffer_radius: float) -> np.ndarray:
    """Mark samples within ``buffer_radius`` of any mineral in ``minerals``.

    Callers that enforce leave-one-unit blindness should pass *train* minerals only,
    so held-out test deposit locations do not carve the unlabeled pool.
    """
    if minerals is None or len(minerals) == 0 or float(buffer_radius) <= 0:
        return np.zeros(len(sample_coords), dtype=bool)

    geo_coords = _patch_indices_to_geo(sample_coords, metadata)
    if geo_coords is None or len(geo_coords) == 0:
        return np.zeros(len(sample_coords), dtype=bool)

    geo_coords = np.asarray(geo_coords, dtype=np.float64)
    mineral_coords = minerals[["x", "y"]].to_numpy(dtype=np.float64)
    mask = np.zeros(len(geo_coords), dtype=bool)
    radius = float(buffer_radius)
    for mineral_x, mineral_y in mineral_coords:
        distances = np.sqrt((geo_coords[:, 0] - mineral_x) ** 2 + (geo_coords[:, 1] - mineral_y) ** 2)
        mask |= distances <= radius
    return mask


def _split_minerals_by_kmeans(
    minerals: pd.DataFrame,
    *,
    n_clusters: int = 10,
    train_ratio: float = 0.7,
    random_state: int = 42,
) -> dict:
    frame = minerals.reset_index(drop=True).copy() if minerals is not None else pd.DataFrame(columns=["x", "y"])
    if len(frame) == 0:
        empty = frame.iloc[0:0].copy()
        empty["kmeans_cluster"] = np.array([], dtype=np.int64)
        return {
            "train": empty.copy(),
            "test": empty.copy(),
            "n_clusters": int(n_clusters),
            "train_ratio": float(train_ratio),
            "cluster_ids": [],
        }

    coords = frame[["x", "y"]].to_numpy(dtype=np.float64)
    unique_count = len(np.unique(coords, axis=0))
    cluster_count = max(1, min(int(n_clusters), len(frame), unique_count))
    ratio = float(train_ratio)
    if not np.isfinite(ratio):
        ratio = 0.7
    ratio = float(min(max(ratio, 0.1), 0.9))
    rng = np.random.default_rng(int(random_state))

    if cluster_count == 1 or len(frame) < 3:
        frame_with_cluster = frame.copy()
        frame_with_cluster["kmeans_cluster"] = 0
        order = rng.permutation(len(frame))
        train_count = max(1, int(round(len(frame) * ratio)))
        train_count = min(train_count, len(frame) - 1) if len(frame) > 1 else len(frame)
        train_rows = np.sort(order[:train_count]).astype(np.int64)
        test_rows = np.sort(order[train_count:]).astype(np.int64)
        return {
            "train": frame_with_cluster.iloc[train_rows].reset_index(drop=True),
            "test": frame_with_cluster.iloc[test_rows].reset_index(drop=True),
            "n_clusters": int(cluster_count),
            "train_ratio": float(ratio),
            "cluster_ids": [0] * len(frame),
        }

    kmeans = KMeans(n_clusters=cluster_count, random_state=int(random_state), n_init=10)
    cluster_ids = kmeans.fit_predict(coords)
    frame_with_cluster = frame.copy()
    frame_with_cluster["kmeans_cluster"] = np.asarray(cluster_ids, dtype=np.int64)

    train_rows = []
    test_rows = []
    for cluster_id in sorted(np.unique(cluster_ids)):
        cluster_rows = np.where(cluster_ids == cluster_id)[0]
        if len(cluster_rows) == 0:
            continue
        order = rng.permutation(cluster_rows)
        if len(cluster_rows) == 1:
            train_count = 1
        else:
            train_count = int(round(len(cluster_rows) * ratio))
            train_count = max(1, min(train_count, len(cluster_rows) - 1))
        train_rows.extend(np.sort(order[:train_count]).astype(np.int64).tolist())
        test_rows.extend(np.sort(order[train_count:]).astype(np.int64).tolist())

    train_rows = np.asarray(sorted(set(train_rows)), dtype=np.int64)
    test_rows = np.asarray(sorted(set(test_rows)), dtype=np.int64)
    target_train = int(round(len(frame) * ratio))
    target_train = max(1, min(target_train, len(frame) - 1)) if len(frame) > 1 else len(frame)
    if len(train_rows) > target_train:
        move_count = len(train_rows) - target_train
        move_indices = rng.choice(train_rows, move_count, replace=False)
        train_rows = np.asarray(sorted(set(train_rows.tolist()) - set(move_indices.tolist())), dtype=np.int64)
        test_rows = np.asarray(sorted(set(test_rows.tolist()) | set(move_indices.tolist())), dtype=np.int64)
    elif len(train_rows) < target_train:
        move_count = target_train - len(train_rows)
        if len(test_rows) > 0:
            move_indices = rng.choice(test_rows, min(move_count, len(test_rows)), replace=False)
            test_rows = np.asarray(sorted(set(test_rows.tolist()) - set(move_indices.tolist())), dtype=np.int64)
            train_rows = np.asarray(sorted(set(train_rows.tolist()) | set(move_indices.tolist())), dtype=np.int64)

    return {
        "train": frame_with_cluster.iloc[train_rows].reset_index(drop=True),
        "test": frame_with_cluster.iloc[test_rows].reset_index(drop=True),
        "n_clusters": int(cluster_count),
        "train_ratio": float(ratio),
        "cluster_ids": np.asarray(cluster_ids, dtype=np.int64).tolist(),
    }


def _split_minerals_by_hard_clusters(
    minerals: pd.DataFrame,
    *,
    n_clusters: int = 10,
    train_ratio: float = 0.7,
    random_state: int = 42,
) -> dict:
    frame = minerals.reset_index(drop=True).copy() if minerals is not None else pd.DataFrame(columns=["x", "y"])
    if len(frame) == 0:
        empty = frame.iloc[0:0].copy()
        return {
            "train": empty.copy(),
            "test": empty.copy(),
            "n_clusters": int(n_clusters),
            "train_ratio": float(train_ratio),
            "cluster_ids": [],
            "cluster_centers": [],
            "train_cluster_ids": [],
            "test_cluster_ids": [],
            "cluster_summaries": [],
            "algorithm": "spatial_hard",
        }

    coords = frame[["x", "y"]].to_numpy(dtype=np.float64)
    unique_count = len(np.unique(coords, axis=0))
    cluster_count = max(1, min(int(n_clusters), len(frame), unique_count))
    ratio = float(train_ratio)
    if not np.isfinite(ratio):
        ratio = 0.7
    ratio = float(min(max(ratio, 0.1), 0.9))
    rng = np.random.default_rng(int(random_state))

    if cluster_count == 1 or len(frame) < 3:
        frame_with_cluster = frame.copy()
        frame_with_cluster["kmeans_cluster"] = 0
        order = rng.permutation(len(frame))
        train_count = max(1, int(round(len(frame) * ratio)))
        train_count = min(train_count, len(frame) - 1) if len(frame) > 1 else len(frame)
        train_rows = np.sort(order[:train_count]).astype(np.int64)
        test_rows = np.sort(order[train_count:]).astype(np.int64)
        return {
            "train": frame_with_cluster.iloc[train_rows].reset_index(drop=True),
            "test": frame_with_cluster.iloc[test_rows].reset_index(drop=True),
            "n_clusters": 1,
            "train_ratio": float(ratio),
            "cluster_ids": [0] * len(frame),
            "cluster_centers": [np.mean(coords, axis=0).astype(float).tolist()],
            "train_cluster_ids": [0],
            "test_cluster_ids": [],
            "cluster_summaries": [
                {
                    "cluster_id": 0,
                    "sample_count": int(len(frame)),
                    "train_count": int(len(train_rows)),
                    "test_count": int(len(test_rows)),
                    "assigned_split": "mixed",
                }
            ],
            "algorithm": "spatial_hard",
        }

    kmeans = KMeans(n_clusters=cluster_count, random_state=int(random_state), n_init=10)
    cluster_ids = kmeans.fit_predict(coords)
    frame_with_cluster = frame.copy()
    frame_with_cluster["kmeans_cluster"] = np.asarray(cluster_ids, dtype=np.int64)

    cluster_sizes = {int(cluster_id): int(np.sum(cluster_ids == cluster_id)) for cluster_id in np.unique(cluster_ids)}
    target_train = int(round(len(frame) * ratio))
    target_train = max(1, min(target_train, len(frame) - 1))
    ordered_clusters = [int(cluster_id) for cluster_id in rng.permutation(sorted(cluster_sizes.keys()))]

    train_clusters = []
    train_total = 0
    for cluster_id in ordered_clusters:
        remaining_clusters = [cid for cid in ordered_clusters if cid not in train_clusters and cid != cluster_id]
        if train_total < target_train and remaining_clusters:
            train_clusters.append(cluster_id)
            train_total += cluster_sizes[cluster_id]

    if not train_clusters:
        train_clusters = [ordered_clusters[0]]
    test_clusters = [cluster_id for cluster_id in sorted(cluster_sizes) if cluster_id not in set(train_clusters)]
    if not test_clusters and len(train_clusters) > 1:
        move_cluster = train_clusters.pop()
        test_clusters = [move_cluster]

    train_cluster_set = set(train_clusters)
    test_cluster_set = set(test_clusters)
    train_rows = np.where(np.isin(cluster_ids, list(train_cluster_set)))[0].astype(np.int64)
    test_rows = np.where(np.isin(cluster_ids, list(test_cluster_set)))[0].astype(np.int64)

    cluster_summaries = []
    for cluster_id in sorted(cluster_sizes):
        assigned_split = "train" if cluster_id in train_cluster_set else "test"
        cluster_summaries.append(
            {
                "cluster_id": int(cluster_id),
                "sample_count": int(cluster_sizes[cluster_id]),
                "train_count": int(np.sum(cluster_ids[train_rows] == cluster_id)),
                "test_count": int(np.sum(cluster_ids[test_rows] == cluster_id)),
                "assigned_split": assigned_split,
            }
        )

    return {
        "train": frame_with_cluster.iloc[train_rows].reset_index(drop=True),
        "test": frame_with_cluster.iloc[test_rows].reset_index(drop=True),
        "n_clusters": int(cluster_count),
        "train_ratio": float(ratio),
        "cluster_ids": np.asarray(cluster_ids, dtype=np.int64).tolist(),
        "cluster_centers": np.asarray(kmeans.cluster_centers_, dtype=np.float64).tolist(),
        "train_cluster_ids": sorted(int(item) for item in train_cluster_set),
        "test_cluster_ids": sorted(int(item) for item in test_cluster_set),
        "cluster_summaries": cluster_summaries,
        "algorithm": "spatial_hard",
    }


def _split_minerals_by_cluster_holdout_cv(
    minerals: pd.DataFrame,
    *,
    n_clusters: int = 10,
    train_ratio: float = 0.9,
    random_state: int = 42,
) -> dict:
    """Split every KMeans cluster into dev/test, then keep cluster ids for inner CV."""
    frame = minerals.reset_index(drop=True).copy() if minerals is not None else pd.DataFrame(columns=["x", "y"])
    if len(frame) == 0:
        empty = frame.iloc[0:0].copy()
        return {
            "train": empty.copy(),
            "test": empty.copy(),
            "n_clusters": int(n_clusters),
            "train_ratio": float(train_ratio),
            "requested_train_ratio": float(train_ratio),
            "cluster_ids": [],
            "cluster_centers": [],
            "train_fold_ids": [0],
            "test_fold_ids": [1],
            "val_fold_ids": [1],
            "cluster_summaries": [],
            "algorithm": "spatial_cluster_holdout_cv",
        }

    coords = frame[["x", "y"]].to_numpy(dtype=np.float64)
    unique_count = len(np.unique(coords, axis=0))
    cluster_count = max(1, min(int(n_clusters), len(frame), unique_count))
    ratio = float(train_ratio)
    if not np.isfinite(ratio):
        ratio = 0.9
    ratio = float(min(max(ratio, 0.1), 0.99))

    if cluster_count == 1 or len(frame) < 3:
        cluster_ids = np.zeros(len(frame), dtype=np.int64)
    else:
        kmeans = KMeans(n_clusters=cluster_count, random_state=int(random_state), n_init=10)
        cluster_ids = kmeans.fit_predict(coords).astype(np.int64)

    frame_with_cluster = frame.copy()
    frame_with_cluster["kmeans_cluster"] = np.asarray(cluster_ids, dtype=np.int64)
    frame_with_cluster["external_split_fold"] = -1

    train_rows = []
    test_rows = []
    cluster_summaries = []
    cluster_centers = []
    rng = np.random.default_rng(int(random_state))
    valid_cluster_ids = sorted(int(item) for item in np.unique(cluster_ids) if int(item) >= 0)
    for cluster_order, cluster_id in enumerate(valid_cluster_ids):
        cluster_rows = np.where(cluster_ids == cluster_id)[0].astype(np.int64)
        cluster_coords = coords[cluster_rows]
        cluster_centers.append(np.mean(cluster_coords[:, :2], axis=0).astype(float).tolist())

        if len(cluster_rows) <= 1:
            cluster_train_rows = cluster_rows
            cluster_test_rows = np.array([], dtype=np.int64)
            axis = 0
        else:
            test_count = int(round(len(cluster_rows) * (1.0 - ratio)))
            test_count = max(1, min(test_count, len(cluster_rows) - 1))
            x_span = float(np.max(cluster_coords[:, 0]) - np.min(cluster_coords[:, 0]))
            y_span = float(np.max(cluster_coords[:, 1]) - np.min(cluster_coords[:, 1]))
            axis = 0 if x_span >= y_span else 1
            ordered_rows = cluster_rows[np.argsort(cluster_coords[:, axis], kind="mergesort")]
            take_high_end = bool((cluster_order + int(rng.integers(0, 2))) % 2)
            if take_high_end:
                cluster_test_rows = ordered_rows[-test_count:]
                cluster_train_rows = ordered_rows[:-test_count]
            else:
                cluster_test_rows = ordered_rows[:test_count]
                cluster_train_rows = ordered_rows[test_count:]

        train_rows.extend(int(item) for item in cluster_train_rows.tolist())
        test_rows.extend(int(item) for item in cluster_test_rows.tolist())
        cluster_summaries.append(
            {
                "cluster_id": int(cluster_id),
                "sample_count": int(len(cluster_rows)),
                "train_count": int(len(cluster_train_rows)),
                "test_count": int(len(cluster_test_rows)),
                "axis": "x" if axis == 0 else "y",
                "assigned_split": "mixed_holdout",
            }
        )

    if not test_rows and len(frame_with_cluster) > 1:
        ordered = np.asarray(sorted(train_rows), dtype=np.int64)
        forced_test_row = int(ordered[-1])
        train_rows = [int(item) for item in train_rows if int(item) != forced_test_row]
        test_rows = [forced_test_row]
        forced_cluster = int(frame_with_cluster.loc[forced_test_row, "kmeans_cluster"])
        for summary in cluster_summaries:
            if int(summary.get("cluster_id", -1)) == forced_cluster:
                summary["train_count"] = int(max(int(summary.get("train_count", 0)) - 1, 0))
                summary["test_count"] = int(int(summary.get("test_count", 0)) + 1)
                break

    train_rows = np.asarray(sorted(set(train_rows)), dtype=np.int64)
    test_rows = np.asarray(sorted(set(test_rows)), dtype=np.int64)
    frame_with_cluster.loc[train_rows, "external_split_fold"] = 0
    frame_with_cluster.loc[test_rows, "external_split_fold"] = 1

    return {
        "train": frame_with_cluster.iloc[train_rows].reset_index(drop=True),
        "test": frame_with_cluster.iloc[test_rows].reset_index(drop=True),
        "n_clusters": int(cluster_count),
        "train_ratio": float(len(train_rows) / max(len(frame_with_cluster), 1)),
        "requested_train_ratio": float(ratio),
        "cluster_ids": np.asarray(cluster_ids, dtype=np.int64).tolist(),
        "cluster_centers": cluster_centers,
        "train_fold_ids": [0],
        "test_fold_ids": [1],
        "val_fold_ids": [1],
        "cluster_summaries": cluster_summaries,
        "algorithm": "spatial_cluster_holdout_cv",
    }


def _normalize_spatial_fold_count(n_folds: int, mineral_count: int) -> int:
    try:
        fold_count = int(n_folds)
    except (TypeError, ValueError):
        fold_count = 5
    fold_count = max(2, fold_count)
    mineral_count = int(max(mineral_count, 0))
    if mineral_count > 0:
        fold_count = min(fold_count, mineral_count)
    return max(1, fold_count)


def _assign_spatial_stratified_mineral_folds(
    minerals: pd.DataFrame,
    *,
    n_clusters: int = 10,
    n_folds: int = 5,
    random_state: int = 42,
    prefer_existing_clusters: bool = False,
) -> dict:
    frame = minerals.reset_index(drop=True).copy() if minerals is not None else pd.DataFrame(columns=["x", "y"])
    if len(frame) == 0:
        empty = frame.iloc[0:0].copy()
        empty["kmeans_cluster"] = np.array([], dtype=np.int64)
        empty["spatial_subgroup"] = np.array([], dtype=np.int64)
        empty["spatial_fold"] = np.array([], dtype=np.int64)
        return {
            "frame": empty,
            "n_clusters": int(n_clusters),
            "fold_count": int(max(int(n_folds or 5), 1)),
            "cluster_ids": [],
            "cluster_centers": [],
            "fold_ids": [],
            "cluster_summaries": [],
        }

    coords = frame[["x", "y"]].to_numpy(dtype=np.float64)
    unique_count = len(np.unique(coords, axis=0))
    cluster_count = max(1, min(int(n_clusters), len(frame), unique_count))
    fold_count = _normalize_spatial_fold_count(n_folds, len(frame))

    use_existing_clusters = False
    if prefer_existing_clusters and "kmeans_cluster" in frame.columns:
        existing = pd.to_numeric(frame["kmeans_cluster"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        valid_existing = sorted(int(item) for item in np.unique(existing) if int(item) >= 0)
        if valid_existing:
            cluster_ids = existing
            cluster_count = int(len(valid_existing))
            use_existing_clusters = True

    if not use_existing_clusters:
        if cluster_count == 1 or len(frame) < 3:
            cluster_ids = np.zeros(len(frame), dtype=np.int64)
        else:
            kmeans = KMeans(n_clusters=cluster_count, random_state=int(random_state), n_init=10)
            cluster_ids = kmeans.fit_predict(coords).astype(np.int64)

    frame_with_folds = frame.copy()
    frame_with_folds["kmeans_cluster"] = np.asarray(cluster_ids, dtype=np.int64)
    subgroup_ids = np.full(len(frame_with_folds), -1, dtype=np.int64)
    fold_ids = np.full(len(frame_with_folds), -1, dtype=np.int64)

    cluster_summaries = []
    valid_cluster_ids = sorted(int(item) for item in np.unique(cluster_ids) if int(item) >= 0)
    cluster_centers = []
    for cluster_order, cluster_id in enumerate(valid_cluster_ids):
        cluster_rows = np.where(cluster_ids == cluster_id)[0].astype(np.int64)
        if len(cluster_rows) == 0:
            continue
        cluster_coords = coords[cluster_rows]
        cluster_centers.append(np.mean(cluster_coords[:, :2], axis=0).astype(float).tolist())
        x_span = float(np.max(cluster_coords[:, 0]) - np.min(cluster_coords[:, 0])) if len(cluster_coords) else 0.0
        y_span = float(np.max(cluster_coords[:, 1]) - np.min(cluster_coords[:, 1])) if len(cluster_coords) else 0.0
        axis = 0 if x_span >= y_span else 1
        ordered_rows = cluster_rows[np.argsort(cluster_coords[:, axis], kind="mergesort")]
        fold_counts = [0 for _ in range(fold_count)]
        for subgroup_id, subgroup_rows in enumerate(np.array_split(ordered_rows, fold_count)):
            subgroup_rows = np.asarray(subgroup_rows, dtype=np.int64)
            if len(subgroup_rows) == 0:
                continue
            # Rotate small clusters across folds so singleton clusters do not all land in fold 0.
            fold_id = int((subgroup_id + cluster_order) % fold_count)
            subgroup_ids[subgroup_rows] = int(subgroup_id)
            fold_ids[subgroup_rows] = fold_id
            fold_counts[fold_id] += int(len(subgroup_rows))
        cluster_summaries.append(
            {
                "cluster_id": int(cluster_id),
                "sample_count": int(len(cluster_rows)),
                "axis": "x" if axis == 0 else "y",
                "fold_mineral_counts": fold_counts,
            }
        )

    frame_with_folds["spatial_subgroup"] = subgroup_ids
    frame_with_folds["spatial_fold"] = fold_ids
    return {
        "frame": frame_with_folds,
        "n_clusters": int(cluster_count),
        "fold_count": int(fold_count),
        "cluster_ids": np.asarray(cluster_ids, dtype=np.int64).tolist(),
        "cluster_centers": cluster_centers,
        "fold_ids": np.asarray(fold_ids, dtype=np.int64).tolist(),
        "cluster_summaries": cluster_summaries,
    }


def _split_minerals_by_spatial_stratified_folds(
    minerals: pd.DataFrame,
    *,
    n_clusters: int = 10,
    n_folds: int = 5,
    train_ratio: float = 0.7,
    random_state: int = 42,
) -> dict:
    assignment = _assign_spatial_stratified_mineral_folds(
        minerals,
        n_clusters=n_clusters,
        n_folds=n_folds,
        random_state=random_state,
    )
    frame = assignment["frame"]
    if len(frame) == 0:
        empty = frame.iloc[0:0].copy()
        return {
            "train": empty.copy(),
            "test": empty.copy(),
            "n_clusters": int(assignment.get("n_clusters", n_clusters)),
            "train_ratio": float(train_ratio),
            "fold_count": int(assignment.get("fold_count", n_folds)),
            "cluster_ids": [],
            "cluster_centers": [],
            "train_fold_ids": [],
            "test_fold_ids": [],
            "cluster_summaries": [],
            "algorithm": "spatial_stratified",
        }

    ratio = float(train_ratio)
    if not np.isfinite(ratio):
        ratio = 0.7
    ratio = float(min(max(ratio, 0.1), 0.9))
    rng = np.random.default_rng(int(random_state))

    fold_ids = pd.to_numeric(frame["spatial_fold"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    valid_fold_ids = sorted(int(item) for item in np.unique(fold_ids) if int(item) >= 0)
    if len(valid_fold_ids) <= 1:
        order = rng.permutation(len(frame))
        train_count = max(1, int(round(len(frame) * ratio)))
        train_count = min(train_count, len(frame) - 1) if len(frame) > 1 else len(frame)
        train_rows = np.sort(order[:train_count]).astype(np.int64)
        test_rows = np.sort(order[train_count:]).astype(np.int64)
        train_fold_ids = sorted(int(item) for item in np.unique(fold_ids[train_rows]) if int(item) >= 0)
        test_fold_ids = sorted(int(item) for item in np.unique(fold_ids[test_rows]) if int(item) >= 0)
    else:
        target_test_fold_count = int(round(len(valid_fold_ids) * (1.0 - ratio)))
        target_test_fold_count = max(1, min(target_test_fold_count, len(valid_fold_ids) - 1))
        shuffled_folds = np.asarray(valid_fold_ids, dtype=np.int64)
        rng.shuffle(shuffled_folds)
        test_fold_ids = sorted(int(item) for item in shuffled_folds[:target_test_fold_count])
        train_fold_ids = [int(item) for item in valid_fold_ids if int(item) not in set(test_fold_ids)]
        train_rows = np.where(np.isin(fold_ids, train_fold_ids))[0].astype(np.int64)
        test_rows = np.where(np.isin(fold_ids, test_fold_ids))[0].astype(np.int64)
        if len(train_rows) == 0 or len(test_rows) == 0:
            order = rng.permutation(len(frame))
            train_count = max(1, int(round(len(frame) * ratio)))
            train_count = min(train_count, len(frame) - 1) if len(frame) > 1 else len(frame)
            train_rows = np.sort(order[:train_count]).astype(np.int64)
            test_rows = np.sort(order[train_count:]).astype(np.int64)
            train_fold_ids = sorted(int(item) for item in np.unique(fold_ids[train_rows]) if int(item) >= 0)
            test_fold_ids = sorted(int(item) for item in np.unique(fold_ids[test_rows]) if int(item) >= 0)

    return {
        "train": frame.iloc[train_rows].reset_index(drop=True),
        "test": frame.iloc[test_rows].reset_index(drop=True),
        "n_clusters": int(assignment.get("n_clusters", n_clusters)),
        "train_ratio": float(len(train_rows) / max(len(frame), 1)),
        "requested_train_ratio": float(ratio),
        "fold_count": int(assignment.get("fold_count", n_folds)),
        "cluster_ids": list(assignment.get("cluster_ids", [])),
        "cluster_centers": list(assignment.get("cluster_centers", [])),
        "train_fold_ids": sorted(int(item) for item in train_fold_ids),
        "test_fold_ids": sorted(int(item) for item in test_fold_ids),
        "val_fold_ids": sorted(int(item) for item in test_fold_ids),
        "cluster_summaries": list(assignment.get("cluster_summaries", [])),
        "algorithm": "spatial_stratified",
    }


def _all_train_spatial_stratified_split(
    minerals: pd.DataFrame,
    *,
    n_clusters: int = 10,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    assignment = _assign_spatial_stratified_mineral_folds(
        minerals,
        n_clusters=n_clusters,
        n_folds=n_folds,
        random_state=random_state,
    )
    frame = assignment["frame"].reset_index(drop=True)
    empty = frame.iloc[0:0].copy()
    if "spatial_fold" in frame.columns:
        all_fold_ids = pd.to_numeric(frame["spatial_fold"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    else:
        all_fold_ids = np.full(len(frame), -1, dtype=np.int64)
    valid_fold_ids = sorted(int(item) for item in np.unique(all_fold_ids) if int(item) >= 0)
    return {
        "train": frame,
        "test": empty,
        "n_clusters": int(assignment.get("n_clusters", n_clusters)),
        "train_ratio": 1.0,
        "fold_count": int(assignment.get("fold_count", n_folds)),
        "cluster_ids": list(assignment.get("cluster_ids", [])),
        "cluster_centers": list(assignment.get("cluster_centers", [])),
        "train_fold_ids": valid_fold_ids,
        "test_fold_ids": [],
        "val_fold_ids": [],
        "cluster_summaries": list(assignment.get("cluster_summaries", [])),
        "algorithm": "spatial_stratified",
    }


def _enrich_mineral_split_basins(mineral_split: dict, basin_grid: dict) -> dict:
    """Attach basin IDs and enforce whole-catchment train/test assignment."""
    if mineral_split is None or basin_grid is None:
        return mineral_split
    if apply_whole_basin_split is not None:
        return apply_whole_basin_split(mineral_split, basin_grid)
    # Fallback: only attach basin ids without reassignment.
    out = dict(mineral_split)
    if sample_basin_ids_at_coords is None:
        return out
    for key in ("train", "test"):
        frame = out.get(key)
        if frame is None or len(frame) == 0:
            continue
        frame = frame.copy()
        xy = frame[["x", "y"]].to_numpy(dtype=np.float64)
        frame["basin_id"] = sample_basin_ids_at_coords(basin_grid, xy)
        out[key] = frame
    test_basins = set()
    train_basins = set()
    for key, bucket in (("test", test_basins), ("train", train_basins)):
        frame = out.get(key)
        if frame is None or "basin_id" not in getattr(frame, "columns", []):
            continue
        vals = pd.to_numeric(frame["basin_id"], errors="coerce").dropna().to_numpy(dtype=np.float64)
        for v in vals:
            if np.isfinite(v):
                bucket.add(int(round(float(v))))
    # Train priority: basins with train deposits stay train.
    test_basins = {b for b in test_basins if b not in train_basins}
    out["test_basin_ids"] = sorted(test_basins)
    out["train_basin_ids"] = sorted(train_basins)
    out["basin_split_rule"] = "whole_basin_train_priority_fallback"
    return out


def _split_unlabeled_indices_by_basin(
    *,
    coords: np.ndarray,
    mineral_split: dict,
    basin_grid: dict,
    buffer_distance: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Hard-isolate unlabeled patches by whole catchment basins (no basin cutting).

    Rule (atomic catchment, train priority):
    - basins with any train deposit -> wholly train
    - basins with only test deposits -> wholly test
    - empty basins (no deposits): follow nearest deposit's train/test side
    - invalid/nan basin ids: gray (excluded)
    Mineral frames should already be reassigned by ``apply_whole_basin_split``.
    """
    if coords is None or len(coords) == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, 0

    coords = np.asarray(coords, dtype=np.float64)[:, :2]
    if sample_basin_ids_at_coords is None:
        empty = np.array([], dtype=np.int64)
        return empty, empty, 0
    basin_ids = sample_basin_ids_at_coords(basin_grid, coords)
    test_basins = {
        int(b)
        for b in (mineral_split.get("test_basin_ids") or [])
        if b is not None and str(b).strip() not in {"", "nan", "None"}
    }
    train_basins = {
        int(b)
        for b in (mineral_split.get("train_basin_ids") or [])
        if b is not None and str(b).strip() not in {"", "nan", "None"}
    }
    # Prefer recomputing from already-reassigned mineral frames when available.
    for key, bucket in (("test", test_basins), ("train", train_basins)):
        frame = mineral_split.get(key)
        if frame is None or "basin_id" not in getattr(frame, "columns", []):
            continue
        vals = pd.to_numeric(frame["basin_id"], errors="coerce").dropna().to_numpy(dtype=np.float64)
        for v in vals:
            if np.isfinite(v):
                bucket.add(int(round(float(v))))
    # Keep basins atomic: train list wins if lists ever overlap.
    test_basins = {b for b in test_basins if b not in train_basins}

    mineral_coords = []
    mineral_is_test = []
    for key, is_test in (("test", True), ("train", False)):
        frame = mineral_split.get(key)
        if frame is None or len(frame) == 0:
            continue
        xy = frame[["x", "y"]].to_numpy(dtype=np.float64)
        mineral_coords.append(xy)
        mineral_is_test.extend([is_test] * len(xy))
    if mineral_coords:
        mineral_coords_arr = np.concatenate(mineral_coords, axis=0)
        mineral_is_test_arr = np.asarray(mineral_is_test, dtype=bool)
    else:
        mineral_coords_arr = np.empty((0, 2), dtype=np.float64)
        mineral_is_test_arr = np.empty((0,), dtype=bool)

    train_mask = np.zeros(len(coords), dtype=bool)
    test_mask = np.zeros(len(coords), dtype=bool)
    gray_mask = np.zeros(len(coords), dtype=bool)

    valid = np.isfinite(basin_ids)
    gray_mask |= ~valid

    # Empty basins (no deposit): assign whole basin by nearest deposit to basin centroid.
    # Vectorized membership + KDTree centroids (avoids slow per-basin full-array scans).
    empty_basin_side: dict = {}
    basin_int_ids = np.full(len(basin_ids), -1, dtype=np.int64)
    if np.any(valid):
        basin_int_ids[valid] = np.rint(basin_ids[valid]).astype(np.int64)

    def _membership(ids: set) -> np.ndarray:
        mask = np.zeros(len(basin_int_ids), dtype=bool)
        if not ids:
            return mask
        arr = np.fromiter(sorted(ids), dtype=np.int64)
        idx = np.searchsorted(arr, basin_int_ids)
        ok = valid & (idx < len(arr))
        mask[ok] = arr[idx[ok]] == basin_int_ids[ok]
        return mask

    in_train = _membership(train_basins)
    in_test = _membership(test_basins) & (~in_train)
    assigned = in_test | in_train
    unknown_mask = valid & (~assigned)
    if len(mineral_coords_arr) > 0 and np.any(unknown_mask):
        from scipy.spatial import cKDTree

        uniq, inv = np.unique(basin_int_ids[unknown_mask], return_inverse=True)
        coords_u = coords[unknown_mask]
        sums = np.zeros((len(uniq), 2), dtype=np.float64)
        counts = np.zeros(len(uniq), dtype=np.float64)
        np.add.at(sums, inv, coords_u)
        np.add.at(counts, inv, 1.0)
        centroids = sums / np.maximum(counts[:, None], 1.0)
        tree = cKDTree(mineral_coords_arr)
        _, nn = tree.query(centroids, k=1)
        nn = np.atleast_1d(nn)
        for i, basin_int in enumerate(uniq.tolist()):
            empty_basin_side[int(basin_int)] = bool(mineral_is_test_arr[int(nn[i])])

    empty_test = np.zeros(len(basin_ids), dtype=bool)
    empty_train = np.zeros(len(basin_ids), dtype=bool)
    if empty_basin_side and np.any(unknown_mask):
        u_ids = basin_int_ids[unknown_mask]
        uniq2, inv2 = np.unique(u_ids, return_inverse=True)
        side_per = np.array([bool(empty_basin_side[int(b)]) for b in uniq2.tolist()], dtype=bool)
        sides = side_per[inv2]
        empty_test[unknown_mask] = sides
        empty_train[unknown_mask] = ~sides

    test_mask |= in_test | empty_test
    train_mask |= in_train | empty_train
    gray_mask |= valid & (~test_mask) & (~train_mask)

    # Train/test isolation strip: drop unlabeled near the opposite-side deposits.
    # Not a positive-sample halo (that is buffer_radius / --buffer-radius in metres).
    # OGR M/N/P UI defaults halo to 500 m; pass 0 to disable.
    if float(buffer_distance) > 0 and len(mineral_coords_arr) > 0:
        from scipy.spatial import cKDTree

        train_xy = mineral_coords_arr[~mineral_is_test_arr] if len(mineral_is_test_arr) else np.empty((0, 2))
        test_xy = mineral_coords_arr[mineral_is_test_arr] if len(mineral_is_test_arr) else np.empty((0, 2))
        distance = float(buffer_distance)
        if len(train_xy) and len(test_xy):
            dist_to_train, _ = cKDTree(train_xy).query(coords, k=1)
            dist_to_test, _ = cKDTree(test_xy).query(coords, k=1)
            gray_mask |= train_mask & (dist_to_test <= distance)
            gray_mask |= test_mask & (dist_to_train <= distance)
            train_mask &= ~gray_mask
            test_mask &= ~gray_mask

    return (
        np.where(train_mask)[0].astype(np.int64),
        np.where(test_mask)[0].astype(np.int64),
        int(np.sum(gray_mask)),
    )


def _split_unlabeled_indices_by_hard_clusters(
    *,
    coords: np.ndarray,
    mineral_split: dict,
    buffer_distance: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if coords is None or len(coords) == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, 0

    centers = np.asarray(mineral_split.get("cluster_centers", []), dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] < 2 or len(centers) == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, 0

    coords = np.asarray(coords, dtype=np.float64)[:, :2]
    centers = centers[:, :2]
    finite_center_mask = np.isfinite(centers).all(axis=1)
    if not np.any(finite_center_mask):
        empty = np.array([], dtype=np.int64)
        return empty, empty, 0
    usable_centers = centers.copy()
    if np.any(~finite_center_mask):
        fallback = np.nanmean(centers, axis=0)
        if not np.all(np.isfinite(fallback)):
            fallback = np.mean(coords, axis=0)
        usable_centers[~finite_center_mask] = fallback

    distances = np.sqrt(np.sum((coords[:, None, :] - usable_centers[None, :, :]) ** 2, axis=2))
    nearest_indices = np.argmin(distances, axis=1).astype(np.int64)
    center_ids = np.asarray(mineral_split.get("cluster_center_ids", []), dtype=np.int64).reshape(-1)
    if len(center_ids) == len(usable_centers):
        nearest_cluster_ids = center_ids[nearest_indices]
    else:
        nearest_cluster_ids = nearest_indices
    sorted_distances = np.sort(distances, axis=1)
    if sorted_distances.shape[1] > 1:
        gray_mask = (sorted_distances[:, 1] - sorted_distances[:, 0]) <= float(max(buffer_distance, 0.0))
    else:
        gray_mask = np.zeros(len(coords), dtype=bool)

    train_clusters = set(int(item) for item in mineral_split.get("train_cluster_ids", mineral_split.get("train_camp_ids", [])))
    test_clusters = set(int(item) for item in mineral_split.get("test_cluster_ids", mineral_split.get("test_camp_ids", [])))
    train_mask = np.asarray([cluster_id in train_clusters for cluster_id in nearest_cluster_ids], dtype=bool) & (~gray_mask)
    test_mask = np.asarray([cluster_id in test_clusters for cluster_id in nearest_cluster_ids], dtype=bool) & (~gray_mask)
    return np.where(train_mask)[0].astype(np.int64), np.where(test_mask)[0].astype(np.int64), int(np.sum(gray_mask))


def _split_unlabeled_indices_by_fault(
    *,
    coords: np.ndarray,
    mineral_split: dict,
    buffer_distance: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Hard-isolate unlabeled patches by the same held-out fault_id as minerals."""
    if coords is None or len(coords) == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, 0
    coords = np.asarray(coords, dtype=np.float64)[:, :2]
    train_faults = {str(v).strip() for v in mineral_split.get("train_fault_ids", []) if str(v).strip()}
    test_faults = {str(v).strip() for v in mineral_split.get("test_fault_ids", []) if str(v).strip()}
    if not test_faults:
        empty = np.array([], dtype=np.int64)
        return empty, empty, 0

    fault_ids = None
    distances = None
    vertices = np.asarray(mineral_split.get("fault_line_vertices", []), dtype=np.float64)
    vertex_ids = np.asarray(mineral_split.get("fault_line_vertex_ids", []), dtype=object).reshape(-1)
    if vertices.ndim == 2 and vertices.shape[1] >= 2 and len(vertices) == len(vertex_ids) and len(vertices) > 0:
        assigned = []
        dist_list = []
        for xy in coords:
            d = np.sqrt(np.sum((vertices[:, :2] - xy) ** 2, axis=1))
            idx = int(np.argmin(d))
            assigned.append(str(vertex_ids[idx]))
            dist_list.append(float(d[idx]))
        fault_ids = np.asarray(assigned, dtype=object)
        distances = np.asarray(dist_list, dtype=np.float64)
    else:
        # Fallback: nearest mineral deposit's fault_id
        train_df = mineral_split.get("train")
        test_df = mineral_split.get("test")
        parts = []
        if train_df is not None and len(train_df) > 0 and "fault_id" in train_df.columns:
            parts.append(train_df)
        if test_df is not None and len(test_df) > 0 and "fault_id" in test_df.columns:
            parts.append(test_df)
        if not parts:
            empty = np.array([], dtype=np.int64)
            return empty, empty, 0
        mineral_frame = pd.concat(parts, ignore_index=True)
        mineral_coords = mineral_frame[["x", "y"]].to_numpy(dtype=np.float64)
        mineral_faults = mineral_frame["fault_id"].astype(str).str.strip().to_numpy()
        from scipy.spatial import cKDTree

        tree = cKDTree(mineral_coords)
        dist_nn, nn = tree.query(coords, k=1)
        fault_ids = mineral_faults[np.asarray(nn, dtype=np.int64)]
        distances = np.asarray(dist_nn, dtype=np.float64)

    train_mask = np.asarray([str(fid).strip() in train_faults for fid in fault_ids], dtype=bool)
    test_mask = np.asarray([str(fid).strip() in test_faults for fid in fault_ids], dtype=bool)
    gray_mask = np.zeros(len(coords), dtype=bool)
    distance = float(max(buffer_distance, 0.0))
    if distance > 0 and np.any(train_mask) and np.any(test_mask) and distances is not None:
        # Gray-zone near opposite-set mineral deposits if available
        train_df = mineral_split.get("train")
        test_df = mineral_split.get("test")
        if train_df is not None and test_df is not None and len(train_df) and len(test_df):
            dist_to_test = _min_distance_to_points(coords, test_df[["x", "y"]].to_numpy(dtype=np.float64))
            dist_to_train = _min_distance_to_points(coords, train_df[["x", "y"]].to_numpy(dtype=np.float64))
            gray_mask |= train_mask & (dist_to_test <= distance)
            gray_mask |= test_mask & (dist_to_train <= distance)

    return (
        np.where(train_mask & (~gray_mask))[0].astype(np.int64),
        np.where(test_mask & (~gray_mask))[0].astype(np.int64),
        int(np.sum(gray_mask)),
    )


def _split_unlabeled_indices_by_variogram_blocks(
    *,
    coords: np.ndarray,
    mineral_split: dict,
    buffer_distance: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Hard-isolate unlabeled patches by the same variogram grid cell as minerals."""
    if coords is None or len(coords) == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, 0
    if assign_coords_to_variogram_blocks is None:
        empty = np.array([], dtype=np.int64)
        return empty, empty, 0

    coords = np.asarray(coords, dtype=np.float64)[:, :2]
    grid_x0 = float(mineral_split.get("grid_x0", 0.0) or 0.0)
    grid_y0 = float(mineral_split.get("grid_y0", 0.0) or 0.0)
    block_size = float(mineral_split.get("block_size_m", mineral_split.get("variogram_range_m", 2000.0)) or 2000.0)
    test_pairs = {
        (int(p[0]), int(p[1]))
        for p in mineral_split.get("test_block_pairs", [])
        if isinstance(p, (list, tuple)) and len(p) >= 2
    }
    if not test_pairs:
        held = mineral_split.get("held_out_block_pair")
        if isinstance(held, (list, tuple)) and len(held) >= 2:
            test_pairs = {(int(held[0]), int(held[1]))}
    if not test_pairs:
        empty = np.array([], dtype=np.int64)
        return empty, empty, 0

    ix, iy = assign_coords_to_variogram_blocks(
        coords, grid_x0=grid_x0, grid_y0=grid_y0, block_size_m=block_size
    )
    test_mask = np.asarray([(int(a), int(b)) in test_pairs for a, b in zip(ix.tolist(), iy.tolist())], dtype=bool)
    train_mask = ~test_mask

    gray_mask = np.zeros(len(coords), dtype=bool)
    distance = float(max(buffer_distance, 0.0))
    if distance > 0:
        # Strip a buffer strip along block boundaries of the held-out cell.
        for pair in test_pairs:
            x_left = grid_x0 + pair[0] * block_size
            x_right = x_left + block_size
            y_bottom = grid_y0 + pair[1] * block_size
            y_top = y_bottom + block_size
            near_x = (np.abs(coords[:, 0] - x_left) <= distance) | (np.abs(coords[:, 0] - x_right) <= distance)
            near_y = (np.abs(coords[:, 1] - y_bottom) <= distance) | (np.abs(coords[:, 1] - y_top) <= distance)
            inside_y = (coords[:, 1] >= y_bottom - distance) & (coords[:, 1] <= y_top + distance)
            inside_x = (coords[:, 0] >= x_left - distance) & (coords[:, 0] <= x_right + distance)
            gray_mask |= (near_x & inside_y) | (near_y & inside_x)

    return (
        np.where(train_mask & (~gray_mask))[0].astype(np.int64),
        np.where(test_mask & (~gray_mask))[0].astype(np.int64),
        int(np.sum(gray_mask)),
    )


def _build_spatial_unit_geometry(spatial_split_mode: str, mineral_split: dict) -> dict:
    """Compact geometry payload for sample_split preview overlays."""
    mode = str(spatial_split_mode or "")
    geometry = {
        "mode": mode,
        "algorithm": str(mineral_split.get("algorithm", mode)),
    }
    if mode in {"leave_one_camp", "spatial_hard", "spatial_cluster"}:
        geometry["cluster_centers"] = mineral_split.get("cluster_centers", [])
        geometry["cluster_center_ids"] = mineral_split.get("cluster_center_ids", [])
        geometry["train_unit_ids"] = mineral_split.get("train_cluster_ids", mineral_split.get("train_camp_ids", []))
        geometry["test_unit_ids"] = mineral_split.get("test_cluster_ids", mineral_split.get("test_camp_ids", []))
        geometry["camp_source"] = mineral_split.get("camp_source", "")
    if mode == "leave_one_fault":
        geometry["train_unit_ids"] = mineral_split.get("train_fault_ids", [])
        geometry["test_unit_ids"] = mineral_split.get("test_fault_ids", [])
        geometry["fault_line_vertices"] = mineral_split.get("fault_line_vertices", [])
        geometry["fault_line_vertex_ids"] = mineral_split.get("fault_line_vertex_ids", [])
        geometry["assignment_source"] = mineral_split.get("assignment_source", "")
    if mode == "variogram_block_cv":
        geometry["grid_x0"] = float(mineral_split.get("grid_x0", 0.0) or 0.0)
        geometry["grid_y0"] = float(mineral_split.get("grid_y0", 0.0) or 0.0)
        geometry["block_size_m"] = float(mineral_split.get("block_size_m", 0.0) or 0.0)
        geometry["held_out_block_pair"] = mineral_split.get("held_out_block_pair", [])
        geometry["train_block_pairs"] = mineral_split.get("train_block_pairs", [])
        geometry["test_block_pairs"] = mineral_split.get("test_block_pairs", [])
        geometry["single_block_warning"] = bool(mineral_split.get("single_block_warning", False))
    return geometry


def _split_unlabeled_indices_by_spatial_stratified_folds(
    *,
    coords: np.ndarray,
    mineral_split: dict,
    buffer_distance: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if coords is None or len(coords) == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, 0

    mineral_coords = np.asarray(mineral_split.get("mineral_coords", []), dtype=np.float64)
    mineral_fold_ids = np.asarray(mineral_split.get("mineral_fold_ids", []), dtype=np.int64).reshape(-1)
    if mineral_coords.ndim != 2 or mineral_coords.shape[1] < 2 or len(mineral_coords) == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, 0
    if len(mineral_coords) != len(mineral_fold_ids):
        empty = np.array([], dtype=np.int64)
        return empty, empty, 0

    coords = np.asarray(coords, dtype=np.float64)[:, :2]
    assigned_fold_ids = _nearest_mineral_fold_ids(coords, mineral_coords[:, :2], mineral_fold_ids)
    train_folds = set(int(item) for item in mineral_split.get("train_fold_ids", []))
    test_folds = set(int(item) for item in mineral_split.get("test_fold_ids", []))
    train_mask = np.asarray([int(fold_id) in train_folds for fold_id in assigned_fold_ids], dtype=bool)
    test_mask = np.asarray([int(fold_id) in test_folds for fold_id in assigned_fold_ids], dtype=bool)

    gray_mask = np.zeros(len(coords), dtype=bool)
    distance = float(max(buffer_distance, 0.0))
    if distance > 0 and np.any(train_mask) and np.any(test_mask):
        train_mineral_mask = np.asarray([int(fold_id) in train_folds for fold_id in mineral_fold_ids], dtype=bool)
        test_mineral_mask = np.asarray([int(fold_id) in test_folds for fold_id in mineral_fold_ids], dtype=bool)
        if np.any(test_mineral_mask):
            dist_to_test = _min_distance_to_points(coords, mineral_coords[test_mineral_mask])
            gray_mask |= train_mask & (dist_to_test <= distance)
        if np.any(train_mineral_mask):
            dist_to_train = _min_distance_to_points(coords, mineral_coords[train_mineral_mask])
            gray_mask |= test_mask & (dist_to_train <= distance)

    return (
        np.where(train_mask & (~gray_mask))[0].astype(np.int64),
        np.where(test_mask & (~gray_mask))[0].astype(np.int64),
        int(np.sum(gray_mask)),
    )


def _split_unlabeled_indices(
    *,
    coords: np.ndarray,
    train_ratio: float,
    n_clusters: int,
    buffer_distance: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if coords is None or len(coords) == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty
    labels = np.full(len(coords), -1, dtype=np.int32)
    train_idx, test_idx, _ = _spatial_cluster_split_indices(
        coords,
        labels,
        n_clusters=n_clusters,
        train_ratio=train_ratio,
        buffer_distance=buffer_distance,
        random_state=random_state,
    )
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def _sample_split_arrays(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    sample_ratio: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if sample_ratio >= 1.0:
        return x_arr, y_arr
    rng = np.random.default_rng(int(random_state))
    y_flat = np.asarray(y_arr).reshape(-1)
    pos_indices = np.where(y_flat == 1)[0]
    neg_indices = np.where(y_flat == -1)[0]
    pos_keep = min(len(pos_indices), max(1, int(round(len(pos_indices) * sample_ratio)))) if len(pos_indices) else 0
    neg_keep = min(len(neg_indices), max(1, int(round(len(neg_indices) * sample_ratio)))) if len(neg_indices) else 0
    pos_selected = rng.choice(pos_indices, pos_keep, replace=False) if pos_keep > 0 else np.array([], dtype=np.int64)
    neg_selected = rng.choice(neg_indices, neg_keep, replace=False) if neg_keep > 0 else np.array([], dtype=np.int64)
    selected = np.concatenate((pos_selected, neg_selected))
    if len(selected) == 0:
        return x_arr, y_arr
    rng.shuffle(selected)
    return x_arr[selected], y_flat[selected]


def _sample_class_indices(y_arr: np.ndarray, sample_ratio: float, random_state: int) -> np.ndarray:
    y_flat = np.asarray(y_arr).reshape(-1)
    if sample_ratio >= 1.0:
        return np.arange(len(y_flat), dtype=np.int64)
    rng = np.random.default_rng(int(random_state))
    pos_indices = np.where(y_flat == 1)[0]
    neg_indices = np.where(y_flat == -1)[0]
    pos_keep = min(len(pos_indices), max(1, int(round(len(pos_indices) * sample_ratio)))) if len(pos_indices) else 0
    neg_keep = min(len(neg_indices), max(1, int(round(len(neg_indices) * sample_ratio)))) if len(neg_indices) else 0
    pos_selected = rng.choice(pos_indices, pos_keep, replace=False) if pos_keep > 0 else np.array([], dtype=np.int64)
    neg_selected = rng.choice(neg_indices, neg_keep, replace=False) if neg_keep > 0 else np.array([], dtype=np.int64)
    selected = np.concatenate((pos_selected, neg_selected))
    if len(selected) == 0:
        return np.arange(len(y_flat), dtype=np.int64)
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def _sample_dev_indices_by_mineral(
    y_arr: np.ndarray,
    mineral_ids: np.ndarray,
    sample_ratio: float,
    random_state: int,
) -> np.ndarray:
    y_flat = np.asarray(y_arr).reshape(-1)
    mineral_ids = np.asarray(mineral_ids).reshape(-1)
    if len(y_flat) != len(mineral_ids):
        return _sample_class_indices(y_flat, sample_ratio, random_state)
    if sample_ratio >= 1.0:
        return np.arange(len(y_flat), dtype=np.int64)

    rng = np.random.default_rng(int(random_state))
    selected_parts = []

    positive_indices = np.where(y_flat == 1)[0]
    if len(positive_indices) > 0:
        positive_mineral_ids = mineral_ids[positive_indices]
        valid_positive_mask = positive_mineral_ids >= 0
        grouped_positive_indices = positive_indices[valid_positive_mask]
        grouped_positive_ids = positive_mineral_ids[valid_positive_mask]
        for mineral_id in np.unique(grouped_positive_ids):
            group_indices = grouped_positive_indices[grouped_positive_ids == mineral_id]
            keep_count = min(len(group_indices), max(1, int(round(len(group_indices) * sample_ratio))))
            group_selected = rng.choice(group_indices, keep_count, replace=False)
            selected_parts.append(np.asarray(group_selected, dtype=np.int64))
        ungrouped_positive = positive_indices[~valid_positive_mask]
        if len(ungrouped_positive) > 0:
            keep_count = min(len(ungrouped_positive), max(1, int(round(len(ungrouped_positive) * sample_ratio))))
            selected_parts.append(np.asarray(rng.choice(ungrouped_positive, keep_count, replace=False), dtype=np.int64))

    negative_indices = np.where(y_flat == -1)[0]
    if len(negative_indices) > 0:
        keep_count = min(len(negative_indices), max(1, int(round(len(negative_indices) * sample_ratio))))
        selected_parts.append(np.asarray(rng.choice(negative_indices, keep_count, replace=False), dtype=np.int64))

    if not selected_parts:
        return np.arange(len(y_flat), dtype=np.int64)
    selected = np.concatenate(selected_parts)
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def _build_spatial_cv_folds(
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    n_folds: int = 5,
    buffer_distance: float = 0.0,
    partition_coords: Optional[np.ndarray] = None,
    area_coords: Optional[np.ndarray] = None,
) -> list:
    coords = np.asarray(coords, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    if len(coords) != len(labels) or len(coords) == 0:
        return []
    if coords.ndim != 2 or coords.shape[1] < 2:
        return []

    partition_array = coords
    if partition_coords is not None:
        partition_array = np.asarray(partition_coords, dtype=np.float64)
        if partition_array.ndim != 2 or partition_array.shape[1] < 2 or len(partition_array) == 0:
            partition_array = coords
        else:
            partition_array = partition_array[:, :2]

    n_folds = int(max(n_folds, 2))
    axis = 0
    axis_source = "x_train_mineral_quantile"
    partition_axis_values = partition_array[:, axis]
    axis_values = coords[:, axis]
    if float(np.max(partition_axis_values)) == float(np.min(partition_axis_values)):
        axis = 1 - axis
        axis_source = "y_fallback"
        partition_axis_values = partition_array[:, axis]
        axis_values = coords[:, axis]
    if float(np.max(partition_axis_values)) == float(np.min(partition_axis_values)):
        return []

    edges = np.quantile(partition_axis_values, np.linspace(0.0, 1.0, n_folds + 1))
    edges = np.asarray(edges, dtype=np.float64)
    if np.unique(edges).size < len(edges):
        edges = np.linspace(float(np.min(partition_axis_values)), float(np.max(partition_axis_values)), n_folds + 1)
    boundaries = edges[1:-1]
    block_ids = np.searchsorted(boundaries, axis_values, side="right")
    block_ids = np.clip(block_ids, 0, n_folds - 1)

    gray_mask = np.zeros(len(axis_values), dtype=bool)
    distance = float(max(buffer_distance, 0.0))
    if distance > 0:
        for boundary in boundaries:
            gray_mask |= np.abs(axis_values - float(boundary)) <= distance

    area_axis_values = None
    area_block_ids = None
    area_gray_mask = None
    if area_coords is not None:
        area_array = np.asarray(area_coords, dtype=np.float64)
        if area_array.ndim == 2 and area_array.shape[1] >= 2 and len(area_array) > 0:
            area_axis_values = area_array[:, axis]
            area_block_ids = np.searchsorted(boundaries, area_axis_values, side="right")
            area_block_ids = np.clip(area_block_ids, 0, n_folds - 1)
            area_gray_mask = np.zeros(len(area_axis_values), dtype=bool)
            if distance > 0:
                for boundary in boundaries:
                    area_gray_mask |= np.abs(area_axis_values - float(boundary)) <= distance

    folds = []
    for fold_idx in range(n_folds):
        val_mask = (block_ids == fold_idx) & (~gray_mask)
        train_mask = (block_ids != fold_idx) & (~gray_mask)
        train_idx = np.where(train_mask)[0].astype(np.int64)
        val_idx = np.where(val_mask)[0].astype(np.int64)
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        folds.append({
            "fold": int(fold_idx),
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "train_count": int(len(train_idx)),
            "val_count": int(len(val_idx)),
            "axis": "x" if axis == 0 else "y",
            "axis_source": axis_source,
        })
        if area_block_ids is not None and area_gray_mask is not None:
            area_val_mask = (area_block_ids == fold_idx) & (~area_gray_mask)
            area_val_idx = np.where(area_val_mask)[0].astype(np.int64)
            folds[-1]["val_area_indices"] = area_val_idx.tolist()
            folds[-1]["val_area_count"] = int(len(area_val_idx))
    return folds


def _nearest_mineral_fold_ids(coords: np.ndarray, mineral_coords: np.ndarray, mineral_fold_ids: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    mineral_coords = np.asarray(mineral_coords, dtype=np.float64)
    mineral_fold_ids = np.asarray(mineral_fold_ids, dtype=np.int64).reshape(-1)
    if coords.ndim != 2 or coords.shape[1] < 2 or len(coords) == 0:
        return np.empty(0, dtype=np.int64)
    if mineral_coords.ndim != 2 or mineral_coords.shape[1] < 2 or len(mineral_coords) == 0:
        return np.full(len(coords), -1, dtype=np.int64)

    result = np.empty(len(coords), dtype=np.int64)
    chunk_size = 20000
    for start in range(0, len(coords), chunk_size):
        stop = min(start + chunk_size, len(coords))
        delta = coords[start:stop, None, :2] - mineral_coords[None, :, :2]
        distances = np.sum(delta * delta, axis=2)
        nearest = np.argmin(distances, axis=1)
        result[start:stop] = mineral_fold_ids[nearest]
    return result


def _min_distance_to_points(coords: np.ndarray, points: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2 or len(coords) == 0:
        return np.empty(0, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) == 0:
        return np.full(len(coords), np.inf, dtype=np.float64)

    result = np.empty(len(coords), dtype=np.float64)
    chunk_size = 20000
    for start in range(0, len(coords), chunk_size):
        stop = min(start + chunk_size, len(coords))
        delta = coords[start:stop, None, :2] - points[None, :, :2]
        result[start:stop] = np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1))
    return result


def _unlabeled_masks_by_whole_basin(
    *,
    basin_ids: np.ndarray,
    unlabeled_mask: np.ndarray,
    mineral_basin_ids: np.ndarray,
    mineral_cluster_ids: np.ndarray,
    train_cluster_set,
    val_cluster_set,
    coords: np.ndarray,
    mineral_coords: np.ndarray,
):
    """Assign unlabeled cells by whole-basin membership (outer train-priority rule).

    - basin with any inner-train deposit -> train
    - basin with only inner-val deposits -> val
    - empty basin (no deposit) -> nearest deposit's camp, then train/val
    - invalid basin id -> gray
    """
    unlabeled_mask = np.asarray(unlabeled_mask, dtype=bool).reshape(-1)
    n = int(len(unlabeled_mask))
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    gray_mask = np.zeros(n, dtype=bool)
    if n == 0:
        return train_mask, val_mask, gray_mask

    def _int_ids(values):
        arr = np.asarray(values if values is not None else [], dtype=np.float64).reshape(-1)
        out = np.full(len(arr), -1, dtype=np.int64)
        valid = np.isfinite(arr)
        if np.any(valid):
            out[valid] = np.rint(arr[valid]).astype(np.int64)
        return out, valid

    sample_basin, sample_valid = _int_ids(basin_ids)
    if len(sample_basin) != n:
        gray_mask |= unlabeled_mask
        return train_mask, val_mask, gray_mask
    mineral_basin, mineral_valid = _int_ids(mineral_basin_ids)
    mineral_cluster_ids = np.asarray(mineral_cluster_ids, dtype=np.int64).reshape(-1)
    if len(mineral_cluster_ids) != len(mineral_basin):
        gray_mask |= unlabeled_mask
        return train_mask, val_mask, gray_mask

    train_cluster_set = {int(item) for item in train_cluster_set}
    val_cluster_set = {int(item) for item in val_cluster_set}
    basin_has_train = set()
    basin_has_val = set()
    for basin_id, cluster_id, valid in zip(
        mineral_basin.tolist(),
        mineral_cluster_ids.tolist(),
        mineral_valid.tolist(),
    ):
        if not valid or int(cluster_id) < 0:
            continue
        basin_id = int(basin_id)
        cluster_id = int(cluster_id)
        if cluster_id in train_cluster_set:
            basin_has_train.add(basin_id)
        if cluster_id in val_cluster_set:
            basin_has_val.add(basin_id)

    gray_mask |= unlabeled_mask & (~sample_valid)
    unlabeled_valid = unlabeled_mask & sample_valid
    empty_idx = np.empty((0,), dtype=np.int64)
    if np.any(unlabeled_valid):
        idx = np.where(unlabeled_valid)[0]
        uniq, inv = np.unique(sample_basin[idx], return_inverse=True)
        has_train = np.array(
            [int(basin_id) in basin_has_train for basin_id in uniq.tolist()],
            dtype=bool,
        )
        has_val = np.array(
            [int(basin_id) in basin_has_val for basin_id in uniq.tolist()],
            dtype=bool,
        )
        point_train = has_train[inv]
        point_val = has_val[inv] & (~point_train)
        train_mask[idx[point_train]] = True
        val_mask[idx[point_val]] = True
        empty_idx = idx[~(has_train[inv] | has_val[inv])]

    if len(empty_idx) and mineral_coords is not None and len(np.asarray(mineral_coords)) > 0:
        from scipy.spatial import cKDTree

        coords_xy = np.asarray(coords, dtype=np.float64)
        mineral_xy = np.asarray(mineral_coords, dtype=np.float64)
        if coords_xy.ndim == 2 and coords_xy.shape[1] >= 2 and mineral_xy.ndim == 2 and mineral_xy.shape[1] >= 2:
            empty_basins = sample_basin[empty_idx]
            uniq, inv = np.unique(empty_basins, return_inverse=True)
            sums = np.zeros((len(uniq), 2), dtype=np.float64)
            counts = np.zeros(len(uniq), dtype=np.float64)
            np.add.at(sums, inv, coords_xy[empty_idx][:, :2])
            np.add.at(counts, inv, 1.0)
            centroids = sums / np.maximum(counts[:, None], 1.0)
            tree = cKDTree(mineral_xy[:, :2])
            _, nn = tree.query(centroids, k=1)
            nn = np.atleast_1d(np.asarray(nn, dtype=np.int64))
            nearest_camps = mineral_cluster_ids[np.clip(nn, 0, len(mineral_cluster_ids) - 1)]
            camp_per_point = nearest_camps[inv]
            is_train = np.isin(camp_per_point, list(train_cluster_set))
            is_val = np.isin(camp_per_point, list(val_cluster_set)) & (~is_train)
            train_mask[empty_idx[is_train]] = True
            val_mask[empty_idx[is_val]] = True

    gray_mask |= unlabeled_mask & (~train_mask) & (~val_mask)
    return train_mask, val_mask, gray_mask


def _basins_of_clusters(mineral_basin_ids: np.ndarray, cluster_ids: np.ndarray, cluster_set) -> set:
    basins = np.asarray(mineral_basin_ids, dtype=np.float64).reshape(-1)
    clusters = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    if len(basins) != len(clusters) or not cluster_set:
        return set()
    mask = np.isin(clusters, list(cluster_set))
    picked = basins[mask]
    finite = np.isfinite(picked)
    if not np.any(finite):
        return set()
    return set(np.rint(picked[finite]).astype(np.int64).tolist())


def _catchment_boundary_buffer_mask(
    coords: np.ndarray,
    basin_ids: np.ndarray,
    train_basins,
    val_basins,
    buffer_distance: float,
) -> np.ndarray:
    """Grey cells that sit within ``buffer_distance`` of the opposite catchment set.

    This is the manuscript inner isolation strip along catchment boundaries
    (default 500 m), not the positive-sample halo around deposits.
    """
    from scipy.spatial import cKDTree

    coords_xy = np.asarray(coords, dtype=np.float64)
    basin_arr = np.asarray(basin_ids, dtype=np.float64).reshape(-1)
    n = int(len(coords_xy)) if coords_xy.ndim == 2 else 0
    gray = np.zeros(n, dtype=bool)
    distance = float(buffer_distance)
    if n == 0 or coords_xy.shape[1] < 2 or distance <= 0 or len(basin_arr) != n:
        return gray
    train_set = {int(item) for item in (train_basins or set())}
    val_set = {int(item) for item in (val_basins or set())}
    if not train_set or not val_set:
        return gray
    basin_int = np.full(n, -1, dtype=np.int64)
    valid = np.isfinite(basin_arr)
    if np.any(valid):
        basin_int[valid] = np.rint(basin_arr[valid]).astype(np.int64)
    in_train = valid & np.isin(basin_int, list(train_set))
    in_val = valid & np.isin(basin_int, list(val_set))
    if not np.any(in_train) or not np.any(in_val):
        return gray
    train_xy = coords_xy[in_train][:, :2]
    val_xy = coords_xy[in_val][:, :2]
    dist_train, _ = cKDTree(val_xy).query(train_xy, k=1)
    dist_val, _ = cKDTree(train_xy).query(val_xy, k=1)
    train_near = np.asarray(dist_train, dtype=np.float64).reshape(-1) <= distance
    val_near = np.asarray(dist_val, dtype=np.float64).reshape(-1) <= distance
    gray[in_train] = train_near
    gray[in_val] = val_near
    return gray


def _assign_samples_to_mineral_clusters(
    coords: np.ndarray,
    mineral_ids: np.ndarray,
    mineral_coords: np.ndarray,
    mineral_cluster_ids: np.ndarray,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    mineral_ids = np.asarray(mineral_ids, dtype=np.int64).reshape(-1)
    mineral_coords = np.asarray(mineral_coords, dtype=np.float64)
    mineral_cluster_ids = np.asarray(mineral_cluster_ids, dtype=np.int64).reshape(-1)
    if len(coords) != len(mineral_ids):
        return np.empty(0, dtype=np.int64)
    if mineral_coords.ndim != 2 or mineral_coords.shape[1] < 2 or len(mineral_coords) == 0:
        return np.full(len(coords), -1, dtype=np.int64)
    if len(mineral_cluster_ids) != len(mineral_coords):
        return np.full(len(coords), -1, dtype=np.int64)

    sample_cluster_ids = np.full(len(coords), -1, dtype=np.int64)
    positive_mask = (mineral_ids >= 0) & (mineral_ids < len(mineral_cluster_ids))
    sample_cluster_ids[positive_mask] = mineral_cluster_ids[mineral_ids[positive_mask]]

    unlabeled_mask = sample_cluster_ids < 0
    if np.any(unlabeled_mask):
        nearest_cluster_ids = _nearest_mineral_fold_ids(
            coords[unlabeled_mask],
            mineral_coords,
            mineral_cluster_ids,
        )
        sample_cluster_ids[unlabeled_mask] = nearest_cluster_ids
    return sample_cluster_ids


def _ensure_unit_ids_on_train_minerals(
    train_minerals: pd.DataFrame,
    spatial_split_mode: str,
) -> pd.DataFrame:
    """Ensure train minerals carry a numeric ``kmeans_cluster`` unit id for hard CV.

    leave_one_camp / variogram already provide camp/block ids; leave_one_fault uses
    string fault_id and must be factorized into dense integers.
    """
    if train_minerals is None or len(train_minerals) == 0:
        return train_minerals
    frame = train_minerals.reset_index(drop=True).copy()
    mode = str(spatial_split_mode or "").strip().lower()

    if mode == "leave_one_fault":
        if "fault_id" in frame.columns:
            codes, _ = pd.factorize(frame["fault_id"].astype(str).str.strip(), sort=True)
            frame["kmeans_cluster"] = np.asarray(codes, dtype=np.int64)
            frame["unit_id"] = frame["kmeans_cluster"]
            frame["unit_label"] = frame["fault_id"].astype(str).str.strip()
            return frame
    if mode == "variogram_block_cv":
        if "block_id" in frame.columns:
            frame["kmeans_cluster"] = pd.to_numeric(frame["block_id"], errors="coerce").fillna(-1).astype(np.int64)
            frame["unit_id"] = frame["kmeans_cluster"]
            return frame
    if mode == "leave_one_camp":
        if "camp_id" in frame.columns:
            frame["kmeans_cluster"] = pd.to_numeric(frame["camp_id"], errors="coerce").fillna(-1).astype(np.int64)
            frame["unit_id"] = frame["kmeans_cluster"]
            return frame

    if "kmeans_cluster" in frame.columns:
        frame["kmeans_cluster"] = pd.to_numeric(frame["kmeans_cluster"], errors="coerce").fillna(-1).astype(np.int64)
        frame["unit_id"] = frame["kmeans_cluster"]
        return frame
    if "camp_id" in frame.columns:
        frame["kmeans_cluster"] = pd.to_numeric(frame["camp_id"], errors="coerce").fillna(-1).astype(np.int64)
        frame["unit_id"] = frame["kmeans_cluster"]
        return frame
    if "block_id" in frame.columns:
        frame["kmeans_cluster"] = pd.to_numeric(frame["block_id"], errors="coerce").fillna(-1).astype(np.int64)
        frame["unit_id"] = frame["kmeans_cluster"]
        return frame

    # Last resort: each mineral is its own unit (still hard-isolates points, not ideal)
    frame["kmeans_cluster"] = np.arange(len(frame), dtype=np.int64)
    frame["unit_id"] = frame["kmeans_cluster"]
    return frame


def _build_hard_isolated_spatial_cv_folds(
    coords: np.ndarray,
    labels: np.ndarray,
    mineral_ids: np.ndarray,
    train_minerals: pd.DataFrame,
    *,
    n_folds: int = 5,
    buffer_distance: float = 0.0,
    area_coords: Optional[np.ndarray] = None,
    area_mineral_ids: Optional[np.ndarray] = None,
    sample_to_area_indices: Optional[np.ndarray] = None,
    strategy_name: str = "hard_cluster",
    unit_fold_mode: str = "packed",
    random_state: int = 42,
    footprint_embargo_grid_cells: int = 0,
    area_basin_ids: Optional[np.ndarray] = None,
    sample_basin_ids: Optional[np.ndarray] = None,
    mineral_basin_ids: Optional[np.ndarray] = None,
) -> list:
    coords = np.asarray(coords, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    mineral_ids = np.asarray(mineral_ids, dtype=np.int64).reshape(-1)
    if len(coords) != len(labels) or len(coords) != len(mineral_ids) or len(coords) == 0:
        return []
    if train_minerals is None or len(train_minerals) == 0:
        return []

    mineral_frame = train_minerals.reset_index(drop=True).copy()
    mineral_coords = mineral_frame[["x", "y"]].to_numpy(dtype=np.float64)
    if mineral_coords.ndim != 2 or mineral_coords.shape[1] < 2 or len(mineral_coords) == 0:
        return []
    if "kmeans_cluster" not in mineral_frame.columns:
        return []

    cluster_ids = pd.to_numeric(mineral_frame["kmeans_cluster"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    valid_cluster_ids = sorted(int(item) for item in np.unique(cluster_ids) if int(item) >= 0)
    if len(valid_cluster_ids) < 2:
        return []

    requested_folds = int(max(n_folds, 1))
    fold_mode = str(unit_fold_mode or "packed").strip().lower()

    cluster_centers = []
    for cluster_id in valid_cluster_ids:
        cluster_coords = mineral_coords[cluster_ids == cluster_id]
        if len(cluster_coords) == 0:
            continue
        cluster_centers.append((cluster_id, np.mean(cluster_coords[:, :2], axis=0)))
    if len(cluster_centers) < 2:
        return []

    centers_array = np.asarray([item[1] for item in cluster_centers], dtype=np.float64)
    x_span = float(np.max(centers_array[:, 0]) - np.min(centers_array[:, 0]))
    y_span = float(np.max(centers_array[:, 1]) - np.min(centers_array[:, 1]))
    axis = 0 if x_span >= y_span else 1
    ordered_clusters = [
        int(cluster_centers[index][0])
        for index in np.argsort(centers_array[:, axis], kind="mergesort")
    ]

    if fold_mode in {"leave_one_unit", "leave_one_camp"}:
        # One complete unit/camp per validation fold (OOF-inner).
        cluster_fold_parts = [[int(cluster_id)] for cluster_id in ordered_clusters]
        effective_folds = len(cluster_fold_parts)
        axis_source = "leave_one_unit"
    elif fold_mode in {"random_one_unit", "random_one_camp_val"}:
        rng = np.random.RandomState(int(random_state))
        chosen = int(ordered_clusters[int(rng.randint(0, len(ordered_clusters)))])
        cluster_fold_parts = [[chosen]]
        effective_folds = 1
        axis_source = "random_one_unit"
    else:
        effective_folds = min(max(requested_folds, 2), len(valid_cluster_ids))
        cluster_fold_parts = [
            [int(item) for item in part.tolist()]
            for part in np.array_split(np.asarray(ordered_clusters, dtype=np.int64), effective_folds)
            if len(part) > 0
        ]
        axis_source = "outer_train_unit_centroid_order"
    if len(cluster_fold_parts) < 1:
        return []
    if fold_mode not in {"random_one_unit", "random_one_camp_val"} and len(cluster_fold_parts) < 2:
        return []

    area_coords = np.asarray(area_coords if area_coords is not None else coords, dtype=np.float64)
    area_mineral_ids = np.asarray(
        area_mineral_ids if area_mineral_ids is not None else mineral_ids,
        dtype=np.int64,
    ).reshape(-1)
    if len(area_coords) != len(area_mineral_ids):
        return []

    area_cluster_ids = _assign_samples_to_mineral_clusters(
        area_coords,
        area_mineral_ids,
        mineral_coords,
        cluster_ids,
    )
    if len(area_cluster_ids) != len(area_coords):
        return []

    if sample_to_area_indices is not None:
        sample_to_area_indices = np.asarray(sample_to_area_indices, dtype=np.int64).reshape(-1)
        if len(sample_to_area_indices) != len(coords):
            return []
        if np.any(sample_to_area_indices < 0) or np.any(sample_to_area_indices >= len(area_cluster_ids)):
            return []
        sample_cluster_ids = area_cluster_ids[sample_to_area_indices]
        sample_gray_source = sample_to_area_indices
    else:
        sample_cluster_ids = _assign_samples_to_mineral_clusters(
            coords,
            mineral_ids,
            mineral_coords,
            cluster_ids,
        )
        if len(sample_cluster_ids) != len(coords):
            return []
        sample_gray_source = None

    area_unlabeled_mask = area_mineral_ids < 0
    sample_unlabeled_mask = labels != 1
    distance = float(max(buffer_distance, 0.0))
    strategy = str(strategy_name or "hard_cluster")
    area_basin_arr = None if area_basin_ids is None else np.asarray(area_basin_ids, dtype=np.float64).reshape(-1)
    sample_basin_arr = None if sample_basin_ids is None else np.asarray(sample_basin_ids, dtype=np.float64).reshape(-1)
    mineral_basin_arr = (
        None if mineral_basin_ids is None else np.asarray(mineral_basin_ids, dtype=np.float64).reshape(-1)
    )
    use_whole_basin_unlabeled = (
        area_basin_arr is not None
        and mineral_basin_arr is not None
        and len(area_basin_arr) == len(area_coords)
        and len(mineral_basin_arr) == len(mineral_coords)
    )
    unlabeled_isolation = "whole_basin" if use_whole_basin_unlabeled else "nearest_mineral"

    folds = []
    all_cluster_set = set(valid_cluster_ids)
    for fold_idx, val_cluster_ids in enumerate(cluster_fold_parts):
        val_cluster_set = set(int(item) for item in val_cluster_ids)
        train_cluster_set = all_cluster_set - val_cluster_set
        if not val_cluster_set or not train_cluster_set:
            continue

        val_mineral_mask = np.isin(cluster_ids, list(val_cluster_set))
        train_mineral_mask = np.isin(cluster_ids, list(train_cluster_set))
        if not np.any(val_mineral_mask) or not np.any(train_mineral_mask):
            continue

        area_positive_mask = area_mineral_ids >= 0
        area_val_pos = area_positive_mask & np.isin(area_cluster_ids, list(val_cluster_set))
        area_train_pos = area_positive_mask & np.isin(area_cluster_ids, list(train_cluster_set))
        area_u_train = area_u_val = area_u_gray = None
        if use_whole_basin_unlabeled:
            area_u_train, area_u_val, area_u_gray = _unlabeled_masks_by_whole_basin(
                basin_ids=area_basin_arr,
                unlabeled_mask=area_unlabeled_mask,
                mineral_basin_ids=mineral_basin_arr,
                mineral_cluster_ids=cluster_ids,
                train_cluster_set=train_cluster_set,
                val_cluster_set=val_cluster_set,
                coords=area_coords,
                mineral_coords=mineral_coords,
            )
            area_val_mask = area_val_pos | area_u_val
            area_train_mask = area_train_pos | area_u_train
            area_gray_mask = np.asarray(area_u_gray, dtype=bool).copy()
        else:
            area_val_mask = np.isin(area_cluster_ids, list(val_cluster_set))
            area_train_mask = np.isin(area_cluster_ids, list(train_cluster_set))
            area_gray_mask = np.zeros(len(area_coords), dtype=bool)
        if distance > 0:
            if use_whole_basin_unlabeled:
                train_basins = _basins_of_clusters(mineral_basin_arr, cluster_ids, train_cluster_set)
                val_basins = _basins_of_clusters(mineral_basin_arr, cluster_ids, val_cluster_set)
                area_boundary = _catchment_boundary_buffer_mask(
                    area_coords,
                    area_basin_arr,
                    train_basins,
                    val_basins,
                    distance,
                )
                # Drop unlabeled windows whose centers sit in the catchment-boundary strip.
                area_gray_mask = area_gray_mask | (area_unlabeled_mask & area_boundary)
            else:
                dist_to_val = _min_distance_to_points(area_coords, mineral_coords[val_mineral_mask])
                dist_to_train = _min_distance_to_points(area_coords, mineral_coords[train_mineral_mask])
                area_gray_mask = area_gray_mask | (
                    area_unlabeled_mask
                    & (
                        (area_val_mask & (dist_to_train <= distance))
                        | (area_train_mask & (dist_to_val <= distance))
                    )
                )
            area_val_mask = area_val_mask & (~area_gray_mask)
            area_train_mask = area_train_mask & (~area_gray_mask)

        sample_positive_mask = labels == 1
        sample_val_pos = sample_positive_mask & np.isin(sample_cluster_ids, list(val_cluster_set))
        sample_train_pos = sample_positive_mask & np.isin(sample_cluster_ids, list(train_cluster_set))
        if use_whole_basin_unlabeled:
            if sample_basin_arr is not None and len(sample_basin_arr) == len(coords):
                sample_u_train, sample_u_val, sample_u_gray = _unlabeled_masks_by_whole_basin(
                    basin_ids=sample_basin_arr,
                    unlabeled_mask=sample_unlabeled_mask,
                    mineral_basin_ids=mineral_basin_arr,
                    mineral_cluster_ids=cluster_ids,
                    train_cluster_set=train_cluster_set,
                    val_cluster_set=val_cluster_set,
                    coords=coords,
                    mineral_coords=mineral_coords,
                )
            elif sample_gray_source is not None and area_u_train is not None:
                sample_u_train = sample_unlabeled_mask & area_u_train[sample_gray_source]
                sample_u_val = sample_unlabeled_mask & area_u_val[sample_gray_source]
                sample_u_gray = sample_unlabeled_mask & area_u_gray[sample_gray_source]
            else:
                sample_u_train = sample_unlabeled_mask & np.isin(sample_cluster_ids, list(train_cluster_set))
                sample_u_val = sample_unlabeled_mask & np.isin(sample_cluster_ids, list(val_cluster_set))
                sample_u_gray = sample_unlabeled_mask & (~sample_u_train) & (~sample_u_val)
            sample_val_mask = sample_val_pos | sample_u_val
            sample_train_mask = sample_train_pos | sample_u_train
            sample_gray_mask = np.asarray(sample_u_gray, dtype=bool).copy()
            if distance > 0:
                if sample_basin_arr is not None and len(sample_basin_arr) == len(coords):
                    train_basins = _basins_of_clusters(mineral_basin_arr, cluster_ids, train_cluster_set)
                    val_basins = _basins_of_clusters(mineral_basin_arr, cluster_ids, val_cluster_set)
                    sample_boundary = _catchment_boundary_buffer_mask(
                        coords,
                        sample_basin_arr,
                        train_basins,
                        val_basins,
                        distance,
                    )
                    sample_gray_mask = sample_gray_mask | (sample_unlabeled_mask & sample_boundary)
                else:
                    dist_to_val = _min_distance_to_points(coords, mineral_coords[val_mineral_mask])
                    dist_to_train = _min_distance_to_points(coords, mineral_coords[train_mineral_mask])
                    sample_gray_mask = sample_gray_mask | (
                        sample_unlabeled_mask
                        & (
                            (sample_val_mask & (dist_to_train <= distance))
                            | (sample_train_mask & (dist_to_val <= distance))
                        )
                    )
        else:
            sample_val_mask = np.isin(sample_cluster_ids, list(val_cluster_set))
            sample_train_mask = np.isin(sample_cluster_ids, list(train_cluster_set))
            if sample_gray_source is not None:
                sample_gray_mask = area_gray_mask[sample_gray_source]
            else:
                sample_gray_mask = np.zeros(len(coords), dtype=bool)
                if distance > 0:
                    dist_to_val = _min_distance_to_points(coords, mineral_coords[val_mineral_mask])
                    dist_to_train = _min_distance_to_points(coords, mineral_coords[train_mineral_mask])
                    sample_gray_mask = sample_unlabeled_mask & (
                        (sample_val_mask & (dist_to_train <= distance))
                        | (sample_train_mask & (dist_to_val <= distance))
                    )
        footprint_embargo_mask = np.zeros(len(coords), dtype=bool)
        if int(footprint_embargo_grid_cells) > 0 and np.any(area_val_mask):
            footprint_embargo_mask = sample_train_mask & _grid_chebyshev_proximity_mask(
                coords,
                area_coords[area_val_mask],
                int(footprint_embargo_grid_cells),
            )
            sample_gray_mask |= footprint_embargo_mask

        train_idx = np.where(sample_train_mask & (~sample_gray_mask))[0].astype(np.int64)
        val_idx = np.where(sample_val_mask & (~sample_gray_mask))[0].astype(np.int64)
        area_val_idx = np.where(area_val_mask & (~area_gray_mask))[0].astype(np.int64)
        if len(train_idx) == 0 or len(val_idx) == 0 or len(area_val_idx) == 0:
            continue

        train_positive_count = int(np.sum(labels[train_idx] == 1))
        val_positive_count = int(np.sum(labels[val_idx] == 1))
        train_unlabeled_count = int(np.sum(labels[train_idx] != 1))
        val_unlabeled_count = int(np.sum(labels[val_idx] != 1))
        if train_positive_count == 0 or val_positive_count == 0:
            continue

        folds.append(
            {
                "fold": int(fold_idx),
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist(),
                "val_area_indices": area_val_idx.tolist(),
                "train_count": int(len(train_idx)),
                "val_count": int(len(val_idx)),
                "val_area_count": int(len(area_val_idx)),
                "train_positive_count": train_positive_count,
                "train_unlabeled_count": train_unlabeled_count,
                "val_positive_count": val_positive_count,
                "val_unlabeled_count": val_unlabeled_count,
                "strategy": strategy,
                "spatial_cv_strategy": strategy,
                "spatial_cv_hard_isolation": True,
                "unit_fold_mode": fold_mode,
                "axis": "x" if axis == 0 else "y",
                "axis_source": axis_source,
                "requested_fold_count": int(requested_folds),
                "effective_fold_count": int(effective_folds),
                "train_mineral_count": int(np.sum(train_mineral_mask)),
                "val_mineral_count": int(np.sum(val_mineral_mask)),
                "train_cluster_ids": sorted(int(item) for item in train_cluster_set),
                "val_cluster_ids": sorted(int(item) for item in val_cluster_set),
                "gray_count": int(np.sum(sample_gray_mask)),
                "gray_area_count": int(np.sum(area_gray_mask)),
                "buffer_excluded_sample_count": int(np.sum(sample_gray_mask)),
                "buffer_excluded_area_count": int(np.sum(area_gray_mask)),
                "footprint_embargo_grid_cells": int(footprint_embargo_grid_cells),
                "footprint_embargo_removed_train_count": int(
                    np.sum(footprint_embargo_mask)
                ),
                "evaluation_area_mask_fixed": bool(
                    int(footprint_embargo_grid_cells) > 0
                ),
                "unlabeled_isolation": unlabeled_isolation,
            }
        )
    return folds


def _build_cluster_stratified_spatial_cv_folds(
    coords: np.ndarray,
    labels: np.ndarray,
    mineral_ids: np.ndarray,
    train_minerals: pd.DataFrame,
    *,
    n_folds: int = 5,
    buffer_distance: float = 0.0,
    area_coords: Optional[np.ndarray] = None,
    area_mineral_ids: Optional[np.ndarray] = None,
    sample_to_area_indices: Optional[np.ndarray] = None,
) -> list:
    coords = np.asarray(coords, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    mineral_ids = np.asarray(mineral_ids, dtype=np.int64).reshape(-1)
    if len(coords) != len(labels) or len(coords) != len(mineral_ids) or len(coords) == 0:
        return []
    if train_minerals is None or len(train_minerals) == 0:
        return []

    mineral_frame = train_minerals.reset_index(drop=True).copy()
    mineral_coords = mineral_frame[["x", "y"]].to_numpy(dtype=np.float64)
    if mineral_coords.ndim != 2 or mineral_coords.shape[1] < 2 or len(mineral_coords) == 0:
        return []
    if "kmeans_cluster" in mineral_frame.columns:
        cluster_ids = pd.to_numeric(mineral_frame["kmeans_cluster"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    else:
        cluster_ids = np.zeros(len(mineral_frame), dtype=np.int64)

    n_folds = int(max(n_folds, 2))
    mineral_fold_ids = np.full(len(mineral_frame), -1, dtype=np.int64)
    cluster_summaries = []
    for cluster_id in sorted(np.unique(cluster_ids)):
        cluster_mineral_ids = np.where(cluster_ids == cluster_id)[0].astype(np.int64)
        if len(cluster_mineral_ids) == 0:
            continue
        cluster_coords = mineral_coords[cluster_mineral_ids]
        x_span = float(np.max(cluster_coords[:, 0]) - np.min(cluster_coords[:, 0])) if len(cluster_coords) else 0.0
        y_span = float(np.max(cluster_coords[:, 1]) - np.min(cluster_coords[:, 1])) if len(cluster_coords) else 0.0
        axis = 0 if x_span >= y_span else 1
        ordered = cluster_mineral_ids[np.argsort(cluster_coords[:, axis], kind="mergesort")]
        subgroups = np.array_split(ordered, n_folds)
        fold_counts = []
        for fold_idx, subgroup in enumerate(subgroups):
            subgroup = np.asarray(subgroup, dtype=np.int64)
            if len(subgroup) == 0:
                fold_counts.append(0)
                continue
            mineral_fold_ids[subgroup] = int(fold_idx)
            fold_counts.append(int(len(subgroup)))
        cluster_summaries.append(
            {
                "cluster": int(cluster_id),
                "mineral_count": int(len(cluster_mineral_ids)),
                "axis": "x" if axis == 0 else "y",
                "fold_mineral_counts": fold_counts,
            }
        )

    valid_mineral_mask = mineral_fold_ids >= 0
    if not np.any(valid_mineral_mask):
        return []

    area_coords = np.asarray(area_coords if area_coords is not None else coords, dtype=np.float64)
    area_mineral_ids = np.asarray(
        area_mineral_ids if area_mineral_ids is not None else mineral_ids,
        dtype=np.int64,
    ).reshape(-1)
    if len(area_coords) != len(area_mineral_ids):
        return []

    area_fold_ids = np.full(len(area_coords), -1, dtype=np.int64)
    valid_area_positive = (area_mineral_ids >= 0) & (area_mineral_ids < len(mineral_fold_ids))
    area_fold_ids[valid_area_positive] = mineral_fold_ids[area_mineral_ids[valid_area_positive]]
    unlabeled_area = area_fold_ids < 0
    if np.any(unlabeled_area):
        area_fold_ids[unlabeled_area] = _nearest_mineral_fold_ids(
            area_coords[unlabeled_area],
            mineral_coords[valid_mineral_mask],
            mineral_fold_ids[valid_mineral_mask],
        )

    if sample_to_area_indices is not None:
        sample_to_area_indices = np.asarray(sample_to_area_indices, dtype=np.int64).reshape(-1)
        if len(sample_to_area_indices) != len(coords):
            return []
        sample_fold_ids = area_fold_ids[sample_to_area_indices]
        sample_gray_source = sample_to_area_indices
    else:
        sample_fold_ids = np.full(len(coords), -1, dtype=np.int64)
        valid_sample_positive = (mineral_ids >= 0) & (mineral_ids < len(mineral_fold_ids))
        sample_fold_ids[valid_sample_positive] = mineral_fold_ids[mineral_ids[valid_sample_positive]]
        unlabeled_sample = sample_fold_ids < 0
        if np.any(unlabeled_sample):
            sample_fold_ids[unlabeled_sample] = _nearest_mineral_fold_ids(
                coords[unlabeled_sample],
                mineral_coords[valid_mineral_mask],
                mineral_fold_ids[valid_mineral_mask],
            )
        sample_gray_source = None

    distance = float(max(buffer_distance, 0.0))
    folds = []
    for fold_idx in range(n_folds):
        val_mineral_mask = mineral_fold_ids == fold_idx
        train_mineral_mask = (mineral_fold_ids >= 0) & (mineral_fold_ids != fold_idx)
        if not np.any(val_mineral_mask) or not np.any(train_mineral_mask):
            continue

        area_val_mask = area_fold_ids == fold_idx
        area_train_mask = (area_fold_ids >= 0) & (area_fold_ids != fold_idx)
        area_gray_mask = np.zeros(len(area_coords), dtype=bool)
        if distance > 0:
            dist_to_val = _min_distance_to_points(area_coords, mineral_coords[val_mineral_mask])
            dist_to_train = _min_distance_to_points(area_coords, mineral_coords[train_mineral_mask])
            area_gray_mask = (area_val_mask & (dist_to_train <= distance)) | (area_train_mask & (dist_to_val <= distance))

        sample_val_mask = sample_fold_ids == fold_idx
        sample_train_mask = (sample_fold_ids >= 0) & (sample_fold_ids != fold_idx)
        if sample_gray_source is not None:
            sample_gray_mask = area_gray_mask[sample_gray_source]
        else:
            sample_gray_mask = np.zeros(len(coords), dtype=bool)
            if distance > 0:
                dist_to_val = _min_distance_to_points(coords, mineral_coords[val_mineral_mask])
                dist_to_train = _min_distance_to_points(coords, mineral_coords[train_mineral_mask])
                sample_gray_mask = (sample_val_mask & (dist_to_train <= distance)) | (sample_train_mask & (dist_to_val <= distance))

        train_idx = np.where(sample_train_mask & (~sample_gray_mask))[0].astype(np.int64)
        val_idx = np.where(sample_val_mask & (~sample_gray_mask))[0].astype(np.int64)
        area_val_idx = np.where(area_val_mask & (~area_gray_mask))[0].astype(np.int64)
        if len(train_idx) == 0 or len(val_idx) == 0 or len(area_val_idx) == 0:
            continue

        val_clusters = sorted(set(cluster_ids[val_mineral_mask].astype(int).tolist()))
        folds.append(
            {
                "fold": int(fold_idx),
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist(),
                "val_area_indices": area_val_idx.tolist(),
                "train_count": int(len(train_idx)),
                "val_count": int(len(val_idx)),
                "val_area_count": int(len(area_val_idx)),
                "strategy": "cluster_stratified_spatial",
                "axis": "cluster_internal",
                "axis_source": "outer_cluster_internal_spatial_order",
                "val_mineral_count": int(np.sum(val_mineral_mask)),
                "train_mineral_count": int(np.sum(train_mineral_mask)),
                "val_cluster_count": int(len(val_clusters)),
                "val_clusters": val_clusters,
                "buffer_excluded_sample_count": int(np.sum(sample_gray_mask)),
                "buffer_excluded_area_count": int(np.sum(area_gray_mask)),
                "cluster_summaries": cluster_summaries,
            }
        )
    return folds


def _build_spatial_stratified_cv_folds(
    coords: np.ndarray,
    labels: np.ndarray,
    mineral_ids: np.ndarray,
    train_minerals: pd.DataFrame,
    *,
    n_clusters: int = 10,
    n_folds: int = 5,
    buffer_distance: float = 0.0,
    random_state: int = 42,
    area_coords: Optional[np.ndarray] = None,
    area_mineral_ids: Optional[np.ndarray] = None,
    sample_to_area_indices: Optional[np.ndarray] = None,
) -> list:
    coords = np.asarray(coords, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    mineral_ids = np.asarray(mineral_ids, dtype=np.int64).reshape(-1)
    if len(coords) != len(labels) or len(coords) != len(mineral_ids) or len(coords) == 0:
        return []
    if train_minerals is None or len(train_minerals) == 0:
        return []

    assignment = _assign_spatial_stratified_mineral_folds(
        train_minerals,
        n_clusters=n_clusters,
        n_folds=n_folds,
        random_state=random_state,
        prefer_existing_clusters=True,
    )
    mineral_frame = assignment["frame"].reset_index(drop=True)
    mineral_coords = mineral_frame[["x", "y"]].to_numpy(dtype=np.float64)
    if mineral_coords.ndim != 2 or mineral_coords.shape[1] < 2 or len(mineral_coords) == 0:
        return []

    mineral_fold_ids = pd.to_numeric(mineral_frame["spatial_fold"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    cluster_ids = pd.to_numeric(mineral_frame["kmeans_cluster"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    valid_mineral_mask = mineral_fold_ids >= 0
    valid_fold_ids = sorted(int(item) for item in np.unique(mineral_fold_ids[valid_mineral_mask]) if int(item) >= 0)
    if len(valid_fold_ids) < 2:
        return []

    area_coords = np.asarray(area_coords if area_coords is not None else coords, dtype=np.float64)
    area_mineral_ids = np.asarray(
        area_mineral_ids if area_mineral_ids is not None else mineral_ids,
        dtype=np.int64,
    ).reshape(-1)
    if len(area_coords) != len(area_mineral_ids):
        return []

    area_fold_ids = np.full(len(area_coords), -1, dtype=np.int64)
    valid_area_positive = (area_mineral_ids >= 0) & (area_mineral_ids < len(mineral_fold_ids))
    area_fold_ids[valid_area_positive] = mineral_fold_ids[area_mineral_ids[valid_area_positive]]
    unlabeled_area = area_fold_ids < 0
    if np.any(unlabeled_area):
        area_fold_ids[unlabeled_area] = _nearest_mineral_fold_ids(
            area_coords[unlabeled_area],
            mineral_coords[valid_mineral_mask],
            mineral_fold_ids[valid_mineral_mask],
        )

    if sample_to_area_indices is not None:
        sample_to_area_indices = np.asarray(sample_to_area_indices, dtype=np.int64).reshape(-1)
        if len(sample_to_area_indices) != len(coords):
            return []
        sample_fold_ids = area_fold_ids[sample_to_area_indices]
        sample_gray_source = sample_to_area_indices
    else:
        sample_fold_ids = np.full(len(coords), -1, dtype=np.int64)
        valid_sample_positive = (mineral_ids >= 0) & (mineral_ids < len(mineral_fold_ids))
        sample_fold_ids[valid_sample_positive] = mineral_fold_ids[mineral_ids[valid_sample_positive]]
        unlabeled_sample = sample_fold_ids < 0
        if np.any(unlabeled_sample):
            sample_fold_ids[unlabeled_sample] = _nearest_mineral_fold_ids(
                coords[unlabeled_sample],
                mineral_coords[valid_mineral_mask],
                mineral_fold_ids[valid_mineral_mask],
            )
        sample_gray_source = None

    distance = float(max(buffer_distance, 0.0))
    folds = []
    for fold_idx in valid_fold_ids:
        val_mineral_mask = mineral_fold_ids == fold_idx
        train_mineral_mask = (mineral_fold_ids >= 0) & (mineral_fold_ids != fold_idx)
        if not np.any(val_mineral_mask) or not np.any(train_mineral_mask):
            continue

        area_val_mask = area_fold_ids == fold_idx
        area_train_mask = (area_fold_ids >= 0) & (area_fold_ids != fold_idx)
        area_gray_mask = np.zeros(len(area_coords), dtype=bool)
        if distance > 0:
            dist_to_val = _min_distance_to_points(area_coords, mineral_coords[val_mineral_mask])
            dist_to_train = _min_distance_to_points(area_coords, mineral_coords[train_mineral_mask])
            area_gray_mask = (area_val_mask & (dist_to_train <= distance)) | (area_train_mask & (dist_to_val <= distance))

        sample_val_mask = sample_fold_ids == fold_idx
        sample_train_mask = (sample_fold_ids >= 0) & (sample_fold_ids != fold_idx)
        if sample_gray_source is not None:
            sample_gray_mask = area_gray_mask[sample_gray_source]
        else:
            sample_gray_mask = np.zeros(len(coords), dtype=bool)
            if distance > 0:
                dist_to_val = _min_distance_to_points(coords, mineral_coords[val_mineral_mask])
                dist_to_train = _min_distance_to_points(coords, mineral_coords[train_mineral_mask])
                sample_gray_mask = (sample_val_mask & (dist_to_train <= distance)) | (sample_train_mask & (dist_to_val <= distance))

        train_idx = np.where(sample_train_mask & (~sample_gray_mask))[0].astype(np.int64)
        val_idx = np.where(sample_val_mask & (~sample_gray_mask))[0].astype(np.int64)
        area_val_idx = np.where(area_val_mask & (~area_gray_mask))[0].astype(np.int64)
        if len(train_idx) == 0 or len(val_idx) == 0 or len(area_val_idx) == 0:
            continue

        train_positive_count = int(np.sum(labels[train_idx] == 1))
        val_positive_count = int(np.sum(labels[val_idx] == 1))
        if train_positive_count == 0 or val_positive_count == 0:
            continue

        val_clusters = sorted(set(cluster_ids[val_mineral_mask].astype(int).tolist()))
        train_clusters = sorted(set(cluster_ids[train_mineral_mask].astype(int).tolist()))
        folds.append(
            {
                "fold": int(fold_idx),
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist(),
                "val_area_indices": area_val_idx.tolist(),
                "train_count": int(len(train_idx)),
                "val_count": int(len(val_idx)),
                "val_area_count": int(len(area_val_idx)),
                "train_positive_count": train_positive_count,
                "train_unlabeled_count": int(np.sum(labels[train_idx] != 1)),
                "val_positive_count": val_positive_count,
                "val_unlabeled_count": int(np.sum(labels[val_idx] != 1)),
                "strategy": "spatial_stratified",
                "spatial_cv_strategy": "spatial_stratified",
                "spatial_cv_hard_isolation": False,
                "axis": "cluster_internal",
                "axis_source": "kmeans_cluster_internal_spatial_subgroup",
                "requested_fold_count": int(n_folds),
                "effective_fold_count": int(len(valid_fold_ids)),
                "train_mineral_count": int(np.sum(train_mineral_mask)),
                "val_mineral_count": int(np.sum(val_mineral_mask)),
                "train_cluster_ids": train_clusters,
                "val_cluster_ids": val_clusters,
                "val_clusters": val_clusters,
                "gray_count": int(np.sum(sample_gray_mask)),
                "gray_area_count": int(np.sum(area_gray_mask)),
                "buffer_excluded_sample_count": int(np.sum(sample_gray_mask)),
                "buffer_excluded_area_count": int(np.sum(area_gray_mask)),
                "cluster_summaries": list(assignment.get("cluster_summaries", [])),
            }
        )
    return folds


def export_spatial_mineral_split(
    label_path: str,
    output_dir: str,
    *,
    n_clusters: int = 10,
    cv_folds: int = 5,
    train_ratio: float = 0.7,
    random_state: int = 42,
    split_mode: str = "spatial_cluster",
    all_train: bool = False,
) -> dict:
    minerals = _read_mineral_points(label_path)
    normalized_split_mode = str(split_mode).strip().lower()
    if all_train and normalized_split_mode == "spatial_stratified":
        split = _all_train_spatial_stratified_split(
            minerals,
            n_clusters=n_clusters,
            n_folds=cv_folds,
            random_state=random_state,
        )
    elif all_train:
        train_df = minerals.reset_index(drop=True).copy()
        split = {
            "train": train_df,
            "test": train_df.iloc[0:0].copy(),
            "n_clusters": int(n_clusters),
            "train_ratio": 1.0,
            "cluster_summaries": [],
            "algorithm": normalized_split_mode,
        }
    elif normalized_split_mode == "spatial_hard":
        split = _split_minerals_by_hard_clusters(
            minerals,
            n_clusters=n_clusters,
            train_ratio=train_ratio,
            random_state=random_state,
        )
    elif normalized_split_mode == "spatial_stratified":
        split = _split_minerals_by_spatial_stratified_folds(
            minerals,
            n_clusters=n_clusters,
            n_folds=cv_folds,
            train_ratio=train_ratio,
            random_state=random_state,
        )
    elif normalized_split_mode == "spatial_cluster_holdout_cv":
        split = _split_minerals_by_cluster_holdout_cv(
            minerals,
            n_clusters=n_clusters,
            train_ratio=train_ratio,
            random_state=random_state,
        )
    elif normalized_split_mode == "leave_one_camp":
        if split_leave_one_camp is None:
            raise RuntimeError("leave_one_camp splitter is unavailable.")
        split = split_leave_one_camp(
            minerals,
            n_clusters=n_clusters,
            held_out_camp_index=0,
            random_state=random_state,
        )
    elif normalized_split_mode == "leave_one_fault":
        if split_leave_one_fault is None:
            raise RuntimeError("leave_one_fault splitter is unavailable.")
        assignment = minerals if "fault_id" in minerals.columns else None
        split = split_leave_one_fault(minerals, held_out_fault_id="", assignment=assignment)
    elif normalized_split_mode == "variogram_block_cv":
        if split_variogram_block_cv is None:
            raise RuntimeError("variogram_block_cv splitter is unavailable.")
        split = split_variogram_block_cv(
            minerals,
            variogram_range_m=2000.0,
            held_out_block_index=0,
            random_state=random_state,
        )
    else:
        split = _split_minerals_by_kmeans(
            minerals,
            n_clusters=n_clusters,
            train_ratio=train_ratio,
            random_state=random_state,
        )
    train_df = split["train"]
    test_df = split["test"]
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["split_set"] = "train"
    test_df["split_set"] = "test"
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    train_csv = os.path.join(output_dir, "spatial_train_minerals.csv")
    test_csv = os.path.join(output_dir, "spatial_test_minerals.csv")
    combined_csv = os.path.join(output_dir, "spatial_all_minerals_with_clusters.csv")
    train_txt = os.path.join(output_dir, "spatial_train_minerals.txt")
    test_txt = os.path.join(output_dir, "spatial_test_minerals.txt")
    summary_json = os.path.join(output_dir, "spatial_split_summary.json")

    train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_csv, index=False, encoding="utf-8-sig")
    combined_df.to_csv(combined_csv, index=False, encoding="utf-8-sig")
    train_df[["x", "y"]].to_csv(train_txt, sep="\t", index=False, header=True, encoding="utf-8-sig")
    test_df[["x", "y"]].to_csv(test_txt, sep="\t", index=False, header=True, encoding="utf-8-sig")

    summary = {
        "total_minerals": int(len(minerals)),
        "train_minerals": int(len(train_df)),
        "test_minerals": int(len(test_df)),
        "n_clusters": int(split.get("n_clusters", n_clusters)),
        "train_ratio": float(split.get("train_ratio", train_ratio)),
        "cv_folds": int(split.get("fold_count", cv_folds)),
        "random_state": int(random_state),
        "split_mode": str(split_mode),
        "split_algorithm": str(split.get("algorithm", "spatial_cluster")),
        "train_fold_ids": split.get("train_fold_ids", []),
        "test_fold_ids": split.get("test_fold_ids", []),
        "val_fold_ids": split.get("val_fold_ids", split.get("test_fold_ids", [])),
        "train_csv": train_csv,
        "test_csv": test_csv,
        "train_txt": train_txt,
        "test_txt": test_txt,
        "combined_csv": combined_csv,
    }
    import json

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary["summary_json"] = summary_json
    return summary


def _spatial_cluster_split_indices(
    sample_coords: np.ndarray,
    labels: np.ndarray,
    *,
    n_clusters: int = 10,
    train_ratio: float = 0.7,
    buffer_distance: float = 0.0,
    random_state: int = 42,
):
    coords = np.asarray(sample_coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Spatial split requires 2D coordinates.")
    coords = coords[:, :2]
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if len(coords) != len(labels):
        raise ValueError("Coordinate count and label count must match for spatial split.")
    if len(coords) == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, {
            "n_clusters": int(n_clusters),
            "train_ratio": float(train_ratio),
            "buffer_distance": float(buffer_distance),
            "cluster_ids": [],
            "cluster_summaries": [],
            "gray_count": 0,
        }

    unique_count = len(np.unique(coords, axis=0))
    cluster_count = max(1, min(int(n_clusters), len(coords), unique_count))
    train_ratio = float(train_ratio)
    if not np.isfinite(train_ratio):
        train_ratio = 0.7
    train_ratio = float(min(max(train_ratio, 0.1), 0.9))
    buffer_distance = float(max(buffer_distance, 0.0))
    rng = np.random.default_rng(int(random_state))

    if cluster_count == 1 or len(coords) < 3:
        order = rng.permutation(len(coords))
        train_count = max(1, int(round(len(coords) * train_ratio)))
        train_count = min(train_count, len(coords) - 1) if len(coords) > 1 else len(coords)
        train_idx = np.sort(order[:train_count]).astype(np.int64)
        test_idx = np.sort(order[train_count:]).astype(np.int64)
        return train_idx, test_idx, {
            "n_clusters": int(cluster_count),
            "train_ratio": float(train_ratio),
            "buffer_distance": float(buffer_distance),
            "cluster_ids": [0] * len(coords),
            "cluster_summaries": [{
                "cluster_id": 0,
                "sample_count": int(len(coords)),
                "train_count": int(len(train_idx)),
                "test_count": int(len(test_idx)),
            }],
            "gray_count": 0,
        }

    kmeans = KMeans(n_clusters=cluster_count, random_state=int(random_state), n_init=10)
    cluster_ids = kmeans.fit_predict(coords)
    distances = kmeans.transform(coords)
    sorted_distances = np.sort(distances, axis=1)
    margin = sorted_distances[:, 1] - sorted_distances[:, 0] if distances.shape[1] > 1 else np.full(len(coords), np.inf)
    gray_mask = margin <= buffer_distance

    train_rows = []
    test_rows = []
    cluster_summaries = []
    for cluster_id in sorted(np.unique(cluster_ids)):
        cluster_rows = np.where((cluster_ids == cluster_id) & (~gray_mask))[0]
        cluster_size = int(len(cluster_rows))
        if cluster_size == 0:
            continue
        order = rng.permutation(cluster_rows)
        if cluster_size == 1:
            train_count = 1
        else:
            train_count = int(round(cluster_size * train_ratio))
            train_count = max(1, min(train_count, cluster_size - 1))
        cluster_train_rows = np.sort(order[:train_count]).astype(np.int64)
        cluster_test_rows = np.sort(order[train_count:]).astype(np.int64)
        train_rows.extend(cluster_train_rows.tolist())
        test_rows.extend(cluster_test_rows.tolist())
        cluster_summaries.append(
            {
                "cluster_id": int(cluster_id),
                "sample_count": cluster_size,
                "train_count": int(len(cluster_train_rows)),
                "test_count": int(len(cluster_test_rows)),
            }
        )

    train_idx = np.asarray(sorted(set(train_rows)), dtype=np.int64)
    test_idx = np.asarray(sorted(set(test_rows)), dtype=np.int64)
    return train_idx, test_idx, {
        "n_clusters": int(cluster_count),
        "train_ratio": float(train_ratio),
        "buffer_distance": float(buffer_distance),
        "cluster_ids": np.asarray(cluster_ids, dtype=np.int64).tolist(),
        "cluster_summaries": cluster_summaries,
        "gray_count": int(np.sum(gray_mask)),
    }


def make_dataset(dataset):
    def make_pu_dataset_from_binary_dataset(x, y):
        print("寮€濮嬪垱寤?PU 鏁版嵁闆?..")
        labels = np.unique(y)
        print(f"鏁版嵁闆嗕腑鐨勫敮涓€鏍囩鍊? {labels}")

        positive, negative = labels[1], labels[0]
        print(f"姝ｇ被鏍囩: {positive}, 璐熺被鏍囩: {negative}")
        x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        n_p = (y == positive).sum()
        n_n = (y == negative).sum()
        print(f"姝ｆ牱鏈暟閲? {n_p}, 璐熸牱鏈暟閲? {n_n}")

        perm = np.random.permutation(len(y))
        x, y = x[perm], y[perm]

        pos_indices = np.where(y == positive)[0]
        neg_indices = np.where(y == negative)[0]

        xlp = x[pos_indices]
        n_lp = len(pos_indices)

        xu = x[neg_indices]
        n_u = len(neg_indices)

        prior = float(n_p) / float(len(y))

        x = np.concatenate((xlp, xu), axis=0)
        y = np.concatenate((np.ones(n_lp), -np.ones(n_u)))

        perm = np.random.permutation(len(y))
        x, y = x[perm], y[perm]

        print(f"鏈€缁堟暟鎹泦澶у皬: {len(x)}")
        print(f"姝ｆ爣璁版牱鏈暟: {n_lp}")
        print(f"鏈爣璁版牱鏈暟: {n_u}")
        print(f"鍏堥獙姒傜巼: {prior}")

        return x, y, prior

    def make_pn_dataset_from_binary_dataset(x, y):
        print("\n鍒涘缓娴嬭瘯鏁版嵁闆?..")
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]

        X, Y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        Y = (Y == positive).astype(np.int32) * 2 - 1

        n_p_test = (Y == 1).sum()
        n_n_test = (Y == -1).sum()
        print(f"娴嬭瘯闆嗘鏍锋湰鏁伴噺: {n_p_test}")
        print(f"娴嬭瘯闆嗚礋鏍锋湰鏁伴噺: {n_n_test}")
        print(f"娴嬭瘯闆嗘€诲ぇ灏? {len(X)}")
        print(f"娴嬭瘯闆嗘鏍锋湰姣斾緥: {n_p_test / len(X):.2%}")

        return X, Y

    (x_train, y_train), (x_test, y_test) = dataset
    x_train, y_train, prior = make_pu_dataset_from_binary_dataset(x_train, y_train)
    x_test, y_test = make_pn_dataset_from_binary_dataset(x_test, y_test)

    return (x_train, y_train), (x_test, y_test), prior


def get_h5_data(
    data_path,
    label_path,
    test_size=0.2,
    random_state=42,
    sample_ratio=1.0,
    split_mode="legacy",
    patch_size=None,
    patch_stride=None,
    buffer_radius=0.0,
    spatial_cluster_n_clusters=10,
    spatial_cluster_train_ratio=0.7,
    full_mineral_training=False,
    mineral_training_strategy="holdout",
    spatial_cluster_cv_buffer_distance=0.0,
    spatial_cv_folds=1,
    no_ore_path=None,
    use_reflect_padding=False,
    selected_channels=None,
    positive_window_mode="three_windows_equal_weight",
    deposit_loss_weighting=True,
    leave_one_camp_index=0,
    leave_one_fault_id="",
    deposit_fault_assignment="",
    fault_lines_path="",
    variogram_range_m=2000.0,
    metric_unit="deposit_unit",
    ogr_options=None,
):
    print("Loading and preparing dataset...")
    ogr_options = dict(ogr_options or {})
    ogr_options.setdefault("leave_one_camp_index", leave_one_camp_index)
    ogr_options.setdefault("leave_one_fault_id", leave_one_fault_id)
    ogr_options.setdefault("deposit_fault_assignment", deposit_fault_assignment)
    ogr_options.setdefault("deposit_camp_assignment", ogr_options.get("deposit_camp_assignment", ""))
    ogr_options.setdefault("fault_lines_path", fault_lines_path)
    ogr_options.setdefault("basin_grd_path", ogr_options.get("basin_grd_path", ""))
    ogr_options.setdefault("variogram_range_m", variogram_range_m)
    ogr_options.setdefault("fail_on_single_variogram_block", False)
    ogr_options.setdefault("positive_window_mode", positive_window_mode)
    ogr_options.setdefault("deposit_loss_weighting", deposit_loss_weighting)
    ogr_options.setdefault("multi_window_max_per_unit", DEFAULT_MULTI_WINDOW_MAX_PER_UNIT)
    ogr_options.setdefault("multi_window_sample_ref_window", DEFAULT_MULTI_WINDOW_SAMPLE_REF_WINDOW)
    ogr_options.setdefault("multi_window_sample_halfwidth", None)
    # Leave sigma unset here so ±2 / ±5 distance modes can apply mode-aware defaults later.
    ogr_options.setdefault("multi_window_distance_sigma", None)
    ogr_options.setdefault("metric_unit", metric_unit)
    ogr_options.setdefault("reviewer_protocol", False)
    ogr_options.setdefault("expert_camp_confirmed", False)
    ogr_options.setdefault("expert_fault_confirmed", False)
    ogr_options.setdefault("meters_per_coordinate_unit", None)
    ogr_options.setdefault("merge_distance_coordinate_units", 0.0)
    ogr_options.setdefault("merge_distance_m", 0.0)
    ogr_options.setdefault("inner_camp_cv_mode", "leave_one_camp")
    ogr_options.setdefault("reviewer_primary_protocol", False)
    ogr_options.setdefault("footprint_reference_patch_size", 0)
    ogr_options.setdefault("fixed_evaluation_mask", False)
    ogr_options.setdefault("unlabeled_contamination_scope", "train_minerals")

    if not 0 < sample_ratio <= 1:
        raise ValueError("sample_ratio must be in (0, 1].")
    if patch_size is None:
        patch_size = 16
    if patch_stride is None:
        patch_stride = patch_size

    x, coordinates, metadata, feature_mode = _load_feature_tensor(
        data_path,
        patch_size,
        patch_stride,
        use_reflect_padding=use_reflect_padding,
        selected_channels=selected_channels,
    )
    label_ext = os.path.splitext(label_path)[1].lower()
    spatial_split_mode = str(split_mode or "legacy").strip().lower()
    use_spatial_mineral_split = spatial_split_mode in SPATIAL_MINERAL_SPLIT_MODES and label_ext in {".txt", ".csv", ".tsv"}
    unlabeled_subsample_ratio = None

    split_summary = {
        "split_mode": str(split_mode),
        "feature_mode": str(feature_mode),
        "label_mode": "coordinates" if label_ext in {".txt", ".csv", ".tsv"} else "h5",
        "patch_size": int(patch_size),
        "patch_stride": int(patch_stride),
        "sample_ratio": float(sample_ratio),
        "reflect_padding": bool(use_reflect_padding),
        "selected_channel_indices": metadata.get("selected_channel_indices", list(range(int(x.shape[1])))),
        "selected_channel_names": metadata.get("selected_channel_names", metadata.get("available_channel_names", [])),
        "channel_selection_active": bool(metadata.get("channel_selection_active", False)),
    }

    mineral_training_strategy = str(mineral_training_strategy or "holdout").strip().lower()
    if full_mineral_training:
        mineral_training_strategy = "all_minerals"
    if mineral_training_strategy not in {"holdout", "train_val", "all_minerals", "holdout_cv"}:
        mineral_training_strategy = "holdout"
    full_mineral_training = mineral_training_strategy == "all_minerals"

    if use_spatial_mineral_split:
        all_minerals = _read_mineral_points(label_path)
        merge_threshold = float(ogr_options.get("merge_distance_coordinate_units", 0.0) or 0.0)
        protected_columns = []
        if spatial_split_mode == "leave_one_camp":
            assignment_path = str(ogr_options.get("deposit_camp_assignment", "") or "").strip()
            if assignment_path and load_deposit_camp_assignment is not None and map_assignment_to_minerals is not None:
                assignment_for_merge = load_deposit_camp_assignment(assignment_path)
                all_minerals = map_assignment_to_minerals(
                    all_minerals, assignment_for_merge, id_column="camp_id"
                )
                protected_columns.append("camp_id")
        elif spatial_split_mode == "leave_one_fault":
            assignment_path = str(ogr_options.get("deposit_fault_assignment", "") or "").strip()
            if assignment_path and load_deposit_fault_assignment is not None and map_assignment_to_minerals is not None:
                assignment_for_merge = load_deposit_fault_assignment(assignment_path)
                all_minerals = map_assignment_to_minerals(
                    all_minerals, assignment_for_merge, id_column="fault_id"
                )
                protected_columns.append("fault_id")
        all_minerals, mineral_unit_mapping, merge_audit = merge_mineralization_units(
            all_minerals,
            merge_threshold,
            protected_group_columns=protected_columns,
        )
        merge_audit["threshold_m"] = float(ogr_options.get("merge_distance_m", 0.0) or 0.0)
        merge_audit["meters_per_coordinate_unit"] = ogr_options.get("meters_per_coordinate_unit")
        print(
            "矿化单元合并: "
            f"{merge_audit['source_deposit_count']} 矿点 → {merge_audit['merged_unit_count']} 单元, "
            f"阈值={merge_audit['threshold_m']} m ({merge_threshold:g} 坐标单位)"
        )
        split_coords = _patch_indices_to_geo(coordinates, metadata)
        if split_coords is None or len(split_coords) == 0:
            raise ValueError("Spatial split requires valid patch coordinates.")

        if full_mineral_training:
            if spatial_split_mode == "spatial_stratified":
                mineral_split = _all_train_spatial_stratified_split(
                    all_minerals,
                    n_clusters=spatial_cluster_n_clusters,
                    n_folds=spatial_cv_folds,
                    random_state=random_state,
                )
                train_minerals = mineral_split["train"]
                test_minerals = mineral_split["test"]
            else:
                train_minerals = all_minerals.reset_index(drop=True).copy()
                test_minerals = all_minerals.iloc[0:0].copy()
                mineral_split = {
                    "train": train_minerals,
                    "test": test_minerals,
                    "n_clusters": int(spatial_cluster_n_clusters),
                    "train_ratio": 1.0,
                    "cluster_summaries": [],
                }
        else:
            if spatial_split_mode == "spatial_hard":
                mineral_split = _split_minerals_by_hard_clusters(
                    all_minerals,
                    n_clusters=spatial_cluster_n_clusters,
                    train_ratio=spatial_cluster_train_ratio,
                    random_state=random_state,
                )
            elif spatial_split_mode == "spatial_stratified":
                mineral_split = _split_minerals_by_spatial_stratified_folds(
                    all_minerals,
                    n_clusters=spatial_cluster_n_clusters,
                    n_folds=spatial_cv_folds,
                    train_ratio=spatial_cluster_train_ratio,
                    random_state=random_state,
                )
            elif spatial_split_mode == "spatial_cluster_holdout_cv":
                mineral_split = _split_minerals_by_cluster_holdout_cv(
                    all_minerals,
                    n_clusters=spatial_cluster_n_clusters,
                    train_ratio=spatial_cluster_train_ratio,
                    random_state=random_state,
                )
            elif spatial_split_mode == "leave_one_camp":
                if split_leave_one_camp is None:
                    raise RuntimeError("leave_one_camp splitter is unavailable.")
                camp_assignment = None
                camp_assignment_path = str(ogr_options.get("deposit_camp_assignment", "") or "").strip()
                if bool(ogr_options.get("reviewer_protocol", False)) and not camp_assignment_path:
                    raise ValueError("Reviewer leave-one-camp requires an expert camp assignment CSV.")
                if bool(ogr_options.get("reviewer_protocol", False)) and not bool(ogr_options.get("expert_camp_confirmed", False)):
                    raise ValueError("Reviewer leave-one-camp requires explicit expert confirmation.")
                if camp_assignment_path:
                    if load_deposit_camp_assignment is None:
                        raise RuntimeError("deposit_camp_assignment loader is unavailable.")
                    if not os.path.exists(camp_assignment_path):
                        raise FileNotFoundError(f"deposit_camp_assignment not found: {camp_assignment_path}")
                    camp_assignment = load_deposit_camp_assignment(camp_assignment_path)
                mineral_split = split_leave_one_camp(
                    all_minerals,
                    n_clusters=spatial_cluster_n_clusters,
                    held_out_camp_index=int(ogr_options.get("leave_one_camp_index", 0) or 0),
                    random_state=random_state,
                    camp_assignment=camp_assignment,
                )
                if mineral_split.get("camp_source") == "kmeans_proxy":
                    if bool(ogr_options.get("reviewer_protocol", False)):
                        raise ValueError("KMeans proxy camps are forbidden in the reviewer protocol.")
                    print(
                        "Warning: leave_one_camp 未提供矿田归属表，当前使用 KMeans 代理矿田"
                        "（非地质专家矿田划分）。"
                    )
                mineral_split = _apply_basin_split_if_available(mineral_split, ogr_options)
            elif spatial_split_mode == "leave_one_fault":
                if split_leave_one_fault is None:
                    raise RuntimeError("leave_one_fault splitter is unavailable.")
                assignment = None
                assignment_source = ""
                fault_lines_df = None
                fault_assignment_path = str(ogr_options.get("deposit_fault_assignment", "") or "")
                fault_lines_path_local = str(ogr_options.get("fault_lines_path", "") or "")
                held_fault = str(ogr_options.get("leave_one_fault_id", "") or "")
                if bool(ogr_options.get("reviewer_protocol", False)):
                    if not fault_assignment_path:
                        raise ValueError("Reviewer leave-one-fault requires an expert fault assignment CSV.")
                    if not bool(ogr_options.get("expert_fault_confirmed", False)):
                        raise ValueError("Reviewer leave-one-fault requires explicit expert confirmation.")
                if fault_assignment_path and load_deposit_fault_assignment is not None:
                    assignment = load_deposit_fault_assignment(fault_assignment_path)
                    assignment_source = "expert_table"
                elif fault_lines_path_local and auto_assign_faults_from_lines is not None:
                    if bool(ogr_options.get("reviewer_protocol", False)):
                        raise ValueError("Nearest-fault auto-assignment is forbidden in the reviewer protocol.")
                    fault_lines_df = (
                        load_fault_lines(fault_lines_path_local)
                        if load_fault_lines is not None
                        else pd.read_csv(fault_lines_path_local)
                    )
                    assignment = auto_assign_faults_from_lines(all_minerals, fault_lines_df)
                    assignment_source = "auto_nearest"
                    print("Warning: leave_one_fault 使用最近断裂自动归属（非专家断裂段表）。")
                elif "fault_id" in all_minerals.columns:
                    assignment = all_minerals
                    assignment_source = "column"
                else:
                    raise ValueError(
                        "leave_one_fault requires deposit_fault_assignment, fault_lines_path, or fault_id in label file."
                    )
                if fault_lines_df is None and fault_lines_path_local:
                    try:
                        fault_lines_df = (
                            load_fault_lines(fault_lines_path_local)
                            if load_fault_lines is not None
                            else pd.read_csv(fault_lines_path_local)
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"Warning: failed to load fault lines for unlabeled hard split: {exc}")
                mineral_split = split_leave_one_fault(
                    all_minerals,
                    held_out_fault_id=held_fault,
                    assignment=assignment,
                    assignment_source=assignment_source,
                    fault_lines=fault_lines_df,
                )
                mineral_split = _apply_basin_split_if_available(mineral_split, ogr_options)
            elif spatial_split_mode == "variogram_block_cv":
                if split_variogram_block_cv is None:
                    raise RuntimeError("variogram_block_cv splitter is unavailable.")
                mineral_split = split_variogram_block_cv(
                    all_minerals,
                    variogram_range_m=float(ogr_options.get("variogram_range_m", 2000.0) or 2000.0),
                    held_out_block_index=int(ogr_options.get("leave_one_camp_index", 0) or 0),
                    random_state=random_state,
                )
                if bool(mineral_split.get("single_block_warning")):
                    message = (
                        "variogram_block_cv 仅形成 1 个矿点块：全体矿点将进入外部测试，"
                        "开发集无训练正样本。请增大研究区覆盖、减小 variogram_range_m，"
                        "或确认主导变程设置。"
                    )
                    if bool(ogr_options.get("fail_on_single_variogram_block", False)):
                        raise ValueError(message)
                    print(f"Warning: {message}")
            else:
                mineral_split = _split_minerals_by_kmeans(
                    all_minerals,
                    n_clusters=spatial_cluster_n_clusters,
                    train_ratio=spatial_cluster_train_ratio,
                    random_state=random_state,
                )
            train_minerals = mineral_split["train"]
            test_minerals = mineral_split["test"]
        if spatial_split_mode == "spatial_stratified":
            all_partition = pd.concat([train_minerals, test_minerals], ignore_index=True)
            mineral_split["mineral_coords"] = all_partition[["x", "y"]].to_numpy(dtype=np.float64)
            mineral_split["mineral_fold_ids"] = pd.to_numeric(
                all_partition.get("spatial_fold", pd.Series(np.full(len(all_partition), -1))),
                errors="coerce",
            ).fillna(-1).to_numpy(dtype=np.int64)
        elif spatial_split_mode == "spatial_cluster_holdout_cv":
            all_partition = pd.concat([train_minerals, test_minerals], ignore_index=True)
            mineral_split["mineral_coords"] = all_partition[["x", "y"]].to_numpy(dtype=np.float64)
            mineral_split["mineral_fold_ids"] = np.concatenate(
                (
                    np.zeros(len(train_minerals), dtype=np.int64),
                    np.ones(len(test_minerals), dtype=np.int64),
                )
            )
            mineral_split["train_fold_ids"] = [0]
            mineral_split["test_fold_ids"] = [1]
            mineral_split["val_fold_ids"] = [1]
        fixed_evaluation_mask = bool(ogr_options.get("fixed_evaluation_mask", False))
        fixed_outer_train_area_indices = np.array([], dtype=np.int64)
        fixed_outer_test_area_indices = np.array([], dtype=np.int64)
        fixed_outer_gray_count = 0
        if fixed_evaluation_mask and not full_mineral_training:
            if spatial_split_mode == "leave_one_camp":
                basin_path = str(ogr_options.get("basin_grd_path", "") or "").strip()
                if basin_path and os.path.exists(basin_path) and read_basin_grd is not None:
                    basin_grid_fixed = read_basin_grd(basin_path)
                    mineral_split = _enrich_mineral_split_basins(
                        mineral_split,
                        basin_grid_fixed,
                    )
                    train_minerals = mineral_split["train"]
                    test_minerals = mineral_split["test"]
                    (
                        fixed_outer_train_area_indices,
                        fixed_outer_test_area_indices,
                        fixed_outer_gray_count,
                    ) = _split_unlabeled_indices_by_basin(
                        coords=np.asarray(split_coords, dtype=np.float64),
                        mineral_split=mineral_split,
                        basin_grid=basin_grid_fixed,
                        buffer_distance=0.0,
                    )
                elif bool(ogr_options.get("reviewer_primary_protocol", False)):
                    raise ValueError(
                        "Reviewer-primary fixed evaluation mask requires a valid basin GRD."
                    )
                else:
                    (
                        fixed_outer_train_area_indices,
                        fixed_outer_test_area_indices,
                        fixed_outer_gray_count,
                    ) = _split_unlabeled_indices_by_hard_clusters(
                        coords=np.asarray(split_coords, dtype=np.float64),
                        mineral_split=mineral_split,
                        buffer_distance=0.0,
                    )
            if len(fixed_outer_train_area_indices) == 0 or len(fixed_outer_test_area_indices) == 0:
                raise ValueError(
                    "Fixed evaluation mask produced an empty outer train/test area."
                )
            print(
                "固定评价格网: "
                f"outer-train={len(fixed_outer_train_area_indices)}, "
                f"outer-test={len(fixed_outer_test_area_indices)}, "
                f"gray={fixed_outer_gray_count}; "
                "不使用矿床窗口/正样本缓冲/no-ore/抽样过滤"
            )
        train_mineral_coords = train_minerals[["x", "y"]].to_numpy(dtype=np.float64)
        positive_window_mode = str(ogr_options.get("positive_window_mode", "multi_window") or "multi_window").strip().lower()
        if positive_window_mode in {"multi_window_deposit_weighted"}:
            positive_window_mode = "multi_window_weighted"
            ogr_options["positive_window_mode"] = "multi_window_weighted"
        use_hw5_neighborhood = positive_window_mode in HW5_MULTI_WINDOW_MODES
        use_distance_weighted_multi = positive_window_mode in DISTANCE_WEIGHTED_MULTI_WINDOW_MODES
        use_capped_multi_windows = (
            positive_window_mode in CAPPED_MULTI_WINDOW_MODES
            or use_distance_weighted_multi
        )
        use_three_windows_equal_weight = positive_window_mode == THREE_WINDOWS_EQUAL_WEIGHT_MODE
        if use_capped_multi_windows:
            # Capped neighborhood sampling; equal 1/n or Gaussian distance-decay loss.
            # Keep explicit mode names (±2 vs ±5 vs 3-window) so audit / run tags stay distinguishable.
            ogr_options["deposit_loss_weighting"] = True
            if use_three_windows_equal_weight:
                ogr_options["multi_window_max_per_unit"] = int(DEFAULT_THREE_WINDOWS_PER_UNIT)
                if ogr_options.get("multi_window_sample_halfwidth") is None:
                    ogr_options["multi_window_sample_halfwidth"] = int(
                        DEFAULT_THREE_WINDOWS_SAMPLE_HALFWIDTH
                    )
                if ogr_options.get("multi_window_sample_ref_window") in (
                    None,
                    "",
                    DEFAULT_MULTI_WINDOW_SAMPLE_REF_WINDOW,
                ):
                    ogr_options["multi_window_sample_ref_window"] = int(
                        DEFAULT_THREE_WINDOWS_SAMPLE_REF_WINDOW
                    )
            if use_hw5_neighborhood and ogr_options.get("multi_window_sample_halfwidth") is None:
                ogr_options["multi_window_sample_halfwidth"] = int(
                    DEFAULT_MULTI_WINDOW_SAMPLE_HALFWIDTH_HW5
                )
            if use_hw5_neighborhood and ogr_options.get("multi_window_sample_ref_window") in (
                None,
                "",
                DEFAULT_MULTI_WINDOW_SAMPLE_REF_WINDOW,
            ):
                # Prefer the matching odd ref window for audit when UI still passes legacy 5.
                ogr_options["multi_window_sample_ref_window"] = int(
                    DEFAULT_MULTI_WINDOW_SAMPLE_REF_WINDOW_HW5
                )
            if (
                use_hw5_neighborhood
                and use_distance_weighted_multi
                and ogr_options.get("multi_window_distance_sigma") is None
            ):
                ogr_options["multi_window_distance_sigma"] = float(
                    DEFAULT_MULTI_WINDOW_DISTANCE_SIGMA_HW5
                )
        use_mineral_center_positives = (
            positive_window_mode == "one_window_per_unit" or use_capped_multi_windows
        )
        try:
            default_max_windows = (
                DEFAULT_THREE_WINDOWS_PER_UNIT
                if use_three_windows_equal_weight
                else DEFAULT_MULTI_WINDOW_MAX_PER_UNIT
            )
            multi_window_max_per_unit = int(
                ogr_options.get("multi_window_max_per_unit", default_max_windows)
                or default_max_windows
            )
        except (TypeError, ValueError):
            multi_window_max_per_unit = (
                DEFAULT_THREE_WINDOWS_PER_UNIT
                if use_three_windows_equal_weight
                else DEFAULT_MULTI_WINDOW_MAX_PER_UNIT
            )
        if use_three_windows_equal_weight:
            multi_window_max_per_unit = int(DEFAULT_THREE_WINDOWS_PER_UNIT)
        multi_window_max_per_unit = max(1, multi_window_max_per_unit)
        default_ref_window = (
            DEFAULT_THREE_WINDOWS_SAMPLE_REF_WINDOW
            if use_three_windows_equal_weight
            else (
                DEFAULT_MULTI_WINDOW_SAMPLE_REF_WINDOW_HW5
                if use_hw5_neighborhood
                else DEFAULT_MULTI_WINDOW_SAMPLE_REF_WINDOW
            )
        )
        multi_window_sample_ref_window = ogr_options.get(
            "multi_window_sample_ref_window", default_ref_window
        )
        multi_window_sample_halfwidth = _resolve_multi_window_sample_halfwidth(
            sample_halfwidth=ogr_options.get("multi_window_sample_halfwidth"),
            sample_ref_window=multi_window_sample_ref_window,
            patch_size=patch_size,
        )
        default_distance_sigma = (
            DEFAULT_MULTI_WINDOW_DISTANCE_SIGMA_HW5
            if use_hw5_neighborhood and use_distance_weighted_multi
            else DEFAULT_MULTI_WINDOW_DISTANCE_SIGMA
        )
        try:
            multi_window_distance_sigma = float(
                ogr_options.get("multi_window_distance_sigma", default_distance_sigma)
                if ogr_options.get("multi_window_distance_sigma", None) is not None
                else default_distance_sigma
            )
        except (TypeError, ValueError):
            multi_window_distance_sigma = float(default_distance_sigma)
        if not np.isfinite(multi_window_distance_sigma) or multi_window_distance_sigma <= 0:
            multi_window_distance_sigma = float(default_distance_sigma)
        mineral_center_collision_count = 0
        mineral_center_mean_snap_distance = float("nan")
        try:
            footprint_reference_patch_size = int(
                ogr_options.get("footprint_reference_patch_size", 0) or 0
            )
        except (TypeError, ValueError):
            footprint_reference_patch_size = 0
        if footprint_reference_patch_size < 0:
            footprint_reference_patch_size = 0
        # Embargo off (0): unlabeled contamination uses the current feature window.
        # Embargo on (odd W): keep W-containment filter aligned with the embargo radius.
        label_filter_patch_size = (
            int(footprint_reference_patch_size)
            if footprint_reference_patch_size > 0
            else int(patch_size)
        )

        # Per-mineral center-window indices (allow duplicate patches when two minerals snap to the same cell).
        train_pos_window_indices = np.array([], dtype=np.int64)
        train_pos_mineral_ids = np.array([], dtype=np.int64)
        train_pos_chebyshev = np.array([], dtype=np.float64)
        test_pos_window_indices = np.array([], dtype=np.int64)
        test_pos_mineral_ids = np.array([], dtype=np.int64)
        mineral_center_train_dropped_overlap = 0

        if use_mineral_center_positives:
            n_windows = int(len(coordinates))
            train_positive_mask = np.zeros(n_windows, dtype=bool)
            test_positive_mask = np.zeros(n_windows, dtype=bool)
            train_primary_mineral_ids = np.full(n_windows, -1, dtype=np.int64)
            test_primary_mineral_ids = np.full(n_windows, -1, dtype=np.int64)

            if use_capped_multi_windows:
                neighbor_selection = (
                    "nearest" if use_three_windows_equal_weight else "random"
                )
                train_raw_w, train_raw_m, train_center_dist, train_raw_cheby = _select_capped_multi_windows_per_unit(
                    coordinates,
                    train_minerals,
                    metadata,
                    patch_size,
                    max_windows_per_unit=multi_window_max_per_unit,
                    sample_halfwidth=multi_window_sample_halfwidth,
                    sample_ref_window=multi_window_sample_ref_window,
                    random_state=int(random_state),
                    neighbor_selection=neighbor_selection,
                )
                test_raw_w, test_raw_m, test_center_dist, test_raw_cheby = _select_capped_multi_windows_per_unit(
                    coordinates,
                    test_minerals,
                    metadata,
                    patch_size,
                    max_windows_per_unit=multi_window_max_per_unit,
                    sample_halfwidth=multi_window_sample_halfwidth,
                    sample_ref_window=multi_window_sample_ref_window,
                    random_state=int(random_state) + 17,
                    neighbor_selection=neighbor_selection,
                )
                if use_three_windows_equal_weight:
                    construction_name = (
                        f"1center_{max(0, multi_window_max_per_unit - 1)}nearest_"
                        f"hw{multi_window_sample_halfwidth}_equal_1_over_n"
                    )
                else:
                    construction_name = (
                        f"fixed_neighborhood_hw{multi_window_sample_halfwidth}_"
                        f"1center_{max(0, multi_window_max_per_unit - 1)}random"
                    )
                if use_distance_weighted_multi:
                    construction_name = (
                        f"{construction_name}_gaussian_sigma{multi_window_distance_sigma:g}"
                    )
                if use_three_windows_equal_weight:
                    print(
                        "正样本: 每矿点 3 窗等权 1/3（矿点中心窗 + 2 个最近 Chebyshev 邻窗；"
                        f"邻域半宽={multi_window_sample_halfwidth}）"
                    )
            else:
                # Clean positives: one window whose geo-center is nearest to each mineral.
                train_center_idx, train_center_dist = _assign_minerals_to_center_windows(
                    coordinates, train_minerals, metadata
                )
                test_center_idx, test_center_dist = _assign_minerals_to_center_windows(
                    coordinates, test_minerals, metadata
                )
                train_raw_w = np.asarray(train_center_idx, dtype=np.int64)
                train_raw_m = np.arange(len(train_raw_w), dtype=np.int64)
                train_raw_cheby = np.zeros(len(train_raw_w), dtype=np.float64)
                test_raw_w = np.asarray(test_center_idx, dtype=np.int64)
                test_raw_m = np.arange(len(test_raw_w), dtype=np.int64)
                test_raw_cheby = np.zeros(len(test_raw_w), dtype=np.float64)
                # Drop invalid center indices before pairing.
                train_valid = (train_raw_w >= 0) & (train_raw_w < n_windows)
                test_valid = (test_raw_w >= 0) & (test_raw_w < n_windows)
                train_raw_w = train_raw_w[train_valid]
                train_raw_m = train_raw_m[train_valid]
                train_raw_cheby = train_raw_cheby[train_valid]
                train_center_dist = np.asarray(train_center_dist, dtype=np.float64)[train_valid]
                test_raw_w = test_raw_w[test_valid]
                test_raw_m = test_raw_m[test_valid]
                test_raw_cheby = test_raw_cheby[test_valid]
                test_center_dist = np.asarray(test_center_dist, dtype=np.float64)[test_valid]
                construction_name = "mineral_geo_center_nearest_patch"

            test_window_set = set(int(i) for i in test_raw_w.tolist() if 0 <= int(i) < n_windows)
            train_keep_mids = []
            train_keep_widxs = []
            train_keep_cheby = []
            for mineral_id, window_idx, cheby in zip(
                train_raw_m.tolist(),
                train_raw_w.tolist(),
                np.asarray(train_raw_cheby, dtype=np.float64).reshape(-1).tolist(),
            ):
                window_idx = int(window_idx)
                if window_idx < 0 or window_idx >= n_windows:
                    continue
                if window_idx in test_window_set:
                    # Train/test claim the same discrete window: keep as external test positive.
                    mineral_center_train_dropped_overlap += 1
                    continue
                train_keep_mids.append(int(mineral_id))
                train_keep_widxs.append(window_idx)
                train_keep_cheby.append(float(cheby))
            test_keep_mids = []
            test_keep_widxs = []
            for mineral_id, window_idx in zip(test_raw_m.tolist(), test_raw_w.tolist()):
                window_idx = int(window_idx)
                if window_idx < 0 or window_idx >= n_windows:
                    continue
                test_keep_mids.append(int(mineral_id))
                test_keep_widxs.append(window_idx)

            train_pos_window_indices = np.asarray(train_keep_widxs, dtype=np.int64)
            train_pos_mineral_ids = np.asarray(train_keep_mids, dtype=np.int64)
            train_pos_chebyshev = np.asarray(train_keep_cheby, dtype=np.float64)
            test_pos_window_indices = np.asarray(test_keep_widxs, dtype=np.int64)
            test_pos_mineral_ids = np.asarray(test_keep_mids, dtype=np.int64)

            # Unique-window masks for unlabeled contamination / buffer exclusion.
            if len(train_pos_window_indices):
                train_positive_mask[train_pos_window_indices] = True
                for mid, widx in zip(train_pos_mineral_ids.tolist(), train_pos_window_indices.tolist()):
                    train_primary_mineral_ids[int(widx)] = int(mid)
            if len(test_pos_window_indices):
                test_positive_mask[test_pos_window_indices] = True
                for mid, widx in zip(test_pos_mineral_ids.tolist(), test_pos_window_indices.tolist()):
                    test_primary_mineral_ids[int(widx)] = int(mid)

            # Same discrete center claimed by >=2 minerals (within train or within test).
            if len(train_pos_window_indices):
                _, train_counts = np.unique(train_pos_window_indices, return_counts=True)
                mineral_center_collision_count += int(np.sum(train_counts > 1))
            if len(test_pos_window_indices):
                _, test_counts = np.unique(test_pos_window_indices, return_counts=True)
                mineral_center_collision_count += int(np.sum(test_counts > 1))

            snap_dists = []
            for dist_arr in (train_center_dist, test_center_dist):
                dist_arr = np.asarray(dist_arr, dtype=np.float64).reshape(-1)
                finite = dist_arr[np.isfinite(dist_arr)]
                if len(finite):
                    snap_dists.append(finite)
            if snap_dists:
                mineral_center_mean_snap_distance = float(np.mean(np.concatenate(snap_dists)))

            overlap_removed_count = int(mineral_center_train_dropped_overlap)

            contamination_scope = str(
                ogr_options.get("unlabeled_contamination_scope") or "train_minerals"
            ).strip().lower().replace("-", "_")
            if contamination_scope in {"all", "all_minerals", "august"}:
                contamination_minerals = all_minerals
                contamination_scope = "all_minerals"
            else:
                contamination_minerals = train_minerals
                contamination_scope = "train_minerals"

            # Unlabeled contamination: for capped multi-window use the same fixed
            # neighborhood as positive sampling (not W-containment), so unlabeled
            # exclusion does not dilate with feature window size.
            if use_capped_multi_windows:
                contaminated_mask = np.zeros(n_windows, dtype=bool)
                all_center_idx, _ = _assign_minerals_to_center_windows(
                    coordinates, contamination_minerals, metadata
                )
                rows_all = np.asarray(coordinates, dtype=np.float64)[:, 0]
                cols_all = np.asarray(coordinates, dtype=np.float64)[:, 1]
                for c_idx in np.asarray(all_center_idx, dtype=np.int64).tolist():
                    c_idx = int(c_idx)
                    if not (0 <= c_idx < n_windows):
                        continue
                    chebyshev = np.maximum(
                        np.abs(rows_all - rows_all[c_idx]),
                        np.abs(cols_all - cols_all[c_idx]),
                    )
                    contaminated_mask |= chebyshev <= float(multi_window_sample_halfwidth) + 1e-9
                center_positive_mask = train_positive_mask | test_positive_mask
                contaminated_mask = contaminated_mask & (~center_positive_mask)
            else:
                contain_any_ids = _window_primary_mineral_ids(
                    coordinates,
                    contamination_minerals,
                    metadata,
                    label_filter_patch_size,
                )
                contain_any_mask = contain_any_ids >= 0
                center_positive_mask = train_positive_mask | test_positive_mask
                contaminated_mask = contain_any_mask & (~center_positive_mask)
            buffer_mask = _buffer_exclusion_mask(coordinates, train_minerals, metadata, buffer_radius)
            unlabeled_mask = (~center_positive_mask) & (~contaminated_mask) & (~buffer_mask)
            buffer_removed_count = int(np.sum((buffer_mask | contaminated_mask) & (~center_positive_mask)))
            print(
                f"Unlabeled contamination scope={contamination_scope} "
                f"(n={len(contamination_minerals)}); "
                f"contaminated={int(np.sum(contaminated_mask))}, "
                f"buffer={int(np.sum(buffer_mask))}, "
                f"unlabeled={int(np.sum(unlabeled_mask))}"
            )
            if use_capped_multi_windows:
                avg_per = (
                    float(len(train_pos_window_indices) / max(len(train_minerals), 1))
                    if len(train_minerals)
                    else 0.0
                )
                print(
                    f"Capped multi-window positives: train={len(train_pos_window_indices)} "
                    f"from {len(train_minerals)} minerals "
                    f"(max={multi_window_max_per_unit}/unit = 1 center + "
                    f"{max(0, multi_window_max_per_unit - 1)} random in fixed "
                    f"±{multi_window_sample_halfwidth} cells, refW="
                    f"{multi_window_sample_ref_window}; avg={avg_per:.2f}/unit; "
                    f"featureW={patch_size}), "
                    f"test={len(test_pos_window_indices)}/{len(test_minerals)}, "
                    f"mean center snap={mineral_center_mean_snap_distance:.2f}, "
                    f"shared-window collisions={mineral_center_collision_count}, "
                    f"train dropped by test overlap={mineral_center_train_dropped_overlap}"
                )
            else:
                print(
                    f"Mineral-centered positives: train={len(train_pos_window_indices)}/{len(train_minerals)}, "
                    f"test={len(test_pos_window_indices)}/{len(test_minerals)}, "
                    f"mean snap distance={mineral_center_mean_snap_distance:.2f}, "
                    f"shared-center collisions={mineral_center_collision_count}, "
                    f"train dropped by test overlap={mineral_center_train_dropped_overlap}"
                )
        else:
            construction_name = "sliding_window_containment"
            train_primary_mineral_ids = _window_primary_mineral_ids(coordinates, train_minerals, metadata, patch_size)
            test_primary_mineral_ids = _window_primary_mineral_ids(coordinates, test_minerals, metadata, patch_size)
            train_positive_mask = train_primary_mineral_ids >= 0
            test_positive_mask = test_primary_mineral_ids >= 0

            overlap_mask = train_positive_mask & test_positive_mask
            overlap_removed_count = int(np.sum(overlap_mask))
            if np.any(overlap_mask):
                train_positive_mask = train_positive_mask & (~overlap_mask)
                train_primary_mineral_ids[overlap_mask] = -1
                test_positive_mask = test_positive_mask | overlap_mask

            positive_mask = train_positive_mask | test_positive_mask
            buffer_mask = _buffer_exclusion_mask(coordinates, train_minerals, metadata, buffer_radius)
            unlabeled_mask = (~positive_mask) & (~buffer_mask)
            buffer_removed_count = int(np.sum(buffer_mask & (~positive_mask)))

        no_ore_removed_count = 0

        if no_ore_path:
            try:
                no_ore_points = _read_mineral_points(no_ore_path)
                if len(no_ore_points) > 0:
                    no_ore_mask = _buffer_exclusion_mask(coordinates, no_ore_points, metadata, buffer_radius)
                    no_ore_removed_count = int(np.sum(no_ore_mask & unlabeled_mask))
                    unlabeled_mask = unlabeled_mask & (~no_ore_mask)
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: failed to read no-ore coordinates: {exc}")

        unlabeled_indices = np.where(unlabeled_mask)[0]
        hard_unlabeled_gray_count = 0
        stratified_unlabeled_gray_count = 0
        unlabeled_hard_isolation_mode = ""
        if full_mineral_training:
            unlabeled_train_indices = unlabeled_indices.astype(np.int64)
            unlabeled_test_indices = np.array([], dtype=np.int64)
        elif len(unlabeled_indices) > 0:
            unlabeled_coords = np.asarray(split_coords)[unlabeled_indices]
            if spatial_split_mode == "spatial_hard":
                unlabeled_train_rel, unlabeled_test_rel, hard_unlabeled_gray_count = _split_unlabeled_indices_by_hard_clusters(
                    coords=unlabeled_coords,
                    mineral_split=mineral_split,
                    buffer_distance=spatial_cluster_cv_buffer_distance,
                )
                unlabeled_hard_isolation_mode = "hard_clusters"
            elif spatial_split_mode == "leave_one_camp":
                basin_path = str(ogr_options.get("basin_grd_path", "") or "").strip()
                basin_grid = None
                if basin_path and os.path.exists(basin_path) and read_basin_grd is not None:
                    basin_grid = read_basin_grd(basin_path)
                    mineral_split = _enrich_mineral_split_basins(mineral_split, basin_grid)
                    unlabeled_train_rel, unlabeled_test_rel, hard_unlabeled_gray_count = _split_unlabeled_indices_by_basin(
                        coords=unlabeled_coords,
                        mineral_split=mineral_split,
                        basin_grid=basin_grid,
                        buffer_distance=spatial_cluster_cv_buffer_distance,
                    )
                    unlabeled_hard_isolation_mode = "leave_one_camp_basin"
                    print(
                        f"未标记按汇水盆地整盆隔离: test_basins={mineral_split.get('test_basin_ids')} "
                        f"train_basins={len(mineral_split.get('train_basin_ids') or [])} "
                        f"gray={hard_unlabeled_gray_count}"
                    )
                else:
                    unlabeled_train_rel, unlabeled_test_rel, hard_unlabeled_gray_count = _split_unlabeled_indices_by_hard_clusters(
                        coords=unlabeled_coords,
                        mineral_split=mineral_split,
                        buffer_distance=spatial_cluster_cv_buffer_distance,
                    )
                    unlabeled_hard_isolation_mode = "leave_one_camp_hard"
                    if basin_path:
                        print(f"Warning: 汇水盆地 GRD 不可用（{basin_path}），回退最近矿田中心划分（会切开盆地）")
            elif spatial_split_mode == "leave_one_fault":
                basin_path = str(ogr_options.get("basin_grd_path", "") or "").strip()
                basin_grid = None
                if basin_path and os.path.exists(basin_path) and read_basin_grd is not None:
                    basin_grid = read_basin_grd(basin_path)
                    mineral_split = _enrich_mineral_split_basins(mineral_split, basin_grid)
                    unlabeled_train_rel, unlabeled_test_rel, hard_unlabeled_gray_count = _split_unlabeled_indices_by_basin(
                        coords=unlabeled_coords,
                        mineral_split=mineral_split,
                        basin_grid=basin_grid,
                        buffer_distance=spatial_cluster_cv_buffer_distance,
                    )
                    unlabeled_hard_isolation_mode = "leave_one_fault_basin"
                    print(
                        f"未标记按汇水盆地整盆隔离: test_basins={mineral_split.get('test_basin_ids')} "
                        f"train_basins={len(mineral_split.get('train_basin_ids') or [])} "
                        f"gray={hard_unlabeled_gray_count}"
                    )
                else:
                    unlabeled_train_rel, unlabeled_test_rel, hard_unlabeled_gray_count = _split_unlabeled_indices_by_fault(
                        coords=unlabeled_coords,
                        mineral_split=mineral_split,
                        buffer_distance=spatial_cluster_cv_buffer_distance,
                    )
                    unlabeled_hard_isolation_mode = "leave_one_fault_hard"
                    if basin_path:
                        print(f"Warning: 汇水盆地 GRD 不可用（{basin_path}），回退断裂最近邻划分")
            elif spatial_split_mode == "variogram_block_cv":
                unlabeled_train_rel, unlabeled_test_rel, hard_unlabeled_gray_count = _split_unlabeled_indices_by_variogram_blocks(
                    coords=unlabeled_coords,
                    mineral_split=mineral_split,
                    buffer_distance=spatial_cluster_cv_buffer_distance,
                )
                unlabeled_hard_isolation_mode = "variogram_block_hard"
            elif spatial_split_mode == "spatial_stratified":
                unlabeled_train_rel, unlabeled_test_rel, stratified_unlabeled_gray_count = _split_unlabeled_indices_by_spatial_stratified_folds(
                    coords=unlabeled_coords,
                    mineral_split=mineral_split,
                    buffer_distance=spatial_cluster_cv_buffer_distance,
                )
            elif spatial_split_mode == "spatial_cluster_holdout_cv":
                unlabeled_train_rel, unlabeled_test_rel, stratified_unlabeled_gray_count = _split_unlabeled_indices_by_spatial_stratified_folds(
                    coords=unlabeled_coords,
                    mineral_split=mineral_split,
                    buffer_distance=spatial_cluster_cv_buffer_distance,
                )
            else:
                unlabeled_train_rel, unlabeled_test_rel = _split_unlabeled_indices(
                    coords=unlabeled_coords,
                    train_ratio=spatial_cluster_train_ratio,
                    n_clusters=spatial_cluster_n_clusters,
                    buffer_distance=spatial_cluster_cv_buffer_distance,
                    random_state=random_state,
                )
            unlabeled_train_indices = unlabeled_indices[unlabeled_train_rel] if len(unlabeled_train_rel) else np.array([], dtype=np.int64)
            unlabeled_test_indices = unlabeled_indices[unlabeled_test_rel] if len(unlabeled_test_rel) else np.array([], dtype=np.int64)
        else:
            unlabeled_train_indices = np.array([], dtype=np.int64)
            unlabeled_test_indices = np.array([], dtype=np.int64)

        outer_footprint_embargo_removed_count = 0
        if fixed_evaluation_mask and not full_mineral_training:
            fixed_train_mask = np.zeros(len(coordinates), dtype=bool)
            fixed_train_mask[fixed_outer_train_area_indices] = True

            def _keep_outer_train_without_overlap(values):
                nonlocal outer_footprint_embargo_removed_count
                values = np.asarray(values, dtype=np.int64).reshape(-1)
                if len(values) == 0:
                    return np.zeros(0, dtype=bool)
                in_train_area = fixed_train_mask[values]
                embargoed = _patch_footprint_embargo_mask(
                    coordinates,
                    values,
                    fixed_outer_test_area_indices,
                    footprint_reference_patch_size,
                )
                outer_footprint_embargo_removed_count += int(
                    np.sum(in_train_area & embargoed)
                )
                return in_train_area & (~embargoed)

            unlabeled_keep = _keep_outer_train_without_overlap(unlabeled_train_indices)
            unlabeled_train_indices = np.asarray(
                unlabeled_train_indices,
                dtype=np.int64,
            )[unlabeled_keep]
            if use_mineral_center_positives:
                positive_keep = _keep_outer_train_without_overlap(
                    train_pos_window_indices
                )
                train_pos_window_indices = train_pos_window_indices[positive_keep]
                train_pos_mineral_ids = train_pos_mineral_ids[positive_keep]
                train_pos_chebyshev = train_pos_chebyshev[positive_keep]
            else:
                positive_indices_now = np.where(train_positive_mask)[0].astype(np.int64)
                positive_keep = _keep_outer_train_without_overlap(positive_indices_now)
                dropped_positive_indices = positive_indices_now[~positive_keep]
                if len(dropped_positive_indices):
                    train_positive_mask[dropped_positive_indices] = False
                    train_primary_mineral_ids[dropped_positive_indices] = -1
            print(
                (
                    "足迹禁运(footprint embargo): "
                    f"referenceW={footprint_reference_patch_size}, "
                    f"radius={max(footprint_reference_patch_size - 1, 0)} grid intervals, "
                    f"removed_train_centers={outer_footprint_embargo_removed_count}"
                )
                if footprint_reference_patch_size > 0
                else (
                    "足迹禁运(footprint embargo): 关闭"
                    f"（未剔除训练中心；label_filterW={label_filter_patch_size}）"
                )
            )

        if use_mineral_center_positives:
            # Keep one row per mineral (duplicate window features allowed).
            train_indices = np.concatenate((train_pos_window_indices, unlabeled_train_indices)).astype(np.int64)
            test_indices = np.concatenate((test_pos_window_indices, unlabeled_test_indices)).astype(np.int64)
            n_train_pos = int(len(train_pos_window_indices))
            n_test_pos = int(len(test_pos_window_indices))
            y_tr_full = np.concatenate(
                (
                    np.ones(n_train_pos, dtype=np.int32),
                    -np.ones(len(unlabeled_train_indices), dtype=np.int32),
                )
            )
            y_te_full = np.concatenate(
                (
                    np.ones(n_test_pos, dtype=np.int32),
                    -np.ones(len(unlabeled_test_indices), dtype=np.int32),
                )
            )
            train_mineral_ids_full = np.concatenate(
                (
                    train_pos_mineral_ids.astype(np.int64),
                    np.full(len(unlabeled_train_indices), -1, dtype=np.int64),
                )
            )
            if len(train_pos_chebyshev) == n_train_pos:
                train_chebyshev_full = np.concatenate(
                    (
                        np.asarray(train_pos_chebyshev, dtype=np.float64),
                        np.zeros(len(unlabeled_train_indices), dtype=np.float64),
                    )
                )
            else:
                train_chebyshev_full = np.zeros(len(y_tr_full), dtype=np.float64)
        else:
            train_indices = np.concatenate((np.where(train_positive_mask)[0], unlabeled_train_indices))
            test_indices = np.concatenate((np.where(test_positive_mask)[0], unlabeled_test_indices))
            train_indices = np.asarray(sorted(set(train_indices.tolist())), dtype=np.int64)
            test_indices = np.asarray(sorted(set(test_indices.tolist())), dtype=np.int64)
            y_tr_full = np.where(train_positive_mask[train_indices], 1, -1).astype(np.int32)
            y_te_full = np.where(test_positive_mask[test_indices], 1, -1).astype(np.int32)
            train_mineral_ids_full = np.where(
                train_positive_mask[train_indices], train_primary_mineral_ids[train_indices], -1
            )
            train_chebyshev_full = np.zeros(len(y_tr_full), dtype=np.float64)

        test_mineral_ids_full = None
        if fixed_evaluation_mask and not full_mineral_training:
            test_indices = np.asarray(fixed_outer_test_area_indices, dtype=np.int64)
            y_te_full = -np.ones(len(test_indices), dtype=np.int32)
            test_mineral_ids_full = _area_mineral_ids(
                np.asarray(coordinates)[test_indices],
                test_minerals,
                metadata,
            )
            y_te_full[test_mineral_ids_full >= 0] = 1

        if len(train_indices) == 0 or len(test_indices) == 0:
            if not full_mineral_training or len(train_indices) == 0:
                raise ValueError("Spatial split produced empty train/test samples. Please adjust parameters.")

        x_tr_full = x[train_indices]
        train_coords_full = np.asarray(split_coords)[train_indices]
        x_te_full = x[test_indices]
        train_pos_full = int(np.sum(y_tr_full == 1))
        train_neg_full = int(np.sum(y_tr_full == -1))
        test_pos_full = int(np.sum(y_te_full == 1))
        test_neg_full = int(np.sum(y_te_full == -1))

        train_sample_indices = (
            _sample_unlabeled_keep_all_positives(y_tr_full, sample_ratio, random_state)
            if use_mineral_center_positives
            else _sample_dev_indices_by_mineral(y_tr_full, train_mineral_ids_full, sample_ratio, random_state)
        )
        unlabeled_subsample_ratio = (
            float(sample_ratio)
            if use_mineral_center_positives and 0.0 < float(sample_ratio) < 1.0
            else None
        )
        x_tr = x_tr_full[train_sample_indices]
        y_tr = y_tr_full[train_sample_indices]
        train_coords = train_coords_full[train_sample_indices]
        sampled_mineral_ids = np.asarray(train_mineral_ids_full, dtype=np.int64)[train_sample_indices]
        sampled_chebyshev = np.asarray(train_chebyshev_full, dtype=np.float64)[train_sample_indices]
        if (not use_mineral_center_positives) and positive_window_mode == "one_window_per_unit":
            # Legacy fallback path (should be unused when center positives are enabled).
            keep_rel = _select_one_window_per_unit(train_coords, y_tr, sampled_mineral_ids, train_minerals)
            x_tr = x_tr[keep_rel]
            y_tr = y_tr[keep_rel]
            train_coords = train_coords[keep_rel]
            sampled_mineral_ids = sampled_mineral_ids[keep_rel]
            sampled_chebyshev = sampled_chebyshev[keep_rel]
            train_sample_indices = np.asarray(train_sample_indices, dtype=np.int64)[keep_rel]
        if use_distance_weighted_multi:
            deposit_weights = _deposit_loss_weights_gaussian_distance(
                sampled_mineral_ids,
                y_tr,
                sampled_chebyshev,
                sigma=multi_window_distance_sigma,
            )
        else:
            deposit_weights = _deposit_loss_weights_from_mineral_ids(sampled_mineral_ids, y_tr)
            if not bool(ogr_options.get("deposit_loss_weighting", False)):
                deposit_weights = np.ones(len(y_tr), dtype=np.float64)
        windows_per_unit = {}
        for mid in np.unique(sampled_mineral_ids[sampled_mineral_ids >= 0]):
            windows_per_unit[str(int(mid))] = int(np.sum(sampled_mineral_ids == int(mid)))
        if fixed_evaluation_mask and not full_mineral_training:
            spatial_cv_area_dataset_indices = np.asarray(
                fixed_outer_train_area_indices,
                dtype=np.int64,
            )
            spatial_cv_area_features = np.array(
                x[spatial_cv_area_dataset_indices],
                copy=True,
            )
            spatial_cv_area_positions = np.asarray(split_coords, dtype=np.float64)[
                spatial_cv_area_dataset_indices
            ]
            spatial_cv_area_mineral_ids = _area_mineral_ids(
                np.asarray(coordinates)[spatial_cv_area_dataset_indices],
                train_minerals,
                metadata,
            )
            sampled_dataset_indices = np.asarray(train_indices, dtype=np.int64)[
                np.asarray(train_sample_indices, dtype=np.int64)
            ]
            spatial_cv_sample_to_area_indices = _sample_to_fixed_area_indices(
                sampled_dataset_indices,
                spatial_cv_area_dataset_indices,
            )
        else:
            spatial_cv_area_dataset_indices = np.asarray(train_indices, dtype=np.int64)
            spatial_cv_area_features = np.array(x_tr_full, copy=True)
            spatial_cv_area_positions = np.asarray(train_coords_full, dtype=np.float64)
            spatial_cv_area_mineral_ids = np.asarray(
                train_mineral_ids_full,
                dtype=np.int64,
            )
            spatial_cv_sample_to_area_indices = np.asarray(
                train_sample_indices,
                dtype=np.int64,
            )
        x_te = x_te_full
        y_te = y_te_full
        train_pos_sampled = int(np.sum(y_tr == 1))
        train_neg_sampled = int(np.sum(y_tr == -1))
        test_pos_sampled = int(np.sum(y_te == 1))
        test_neg_sampled = int(np.sum(y_te == -1))

        print(
            f"Spatial mineral split: train minerals={len(train_minerals)}, "
            f"test minerals={len(test_minerals)}, test-window overlap removed from train={overlap_removed_count}"
        )

        split_summary.update(
            {
                "spatial_cluster_active": True,
                "spatial_split_mode": spatial_split_mode,
                "spatial_split_algorithm": str(mineral_split.get("algorithm", spatial_split_mode)),
                "spatial_cluster_n_clusters": int(mineral_split.get("n_clusters", spatial_cluster_n_clusters)),
                "spatial_cluster_train_ratio": float(mineral_split.get("train_ratio", spatial_cluster_train_ratio)),
                "spatial_cluster_random_state": int(random_state),
                "spatial_hard_isolation": bool(spatial_split_mode == "spatial_hard"),
                "spatial_hard_train_cluster_ids": mineral_split.get("train_cluster_ids", []),
                "spatial_hard_test_cluster_ids": mineral_split.get("test_cluster_ids", []),
                "spatial_hard_unlabeled_gray_count": int(locals().get("hard_unlabeled_gray_count", 0)),
                "train_basin_ids": mineral_split.get("train_basin_ids", []),
                "test_basin_ids": mineral_split.get("test_basin_ids", []),
                "basin_split_rule": str(mineral_split.get("basin_split_rule", "")),
                "basin_absorbed_train_to_test": int(mineral_split.get("basin_absorbed_train_to_test", 0) or 0),
                "basin_absorbed_test_to_train": int(mineral_split.get("basin_absorbed_test_to_train", 0) or 0),
                "basin_grd_path": str(ogr_options.get("basin_grd_path", "") or ""),
                "basin_hard_isolation_active": str(
                    locals().get("unlabeled_hard_isolation_mode", "")
                ).endswith("_basin"),
                "spatial_cluster_holdout_cv_active": bool(spatial_split_mode == "spatial_cluster_holdout_cv"),
                "spatial_cluster_holdout_requested_train_ratio": float(mineral_split.get("requested_train_ratio", spatial_cluster_train_ratio)),
                "spatial_cluster_holdout_unlabeled_gray_count": int(locals().get("stratified_unlabeled_gray_count", 0))
                if spatial_split_mode == "spatial_cluster_holdout_cv"
                else 0,
                "leave_one_camp_active": bool(spatial_split_mode == "leave_one_camp"),
                "leave_one_fault_active": bool(spatial_split_mode == "leave_one_fault"),
                "variogram_block_cv_active": bool(spatial_split_mode == "variogram_block_cv"),
                "camp_source": str(mineral_split.get("camp_source", "")),
                "fault_assignment_source": str(mineral_split.get("assignment_source", "")),
                "unlabeled_hard_isolation_mode": str(locals().get("unlabeled_hard_isolation_mode", "")),
                "unlabeled_hard_isolation_active": bool(locals().get("unlabeled_hard_isolation_mode", "")),
                "single_variogram_block_warning": bool(mineral_split.get("single_block_warning", False)),
                "variogram_grid_x0": mineral_split.get("grid_x0"),
                "variogram_grid_y0": mineral_split.get("grid_y0"),
                "held_out_block_pair": mineral_split.get("held_out_block_pair"),
                "held_out_camp_id": mineral_split.get("held_out_camp_id"),
                "held_out_fault_id": mineral_split.get("held_out_fault_id"),
                "held_out_block_id": mineral_split.get("held_out_block_id"),
                "variogram_range_m": float(ogr_options.get("variogram_range_m", 2000.0) or 2000.0),
                "variogram_range_coordinate_units": float(ogr_options.get("variogram_range_m", 2000.0) or 2000.0),
                "meters_per_coordinate_unit": ogr_options.get("meters_per_coordinate_unit"),
                "distance_protocol": ogr_options.get("distance_protocol") or {},
                "mineralization_unit_merge": merge_audit if "merge_audit" in locals() else {},
                "spatial_stratified_isolation": bool(spatial_split_mode == "spatial_stratified"),
                "spatial_stratified_fold_count": int(mineral_split.get("fold_count", spatial_cv_folds) or spatial_cv_folds),
                "spatial_stratified_train_fold_ids": mineral_split.get("train_fold_ids", []),
                "spatial_stratified_test_fold_ids": mineral_split.get("test_fold_ids", []),
                "spatial_stratified_val_fold_ids": mineral_split.get("val_fold_ids", mineral_split.get("test_fold_ids", [])),
                "spatial_stratified_unlabeled_gray_count": int(locals().get("stratified_unlabeled_gray_count", 0)),
                "mineral_training_strategy": mineral_training_strategy,
                "full_mineral_training": bool(full_mineral_training),
                "spatial_cluster_train_mineral_count": int(len(train_minerals)),
                "spatial_cluster_test_mineral_count": int(len(test_minerals)),
                "spatial_cluster_cluster_summaries": mineral_split.get("cluster_summaries", []),
                "buffer_exclusion_enabled": bool(float(buffer_radius) > 0),
                "buffer_exclusion_distance": float(buffer_radius),
                "buffer_exclusion_scope": "train_minerals_only",
                "buffer_exclusion_removed_count": int(buffer_removed_count),
                "unlabeled_contamination_scope": str(
                    locals().get("contamination_scope") or "train_minerals"
                ),
                "no_ore_active": bool(no_ore_path),
                "no_ore_exclusion_removed_count": int(no_ore_removed_count),
                "test_window_overlap_exclusion_enabled": True,
                "test_window_overlap_exclusion_applied": bool(overlap_removed_count > 0),
                "test_window_overlap_exclusion_removed_count": int(overlap_removed_count),
                "fixed_evaluation_mask": bool(fixed_evaluation_mask),
                "fixed_evaluation_mask_rule": (
                    "all valid centers in whole-basin outer partition; label/window independent"
                    if fixed_evaluation_mask
                    else "legacy label-filtered sample pool"
                ),
                "fixed_outer_train_area_count": int(
                    len(fixed_outer_train_area_indices)
                    if fixed_evaluation_mask
                    else len(train_indices)
                ),
                "fixed_outer_test_area_count": int(
                    len(fixed_outer_test_area_indices)
                    if fixed_evaluation_mask
                    else len(test_indices)
                ),
                "fixed_outer_train_mask_sha256": (
                    _indices_sha256(fixed_outer_train_area_indices)
                    if fixed_evaluation_mask
                    else ""
                ),
                "fixed_outer_test_mask_sha256": (
                    _indices_sha256(fixed_outer_test_area_indices)
                    if fixed_evaluation_mask
                    else ""
                ),
                "footprint_reference_patch_size": int(
                    footprint_reference_patch_size
                ),
                "training_label_filter_reference_patch_size": int(
                    label_filter_patch_size
                ),
                "footprint_embargo_enabled": bool(footprint_reference_patch_size > 0),
                "footprint_embargo_radius_grid_intervals": int(
                    max(footprint_reference_patch_size - 1, 0)
                ),
                "outer_footprint_embargo_removed_train_centers": int(
                    outer_footprint_embargo_removed_count
                ),
                "train_total_full": int(len(y_tr_full)),
                "train_positive_full": int(train_pos_full),
                "train_negative_full": int(train_neg_full),
                "test_total_full": int(len(y_te_full)),
                "test_positive_full": int(test_pos_full),
                "test_negative_full": int(test_neg_full),
                "train_total_sampled": int(len(y_tr)),
                "train_positive_sampled": int(train_pos_sampled),
                "train_negative_sampled": int(train_neg_sampled),
                "test_total_sampled": int(len(y_te)),
                "test_positive_sampled": int(test_pos_sampled),
                "test_negative_sampled": int(test_neg_sampled),
                "sampling_scope": "unlabeled_only" if use_mineral_center_positives else "dev_only",
                "positive_sampling_unit": (
                    "deposit_unit"
                    if positive_window_mode
                    in {
                        "one_window_per_unit",
                        THREE_WINDOWS_EQUAL_WEIGHT_MODE,
                        "multi_window_weighted",
                        "multi_window_distance_weighted",
                        "multi_window_weighted_hw5",
                        "multi_window_distance_weighted_hw5",
                    }
                    else "mineral_point"
                ),
                "positive_window_mode": positive_window_mode,
                "deposit_loss_weighting": bool(ogr_options.get("deposit_loss_weighting", False)),
                "deposit_loss_weighting_scheme": (
                    "gaussian_distance"
                    if use_distance_weighted_multi
                    else ("inverse_count_1_over_n" if bool(ogr_options.get("deposit_loss_weighting", False)) else "none")
                ),
                "multi_window_distance_sigma": (
                    float(multi_window_distance_sigma) if use_distance_weighted_multi else None
                ),
                "positive_window_construction": str(
                    construction_name
                    if use_mineral_center_positives
                    else "sliding_window_containment"
                ),
                "multi_window_max_per_unit": (
                    int(multi_window_max_per_unit) if use_capped_multi_windows else None
                ),
                "multi_window_sample_halfwidth": (
                    int(multi_window_sample_halfwidth) if use_capped_multi_windows else None
                ),
                "multi_window_sample_ref_window": (
                    int(multi_window_sample_ref_window)
                    if use_capped_multi_windows
                    and str(multi_window_sample_ref_window).strip() != ""
                    else None
                ),
                "mineral_center_mean_snap_distance_coordinate_units": (
                    float(mineral_center_mean_snap_distance)
                    if use_mineral_center_positives and np.isfinite(mineral_center_mean_snap_distance)
                    else None
                ),
                "mineral_center_mean_snap_distance_m": (
                    float(mineral_center_mean_snap_distance)
                    * float(ogr_options.get("meters_per_coordinate_unit", 1.0) or 1.0)
                    if use_mineral_center_positives and np.isfinite(mineral_center_mean_snap_distance)
                    else None
                ),
                "mineral_center_collision_count": int(mineral_center_collision_count) if use_mineral_center_positives else 0,
                "mineral_center_train_dropped_by_test_overlap": int(mineral_center_train_dropped_overlap) if use_mineral_center_positives else 0,
                "deposit_loss_weighting": bool(ogr_options.get("deposit_loss_weighting", False)),
                "windows_per_unit": windows_per_unit,
                "metric_unit": str(ogr_options.get("metric_unit", "deposit_unit") or "deposit_unit"),
                "unlabeled_sampling_scope": "unlabeled_only" if use_mineral_center_positives else "dev_only",
                "test_sampling_applied": False,
                "unlabeled_candidate_count": int(np.sum(unlabeled_mask)),
                "unlabeled_train_count": int(len(unlabeled_train_indices)),
                "unlabeled_test_count": int(len(unlabeled_test_indices)),
            }
        )
    else:
        if label_ext in {".txt", ".csv", ".tsv"}:
            minerals = _read_mineral_points(label_path)
            positive_mask = _window_contains_minerals(coordinates, minerals, metadata, patch_size) > 0
            y = np.where(positive_mask, 1, -1).astype(np.int32)
            buffer_mask = _buffer_exclusion_mask(coordinates, minerals, metadata, buffer_radius)
            keep_mask = positive_mask | (~buffer_mask)
        elif label_ext in {".h5", ".hdf5"}:
            y = _read_label_h5(label_path)
            keep_mask = np.ones(len(y), dtype=bool)
        else:
            raise ValueError("Label file must be TXT/CSV/TSV mineral points or H5 labels.")

        if no_ore_path:
            try:
                no_ore_points = _read_mineral_points(no_ore_path)
                if len(no_ore_points) > 0:
                    no_ore_mask = _buffer_exclusion_mask(coordinates, no_ore_points, metadata, buffer_radius)
                    y = np.asarray(y, dtype=np.int32).copy()
                    y[no_ore_mask] = -1
                    keep_mask |= no_ore_mask
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: failed to read no-ore coordinates: {exc}")

        if label_ext in {".txt", ".csv", ".tsv"} and not np.all(keep_mask):
            x = x[keep_mask]
            y = y[keep_mask]
            if coordinates is not None:
                coordinates = np.asarray(coordinates)[keep_mask]

        if len(x) != len(y):
            raise ValueError(f"Feature/label length mismatch: features={len(x)}, labels={len(y)}")

        if sample_ratio < 1.0:
            rng = np.random.default_rng(random_state)
            pos_indices = np.where(y == 1)[0]
            neg_indices = np.where(y == -1)[0]
            pos_sample_size = max(1, int(len(pos_indices) * sample_ratio)) if len(pos_indices) > 0 else 0
            neg_sample_size = max(1, int(len(neg_indices) * sample_ratio)) if len(neg_indices) > 0 else 0
            pos_selected = rng.choice(pos_indices, pos_sample_size, replace=False) if pos_sample_size > 0 else np.array([], dtype=int)
            neg_selected = rng.choice(neg_indices, neg_sample_size, replace=False) if neg_sample_size > 0 else np.array([], dtype=int)
            selected_indices = np.concatenate((pos_selected, neg_selected))
            rng.shuffle(selected_indices)
            x = x[selected_indices]
            y = y[selected_indices]
            if coordinates is not None:
                coordinates = np.asarray(coordinates)[selected_indices]

        if split_mode in SPATIAL_MINERAL_SPLIT_MODES:
            split_coords = _patch_indices_to_geo(coordinates, metadata)
            if split_coords is None:
                raise ValueError("Spatial split requires valid patch coordinates.")
            train_index, test_index, _ = _spatial_cluster_split_indices(
                split_coords,
                y,
                n_clusters=spatial_cluster_n_clusters,
                train_ratio=spatial_cluster_train_ratio,
                buffer_distance=spatial_cluster_cv_buffer_distance,
                random_state=random_state,
            )
            if len(train_index) == 0 or len(test_index) == 0:
                raise ValueError("Spatial split produced empty train/test samples.")
            x_tr, x_te = x[train_index], x[test_index]
            y_tr, y_te = y[train_index], y[test_index]
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            for train_index, test_index in sss.split(x, y):
                x_tr, x_te = x[train_index], x[test_index]
                y_tr, y_te = y[train_index], y[test_index]

    print(f"Input mode: {feature_mode}, label mode: {'coordinates' if label_ext in {'.txt', '.csv', '.tsv'} else 'h5'}")
    print(f"Train size: {len(x_tr)}, Test size: {len(x_te)}")
    print(f"Train positives: {int(np.sum(y_tr == 1))}, Test positives: {int(np.sum(y_te == 1))}")

    print("\nStart normalization...")
    n_channels = x_tr.shape[1]
    mean_per_channel = []
    std_per_channel = []
    channel_names_for_fault = list(
        metadata.get("selected_channel_names")
        or metadata.get("available_channel_names")
        or []
    )
    fault_raw_payload = {}
    try:
        from nested_fault_calibration import capture_raw_fault_distance

        area_for_raw = spatial_cv_area_features if "spatial_cv_area_features" in locals() else None
        fault_raw_payload = capture_raw_fault_distance(
            x_tr,
            x_te,
            channel_names_for_fault,
            area_features=area_for_raw,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to snapshot raw fault distance channel: {exc}")

    for channel in range(n_channels):
        channel_data = x_tr[:, channel, :, :]
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        mean_per_channel.append(mean)
        std_per_channel.append(std)

        print(f"Channel {channel}: mean={mean:.4f}, std={std:.4f}")
        x_tr[:, channel, :, :] = (channel_data - mean) / (std + 1e-8)
        x_te[:, channel, :, :] = (x_te[:, channel, :, :] - mean) / (std + 1e-8)
        if use_spatial_mineral_split and "spatial_cv_area_features" in locals():
            spatial_cv_area_features[:, channel, :, :] = (
                spatial_cv_area_features[:, channel, :, :] - mean
            ) / (std + 1e-8)

    normalization_params = {"mean": mean_per_channel, "std": std_per_channel}
    unlabeled_ratio_now = locals().get("unlabeled_subsample_ratio")
    raw_prior = _empirical_positive_rate(y_tr, unlabeled_subsample_ratio=None)
    restored_prior = _empirical_positive_rate(
        y_tr, unlabeled_subsample_ratio=unlabeled_ratio_now
    )
    normalization_params["unlabeled_subsample_ratio"] = unlabeled_ratio_now
    normalization_params["calculated_prior_raw_subsampled"] = float(raw_prior)
    normalization_params["calculated_prior"] = float(restored_prior)
    normalization_params["calculated_prior_rule"] = (
        "positives_kept; unlabeled restored by 1/sample_ratio"
        if unlabeled_ratio_now is not None
        else "empirical_positive_rate_on_sampled_rows"
    )
    if channel_names_for_fault:
        normalization_params["selected_channel_names"] = list(channel_names_for_fault)
    if fault_raw_payload:
        normalization_params.update(fault_raw_payload)
        # Immutable source snapshots for nested multi-fold rematerialize + decay_only drop.
        idxs = fault_raw_payload.get("fault_channel_indices")
        if isinstance(idxs, dict):
            normalization_params["fault_channel_indices_source"] = dict(idxs)
        if channel_names_for_fault:
            normalization_params["selected_channel_names_source"] = list(channel_names_for_fault)
    if use_spatial_mineral_split:
        if "mineral_unit_mapping" in locals():
            normalization_params["mineral_unit_mapping"] = mineral_unit_mapping.to_dict(orient="records")
            normalization_params["mineralization_unit_merge"] = merge_audit
        normalization_params["distance_protocol"] = ogr_options.get("distance_protocol") or {}
        if "deposit_weights" in locals():
            normalization_params["deposit_loss_weights"] = np.asarray(deposit_weights, dtype=np.float64)
            normalization_params["deposit_loss_weighting"] = bool(ogr_options.get("deposit_loss_weighting", False))
            normalization_params["positive_window_mode"] = str(ogr_options.get("positive_window_mode", "multi_window"))
            normalization_params["deposit_loss_weighting_scheme"] = (
                "gaussian_distance"
                if use_distance_weighted_multi
                else ("inverse_count_1_over_n" if bool(ogr_options.get("deposit_loss_weighting", False)) else "none")
            )
            normalization_params["multi_window_distance_sigma"] = (
                float(multi_window_distance_sigma) if use_distance_weighted_multi else None
            )
            normalization_params["multi_window_max_per_unit"] = (
                int(multi_window_max_per_unit) if use_capped_multi_windows else None
            )
            normalization_params["multi_window_sample_halfwidth"] = (
                int(multi_window_sample_halfwidth) if use_capped_multi_windows else None
            )
            normalization_params["multi_window_sample_ref_window"] = (
                int(DEFAULT_MULTI_WINDOW_SAMPLE_REF_WINDOW)
                if use_capped_multi_windows
                else None
            )
            try:
                if use_capped_multi_windows:
                    normalization_params["multi_window_sample_ref_window"] = int(
                        multi_window_sample_ref_window
                    )
            except (TypeError, ValueError):
                pass
            normalization_params["metric_unit"] = str(ogr_options.get("metric_unit", "deposit_unit"))
            prior_sampling_weights = np.ones(len(y_tr), dtype=np.float64)
            if 0.0 < float(sample_ratio) < 1.0:
                prior_sampling_weights[np.asarray(y_tr) != 1] = 1.0 / float(sample_ratio)
            normalization_params["prior_sampling_weights"] = prior_sampling_weights
            normalization_params["prior_sampling_weight_rule"] = (
                "positive=1; sampled_unlabeled=1/sample_ratio"
            )
        if "train_minerals" in locals() and "test_minerals" in locals():
            try:
                from spatial_split_report import summarize_train_test_distances

                normalization_params["train_test_distance_summary"] = summarize_train_test_distances(
                    train_minerals,
                    test_minerals,
                    meters_per_coordinate_unit=float(
                        ogr_options.get("meters_per_coordinate_unit", 1.0) or 1.0
                    ),
                )
            except Exception:
                pass
        if "mineral_split" in locals() and isinstance(mineral_split, dict):
            geometry = _build_spatial_unit_geometry(spatial_split_mode, mineral_split)
            normalization_params["spatial_unit_geometry"] = geometry
            split_summary["spatial_unit_geometry"] = geometry
            split_summary["unlabeled_hard_isolation_mode"] = str(
                locals().get("unlabeled_hard_isolation_mode", "")
            )
            split_summary["unlabeled_hard_isolation_active"] = bool(
                locals().get("unlabeled_hard_isolation_mode", "")
            )
        full_count = int(len(split_coords)) if "split_coords" in locals() and split_coords is not None else 0
        saved_test_indices = np.asarray(test_indices if "test_indices" in locals() else [], dtype=np.int64)
        test_mask = np.zeros(full_count, dtype=bool)
        if full_count > 0 and len(saved_test_indices) > 0:
            valid_test_indices = saved_test_indices[(saved_test_indices >= 0) & (saved_test_indices < full_count)]
            test_mask[valid_test_indices] = True
        normalization_params["test_indices"] = saved_test_indices
        normalization_params["test_mask"] = test_mask
        if full_count > 0 and len(saved_test_indices) > 0:
            normalization_params["test_positions"] = np.asarray(split_coords, dtype=np.float64)[saved_test_indices]
        else:
            normalization_params["test_positions"] = np.empty((0, 2), dtype=np.float64)
        if "test_minerals" in locals():
            normalization_params["test_mineral_positions"] = test_minerals[["x", "y"]].to_numpy(dtype=np.float64)
            if "test_mineral_ids_full" in locals() and test_mineral_ids_full is not None:
                normalization_params["test_mineral_ids"] = np.asarray(
                    test_mineral_ids_full,
                    dtype=np.int64,
                )
            elif "test_primary_mineral_ids" in locals() and "test_indices" in locals():
                normalization_params["test_mineral_ids"] = np.asarray(test_primary_mineral_ids, dtype=np.int64)[
                    np.asarray(test_indices, dtype=np.int64)
                ]
        if "train_minerals" in locals():
            normalization_params["train_mineral_positions"] = train_minerals[["x", "y"]].to_numpy(dtype=np.float64)
        if "train_indices" in locals() and "train_sample_indices" in locals():
            sampled_train_indices = np.asarray(train_indices, dtype=np.int64)[np.asarray(train_sample_indices, dtype=np.int64)]
            normalization_params["train_indices"] = sampled_train_indices
            sample_area_indices = np.asarray(
                spatial_cv_sample_to_area_indices
                if "spatial_cv_sample_to_area_indices" in locals()
                else train_sample_indices,
                dtype=np.int64,
            )
            normalization_params["train_sample_area_indices"] = sample_area_indices
            sampled_area_mask = np.zeros(
                len(spatial_cv_area_dataset_indices)
                if "spatial_cv_area_dataset_indices" in locals()
                else len(train_indices),
                dtype=bool,
            )
            valid_area_indices = sample_area_indices[(sample_area_indices >= 0) & (sample_area_indices < len(sampled_area_mask))]
            sampled_area_mask[valid_area_indices] = True
            normalization_params["spatial_cv_area_sampled_mask"] = sampled_area_mask
        if "train_coords" in locals():
            normalization_params["train_positions"] = np.asarray(train_coords, dtype=np.float64)
        if "train_mineral_ids_full" in locals() and "train_sample_indices" in locals():
            normalization_params["train_mineral_ids"] = np.asarray(train_mineral_ids_full, dtype=np.int64)[
                np.asarray(train_sample_indices, dtype=np.int64)
            ]
            prior_ids = np.asarray(normalization_params["train_mineral_ids"], dtype=np.int64)
            prior_groups = np.empty(len(prior_ids), dtype=object)
            positive_group_mask = prior_ids >= 0
            prior_groups[positive_group_mask] = [f"unit_{int(value)}" for value in prior_ids[positive_group_mask]]
            unlabeled_group_mask = ~positive_group_mask
            if np.any(unlabeled_group_mask) and "train_coords" in locals():
                prior_coords = np.asarray(train_coords, dtype=np.float64)
                block_size = float(ogr_options.get("variogram_range_m", 1.0) or 1.0)
                block_size = max(block_size, 1e-12)
                x0 = float(np.nanmin(prior_coords[:, 0]))
                y0 = float(np.nanmin(prior_coords[:, 1]))
                block_x = np.floor((prior_coords[:, 0] - x0) / block_size).astype(np.int64)
                block_y = np.floor((prior_coords[:, 1] - y0) / block_size).astype(np.int64)
                prior_groups[unlabeled_group_mask] = [
                    f"block_{int(x)}_{int(y)}"
                    for x, y in zip(block_x[unlabeled_group_mask], block_y[unlabeled_group_mask])
                ]
            elif np.any(unlabeled_group_mask):
                prior_groups[unlabeled_group_mask] = [
                    f"unlabeled_{index}" for index in np.where(unlabeled_group_mask)[0]
                ]
            normalization_params["prior_estimation_groups"] = prior_groups
            normalization_params["prior_group_rule"] = (
                "positive=mineralization_unit; unlabeled=variogram_range_spatial_block"
            )
            group_column = {
                "leave_one_camp": "camp_id",
                "leave_one_fault": "fault_id",
                "variogram_block_cv": "block_id",
            }.get(spatial_split_mode)
            if group_column and group_column in train_minerals.columns:
                group_values = train_minerals[group_column].astype(str).to_numpy(dtype=object)
                train_ids = np.asarray(normalization_params["train_mineral_ids"], dtype=np.int64)
                train_outer_groups = np.full(len(train_ids), "unlabeled", dtype=object)
                valid = (train_ids >= 0) & (train_ids < len(group_values))
                train_outer_groups[valid] = group_values[train_ids[valid]]
                normalization_params["train_outer_group_ids"] = train_outer_groups
            if group_column and group_column in test_minerals.columns and normalization_params.get("test_mineral_ids") is not None:
                group_values = test_minerals[group_column].astype(str).to_numpy(dtype=object)
                test_ids = np.asarray(normalization_params["test_mineral_ids"], dtype=np.int64)
                test_outer_groups = np.full(len(test_ids), "unlabeled", dtype=object)
                valid = (test_ids >= 0) & (test_ids < len(group_values))
                test_outer_groups[valid] = group_values[test_ids[valid]]
                normalization_params["test_outer_group_ids"] = test_outer_groups
        if "train_minerals" in locals():
            normalization_params["train_mineral_positions"] = train_minerals[["x", "y"]].to_numpy(dtype=np.float64)
        # Preview overlays: train-only buffer radius + patch window size (geo units ≈ coord units).
        normalization_params["buffer_exclusion_distance"] = float(buffer_radius)
        normalization_params["buffer_exclusion_scope"] = "train_minerals_only"
        normalization_params["patch_size"] = int(patch_size)
        try:
            nx = int(metadata.get("nx", metadata.get("image_width", metadata.get("width", 0)) or 0))
            ny = int(metadata.get("ny", metadata.get("image_height", metadata.get("height", 0)) or 0))
            x_min = float(metadata["x_min"])
            x_max = float(metadata["x_max"])
            y_min = float(metadata["y_min"])
            y_max = float(metadata["y_max"])
            if nx > 1 and ny > 1:
                normalization_params["grid_x_step"] = abs(x_max - x_min) / max(nx - 1, 1)
                normalization_params["grid_y_step"] = abs(y_max - y_min) / max(ny - 1, 1)
                if "footprint_reference_patch_size" in locals():
                    meters_per_unit = float(
                        ogr_options.get("meters_per_coordinate_unit", 1.0) or 1.0
                    )
                    radius_intervals = max(
                        int(footprint_reference_patch_size) - 1,
                        0,
                    )
                    split_summary["footprint_embargo_distance_x_m"] = float(
                        radius_intervals
                        * normalization_params["grid_x_step"]
                        * meters_per_unit
                    )
                    split_summary["footprint_embargo_distance_y_m"] = float(
                        radius_intervals
                        * normalization_params["grid_y_step"]
                        * meters_per_unit
                    )
        except (KeyError, TypeError, ValueError):
            pass
        if "spatial_cv_area_features" in locals():
            normalization_params["spatial_cv_area_features"] = spatial_cv_area_features
        if "spatial_cv_area_positions" in locals():
            normalization_params["spatial_cv_area_positions"] = np.asarray(
                spatial_cv_area_positions,
                dtype=np.float64,
            )
        if "spatial_cv_area_mineral_ids" in locals():
            normalization_params["spatial_cv_area_mineral_ids"] = np.asarray(
                spatial_cv_area_mineral_ids,
                dtype=np.int64,
            )
        if "spatial_cv_area_dataset_indices" in locals():
            normalization_params["spatial_cv_area_dataset_indices"] = np.asarray(
                spatial_cv_area_dataset_indices,
                dtype=np.int64,
            )
    inner_camp_cv_mode_opt = str(ogr_options.get("inner_camp_cv_mode") or "").strip().lower()
    force_inner_camp_cv = (
        str(spatial_split_mode) == "leave_one_camp"
        and inner_camp_cv_mode_opt
        in {"leave_one_camp", "leave_one_unit", "random_one_camp_val", "random_one_unit"}
    )
    if use_spatial_mineral_split and (int(spatial_cv_folds) > 1 or force_inner_camp_cv):
        sampled_train_mineral_ids = np.asarray(train_mineral_ids_full, dtype=np.int64)[
            np.asarray(train_sample_indices, dtype=np.int64)
        ]
        spatial_cv_fallback_used = False
        spatial_cv_fallback_reason = ""
        if spatial_split_mode == "spatial_hard":
            folds = _build_hard_isolated_spatial_cv_folds(
                train_coords,
                y_tr,
                sampled_train_mineral_ids,
                train_minerals,
                n_folds=int(spatial_cv_folds),
                buffer_distance=float(spatial_cluster_cv_buffer_distance),
                area_coords=spatial_cv_area_positions,
                area_mineral_ids=spatial_cv_area_mineral_ids,
                sample_to_area_indices=spatial_cv_sample_to_area_indices,
                strategy_name="hard_cluster",
            )
            if not folds:
                spatial_cv_fallback_used = True
                spatial_cv_fallback_reason = "hard_cluster_cv_unavailable"
        elif spatial_split_mode in {"leave_one_camp", "leave_one_fault", "variogram_block_cv"}:
            train_minerals_for_cv = _ensure_unit_ids_on_train_minerals(train_minerals, spatial_split_mode)
            strategy_by_mode = {
                "leave_one_camp": "hard_unit_camp",
                "leave_one_fault": "hard_unit_fault",
                "variogram_block_cv": "hard_unit_block",
            }
            inner_camp_cv_mode = str(ogr_options.get("inner_camp_cv_mode") or "").strip().lower()
            if spatial_split_mode == "leave_one_camp" and inner_camp_cv_mode in {
                "leave_one_camp",
                "leave_one_unit",
                "random_one_camp_val",
                "random_one_unit",
            }:
                unit_fold_mode = inner_camp_cv_mode
            elif spatial_split_mode == "leave_one_camp" and bool(ogr_options.get("reviewer_primary_protocol")):
                # M / reviewer-primary default: explicit leave-one among modeling camps.
                unit_fold_mode = "leave_one_camp"
            else:
                unit_fold_mode = "packed"
            area_basin_ids = None
            sample_basin_ids = None
            mineral_basin_ids = None
            reviewer_primary_inner = bool(ogr_options.get("reviewer_primary_protocol", False))
            if spatial_split_mode == "leave_one_camp":
                basin_path = str(ogr_options.get("basin_grd_path", "") or "").strip()
                basin_ready = (
                    bool(basin_path)
                    and os.path.exists(basin_path)
                    and read_basin_grd is not None
                    and sample_basin_ids_at_coords is not None
                )
                if basin_ready:
                    try:
                        basin_grid_inner = read_basin_grd(basin_path)
                        area_xy = np.asarray(spatial_cv_area_positions, dtype=np.float64)
                        sample_xy = np.asarray(train_coords, dtype=np.float64)
                        if area_xy.ndim != 2 or area_xy.shape[1] < 2:
                            raise ValueError("inner CV area coordinates are not 2-D")
                        if sample_xy.ndim != 2 or sample_xy.shape[1] < 2:
                            raise ValueError("inner CV sample coordinates are not 2-D")
                        area_basin_ids = sample_basin_ids_at_coords(
                            basin_grid_inner,
                            area_xy[:, :2],
                        )
                        sample_basin_ids = sample_basin_ids_at_coords(
                            basin_grid_inner,
                            sample_xy[:, :2],
                        )
                        if "basin_id" in train_minerals_for_cv.columns:
                            mineral_basin_ids = pd.to_numeric(
                                train_minerals_for_cv["basin_id"],
                                errors="coerce",
                            ).to_numpy(dtype=np.float64)
                        else:
                            mineral_basin_ids = sample_basin_ids_at_coords(
                                basin_grid_inner,
                                train_minerals_for_cv[["x", "y"]].to_numpy(dtype=np.float64),
                            )
                        print("内层未标记按汇水盆地整盆归属（与外层相同：训练优先、空盆跟最近矿点）。")
                    except Exception as exc:
                        area_basin_ids = None
                        sample_basin_ids = None
                        mineral_basin_ids = None
                        if reviewer_primary_inner:
                            raise ValueError(
                                "Reviewer-primary inner leave-one-camp unlabeled split requires a valid "
                                "basin GRD so unlabeled cells follow the same whole-basin assignment "
                                "as the outer split."
                            ) from exc
                        print(f"Warning: 内层汇水盆地整盆归属失败（{exc}），回退最近矿田中心划分。")
                elif reviewer_primary_inner:
                    raise ValueError(
                        "Reviewer-primary inner leave-one-camp unlabeled split requires a valid "
                        "basin GRD so unlabeled cells follow the same whole-basin assignment "
                        "as the outer split."
                    )
            folds = _build_hard_isolated_spatial_cv_folds(
                train_coords,
                y_tr,
                sampled_train_mineral_ids,
                train_minerals_for_cv,
                n_folds=int(spatial_cv_folds),
                buffer_distance=float(spatial_cluster_cv_buffer_distance),
                area_coords=spatial_cv_area_positions,
                area_mineral_ids=spatial_cv_area_mineral_ids,
                sample_to_area_indices=spatial_cv_sample_to_area_indices,
                strategy_name=strategy_by_mode.get(spatial_split_mode, "hard_unit"),
                unit_fold_mode=unit_fold_mode,
                random_state=int(random_state),
                footprint_embargo_grid_cells=(
                    max(footprint_reference_patch_size - 1, 0)
                    if fixed_evaluation_mask
                    else 0
                ),
                area_basin_ids=area_basin_ids,
                sample_basin_ids=sample_basin_ids,
                mineral_basin_ids=mineral_basin_ids,
            )
            if folds:
                normalization_params["inner_camp_cv_mode"] = unit_fold_mode
                print(
                    f"内层矿田CV模式={unit_fold_mode}，有效折数={len(folds)}"
                    + (
                        f"（请求={int(spatial_cv_folds)}）"
                        if int(spatial_cv_folds) != len(folds)
                        else ""
                    )
                    + (
                        f"；隔离缓冲={float(spatial_cluster_cv_buffer_distance):g} 坐标单位"
                        f"（汇水盆地边界，固定评价格网下仍生效）"
                        if float(spatial_cluster_cv_buffer_distance) > 0
                        else "；隔离缓冲=0"
                    )
                )
                if reviewer_primary_inner and spatial_split_mode == "leave_one_camp":
                    isolations = {
                        str(fold.get("unlabeled_isolation") or "") for fold in folds
                    }
                    if isolations != {"whole_basin"}:
                        raise ValueError(
                            "Reviewer-primary inner leave-one-camp unlabeled split did not "
                            "apply whole-basin assignment (got "
                            f"{sorted(isolations)})."
                        )
            if not folds:
                if spatial_split_mode == "leave_one_camp":
                    raise RuntimeError(
                        "leave-one-camp inner hard-isolation folds are unavailable; "
                        "refusing a same-camp cluster-stratified fallback."
                    )
                spatial_cv_fallback_used = True
                spatial_cv_fallback_reason = "hard_unit_cv_unavailable"
                folds = _build_cluster_stratified_spatial_cv_folds(
                    train_coords,
                    y_tr,
                    sampled_train_mineral_ids,
                    train_minerals_for_cv,
                    n_folds=int(spatial_cv_folds),
                    buffer_distance=float(spatial_cluster_cv_buffer_distance),
                    area_coords=spatial_cv_area_positions,
                    area_mineral_ids=spatial_cv_area_mineral_ids,
                    sample_to_area_indices=spatial_cv_sample_to_area_indices,
                )
                if folds:
                    print(
                        f"Warning: {spatial_split_mode} 整单元硬隔离 CV 不可用，"
                        "已回退到簇内空间分层 CV（训练/验证可能同单元）。"
                    )
                else:
                    spatial_cv_fallback_reason = "hard_unit_and_cluster_stratified_cv_unavailable"
        elif spatial_split_mode == "spatial_cluster_holdout_cv":
            folds = _build_spatial_stratified_cv_folds(
                train_coords,
                y_tr,
                sampled_train_mineral_ids,
                train_minerals,
                n_clusters=int(spatial_cluster_n_clusters),
                n_folds=int(spatial_cv_folds),
                buffer_distance=float(spatial_cluster_cv_buffer_distance),
                random_state=int(random_state),
                area_coords=spatial_cv_area_positions,
                area_mineral_ids=spatial_cv_area_mineral_ids,
                sample_to_area_indices=spatial_cv_sample_to_area_indices,
            )
            if not folds:
                spatial_cv_fallback_used = True
                spatial_cv_fallback_reason = "cluster_holdout_dev_spatial_cv_unavailable"
        elif spatial_split_mode == "spatial_stratified":
            folds = _build_spatial_stratified_cv_folds(
                train_coords,
                y_tr,
                sampled_train_mineral_ids,
                train_minerals,
                n_clusters=int(spatial_cluster_n_clusters),
                n_folds=int(spatial_cv_folds),
                buffer_distance=float(spatial_cluster_cv_buffer_distance),
                random_state=int(random_state),
                area_coords=spatial_cv_area_positions,
                area_mineral_ids=spatial_cv_area_mineral_ids,
                sample_to_area_indices=spatial_cv_sample_to_area_indices,
            )
            if not folds:
                spatial_cv_fallback_used = True
                spatial_cv_fallback_reason = "spatial_stratified_cv_unavailable"
        else:
            folds = _build_cluster_stratified_spatial_cv_folds(
                train_coords,
                y_tr,
                sampled_train_mineral_ids,
                train_minerals,
                n_folds=int(spatial_cv_folds),
                buffer_distance=float(spatial_cluster_cv_buffer_distance),
                area_coords=spatial_cv_area_positions,
                area_mineral_ids=spatial_cv_area_mineral_ids,
                sample_to_area_indices=spatial_cv_sample_to_area_indices,
            )
            if not folds:
                spatial_cv_fallback_used = True
                spatial_cv_fallback_reason = "cluster_stratified_cv_unavailable"
        if not folds:
            if spatial_split_mode == "leave_one_camp":
                raise RuntimeError(
                    "leave-one-camp inner hard-isolation folds are unavailable; "
                    "refusing an axis-block fallback that can mix camps."
                )
            folds = _build_spatial_cv_folds(
                train_coords,
                y_tr,
                n_folds=int(spatial_cv_folds),
                buffer_distance=float(spatial_cluster_cv_buffer_distance),
                partition_coords=train_mineral_coords,
                area_coords=spatial_cv_area_positions,
            )
        if folds:
            normalization_params["spatial_cv_folds"] = folds
            normalization_params["spatial_cv_fold_count"] = int(len(folds))
            normalization_params["spatial_cv_buffer_distance"] = float(spatial_cluster_cv_buffer_distance)
            normalization_params["spatial_cv_axis"] = str(folds[0].get("axis", "x"))
            normalization_params["spatial_cv_axis_source"] = str(folds[0].get("axis_source", "x_train_mineral_quantile"))
            normalization_params["spatial_cv_strategy"] = str(folds[0].get("strategy", "axis_block"))
            normalization_params["spatial_cv_hard_isolation"] = bool(folds[0].get("spatial_cv_hard_isolation", False))
            normalization_params["spatial_cv_fallback_used"] = bool(spatial_cv_fallback_used)
            normalization_params["spatial_cv_fallback_reason"] = str(spatial_cv_fallback_reason)
            normalization_params["spatial_cv_requested_fold_count"] = int(spatial_cv_folds)
            normalization_params["spatial_cv_effective_fold_count"] = int(len(folds))
            split_summary["spatial_cv_enabled"] = True
            split_summary["spatial_cv_fold_count"] = int(len(folds))
            split_summary["spatial_cv_buffer_distance"] = float(spatial_cluster_cv_buffer_distance)
            split_summary["spatial_cv_selected_fold"] = 0
            split_summary["spatial_cv_axis"] = str(folds[0].get("axis", "x"))
            split_summary["spatial_cv_axis_source"] = str(folds[0].get("axis_source", "x_train_mineral_quantile"))
            split_summary["spatial_cv_strategy"] = str(folds[0].get("strategy", "axis_block"))
            split_summary["spatial_cv_hard_isolation"] = bool(folds[0].get("spatial_cv_hard_isolation", False))
            split_summary["spatial_cv_fallback_used"] = bool(spatial_cv_fallback_used)
            split_summary["spatial_cv_fallback_reason"] = str(spatial_cv_fallback_reason)
            split_summary["spatial_cv_requested_fold_count"] = int(spatial_cv_folds)
            split_summary["spatial_cv_effective_fold_count"] = int(len(folds))
            split_summary["spatial_cv_fold_summaries"] = [
                {
                    "fold": int(fold.get("fold", index)),
                    "strategy": str(fold.get("strategy", "")),
                    "train_count": int(fold.get("train_count", len(fold.get("train_indices", [])))),
                    "val_count": int(fold.get("val_count", len(fold.get("val_indices", [])))),
                    "val_positive_count": int(fold.get("val_positive_count", -1)),
                    "val_unlabeled_count": int(fold.get("val_unlabeled_count", -1)),
                    "train_cluster_ids": fold.get("train_cluster_ids", []),
                    "val_cluster_ids": fold.get("val_cluster_ids", []),
                    "gray_count": int(fold.get("gray_count", fold.get("buffer_excluded_sample_count", 0))),
                }
                for index, fold in enumerate(folds)
            ]
        else:
            split_summary["spatial_cv_enabled"] = False
            split_summary["spatial_cv_fold_count"] = 0
            split_summary["spatial_cv_buffer_distance"] = float(spatial_cluster_cv_buffer_distance)
            split_summary["spatial_cv_fallback_used"] = bool(spatial_cv_fallback_used)
            split_summary["spatial_cv_fallback_reason"] = str(spatial_cv_fallback_reason or "no_valid_spatial_cv_folds")
    elif use_spatial_mineral_split:
        split_summary["spatial_cv_enabled"] = False
        split_summary["spatial_cv_fold_count"] = int(spatial_cv_folds)
        split_summary["spatial_cv_buffer_distance"] = float(spatial_cluster_cv_buffer_distance)

    if not use_spatial_mineral_split:
        split_summary.setdefault("spatial_cluster_active", bool(str(split_mode).strip().lower() in SPATIAL_MINERAL_SPLIT_MODES))
        split_summary.setdefault("buffer_exclusion_enabled", bool(float(buffer_radius) > 0))
        split_summary.setdefault("buffer_exclusion_distance", float(buffer_radius))
        split_summary.setdefault("buffer_exclusion_scope", "train_minerals_only")

    normalization_params["split_summary"] = split_summary

    print(f"Train tensor shape: {x_tr.shape}")
    print(f"Test tensor shape: {x_te.shape}")

    return (x_tr, y_tr), (x_te, y_te), normalization_params

def binarize_h5_class(y_train, y_test, positive_label=1):
    y_train_bin = np.ones(len(y_train), dtype=np.int32)
    y_test_bin = np.ones(len(y_test), dtype=np.int32)
    y_train_bin[y_train != positive_label] = -1
    y_test_bin[y_test != positive_label] = -1
    return y_train_bin, y_test_bin


def load_dataset(
    data_path=None,
    label_path=None,
    test_size=0.2,
    random_state=42,
    sample_ratio=1.0,
    split_mode="legacy",
    patch_size=None,
    patch_stride=None,
    buffer_radius=0.0,
    spatial_cluster_n_clusters=10,
    spatial_cluster_train_ratio=0.7,
    full_mineral_training=False,
    mineral_training_strategy="holdout",
    spatial_cluster_cv_buffer_distance=0.0,
    spatial_cv_folds=1,
    no_ore_path=None,
    use_reflect_padding=False,
    selected_channels=None,
    positive_window_mode="three_windows_equal_weight",
    deposit_loss_weighting=True,
    leave_one_camp_index=0,
    leave_one_fault_id="",
    deposit_fault_assignment="",
    fault_lines_path="",
    variogram_range_m=2000.0,
    metric_unit="deposit_unit",
    ogr_options=None,
):
    if not data_path:
        raise ValueError("Need feature H5 path.")
    if not label_path:
        raise ValueError("Need label path.")

    (x_tr, y_tr), (x_te, y_te), normalization_params = get_h5_data(
        data_path,
        label_path,
        test_size=test_size,
        random_state=random_state,
        sample_ratio=sample_ratio,
        split_mode=split_mode,
        patch_size=patch_size,
        patch_stride=patch_stride,
        buffer_radius=buffer_radius,
        spatial_cluster_n_clusters=spatial_cluster_n_clusters,
        spatial_cluster_train_ratio=spatial_cluster_train_ratio,
        full_mineral_training=full_mineral_training,
        mineral_training_strategy=mineral_training_strategy,
        spatial_cluster_cv_buffer_distance=spatial_cluster_cv_buffer_distance,
        spatial_cv_folds=spatial_cv_folds,
        no_ore_path=no_ore_path,
        use_reflect_padding=use_reflect_padding,
        selected_channels=selected_channels,
        positive_window_mode=positive_window_mode,
        deposit_loss_weighting=deposit_loss_weighting,
        leave_one_camp_index=leave_one_camp_index,
        leave_one_fault_id=leave_one_fault_id,
        deposit_fault_assignment=deposit_fault_assignment,
        fault_lines_path=fault_lines_path,
        variogram_range_m=variogram_range_m,
        metric_unit=metric_unit,
        ogr_options=ogr_options,
    )

    unlabeled_subsample_ratio = None
    if isinstance(normalization_params, dict):
        unlabeled_subsample_ratio = normalization_params.get("unlabeled_subsample_ratio")
    prior = _empirical_positive_rate(
        y_tr, unlabeled_subsample_ratio=unlabeled_subsample_ratio
    )
    if isinstance(normalization_params, dict):
        normalization_params["calculated_prior"] = float(prior)
        if unlabeled_subsample_ratio is not None:
            raw_prior = _empirical_positive_rate(y_tr, unlabeled_subsample_ratio=None)
            normalization_params["calculated_prior_raw_subsampled"] = float(raw_prior)
            print(
                f"Calculated prior: {prior:.4f} "
                f"(restored unlabeled by 1/{float(unlabeled_subsample_ratio):g}; "
                f"raw subsampled P/(P+U)={raw_prior:.4f} was not used)"
            )
        else:
            print(f"Calculated prior: {prior:.4f}")
    else:
        print(f"Calculated prior: {prior:.4f}")
    return (x_tr, y_tr), (x_te, y_te), prior, normalization_params


def calculate_sampling_stats(
    data_path,
    label_path,
    sample_ratio=1.0,
    test_size=0.2,
    random_state=42,
    split_mode="legacy",
    patch_size=None,
    patch_stride=None,
    buffer_radius=0.0,
    spatial_cluster_n_clusters=10,
    spatial_cluster_train_ratio=0.7,
    full_mineral_training=False,
    mineral_training_strategy="holdout",
    spatial_cluster_cv_buffer_distance=0.0,
    cv_folds=1,
    no_ore_path=None,
    use_reflect_padding=False,
    selected_channels=None,
):
    if not data_path:
        raise ValueError("Need feature H5 path.")
    if not label_path:
        raise ValueError("Need label path.")
    if not 0 < sample_ratio <= 1:
        raise ValueError("sample_ratio must be in (0, 1].")
    spatial_split_mode = str(split_mode or "legacy").strip().lower()
    if spatial_split_mode not in SPATIAL_MINERAL_SPLIT_MODES and not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1) for legacy mode.")

    patch_size = patch_size or 16
    patch_stride = patch_stride or patch_size
    x, coordinates, metadata, _ = _load_feature_tensor(
        data_path,
        patch_size,
        patch_stride,
        use_reflect_padding=use_reflect_padding,
        selected_channels=selected_channels,
    )
    label_ext = os.path.splitext(label_path)[1].lower()
    use_spatial_mineral_split = spatial_split_mode in SPATIAL_MINERAL_SPLIT_MODES and label_ext in {".txt", ".csv", ".tsv"}

    mineral_training_strategy = str(mineral_training_strategy or "holdout").strip().lower()
    if full_mineral_training:
        mineral_training_strategy = "all_minerals"
    if mineral_training_strategy not in {"holdout", "train_val", "all_minerals", "holdout_cv"}:
        mineral_training_strategy = "holdout"
    full_mineral_training = mineral_training_strategy == "all_minerals"

    if use_spatial_mineral_split:
        all_minerals = _read_mineral_points(label_path)
        split_coords = _patch_indices_to_geo(coordinates, metadata)
        if split_coords is None or len(split_coords) == 0:
            raise ValueError("Spatial split requires valid patch coordinates.")

        if full_mineral_training:
            if spatial_split_mode == "spatial_stratified":
                mineral_split = _all_train_spatial_stratified_split(
                    all_minerals,
                    n_clusters=spatial_cluster_n_clusters,
                    n_folds=cv_folds,
                    random_state=random_state,
                )
                train_minerals = mineral_split["train"]
                test_minerals = mineral_split["test"]
            else:
                train_minerals = all_minerals.reset_index(drop=True).copy()
                test_minerals = all_minerals.iloc[0:0].copy()
                mineral_split = {
                    "train": train_minerals,
                    "test": test_minerals,
                    "n_clusters": int(spatial_cluster_n_clusters),
                    "train_ratio": 1.0,
                }
        else:
            if spatial_split_mode == "spatial_hard":
                mineral_split = _split_minerals_by_hard_clusters(
                    all_minerals,
                    n_clusters=spatial_cluster_n_clusters,
                    train_ratio=spatial_cluster_train_ratio,
                    random_state=random_state,
                )
            elif spatial_split_mode == "spatial_stratified":
                mineral_split = _split_minerals_by_spatial_stratified_folds(
                    all_minerals,
                    n_clusters=spatial_cluster_n_clusters,
                    n_folds=cv_folds,
                    train_ratio=spatial_cluster_train_ratio,
                    random_state=random_state,
                )
            elif spatial_split_mode == "spatial_cluster_holdout_cv":
                mineral_split = _split_minerals_by_cluster_holdout_cv(
                    all_minerals,
                    n_clusters=spatial_cluster_n_clusters,
                    train_ratio=spatial_cluster_train_ratio,
                    random_state=random_state,
                )
            else:
                mineral_split = _split_minerals_by_kmeans(
                    all_minerals,
                    n_clusters=spatial_cluster_n_clusters,
                    train_ratio=spatial_cluster_train_ratio,
                    random_state=random_state,
                )
            train_minerals = mineral_split["train"]
            test_minerals = mineral_split["test"]
        if spatial_split_mode == "spatial_stratified":
            all_partition = pd.concat([train_minerals, test_minerals], ignore_index=True)
            mineral_split["mineral_coords"] = all_partition[["x", "y"]].to_numpy(dtype=np.float64)
            mineral_split["mineral_fold_ids"] = pd.to_numeric(
                all_partition.get("spatial_fold", pd.Series(np.full(len(all_partition), -1))),
                errors="coerce",
            ).fillna(-1).to_numpy(dtype=np.int64)
        elif spatial_split_mode == "spatial_cluster_holdout_cv":
            all_partition = pd.concat([train_minerals, test_minerals], ignore_index=True)
            mineral_split["mineral_coords"] = all_partition[["x", "y"]].to_numpy(dtype=np.float64)
            mineral_split["mineral_fold_ids"] = np.concatenate(
                (
                    np.zeros(len(train_minerals), dtype=np.int64),
                    np.ones(len(test_minerals), dtype=np.int64),
                )
            )
            mineral_split["train_fold_ids"] = [0]
            mineral_split["test_fold_ids"] = [1]
            mineral_split["val_fold_ids"] = [1]
        train_mineral_coords = train_minerals[["x", "y"]].to_numpy(dtype=np.float64)

        train_primary_mineral_ids = _window_primary_mineral_ids(coordinates, train_minerals, metadata, patch_size)
        test_primary_mineral_ids = _window_primary_mineral_ids(coordinates, test_minerals, metadata, patch_size)
        train_positive_mask = train_primary_mineral_ids >= 0
        test_positive_mask = test_primary_mineral_ids >= 0
        overlap_mask = train_positive_mask & test_positive_mask
        overlap_removed_count = int(np.sum(overlap_mask))
        if np.any(overlap_mask):
            train_positive_mask = train_positive_mask & (~overlap_mask)
            train_primary_mineral_ids[overlap_mask] = -1
            test_positive_mask = test_positive_mask | overlap_mask

        positive_mask = train_positive_mask | test_positive_mask
        buffer_mask = _buffer_exclusion_mask(coordinates, train_minerals, metadata, buffer_radius)
        unlabeled_mask = (~positive_mask) & (~buffer_mask)

        if no_ore_path:
            try:
                no_ore_points = _read_mineral_points(no_ore_path)
                if len(no_ore_points) > 0:
                    unlabeled_mask |= _buffer_exclusion_mask(coordinates, no_ore_points, metadata, buffer_radius)
            except Exception:
                pass

        unlabeled_indices = np.where(unlabeled_mask)[0]
        if full_mineral_training:
            unlabeled_train_indices = unlabeled_indices.astype(np.int64)
            unlabeled_test_indices = np.array([], dtype=np.int64)
        elif len(unlabeled_indices) > 0:
            unlabeled_coords = np.asarray(split_coords)[unlabeled_indices]
            if spatial_split_mode == "spatial_hard":
                unlabeled_train_rel, unlabeled_test_rel, _ = _split_unlabeled_indices_by_hard_clusters(
                    coords=unlabeled_coords,
                    mineral_split=mineral_split,
                    buffer_distance=spatial_cluster_cv_buffer_distance,
                )
            elif spatial_split_mode == "spatial_stratified":
                unlabeled_train_rel, unlabeled_test_rel, _ = _split_unlabeled_indices_by_spatial_stratified_folds(
                    coords=unlabeled_coords,
                    mineral_split=mineral_split,
                    buffer_distance=spatial_cluster_cv_buffer_distance,
                )
            elif spatial_split_mode == "spatial_cluster_holdout_cv":
                unlabeled_train_rel, unlabeled_test_rel, _ = _split_unlabeled_indices_by_spatial_stratified_folds(
                    coords=unlabeled_coords,
                    mineral_split=mineral_split,
                    buffer_distance=spatial_cluster_cv_buffer_distance,
                )
            else:
                unlabeled_train_rel, unlabeled_test_rel = _split_unlabeled_indices(
                    coords=unlabeled_coords,
                    train_ratio=spatial_cluster_train_ratio,
                    n_clusters=spatial_cluster_n_clusters,
                    buffer_distance=spatial_cluster_cv_buffer_distance,
                    random_state=random_state,
                )
            unlabeled_train_indices = unlabeled_indices[unlabeled_train_rel] if len(unlabeled_train_rel) else np.array([], dtype=np.int64)
            unlabeled_test_indices = unlabeled_indices[unlabeled_test_rel] if len(unlabeled_test_rel) else np.array([], dtype=np.int64)
        else:
            unlabeled_train_indices = np.array([], dtype=np.int64)
            unlabeled_test_indices = np.array([], dtype=np.int64)

        train_indices = np.concatenate((np.where(train_positive_mask)[0], unlabeled_train_indices))
        test_indices = np.concatenate((np.where(test_positive_mask)[0], unlabeled_test_indices))
        train_indices = np.asarray(sorted(set(train_indices.tolist())), dtype=np.int64)
        test_indices = np.asarray(sorted(set(test_indices.tolist())), dtype=np.int64)

        y_dev = np.where(train_positive_mask[train_indices], 1, -1).astype(np.int32)
        dev_mineral_ids = np.where(train_positive_mask[train_indices], train_primary_mineral_ids[train_indices], -1)
        y_test = np.where(test_positive_mask[test_indices], 1, -1).astype(np.int32)

        dev_sample_indices = np.arange(len(y_dev), dtype=np.int64)
        if sample_ratio < 1.0:
            dev_sample_indices = _sample_dev_indices_by_mineral(y_dev, dev_mineral_ids, sample_ratio, random_state)
            y_dev = y_dev[dev_sample_indices]
        dev_mineral_ids_sampled = dev_mineral_ids[dev_sample_indices]

        if len(y_dev) >= 2:
            dev_coords_full = np.asarray(split_coords)[train_indices]
            dev_coords = dev_coords_full[dev_sample_indices]
            if int(cv_folds) > 1:
                if spatial_split_mode == "spatial_cluster_holdout_cv":
                    spatial_folds = _build_spatial_stratified_cv_folds(
                        dev_coords,
                        y_dev,
                        dev_mineral_ids_sampled,
                        train_minerals,
                        n_clusters=int(spatial_cluster_n_clusters),
                        n_folds=int(cv_folds),
                        buffer_distance=float(spatial_cluster_cv_buffer_distance),
                        random_state=int(random_state),
                        area_coords=dev_coords_full,
                        area_mineral_ids=dev_mineral_ids,
                        sample_to_area_indices=dev_sample_indices,
                    )
                elif spatial_split_mode == "spatial_stratified":
                    spatial_folds = _build_spatial_stratified_cv_folds(
                        dev_coords,
                        y_dev,
                        dev_mineral_ids_sampled,
                        train_minerals,
                        n_clusters=int(spatial_cluster_n_clusters),
                        n_folds=int(cv_folds),
                        buffer_distance=float(spatial_cluster_cv_buffer_distance),
                        random_state=int(random_state),
                        area_coords=dev_coords_full,
                        area_mineral_ids=dev_mineral_ids,
                        sample_to_area_indices=dev_sample_indices,
                    )
                else:
                    spatial_folds = _build_cluster_stratified_spatial_cv_folds(
                        dev_coords,
                        y_dev,
                        dev_mineral_ids_sampled,
                        train_minerals,
                        n_folds=int(cv_folds),
                        buffer_distance=float(spatial_cluster_cv_buffer_distance),
                        area_coords=dev_coords_full,
                        area_mineral_ids=dev_mineral_ids,
                        sample_to_area_indices=dev_sample_indices,
                    )
                if not spatial_folds:
                    spatial_folds = _build_spatial_cv_folds(
                        dev_coords,
                        y_dev,
                        n_folds=int(cv_folds),
                        buffer_distance=float(spatial_cluster_cv_buffer_distance),
                        partition_coords=train_mineral_coords,
                    )
            else:
                spatial_folds = []
            if spatial_folds:
                first_fold = spatial_folds[0]
                inner_train_idx = np.asarray(first_fold["train_indices"], dtype=np.int64)
                inner_val_idx = np.asarray(first_fold["val_indices"], dtype=np.int64)
                y_inner_train = y_dev[inner_train_idx]
                y_inner_val = y_dev[inner_val_idx]
            else:
                folds = max(int(cv_folds), 1)
                inner_val_ratio = (1.0 / float(folds)) if folds > 1 else 0.2
                inner_val_ratio = float(min(max(inner_val_ratio, 0.1), 0.5))
                dummy = np.zeros((len(y_dev), 1), dtype=np.float32)
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=inner_val_ratio, random_state=random_state)
                inner_train_idx, inner_val_idx = next(splitter.split(dummy, y_dev))
                y_inner_train = y_dev[inner_train_idx]
                y_inner_val = y_dev[inner_val_idx]
        else:
            y_inner_train = y_dev
            y_inner_val = np.array([], dtype=np.int32)

        if mineral_training_strategy in {"train_val", "all_minerals"}:
            reported_train = y_dev
            reported_val = np.array([], dtype=np.int32)
        else:
            reported_train = y_inner_train
            reported_val = y_inner_val

        return {
            "original_positive": int(np.sum(positive_mask)),
            "original_negative": int(np.sum(unlabeled_mask)),
            "sampled_positive": int(np.sum(y_dev == 1)),
            "sampled_negative": int(np.sum(y_dev == -1)),
            "sampled_total": int(len(y_dev)),
            "reflect_padding": bool(use_reflect_padding),
            "selected_channel_names": metadata.get("selected_channel_names", metadata.get("available_channel_names", [])),
            "buffer_removed_count": int(np.sum(buffer_mask & (~positive_mask))),
            "train_total": int(len(reported_train)),
            "train_positive": int(np.sum(reported_train == 1)),
            "train_negative": int(np.sum(reported_train == -1)),
            "val_total": int(len(reported_val)),
            "val_positive": int(np.sum(reported_val == 1)),
            "val_negative": int(np.sum(reported_val == -1)),
            "external_test_total": int(len(y_test)),
            "external_test_positive": int(np.sum(y_test == 1)),
            "external_test_negative": int(np.sum(y_test == -1)),
            "mineral_training_strategy": mineral_training_strategy,
            "full_mineral_training": bool(full_mineral_training),
            "dev_total": int(len(y_dev)),
            "dev_positive": int(np.sum(y_dev == 1)),
            "dev_negative": int(np.sum(y_dev == -1)),
            "spatial_train_minerals": int(len(train_minerals)),
            "spatial_test_minerals": int(len(test_minerals)),
            "spatial_split_mode": spatial_split_mode,
            "spatial_split_algorithm": str(mineral_split.get("algorithm", spatial_split_mode)),
            "spatial_hard_isolation": bool(spatial_split_mode == "spatial_hard"),
            "spatial_cluster_holdout_cv_active": bool(spatial_split_mode == "spatial_cluster_holdout_cv"),
            "spatial_cluster_holdout_requested_train_ratio": float(mineral_split.get("requested_train_ratio", spatial_cluster_train_ratio)),
            "spatial_stratified_isolation": bool(spatial_split_mode == "spatial_stratified"),
            "spatial_stratified_fold_count": int(mineral_split.get("fold_count", cv_folds) or cv_folds),
            "spatial_stratified_train_fold_ids": mineral_split.get("train_fold_ids", []),
            "spatial_stratified_test_fold_ids": mineral_split.get("test_fold_ids", []),
            "spatial_stratified_val_fold_ids": mineral_split.get("val_fold_ids", mineral_split.get("test_fold_ids", [])),
            "spatial_overlap_removed": int(overlap_removed_count),
            "sampling_scope": "dev_only",
            "positive_sampling_unit": "mineral_point",
            "unlabeled_sampling_scope": "dev_only",
            "test_sampling_applied": False,
        }

    if label_ext in {".txt", ".csv", ".tsv"}:
        minerals = _read_mineral_points(label_path)
        positive_mask = _window_contains_minerals(coordinates, minerals, metadata, patch_size) > 0
        labels = np.where(positive_mask, 1, -1).astype(np.int32)
        buffer_mask = _buffer_exclusion_mask(coordinates, minerals, metadata, buffer_radius)
        keep_mask = positive_mask | (~buffer_mask)
    elif label_ext in {".h5", ".hdf5"}:
        labels = _read_label_h5(label_path)
        keep_mask = np.ones(len(labels), dtype=bool)
        buffer_mask = np.zeros(len(labels), dtype=bool)
        positive_mask = labels == 1
    else:
        raise ValueError("Label file must be TXT/CSV/TSV mineral points or H5 labels.")

    if no_ore_path:
        try:
            no_ore_points = _read_mineral_points(no_ore_path)
            if len(no_ore_points) > 0:
                no_ore_mask = _buffer_exclusion_mask(coordinates, no_ore_points, metadata, buffer_radius)
                labels = np.asarray(labels, dtype=np.int32).copy()
                labels[no_ore_mask] = -1
                keep_mask |= no_ore_mask
        except Exception:
            pass

    if not np.all(keep_mask):
        labels = labels[keep_mask]

    pos_total = int(np.sum(labels == 1))
    neg_total = int(np.sum(labels == -1))
    if pos_total + neg_total == 0:
        raise ValueError("No valid samples after filtering.")

    if sample_ratio < 1.0:
        dummy = np.zeros((len(labels), 1), dtype=np.float32)
        _, sampled_labels = _sample_split_arrays(dummy, labels, sample_ratio, random_state)
    else:
        sampled_labels = labels

    sampled_total = int(len(sampled_labels))
    dummy_x = np.zeros((sampled_total, 1), dtype=np.float32)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_total = val_total = train_pos = val_pos = 0
    for train_idx, val_idx in sss.split(dummy_x, sampled_labels):
        train_total = len(train_idx)
        val_total = len(val_idx)
        train_pos = int(np.sum(sampled_labels[train_idx] == 1))
        val_pos = int(np.sum(sampled_labels[val_idx] == 1))
        break

    return {
        "original_positive": pos_total,
        "original_negative": neg_total,
        "sampled_positive": int(np.sum(sampled_labels == 1)),
        "sampled_negative": int(np.sum(sampled_labels == -1)),
        "sampled_total": sampled_total,
        "reflect_padding": bool(use_reflect_padding),
        "selected_channel_names": metadata.get("selected_channel_names", metadata.get("available_channel_names", [])),
        "buffer_removed_count": int(np.sum(buffer_mask & (~positive_mask))) if len(buffer_mask) == len(positive_mask) else 0,
        "train_total": train_total,
        "train_positive": train_pos,
        "train_negative": train_total - train_pos,
        "val_total": val_total,
        "val_positive": val_pos,
        "val_negative": val_total - val_pos,
    }

if __name__ == "__main__":
    try:
        print("=== 绋嬪簭寮€濮嬫墽琛?===")
        print("寮€濮嬪姞杞藉拰澶勭悊鏁版嵁闆?..")
        xy_train, xy_test, prior, normalization_params = load_dataset()
        print("\n=== 鏁版嵁闆嗗垱寤哄畬鎴?===")
        print(f"璁粌闆嗗ぇ灏? {len(xy_train)}")
        print(f"娴嬭瘯闆嗗ぇ灏? {len(xy_test)}")
        print(f"鍏堥獙姒傜巼: {prior}")
    except Exception as e:  # noqa: BLE001
        print(f"\n绋嬪簭鎵ц鍑洪敊: {e}")
        print(f"閿欒绫诲瀷: {type(e).__name__}")
        import traceback

        print("\n璇︾粏閿欒淇℃伅:")
        traceback.print_exc()


