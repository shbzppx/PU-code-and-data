"""Spatial splitting helpers for mineral-driven model comparison."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def _find_column_alias(frame: pd.DataFrame, aliases) -> Optional[str]:
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for alias in aliases:
        matched = normalized.get(str(alias).strip().lower())
        if matched is not None:
            return matched
    return None


def _resolve_xy_columns(frame: pd.DataFrame):
    x_column = _find_column_alias(
        frame,
        ["x", "coord_x", "point_x", "east", "easting", "x坐标", "横坐标"],
    )
    y_column = _find_column_alias(
        frame,
        ["y", "coord_y", "point_y", "north", "northing", "y坐标", "纵坐标"],
    )
    if x_column is None or y_column is None:
        raise KeyError("Mineral file must contain X/Y coordinate columns.")
    return x_column, y_column


def _ensure_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(columns=["x", "y"])
    return frame.reset_index(drop=True).copy()


def split_minerals_by_kmeans(
    mineral_df: pd.DataFrame,
    *,
    n_clusters: int = 10,
    train_ratio: float = 0.7,
    random_state: int = 42,
) -> Dict[str, object]:
    """Split minerals into train/test groups with spatial KMeans stratification."""

    frame = _ensure_frame(mineral_df)
    if len(frame) == 0:
        empty = frame.iloc[0:0].copy()
        return {
            "train": empty.copy(),
            "test": empty.copy(),
            "train_indices": [],
            "test_indices": [],
            "cluster_ids": [],
            "cluster_summaries": [],
            "n_clusters": int(n_clusters),
            "train_ratio": float(train_ratio),
            "random_state": int(random_state),
        }

    x_column, y_column = _resolve_xy_columns(frame)
    coords = frame[[x_column, y_column]].to_numpy(dtype=np.float64)
    unique_count = len(np.unique(coords, axis=0))
    cluster_count = max(1, min(int(n_clusters), len(frame), unique_count))
    train_ratio = float(train_ratio)
    if not np.isfinite(train_ratio):
        train_ratio = 0.7
    train_ratio = float(min(max(train_ratio, 0.1), 0.9))

    if cluster_count == 1 or len(frame) < 3:
        rng = np.random.default_rng(int(random_state))
        order = rng.permutation(len(frame))
        train_count = max(1, int(round(len(frame) * train_ratio)))
        train_count = min(train_count, len(frame) - 1) if len(frame) > 1 else len(frame)
        train_rows = np.sort(order[:train_count])
        test_rows = np.sort(order[train_count:])
        return {
            "train": frame.iloc[train_rows].reset_index(drop=True),
            "test": frame.iloc[test_rows].reset_index(drop=True),
            "train_indices": train_rows.astype(np.int64).tolist(),
            "test_indices": test_rows.astype(np.int64).tolist(),
            "cluster_ids": [0] * len(frame),
            "cluster_summaries": [
                {
                    "cluster_id": 0,
                    "sample_count": int(len(frame)),
                    "train_count": int(len(train_rows)),
                    "test_count": int(len(test_rows)),
                }
            ],
            "n_clusters": 1,
            "train_ratio": float(train_ratio),
            "random_state": int(random_state),
        }

    kmeans = KMeans(n_clusters=cluster_count, random_state=int(random_state), n_init=10)
    cluster_ids = kmeans.fit_predict(coords)
    rng = np.random.default_rng(int(random_state))
    train_rows = []
    test_rows = []
    cluster_summaries = []

    for cluster_id in sorted(np.unique(cluster_ids)):
        cluster_rows = np.where(cluster_ids == cluster_id)[0]
        cluster_size = int(len(cluster_rows))
        if cluster_size == 0:
            continue
        order = rng.permutation(cluster_rows)
        if cluster_size == 1:
            train_count = 1
        else:
            train_count = int(round(cluster_size * train_ratio))
            train_count = max(1, min(train_count, cluster_size - 1))
        cluster_train_rows = np.sort(order[:train_count])
        cluster_test_rows = np.sort(order[train_count:])
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

    train_rows = np.asarray(sorted(set(train_rows)), dtype=np.int64)
    test_rows = np.asarray(sorted(set(test_rows)), dtype=np.int64)

    # Match the training module semantics: after per-cluster allocation,
    # enforce the global train/test target implied by the requested ratio.
    target_train = int(round(len(frame) * train_ratio))
    target_train = max(1, min(target_train, len(frame) - 1)) if len(frame) > 1 else len(frame)
    if len(train_rows) > target_train:
        move_count = len(train_rows) - target_train
        move_indices = rng.choice(train_rows, move_count, replace=False)
        train_rows = np.asarray(sorted(set(train_rows.tolist()) - set(move_indices.tolist())), dtype=np.int64)
        test_rows = np.asarray(sorted(set(test_rows.tolist()) | set(move_indices.tolist())), dtype=np.int64)
    elif len(train_rows) < target_train and len(test_rows) > 0:
        move_count = target_train - len(train_rows)
        move_indices = rng.choice(test_rows, min(move_count, len(test_rows)), replace=False)
        test_rows = np.asarray(sorted(set(test_rows.tolist()) - set(move_indices.tolist())), dtype=np.int64)
        train_rows = np.asarray(sorted(set(train_rows.tolist()) | set(move_indices.tolist())), dtype=np.int64)

    cluster_summaries = []
    train_row_set = set(train_rows.tolist())
    test_row_set = set(test_rows.tolist())
    for cluster_id in sorted(np.unique(cluster_ids)):
        cluster_rows = np.where(cluster_ids == cluster_id)[0]
        cluster_row_set = set(cluster_rows.tolist())
        cluster_summaries.append(
            {
                "cluster_id": int(cluster_id),
                "sample_count": int(len(cluster_rows)),
                "train_count": int(len(cluster_row_set & train_row_set)),
                "test_count": int(len(cluster_row_set & test_row_set)),
            }
        )

    return {
        "train": frame.iloc[train_rows].reset_index(drop=True),
        "test": frame.iloc[test_rows].reset_index(drop=True),
        "train_indices": train_rows.astype(np.int64).tolist(),
        "test_indices": test_rows.astype(np.int64).tolist(),
        "cluster_ids": np.asarray(cluster_ids, dtype=np.int64).tolist(),
        "cluster_summaries": cluster_summaries,
        "n_clusters": int(cluster_count),
        "train_ratio": float(train_ratio),
        "random_state": int(random_state),
    }


def _quantile_edges(values: np.ndarray, n_folds: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0.0, 1.0, int(n_folds) + 1))
    edges = np.asarray(edges, dtype=np.float64)
    edges[0] = float(np.min(values))
    edges[-1] = float(np.max(values))
    if np.unique(edges).size < len(edges):
        edges = np.linspace(float(np.min(values)), float(np.max(values)), int(n_folds) + 1)
    return edges


def _partition_score(result: Dict[str, object]) -> float:
    fold_sizes = np.asarray(result.get("fold_sizes", []), dtype=np.float64)
    if len(fold_sizes) == 0:
        return float("inf")

    score = float(np.std(fold_sizes))
    if np.any(fold_sizes <= 0):
        score += 1e6

    positive_rates = np.asarray(result.get("fold_positive_rates", []), dtype=np.float64)
    overall_positive_rate = result.get("overall_positive_rate")
    if overall_positive_rate is not None and len(positive_rates) > 0:
        score += float(np.nanmean(np.abs(positive_rates - float(overall_positive_rate))))
    return score


def _build_axis_partition(
    coords: np.ndarray,
    *,
    axis: int,
    n_folds: int,
    buffer_distance: float,
    labels: Optional[np.ndarray] = None,
    partition_coords: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    axis = int(axis)
    axis_name = "x" if axis == 0 else "y"
    values = np.asarray(coords[:, axis], dtype=np.float64)
    reference_coords = coords
    if partition_coords is not None:
        reference_coords = np.asarray(partition_coords, dtype=np.float64)
        if reference_coords.ndim != 2 or reference_coords.shape[1] < 2 or len(reference_coords) == 0:
            reference_coords = coords
        else:
            reference_coords = reference_coords[:, :2]
    reference_values = np.asarray(reference_coords[:, axis], dtype=np.float64)
    if np.nanmax(reference_values) == np.nanmin(reference_values):
        raise ValueError(f"Spatial CV requires variability along axis {axis_name}.")

    edges = _quantile_edges(reference_values, n_folds)
    boundaries = edges[1:-1]
    block_ids = np.searchsorted(boundaries, values, side="right")
    block_ids = np.clip(block_ids, 0, int(n_folds) - 1)

    gray_mask = np.zeros(len(values), dtype=bool)
    for boundary in boundaries:
        gray_mask |= np.abs(values - float(boundary)) <= float(buffer_distance)

    x_min = float(np.min(coords[:, 0]))
    x_max = float(np.max(coords[:, 0]))
    y_min = float(np.min(coords[:, 1]))
    y_max = float(np.max(coords[:, 1]))

    folds = []
    fold_sizes = []
    fold_positive_rates = []
    overall_positive_rate = None
    if labels is not None and len(labels) > 0:
        labels = np.asarray(labels).reshape(-1)
        overall_positive_rate = float(np.mean(labels > 0)) if len(labels) > 0 else None

    for fold_index in range(int(n_folds)):
        val_mask = (block_ids == fold_index) & (~gray_mask)
        train_mask = (block_ids != fold_index) & (~gray_mask)
        train_indices = np.where(train_mask)[0].astype(np.int64)
        val_indices = np.where(val_mask)[0].astype(np.int64)
        fold_sizes.append(int(len(val_indices)))
        if labels is not None and len(labels) > 0 and len(val_indices) > 0:
            fold_positive_rates.append(float(np.mean(labels[val_indices] > 0)))
        else:
            fold_positive_rates.append(0.0)
        if axis == 0:
            val_region = {
                "xmin": float(edges[fold_index]),
                "xmax": float(edges[fold_index + 1]),
                "ymin": y_min,
                "ymax": y_max,
            }
        else:
            val_region = {
                "xmin": x_min,
                "xmax": x_max,
                "ymin": float(edges[fold_index]),
                "ymax": float(edges[fold_index + 1]),
            }
        folds.append(
            {
                "fold_index": int(fold_index),
                "train_indices": train_indices.tolist(),
                "val_indices": val_indices.tolist(),
                "train_count": int(len(train_indices)),
                "val_count": int(len(val_indices)),
                "val_region": val_region,
            }
        )

    return {
        "axis": axis,
        "axis_name": axis_name,
        "edges": edges.astype(np.float64).tolist(),
        "buffer_distance": float(buffer_distance),
        "gray_count": int(np.sum(gray_mask)),
        "folds": folds,
        "fold_sizes": fold_sizes,
        "fold_positive_rates": fold_positive_rates,
        "overall_positive_rate": overall_positive_rate,
        "sample_count": int(len(values)),
    }


def build_spatial_cv_folds(
    coords: np.ndarray,
    *,
    n_folds: int = 5,
    buffer_distance: float = 0.0,
    labels: Optional[np.ndarray] = None,
    partition_coords: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Split coordinates into rectangular spatial CV folds."""

    coord_array = np.asarray(coords, dtype=np.float64)
    if coord_array.ndim != 2 or coord_array.shape[1] < 2:
        raise ValueError("Coordinates must be a 2D array with at least two columns.")
    coord_array = coord_array[:, :2]
    sample_count = int(len(coord_array))
    if sample_count == 0:
        return {
            "axis": 0,
            "axis_name": "x",
            "strategy": "x_quantile",
            "axis_source": "x_train_mineral_quantile",
            "fallback_reason": "-",
            "edges": [],
            "buffer_distance": float(buffer_distance),
            "gray_count": 0,
            "folds": [],
            "fold_sizes": [],
            "fold_positive_rates": [],
            "overall_positive_rate": None,
            "sample_count": 0,
            "fold_count": int(n_folds),
            "selected_fold_index": 0,
        }

    fold_count = max(2, min(int(n_folds), sample_count))
    buffer_distance = float(buffer_distance or 0.0)

    # Spatial CV policy: use quantile boundaries from the development mineral
    # systems along X so each fold contains a similar number of minerals. Fall
    # back to Y only when X has no variability.
    axis_source = "x_train_mineral_quantile"
    fallback_reason = "-"
    try:
        best_result = _build_axis_partition(
            coord_array,
            axis=0,
            n_folds=fold_count,
            buffer_distance=buffer_distance,
            labels=labels,
            partition_coords=partition_coords,
        )
    except ValueError:
        best_result = _build_axis_partition(
            coord_array,
            axis=1,
            n_folds=fold_count,
            buffer_distance=buffer_distance,
            labels=labels,
            partition_coords=partition_coords,
        )
        axis_source = "y_fallback"
        fallback_reason = "x_axis_has_no_variability"

    overall_positive_rate = best_result.get("overall_positive_rate")
    target_size = float(sample_count) / float(fold_count)
    selected_fold_index = 0
    best_fold_score = float("inf")
    for fold in best_result["folds"]:
        fold_index = int(fold["fold_index"])
        val_count = int(fold.get("val_count", 0))
        val_score = abs(float(val_count) - target_size)
        if val_count <= 0:
            val_score += 1e6
        if labels is not None and len(labels) > 0:
            val_indices = np.asarray(fold.get("val_indices") or [], dtype=np.int64)
            if len(val_indices) > 0:
                val_positive_rate = float(np.mean(np.asarray(labels)[val_indices] > 0))
                if overall_positive_rate is not None:
                    val_score += abs(val_positive_rate - float(overall_positive_rate))
        if val_score < best_fold_score:
            best_fold_score = val_score
            selected_fold_index = fold_index

    best_result["fold_count"] = int(fold_count)
    best_result["selected_fold_index"] = int(selected_fold_index)
    best_result["target_val_count"] = float(target_size)
    best_result["strategy"] = "x_quantile"
    best_result["axis_source"] = str(axis_source)
    best_result["fallback_reason"] = str(fallback_reason)
    return best_result
