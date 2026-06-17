"""Dataset construction utilities for model comparison workflows."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(CURRENT_DIR)
COMMON_DIR = os.path.join(CODE_ROOT, "common")
for path in (CODE_ROOT, COMMON_DIR):
    if path not in sys.path:
        sys.path.append(path)

from feature_channel_utils import infer_h5_channel_names, subset_samples_by_channels
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - 运行环境可能未安装 torch
    class _TorchFallback:
        @staticmethod
        def as_tensor(*args, **kwargs):
            raise ImportError("torch is required for dataset building")

        class float32:  # pragma: no cover - simple sentinel
            pass

        class long:  # pragma: no cover - simple sentinel
            pass

    class Dataset:  # type: ignore[override]
        pass

    class DataLoader:  # type: ignore[override]
        pass

    torch = _TorchFallback()
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit

from .data_splitter import MineralBasedSplitter
from .coordinate_utils import infer_grid_metadata_from_coordinates
from .metric_protocol import (
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_THRESHOLD_STEP,
    METRIC_PROTOCOL,
    PAF_SCOPE,
    THRESHOLD_RULE,
    THRESHOLD_STRATEGY,
    metric_protocol_fields,
)
from .spatial_mineral_splitter import build_spatial_cv_folds
from .spatial_region_splitter import build_coordinate_partition, normalize_region_split_config
from ..feature.patch_creator import PatchCreator


def _sanitize_std(std_values, eps: float = 1e-6) -> np.ndarray:
    std_values = np.asarray(std_values, dtype=np.float32)
    std_values = np.where(np.abs(std_values) < eps, 1.0, std_values)
    return std_values.astype(np.float32)


def _normalize_stats_dict(mean_values, std_values) -> Dict[str, list]:
    return {
        "mean": np.asarray(mean_values, dtype=np.float32).tolist(),
        "std": _sanitize_std(std_values).tolist(),
    }


def _prepare_normalization_tensors(normalization_stats):
    if normalization_stats is None:
        return None, None
    mean_values = normalization_stats.get("mean")
    std_values = normalization_stats.get("std")
    if mean_values is None or std_values is None:
        return None, None
    mean_tensor = torch.as_tensor(mean_values, dtype=torch.float32).view(-1, 1, 1)
    std_tensor = torch.as_tensor(_sanitize_std(std_values), dtype=torch.float32).view(-1, 1, 1)
    return mean_tensor, std_tensor


def _compute_sample_normalization_stats(samples: np.ndarray) -> Optional[Dict[str, list]]:
    if samples is None or len(samples) == 0:
        return None
    samples = np.asarray(samples, dtype=np.float32)
    mean_values = samples.mean(axis=(0, 2, 3))
    std_values = samples.std(axis=(0, 2, 3))
    return _normalize_stats_dict(mean_values, std_values)


def _extract_geo_metadata(handle: h5py.File) -> Dict[str, object]:
    metadata = {}
    if "metadata" not in handle:
        return metadata
    for key, value in handle["metadata"].attrs.items():
        if isinstance(value, np.generic):
            metadata[key] = value.item()
        else:
            metadata[key] = value
    return metadata


def _read_prebuilt_arrays(h5_path: str):
    with h5py.File(h5_path, "r") as handle:
        if "samples" in handle:
            samples = np.asarray(handle["samples"][:], dtype=np.float32)
            sample_kind = "samples"
        elif "vectors" in handle:
            vectors = np.asarray(handle["vectors"][:], dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors.reshape(-1, 1)
            if vectors.ndim == 2:
                samples = vectors[:, :, None, None]
            elif vectors.ndim == 3:
                samples = vectors[:, None, :, :]
            elif vectors.ndim == 4:
                samples = vectors
            else:
                raise ValueError(f"Unsupported vectors shape for prebuilt H5: {vectors.shape}")
            sample_kind = "coordinate_vectors"
        else:
            raise KeyError("Prebuilt H5 must contain a 'samples' or 'vectors' dataset.")
        if "labels" in handle:
            labels = np.asarray(handle["labels"][:]).reshape(-1).astype(np.int64)
        elif "label" in handle:
            labels = np.asarray(handle["label"][:]).reshape(-1).astype(np.int64)
        else:
            labels = None
        coordinates = np.asarray(handle["coordinates"][:]) if "coordinates" in handle else None
        metadata = _extract_geo_metadata(handle)
        if not metadata and coordinates is not None:
            metadata = infer_grid_metadata_from_coordinates(coordinates)
        metadata = dict(metadata)
        metadata["available_channel_names"] = infer_h5_channel_names(h5_path)
        metadata["sample_kind"] = sample_kind
    return samples, labels, coordinates, metadata


def _find_column_alias(frame: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for alias in aliases:
        matched = normalized.get(alias.strip().lower())
        if matched is not None:
            return matched
    return None


def _resolve_xy_columns(frame: pd.DataFrame) -> Tuple[str, str]:
    x_column = _find_column_alias(frame, ["x", "coord_x", "point_x", "east", "easting", "x坐标", "横坐标"])
    y_column = _find_column_alias(frame, ["y", "coord_y", "point_y", "north", "northing", "y坐标", "纵坐标"])
    if x_column is None or y_column is None:
        raise KeyError("Mineral file must contain X/Y coordinate columns.")
    return x_column, y_column


def _resolve_label_column(frame: pd.DataFrame) -> Optional[str]:
    return _find_column_alias(frame, ["label", "class", "target", "category", "ore_type", "类别"])


def _build_label_mapping(*frames: Optional[pd.DataFrame]) -> Dict[object, int]:
    raw_values = []
    for frame in frames:
        if frame is None or len(frame) == 0:
            continue
        label_column = _resolve_label_column(frame)
        if label_column is None:
            continue
        raw_values.extend(frame[label_column].dropna().tolist())
    if not raw_values:
        return {}
    unique_values = []
    for value in raw_values:
        if value not in unique_values:
            unique_values.append(value)
    return {value: index + 1 for index, value in enumerate(unique_values)}


def _read_patch_source_metadata(h5_path: str) -> Dict[str, object]:
    with h5py.File(h5_path, "r") as handle:
        if "metadata" not in handle:
            raise KeyError("H5 file is missing metadata required for spatial splitting.")
        return _extract_geo_metadata(handle)


def _patch_geo_coordinates(coords, metadata):
    from ..cnn.data_loader import _patch_indices_to_geo

    if coords is None:
        return None
    return _patch_indices_to_geo(np.asarray(coords), metadata or {})


def _use_reflect_padding(build_config: Optional[Dict[str, object]]) -> bool:
    return bool((build_config or {}).get("use_reflect_padding", False))


def _split_relative_indices(labels, coords, validation_split, n_blocks, metadata):
    from ..cnn.data_loader import _split_relative_indices as cnn_split_relative_indices

    label_frame = pd.DataFrame({"label": np.asarray(labels, dtype=np.int64)})
    stratify = len(np.unique(label_frame["label"].values)) > 1
    return cnn_split_relative_indices(
        labels_df=label_frame,
        coords=coords,
        validation_split=float(validation_split),
        stratify=stratify,
        n_blocks=int(n_blocks),
        random_state=42,
        metadata=metadata,
    )


def _estimate_spatial_cv_buffer_distance(metadata, patch_size: int) -> float:
    meta = metadata or {}
    x_scale = abs(float(meta.get("x_scale", 1.0) or 1.0))
    y_scale = abs(float(meta.get("y_scale", 1.0) or 1.0))
    window_width = int(meta.get("window_width", patch_size) or patch_size)
    window_height = int(meta.get("window_height", patch_size) or patch_size)
    window_extent = max(float(window_width) * x_scale, float(window_height) * y_scale)
    return max(window_extent / 2.0, 0.0)


def _split_spatial_relative_indices(
    labels,
    coords,
    metadata,
    patch_size: int,
    n_folds: int,
    buffer_distance: Optional[float] = None,
    partition_coords: Optional[np.ndarray] = None,
):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    coords = np.asarray(coords, dtype=np.float64)
    if len(labels) <= 1:
        train_indices = list(range(len(labels)))
        return train_indices, [], {
            "strategy": "degenerate_spatial_cv",
            "effective_strategy": "degenerate_spatial_cv",
            "train_count": int(len(train_indices)),
            "val_count": 0,
            "validation_split": 0.0,
            "fold_count": int(n_folds),
            "selected_fold_index": 0,
            "axis": "x",
            "axis_name": "x",
            "buffer_distance": 0.0,
            "gray_count": 0,
            "fold_counts": [0 for _ in range(max(int(n_folds), 1))],
            "spatial_cv_strategy": "x_quantile",
            "spatial_cv_axis_source": "x_train_mineral_quantile",
            "spatial_cv_fallback_reason": "-",
        }, {
            "axis": 0,
            "axis_name": "x",
            "strategy": "x_quantile",
            "axis_source": "x_train_mineral_quantile",
            "fallback_reason": "-",
            "edges": [],
            "buffer_distance": 0.0,
            "gray_count": 0,
            "folds": [],
            "fold_sizes": [],
            "fold_positive_rates": [],
            "overall_positive_rate": None,
            "sample_count": int(len(labels)),
            "fold_count": int(n_folds),
            "selected_fold_index": 0,
        }

    if buffer_distance is None:
        buffer_distance = _estimate_spatial_cv_buffer_distance(metadata, patch_size)
    else:
        buffer_distance = max(float(buffer_distance), 0.0)
    cv_info = build_spatial_cv_folds(
        coords,
        n_folds=int(n_folds),
        buffer_distance=float(buffer_distance),
        labels=labels,
        partition_coords=partition_coords,
    )
    selected_fold = int(cv_info.get("selected_fold_index", 0))
    folds = cv_info.get("folds") or []
    if selected_fold < 0 or selected_fold >= len(folds):
        selected_fold = 0
    selected_fold_info = folds[selected_fold] if folds else {"train_indices": [], "val_indices": []}
    train_indices = np.asarray(selected_fold_info.get("train_indices") or [], dtype=np.int64)
    val_indices = np.asarray(selected_fold_info.get("val_indices") or [], dtype=np.int64)

    split_info = {
        "strategy": "spatial_block_cv",
        "effective_strategy": "spatial_block_cv",
        "train_count": int(len(train_indices)),
        "val_count": int(len(val_indices)),
        "validation_split": float(1.0 / max(int(n_folds), 1)),
        "fold_count": int(n_folds),
        "selected_fold_index": int(selected_fold),
        "axis": cv_info.get("axis"),
        "axis_name": cv_info.get("axis_name"),
        "buffer_distance": float(cv_info.get("buffer_distance", buffer_distance) or 0.0),
        "gray_count": int(cv_info.get("gray_count", 0) or 0),
        "fold_counts": [int(fold.get("val_count", 0)) for fold in folds],
        "fold_positive_rates": [float(rate) for rate in (cv_info.get("fold_positive_rates") or [])],
        "overall_positive_rate": cv_info.get("overall_positive_rate"),
        "spatial_cv_strategy": str(cv_info.get("strategy", "x_quantile") or "x_quantile"),
        "spatial_cv_axis_source": str(cv_info.get("axis_source", "x_train_mineral_quantile") or "x_train_mineral_quantile"),
        "spatial_cv_fallback_reason": str(cv_info.get("fallback_reason", "-") or "-"),
    }
    return train_indices.tolist(), val_indices.tolist(), split_info, cv_info


def _split_frame_by_spatial_cv(frame: pd.DataFrame, cv_info: Dict[str, object]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    if frame is None or len(frame) == 0:
        empty = pd.DataFrame(columns=[] if frame is None else frame.columns)
        return empty.copy(), empty.copy(), {
            "train_count": 0,
            "val_count": 0,
            "fold_index": int(cv_info.get("selected_fold_index", 0) or 0),
        }

    x_column, y_column = _resolve_xy_columns(frame)
    coords = frame[[x_column, y_column]].to_numpy(dtype=np.float64)
    axis = int(cv_info.get("axis", 0) or 0)
    axis_values = coords[:, axis]
    edges = np.asarray(cv_info.get("edges") or [], dtype=np.float64)
    if len(edges) < 2:
        return frame.reset_index(drop=True), frame.iloc[0:0].copy(), {
            "train_count": int(len(frame)),
            "val_count": 0,
            "fold_index": int(cv_info.get("selected_fold_index", 0) or 0),
        }

    boundaries = edges[1:-1]
    block_ids = np.searchsorted(boundaries, axis_values, side="right")
    block_ids = np.clip(block_ids, 0, len(edges) - 2)
    selected_fold = int(cv_info.get("selected_fold_index", 0) or 0)
    val_mask = block_ids == selected_fold
    train_mask = block_ids != selected_fold
    train_frame = frame.loc[train_mask].reset_index(drop=True)
    val_frame = frame.loc[val_mask].reset_index(drop=True)
    return train_frame, val_frame, {
        "train_count": int(len(train_frame)),
        "val_count": int(len(val_frame)),
        "fold_index": int(selected_fold),
    }


def _build_split_info(split_info, *, sample_count: int, train_count: int, val_count: int, n_blocks: int, validation_split: float):
    info = dict(split_info or {})
    info["sample_count"] = int(sample_count)
    info["train_size"] = int(train_count)
    info["val_size"] = int(val_count)
    info["n_blocks"] = int(n_blocks)
    info["validation_split"] = float(validation_split)
    return info


def _attach_loader_metadata(loader: Optional[DataLoader], *, normalization_stats=None, split_info=None):
    if loader is None:
        return None
    loader.normalization_stats = normalization_stats
    loader.normalization_applied = bool(normalization_stats)
    loader.split_info = split_info
    return loader


def _should_drop_last_for_training(sample_count, batch_size) -> bool:
    batch_size = int(batch_size or 0)
    sample_count = int(sample_count or 0)
    return batch_size > 1 and sample_count > 1 and (sample_count % batch_size) == 1


def _distance_mask(sample_coords: np.ndarray, minerals_df: Optional[pd.DataFrame], buffer_radius: float) -> np.ndarray:
    if minerals_df is None or len(minerals_df) == 0:
        return np.zeros(len(sample_coords), dtype=bool)
    x_column, y_column = _resolve_xy_columns(minerals_df)
    mask = np.zeros(len(sample_coords), dtype=bool)
    mineral_coords = minerals_df[[x_column, y_column]].to_numpy(dtype=np.float64)
    for mineral_x, mineral_y in mineral_coords:
        distances = np.sqrt((sample_coords[:, 0] - mineral_x) ** 2 + (sample_coords[:, 1] - mineral_y) ** 2)
        mask |= distances <= float(buffer_radius)
    return mask


def _resolve_radius_column(frame: pd.DataFrame) -> Optional[str]:
    return _find_column_alias(
        frame,
        [
            "radius",
            "r",
            "buffer_radius",
            "negative_radius",
            "no_ore_radius",
            "影响半径",
            "半径",
            "无矿半径",
        ],
    )


def _negative_anchor_mask(sample_coords: np.ndarray, anchors_df: Optional[pd.DataFrame], default_radius: float) -> np.ndarray:
    if anchors_df is None or len(anchors_df) == 0:
        return np.zeros(len(sample_coords), dtype=bool)

    x_column, y_column = _resolve_xy_columns(anchors_df)
    radius_column = _resolve_radius_column(anchors_df)
    anchor_coords = anchors_df[[x_column, y_column]].to_numpy(dtype=np.float64)
    mask = np.zeros(len(sample_coords), dtype=bool)

    if radius_column is None:
        for anchor_x, anchor_y in anchor_coords:
            distances = np.sqrt((sample_coords[:, 0] - anchor_x) ** 2 + (sample_coords[:, 1] - anchor_y) ** 2)
            mask |= distances <= float(default_radius)
        return mask

    radii = pd.to_numeric(anchors_df[radius_column], errors="coerce").to_numpy(dtype=np.float64)
    for (anchor_x, anchor_y), radius in zip(anchor_coords, radii):
        effective_radius = float(default_radius) if not np.isfinite(radius) else float(radius)
        if effective_radius < 0:
            effective_radius = 0.0
        distances = np.sqrt((sample_coords[:, 0] - anchor_x) ** 2 + (sample_coords[:, 1] - anchor_y) ** 2)
        mask |= distances <= effective_radius
    return mask


def _label_array_from_minerals(
    sample_coords: np.ndarray,
    minerals_df: Optional[pd.DataFrame],
    *,
    metadata: Dict[str, object],
    patch_size: int,
    label_mapping: Dict[object, int],
) -> np.ndarray:
    labels = np.zeros(len(sample_coords), dtype=np.int64)
    if minerals_df is None or len(minerals_df) == 0:
        return labels
    half_width, half_height = _window_half_extent(metadata, patch_size)
    x_column, y_column = _resolve_xy_columns(minerals_df)
    label_column = _resolve_label_column(minerals_df)
    for _, row in minerals_df.iterrows():
        mineral_x = float(row[x_column])
        mineral_y = float(row[y_column])
        if label_column is None:
            label_value = 1
        else:
            label_value = int(label_mapping.get(row[label_column], 1))
        in_window_mask = (
            (np.abs(sample_coords[:, 0] - mineral_x) <= half_width)
            & (np.abs(sample_coords[:, 1] - mineral_y) <= half_height)
        )
        labels[in_window_mask] = np.maximum(labels[in_window_mask], label_value)
    return labels


def _label_array_from_window_bounds(
    sample_indices: np.ndarray,
    minerals_df: Optional[pd.DataFrame],
    *,
    metadata: Dict[str, object],
    patch_size: int,
    label_mapping: Dict[object, int],
) -> np.ndarray:
    labels = np.zeros(len(sample_indices), dtype=np.int64)
    if minerals_df is None or len(minerals_df) == 0:
        return labels

    indices = np.asarray(sample_indices, dtype=np.float64)
    if indices.ndim != 2 or indices.shape[1] < 2:
        return labels

    meta = metadata or {}
    try:
        x_min = float(meta["x_min"])
        x_max = float(meta["x_max"])
        y_min = float(meta["y_min"])
        y_max = float(meta["y_max"])
        nx = int(meta.get("nx", meta.get("image_width", meta.get("width", 0))))
        ny = int(meta.get("ny", meta.get("image_height", meta.get("height", 0))))
    except (KeyError, TypeError, ValueError):
        geo_coords = _patch_geo_coordinates(indices, meta)
        return _label_array_from_minerals(
            geo_coords if geo_coords is not None else indices,
            minerals_df,
            metadata=metadata,
            patch_size=patch_size,
            label_mapping=label_mapping,
        )

    if nx <= 1 or ny <= 1:
        geo_coords = _patch_geo_coordinates(indices, meta)
        return _label_array_from_minerals(
            geo_coords if geo_coords is not None else indices,
            minerals_df,
            metadata=metadata,
            patch_size=patch_size,
            label_mapping=label_mapping,
        )

    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span == 0 or y_span == 0:
        return labels

    x_step = x_span / max(nx - 1, 1)
    y_step = y_span / max(ny - 1, 1)
    window_width = int(meta.get("window_width", patch_size) or patch_size)
    window_height = int(meta.get("window_height", patch_size) or patch_size)
    rows = indices[:, 0]
    cols = indices[:, 1]
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

    x_column, y_column = _resolve_xy_columns(minerals_df)
    label_column = _resolve_label_column(minerals_df)
    for _, row in minerals_df.iterrows():
        mineral_x = float(row[x_column])
        mineral_y = float(row[y_column])
        if label_column is None:
            label_value = 1
        else:
            label_value = int(label_mapping.get(row[label_column], 1))
        in_window_mask = (
            (mineral_x >= x_left)
            & (mineral_x < x_right)
            & (mineral_y <= y_top)
            & (mineral_y > y_bottom)
        )
        labels[in_window_mask] = np.maximum(labels[in_window_mask], label_value)
    return labels


def _nearest_mineral_distance(sample_coords: np.ndarray, minerals_df: Optional[pd.DataFrame]) -> np.ndarray:
    if minerals_df is None or len(minerals_df) == 0:
        return np.full(len(sample_coords), np.inf, dtype=np.float64)
    x_column, y_column = _resolve_xy_columns(minerals_df)
    mineral_coords = minerals_df[[x_column, y_column]].to_numpy(dtype=np.float64)
    nearest_distances = np.full(len(sample_coords), np.inf, dtype=np.float64)
    for mineral_x, mineral_y in mineral_coords:
        distances = np.sqrt((sample_coords[:, 0] - mineral_x) ** 2 + (sample_coords[:, 1] - mineral_y) ** 2)
        nearest_distances = np.minimum(nearest_distances, distances)
    return nearest_distances


def _resolve_negative_sampling_scheme(build_config, buffer_radius: float) -> Tuple[str, float, Optional[float]]:
    mode = str(build_config.get("negative_sampling_mode") or "default").strip().lower()
    if mode in {"true", "1", "yes", "enabled", "far", "far_distance", "distance", "distance_confirmed"}:
        mode = "far_distance"
    else:
        mode = "default"

    multiplier = build_config.get("negative_distance_multiplier", 2.0)
    if multiplier in (None, ""):
        multiplier = 2.0
    multiplier = float(multiplier)
    if multiplier <= 0:
        raise ValueError("negative_distance_multiplier must be greater than 0.")

    negative_distance_radius = float(buffer_radius) * multiplier if mode == "far_distance" else None
    return mode, multiplier, negative_distance_radius


def _negative_candidate_mask(
    sample_coords: np.ndarray,
    minerals_df: Optional[pd.DataFrame],
    negative_sampling_mode: str,
    negative_distance_radius: Optional[float],
) -> np.ndarray:
    if negative_sampling_mode != "far_distance" or negative_distance_radius is None:
        return np.ones(len(sample_coords), dtype=bool)
    nearest_distances = _nearest_mineral_distance(sample_coords, minerals_df)
    return nearest_distances >= float(negative_distance_radius)


def _resolve_buffer_policy(build_config: Dict[str, object], *, default_radius: float) -> Dict[str, object]:
    policy = dict(build_config.get("buffer_policy") or {})
    enabled = bool(policy.get("enable", build_config.get("buffer_exclusion_enabled", False)))
    remove_unlabeled_only = bool(policy.get("remove_unlabeled_only", True))
    radius_value = policy.get("radius_m", policy.get("radius", default_radius))
    radius = float(radius_value if radius_value not in (None, "") else default_radius)
    if radius < 0:
        radius = 0.0
    return {
        "enable": enabled,
        "radius_m": radius,
        "remove_unlabeled_only": remove_unlabeled_only,
    }


def _window_coverage_mask(
    sample_coords: np.ndarray,
    minerals_df: Optional[pd.DataFrame],
    metadata: Dict[str, object],
    patch_size: int,
) -> np.ndarray:
    radius = _estimate_spatial_cv_buffer_distance(metadata, patch_size)
    if radius <= 0:
        return np.zeros(len(sample_coords), dtype=bool)
    return _distance_mask(sample_coords, minerals_df, radius)


def _window_half_extent(metadata: Dict[str, object], patch_size: int) -> Tuple[float, float]:
    meta = metadata or {}
    x_scale = abs(float(meta.get("x_scale", 1.0) or 1.0))
    y_scale = abs(float(meta.get("y_scale", 1.0) or 1.0))
    window_width = int(meta.get("window_width", patch_size) or patch_size)
    window_height = int(meta.get("window_height", patch_size) or patch_size)
    half_width = max((float(window_width) * x_scale) / 2.0, 0.0)
    half_height = max((float(window_height) * y_scale) / 2.0, 0.0)
    return half_width, half_height


def _resolve_region_partition(sample_coords: np.ndarray, build_config: Dict[str, object]) -> Optional[Dict[str, object]]:
    region_config = normalize_region_split_config(build_config.get("spatial_region_split"))
    if region_config is None:
        return None
    return build_coordinate_partition(
        sample_coords,
        region_config["train_region"],
        region_config["test_region"],
        buffer_distance=float(region_config["buffer_distance"]),
    )


def _empty_index_array() -> np.ndarray:
    return np.asarray([], dtype=np.int64)


def _filter_indices_by_mask(indices: np.ndarray, allowed_mask: Optional[np.ndarray]) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if allowed_mask is None or len(indices) == 0:
        return indices
    return indices[np.asarray(allowed_mask[indices], dtype=bool)].astype(np.int64)


def _region_summary_fields(
    region_partition: Optional[Dict[str, object]],
    *,
    train_no_ore_count: int = 0,
    test_no_ore_count: int = 0,
) -> Dict[str, object]:
    if not region_partition:
        return {
            "spatial_region_active": False,
            "spatial_region_buffer_distance": 0.0,
            "spatial_region_train_sample_count": 0,
            "spatial_region_test_sample_count": 0,
            "spatial_region_gray_sample_count": 0,
            "spatial_region_outside_sample_count": 0,
            "spatial_region_overlap_sample_count": 0,
            "spatial_region_train_no_ore_sample_count": int(train_no_ore_count),
            "spatial_region_test_no_ore_sample_count": int(test_no_ore_count),
        }

    return {
        "spatial_region_active": True,
        "spatial_region_train_bounds": dict(region_partition.get("train_region") or {}),
        "spatial_region_test_bounds": dict(region_partition.get("test_region") or {}),
        "spatial_region_buffer_distance": float(region_partition.get("buffer_distance", 0.0) or 0.0),
        "spatial_region_train_sample_count": int(region_partition.get("train_count", 0) or 0),
        "spatial_region_test_sample_count": int(region_partition.get("test_count", 0) or 0),
        "spatial_region_gray_sample_count": int(region_partition.get("gray_count", 0) or 0),
        "spatial_region_outside_sample_count": int(region_partition.get("outside_count", 0) or 0),
        "spatial_region_overlap_sample_count": int(region_partition.get("overlap_count", 0) or 0),
        "spatial_region_train_no_ore_sample_count": int(train_no_ore_count),
        "spatial_region_test_no_ore_sample_count": int(test_no_ore_count),
    }


def _sample_background_indices(background_indices: np.ndarray, target_count: int, rng: np.random.Generator) -> np.ndarray:
    background_indices = np.asarray(background_indices, dtype=np.int64)
    if target_count <= 0 or len(background_indices) == 0:
        return np.asarray([], dtype=np.int64)
    count = min(int(target_count), len(background_indices))
    return np.sort(rng.choice(background_indices, size=count, replace=False).astype(np.int64))


def _split_indices_by_spatial_clusters(
    coords: np.ndarray,
    *,
    train_ratio: float,
    n_clusters: int,
    buffer_distance: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2 or len(coords) == 0:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty

    coords = coords[:, :2]
    unique_count = len(np.unique(coords, axis=0))
    cluster_count = max(1, min(int(n_clusters), len(coords), unique_count))
    ratio = float(train_ratio)
    if not np.isfinite(ratio):
        ratio = 0.7
    ratio = float(min(max(ratio, 0.1), 0.9))
    rng = np.random.default_rng(int(random_state))

    if cluster_count == 1 or len(coords) < 3:
        order = rng.permutation(len(coords))
        train_count = max(1, int(round(len(coords) * ratio)))
        train_count = min(train_count, len(coords) - 1) if len(coords) > 1 else len(coords)
        return (
            np.sort(order[:train_count]).astype(np.int64),
            np.sort(order[train_count:]).astype(np.int64),
        )

    kmeans = KMeans(n_clusters=cluster_count, random_state=int(random_state), n_init=10)
    cluster_ids = kmeans.fit_predict(coords)
    distances = kmeans.transform(coords)
    if distances.shape[1] > 1:
        sorted_distances = np.sort(distances, axis=1)
        margin = sorted_distances[:, 1] - sorted_distances[:, 0]
    else:
        margin = np.full(len(coords), np.inf, dtype=np.float64)
    gray_mask = margin <= float(max(buffer_distance, 0.0))

    train_rows = []
    test_rows = []
    for cluster_id in sorted(np.unique(cluster_ids)):
        cluster_rows = np.where((cluster_ids == cluster_id) & (~gray_mask))[0]
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

    return (
        np.asarray(sorted(set(train_rows)), dtype=np.int64),
        np.asarray(sorted(set(test_rows)), dtype=np.int64),
    )


def _split_background_indices_for_pu(
    background_indices: np.ndarray,
    split_coords: Optional[np.ndarray],
    build_config: Dict[str, object],
    *,
    split_mode: str,
    default_train_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split all unlabeled/background candidates without down-sampling them."""

    background_indices = np.asarray(background_indices, dtype=np.int64)
    if len(background_indices) == 0:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty

    spatial_config = dict(build_config.get("spatial_cluster_split") or {})
    train_ratio = float(spatial_config.get("train_ratio", default_train_ratio) or default_train_ratio)
    random_state = int(spatial_config.get("random_state", 42) or 42)

    if split_mode == "spatial_cluster" and split_coords is not None:
        rel_train, rel_test = _split_indices_by_spatial_clusters(
            np.asarray(split_coords)[background_indices],
            train_ratio=train_ratio,
            n_clusters=int(spatial_config.get("n_clusters", 10) or 10),
            buffer_distance=float(spatial_config.get("cv_buffer_distance", 0.0) or 0.0),
            random_state=random_state,
        )
        return (
            np.sort(background_indices[rel_train]).astype(np.int64),
            np.sort(background_indices[rel_test]).astype(np.int64),
        )

    ratio = float(train_ratio)
    if not np.isfinite(ratio):
        ratio = float(default_train_ratio)
    ratio = float(min(max(ratio, 0.1), 0.9))
    rng = np.random.default_rng(random_state)
    order = rng.permutation(background_indices)
    train_count = max(1, int(round(len(order) * ratio)))
    train_count = min(train_count, len(order) - 1) if len(order) > 1 else len(order)
    return (
        np.sort(order[:train_count]).astype(np.int64),
        np.sort(order[train_count:]).astype(np.int64),
    )


def _normalize_sampling_fraction(value: object, default: float = 1.0) -> float:
    if value is None:
        return float(default)

    fraction = float(value)
    if fraction <= 0:
        return 0.0
    if fraction > 1.0:
        fraction = fraction / 100.0
    return float(min(fraction, 1.0))


def _sample_index_pool(
    indices: np.ndarray,
    labels: Optional[np.ndarray],
    sampling_percentage: float,
    *,
    rng: Optional[np.random.Generator] = None,
    protected_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    indices = np.asarray(indices, dtype=np.int64)
    if labels is not None:
        labels = np.asarray(labels, dtype=np.int64)
    protected_indices = None if protected_indices is None else np.asarray(protected_indices, dtype=np.int64)

    if len(indices) == 0:
        return indices, labels

    order = np.argsort(indices)
    indices = indices[order]
    if labels is not None:
        labels = labels[order]

    protected_mask = np.zeros(len(indices), dtype=bool)
    if protected_indices is not None and len(protected_indices) > 0:
        protected_mask = np.isin(indices, protected_indices)

    protected_pool = indices[protected_mask]
    protected_labels = labels[protected_mask] if labels is not None else None
    remaining_indices = indices[~protected_mask]
    remaining_labels = labels[~protected_mask] if labels is not None else None

    fraction = _normalize_sampling_fraction(sampling_percentage)
    if fraction >= 1.0:
        return indices, labels

    remaining_target = 0
    if len(remaining_indices) > 0:
        remaining_target = max(1, int(round(len(remaining_indices) * fraction)))
    sampled_remaining = None

    if remaining_target >= len(remaining_indices):
        sampled_remaining = np.arange(len(remaining_indices), dtype=np.int64)
    elif remaining_target > 0 and remaining_labels is not None and len(remaining_indices) > 0 and len(np.unique(remaining_labels)) > 1:
        try:
            remaining_fraction = remaining_target / float(len(remaining_indices))
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=remaining_fraction, random_state=42)
            sampled_remaining, _ = next(splitter.split(np.zeros(len(remaining_indices)), remaining_labels))
            sampled_remaining = np.sort(np.asarray(sampled_remaining, dtype=np.int64))
        except ValueError:
            sampled_remaining = None

    if sampled_remaining is None and remaining_target > 0:
        rng = rng or np.random.default_rng(42)
        sampled_remaining = np.sort(rng.choice(len(remaining_indices), size=remaining_target, replace=False).astype(np.int64))

    if sampled_remaining is None:
        sampled_remaining = np.asarray([], dtype=np.int64)

    sampled_indices = np.concatenate([protected_pool, remaining_indices[sampled_remaining]]).astype(np.int64)
    sampled_labels = None
    if labels is not None:
        sampled_labels = np.concatenate([
            protected_labels if protected_labels is not None else np.asarray([], dtype=np.int64),
            remaining_labels[sampled_remaining] if remaining_labels is not None else np.asarray([], dtype=np.int64),
        ]).astype(np.int64)
    order = np.argsort(sampled_indices)
    sampled_indices = sampled_indices[order]
    if sampled_labels is not None:
        sampled_labels = sampled_labels[order]
    return sampled_indices, sampled_labels


def _assign_indices_to_nearest_minerals(
    sample_coords: Optional[np.ndarray],
    indices: np.ndarray,
    minerals_df: Optional[pd.DataFrame],
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    assigned_ids = np.full(len(indices), -1, dtype=np.int64)
    if sample_coords is None or minerals_df is None or len(minerals_df) == 0 or len(indices) == 0:
        return assigned_ids

    coords = np.asarray(sample_coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        return assigned_ids

    x_column, y_column = _resolve_xy_columns(minerals_df)
    mineral_coords = minerals_df[[x_column, y_column]].to_numpy(dtype=np.float64)
    if len(mineral_coords) == 0:
        return assigned_ids

    selected_coords = coords[indices][:, :2]
    for pos, coord in enumerate(selected_coords):
        distances = np.sqrt(np.sum((mineral_coords - coord) ** 2, axis=1))
        assigned_ids[pos] = int(np.argmin(distances))
    return assigned_ids


def _sample_dev_indices_by_mineral(
    indices: np.ndarray,
    labels: np.ndarray,
    sampling_percentage: float,
    *,
    sample_coords: Optional[np.ndarray],
    minerals_df: Optional[pd.DataFrame],
    rng: Optional[np.random.Generator] = None,
    protected_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample the development pool like the training module.

    Positive samples are sampled within each mineral group so every retained
    mineral still contributes windows. Unlabeled/negative samples are sampled
    randomly by ratio, while protected hard negatives stay in the pool.
    """

    indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    protected_indices = None if protected_indices is None else np.asarray(protected_indices, dtype=np.int64)

    if len(indices) == 0:
        return indices, labels

    order = np.argsort(indices)
    indices = indices[order]
    labels = labels[order]

    fraction = _normalize_sampling_fraction(sampling_percentage)
    if fraction >= 1.0:
        return indices, labels

    rng = rng or np.random.default_rng(42)
    selected_parts = []

    positive_positions = np.where(labels > 0)[0]
    if len(positive_positions) > 0:
        positive_indices = indices[positive_positions]
        mineral_ids = _assign_indices_to_nearest_minerals(sample_coords, positive_indices, minerals_df)
        grouped_mask = mineral_ids >= 0
        grouped_indices = positive_indices[grouped_mask]
        grouped_ids = mineral_ids[grouped_mask]

        for mineral_id in np.unique(grouped_ids):
            group_indices = grouped_indices[grouped_ids == mineral_id]
            keep_count = min(len(group_indices), max(1, int(round(len(group_indices) * fraction))))
            selected_parts.append(rng.choice(group_indices, keep_count, replace=False).astype(np.int64))

        ungrouped_indices = positive_indices[~grouped_mask]
        if len(ungrouped_indices) > 0:
            keep_count = min(len(ungrouped_indices), max(1, int(round(len(ungrouped_indices) * fraction))))
            selected_parts.append(rng.choice(ungrouped_indices, keep_count, replace=False).astype(np.int64))

    negative_positions = np.where(labels <= 0)[0]
    if len(negative_positions) > 0:
        negative_indices = indices[negative_positions]
        protected_negative = np.asarray([], dtype=np.int64)
        remaining_negative = negative_indices
        if protected_indices is not None and len(protected_indices) > 0:
            protected_mask = np.isin(negative_indices, protected_indices)
            protected_negative = negative_indices[protected_mask]
            remaining_negative = negative_indices[~protected_mask]
            if len(protected_negative) > 0:
                selected_parts.append(protected_negative.astype(np.int64))
        if len(remaining_negative) > 0:
            keep_count = min(len(remaining_negative), max(1, int(round(len(remaining_negative) * fraction))))
            selected_parts.append(rng.choice(remaining_negative, keep_count, replace=False).astype(np.int64))

    if not selected_parts:
        return indices, labels

    selected_indices = np.sort(np.unique(np.concatenate(selected_parts).astype(np.int64)))
    selected_labels = labels[np.searchsorted(indices, selected_indices)]
    return selected_indices, selected_labels


def _balance_index_pool(
    indices: np.ndarray,
    labels: np.ndarray,
    balance_ratio: float,
    *,
    rng: Optional[np.random.Generator] = None,
    protected_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    protected_indices = None if protected_indices is None else np.asarray(protected_indices, dtype=np.int64)
    if len(indices) == 0:
        return indices, labels

    order = np.argsort(indices)
    indices = indices[order]
    labels = labels[order]

    if balance_ratio is None:
        return indices, labels

    ratio = float(balance_ratio)
    if ratio <= 0:
        raise ValueError("balance_ratio must be greater than 0.")

    positive_mask = labels > 0
    negative_mask = labels == 0
    positive_indices = indices[positive_mask]
    negative_indices = indices[negative_mask]
    positive_labels = labels[positive_mask]
    negative_labels = labels[negative_mask]

    if len(positive_indices) == 0 or len(negative_indices) == 0:
        return indices, labels

    protected_negative_mask = np.zeros(len(negative_indices), dtype=bool)
    if protected_indices is not None and len(protected_indices) > 0:
        protected_negative_mask = np.isin(negative_indices, protected_indices)
    protected_negative_indices = negative_indices[protected_negative_mask]
    protected_negative_labels = negative_labels[protected_negative_mask]
    remaining_negative_indices = negative_indices[~protected_negative_mask]
    remaining_negative_labels = negative_labels[~protected_negative_mask]

    target_negatives = int(round(len(positive_indices) / ratio))
    target_negatives = max(target_negatives, 0)
    if target_negatives <= len(protected_negative_indices):
        sampled_negative_indices = protected_negative_indices
        sampled_negative_labels = protected_negative_labels
    else:
        target_remaining_negatives = min(target_negatives - len(protected_negative_indices), len(remaining_negative_indices))
        if target_remaining_negatives >= len(remaining_negative_indices):
            sampled_negative_indices = np.concatenate([protected_negative_indices, remaining_negative_indices]).astype(np.int64)
            sampled_negative_labels = np.concatenate([protected_negative_labels, remaining_negative_labels]).astype(np.int64)
        else:
            rng = rng or np.random.default_rng(42)
            sampled_positions = np.sort(
                rng.choice(len(remaining_negative_indices), size=target_remaining_negatives, replace=False).astype(np.int64)
            )
            sampled_negative_indices = np.concatenate([protected_negative_indices, remaining_negative_indices[sampled_positions]]).astype(np.int64)
            sampled_negative_labels = np.concatenate([protected_negative_labels, remaining_negative_labels[sampled_positions]]).astype(np.int64)

    balanced_indices = np.concatenate([positive_indices, sampled_negative_indices]).astype(np.int64)
    balanced_labels = np.concatenate([positive_labels, sampled_negative_labels]).astype(np.int64)
    order = np.argsort(balanced_indices)
    return balanced_indices[order], balanced_labels[order]


def _max_label(*arrays: np.ndarray) -> int:
    values = [int(np.max(array)) for array in arrays if array is not None and len(array) > 0]
    return max(values) if values else 0


def _assign_dev_minerals_to_split(dev_minerals_df: pd.DataFrame, positive_coords: np.ndarray, positive_indices: np.ndarray, val_indices: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if dev_minerals_df is None or len(dev_minerals_df) == 0:
        empty = pd.DataFrame(columns=["x", "y"])
        return empty.copy(), empty.copy()
    if positive_coords is None or len(positive_coords) == 0 or len(positive_indices) == 0:
        return dev_minerals_df.reset_index(drop=True), dev_minerals_df.iloc[0:0].copy()

    x_column, y_column = _resolve_xy_columns(dev_minerals_df)
    val_index_set = set(np.asarray(val_indices, dtype=np.int64).tolist())
    train_rows = []
    val_rows = []
    mineral_coords = dev_minerals_df[[x_column, y_column]].to_numpy(dtype=np.float64)
    for row_index, mineral_coord in enumerate(mineral_coords):
        distances = np.sqrt(np.sum((positive_coords - mineral_coord) ** 2, axis=1))
        nearest_pos = int(np.argmin(distances))
        target_index = int(positive_indices[nearest_pos])
        if target_index in val_index_set:
            val_rows.append(row_index)
        else:
            train_rows.append(row_index)
    return (
        dev_minerals_df.iloc[train_rows].reset_index(drop=True),
        dev_minerals_df.iloc[val_rows].reset_index(drop=True),
    )


@dataclass
class ComparisonDatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_data_array: Tuple[np.ndarray, np.ndarray]
    val_data_array: Tuple[np.ndarray, np.ndarray]
    test_data_array: Tuple[np.ndarray, np.ndarray]
    dev_data_array: Optional[Tuple[np.ndarray, np.ndarray]]
    dataset_meta: Dict[str, object]
    dataset_summary: Dict[str, object]
    build_config: Dict[str, object]
    h5_path: str
    train_minerals_df: pd.DataFrame
    val_minerals_df: pd.DataFrame
    test_minerals_df: pd.DataFrame
    spatial_cv_splits: Optional[Dict[str, object]] = None
    sample_coordinates: Optional[np.ndarray] = None
    train_indices: Optional[np.ndarray] = None
    val_indices: Optional[np.ndarray] = None
    test_indices: Optional[np.ndarray] = None
    test_area_positions: Optional[np.ndarray] = None


class H5ClassificationDataset(Dataset):
    def __init__(self, samples: np.ndarray, labels: np.ndarray, normalization_stats=None):
        self.samples = np.asarray(samples, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.normalization_stats = normalization_stats
        self.normalization_mean, self.normalization_std = _prepare_normalization_tensors(normalization_stats)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = torch.as_tensor(self.samples[index], dtype=torch.float32)
        if self.normalization_mean is not None and self.normalization_std is not None:
            sample = (sample - self.normalization_mean) / self.normalization_std
        return sample, int(self.labels[index] > 0)


class ArrayClassificationDataset(H5ClassificationDataset):
    pass


class ComparisonDataBuilder:
    """Build train/val/test bundles for model comparison."""

    def __init__(self, negative_ratio: int = 1):
        self.negative_ratio = int(negative_ratio)

    def detect_h5_mode(self, h5_path: str) -> str:
        with h5py.File(h5_path, "r") as handle:
            if "samples" in handle:
                return "prebuilt"
            if "vectors" in handle and "coordinates" in handle:
                return "prebuilt"
            if "fused_features" in handle:
                return "raw_features"
        raise ValueError("Unsupported H5 structure for model comparison.")

    def _resolve_sampling_controls(self, build_config) -> Tuple[float, Optional[float]]:
        sampling_percentage = _normalize_sampling_fraction(build_config.get("sampling_percentage", 1.0))
        balance_ratio = build_config.get("balance_ratio")
        if balance_ratio in (None, ""):
            return sampling_percentage, None

        balance_ratio = float(balance_ratio)
        if balance_ratio <= 0:
            raise ValueError("balance_ratio must be greater than 0.")
        return sampling_percentage, balance_ratio

    def release_bundle(self, bundle) -> None:
        del bundle

    def build_bundle(
        self,
        h5_path,
        train_minerals,
        val_minerals,
        test_minerals,
        h5_mode,
        build_config,
        progress_callback=None,
        no_ore_minerals=None,
    ) -> ComparisonDatasetBundle:
        if progress_callback is not None:
            progress_callback(f"正在以 {h5_mode} 模式构建数据集...")

        if h5_mode == "prebuilt":
            return self._build_prebuilt_bundle(
                h5_path,
                train_minerals,
                val_minerals,
                test_minerals,
                build_config,
                progress_callback=progress_callback,
                no_ore_minerals=no_ore_minerals,
            )
        if h5_mode == "raw_features":
            return self._build_raw_bundle(
                h5_path,
                train_minerals,
                val_minerals,
                test_minerals,
                build_config,
                no_ore_minerals=no_ore_minerals,
            )
        raise ValueError(f"Unsupported H5 mode: {h5_mode}")

    def _create_loader(self, samples, labels, *, batch_size, shuffle, normalization_stats=None, split_info=None):
        dataset = ArrayClassificationDataset(samples, labels, normalization_stats=normalization_stats)
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            num_workers=0,
            drop_last=_should_drop_last_for_training(len(dataset), batch_size) if shuffle else False,
        )
        return _attach_loader_metadata(loader, normalization_stats=normalization_stats, split_info=split_info)

    def _finalize_bundle(
        self,
        *,
        h5_path,
        h5_mode,
        build_config,
        train_samples,
        train_labels,
        val_samples,
        val_labels,
        test_samples,
        test_labels,
        dataset_meta,
        dataset_summary,
        train_minerals_df,
        val_minerals_df,
        test_minerals_df,
        dev_samples=None,
        dev_labels=None,
        spatial_cv_splits=None,
        sample_coordinates=None,
        train_indices=None,
        val_indices=None,
        test_indices=None,
    ) -> ComparisonDatasetBundle:
        normalization_stats = _compute_sample_normalization_stats(train_samples)
        split_info = dataset_summary.get("split_info")
        batch_size = int(build_config.get("batch_size", 32))

        train_loader = self._create_loader(
            train_samples,
            train_labels,
            batch_size=batch_size,
            shuffle=True,
            normalization_stats=normalization_stats,
            split_info=split_info,
        )
        val_loader = self._create_loader(
            val_samples,
            val_labels,
            batch_size=batch_size,
            shuffle=False,
            normalization_stats=normalization_stats,
            split_info=split_info,
        )
        test_loader = self._create_loader(
            test_samples,
            test_labels,
            batch_size=batch_size,
            shuffle=False,
            normalization_stats=normalization_stats,
            split_info=split_info,
        )

        dataset_summary = dict(dataset_summary)
        dataset_summary["normalization_stats"] = normalization_stats
        dataset_summary["channel_normalization_enabled"] = bool(normalization_stats)
        dataset_meta = dict(dataset_meta)
        dataset_meta.setdefault("input_kind", dataset_summary.get("input_kind", "vector"))
        dataset_meta.setdefault("label_mode", dataset_summary.get("label_mode", "binary"))
        dataset_summary.setdefault("input_kind", dataset_meta["input_kind"])
        dataset_summary.setdefault("label_mode", dataset_meta["label_mode"])
        evaluation_protocol = dict(dataset_summary.get("evaluation_protocol") or {})
        evaluation_protocol.update(
            metric_protocol_fields(
                threshold_step=float(evaluation_protocol.get("threshold_step", DEFAULT_THRESHOLD_STEP) or DEFAULT_THRESHOLD_STEP),
                distance_threshold=float(
                    evaluation_protocol.get("distance_threshold", DEFAULT_DISTANCE_THRESHOLD)
                    or DEFAULT_DISTANCE_THRESHOLD
                ),
                threshold_strategy=str(evaluation_protocol.get("threshold_strategy") or THRESHOLD_STRATEGY),
            )
        )
        dataset_summary["evaluation_protocol"] = evaluation_protocol
        dataset_summary["metric_protocol"] = METRIC_PROTOCOL
        dataset_summary["threshold_strategy"] = evaluation_protocol.get("threshold_strategy", THRESHOLD_STRATEGY)
        dataset_summary["paf_scope"] = PAF_SCOPE
        dataset_summary["threshold_rule"] = THRESHOLD_RULE

        sample_coordinates_array = (
            None
            if sample_coordinates is None
            else np.asarray(sample_coordinates, dtype=np.float64)
        )
        train_indices_array = None if train_indices is None else np.asarray(train_indices, dtype=np.int64)
        val_indices_array = None if val_indices is None else np.asarray(val_indices, dtype=np.int64)
        test_indices_array = None if test_indices is None else np.asarray(test_indices, dtype=np.int64)
        test_area_positions = None
        if sample_coordinates_array is not None and test_indices_array is not None:
            valid_test_indices = test_indices_array[
                (test_indices_array >= 0) & (test_indices_array < len(sample_coordinates_array))
            ]
            test_area_positions = sample_coordinates_array[valid_test_indices]
            dataset_summary["independent_test_area_count"] = int(len(test_area_positions))
        else:
            dataset_summary["independent_test_area_count"] = int(len(test_labels))

        return ComparisonDatasetBundle(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train_data_array=(train_samples, train_labels),
            val_data_array=(val_samples, val_labels),
            test_data_array=(test_samples, test_labels),
            dev_data_array=None
            if dev_samples is None or dev_labels is None
            else (np.asarray(dev_samples, dtype=np.float32), np.asarray(dev_labels, dtype=np.int64)),
            dataset_meta=dataset_meta,
            dataset_summary=dataset_summary,
            build_config=dict(build_config),
            h5_path=h5_path,
            train_minerals_df=train_minerals_df.reset_index(drop=True),
            val_minerals_df=val_minerals_df.reset_index(drop=True),
            test_minerals_df=test_minerals_df.reset_index(drop=True),
            spatial_cv_splits=spatial_cv_splits,
            sample_coordinates=sample_coordinates_array,
            train_indices=train_indices_array,
            val_indices=val_indices_array,
            test_indices=test_indices_array,
            test_area_positions=test_area_positions,
        )

    def _subset_arrays(self, samples, labels, indices):
        indices = np.asarray(indices, dtype=np.int64)
        binary_labels = (np.asarray(labels[indices], dtype=np.int64) > 0).astype(np.int64)
        return np.asarray(samples[indices], dtype=np.float32), binary_labels

    def _build_prebuilt_bundle(self, h5_path, train_minerals, val_minerals, test_minerals, build_config=None, progress_callback=None, no_ore_minerals=None):
        build_config = build_config or {}
        samples, labels, coordinates, metadata = _read_prebuilt_arrays(h5_path)
        if coordinates is None:
            raise KeyError("Prebuilt H5 must contain coordinates for spatial validation.")
        sample_kind = str(metadata.get("sample_kind", "samples") or "samples")
        requested_patch_size = int(build_config.get("patch_size", samples.shape[-1] if samples.ndim >= 3 else 1))
        requested_patch_stride = int(build_config.get("patch_stride", requested_patch_size))
        vector_mode = sample_kind == "coordinate_vectors"
        coordinate_grid_mode = vector_mode and requested_patch_size > 1

        if coordinate_grid_mode:
            use_reflect_padding = _use_reflect_padding(build_config)
            patch_creator = PatchCreator(h5_path)
            try:
                samples, coordinates = patch_creator.generate_patches(
                    (requested_patch_size, requested_patch_size),
                    (requested_patch_stride, requested_patch_stride),
                    enable_padding=bool(use_reflect_padding),
                    padding_mode="reflect",
                )
            finally:
                patch_creator.close()
            samples = np.asarray(samples, dtype=np.float32)
            coordinates = np.asarray(coordinates)
            if labels is not None and len(labels) != len(samples):
                labels = None
            sample_kind = "coordinate_grid"
            vector_mode = False
        samples, patch_metadata = subset_samples_by_channels(samples, metadata, build_config.get("selected_channels"))
        metadata = dict(patch_metadata)

        effective_patch_size = 1 if vector_mode else requested_patch_size
        effective_patch_stride = 1 if vector_mode else requested_patch_stride

        sampling_percentage, balance_ratio = self._resolve_sampling_controls(build_config)
        requested_buffer_radius = float(build_config.get("buffer_radius", 500.0))
        buffer_radius = requested_buffer_radius
        buffer_policy = _resolve_buffer_policy(build_config, default_radius=requested_buffer_radius)
        evaluation_protocol = dict(build_config.get("evaluation_protocol") or {})
        negative_sampling_mode, negative_distance_multiplier, negative_distance_radius = _resolve_negative_sampling_scheme(
            build_config,
            buffer_radius,
        )
        dev_minerals = pd.concat([train_minerals, val_minerals], ignore_index=True)
        no_ore_minerals = None if no_ore_minerals is None or len(no_ore_minerals) == 0 else no_ore_minerals.reset_index(drop=True)
        no_ore_active = no_ore_minerals is not None and len(no_ore_minerals) > 0
        split_mode = str(build_config.get("split_mode", "single") or "single").strip().lower()

        patch_metadata = dict(metadata)
        if "window_width" not in patch_metadata:
            patch_metadata["window_width"] = int(effective_patch_size)
        if "window_height" not in patch_metadata:
            patch_metadata["window_height"] = int(effective_patch_size)
        patch_metadata["sample_kind"] = sample_kind
        patch_metadata["reflect_padding"] = bool(coordinate_grid_mode and _use_reflect_padding(build_config))
        patch_metadata["coordinates_are_centers"] = bool(coordinate_grid_mode and _use_reflect_padding(build_config))
        if coordinate_grid_mode:
            patch_metadata["grid_layout"] = "coordinate_table"
            patch_metadata["coordinate_order"] = "row_col"
            if progress_callback is not None:
                progress_callback(
                    f"检测到坐标向量型 H5，正在按窗口大小 {requested_patch_size}、步长 {requested_patch_stride} 重建内部窗口。"
                )

        if vector_mode or coordinate_grid_mode:
            grid_step = max(
                abs(float(patch_metadata.get("x_scale", 1.0) or 1.0)),
                abs(float(patch_metadata.get("y_scale", 1.0) or 1.0)),
            )
            adaptive_radius = max(grid_step * 5.0, grid_step)
            buffer_radius = min(requested_buffer_radius, adaptive_radius)
            if progress_callback is not None and buffer_radius < requested_buffer_radius:
                progress_callback(
                    f"检测到坐标向量型 H5，缓冲半径已根据网格间距从 {requested_buffer_radius:g} 调整为 {buffer_radius:g}。"
                )
            negative_sampling_mode, negative_distance_multiplier, negative_distance_radius = _resolve_negative_sampling_scheme(
                build_config,
                buffer_radius,
            )

        split_coords = _patch_geo_coordinates(coordinates, patch_metadata)
        test_window_overlap_mask = _window_coverage_mask(
            split_coords,
            test_minerals,
            patch_metadata,
            int(effective_patch_size),
        )
        test_window_overlap_radius = _estimate_spatial_cv_buffer_distance(
            patch_metadata,
            int(effective_patch_size),
        )
        test_window_overlap_removed_count = int(np.sum(test_window_overlap_mask))
        region_partition = _resolve_region_partition(split_coords, build_config)
        region_active = region_partition is not None
        train_region_mask = None if not region_active else np.asarray(region_partition["train_mask"], dtype=bool)
        test_region_mask = None if not region_active else np.asarray(region_partition["test_mask"], dtype=bool)
        label_mapping = {}
        label_source = "h5"
        conflict_sample_count = 0
        no_ore_point_count = int(len(no_ore_minerals)) if no_ore_minerals is not None else 0
        no_ore_conflict_count = 0
        no_ore_sample_count = 0
        labels_were_missing = labels is None
        no_ore_mask = _negative_anchor_mask(split_coords, no_ore_minerals, buffer_radius) if no_ore_active else np.zeros(len(samples), dtype=bool)
        hard_negative_indices = np.where(no_ore_mask)[0].astype(np.int64) if no_ore_active else _empty_index_array()
        train_hard_negative_indices = (
            np.where(no_ore_mask & train_region_mask)[0].astype(np.int64)
            if no_ore_active and region_active
            else hard_negative_indices
        )
        test_hard_negative_indices = (
            np.where(no_ore_mask & test_region_mask)[0].astype(np.int64)
            if no_ore_active and region_active
            else _empty_index_array()
        )
        train_region_no_ore_count = int(len(train_hard_negative_indices))
        test_region_no_ore_count = int(len(test_hard_negative_indices))
        if labels is not None:
            no_ore_conflict_count = int(np.sum(no_ore_mask & (labels > 0)))
            if no_ore_active:
                labels = np.asarray(labels, dtype=np.int64).copy()
                labels[no_ore_mask] = 0
                no_ore_sample_count = int(np.sum(no_ore_mask))
        if labels_were_missing:
            if progress_callback is not None:
                progress_callback("预构建 H5 中没有标签，正在根据矿点文件推导标签...")
            label_source = "mineral_files"
            label_mapping = _build_label_mapping(dev_minerals, test_minerals)
            dev_positive_labels = _label_array_from_minerals(
                split_coords,
                dev_minerals,
                metadata=patch_metadata,
                patch_size=int(effective_patch_size),
                label_mapping=label_mapping,
            )
            test_positive_labels = _label_array_from_minerals(
                split_coords,
                test_minerals,
                metadata=patch_metadata,
                patch_size=int(effective_patch_size),
                label_mapping=label_mapping,
            )
            # Match the training module: overlapping train/test windows are
            # reserved for the test side instead of being dropped by radius.
            conflict_mask = (dev_positive_labels > 0) & (test_positive_labels > 0)
            conflict_sample_count = int(np.sum(conflict_mask))
            test_window_overlap_removed_count = int(conflict_sample_count)
            dev_positive_labels[conflict_mask] = 0

            if no_ore_active:
                no_ore_conflict_mask = no_ore_mask & ((dev_positive_labels > 0) | (test_positive_labels > 0))
                no_ore_conflict_count = int(np.sum(no_ore_conflict_mask))
                dev_positive_labels[no_ore_mask] = 0
                test_positive_labels[no_ore_mask] = 0
                no_ore_sample_count = int(np.sum(no_ore_mask))

            dev_positive_indices = np.where(dev_positive_labels > 0)[0].astype(np.int64)
            test_positive_indices = np.where(test_positive_labels > 0)[0].astype(np.int64)
            if region_active:
                dev_positive_indices = _filter_indices_by_mask(dev_positive_indices, train_region_mask)
                test_positive_indices = _filter_indices_by_mask(test_positive_indices, test_region_mask)
            if len(dev_positive_indices) == 0:
                raise ValueError("No positive development samples were derived from mineral files.")
            if len(test_positive_indices) == 0:
                raise ValueError("No positive test samples were derived from mineral files.")

            combined_minerals = pd.concat([dev_minerals, test_minerals], ignore_index=True)
            far_negative_mask = _negative_candidate_mask(
                split_coords,
                combined_minerals,
                negative_sampling_mode,
                negative_distance_radius,
            )
            background_indices = np.where(
                (dev_positive_labels == 0)
                & (test_positive_labels == 0)
                & far_negative_mask
                & (~no_ore_mask)
            )[0].astype(np.int64)

            buffer_policy_removed_count = 0
            buffer_policy_applied = False
            if buffer_policy["enable"] and buffer_policy["radius_m"] > 0:
                combined_mineral_buffer_mask = _distance_mask(
                    split_coords,
                    combined_minerals,
                    float(buffer_policy["radius_m"]),
                )
                buffer_candidate_mask = (
                    combined_mineral_buffer_mask & (dev_positive_labels == 0) & (test_positive_labels == 0)
                    if buffer_policy["remove_unlabeled_only"]
                    else combined_mineral_buffer_mask
                )
                before_count = int(len(background_indices))
                background_indices = background_indices[~buffer_candidate_mask[background_indices]]
                buffer_policy_removed_count = max(before_count - int(len(background_indices)), 0)
                buffer_policy_applied = bool(buffer_policy_removed_count > 0)
            else:
                buffer_policy_removed_count = 0
                buffer_policy_applied = False
            if region_active:
                dev_background_indices = _filter_indices_by_mask(background_indices, train_region_mask)
                test_background_indices = _filter_indices_by_mask(background_indices, test_region_mask)
                dev_negative_indices = np.concatenate([train_hard_negative_indices, dev_background_indices]).astype(np.int64)
                test_negative_indices = np.concatenate([test_hard_negative_indices, test_background_indices]).astype(np.int64)
            else:
                total_positive = len(dev_positive_indices) + len(test_positive_indices)
                default_train_ratio = (
                    float(len(dev_positive_indices)) / float(total_positive)
                    if total_positive > 0
                    else 0.7
                )
                dev_background_indices, test_background_indices = _split_background_indices_for_pu(
                    background_indices,
                    split_coords,
                    build_config,
                    split_mode=split_mode,
                    default_train_ratio=default_train_ratio,
                )
                dev_negative_indices = np.concatenate([hard_negative_indices, dev_background_indices]).astype(np.int64)
                test_negative_indices = test_background_indices.astype(np.int64)
            dev_negative_indices = np.sort(np.unique(dev_negative_indices))
            test_negative_indices = np.sort(np.unique(test_negative_indices))
            if progress_callback is not None:
                progress_callback(
                    "预构建采样: "
                    f"训练/验证正样本={len(dev_positive_indices)}, 训练/验证负样本={len(dev_negative_indices)}, "
                    f"测试正样本={len(test_positive_indices)}, 测试负样本={len(test_negative_indices)}"
                )

            dev_indices = np.sort(np.concatenate([dev_positive_indices, dev_negative_indices])).astype(np.int64)
            test_indices = np.sort(np.concatenate([test_positive_indices, test_negative_indices])).astype(np.int64)
            labels = np.zeros(len(samples), dtype=np.int64)
            labels[dev_positive_indices] = dev_positive_labels[dev_positive_indices]
            labels[test_positive_indices] = np.maximum(
                labels[test_positive_indices],
                test_positive_labels[test_positive_indices],
            )
            derived_positive_count = int(np.sum(labels > 0))
            derived_negative_count = int(np.sum(labels == 0))
            if progress_callback is not None:
                progress_callback(
                    f"已从矿点文件推导标签: 正样本={derived_positive_count}, 负样本={derived_negative_count}"
                )
                if derived_positive_count == 0:
                    progress_callback(
                        "警告: 未能从矿点文件推导出正样本，请检查坐标原点和单位是否一致。"
                    )
        else:
            if region_active:
                dev_indices = np.where(train_region_mask)[0].astype(np.int64)
                test_indices = np.where(test_region_mask)[0].astype(np.int64)
                if np.any(test_window_overlap_mask):
                    dev_indices = dev_indices[~test_window_overlap_mask[dev_indices]]
                if no_ore_active:
                    no_ore_sample_count = int(np.sum(no_ore_mask & (train_region_mask | test_region_mask)))
            else:
                splitter = MineralBasedSplitter(buffer_radius=float(build_config.get("buffer_radius", 500.0)))
                dev_indices, _, test_indices = splitter.split_by_minerals(
                    h5_path,
                    dev_minerals,
                    dev_minerals.iloc[0:0].copy(),
                    test_minerals,
                )
                dev_indices = np.asarray(sorted(set(dev_indices) - set(test_indices)), dtype=np.int64)
                test_indices = np.asarray(sorted(set(test_indices)), dtype=np.int64)
                if no_ore_active:
                    if len(hard_negative_indices) > 0:
                        dev_indices = np.asarray(sorted(set(dev_indices).union(hard_negative_indices.tolist())), dtype=np.int64)
                        test_indices = np.asarray(sorted(set(test_indices) - set(hard_negative_indices.tolist())), dtype=np.int64)
                        no_ore_sample_count = int(len(hard_negative_indices))
                if np.any(test_window_overlap_mask):
                    dev_indices = dev_indices[~test_window_overlap_mask[dev_indices]]

            buffer_policy_removed_count = 0
            buffer_policy_applied = False
            if buffer_policy["enable"] and buffer_policy["radius_m"] > 0 and len(dev_indices) > 0:
                combined_minerals = pd.concat([dev_minerals, test_minerals], ignore_index=True)
                combined_mineral_buffer_mask = _distance_mask(
                    split_coords,
                    combined_minerals,
                    float(buffer_policy["radius_m"]),
                )
                if buffer_policy["remove_unlabeled_only"]:
                    remove_mask = combined_mineral_buffer_mask & (labels == 0)
                else:
                    remove_mask = combined_mineral_buffer_mask
                before_count = int(len(dev_indices))
                dev_indices = dev_indices[~remove_mask[dev_indices]]
                buffer_policy_removed_count = max(before_count - int(len(dev_indices)), 0)
                buffer_policy_applied = bool(buffer_policy_removed_count > 0)
            if len(dev_indices) == 0:
                raise ValueError("Development split is empty after mineral-based masking.")
            if len(test_indices) == 0:
                raise ValueError("Test split is empty after mineral-based masking.")

        if labels_were_missing:
            # Fields defined in missing-label branch for consistent reporting.
            buffer_policy_removed_count = int(locals().get("buffer_policy_removed_count", 0))
            buffer_policy_applied = bool(locals().get("buffer_policy_applied", False))

        negative_sampling_applied = bool(labels_were_missing)
        if sampling_percentage < 1.0:
            dev_indices, _ = _sample_dev_indices_by_mineral(
                dev_indices,
                labels[dev_indices],
                sampling_percentage,
                sample_coords=split_coords,
                minerals_df=dev_minerals,
                rng=np.random.default_rng(int(build_config.get("spatial_cluster_split", {}).get("random_state", 42) or 42)),
                protected_indices=train_hard_negative_indices if no_ore_active else None,
            )
        dev_samples, dev_labels = self._subset_arrays(samples, labels, dev_indices)
        spatial_cv_splits = None
        if split_mode == "spatial_cluster":
            dev_mineral_x, dev_mineral_y = _resolve_xy_columns(dev_minerals)
            train_rel, val_rel, split_info, spatial_cv_splits = _split_spatial_relative_indices(
                labels=labels[dev_indices],
                coords=None if split_coords is None else split_coords[dev_indices],
                metadata=patch_metadata,
                patch_size=effective_patch_size,
                n_folds=int(build_config.get("spatial_cluster_split", {}).get("cv_folds", 5) or 5),
                buffer_distance=build_config.get("spatial_cluster_split", {}).get("cv_buffer_distance"),
                partition_coords=dev_minerals[[dev_mineral_x, dev_mineral_y]].to_numpy(dtype=np.float64),
            )
            train_indices = dev_indices[np.asarray(train_rel, dtype=np.int64)]
            val_indices = dev_indices[np.asarray(val_rel, dtype=np.int64)]
            if len(train_indices) == 0 or len(val_indices) == 0:
                raise ValueError("Spatial validation failed to produce non-empty train/val splits.")
            train_samples, train_labels = self._subset_arrays(samples, labels, train_indices)
            val_samples, val_labels = self._subset_arrays(samples, labels, val_indices)
            test_samples, test_labels = self._subset_arrays(samples, labels, test_indices)
            train_minerals_df, val_minerals_df, cv_mineral_summary = _split_frame_by_spatial_cv(dev_minerals, spatial_cv_splits)
        else:
            dev_coords = None if split_coords is None else split_coords[dev_indices]
            train_rel, val_rel, split_info = _split_relative_indices(
                labels=labels[dev_indices],
                coords=dev_coords,
                validation_split=float(build_config.get("val_ratio", 0.2)),
                n_blocks=int(build_config.get("n_blocks", 3)),
                metadata=patch_metadata,
            )
            train_indices = dev_indices[np.asarray(train_rel, dtype=np.int64)]
            val_indices = dev_indices[np.asarray(val_rel, dtype=np.int64)]
            if len(train_indices) == 0 or len(val_indices) == 0:
                raise ValueError("Spatial validation failed to produce non-empty train/val splits.")
            train_samples, train_labels = self._subset_arrays(samples, labels, train_indices)
            val_samples, val_labels = self._subset_arrays(samples, labels, val_indices)
            test_samples, test_labels = self._subset_arrays(samples, labels, test_indices)
            train_minerals_df = dev_minerals.reset_index(drop=True)
            val_minerals_df = dev_minerals.iloc[0:0].copy()
            cv_mineral_summary = {
                "train_count": int(len(train_minerals_df)),
                "val_count": int(len(val_minerals_df)),
                "fold_index": 0,
            }

        split_info = _build_split_info(
            split_info,
            sample_count=len(dev_indices),
            train_count=len(train_indices),
            val_count=len(val_indices),
            n_blocks=int(build_config.get("spatial_cluster_split", {}).get("cv_folds", 5) if split_mode == "spatial_cluster" else build_config.get("n_blocks", 3)),
            validation_split=float(1.0 / max(int(build_config.get("spatial_cluster_split", {}).get("cv_folds", 5) if split_mode == "spatial_cluster" else build_config.get("n_blocks", 3)), 1)) if split_mode == "spatial_cluster" else float(build_config.get("val_ratio", 0.2)),
        )
        split_info["sampling_percentage"] = sampling_percentage
        split_info["balance_ratio"] = balance_ratio
        split_info["buffer_requested"] = bool(buffer_policy["enable"])
        split_info["buffer_applied"] = bool(buffer_policy_applied)
        split_info["buffer_removed_count"] = int(buffer_policy_removed_count)
        split_info["buffer_distance"] = float(buffer_policy["radius_m"])
        split_info.setdefault("fallback_reason", "-")
        if split_mode == "spatial_cluster":
            split_info["spatial_cluster_active"] = True
            split_info["spatial_cluster_n_clusters"] = int(build_config.get("spatial_cluster_split", {}).get("n_clusters", 0) or 0)
            split_info["spatial_cluster_train_ratio"] = float(build_config.get("spatial_cluster_split", {}).get("train_ratio", 0.7) or 0.7)
            split_info["spatial_cluster_cv_folds"] = int(build_config.get("spatial_cluster_split", {}).get("cv_folds", 5) or 5)
            split_info["spatial_cluster_random_state"] = int(build_config.get("spatial_cluster_split", {}).get("random_state", 42) or 42)
            split_info["spatial_cluster_mineral_train_count"] = int(cv_mineral_summary.get("train_count", 0))
            split_info["spatial_cluster_mineral_val_count"] = int(cv_mineral_summary.get("val_count", 0))
            split_info["spatial_cluster_selected_fold"] = int(cv_mineral_summary.get("fold_index", 0))
        else:
            split_info["spatial_cluster_active"] = False
        dataset_meta = {
            "input_channels": int(samples.shape[1]),
            "image_size": int(samples.shape[-1]),
            "num_classes": 2,
            "input_kind": "vector" if vector_mode else "window",
            "label_mode": "binary",
        }
        dataset_summary = {
            "h5_mode": "prebuilt",
            "sample_kind": sample_kind,
            "input_kind": "vector" if vector_mode else "window",
            "label_mode": "binary",
            "input_channels": dataset_meta["input_channels"],
            "image_size": dataset_meta["image_size"],
            "num_classes": dataset_meta["num_classes"],
            "patch_size": int(samples.shape[-1]) if not vector_mode else 1,
            "patch_stride": effective_patch_stride,
            "reflect_padding": bool(patch_metadata.get("reflect_padding", False)),
            "selected_channel_indices": patch_metadata.get("selected_channel_indices", list(range(int(samples.shape[1])))),
            "selected_channel_names": patch_metadata.get("selected_channel_names", patch_metadata.get("available_channel_names", [])),
            "buffer_radius": buffer_radius,
            "buffer_radius_requested": requested_buffer_radius,
            "validation_split": float(build_config.get("val_ratio", 0.2)),
            "n_blocks": int(build_config.get("n_blocks", 3)),
            "sampling_percentage": sampling_percentage,
            "balance_ratio": balance_ratio,
            "augmentation_enabled": bool(build_config.get("augmentation_enabled", False)),
            "augmentation_noise_std": float(build_config.get("augmentation_noise_std", 0.01) or 0.01),
            "negative_ratio": None,
            "supervised_train_ratio": 1.0,
            "sample_construction_mode": "pu_full_unlabeled",
            "negative_sampling_mode": negative_sampling_mode,
            "negative_sampling_applied": negative_sampling_applied,
            "negative_distance_multiplier": negative_distance_multiplier,
            "negative_distance_radius": negative_distance_radius,
            "buffer_policy": dict(buffer_policy),
            "buffer_exclusion_enabled": bool(buffer_policy["enable"]),
            "buffer_exclusion_applied": bool(buffer_policy_applied),
            "buffer_exclusion_removed_count": int(buffer_policy_removed_count),
            "buffer_exclusion_distance": float(buffer_policy["radius_m"]),
            "test_window_overlap_exclusion_enabled": True,
            "test_window_overlap_exclusion_applied": bool(test_window_overlap_removed_count > 0),
            "test_window_overlap_exclusion_removed_count": int(test_window_overlap_removed_count),
            "test_window_overlap_exclusion_distance": float(test_window_overlap_radius),
            "evaluation_protocol": dict(evaluation_protocol),
            "split_mode": split_mode,
            "spatial_cluster_active": bool(split_mode == "spatial_cluster"),
            "spatial_cluster_n_clusters": int(build_config.get("spatial_cluster_split", {}).get("n_clusters", 0) or 0),
            "spatial_cluster_train_ratio": float(build_config.get("spatial_cluster_split", {}).get("train_ratio", 0.7) or 0.7),
            "spatial_cluster_cv_folds": int(build_config.get("spatial_cluster_split", {}).get("cv_folds", 5) or 5),
            "spatial_cluster_random_state": int(build_config.get("spatial_cluster_split", {}).get("random_state", 42) or 42),
            "spatial_cluster_train_mineral_count": int(len(dev_minerals)),
            "spatial_cluster_test_mineral_count": int(len(test_minerals)),
            "spatial_cv_enabled": bool(spatial_cv_splits),
            "spatial_cv_fold_count": int(spatial_cv_splits.get("fold_count", 0) if spatial_cv_splits else 0),
            "spatial_cv_axis": spatial_cv_splits.get("axis_name", None) if spatial_cv_splits else None,
            "spatial_cv_strategy": spatial_cv_splits.get("strategy", "x_quantile") if spatial_cv_splits else None,
            "spatial_cv_axis_source": spatial_cv_splits.get("axis_source", "x_train_mineral_quantile") if spatial_cv_splits else None,
            "spatial_cv_buffer_distance": float(spatial_cv_splits.get("buffer_distance", 0.0) if spatial_cv_splits else 0.0),
            "spatial_cv_fallback_reason": spatial_cv_splits.get("fallback_reason", "-") if spatial_cv_splits else "-",
            "spatial_cv_selected_fold": int(spatial_cv_splits.get("selected_fold_index", 0) if spatial_cv_splits else 0),
            "no_ore_active": bool(no_ore_active),
            "no_ore_point_count": int(no_ore_point_count),
            "no_ore_sample_count": int(no_ore_sample_count),
            "no_ore_conflict_count": int(no_ore_conflict_count),
            "spatial_validation_enabled": True,
            "split_info": split_info,
            "train_sample_count": int(len(train_samples)),
            "val_sample_count": int(len(val_samples)),
            "test_sample_count": int(len(test_samples)),
            "train_positive_count": int(np.sum(train_labels > 0)),
            "val_positive_count": int(np.sum(val_labels > 0)),
            "test_positive_count": int(np.sum(test_labels > 0)),
            "train_negative_count": int(np.sum(train_labels == 0)),
            "val_negative_count": int(np.sum(val_labels == 0)),
            "test_negative_count": int(np.sum(test_labels == 0)),
            "train_mineral_count": int(len(train_minerals_df)),
            "val_mineral_count": int(len(val_minerals_df)),
            "test_mineral_count": int(len(test_minerals)),
            "dev_mineral_count": int(len(dev_minerals)),
            "dev_pool_sample_count": int(len(dev_indices)),
            "label_mapping": {str(key): int(value) for key, value in label_mapping.items()},
            "label_source": label_source,
            "class_distribution": {
                int(label): int(np.sum((labels > 0).astype(np.int64) == label))
                for label in np.unique((labels > 0).astype(np.int64))
            },
            "conflict_sample_count": int(conflict_sample_count),
        }
        dataset_summary.update(
            _region_summary_fields(
                region_partition,
                train_no_ore_count=train_region_no_ore_count,
                test_no_ore_count=test_region_no_ore_count,
            )
        )
        return self._finalize_bundle(
            h5_path=h5_path,
            h5_mode="prebuilt",
            build_config=build_config,
            train_samples=train_samples,
            train_labels=train_labels,
            val_samples=val_samples,
            val_labels=val_labels,
            test_samples=test_samples,
            test_labels=test_labels,
            dataset_meta=dataset_meta,
            dataset_summary=dataset_summary,
            train_minerals_df=train_minerals_df,
            val_minerals_df=val_minerals_df,
            test_minerals_df=test_minerals,
            dev_samples=dev_samples,
            dev_labels=dev_labels,
            spatial_cv_splits=spatial_cv_splits,
            sample_coordinates=split_coords,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        )

    def _build_raw_bundle(self, h5_path, train_minerals, val_minerals, test_minerals, build_config=None, no_ore_minerals=None):
        build_config = build_config or {}
        dev_minerals = pd.concat([train_minerals, val_minerals], ignore_index=True)
        split_mode = str(build_config.get("split_mode", "single") or "single").strip().lower()
        patch_size = int(build_config.get("patch_size", 64))
        patch_stride = int(build_config.get("patch_stride", patch_size))
        buffer_radius = float(build_config.get("buffer_radius", 500.0))
        buffer_policy = _resolve_buffer_policy(build_config, default_radius=buffer_radius)
        evaluation_protocol = dict(build_config.get("evaluation_protocol") or {})
        n_blocks = int(build_config.get("n_blocks", 3))
        validation_split = float(build_config.get("val_ratio", 0.2))
        sampling_percentage, balance_ratio = self._resolve_sampling_controls(build_config)
        negative_sampling_mode, negative_distance_multiplier, negative_distance_radius = _resolve_negative_sampling_scheme(
            build_config,
            buffer_radius,
        )
        no_ore_minerals = None if no_ore_minerals is None or len(no_ore_minerals) == 0 else no_ore_minerals.reset_index(drop=True)
        no_ore_active = no_ore_minerals is not None and len(no_ore_minerals) > 0
        use_reflect_padding = _use_reflect_padding(build_config)

        patch_creator = PatchCreator(h5_path)
        try:
            samples, coordinates = patch_creator.generate_patches(
                patch_size,
                patch_stride,
                enable_padding=bool(use_reflect_padding),
                padding_mode="reflect",
            )
        finally:
            patch_creator.source_h5_file.close()

        samples = np.asarray(samples, dtype=np.float32)
        coordinates = np.asarray(coordinates)
        metadata = _read_patch_source_metadata(h5_path)
        metadata["available_channel_names"] = infer_h5_channel_names(h5_path)
        split_metadata = dict(metadata)
        split_metadata["window_width"] = patch_size
        split_metadata["window_height"] = patch_size
        split_metadata["patch_stride"] = patch_stride
        split_metadata["reflect_padding"] = bool(use_reflect_padding)
        split_metadata["coordinates_are_centers"] = bool(use_reflect_padding)
        split_metadata["coordinate_order"] = "row_col"
        samples, split_metadata = subset_samples_by_channels(samples, split_metadata, build_config.get("selected_channels"))
        split_coords = _patch_geo_coordinates(coordinates, split_metadata)
        if split_coords is None:
            raise ValueError("Unable to derive geographic coordinates for raw-feature patches.")
        test_window_overlap_mask = _window_coverage_mask(
            split_coords,
            test_minerals,
            split_metadata,
            int(patch_size),
        )
        test_window_overlap_radius = _estimate_spatial_cv_buffer_distance(
            split_metadata,
            int(patch_size),
        )
        test_window_overlap_removed_count = int(np.sum(test_window_overlap_mask))
        region_partition = _resolve_region_partition(split_coords, build_config)
        region_active = region_partition is not None
        train_region_mask = None if not region_active else np.asarray(region_partition["train_mask"], dtype=bool)
        test_region_mask = None if not region_active else np.asarray(region_partition["test_mask"], dtype=bool)

        label_mapping = _build_label_mapping(dev_minerals, test_minerals)
        dev_positive_labels = _label_array_from_window_bounds(
            coordinates,
            dev_minerals,
            metadata=split_metadata,
            patch_size=int(patch_size),
            label_mapping=label_mapping,
        )
        test_positive_labels = _label_array_from_window_bounds(
            coordinates,
            test_minerals,
            metadata=split_metadata,
            patch_size=int(patch_size),
            label_mapping=label_mapping,
        )
        no_ore_mask = _negative_anchor_mask(split_coords, no_ore_minerals, buffer_radius) if no_ore_active else np.zeros(len(samples), dtype=bool)
        no_ore_point_count = int(len(no_ore_minerals)) if no_ore_minerals is not None else 0
        no_ore_conflict_count = 0
        no_ore_sample_count = 0
        train_forced_negative_indices = (
            np.where(no_ore_mask & train_region_mask)[0].astype(np.int64)
            if no_ore_active and region_active
            else np.where(no_ore_mask)[0].astype(np.int64) if no_ore_active else _empty_index_array()
        )
        test_forced_negative_indices = (
            np.where(no_ore_mask & test_region_mask)[0].astype(np.int64)
            if no_ore_active and region_active
            else _empty_index_array()
        )

        # Match the training module: overlapping train/test windows are
        # reserved for the test side instead of being dropped by radius.
        conflict_mask = (dev_positive_labels > 0) & (test_positive_labels > 0)
        test_window_overlap_removed_count = int(np.sum(conflict_mask))
        dev_positive_labels[conflict_mask] = 0

        if no_ore_active:
            no_ore_conflict_mask = no_ore_mask & ((dev_positive_labels > 0) | (test_positive_labels > 0))
            no_ore_conflict_count = int(np.sum(no_ore_conflict_mask))
            dev_positive_labels[no_ore_mask] = 0
            test_positive_labels[no_ore_mask] = 0
            no_ore_sample_count = int(np.sum(no_ore_mask))

        dev_positive_indices = np.where(dev_positive_labels > 0)[0].astype(np.int64)
        test_positive_indices = np.where(test_positive_labels > 0)[0].astype(np.int64)
        if region_active:
            dev_positive_indices = _filter_indices_by_mask(dev_positive_indices, train_region_mask)
            test_positive_indices = _filter_indices_by_mask(test_positive_indices, test_region_mask)
        positive_repartition_fallback = False
        positive_repartition_reason = None
        if len(dev_positive_indices) == 0 or len(test_positive_indices) == 0:
            merged_positive_labels = np.where(dev_positive_labels > 0, dev_positive_labels, test_positive_labels)
            merged_positive_indices = np.where(merged_positive_labels > 0)[0].astype(np.int64)
            if len(merged_positive_indices) < 2:
                raise ValueError(
                    "No positive development patches were generated. "
                    "Please check mineral coordinates, window size and coordinate units."
                )

            rng_fallback = np.random.default_rng(42)
            shuffled = np.array(merged_positive_indices, copy=True)
            rng_fallback.shuffle(shuffled)

            split_mode_hint = str(build_config.get("split_mode", "single") or "single").strip().lower()
            if split_mode_hint == "spatial_cluster":
                train_ratio = float(build_config.get("spatial_cluster_split", {}).get("train_ratio", 0.7) or 0.7)
            else:
                mineral_total = len(dev_minerals) + len(test_minerals)
                train_ratio = (float(len(dev_minerals)) / float(mineral_total)) if mineral_total > 0 else 0.8
            train_ratio = min(max(train_ratio, 0.1), 0.9)

            split_point = int(round(len(shuffled) * train_ratio))
            split_point = min(max(split_point, 1), len(shuffled) - 1)
            fallback_dev_indices = np.sort(shuffled[:split_point]).astype(np.int64)
            fallback_test_indices = np.sort(shuffled[split_point:]).astype(np.int64)

            dev_positive_labels[merged_positive_indices] = 0
            test_positive_labels[merged_positive_indices] = 0
            dev_positive_labels[fallback_dev_indices] = merged_positive_labels[fallback_dev_indices]
            test_positive_labels[fallback_test_indices] = merged_positive_labels[fallback_test_indices]

            dev_positive_indices = fallback_dev_indices
            test_positive_indices = fallback_test_indices
            positive_repartition_fallback = True
            positive_repartition_reason = "dev_or_test_positive_empty"

        if len(dev_positive_indices) == 0 or len(test_positive_indices) == 0:
            raise ValueError(
                "No positive development/test patches were generated after fallback split. "
                "Please check mineral coordinates, window size and coordinate units."
            )

        combined_minerals = pd.concat([dev_minerals, test_minerals], ignore_index=True)
        far_negative_mask = _negative_candidate_mask(
            split_coords,
            combined_minerals,
            negative_sampling_mode,
            negative_distance_radius,
        )
        background_indices = np.where(
            (dev_positive_labels == 0)
            & (test_positive_labels == 0)
            & far_negative_mask
            & (~no_ore_mask)
        )[0].astype(np.int64)

        buffer_policy_removed_count = 0
        buffer_policy_applied = False
        if buffer_policy["enable"] and buffer_policy["radius_m"] > 0:
            combined_mineral_buffer_mask = _distance_mask(
                split_coords,
                combined_minerals,
                float(buffer_policy["radius_m"]),
            )
            buffer_candidate_mask = (
                combined_mineral_buffer_mask & (dev_positive_labels == 0) & (test_positive_labels == 0)
                if buffer_policy["remove_unlabeled_only"]
                else combined_mineral_buffer_mask
            )
            before_count = int(len(background_indices))
            background_indices = background_indices[~buffer_candidate_mask[background_indices]]
            buffer_policy_removed_count = max(before_count - int(len(background_indices)), 0)
            buffer_policy_applied = bool(buffer_policy_removed_count > 0)
        if region_active:
            dev_background_indices = _filter_indices_by_mask(background_indices, train_region_mask)
            test_background_indices = _filter_indices_by_mask(background_indices, test_region_mask)
            dev_negative_indices = np.concatenate([
                train_forced_negative_indices,
                dev_background_indices,
            ]).astype(np.int64)
            test_negative_indices = np.concatenate([
                test_forced_negative_indices,
                test_background_indices,
            ]).astype(np.int64)
        else:
            total_positive = len(dev_positive_indices) + len(test_positive_indices)
            default_train_ratio = (
                float(len(dev_positive_indices)) / float(total_positive)
                if total_positive > 0
                else 0.7
            )
            dev_background_indices, test_background_indices = _split_background_indices_for_pu(
                background_indices,
                split_coords,
                build_config,
                split_mode=split_mode,
                default_train_ratio=default_train_ratio,
            )
            dev_negative_indices = np.concatenate([
                train_forced_negative_indices,
                dev_background_indices,
            ]).astype(np.int64)
            test_negative_indices = test_background_indices.astype(np.int64)
        dev_negative_indices = np.sort(np.unique(dev_negative_indices))
        test_negative_indices = np.sort(np.unique(test_negative_indices))

        dev_candidate_indices = np.sort(np.concatenate([dev_positive_indices, dev_negative_indices])).astype(np.int64)
        test_indices = np.sort(np.concatenate([test_positive_indices, test_negative_indices])).astype(np.int64)
        combined_labels = np.zeros(len(samples), dtype=np.int64)
        combined_labels[dev_positive_indices] = dev_positive_labels[dev_positive_indices]
        combined_labels[test_positive_indices] = test_positive_labels[test_positive_indices]

        if sampling_percentage < 1.0:
            dev_candidate_indices, _ = _sample_dev_indices_by_mineral(
                dev_candidate_indices,
                combined_labels[dev_candidate_indices],
                sampling_percentage,
                sample_coords=split_coords,
                minerals_df=dev_minerals,
                rng=np.random.default_rng(int(build_config.get("spatial_cluster_split", {}).get("random_state", 42) or 42)),
                protected_indices=train_forced_negative_indices if len(train_forced_negative_indices) > 0 else None,
            )

        dev_samples, dev_labels = self._subset_arrays(samples, combined_labels, dev_candidate_indices)
        spatial_cv_splits = None
        if split_mode == "spatial_cluster":
            dev_mineral_x, dev_mineral_y = _resolve_xy_columns(dev_minerals)
            train_rel, val_rel, split_info, spatial_cv_splits = _split_spatial_relative_indices(
                labels=combined_labels[dev_candidate_indices],
                coords=split_coords[dev_candidate_indices],
                metadata=split_metadata,
                patch_size=patch_size,
                n_folds=int(build_config.get("spatial_cluster_split", {}).get("cv_folds", 5) or 5),
                buffer_distance=build_config.get("spatial_cluster_split", {}).get("cv_buffer_distance"),
                partition_coords=dev_minerals[[dev_mineral_x, dev_mineral_y]].to_numpy(dtype=np.float64),
            )
            train_indices = dev_candidate_indices[np.asarray(train_rel, dtype=np.int64)]
            val_indices = dev_candidate_indices[np.asarray(val_rel, dtype=np.int64)]
            if len(train_indices) == 0 or len(val_indices) == 0:
                raise ValueError("Spatial validation failed to produce non-empty raw-feature train/val splits.")
            train_samples, train_labels = self._subset_arrays(samples, combined_labels, train_indices)
            val_samples, val_labels = self._subset_arrays(samples, combined_labels, val_indices)
            test_samples, test_labels = self._subset_arrays(samples, combined_labels, test_indices)
            train_minerals_df, val_minerals_df, cv_mineral_summary = _split_frame_by_spatial_cv(dev_minerals, spatial_cv_splits)
        else:
            train_rel, val_rel, split_info = _split_relative_indices(
                labels=combined_labels[dev_candidate_indices],
                coords=split_coords[dev_candidate_indices],
                validation_split=validation_split,
                n_blocks=n_blocks,
                metadata=split_metadata,
            )
            train_indices = dev_candidate_indices[np.asarray(train_rel, dtype=np.int64)]
            val_indices = dev_candidate_indices[np.asarray(val_rel, dtype=np.int64)]
            if len(train_indices) == 0 or len(val_indices) == 0:
                raise ValueError("Spatial validation failed to produce non-empty raw-feature train/val splits.")
            train_samples, train_labels = self._subset_arrays(samples, combined_labels, train_indices)
            val_samples, val_labels = self._subset_arrays(samples, combined_labels, val_indices)
            test_samples, test_labels = self._subset_arrays(samples, combined_labels, test_indices)
            train_minerals_df = dev_minerals.reset_index(drop=True)
            val_minerals_df = dev_minerals.iloc[0:0].copy()
            cv_mineral_summary = {
                "train_count": int(len(train_minerals_df)),
                "val_count": int(len(val_minerals_df)),
                "fold_index": 0,
            }
        split_info = _build_split_info(
            split_info,
            sample_count=len(dev_candidate_indices),
            train_count=len(train_indices),
            val_count=len(val_indices),
            n_blocks=int(build_config.get("spatial_cluster_split", {}).get("cv_folds", 5) if split_mode == "spatial_cluster" else n_blocks),
            validation_split=float(1.0 / max(int(build_config.get("spatial_cluster_split", {}).get("cv_folds", 5) if split_mode == "spatial_cluster" else n_blocks), 1)) if split_mode == "spatial_cluster" else validation_split,
        )
        split_info["sampling_percentage"] = sampling_percentage
        split_info["balance_ratio"] = balance_ratio
        split_info["buffer_requested"] = bool(buffer_policy["enable"])
        split_info["buffer_applied"] = bool(buffer_policy_applied)
        split_info["buffer_removed_count"] = int(buffer_policy_removed_count)
        split_info["buffer_distance"] = float(buffer_policy["radius_m"])
        split_info.setdefault("fallback_reason", "-")
        if split_mode == "spatial_cluster":
            split_info["spatial_cluster_active"] = True
            split_info["spatial_cluster_n_clusters"] = int(build_config.get("spatial_cluster_split", {}).get("n_clusters", 0) or 0)
            split_info["spatial_cluster_train_ratio"] = float(build_config.get("spatial_cluster_split", {}).get("train_ratio", 0.7) or 0.7)
            split_info["spatial_cluster_cv_folds"] = int(build_config.get("spatial_cluster_split", {}).get("cv_folds", 5) or 5)
            split_info["spatial_cluster_random_state"] = int(build_config.get("spatial_cluster_split", {}).get("random_state", 42) or 42)
            split_info["spatial_cluster_mineral_train_count"] = int(cv_mineral_summary.get("train_count", 0))
            split_info["spatial_cluster_mineral_val_count"] = int(cv_mineral_summary.get("val_count", 0))
            split_info["spatial_cluster_selected_fold"] = int(cv_mineral_summary.get("fold_index", 0))
        else:
            split_info["spatial_cluster_active"] = False
        dataset_meta = {
            "input_channels": int(samples.shape[1]),
            "image_size": int(samples.shape[-1]),
            "num_classes": 2,
            "input_kind": "window",
            "label_mode": "binary",
        }
        dataset_summary = {
            "h5_mode": "raw_features",
            "input_kind": "window",
            "label_mode": "binary",
            "input_channels": dataset_meta["input_channels"],
            "image_size": dataset_meta["image_size"],
            "num_classes": dataset_meta["num_classes"],
            "patch_size": patch_size,
            "patch_stride": patch_stride,
            "reflect_padding": bool(use_reflect_padding),
            "selected_channel_indices": split_metadata.get("selected_channel_indices", list(range(int(samples.shape[1])))),
            "selected_channel_names": split_metadata.get("selected_channel_names", split_metadata.get("available_channel_names", [])),
            "total_patch_count": int(len(samples)),
            "buffer_radius": buffer_radius,
            "validation_split": validation_split,
            "n_blocks": n_blocks,
            "sampling_percentage": sampling_percentage,
            "balance_ratio": balance_ratio,
            "augmentation_enabled": bool(build_config.get("augmentation_enabled", False)),
            "augmentation_noise_std": float(build_config.get("augmentation_noise_std", 0.01) or 0.01),
            "negative_ratio": None,
            "supervised_train_ratio": 1.0,
            "sample_construction_mode": "pu_full_unlabeled",
            "negative_sampling_mode": negative_sampling_mode,
            "negative_sampling_applied": True,
            "negative_distance_multiplier": negative_distance_multiplier,
            "negative_distance_radius": negative_distance_radius,
            "buffer_policy": dict(buffer_policy),
            "buffer_exclusion_enabled": bool(buffer_policy["enable"]),
            "buffer_exclusion_applied": bool(buffer_policy_applied),
            "buffer_exclusion_removed_count": int(buffer_policy_removed_count),
            "buffer_exclusion_distance": float(buffer_policy["radius_m"]),
            "test_window_overlap_exclusion_enabled": True,
            "test_window_overlap_exclusion_applied": bool(test_window_overlap_removed_count > 0),
            "test_window_overlap_exclusion_removed_count": int(test_window_overlap_removed_count),
            "test_window_overlap_exclusion_distance": float(test_window_overlap_radius),
            "evaluation_protocol": dict(evaluation_protocol),
            "split_mode": split_mode,
            "spatial_cluster_active": bool(split_mode == "spatial_cluster"),
            "spatial_cluster_n_clusters": int(build_config.get("spatial_cluster_split", {}).get("n_clusters", 0) or 0),
            "spatial_cluster_train_ratio": float(build_config.get("spatial_cluster_split", {}).get("train_ratio", 0.7) or 0.7),
            "spatial_cluster_cv_folds": int(build_config.get("spatial_cluster_split", {}).get("cv_folds", 5) or 5),
            "spatial_cluster_random_state": int(build_config.get("spatial_cluster_split", {}).get("random_state", 42) or 42),
            "spatial_cluster_train_mineral_count": int(len(dev_minerals)),
            "spatial_cluster_test_mineral_count": int(len(test_minerals)),
            "spatial_cv_enabled": bool(spatial_cv_splits),
            "spatial_cv_fold_count": int(spatial_cv_splits.get("fold_count", 0) if spatial_cv_splits else 0),
            "spatial_cv_axis": spatial_cv_splits.get("axis_name", None) if spatial_cv_splits else None,
            "spatial_cv_strategy": spatial_cv_splits.get("strategy", "x_quantile") if spatial_cv_splits else None,
            "spatial_cv_axis_source": spatial_cv_splits.get("axis_source", "x_train_mineral_quantile") if spatial_cv_splits else None,
            "spatial_cv_buffer_distance": float(spatial_cv_splits.get("buffer_distance", 0.0) if spatial_cv_splits else 0.0),
            "spatial_cv_fallback_reason": spatial_cv_splits.get("fallback_reason", "-") if spatial_cv_splits else "-",
            "spatial_cv_selected_fold": int(spatial_cv_splits.get("selected_fold_index", 0) if spatial_cv_splits else 0),
            "no_ore_active": bool(no_ore_active),
            "no_ore_point_count": int(no_ore_point_count),
            "no_ore_sample_count": int(no_ore_sample_count),
            "no_ore_conflict_count": int(no_ore_conflict_count),
            "spatial_validation_enabled": True,
            "split_info": split_info,
            "train_sample_count": int(len(train_samples)),
            "val_sample_count": int(len(val_samples)),
            "test_sample_count": int(len(test_samples)),
            "train_positive_count": int(np.sum(train_labels > 0)),
            "val_positive_count": int(np.sum(val_labels > 0)),
            "test_positive_count": int(np.sum(test_labels > 0)),
            "train_negative_count": int(np.sum(train_labels == 0)),
            "val_negative_count": int(np.sum(val_labels == 0)),
            "test_negative_count": int(np.sum(test_labels == 0)),
            "train_mineral_count": int(len(train_minerals_df)),
            "val_mineral_count": int(len(val_minerals_df)),
            "test_mineral_count": int(len(test_minerals)),
            "dev_mineral_count": int(len(dev_minerals)),
            "label_mapping": {str(key): int(value) for key, value in label_mapping.items()},
            "class_distribution": {
                int(label): int(np.sum((combined_labels > 0).astype(np.int64) == label))
                for label in np.unique((combined_labels > 0).astype(np.int64))
            },
            "conflict_sample_count": int(np.sum(conflict_mask)),
            "positive_repartition_fallback": bool(positive_repartition_fallback),
            "positive_repartition_reason": positive_repartition_reason,
        }
        dataset_summary.update(
            _region_summary_fields(
                region_partition,
                train_no_ore_count=int(len(train_forced_negative_indices)),
                test_no_ore_count=int(len(test_forced_negative_indices)),
            )
        )
        return self._finalize_bundle(
            h5_path=h5_path,
            h5_mode="raw_features",
            build_config=build_config,
            train_samples=train_samples,
            train_labels=train_labels,
            val_samples=val_samples,
            val_labels=val_labels,
            test_samples=test_samples,
            test_labels=test_labels,
            dataset_meta=dataset_meta,
            dataset_summary=dataset_summary,
            train_minerals_df=train_minerals_df,
            val_minerals_df=val_minerals_df,
            test_minerals_df=test_minerals,
            dev_samples=dev_samples,
            dev_labels=dev_labels,
            spatial_cv_splits=spatial_cv_splits,
            sample_coordinates=split_coords,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        )
