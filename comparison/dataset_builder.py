"""Dataset construction utilities for model comparison workflows."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(CURRENT_DIR)
COMMON_DIR = os.path.join(CODE_ROOT, "common")
for path in (CODE_ROOT, COMMON_DIR):
    if path not in sys.path:
        sys.path.append(path)

from feature_channel_utils import infer_h5_channel_names, subset_samples_by_channels
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

from .data_splitter import MineralBasedSplitter
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
        if "samples" not in handle:
            raise KeyError("Prebuilt H5 must contain a 'samples' dataset.")
        samples = np.asarray(handle["samples"][:], dtype=np.float32)
        if "labels" in handle:
            labels = np.asarray(handle["labels"][:]).reshape(-1).astype(np.int64)
        elif "label" in handle:
            labels = np.asarray(handle["label"][:]).reshape(-1).astype(np.int64)
        else:
            labels = None
        coordinates = np.asarray(handle["coordinates"][:]) if "coordinates" in handle else None
        metadata = _extract_geo_metadata(handle)
        metadata["available_channel_names"] = infer_h5_channel_names(h5_path)
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


def _window_half_extent(metadata: Optional[Dict[str, object]], patch_size: int) -> Tuple[float, float]:
    meta = metadata or {}
    x_scale = abs(float(meta.get("x_scale", 1.0) or 1.0))
    y_scale = abs(float(meta.get("y_scale", 1.0) or 1.0))
    window_width = int(meta.get("window_width", patch_size) or patch_size)
    window_height = int(meta.get("window_height", patch_size) or patch_size)
    return max(float(window_width) * x_scale / 2.0, 0.0), max(float(window_height) * y_scale / 2.0, 0.0)


def _label_array_from_minerals(
    sample_coords: np.ndarray,
    minerals_df: Optional[pd.DataFrame],
    buffer_radius: float,
    label_mapping: Dict[object, int],
    *,
    metadata: Optional[Dict[str, object]] = None,
    patch_size: Optional[int] = None,
) -> np.ndarray:
    labels = np.zeros(len(sample_coords), dtype=np.int64)
    if minerals_df is None or len(minerals_df) == 0:
        return labels
    x_column, y_column = _resolve_xy_columns(minerals_df)
    label_column = _resolve_label_column(minerals_df)
    half_width = half_height = None
    if patch_size is not None:
        half_width, half_height = _window_half_extent(metadata, int(patch_size))
    for _, row in minerals_df.iterrows():
        mineral_x = float(row[x_column])
        mineral_y = float(row[y_column])
        if label_column is None:
            label_value = 1
        else:
            label_value = int(label_mapping.get(row[label_column], 1))
        if half_width is not None and half_height is not None:
            in_window_mask = (
                (np.abs(sample_coords[:, 0] - mineral_x) <= half_width)
                & (np.abs(sample_coords[:, 1] - mineral_y) <= half_height)
            )
        else:
            distances = np.sqrt((sample_coords[:, 0] - mineral_x) ** 2 + (sample_coords[:, 1] - mineral_y) ** 2)
            in_window_mask = distances <= float(buffer_radius)
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
    dataset_meta: Dict[str, int]
    dataset_summary: Dict[str, object]
    build_config: Dict[str, object]
    h5_path: str
    train_minerals_df: pd.DataFrame
    val_minerals_df: pd.DataFrame
    test_minerals_df: pd.DataFrame


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
        return sample, int(self.labels[index])


class ArrayClassificationDataset(H5ClassificationDataset):
    pass


class ComparisonDataBuilder:
    """Build train/val/test bundles for model comparison."""

    def __init__(self, negative_ratio: int = 3):
        self.negative_ratio = int(negative_ratio)

    def detect_h5_mode(self, h5_path: str) -> str:
        with h5py.File(h5_path, "r") as handle:
            if "samples" in handle:
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
            progress_callback(f"Building dataset bundle in {h5_mode} mode...")

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

        return ComparisonDatasetBundle(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train_data_array=(train_samples, train_labels),
            val_data_array=(val_samples, val_labels),
            test_data_array=(test_samples, test_labels),
            dataset_meta=dict(dataset_meta),
            dataset_summary=dataset_summary,
            build_config=dict(build_config),
            h5_path=h5_path,
            train_minerals_df=train_minerals_df.reset_index(drop=True),
            val_minerals_df=val_minerals_df.reset_index(drop=True),
            test_minerals_df=test_minerals_df.reset_index(drop=True),
        )

    def _subset_arrays(self, samples, labels, indices):
        indices = np.asarray(indices, dtype=np.int64)
        return np.asarray(samples[indices], dtype=np.float32), np.asarray(labels[indices], dtype=np.int64)

    def _build_prebuilt_bundle(self, h5_path, train_minerals, val_minerals, test_minerals, build_config=None, progress_callback=None, no_ore_minerals=None):
        build_config = build_config or {}
        samples, labels, coordinates, metadata = _read_prebuilt_arrays(h5_path)
        samples, metadata = subset_samples_by_channels(samples, metadata, build_config.get("selected_channels"))
        if coordinates is None:
            raise KeyError("Prebuilt H5 must contain coordinates for spatial validation.")

        sampling_percentage, balance_ratio = self._resolve_sampling_controls(build_config)
        buffer_radius = float(build_config.get("buffer_radius", 500.0))
        negative_sampling_mode, negative_distance_multiplier, negative_distance_radius = _resolve_negative_sampling_scheme(
            build_config,
            buffer_radius,
        )
        dev_minerals = pd.concat([train_minerals, val_minerals], ignore_index=True)
        no_ore_minerals = None if no_ore_minerals is None or len(no_ore_minerals) == 0 else no_ore_minerals.reset_index(drop=True)
        no_ore_active = no_ore_minerals is not None and len(no_ore_minerals) > 0

        patch_metadata = dict(metadata)
        if "window_width" not in patch_metadata:
            patch_metadata["window_width"] = int(samples.shape[-1])
        if "window_height" not in patch_metadata:
            patch_metadata["window_height"] = int(samples.shape[-2])

        split_coords = _patch_geo_coordinates(coordinates, patch_metadata)
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
                progress_callback("Prebuilt H5 has no labels; deriving labels from mineral files...")
            label_source = "mineral_files"
            label_mapping = _build_label_mapping(dev_minerals, test_minerals)
            dev_positive_labels = _label_array_from_minerals(
                split_coords,
                dev_minerals,
                buffer_radius,
                label_mapping,
                metadata=patch_metadata,
                patch_size=int(patch_metadata.get("window_width", patch_metadata.get("window_height", build_config.get("patch_size", 1))) or 1),
            )
            test_positive_labels = _label_array_from_minerals(
                split_coords,
                test_minerals,
                buffer_radius,
                label_mapping,
                metadata=patch_metadata,
                patch_size=int(patch_metadata.get("window_width", patch_metadata.get("window_height", build_config.get("patch_size", 1))) or 1),
            )
            conflict_mask = (dev_positive_labels > 0) & (test_positive_labels > 0)
            conflict_sample_count = int(np.sum(conflict_mask))
            dev_positive_labels[conflict_mask] = 0
            test_positive_labels[conflict_mask] = 0

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
            rng = np.random.default_rng(42)
            if balance_ratio is not None:
                test_negative_target = int(round(len(test_positive_indices) / balance_ratio))
                dev_negative_target = int(round(len(dev_positive_indices) / balance_ratio))
            else:
                test_negative_target = max(len(test_positive_indices) * self.negative_ratio, len(test_positive_indices))
                dev_negative_target = max(len(dev_positive_indices) * self.negative_ratio, len(dev_positive_indices))
            if region_active:
                dev_background_indices = _filter_indices_by_mask(background_indices, train_region_mask)
                test_background_indices = _filter_indices_by_mask(background_indices, test_region_mask)
                dev_background_target = max(dev_negative_target - len(train_hard_negative_indices), 0)
                test_background_target = max(test_negative_target - len(test_hard_negative_indices), 0)
                dev_negative_indices = np.concatenate(
                    [
                        train_hard_negative_indices,
                        _sample_background_indices(dev_background_indices, dev_background_target, rng),
                    ]
                ).astype(np.int64)
                test_negative_indices = np.concatenate(
                    [
                        test_hard_negative_indices,
                        _sample_background_indices(test_background_indices, test_background_target, rng),
                    ]
                ).astype(np.int64)
            else:
                test_negative_indices = _sample_background_indices(background_indices, test_negative_target, rng)
                remaining_background = np.setdiff1d(background_indices, test_negative_indices, assume_unique=False)
                dev_negative_target = max(dev_negative_target - len(hard_negative_indices), 0)
                dev_negative_indices = np.concatenate([
                    hard_negative_indices,
                    _sample_background_indices(remaining_background, dev_negative_target, rng),
                ]).astype(np.int64)
            dev_negative_indices = np.sort(np.unique(dev_negative_indices))
            test_negative_indices = np.sort(np.unique(test_negative_indices))
            if progress_callback is not None:
                progress_callback(
                    "Prebuilt sampling: "
                    f"dev_pos={len(dev_positive_indices)}, dev_neg={len(dev_negative_indices)}, "
                    f"test_pos={len(test_positive_indices)}, test_neg={len(test_negative_indices)}"
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
                    f"Derived labels from mineral files: positives={derived_positive_count}, negatives={derived_negative_count}"
                )
                if derived_positive_count == 0:
                    progress_callback(
                        "Warning: no positive labels were derived from mineral files; please verify coordinate origin and units."
                    )
        else:
            if region_active:
                dev_indices = np.where(train_region_mask)[0].astype(np.int64)
                test_indices = np.where(test_region_mask)[0].astype(np.int64)
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
            if len(dev_indices) == 0:
                raise ValueError("Development split is empty after mineral-based masking.")
            if len(test_indices) == 0:
                raise ValueError("Test split is empty after mineral-based masking.")

        negative_sampling_applied = bool(labels_were_missing)
        if sampling_percentage < 1.0:
            dev_indices, _ = _sample_index_pool(
                dev_indices,
                labels[dev_indices],
                sampling_percentage,
                protected_indices=train_hard_negative_indices if no_ore_active else None,
            )
        if not labels_were_missing and balance_ratio is not None:
            dev_indices, _ = _balance_index_pool(
                dev_indices,
                labels[dev_indices],
                balance_ratio,
                protected_indices=train_hard_negative_indices if no_ore_active else None,
            )
            test_indices, _ = _balance_index_pool(
                test_indices,
                labels[test_indices],
                balance_ratio,
                protected_indices=test_hard_negative_indices if region_active and no_ore_active else None,
            )

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

        split_info = _build_split_info(
            split_info,
            sample_count=len(dev_indices),
            train_count=len(train_indices),
            val_count=len(val_indices),
            n_blocks=int(build_config.get("n_blocks", 3)),
            validation_split=float(build_config.get("val_ratio", 0.2)),
        )
        split_info["sampling_percentage"] = sampling_percentage
        split_info["balance_ratio"] = balance_ratio
        dataset_meta = {
            "input_channels": int(samples.shape[1]),
            "image_size": int(samples.shape[-1]),
            "num_classes": int(_max_label(labels, test_labels) + 1),
        }
        dataset_summary = {
            "h5_mode": "prebuilt",
            "input_channels": dataset_meta["input_channels"],
            "image_size": dataset_meta["image_size"],
            "num_classes": dataset_meta["num_classes"],
            "patch_size": int(samples.shape[-1]),
            "patch_stride": int(build_config.get("patch_stride", samples.shape[-1])),
            "selected_channel_indices": metadata.get("selected_channel_indices", list(range(int(samples.shape[1])))),
            "selected_channel_names": metadata.get("selected_channel_names", metadata.get("available_channel_names", [])),
            "buffer_radius": float(build_config.get("buffer_radius", 500.0)),
            "validation_split": float(build_config.get("val_ratio", 0.2)),
            "n_blocks": int(build_config.get("n_blocks", 3)),
            "sampling_percentage": sampling_percentage,
            "balance_ratio": balance_ratio,
            "negative_ratio": self.negative_ratio,
            "negative_sampling_mode": negative_sampling_mode,
            "negative_sampling_applied": negative_sampling_applied,
            "negative_distance_multiplier": negative_distance_multiplier,
            "negative_distance_radius": negative_distance_radius,
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
            "class_distribution": {int(label): int(np.sum(labels == label)) for label in np.unique(labels)},
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
        )

    def _build_raw_bundle(self, h5_path, train_minerals, val_minerals, test_minerals, build_config=None, no_ore_minerals=None):
        build_config = build_config or {}
        dev_minerals = pd.concat([train_minerals, val_minerals], ignore_index=True)
        patch_size = int(build_config.get("patch_size", 64))
        patch_stride = int(build_config.get("patch_stride", patch_size))
        buffer_radius = float(build_config.get("buffer_radius", 500.0))
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
        samples, split_metadata = subset_samples_by_channels(samples, split_metadata, build_config.get("selected_channels"))
        split_coords = _patch_geo_coordinates(coordinates, split_metadata)
        if split_coords is None:
            raise ValueError("Unable to derive geographic coordinates for raw-feature patches.")
        region_partition = _resolve_region_partition(split_coords, build_config)
        region_active = region_partition is not None
        train_region_mask = None if not region_active else np.asarray(region_partition["train_mask"], dtype=bool)
        test_region_mask = None if not region_active else np.asarray(region_partition["test_mask"], dtype=bool)

        label_mapping = _build_label_mapping(dev_minerals, test_minerals)
        dev_positive_labels = _label_array_from_minerals(
            split_coords,
            dev_minerals,
            buffer_radius,
            label_mapping,
            metadata=split_metadata,
            patch_size=patch_size,
        )
        test_positive_labels = _label_array_from_minerals(
            split_coords,
            test_minerals,
            buffer_radius,
            label_mapping,
            metadata=split_metadata,
            patch_size=patch_size,
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

        conflict_mask = (dev_positive_labels > 0) & (test_positive_labels > 0)
        dev_positive_labels[conflict_mask] = 0
        test_positive_labels[conflict_mask] = 0

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
            raise ValueError("No positive development patches were generated.")
        if len(test_positive_indices) == 0:
            raise ValueError("No positive test patches were generated.")

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
        rng = np.random.default_rng(42)
        if balance_ratio is not None:
            test_negative_target = int(round(len(test_positive_indices) / balance_ratio))
            dev_negative_target = int(round(len(dev_positive_indices) / balance_ratio))
        else:
            test_negative_target = max(len(test_positive_indices) * self.negative_ratio, len(test_positive_indices))
            dev_negative_target = max(len(dev_positive_indices) * self.negative_ratio, len(dev_positive_indices))
        if region_active:
            dev_background_indices = _filter_indices_by_mask(background_indices, train_region_mask)
            test_background_indices = _filter_indices_by_mask(background_indices, test_region_mask)
            dev_background_target = max(dev_negative_target - len(train_forced_negative_indices), 0)
            test_background_target = max(test_negative_target - len(test_forced_negative_indices), 0)
            dev_negative_indices = np.concatenate([
                train_forced_negative_indices,
                _sample_background_indices(dev_background_indices, dev_background_target, rng),
            ]).astype(np.int64)
            test_negative_indices = np.concatenate([
                test_forced_negative_indices,
                _sample_background_indices(test_background_indices, test_background_target, rng),
            ]).astype(np.int64)
        else:
            test_negative_indices = _sample_background_indices(background_indices, test_negative_target, rng)
            remaining_background = np.setdiff1d(background_indices, test_negative_indices, assume_unique=False)
            dev_background_target = max(dev_negative_target - len(train_forced_negative_indices), 0)
            dev_negative_indices = np.concatenate([
                train_forced_negative_indices,
                _sample_background_indices(remaining_background, dev_background_target, rng),
            ]).astype(np.int64)
        dev_negative_indices = np.sort(np.unique(dev_negative_indices))
        test_negative_indices = np.sort(np.unique(test_negative_indices))

        dev_candidate_indices = np.sort(np.concatenate([dev_positive_indices, dev_negative_indices])).astype(np.int64)
        test_indices = np.sort(np.concatenate([test_positive_indices, test_negative_indices])).astype(np.int64)
        combined_labels = np.zeros(len(samples), dtype=np.int64)
        combined_labels[dev_positive_indices] = dev_positive_labels[dev_positive_indices]
        combined_labels[test_positive_indices] = test_positive_labels[test_positive_indices]

        if sampling_percentage < 1.0:
            dev_candidate_indices, _ = _sample_index_pool(
                dev_candidate_indices,
                combined_labels[dev_candidate_indices],
                sampling_percentage,
                protected_indices=train_forced_negative_indices if len(train_forced_negative_indices) > 0 else None,
            )

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
        split_info = _build_split_info(
            split_info,
            sample_count=len(dev_candidate_indices),
            train_count=len(train_indices),
            val_count=len(val_indices),
            n_blocks=n_blocks,
            validation_split=validation_split,
        )
        split_info["sampling_percentage"] = sampling_percentage
        split_info["balance_ratio"] = balance_ratio
        dataset_meta = {
            "input_channels": int(samples.shape[1]),
            "image_size": int(samples.shape[-1]),
            "num_classes": int(_max_label(combined_labels, test_labels) + 1),
        }
        dataset_summary = {
            "h5_mode": "raw_features",
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
            "negative_ratio": self.negative_ratio,
            "negative_sampling_mode": negative_sampling_mode,
            "negative_sampling_applied": True,
            "negative_distance_multiplier": negative_distance_multiplier,
            "negative_distance_radius": negative_distance_radius,
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
            "class_distribution": {int(label): int(np.sum(combined_labels == label)) for label in np.unique(combined_labels)},
            "conflict_sample_count": int(np.sum(conflict_mask)),
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
        )
