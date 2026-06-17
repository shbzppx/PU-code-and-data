from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """Simple in-memory dataset wrapper."""

    def __init__(self, samples, labels=None, normalization_stats=None):
        self.samples = np.asarray(samples, dtype=np.float32)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64).reshape(-1)
        self.normalization_stats = normalization_stats
        self.normalization_mean = None
        self.normalization_std = None
        if normalization_stats:
            mean_values = np.asarray(normalization_stats.get("mean", []), dtype=np.float32)
            std_values = np.asarray(normalization_stats.get("std", []), dtype=np.float32)
            if mean_values.size and std_values.size:
                self.normalization_mean = torch.as_tensor(mean_values, dtype=torch.float32).view(-1, 1, 1)
                safe_std = np.where(np.abs(std_values) < 1e-6, 1.0, std_values)
                self.normalization_std = torch.as_tensor(safe_std, dtype=torch.float32).view(-1, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = torch.as_tensor(self.samples[index], dtype=torch.float32)
        if self.normalization_mean is not None and self.normalization_std is not None:
            sample = (sample - self.normalization_mean) / self.normalization_std
        if self.labels is None:
            return sample
        return sample, int(self.labels[index])


class H5Dataset(MyDataset):
    """Compatibility wrapper that reads samples from an H5 file."""

    def __init__(self, h5_path, sample_key="samples", label_key="labels", normalization_stats=None):
        with h5py.File(h5_path, "r") as handle:
            if sample_key not in handle:
                raise KeyError(f"H5 file is missing required dataset: {sample_key}")
            samples = np.asarray(handle[sample_key][:], dtype=np.float32)
            labels = None
            if label_key in handle:
                labels = np.asarray(handle[label_key][:], dtype=np.int64).reshape(-1)
        super().__init__(samples, labels, normalization_stats=normalization_stats)


def _patch_indices_to_geo(coords, metadata):
    """Convert patch indices to approximate geographic coordinates."""

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

    # PatchCreator grid windows output (row, col).  Mark those callers explicitly so
    # square grids are not accidentally interpreted as (x_index, y_index).
    swapped_x = arr[:, 1]
    swapped_y = arr[:, 0]
    swapped_ok = np.nanmax(np.abs(swapped_x)) <= x_bound and np.nanmax(np.abs(swapped_y)) <= y_bound
    coordinate_order = str(meta.get("coordinate_order", "") or "").strip().lower()
    row_col_order = coordinate_order in {"row_col", "row,col", "yx", "ij"} or bool(meta.get("coordinates_are_row_col", False))

    if row_col_order:
        x_values = swapped_x
        y_values = swapped_y
    elif not direct_ok and swapped_ok:
        x_values = swapped_x
        y_values = swapped_y
    else:
        x_values = direct_x
        y_values = direct_y

    if np.nanmax(np.abs(x_values)) <= x_bound and np.nanmax(np.abs(y_values)) <= y_bound:
        x_step = x_span / max(nx - 1, 1)
        y_step = y_span / max(ny - 1, 1)

        # For sliding windows, use window center if metadata provides patch size.
        window_width = int(meta.get("window_width", 1) or 1)
        window_height = int(meta.get("window_height", 1) or 1)
        coordinates_are_centers = bool(meta.get("coordinates_are_centers", False))
        x_offset = 0.0 if coordinates_are_centers else (window_width - 1) / 2.0 if window_width > 1 else 0.0
        y_offset = 0.0 if coordinates_are_centers else (window_height - 1) / 2.0 if window_height > 1 else 0.0

        geo_x = x_min + (x_values + x_offset) * x_step
        geo_y = y_max - (y_values + y_offset) * y_step
        return np.column_stack([geo_x, geo_y]).astype(np.float64, copy=False)

    return arr


def _split_relative_indices(
    *,
    labels_df,
    coords=None,
    validation_split=0.2,
    stratify=True,
    n_blocks=3,
    random_state=42,
    metadata=None,
):
    """Return train/val relative indices for the current comparison workflow."""

    del coords, metadata, n_blocks

    if isinstance(labels_df, pd.DataFrame):
        if "label" in labels_df.columns:
            labels = labels_df["label"].to_numpy()
        else:
            labels = labels_df.iloc[:, 0].to_numpy()
    else:
        labels = np.asarray(labels_df)

    labels = np.asarray(labels).reshape(-1)
    sample_count = int(len(labels))
    if sample_count <= 1:
        train_indices = list(range(sample_count))
        return train_indices, [], {
            "strategy": "degenerate",
            "train_count": int(len(train_indices)),
            "val_count": 0,
            "validation_split": float(validation_split),
            "stratified": False,
            "random_state": int(random_state),
        }

    requested_val_count = int(round(sample_count * float(validation_split)))
    val_count = max(1, min(sample_count - 1, requested_val_count))
    indices = np.arange(sample_count)

    use_stratify = bool(stratify) and len(np.unique(labels)) > 1
    if use_stratify:
        class_counts = np.unique(labels, return_counts=True)[1]
        if class_counts.size == 0 or int(class_counts.min()) < 2:
            use_stratify = False
        elif val_count < len(np.unique(labels)):
            use_stratify = False

    try:
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_count,
            random_state=int(random_state),
            shuffle=True,
            stratify=labels if use_stratify else None,
        )
    except ValueError:
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_count,
            random_state=int(random_state),
            shuffle=True,
            stratify=None,
        )
        use_stratify = False

    train_indices = np.sort(np.asarray(train_indices, dtype=np.int64))
    val_indices = np.sort(np.asarray(val_indices, dtype=np.int64))
    split_info = {
        "strategy": "stratified_random" if use_stratify else "random",
        "train_count": int(len(train_indices)),
        "val_count": int(len(val_indices)),
        "validation_split": float(validation_split),
        "stratified": bool(use_stratify),
        "random_state": int(random_state),
    }
    return train_indices.tolist(), val_indices.tolist(), split_info
