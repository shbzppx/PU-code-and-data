from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd


def _as_float32(array):
    return np.asarray(array, dtype=np.float32)


def _normalize_axis(values):
    axis = np.asarray(values, dtype=np.float64).reshape(-1)
    axis = axis[np.isfinite(axis)]
    axis = np.unique(axis)
    axis.sort()
    return axis


def _build_grid_from_table(coords, values):
    coords = np.asarray(coords, dtype=np.float64)
    values = np.asarray(values, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Invalid coordinate shape: {coords.shape}")
    if values.ndim == 1:
        values = values[:, None]
    if len(coords) != len(values):
        raise ValueError("Coordinate count does not match value count.")

    x_values = _normalize_axis(coords[:, 0])
    y_values = _normalize_axis(coords[:, 1])
    if len(x_values) == 0 or len(y_values) == 0:
        raise ValueError("Unable to infer grid axes from coordinates.")

    x_index = {float(np.round(x, 12)): idx for idx, x in enumerate(x_values)}
    y_index = {float(np.round(y, 12)): idx for idx, y in enumerate(y_values)}
    grid = np.full((values.shape[1], len(y_values), len(x_values)), np.nan, dtype=np.float32)

    for coord, value in zip(coords, values):
        x_key = float(np.round(coord[0], 12))
        y_key = float(np.round(coord[1], 12))
        if x_key not in x_index or y_key not in y_index:
            raise ValueError(f"Coordinate {coord!r} does not fit inferred grid axes.")
        channel_values = np.asarray(value, dtype=np.float32).reshape(-1)
        row = y_index[y_key]
        col = x_index[x_key]
        if np.isnan(grid[:, row, col]).any():
            grid[:, row, col] = channel_values
        else:
            raise ValueError("Duplicate coordinate detected while reconstructing grid.")

    if np.isnan(grid).any():
        raise ValueError("Grid reconstruction produced missing cells.")

    metadata = {
        "nx": int(len(x_values)),
        "ny": int(len(y_values)),
        "x_min": float(x_values[0]),
        "x_max": float(x_values[-1]),
        "y_min": float(y_values[0]),
        "y_max": float(y_values[-1]),
        "width": int(len(x_values)),
        "height": int(len(y_values)),
        "channels": int(grid.shape[0]),
        "x_scale": float(np.median(np.diff(x_values))) if len(x_values) > 1 else 1.0,
        "y_scale": float(np.median(np.diff(y_values))) if len(y_values) > 1 else 1.0,
    }
    return grid, metadata


def _compute_channel_stats(channel, method):
    valid = np.asarray(channel, dtype=np.float32)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        if method == "zscore":
            return {"mean": 0.0, "std": 0.0, "normalized": False}
        if method == "normalize":
            return {"min": 0.0, "max": 0.0, "normalized": False}
        raise ValueError(f"Unsupported normalization method: {method}")
    if method == "zscore":
        mean_val = float(np.mean(valid))
        std_val = float(np.std(valid))
        return {
            "mean": mean_val,
            "std": std_val,
            "normalized": std_val > 1e-9,
        }
    if method == "normalize":
        min_val = float(np.min(valid))
        max_val = float(np.max(valid))
        return {
            "min": min_val,
            "max": max_val,
            "normalized": max_val > min_val,
        }
    raise ValueError(f"Unsupported normalization method: {method}")


def _load_normalization_params(params_path):
    if not params_path:
        return None
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Normalization params file not found: {params_path}")
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    if not isinstance(params, dict):
        raise ValueError("Normalization params JSON must be an object.")
    return params


def _apply_normalization_to_grids(grids, file_names, normalize_method, params=None, log_func=print):
    method = (normalize_method or "none").strip().lower()
    if method in {"", "none"}:
        return grids, {
            "method": "none",
            "params_per_channel": {},
            "normalized": False,
        }

    def log(message):
        if log_func:
            log_func(message)

    provided_method = None
    provided_params = {}
    if params:
        provided_method = str(params.get("method", method)).strip().lower()
        provided_params = params.get("params_per_channel", {}) or {}
        if provided_method and provided_method != method:
            log(f"Warning: requested normalization method '{method}' differs from params file method '{provided_method}'. Using '{provided_method}'.")
            method = provided_method

    normalized_grids = []
    preprocess_params = {
        "method": method,
        "params_per_channel": {},
        "normalized": False,
    }
    any_normalized = False

    for grid, file_name in zip(grids, file_names):
        channel_name = os.path.basename(file_name)
        channel = np.asarray(grid, dtype=np.float32)
        channel_params = provided_params.get(channel_name)
        if channel_params:
            if method == "zscore":
                mean_val = channel_params.get("mean")
                std_val = channel_params.get("std")
                if mean_val is None or std_val is None:
                    raise ValueError(f"Normalization params for '{channel_name}' are incomplete.")
                mean_val = float(mean_val)
                std_val = float(std_val)
                if std_val > 1e-9:
                    channel = np.nan_to_num(channel, nan=mean_val)
                    channel = (channel - mean_val) / std_val
                    channel_params = {**channel_params, "normalized": True}
                else:
                    channel = np.zeros_like(channel, dtype=np.float32)
                    channel_params = {**channel_params, "normalized": False}
            elif method == "normalize":
                min_val = channel_params.get("min")
                max_val = channel_params.get("max")
                if min_val is None or max_val is None:
                    raise ValueError(f"Normalization params for '{channel_name}' are incomplete.")
                min_val = float(min_val)
                max_val = float(max_val)
                if max_val > min_val:
                    channel = np.nan_to_num(channel, nan=0.0)
                    channel = (channel - min_val) / (max_val - min_val)
                    channel_params = {**channel_params, "normalized": True}
                else:
                    channel = np.zeros_like(channel, dtype=np.float32)
                    channel_params = {**channel_params, "normalized": False}
            else:
                raise ValueError(f"Unsupported normalization method: {method}")
        else:
            channel_params = _compute_channel_stats(channel, method)
            if method == "zscore":
                mean_val = channel_params["mean"]
                std_val = channel_params["std"]
                if std_val > 1e-9:
                    channel = np.nan_to_num(channel, nan=mean_val)
                    channel = (channel - mean_val) / std_val
                else:
                    channel = np.zeros_like(channel, dtype=np.float32)
                    channel_params["normalized"] = False
            elif method == "normalize":
                min_val = channel_params["min"]
                max_val = channel_params["max"]
                if max_val > min_val:
                    channel = np.nan_to_num(channel, nan=0.0)
                    channel = (channel - min_val) / (max_val - min_val)
                else:
                    channel = np.zeros_like(channel, dtype=np.float32)
                    channel_params["normalized"] = False

        normalized_grids.append(np.asarray(channel, dtype=np.float32))
        preprocess_params["params_per_channel"][channel_name] = channel_params
        any_normalized = any_normalized or bool(channel_params.get("normalized", False))

    preprocess_params["normalized"] = any_normalized
    return normalized_grids, preprocess_params


def _load_h5_grid(file_path: str):
    with h5py.File(file_path, "r") as f:
        if "fused_features" in f:
            fused = np.asarray(f["fused_features"][:], dtype=np.float32)
            if fused.ndim == 2:
                fused = fused[None, :, :]
            elif fused.ndim == 3:
                pass
            elif fused.ndim == 4 and fused.shape[0] == 1:
                fused = fused.reshape(fused.shape[1], fused.shape[2], fused.shape[3])
            else:
                raise ValueError(f"Unsupported fused_features shape: {fused.shape}")

            metadata = {}
            if "metadata" in f:
                metadata = {key: value for key, value in f["metadata"].attrs.items()}
            if not metadata:
                metadata = {
                    "channels": int(fused.shape[0]),
                    "height": int(fused.shape[1]),
                    "width": int(fused.shape[2]),
                    "nx": int(fused.shape[2]),
                    "ny": int(fused.shape[1]),
                }
            return fused, metadata

        if "coordinates" in f and "vectors" in f:
            coords = np.asarray(f["coordinates"][:], dtype=np.float32)
            vectors = np.asarray(f["vectors"][:], dtype=np.float32)
            grid, metadata = _build_grid_from_table(coords, vectors)
            return grid, metadata

        if "x_coords" in f and "y_coords" in f and "data" in f:
            x_coords = np.asarray(f["x_coords"][:], dtype=np.float64).reshape(-1)
            y_coords = np.asarray(f["y_coords"][:], dtype=np.float64).reshape(-1)
            data = np.asarray(f["data"][:], dtype=np.float32)
            if data.ndim == 2:
                grid = data[None, :, :]
            elif data.ndim == 3:
                grid = np.transpose(data, (2, 1, 0))
            else:
                raise ValueError(f"Unsupported data shape: {data.shape}")
            metadata = {
                "nx": int(len(x_coords)),
                "ny": int(len(y_coords)),
                "x_min": float(x_coords.min()) if len(x_coords) else 0.0,
                "x_max": float(x_coords.max()) if len(x_coords) else 0.0,
                "y_min": float(y_coords.min()) if len(y_coords) else 0.0,
                "y_max": float(y_coords.max()) if len(y_coords) else 0.0,
                "width": int(len(x_coords)),
                "height": int(len(y_coords)),
                "channels": int(grid.shape[0]),
                "x_scale": float(np.median(np.diff(np.sort(x_coords)))) if len(x_coords) > 1 else 1.0,
                "y_scale": float(np.median(np.diff(np.sort(y_coords)))) if len(y_coords) > 1 else 1.0,
            }
            return grid, metadata

    raise ValueError(f"Unsupported H5 grid structure: {file_path}")


def parse_surfer_grid(file_path: str):
    """Parse Surfer 6 ASCII Grid (.grd) files into a single-channel grid."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Grid file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        dsaa_id = f.readline().strip()
        if dsaa_id != "DSAA":
            raise ValueError(f"Invalid grid header, expected DSAA: {file_path}")

        nx, ny = map(int, f.readline().split())
        x_min, x_max = map(float, f.readline().split())
        y_min, y_max = map(float, f.readline().split())
        z_min, z_max = map(float, f.readline().split())

        value_tokens: List[str] = []
        for line in f:
            stripped = line.strip()
            if stripped:
                value_tokens.extend(stripped.split())

    try:
        flat_values = np.asarray([float(tok) for tok in value_tokens], dtype=np.float32)
    except ValueError as exc:
        raise ValueError(f"GRD data contains non-numeric values: {file_path}") from exc

    expected = nx * ny
    if flat_values.size < expected:
        raise ValueError(f"GRD data points are insufficient: expected {expected}, got {flat_values.size}")
    if flat_values.size > expected:
        flat_values = flat_values[:expected]

    grid_data = flat_values.reshape(ny, nx)
    grid_data = np.flipud(grid_data)
    grid = grid_data[None, :, :].astype(np.float32, copy=False)

    metadata = {
        "nx": int(nx),
        "ny": int(ny),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "z_min": float(z_min),
        "z_max": float(z_max),
        "width": int(nx),
        "height": int(ny),
        "channels": 1,
        "x_scale": float((x_max - x_min) / max(nx - 1, 1)),
        "y_scale": float((y_max - y_min) / max(ny - 1, 1)),
        "source_format": "DSAA",
    }
    return grid, metadata


def read_grid_like_file(file_path: str):
    """Read a grid-like file and return a channel-first grid tensor plus metadata."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = Path(file_path).suffix.lower()

    if ext == ".grd":
        return parse_surfer_grid(file_path)

    if ext == ".h5":
        return _load_h5_grid(file_path)

    if ext in {".csv", ".dat", ".txt"}:
        sep = r"\s+|,|;|\t"
        df = pd.read_csv(file_path, sep=sep, engine="python", header=None)
        if df.shape[1] < 3:
            raise ValueError(f"Not enough columns in grid file: {file_path}")
        df = df.iloc[:, :3].copy()
        df.columns = ["X", "Y", "Value"]
        coords = df[["X", "Y"]].to_numpy(dtype=np.float32)
        values = df[["Value"]].to_numpy(dtype=np.float32)
        grid, metadata = _build_grid_from_table(coords, values)
        return grid, metadata

    if ext in {".xls", ".xlsx"}:
        df = pd.read_excel(file_path)
        if df.shape[1] < 3:
            raise ValueError(f"Not enough columns in Excel grid file: {file_path}")
        df = df.iloc[:, :3].copy()
        df.columns = ["X", "Y", "Value"]
        coords = df[["X", "Y"]].to_numpy(dtype=np.float32)
        values = df[["Value"]].to_numpy(dtype=np.float32)
        grid, metadata = _build_grid_from_table(coords, values)
        return grid, metadata

    raise ValueError(f"Unsupported grid file format: {file_path}")


def combine_grid_files_to_h5(input_files, output_file, normalize_method="none", params_path=None, log_func=print):
    """Combine multiple grid files into an old-style fused_features H5.

    Output schema:
    - /fused_features: (C, H, W)
    - /metadata (group with attrs)
    - root attrs['file_names']
    - root attrs['source_type'] = 'grid_fused_features'
    """

    if not input_files:
        raise ValueError("Please select at least one input file.")

    def log(msg):
        if log_func:
            log_func(msg)

    grids = []
    metadata_list = []
    file_names = []

    log("Starting grid file fusion...")
    for idx, file_path in enumerate(input_files, 1):
        log(f"[{idx}/{len(input_files)}] Loading: {file_path}")
        grid, metadata = read_grid_like_file(file_path)
        grids.append(np.asarray(grid, dtype=np.float32))
        metadata_list.append(dict(metadata or {}))
        file_names.append(os.path.basename(file_path))
        log(f"  -> grid shape: {grid.shape}")

    reference = metadata_list[0]
    ref_channels, ref_height, ref_width = grids[0].shape
    for file_path, grid, metadata in zip(input_files, grids, metadata_list):
        if grid.shape[1:] != (ref_height, ref_width):
            raise ValueError(f"Grid shape mismatch for {file_path}: {grid.shape} vs {(ref_channels, ref_height, ref_width)}")
        if int(metadata.get("nx", ref_width)) != ref_width or int(metadata.get("ny", ref_height)) != ref_height:
            raise ValueError(f"Grid dimensions mismatch for {file_path}")
        if not np.isclose(float(metadata.get("x_min", reference.get("x_min", 0.0))), float(reference.get("x_min", 0.0))):
            log(f"Warning: x_min differs for {file_path}")
        if not np.isclose(float(metadata.get("x_max", reference.get("x_max", 0.0))), float(reference.get("x_max", 0.0))):
            log(f"Warning: x_max differs for {file_path}")
        if not np.isclose(float(metadata.get("y_min", reference.get("y_min", 0.0))), float(reference.get("y_min", 0.0))):
            log(f"Warning: y_min differs for {file_path}")
        if not np.isclose(float(metadata.get("y_max", reference.get("y_max", 0.0))), float(reference.get("y_max", 0.0))):
            log(f"Warning: y_max differs for {file_path}")

    params = None
    if normalize_method not in {"", "none"} and params_path:
        params = _load_normalization_params(params_path)
    grids, preprocess_params = _apply_normalization_to_grids(
        grids,
        file_names,
        normalize_method=normalize_method,
        params=params,
        log_func=log_func,
    )

    fused_features = np.concatenate(grids, axis=0).astype(np.float32, copy=False)
    channels, height, width = fused_features.shape

    metadata = {
        "channels": int(channels),
        "height": int(height),
        "width": int(width),
        "nx": int(reference.get("nx", width)),
        "ny": int(reference.get("ny", height)),
        "x_min": float(reference.get("x_min", 0.0)),
        "x_max": float(reference.get("x_max", float(width - 1))),
        "y_min": float(reference.get("y_min", 0.0)),
        "y_max": float(reference.get("y_max", float(height - 1))),
        "x_scale": float(reference.get("x_scale", 1.0)),
        "y_scale": float(reference.get("y_scale", 1.0)),
        "z_min": float(np.nanmin(fused_features)),
        "z_max": float(np.nanmax(fused_features)),
        "source_files": str([str(path) for path in input_files]),
        "normalized": bool(preprocess_params.get("normalized", False)),
        "normalization_method": str(preprocess_params.get("method", "none")),
    }

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with h5py.File(output_file, "w") as f:
        f.create_dataset("fused_features", data=fused_features, dtype=np.float32)
        meta_group = f.create_group("metadata")
        for key, value in metadata.items():
            meta_group.attrs[key] = value
        f.attrs["file_names"] = np.asarray(file_names, dtype=h5py.string_dtype(encoding="utf-8"))
        f.attrs["source_type"] = "grid_fused_features"

    if normalize_method not in {"", "none"} and params is None:
        params_output = Path(output_file).with_suffix(".json")
        with open(params_output, "w", encoding="utf-8") as f:
            json.dump(preprocess_params, f, ensure_ascii=False, indent=4)
        log(f"Normalization params saved to: {params_output}")

    log(f"Grid fusion complete: {output_file}")
    log(f"  channels={channels}, height={height}, width={width}")
    return output_file
