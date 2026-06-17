from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Tuple

import h5py
import numpy as np


class PatchCreator:
    """Generate sliding-window patches from sample, fused-feature, or coordinate-vector H5 files."""

    def __init__(self, input_h5_path):
        if not os.path.exists(input_h5_path):
            raise FileNotFoundError(f"Input file not found: {input_h5_path}")

        self.input_h5_path = input_h5_path
        self.source_h5_file = h5py.File(input_h5_path, "r")
        self.source_kind = self._detect_source_kind()

    def close(self):
        if getattr(self, "source_h5_file", None) is not None:
            try:
                self.source_h5_file.close()
            finally:
                self.source_h5_file = None

    def __del__(self):
        self.close()

    @staticmethod
    def _normalize_window(value) -> Tuple[int, int]:
        if isinstance(value, int):
            return int(value), int(value)
        if isinstance(value, Iterable):
            value = tuple(value)
            if len(value) != 2:
                raise ValueError(f"Window parameter must have length 2, got {value!r}")
            return int(value[0]), int(value[1])
        raise TypeError(f"Unsupported window parameter: {value!r}")

    @staticmethod
    def _ensure_patch_tensor(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 4:
            return array
        if array.ndim == 3:
            return array[:, None, :, :]
        if array.ndim == 2:
            return array[:, :, None, None]
        raise ValueError(f"Unsupported patch tensor shape: {array.shape}")

    def _detect_source_kind(self) -> str:
        handle = self.source_h5_file
        if "samples" in handle:
            return "samples"
        if "fused_features" in handle:
            return "fused_features"
        if "vectors" in handle and "coordinates" in handle:
            return "coordinate_vectors"
        raise KeyError("HDF5 file must contain 'samples', 'fused_features', or ('coordinates' and 'vectors').")

    def _load_coordinates(self, sample_count):
        if "coordinates" in self.source_h5_file:
            coords = np.asarray(self.source_h5_file["coordinates"][:], dtype=np.float64)
            if coords.ndim == 1:
                coords = coords.reshape(-1, 1)
            if coords.shape[0] == sample_count and coords.shape[1] >= 2:
                return coords[:, :2].astype(np.float64, copy=False)
        return np.column_stack(
            [
                np.arange(sample_count, dtype=np.float64),
                np.zeros(sample_count, dtype=np.float64),
            ]
        )

    @staticmethod
    def _round_key(value: float, decimals: int = 12) -> float:
        return float(np.round(float(value), decimals))

    def _reconstruct_coordinate_vector_grid(self):
        handle = self.source_h5_file
        coordinates = np.asarray(handle["coordinates"][:], dtype=np.float64)
        vectors = np.asarray(handle["vectors"][:], dtype=np.float32)

        if coordinates.ndim != 2 or coordinates.shape[1] < 2:
            raise ValueError(f"Unsupported coordinates shape: {coordinates.shape}")
        if vectors.ndim == 1:
            vectors = vectors.reshape(-1, 1)
        if vectors.ndim != 2:
            raise ValueError(f"Unsupported vectors shape for coordinate-vector H5: {vectors.shape}")
        if len(coordinates) != len(vectors):
            raise ValueError("Coordinate and vector counts do not match.")

        x_values = np.unique(coordinates[:, 0][np.isfinite(coordinates[:, 0])])
        y_values = np.unique(coordinates[:, 1][np.isfinite(coordinates[:, 1])])
        if len(x_values) == 0 or len(y_values) == 0:
            raise ValueError("Unable to infer grid axes from coordinates.")

        x_values = np.sort(x_values.astype(np.float64))
        y_values = np.sort(y_values.astype(np.float64))[::-1]
        expected_count = len(x_values) * len(y_values)
        if expected_count != len(coordinates):
            raise ValueError(
                "Coordinate-vector H5 does not describe a complete regular grid: "
                f"{len(coordinates)} points cannot form {len(y_values)}x{len(x_values)} cells."
            )

        x_index = {self._round_key(x): idx for idx, x in enumerate(x_values)}
        y_index = {self._round_key(y): idx for idx, y in enumerate(y_values)}

        grid = np.full((len(y_values), len(x_values), vectors.shape[1]), np.nan, dtype=np.float32)
        pixel_coords = np.empty((len(coordinates), 2), dtype=np.int64)

        for row_index, (coord, vector) in enumerate(zip(coordinates, vectors)):
            x_key = self._round_key(coord[0])
            y_key = self._round_key(coord[1])
            if x_key not in x_index or y_key not in y_index:
                raise ValueError(f"Coordinate {coord!r} does not match inferred grid axes.")
            grid_row = y_index[y_key]
            grid_col = x_index[x_key]
            if not np.isnan(grid[grid_row, grid_col]).all():
                raise ValueError("Duplicate coordinate detected while reconstructing coordinate-vector grid.")
            grid[grid_row, grid_col] = vector
            pixel_coords[row_index] = [grid_row, grid_col]

        if np.isnan(grid).any():
            raise ValueError("Coordinate-vector grid reconstruction produced missing cells.")

        metadata = {
            "x_min": float(np.min(x_values)),
            "x_max": float(np.max(x_values)),
            "y_min": float(np.min(y_values)),
            "y_max": float(np.max(y_values)),
            "nx": int(len(x_values)),
            "ny": int(len(y_values)),
            "x_scale": float(np.median(np.diff(x_values))) if len(x_values) > 1 else 1.0,
            "y_scale": float(np.median(np.diff(np.sort(y_values)))) if len(y_values) > 1 else 1.0,
            "grid_layout": "coordinate_table",
        }
        return grid, pixel_coords, metadata

    @staticmethod
    def _extract_windows_from_grid(
        grid: np.ndarray,
        window_size: Tuple[int, int],
        stride: Tuple[int, int],
        *,
        enable_padding: bool = False,
        padding_mode: str = "reflect",
        reference_coords: np.ndarray | None = None,
    ):
        win_h, win_w = window_size
        stride_y, stride_x = stride
        height, width, _ = grid.shape

        patches = []
        coordinates = []

        if enable_padding:
            pad_h = win_h // 2
            pad_w = win_w // 2
            padded = np.pad(grid, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode=padding_mode)

            if reference_coords is not None and stride_y == 1 and stride_x == 1:
                for row, col in np.asarray(reference_coords, dtype=np.int64):
                    patch = padded[row : row + win_h, col : col + win_w, :]
                    if patch.shape[:2] != (win_h, win_w):
                        continue
                    patches.append(np.transpose(patch, (2, 0, 1)))
                    coordinates.append((int(row), int(col)))
            else:
                for row in range(0, height, stride_y):
                    for col in range(0, width, stride_x):
                        patch = padded[row : row + win_h, col : col + win_w, :]
                        if patch.shape[:2] != (win_h, win_w):
                            continue
                        patches.append(np.transpose(patch, (2, 0, 1)))
                        coordinates.append((int(row), int(col)))
            if not patches:
                raise ValueError("No patches were generated from the padded grid.")
            return np.asarray(patches, dtype=np.float32), np.asarray(coordinates, dtype=np.int64)

        for row in range(0, height - win_h + 1, stride_y):
            for col in range(0, width - win_w + 1, stride_x):
                patch = grid[row : row + win_h, col : col + win_w, :]
                if patch.shape[:2] != (win_h, win_w):
                    continue
                patches.append(np.transpose(patch, (2, 0, 1)))
                coordinates.append((int(row), int(col)))

        if not patches:
            raise ValueError("No patches were generated from the grid.")
        return np.asarray(patches, dtype=np.float32), np.asarray(coordinates, dtype=np.int64)

    def generate_patches(self, patch_size, patch_stride, enable_padding=False, padding_mode="reflect", stop_check_callback=None, progress_callback=None):
        del stop_check_callback, progress_callback

        win_h, win_w = self._normalize_window(patch_size)
        stride_y, stride_x = self._normalize_window(patch_stride)
        handle = self.source_h5_file

        if self.source_kind == "samples":
            samples = np.asarray(handle["samples"][:], dtype=np.float32)
            return self._ensure_patch_tensor(samples), self._load_coordinates(len(samples))

        if self.source_kind == "fused_features":
            features = np.asarray(handle["fused_features"][:], dtype=np.float32)
            if features.ndim == 3 and features.shape[0] > 1:
                grid = np.transpose(features, (1, 2, 0))
                return self._extract_windows_from_grid(
                    grid,
                    (win_h, win_w),
                    (stride_y, stride_x),
                    enable_padding=enable_padding,
                    padding_mode=padding_mode,
                )

            if features.ndim == 4:
                return self._ensure_patch_tensor(features), self._load_coordinates(len(features))
            if features.ndim == 3:
                return self._ensure_patch_tensor(features), self._load_coordinates(len(features))
            if features.ndim == 2:
                return self._ensure_patch_tensor(features), self._load_coordinates(len(features))
            raise ValueError(f"Unsupported fused_features shape: {features.shape}")

        if self.source_kind == "coordinate_vectors":
            grid, pixel_coords, _ = self._reconstruct_coordinate_vector_grid()
            return self._extract_windows_from_grid(
                grid,
                (win_h, win_w),
                (stride_y, stride_x),
                enable_padding=enable_padding,
                padding_mode=padding_mode,
                reference_coords=pixel_coords if enable_padding else None,
            )

        raise KeyError(
            "Compatibility PatchCreator only supports H5 files with 'samples', 'fused_features', or 'coordinates' + 'vectors'."
        )
