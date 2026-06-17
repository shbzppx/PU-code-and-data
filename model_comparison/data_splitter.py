"""Mineral-driven sample splitting for prebuilt H5 datasets."""

from __future__ import annotations

import h5py
import numpy as np

from ..cnn.data_loader import _patch_indices_to_geo


class MineralBasedSplitter:
    """Split prebuilt H5 samples according to train/val/test mineral groups."""

    def __init__(self, buffer_radius=500):
        self.buffer_radius = float(buffer_radius)

    def split_by_minerals(
        self,
        h5_path,
        train_minerals_df,
        val_minerals_df,
        test_minerals_df,
    ):
        with h5py.File(h5_path, "r") as handle:
            if "coordinates" not in handle:
                raise KeyError("H5 文件缺少 coordinates 数据集。")

            coords = handle["coordinates"][:]
            metadata = dict(handle["metadata"].attrs)

        sample_coords = _patch_indices_to_geo(coords, metadata)
        if sample_coords is None:
            raise ValueError("Unable to derive geographic coordinates from H5 patch origins.")

        train_mask = self._get_mineral_mask(sample_coords, train_minerals_df)
        val_mask = self._get_mineral_mask(sample_coords, val_minerals_df)
        test_mask = self._get_mineral_mask(sample_coords, test_minerals_df)

        return (
            np.where(train_mask)[0].tolist(),
            np.where(val_mask)[0].tolist(),
            np.where(test_mask)[0].tolist(),
        )

    def _find_column_alias(self, mineral_df, aliases):
        normalized_columns = {
            str(column).strip().lower(): column for column in mineral_df.columns
        }
        for alias in aliases:
            matched_column = normalized_columns.get(alias.strip().lower())
            if matched_column is not None:
                return matched_column
        return None

    def _resolve_xy_columns(self, mineral_df):
        x_column = self._find_column_alias(
            mineral_df,
            ["x", "coord_x", "point_x", "east", "easting", "东坐标", "横坐标"],
        )
        y_column = self._find_column_alias(
            mineral_df,
            ["y", "coord_y", "point_y", "north", "northing", "北坐标", "纵坐标"],
        )
        if x_column is None or y_column is None:
            raise KeyError("矿点文件缺少坐标列，请提供 X/Y 或 x/y 列。")
        return x_column, y_column

    def _get_mineral_mask(self, sample_coords, mineral_df):
        if mineral_df is None or len(mineral_df) == 0:
            return np.zeros(len(sample_coords), dtype=bool)

        x_column, y_column = self._resolve_xy_columns(mineral_df)
        mask = np.zeros(len(sample_coords), dtype=bool)
        for _, mineral in mineral_df.iterrows():
            distances = np.sqrt(
                (sample_coords[:, 0] - float(mineral[x_column])) ** 2
                + (sample_coords[:, 1] - float(mineral[y_column])) ** 2
            )
            mask |= distances < self.buffer_radius

        return mask
