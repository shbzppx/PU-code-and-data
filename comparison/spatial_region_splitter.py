"""Utilities for rectangle-based spatial region splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RectRegion:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "xmin": float(self.xmin),
            "xmax": float(self.xmax),
            "ymin": float(self.ymin),
            "ymax": float(self.ymax),
        }


def normalize_rect_region(raw_region: object, *, title: str = "region") -> RectRegion:
    if raw_region is None:
        raise ValueError(f"{title} is required.")

    if isinstance(raw_region, RectRegion):
        region = raw_region
    elif isinstance(raw_region, Mapping):
        missing = [key for key in ("xmin", "xmax", "ymin", "ymax") if key not in raw_region]
        if missing:
            raise ValueError(f"{title} is missing keys: {', '.join(missing)}")
        region = RectRegion(
            xmin=float(raw_region["xmin"]),
            xmax=float(raw_region["xmax"]),
            ymin=float(raw_region["ymin"]),
            ymax=float(raw_region["ymax"]),
        )
    else:
        values = list(raw_region or [])
        if len(values) != 4:
            raise ValueError(f"{title} must contain xmin,xmax,ymin,ymax.")
        region = RectRegion(
            xmin=float(values[0]),
            xmax=float(values[1]),
            ymin=float(values[2]),
            ymax=float(values[3]),
        )

    xmin = min(float(region.xmin), float(region.xmax))
    xmax = max(float(region.xmin), float(region.xmax))
    ymin = min(float(region.ymin), float(region.ymax))
    ymax = max(float(region.ymin), float(region.ymax))
    if not np.isfinite([xmin, xmax, ymin, ymax]).all():
        raise ValueError(f"{title} contains non-finite values.")
    return RectRegion(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)


def normalize_region_split_config(raw_config: Optional[Mapping[str, object]]) -> Optional[Dict[str, object]]:
    if not raw_config:
        return None
    train_region = normalize_rect_region(raw_config.get("train_region"), title="train_region")
    test_region = normalize_rect_region(raw_config.get("test_region"), title="test_region")
    buffer_distance = float(raw_config.get("buffer_distance", 0.0) or 0.0)
    if buffer_distance < 0:
        raise ValueError("buffer_distance must be greater than or equal to 0.")
    return {
        "train_region": train_region,
        "test_region": test_region,
        "buffer_distance": buffer_distance,
    }


def serialize_region_split_config(region_config: Optional[Mapping[str, object]]) -> Optional[Dict[str, object]]:
    normalized = normalize_region_split_config(region_config)
    if normalized is None:
        return None
    return {
        "train_region": normalized["train_region"].to_dict(),
        "test_region": normalized["test_region"].to_dict(),
        "buffer_distance": float(normalized["buffer_distance"]),
    }


def _ensure_coordinate_array(coords: object) -> np.ndarray:
    array = np.asarray(coords, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("Coordinates must be a 2D array with at least two columns.")
    return array[:, :2]


def _inside_rect_mask(coords: np.ndarray, region: RectRegion) -> np.ndarray:
    return (
        (coords[:, 0] >= float(region.xmin))
        & (coords[:, 0] <= float(region.xmax))
        & (coords[:, 1] >= float(region.ymin))
        & (coords[:, 1] <= float(region.ymax))
    )


def _distance_to_rect(coords: np.ndarray, region: RectRegion) -> np.ndarray:
    dx = np.maximum(np.maximum(float(region.xmin) - coords[:, 0], 0.0), coords[:, 0] - float(region.xmax))
    dy = np.maximum(np.maximum(float(region.ymin) - coords[:, 1], 0.0), coords[:, 1] - float(region.ymax))
    return np.sqrt(dx ** 2 + dy ** 2)


def build_coordinate_partition(
    coords: object,
    train_region: object,
    test_region: object,
    *,
    buffer_distance: float = 0.0,
) -> Dict[str, object]:
    coord_array = _ensure_coordinate_array(coords)
    train_rect = normalize_rect_region(train_region, title="train_region")
    test_rect = normalize_rect_region(test_region, title="test_region")
    buffer_distance = float(buffer_distance or 0.0)
    if buffer_distance < 0:
        raise ValueError("buffer_distance must be greater than or equal to 0.")

    train_mask = _inside_rect_mask(coord_array, train_rect)
    test_mask = _inside_rect_mask(coord_array, test_rect)
    overlap_mask = train_mask & test_mask
    if np.any(overlap_mask):
        raise ValueError("Train region and test region overlap. Please adjust the coordinate ranges.")

    gray_mask = np.zeros(len(coord_array), dtype=bool)
    if buffer_distance > 0:
        distance_to_train = _distance_to_rect(coord_array, train_rect)
        distance_to_test = _distance_to_rect(coord_array, test_rect)
        near_train = distance_to_train <= buffer_distance
        near_test = distance_to_test <= buffer_distance
        gray_mask = (
            (train_mask & near_test)
            | (test_mask & near_train)
            | ((~train_mask) & (~test_mask) & (near_train | near_test))
        )
        train_mask = train_mask & (~gray_mask)
        test_mask = test_mask & (~gray_mask)

    outside_mask = ~(train_mask | test_mask | gray_mask)
    return {
        "train_mask": train_mask,
        "test_mask": test_mask,
        "gray_mask": gray_mask,
        "outside_mask": outside_mask,
        "overlap_mask": overlap_mask,
        "train_region": train_rect.to_dict(),
        "test_region": test_rect.to_dict(),
        "buffer_distance": buffer_distance,
        "train_count": int(np.sum(train_mask)),
        "test_count": int(np.sum(test_mask)),
        "gray_count": int(np.sum(gray_mask)),
        "outside_count": int(np.sum(outside_mask)),
        "overlap_count": int(np.sum(overlap_mask)),
    }


def split_dataframe_by_regions(
    frame: Optional[pd.DataFrame],
    train_region: object,
    test_region: object,
    *,
    buffer_distance: float = 0.0,
    x_column: str = "x",
    y_column: str = "y",
) -> Dict[str, object]:
    if frame is None or len(frame) == 0:
        empty = pd.DataFrame(columns=[] if frame is None else frame.columns)
        partition = build_coordinate_partition(np.empty((0, 2), dtype=np.float64), train_region, test_region, buffer_distance=buffer_distance)
        return {
            "train": empty.copy(),
            "test": empty.copy(),
            "gray": empty.copy(),
            "outside": empty.copy(),
            "stats": partition,
        }

    coords = frame[[x_column, y_column]].to_numpy(dtype=np.float64)
    partition = build_coordinate_partition(coords, train_region, test_region, buffer_distance=buffer_distance)
    return {
        "train": frame.loc[partition["train_mask"]].reset_index(drop=True),
        "test": frame.loc[partition["test_mask"]].reset_index(drop=True),
        "gray": frame.loc[partition["gray_mask"]].reset_index(drop=True),
        "outside": frame.loc[partition["outside_mask"]].reset_index(drop=True),
        "stats": partition,
    }
