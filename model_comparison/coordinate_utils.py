from __future__ import annotations

import numpy as np


class CoordinateConverter:
    """Convert between geographic coordinates and pixel coordinates."""

    def __init__(self, metadata):
        metadata = metadata or {}
        self.x_min = float(metadata["x_min"])
        self.x_max = float(metadata["x_max"])
        self.y_min = float(metadata["y_min"])
        self.y_max = float(metadata["y_max"])
        self.nx = int(metadata["nx"])
        self.ny = int(metadata["ny"])

    def pixel_to_geo(self, pixel_x, pixel_y):
        geo_x = self.x_min + (float(pixel_x) / max(self.nx - 1, 1)) * (self.x_max - self.x_min)
        geo_y = self.y_max - (float(pixel_y) / max(self.ny - 1, 1)) * (self.y_max - self.y_min)
        return geo_x, geo_y

    def geo_to_pixel(self, geo_x, geo_y):
        pixel_x = int(round((float(geo_x) - self.x_min) / (self.x_max - self.x_min) * max(self.nx - 1, 1)))
        pixel_y = int(round((self.y_max - float(geo_y)) / (self.y_max - self.y_min) * max(self.ny - 1, 1)))
        pixel_x = max(0, min(self.nx - 1, pixel_x))
        pixel_y = max(0, min(self.ny - 1, pixel_y))
        return pixel_x, pixel_y


def infer_grid_metadata_from_coordinates(coordinates):
    """Infer regular-grid metadata from a coordinate table."""

    if coordinates is None:
        return {}

    arr = np.asarray(coordinates, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) == 0:
        return {}

    x_values = np.unique(arr[:, 0][np.isfinite(arr[:, 0])])
    y_values = np.unique(arr[:, 1][np.isfinite(arr[:, 1])])
    if len(x_values) == 0 or len(y_values) == 0:
        return {}

    x_values.sort()
    y_values.sort()
    x_step = float(np.median(np.diff(x_values))) if len(x_values) > 1 else 1.0
    y_step = float(np.median(np.diff(y_values))) if len(y_values) > 1 else 1.0

    return {
        "x_min": float(x_values[0]),
        "x_max": float(x_values[-1]),
        "y_min": float(y_values[0]),
        "y_max": float(y_values[-1]),
        "nx": int(len(x_values)),
        "ny": int(len(y_values)),
        "x_scale": x_step,
        "y_scale": y_step,
        "grid_layout": "coordinate_table",
    }
