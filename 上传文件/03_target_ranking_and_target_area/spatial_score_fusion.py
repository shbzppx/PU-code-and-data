"""Label-free spatial score fusion shared by evaluation and prediction."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def window_average_scores(
    positions,
    scores,
    *,
    window_width: int = 1,
    window_height: int = 1,
) -> Optional[np.ndarray]:
    """Average scores of all prediction windows covering each grid cell."""
    coords = np.asarray(positions, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if (
        coords.ndim != 2
        or coords.shape[1] < 2
        or len(coords) != len(values)
        or len(values) == 0
    ):
        return None
    try:
        window_width = int(window_width or 1)
        window_height = int(window_height or window_width or 1)
    except (TypeError, ValueError):
        return None
    if window_width <= 1 and window_height <= 1:
        return values.copy()

    rounded_x = np.round(coords[:, 0], 8)
    rounded_y = np.round(coords[:, 1], 8)
    x_order = {key: idx for idx, key in enumerate(np.sort(np.unique(rounded_x)))}
    y_order = {key: idx for idx, key in enumerate(np.sort(np.unique(rounded_y)))}
    cell_to_indices: Dict[Tuple[int, int], list] = {}
    for idx, (x_key, y_key) in enumerate(zip(rounded_x, rounded_y)):
        cell = (y_order[y_key], x_order[x_key])
        cell_to_indices.setdefault(cell, []).append(idx)

    left = max(window_width // 2, 0)
    right = max(window_width - left, 1)
    lower = max(window_height // 2, 0)
    upper = max(window_height - lower, 1)
    sums = np.zeros(len(values), dtype=np.float64)
    counts = np.zeros(len(values), dtype=np.float64)

    for idx, (x_key, y_key) in enumerate(zip(rounded_x, rounded_y)):
        row = y_order[y_key]
        col = x_order[x_key]
        score = values[idx]
        if not np.isfinite(score):
            continue
        for target_row in range(row - lower, row + upper):
            for target_col in range(col - left, col + right):
                for target_idx in cell_to_indices.get((target_row, target_col), ()):
                    sums[target_idx] += score
                    counts[target_idx] += 1.0

    fused = values.copy()
    valid = counts > 0
    fused[valid] = sums[valid] / counts[valid]
    return np.clip(fused, 0.0, 1.0)


def apply_model_spatial_score_mode(
    model,
    positions,
    scores,
    *,
    window_width: int,
    window_height: int,
):
    """Apply a model's opt-in spatial score mode without changing old models."""
    requested = str(
        getattr(model, "preferred_spatial_score_mode", "center") or "center"
    ).strip().lower()
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if requested != "window_average":
        return values, "center"
    fused = window_average_scores(
        positions,
        values,
        window_width=window_width,
        window_height=window_height,
    )
    if fused is None:
        return values, "center_fallback"
    return fused, "window_average"
