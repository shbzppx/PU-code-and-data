"""Unified independent-test SR/PAF/EI metric protocol."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


METRIC_PROTOCOL = "independent_test_v1"
THRESHOLD_STRATEGY = "max_ei"
PAF_SCOPE = "test_area_only"
THRESHOLD_RULE = "confidence > threshold"
DEFAULT_THRESHOLD_STEP = 0.01
DEFAULT_DISTANCE_THRESHOLD = 4.0


def metric_protocol_fields(
    *,
    threshold_step: float = DEFAULT_THRESHOLD_STEP,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    threshold_strategy: str = THRESHOLD_STRATEGY,
) -> Dict[str, object]:
    return {
        "metric_protocol": METRIC_PROTOCOL,
        "threshold_strategy": threshold_strategy,
        "threshold_range": [0.0, 1.0],
        "threshold_step": float(threshold_step),
        "threshold_rule": THRESHOLD_RULE,
        "paf_scope": PAF_SCOPE,
        "distance_threshold": float(distance_threshold),
        "primary_metric": "independent_test_ei",
        "primary_metric_formula": "EI = SR / PAF",
        "selection_order": ["EI desc", "SR desc", "PAF asc"],
    }


def threshold_candidates(
    *,
    step: float = DEFAULT_THRESHOLD_STEP,
    fixed_threshold: Optional[float] = None,
) -> np.ndarray:
    if fixed_threshold is not None:
        return np.asarray([float(fixed_threshold)], dtype=np.float64)
    step = float(step or DEFAULT_THRESHOLD_STEP)
    if step <= 0 or step > 1:
        step = DEFAULT_THRESHOLD_STEP
    values = np.arange(0.0, 1.0 + step / 2.0, step, dtype=np.float64)
    values = np.unique(np.round(np.concatenate(([0.0, 1.0], values)), 10))
    return values[(values >= 0.0) & (values <= 1.0)]


def resolve_xy_columns(frame: pd.DataFrame) -> Tuple[Optional[object], Optional[object]]:
    if frame is None or len(frame.columns) < 2:
        return None, None
    column_map = {str(col).strip().lower(): col for col in frame.columns}
    x_keys = (
        "x",
        "coord_x",
        "point_x",
        "east",
        "easting",
        "longitude",
        "东坐标",
        "横坐标",
        "x坐标",
    )
    y_keys = (
        "y",
        "coord_y",
        "point_y",
        "north",
        "northing",
        "latitude",
        "北坐标",
        "纵坐标",
        "y坐标",
    )
    x_col = next((column_map[key] for key in x_keys if key in column_map), None)
    y_col = next((column_map[key] for key in y_keys if key in column_map), None)
    if (x_col is None or y_col is None) and len(frame.columns) >= 2:
        x_col = frame.columns[0]
        y_col = frame.columns[1]
    return x_col, y_col


def mineral_coordinates(mineral_points_df) -> np.ndarray:
    if mineral_points_df is None or len(mineral_points_df) == 0:
        return np.empty((0, 2), dtype=np.float64)
    frame = mineral_points_df.copy()
    x_col, y_col = resolve_xy_columns(frame)
    if x_col is None or y_col is None:
        return np.empty((0, 2), dtype=np.float64)
    coords = frame[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if coords.empty:
        return np.empty((0, 2), dtype=np.float64)
    return coords.to_numpy(dtype=np.float64)


def coordinate_membership_mask(
    prediction_positions: Optional[np.ndarray],
    member_positions: Optional[np.ndarray],
    *,
    decimals: int = 6,
) -> Optional[np.ndarray]:
    if prediction_positions is None or member_positions is None:
        return None
    predictions = np.asarray(prediction_positions, dtype=np.float64)
    members = np.asarray(member_positions, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[1] < 2 or members.ndim != 2 or members.shape[1] < 2:
        return None
    if len(predictions) == 0:
        return np.zeros(0, dtype=bool)
    if len(members) == 0:
        return np.zeros(len(predictions), dtype=bool)
    member_keys = {
        (round(float(x), decimals), round(float(y), decimals))
        for x, y in members[:, :2]
    }
    return np.asarray(
        [
            (round(float(x), decimals), round(float(y), decimals)) in member_keys
            for x, y in predictions[:, :2]
        ],
        dtype=bool,
    )


def build_test_area_mask(
    probability_map: np.ndarray,
    metadata: Optional[Dict[str, object]] = None,
    *,
    test_area_mask: Optional[np.ndarray] = None,
    test_positions: Optional[np.ndarray] = None,
    test_indices: Optional[Iterable[int]] = None,
) -> np.ndarray:
    probability_map = np.asarray(probability_map)
    shape = tuple(probability_map.shape)
    if len(shape) != 2:
        raise ValueError("probability_map must be a 2D array.")

    if test_area_mask is not None:
        mask = np.asarray(test_area_mask).astype(bool)
        if mask.shape == shape:
            return mask
        if mask.size == probability_map.size:
            return mask.reshape(shape)
        raise ValueError(
            f"test_area_mask shape mismatch: mask={mask.shape}, probability_map={shape}"
        )

    if test_positions is not None:
        positions = np.asarray(test_positions, dtype=np.float64)
        mask = np.zeros(shape, dtype=bool)
        if positions.ndim == 2 and positions.shape[1] >= 2 and len(positions) > 0:
            rows, cols = positions_to_pixel_indices(positions[:, :2], metadata, shape)
            valid = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
            mask[rows[valid], cols[valid]] = True
        return mask

    if test_indices is not None:
        indices = np.asarray(list(test_indices), dtype=np.int64).reshape(-1)
        mask = np.zeros(probability_map.size, dtype=bool)
        valid = indices[(indices >= 0) & (indices < mask.size)]
        mask[valid] = True
        return mask.reshape(shape)

    raise ValueError("Independent-test metrics require test_area_mask, test_positions, or test_indices.")


def positions_to_pixel_indices(
    positions: np.ndarray,
    metadata: Optional[Dict[str, object]],
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    height, width = int(shape[0]), int(shape[1])
    meta = metadata or {}
    try:
        x_min = float(meta["x_min"])
        x_max = float(meta["x_max"])
        y_min = float(meta["y_min"])
        y_max = float(meta["y_max"])
        nx = int(meta.get("nx", width) or width)
        ny = int(meta.get("ny", height) or height)
    except (KeyError, TypeError, ValueError):
        cols = np.rint(positions[:, 0]).astype(np.int64)
        rows = np.rint(positions[:, 1]).astype(np.int64)
        return rows, cols

    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span == 0 or y_span == 0:
        cols = np.rint(positions[:, 0]).astype(np.int64)
        rows = np.rint(positions[:, 1]).astype(np.int64)
        return rows, cols

    cols = np.rint((positions[:, 0] - x_min) / x_span * max(nx - 1, 1)).astype(np.int64)
    rows = np.rint((y_max - positions[:, 1]) / y_span * max(ny - 1, 1)).astype(np.int64)
    if nx != width and nx > 1:
        cols = np.rint(cols / max(nx - 1, 1) * max(width - 1, 1)).astype(np.int64)
    if ny != height and ny > 1:
        rows = np.rint(rows / max(ny - 1, 1) * max(height - 1, 1)).astype(np.int64)
    return rows, cols


def grid_positions_from_probability_map(
    probability_map: np.ndarray,
    metadata: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    probability_map = np.asarray(probability_map)
    height, width = probability_map.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    meta = metadata or {}
    try:
        x_min = float(meta["x_min"])
        x_max = float(meta["x_max"])
        y_min = float(meta["y_min"])
        y_max = float(meta["y_max"])
    except (KeyError, TypeError, ValueError):
        return np.column_stack([xx.reshape(-1), yy.reshape(-1)]).astype(np.float64)

    x_values = x_min + xx / max(width - 1, 1) * (x_max - x_min)
    y_values = y_max - yy / max(height - 1, 1) * (y_max - y_min)
    return np.column_stack([x_values.reshape(-1), y_values.reshape(-1)]).astype(np.float64)


def deposit_hit_stats(
    selected_positions: np.ndarray,
    deposit_coords: np.ndarray,
    *,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> Tuple[int, np.ndarray, list]:
    selected_positions = np.asarray(selected_positions, dtype=np.float64)
    deposit_coords = np.asarray(deposit_coords, dtype=np.float64)
    total_deposits = len(deposit_coords)
    if total_deposits == 0:
        return 0, np.empty(0, dtype=np.float64), []
    if len(selected_positions) == 0:
        min_distances = np.full(total_deposits, np.inf, dtype=np.float64)
        return 0, min_distances, [False] * total_deposits

    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception:
        cKDTree = None
    if cKDTree is not None and len(selected_positions) >= 10:
        tree = cKDTree(selected_positions[:, :2])
        min_distances, _ = tree.query(deposit_coords[:, :2], k=1)
    else:
        min_distances = np.empty(total_deposits, dtype=np.float64)
        for index, deposit_coord in enumerate(deposit_coords[:, :2]):
            distances = np.sqrt(np.sum((selected_positions[:, :2] - deposit_coord) ** 2, axis=1))
            min_distances[index] = distances.min() if len(distances) else np.inf
    hit_mask = min_distances <= float(distance_threshold)
    return int(np.sum(hit_mask)), min_distances, hit_mask.tolist()


def independent_metric_rank(row: Dict[str, object]) -> Tuple[float, float, float, float, float]:
    threshold = float(row.get("threshold", 0.0) or 0.0)
    return (
        round(float(row.get("test_ei", row.get("ei", 0.0)) or 0.0), 12),
        round(float(row.get("test_sr", row.get("sr", 0.0)) or 0.0), 12),
        -round(float(row.get("test_paf", row.get("pa", row.get("paf", 0.0))) or 0.0), 12),
        -abs(threshold - 0.5),
        -threshold,
    )


def evaluate_independent_test_metrics(
    probability_map: np.ndarray,
    metadata: Optional[Dict[str, object]],
    test_minerals_df,
    *,
    test_area_mask: Optional[np.ndarray] = None,
    test_positions: Optional[np.ndarray] = None,
    test_indices: Optional[Iterable[int]] = None,
    threshold_step: float = DEFAULT_THRESHOLD_STEP,
    fixed_threshold: Optional[float] = None,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> Dict[str, object]:
    probability_map = np.asarray(probability_map, dtype=np.float64)
    if probability_map.ndim != 2:
        raise ValueError("probability_map must be a 2D array.")

    area_mask = build_test_area_mask(
        probability_map,
        metadata,
        test_area_mask=test_area_mask,
        test_positions=test_positions,
        test_indices=test_indices,
    )
    flat_area_mask = area_mask.reshape(-1)
    test_area_count = int(np.sum(flat_area_mask))
    if test_area_count <= 0:
        raise ValueError("Independent test area is empty; cannot compute PAF.")

    deposit_coords = mineral_coordinates(test_minerals_df)
    total_deposits = int(len(deposit_coords))
    if total_deposits <= 0:
        raise ValueError("Independent test minerals are empty; cannot compute SR.")

    flat_probabilities = probability_map.reshape(-1)
    grid_positions = grid_positions_from_probability_map(probability_map, metadata)
    rows = []
    best_row = None
    for threshold in threshold_candidates(step=threshold_step, fixed_threshold=fixed_threshold):
        selected_mask = flat_area_mask & (flat_probabilities > float(threshold))
        selected_positions = grid_positions[selected_mask]
        high_count = int(np.sum(selected_mask))
        paf = float(high_count / test_area_count) if test_area_count else 0.0
        hit_count, min_distances, hit_status = deposit_hit_stats(
            selected_positions,
            deposit_coords,
            distance_threshold=distance_threshold,
        )
        sr = float(hit_count / total_deposits) if total_deposits else 0.0
        ei = float(sr / paf) if paf > 0 else 0.0
        row = {
            "threshold": float(threshold),
            "test_sr": sr,
            "test_paf": paf,
            "test_ei": ei,
            "sr": sr,
            "pa": paf,
            "paf": paf,
            "ei": ei,
            "test_detected_count": int(hit_count),
            "test_mineral_count": int(total_deposits),
            "hit_deposits": int(hit_count),
            "total_deposits": int(total_deposits),
            "high_potential_count": int(high_count),
            "test_area_count": int(test_area_count),
            "min_distances": min_distances,
            "hit_status": hit_status,
        }
        rows.append(row)
        if best_row is None or independent_metric_rank(row) > independent_metric_rank(best_row):
            best_row = row

    if best_row is None:
        raise RuntimeError("Threshold scan failed to produce independent-test metrics.")

    best = dict(best_row)
    distances = np.asarray(best.get("min_distances", []), dtype=np.float64)
    finite_distances = distances[np.isfinite(distances)]
    best["mean_min_distance"] = float(np.mean(finite_distances)) if len(finite_distances) else float("inf")
    best["median_min_distance"] = float(np.median(finite_distances)) if len(finite_distances) else float("inf")
    best["test_area_mask"] = area_mask
    best.update(
        metric_protocol_fields(
            threshold_step=threshold_step,
            distance_threshold=distance_threshold,
            threshold_strategy="fixed" if fixed_threshold is not None else THRESHOLD_STRATEGY,
        )
    )

    json_rows = []
    for row in rows:
        public_row = {key: value for key, value in row.items() if key not in {"min_distances", "hit_status"}}
        public_row.update(
            metric_protocol_fields(
                threshold_step=threshold_step,
                distance_threshold=distance_threshold,
                threshold_strategy="fixed" if fixed_threshold is not None else THRESHOLD_STRATEGY,
            )
        )
        json_rows.append(public_row)

    return {
        "best": best,
        "threshold_curve": json_rows,
        "src_curve": [{"threshold": float(row["threshold"]), "sr": float(row["test_sr"])} for row in json_rows],
        "pac_curve": [{"threshold": float(row["threshold"]), "paf": float(row["test_paf"])} for row in json_rows],
        "test_area_mask": area_mask,
        **metric_protocol_fields(
            threshold_step=threshold_step,
            distance_threshold=distance_threshold,
            threshold_strategy="fixed" if fixed_threshold is not None else THRESHOLD_STRATEGY,
        ),
    }


def detection_from_metric_row(row: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not row:
        return None
    return {
        "detection_rate": float(row.get("test_sr", row.get("sr", 0.0)) or 0.0),
        "avg_detection_rate": float(row.get("test_sr", row.get("sr", 0.0)) or 0.0),
        "detection_ratio": float(row.get("test_sr", row.get("sr", 0.0)) or 0.0),
        "mineral_count": int(row.get("test_mineral_count", row.get("total_deposits", 0)) or 0),
        "detected_count": int(row.get("test_detected_count", row.get("hit_deposits", 0)) or 0),
        "avg_probability": None,
        "threshold": float(row.get("threshold", 0.0) or 0.0),
        "metric_protocol": METRIC_PROTOCOL,
        "paf_scope": PAF_SCOPE,
        "threshold_rule": THRESHOLD_RULE,
        "distance_threshold": float(row.get("distance_threshold", DEFAULT_DISTANCE_THRESHOLD) or DEFAULT_DISTANCE_THRESHOLD),
        "details": [
            {
                "hit": bool(hit),
                "min_distance": float(distance) if np.isfinite(distance) else None,
            }
            for hit, distance in zip(
                list(row.get("hit_status") or []),
                np.asarray(row.get("min_distances", []), dtype=np.float64).reshape(-1),
            )
        ],
    }
