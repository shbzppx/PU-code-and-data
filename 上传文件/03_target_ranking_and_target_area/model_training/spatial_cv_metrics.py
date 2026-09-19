from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

from spatial_score_fusion import window_average_scores

try:
    from model_comparison.metric_protocol import (
        DEFAULT_DISTANCE_THRESHOLD,
        DEFAULT_THRESHOLD_STEP,
        THRESHOLD_STRATEGY,
        deposit_hit_stats,
        metric_protocol_fields,
        threshold_candidates,
    )
except Exception:  # pragma: no cover - keep training usable if comparison module is unavailable
    DEFAULT_THRESHOLD_STEP = 0.01
    DEFAULT_DISTANCE_THRESHOLD = 4.0
    THRESHOLD_STRATEGY = "max_ei"

    def threshold_candidates(*, step=DEFAULT_THRESHOLD_STEP, fixed_threshold=None):
        if fixed_threshold is not None:
            return np.asarray([float(fixed_threshold)], dtype=np.float64)
        step = float(step or DEFAULT_THRESHOLD_STEP)
        if step <= 0 or step > 1:
            step = DEFAULT_THRESHOLD_STEP
        values = np.arange(0.0, 1.0 + step / 2.0, step, dtype=np.float64)
        return np.unique(np.round(np.concatenate(([0.0, 1.0], values)), 10))

    def deposit_hit_stats(selected_positions, deposit_coords, *, distance_threshold=DEFAULT_DISTANCE_THRESHOLD):
        selected_positions = np.asarray(selected_positions, dtype=np.float64)
        deposit_coords = np.asarray(deposit_coords, dtype=np.float64)
        if len(deposit_coords) == 0:
            return 0, np.empty(0, dtype=np.float64), []
        if len(selected_positions) == 0:
            return 0, np.full(len(deposit_coords), np.inf, dtype=np.float64), [False] * len(deposit_coords)
        min_distances = np.empty(len(deposit_coords), dtype=np.float64)
        for index, deposit_coord in enumerate(deposit_coords[:, :2]):
            distances = np.sqrt(np.sum((selected_positions[:, :2] - deposit_coord) ** 2, axis=1))
            min_distances[index] = distances.min() if len(distances) else np.inf
        hit_mask = min_distances <= float(distance_threshold)
        return int(np.sum(hit_mask)), min_distances, hit_mask.tolist()

    def metric_protocol_fields(*, threshold_step=DEFAULT_THRESHOLD_STEP, distance_threshold=DEFAULT_DISTANCE_THRESHOLD, threshold_strategy=THRESHOLD_STRATEGY):
        return {
            "metric_protocol": "independent_test_v1",
            "threshold_strategy": threshold_strategy,
            "threshold_range": [0.0, 1.0],
            "threshold_step": float(threshold_step),
            "threshold_rule": "confidence > threshold",
            "paf_scope": "test_area_only",
            "distance_threshold": float(distance_threshold),
            "primary_metric": "independent_test_ei",
            "primary_metric_formula": "EI = SR / PAF",
            "selection_order": ["EI desc", "SR desc", "PAF asc"],
        }


def build_spatial_cv_context(normalization_params: Optional[Dict[str, object]]) -> Dict[str, np.ndarray]:
    if not isinstance(normalization_params, dict):
        return {}
    positions = normalization_params.get("train_positions")
    mineral_ids = normalization_params.get("train_mineral_ids")
    mineral_positions = normalization_params.get("train_mineral_positions")
    if positions is None or mineral_ids is None or mineral_positions is None:
        return {}

    positions = np.asarray(positions, dtype=np.float64)
    mineral_ids = np.asarray(mineral_ids, dtype=np.int64).reshape(-1)
    mineral_positions = np.asarray(mineral_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2:
        return {}
    if len(positions) != len(mineral_ids):
        return {}
    if mineral_positions.ndim != 2 or mineral_positions.shape[1] < 2 or len(mineral_positions) == 0:
        return {}
    context = {
        "train_positions": positions[:, :2],
        "train_mineral_ids": mineral_ids,
        "train_mineral_positions": mineral_positions[:, :2],
    }
    split_summary = normalization_params.get("split_summary") if isinstance(normalization_params.get("split_summary"), dict) else {}
    window_width = split_summary.get("window_width", split_summary.get("patch_size"))
    window_height = split_summary.get("window_height", split_summary.get("patch_size", window_width))
    try:
        if window_width is not None:
            context["window_width"] = int(window_width)
        if window_height is not None:
            context["window_height"] = int(window_height)
    except (TypeError, ValueError):
        pass
    area_positions = normalization_params.get("spatial_cv_area_positions")
    area_mineral_ids = normalization_params.get("spatial_cv_area_mineral_ids")
    area_features = normalization_params.get("spatial_cv_area_features")
    if area_positions is not None and area_mineral_ids is not None:
        area_positions = np.asarray(area_positions, dtype=np.float64)
        area_mineral_ids = np.asarray(area_mineral_ids, dtype=np.int64).reshape(-1)
        if area_positions.ndim == 2 and area_positions.shape[1] >= 2 and len(area_positions) == len(area_mineral_ids):
            context["area_positions"] = area_positions[:, :2]
            context["area_mineral_ids"] = area_mineral_ids
            if area_features is not None and len(area_features) == len(area_positions):
                area_features = np.asarray(area_features)
                context["area_features"] = area_features
                if "window_width" not in context and area_features.ndim >= 4:
                    context["window_height"] = int(area_features.shape[-2])
                    context["window_width"] = int(area_features.shape[-1])
    return context


def _metric_rank(row: Dict[str, object]) -> tuple:
    threshold = float(row.get("threshold", 0.0) or 0.0)
    return (
        round(float(row.get("val_ei", row.get("ei", 0.0)) or 0.0), 12),
        round(float(row.get("val_sr", row.get("sr", 0.0)) or 0.0), 12),
        -round(float(row.get("val_paf", row.get("paf", 0.0)) or 0.0), 12),
        -abs(threshold - 0.5),
        -threshold,
    )


def _parse_area_fractions(area_fractions) -> list:
    if area_fractions is None:
        return []
    if isinstance(area_fractions, str):
        parts = [p.strip() for p in area_fractions.replace(";", ",").split(",") if p.strip()]
        values = []
        for part in parts:
            try:
                values.append(float(part))
            except (TypeError, ValueError):
                continue
    else:
        try:
            values = [float(v) for v in list(area_fractions)]
        except (TypeError, ValueError):
            return []
    cleaned = []
    for value in values:
        if not np.isfinite(value):
            continue
        # allow 5 or 0.05
        frac = float(value) / 100.0 if float(value) > 1.0 else float(value)
        if 0.0 < frac <= 1.0:
            cleaned.append(frac)
    # unique preserve order
    out = []
    for frac in cleaned:
        if not any(abs(frac - x) < 1e-12 for x in out):
            out.append(frac)
    return out


def _row_from_selection(
    *,
    selected_mask: np.ndarray,
    area_positions: np.ndarray,
    deposit_positions: np.ndarray,
    area_count: int,
    mineral_count: int,
    distance_threshold: float,
    threshold: float,
    target_paf: Optional[float] = None,
) -> Dict[str, object]:
    selected_positions = area_positions[selected_mask]
    high_count = int(np.sum(selected_mask))
    paf = float(high_count / area_count) if area_count else 0.0
    hit_count, min_distances, hit_status = deposit_hit_stats(
        selected_positions,
        deposit_positions,
        distance_threshold=distance_threshold,
    )
    sr = float(hit_count / mineral_count) if mineral_count else 0.0
    ei = float(sr / paf) if paf > 0 else 0.0
    row = {
        "threshold": float(threshold),
        "val_sr": sr,
        "val_paf": paf,
        "val_ei": ei,
        "sr": sr,
        "paf": paf,
        "ei": ei,
        "val_detected_count": int(hit_count),
        "val_mineral_count": int(mineral_count),
        "high_potential_count": int(high_count),
        "val_area_count": int(area_count),
        "min_distances": min_distances,
        "hit_status": hit_status,
    }
    if target_paf is not None:
        row["target_paf"] = float(target_paf)
        row["target_paf_pct"] = float(target_paf * 100.0)
    return row


def _evaluate_fixed_area_rows(
    probabilities: np.ndarray,
    area_positions: np.ndarray,
    deposit_positions: np.ndarray,
    *,
    area_fractions: Iterable[float],
    distance_threshold: float,
    primary_area_fraction: Optional[float] = None,
) -> Optional[Dict[str, object]]:
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    area_count = int(len(area_positions))
    mineral_count = int(len(deposit_positions))
    if len(probabilities) != area_count or area_count <= 0 or mineral_count <= 0:
        return None
    fractions = _parse_area_fractions(area_fractions)
    if not fractions:
        return None
    primary = float(primary_area_fraction) if primary_area_fraction is not None else fractions[0]
    if primary > 1.0:
        primary = primary / 100.0
    # nearest listed fraction as primary
    primary = min(fractions, key=lambda x: abs(x - primary))

    area_xy = np.asarray(area_positions, dtype=np.float64)[:, :2]
    deposit_xy = np.asarray(deposit_positions, dtype=np.float64)[:, :2]
    nearest_area_indices = []
    nearest_area_distances = []
    for deposit_xy_row in deposit_xy:
        distances = np.sqrt(np.sum((area_xy - deposit_xy_row) ** 2, axis=1))
        nearest_index = int(np.argmin(distances))
        nearest_area_indices.append(nearest_index)
        nearest_area_distances.append(float(distances[nearest_index]))
    nearest_area_indices = np.asarray(nearest_area_indices, dtype=np.int64)
    nearest_area_distances = np.asarray(nearest_area_distances, dtype=np.float64)

    order = np.argsort(-probabilities, kind="mergesort")
    rows = []
    primary_row = None
    for target_paf in fractions:
        k = int(round(area_count * float(target_paf)))
        k = max(1, min(area_count, k))
        selected_mask = np.zeros(area_count, dtype=bool)
        selected_mask[order[:k]] = True
        threshold = float(probabilities[order[k - 1]]) if k > 0 else 1.0
        cutoff_tie_mask = probabilities == threshold
        cutoff_tie_count = int(np.sum(cutoff_tie_mask))
        cutoff_tie_selected_count = int(np.sum(cutoff_tie_mask & selected_mask))
        row = _row_from_selection(
            selected_mask=selected_mask,
            area_positions=area_positions,
            deposit_positions=deposit_positions,
            area_count=area_count,
            mineral_count=mineral_count,
            distance_threshold=distance_threshold,
            threshold=threshold,
            target_paf=float(target_paf),
        )
        row["cutoff_tie_count"] = cutoff_tie_count
        row["cutoff_tie_selected_count"] = cutoff_tie_selected_count
        row["cutoff_tie_fraction"] = float(cutoff_tie_count / area_count)
        row["cutoff_tie_budget_ratio"] = float(cutoff_tie_count / k)
        row["cutoff_tie_break_rule"] = "stable_input_order_mergesort"

        center_selected = selected_mask[nearest_area_indices]
        center_sr = float(np.mean(center_selected))
        center_ei = float(center_sr / row["val_paf"]) if row["val_paf"] > 0 else 0.0
        strict_count = int(np.sum(probabilities > threshold))
        needed_from_tie = max(0, min(cutoff_tie_count, k - strict_count))
        tie_selection_probability = (
            float(needed_from_tie / cutoff_tie_count) if cutoff_tie_count else 0.0
        )
        deposit_scores = probabilities[nearest_area_indices]
        expected_center_hits = np.where(
            deposit_scores > threshold,
            1.0,
            np.where(deposit_scores == threshold, tie_selection_probability, 0.0),
        )
        tie_expected_center_sr = float(np.mean(expected_center_hits))
        tie_expected_center_ei = (
            float(tie_expected_center_sr / row["val_paf"])
            if row["val_paf"] > 0
            else 0.0
        )
        row["center_sr_stable_tie"] = center_sr
        row["center_ei_stable_tie"] = center_ei
        row["center_detected_count_stable_tie"] = int(np.sum(center_selected))
        row["tie_expected_center_sr"] = tie_expected_center_sr
        row["tie_expected_center_ei"] = tie_expected_center_ei
        row["tie_selection_probability_at_cutoff"] = tie_selection_probability
        row["center_metric_rule"] = "nearest_area_cell; expected inclusion within cutoff ties"
        rows.append(row)
        if abs(float(target_paf) - primary) < 1e-12:
            primary_row = row
    if primary_row is None:
        primary_row = rows[0]
    result = {key: value for key, value in primary_row.items() if key not in {"min_distances", "hit_status"}}
    result["threshold_curve"] = [
        {key: value for key, value in row.items() if key not in {"min_distances", "hit_status"}}
        for row in rows
    ]
    result["area_budget_metrics"] = list(result["threshold_curve"])
    result["primary_target_paf"] = float(primary_row.get("target_paf", primary))
    result["score_unique_count"] = int(len(np.unique(probabilities)))
    result["score_min"] = float(np.min(probabilities))
    result["score_max"] = float(np.max(probabilities))
    result["score_at_one_fraction"] = float(np.mean(probabilities == 1.0))
    result["score_at_zero_fraction"] = float(np.mean(probabilities == 0.0))
    result["center_metric_nearest_distance_mean"] = float(np.mean(nearest_area_distances))
    result["center_metric_nearest_distance_max"] = float(np.max(nearest_area_distances))
    result["legacy_distance_tolerant_metric_preserved"] = True
    return result


def _unique_mineral_positions(mineral_positions: np.ndarray, mineral_ids: Iterable[int]) -> np.ndarray:
    valid_ids = np.asarray(list(mineral_ids), dtype=np.int64).reshape(-1)
    valid_ids = np.unique(valid_ids[valid_ids >= 0])
    valid_ids = valid_ids[valid_ids < len(mineral_positions)]
    if len(valid_ids) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(mineral_positions[valid_ids], dtype=np.float64)[:, :2]


def _window_average_probabilities(
    positions: np.ndarray,
    probabilities: np.ndarray,
    *,
    window_width: int = 1,
    window_height: int = 1,
) -> Optional[np.ndarray]:
    return window_average_scores(
        positions,
        probabilities,
        window_width=window_width,
        window_height=window_height,
    )


def _evaluate_probability_rows(
    probabilities: np.ndarray,
    area_positions: np.ndarray,
    deposit_positions: np.ndarray,
    *,
    threshold_step: float,
    fixed_threshold: Optional[float],
    distance_threshold: float,
    area_fractions: Optional[Iterable[float]] = None,
    primary_area_fraction: Optional[float] = None,
) -> Optional[Dict[str, object]]:
    fractions = _parse_area_fractions(area_fractions)
    if fractions:
        return _evaluate_fixed_area_rows(
            probabilities,
            area_positions,
            deposit_positions,
            area_fractions=fractions,
            distance_threshold=distance_threshold,
            primary_area_fraction=primary_area_fraction,
        )

    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    area_count = int(len(area_positions))
    mineral_count = int(len(deposit_positions))
    if len(probabilities) != area_count or area_count <= 0 or mineral_count <= 0:
        return None

    rows = []
    best_row = None
    for threshold in threshold_candidates(step=threshold_step, fixed_threshold=fixed_threshold):
        selected_mask = probabilities > float(threshold)
        row = _row_from_selection(
            selected_mask=selected_mask,
            area_positions=area_positions,
            deposit_positions=deposit_positions,
            area_count=area_count,
            mineral_count=mineral_count,
            distance_threshold=distance_threshold,
            threshold=float(threshold),
        )
        rows.append(row)
        if best_row is None or _metric_rank(row) > _metric_rank(best_row):
            best_row = row
    if best_row is None:
        return None
    result = {key: value for key, value in best_row.items() if key not in {"min_distances", "hit_status"}}
    result["threshold_curve"] = [
        {key: value for key, value in row.items() if key not in {"min_distances", "hit_status"}}
        for row in rows
    ]
    return result


def _build_max_ei_sensitivity(
    tau_result: Dict[str, object],
    probabilities: np.ndarray,
    area_positions: np.ndarray,
    deposit_positions: np.ndarray,
    *,
    distance_threshold: float,
    selection_rule: str = "max_ei_on_validation_fold",
    note: Optional[str] = None,
) -> Dict[str, object]:
    """Legacy max-EI τ metrics plus same-area tie-expected center metrics."""
    out: Dict[str, object] = {
        "selection_rule": str(selection_rule),
        "threshold": float(tau_result.get("threshold", 0.0) or 0.0),
        "val_sr": float(tau_result.get("val_sr", tau_result.get("sr", 0.0)) or 0.0),
        "val_paf": float(tau_result.get("val_paf", tau_result.get("paf", 0.0)) or 0.0),
        "val_ei": float(tau_result.get("val_ei", tau_result.get("ei", 0.0)) or 0.0),
        "val_detected_count": int(tau_result.get("val_detected_count", 0) or 0),
        "val_mineral_count": int(tau_result.get("val_mineral_count", 0) or 0),
        "note": note
        or (
            "τ selected by legacy max-EI scan; "
            "tie-expected center EI/SR reported at the same selected-area fraction"
        ),
    }
    paf = float(out["val_paf"])
    if not np.isfinite(paf) or paf <= 0.0:
        return out
    # Clip tiny float noise so fixed-area top-k stays feasible.
    paf = float(min(max(paf, 1.0 / max(len(probabilities), 1)), 1.0))
    same_area = _evaluate_fixed_area_rows(
        probabilities,
        area_positions,
        deposit_positions,
        area_fractions=[paf],
        distance_threshold=distance_threshold,
        primary_area_fraction=paf,
    )
    if same_area is None:
        return out
    for key in (
        "tie_expected_center_sr",
        "tie_expected_center_ei",
        "center_sr_stable_tie",
        "center_ei_stable_tie",
        "score_unique_count",
        "cutoff_tie_count",
        "cutoff_tie_fraction",
        "cutoff_tie_budget_ratio",
    ):
        if same_area.get(key) is not None:
            out[key] = same_area[key]
    out["same_area_paf"] = float(same_area.get("val_paf", same_area.get("paf", paf)) or paf)
    return out


def evaluate_spatial_cv_fold(
    probabilities: np.ndarray,
    val_indices: Iterable[int],
    context: Dict[str, np.ndarray],
    *,
    area_indices: Optional[Iterable[int]] = None,
    threshold_step: float = DEFAULT_THRESHOLD_STEP,
    fixed_threshold: Optional[float] = None,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    area_fractions: Optional[Iterable[float]] = None,
    primary_area_fraction: Optional[float] = None,
    primary_probability_mode: str = "center",
) -> Optional[Dict[str, object]]:
    positions = context.get("train_positions")
    mineral_ids = context.get("train_mineral_ids")
    mineral_positions = context.get("train_mineral_positions")
    if positions is None or mineral_ids is None or mineral_positions is None:
        return None

    if area_indices is not None and context.get("area_positions") is not None and context.get("area_mineral_ids") is not None:
        metric_positions = context["area_positions"]
        metric_mineral_ids = context["area_mineral_ids"]
        val_indices = np.asarray(list(area_indices), dtype=np.int64).reshape(-1)
    else:
        metric_positions = positions
        metric_mineral_ids = mineral_ids
        val_indices = np.asarray(list(val_indices), dtype=np.int64).reshape(-1)

    valid = val_indices[(val_indices >= 0) & (val_indices < len(metric_positions))]
    if len(valid) == 0:
        return None

    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(probabilities) != len(valid):
        return None

    area_positions = np.asarray(metric_positions[valid], dtype=np.float64)[:, :2]
    deposit_positions = _unique_mineral_positions(mineral_positions, metric_mineral_ids[valid])
    area_count = int(len(area_positions))
    mineral_count = int(len(deposit_positions))
    if area_count <= 0 or mineral_count <= 0:
        return None

    fractions = _parse_area_fractions(area_fractions)
    use_fixed_area = bool(fractions)

    center_result = _evaluate_probability_rows(
        probabilities,
        area_positions,
        deposit_positions,
        threshold_step=threshold_step,
        fixed_threshold=None if use_fixed_area else fixed_threshold,
        distance_threshold=distance_threshold,
        area_fractions=fractions if use_fixed_area else None,
        primary_area_fraction=primary_area_fraction,
    )
    if center_result is None:
        return None

    if use_fixed_area:
        strategy_name = "fixed_area"
        primary_name = "spatial_cv_fixed_area_ei"
        primary_paf = float(center_result.get("primary_target_paf", fractions[0]))
        note = (
            "fixed prospective-area budgets (top-k by score); "
            f"primary={primary_paf:.4f}; "
            "threshold is the score cutoff implied by the area budget"
        )
    elif fixed_threshold is not None:
        strategy_name = "fixed"
        primary_name = "spatial_cv_fixed_threshold_ei"
        note = "fixed_threshold"
    else:
        strategy_name = THRESHOLD_STRATEGY
        primary_name = "spatial_cv_max_ei"
        note = (
            "threshold selected on this validation fold by maximum EI; "
            "use fixed or fixed_area strategy for non-optimistic CV reporting"
        )

    result = dict(center_result)
    result.update(
        metric_protocol_fields(
            threshold_step=threshold_step,
            distance_threshold=distance_threshold,
            threshold_strategy=strategy_name,
        )
    )
    result["metric_protocol"] = "spatial_cv_v1"
    result["metric_scope"] = "spatial_cv_validation_fold"
    result["paf_scope"] = "validation_fold_area_only"
    result["probability_mode"] = "center"
    result["probability_mode_label"] = "中心点原值"
    result["primary_metric"] = primary_name
    result["threshold_selection_note"] = note
    result["threshold_strategy"] = strategy_name

    # Sensitivity sidecar: validation-selected τ by max EI (does not replace fixed_area primary).
    if use_fixed_area:
        tau_result = _evaluate_probability_rows(
            probabilities,
            area_positions,
            deposit_positions,
            threshold_step=threshold_step,
            fixed_threshold=None,
            distance_threshold=distance_threshold,
            area_fractions=None,
            primary_area_fraction=None,
        )
        if tau_result is not None:
            result["validation_tau_sensitivity"] = _build_max_ei_sensitivity(
                tau_result,
                probabilities,
                area_positions,
                deposit_positions,
                distance_threshold=distance_threshold,
                selection_rule="max_ei_on_validation_fold",
            )
    for row in result.get("threshold_curve") or []:
        row["probability_mode"] = "center"
        row["probability_mode_label"] = "中心点原值"

    window_probabilities = _window_average_probabilities(
        area_positions,
        probabilities,
        window_width=int(context.get("window_width", 1) or 1),
        window_height=int(context.get("window_height", context.get("window_width", 1)) or 1),
    )
    probability_mode_metrics = [dict(result)]
    window_result = None
    if window_probabilities is not None:
        window_result = _evaluate_probability_rows(
            window_probabilities,
            area_positions,
            deposit_positions,
            threshold_step=threshold_step,
            fixed_threshold=None if use_fixed_area else fixed_threshold,
            distance_threshold=distance_threshold,
            area_fractions=fractions if use_fixed_area else None,
            primary_area_fraction=primary_area_fraction,
        )
        if window_result is not None:
            window_result.update(
                metric_protocol_fields(
                    threshold_step=threshold_step,
                    distance_threshold=distance_threshold,
                    threshold_strategy=strategy_name,
                )
            )
            window_result["metric_protocol"] = "spatial_cv_v1"
            window_result["metric_scope"] = "spatial_cv_validation_fold"
            window_result["paf_scope"] = "validation_fold_area_only"
            window_result["probability_mode"] = "window_average"
            window_result["probability_mode_label"] = "窗口平均融合"
            window_result["primary_metric"] = primary_name
            window_result["threshold_selection_note"] = result["threshold_selection_note"]
            window_result["threshold_strategy"] = strategy_name
            if use_fixed_area:
                window_tau_result = _evaluate_probability_rows(
                    window_probabilities,
                    area_positions,
                    deposit_positions,
                    threshold_step=threshold_step,
                    fixed_threshold=None,
                    distance_threshold=distance_threshold,
                    area_fractions=None,
                    primary_area_fraction=None,
                )
                if window_tau_result is not None:
                    window_result["validation_tau_sensitivity"] = _build_max_ei_sensitivity(
                        window_tau_result,
                        window_probabilities,
                        area_positions,
                        deposit_positions,
                        distance_threshold=distance_threshold,
                        selection_rule="max_ei_on_validation_fold",
                    )
            for row in window_result.get("threshold_curve") or []:
                row["probability_mode"] = "window_average"
                row["probability_mode_label"] = "窗口平均融合"
            probability_mode_metrics.append(window_result)
    requested_mode = str(primary_probability_mode or "center").strip().lower()
    if requested_mode not in {"center", "window_average"}:
        raise ValueError(
            "primary_probability_mode must be center or window_average"
        )
    selected_result = dict(result)
    if requested_mode == "window_average" and window_result is not None:
        selected_result = dict(window_result)
    elif requested_mode == "window_average":
        selected_result["probability_mode_fallback"] = "center"
    selected_result["primary_probability_mode_requested"] = requested_mode
    selected_result["probability_mode_metrics"] = probability_mode_metrics
    return selected_result


def summarize_spatial_cv_metrics(fold_metrics: Iterable[Dict[str, object]]) -> Dict[str, object]:
    items = [item for item in fold_metrics if item]

    def values(key: str) -> np.ndarray:
        arr = np.asarray([float(item.get(key, 0.0) or 0.0) for item in items], dtype=np.float64)
        return arr[np.isfinite(arr)]

    summary: Dict[str, object] = {"spatial_cv_metric_fold_count": int(len(items))}
    strategies = sorted({str(item.get("threshold_strategy", "") or "") for item in items if item.get("threshold_strategy")})
    if strategies:
        summary["spatial_cv_threshold_strategy"] = strategies[0] if len(strategies) == 1 else ",".join(strategies)
    notes = sorted({str(item.get("threshold_selection_note", "") or "") for item in items if item.get("threshold_selection_note")})
    if notes:
        summary["spatial_cv_threshold_selection_note"] = notes[0] if len(notes) == 1 else "mixed"
    threshold_values = values("threshold")
    summary["cv_threshold_values"] = threshold_values.tolist()
    summary["cv_threshold_mean"] = float(np.mean(threshold_values)) if len(threshold_values) else None
    summary["cv_threshold_std"] = float(np.std(threshold_values, ddof=0)) if len(threshold_values) else None
    for public_name, key in (("sr", "val_sr"), ("paf", "val_paf"), ("ei", "val_ei")):
        arr = values(key)
        mean = float(np.mean(arr)) if len(arr) else None
        std = float(np.std(arr, ddof=0)) if len(arr) else None
        summary[f"cv_{public_name}_values"] = arr.tolist()
        summary[f"cv_{public_name}_mean"] = mean
        summary[f"cv_{public_name}_std"] = std
        summary[f"cv_{public_name}_mean_std"] = "" if mean is None or std is None else f"{mean:.4f} ± {std:.4f}"

    for public_name, key in (
        ("center_sr_stable_tie", "center_sr_stable_tie"),
        ("center_ei_stable_tie", "center_ei_stable_tie"),
        ("tie_expected_center_sr", "tie_expected_center_sr"),
        ("tie_expected_center_ei", "tie_expected_center_ei"),
    ):
        arr = values(key)
        mean = float(np.mean(arr)) if len(arr) else None
        std = float(np.std(arr, ddof=0)) if len(arr) else None
        summary[f"cv_{public_name}_values"] = arr.tolist()
        summary[f"cv_{public_name}_mean"] = mean
        summary[f"cv_{public_name}_std"] = std
        summary[f"cv_{public_name}_mean_std"] = "" if mean is None or std is None else f"{mean:.4f} ± {std:.4f}"

    # Aggregate fixed-area budgets across folds when present.
    budget_map: Dict[str, Dict[str, list]] = {}
    for item in items:
        for row in item.get("area_budget_metrics") or item.get("threshold_curve") or []:
            if not isinstance(row, dict) or row.get("target_paf") is None:
                continue
            key = f"{float(row['target_paf']):.4f}"
            bucket = budget_map.setdefault(key, {"sr": [], "paf": [], "ei": [], "threshold": []})
            for metric_name in ("sr", "paf", "ei", "threshold"):
                val = row.get(metric_name, row.get(f"val_{metric_name}"))
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(fval):
                    bucket[metric_name].append(fval)
    if budget_map:
        area_summary = []
        for key in sorted(budget_map, key=lambda x: float(x)):
            bucket = budget_map[key]
            entry: Dict[str, object] = {"target_paf": float(key), "target_paf_pct": float(key) * 100.0}
            for metric_name in ("sr", "paf", "ei", "threshold"):
                arr = np.asarray(bucket[metric_name], dtype=np.float64)
                if len(arr) == 0:
                    continue
                mean = float(np.mean(arr))
                std = float(np.std(arr, ddof=0))
                entry[f"{metric_name}_mean"] = mean
                entry[f"{metric_name}_std"] = std
                entry[f"{metric_name}_mean_std"] = f"{mean:.4f} ± {std:.4f}"
                entry[f"{metric_name}_values"] = arr.tolist()
            area_summary.append(entry)
        summary["cv_area_budget_summary"] = area_summary

    # Aggregate validation-selected τ sensitivity across folds.
    tau_thresholds = []
    tau_srs, tau_pafs, tau_eis = [], [], []
    tau_tie_srs, tau_tie_eis = [], []
    tau_center_srs, tau_center_eis = [], []
    for item in items:
        sens = item.get("validation_tau_sensitivity")
        if not isinstance(sens, dict):
            continue
        try:
            tau = float(sens.get("threshold"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(tau):
            continue
        tau_thresholds.append(tau)
        for bucket, key in ((tau_srs, "val_sr"), (tau_pafs, "val_paf"), (tau_eis, "val_ei")):
            try:
                value = float(sens.get(key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                bucket.append(value)
        for bucket, key in (
            (tau_tie_srs, "tie_expected_center_sr"),
            (tau_tie_eis, "tie_expected_center_ei"),
            (tau_center_srs, "center_sr_stable_tie"),
            (tau_center_eis, "center_ei_stable_tie"),
        ):
            try:
                value = float(sens.get(key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                bucket.append(value)
    if tau_thresholds:
        tau_arr = np.asarray(tau_thresholds, dtype=np.float64)
        summary["cv_validation_tau_values"] = tau_arr.tolist()
        summary["cv_validation_tau_mean"] = float(np.mean(tau_arr))
        summary["cv_validation_tau_std"] = float(np.std(tau_arr, ddof=0))
        summary["cv_validation_tau_median"] = float(np.median(tau_arr))
        for public_name, arr in (
            ("sr", tau_srs),
            ("paf", tau_pafs),
            ("ei", tau_eis),
            ("tie_expected_center_sr", tau_tie_srs),
            ("tie_expected_center_ei", tau_tie_eis),
            ("center_sr_stable_tie", tau_center_srs),
            ("center_ei_stable_tie", tau_center_eis),
        ):
            values_arr = np.asarray(arr, dtype=np.float64)
            if len(values_arr) == 0:
                continue
            mean = float(np.mean(values_arr))
            std = float(np.std(values_arr, ddof=0))
            summary[f"cv_validation_tau_{public_name}_mean"] = mean
            summary[f"cv_validation_tau_{public_name}_std"] = std
            summary[f"cv_validation_tau_{public_name}_mean_std"] = f"{mean:.4f} ± {std:.4f}"
    return summary


def summarize_fold_series(values: Iterable[float], prefix: str) -> Dict[str, Optional[float]]:
    arr = np.asarray([float(value) for value in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return {
        f"mean_{prefix}": float(np.mean(arr)) if len(arr) else None,
        f"std_{prefix}": float(np.std(arr, ddof=0)) if len(arr) else None,
    }
