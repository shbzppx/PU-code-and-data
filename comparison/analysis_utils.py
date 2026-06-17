"""Shared scoring and recommendation helpers for model comparison."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional

import numpy as np


DETECTION_THRESHOLD = 0.5
COMPOSITE_ACCURACY_WEIGHT = 0.4
COMPOSITE_DETECTION_WEIGHT = 0.6
COMPOSITE_F1_WEIGHT = 0.6
VALIDATION_RECALL_WEIGHT = 0.7
VALIDATION_AREA_WEIGHT = 0.3


def compute_composite_score(
    val_accuracy: Optional[float],
    val_detection_rate: Optional[float],
    val_f1: Optional[float] = None,
) -> float:
    """Compute composite score with automatic fallback when mineral validation is unavailable."""

    accuracy = _safe_optional_float(val_accuracy)
    detection_rate = _safe_optional_float(val_detection_rate)
    f1_score = _safe_optional_float(val_f1)

    if detection_rate is not None:
        if accuracy is None:
            return float(detection_rate)
        return (
            COMPOSITE_ACCURACY_WEIGHT * accuracy
            + COMPOSITE_DETECTION_WEIGHT * detection_rate
        )

    if f1_score is not None:
        if accuracy is None:
            return float(f1_score)
        return (
            COMPOSITE_ACCURACY_WEIGHT * accuracy
            + COMPOSITE_F1_WEIGHT * f1_score
        )

    return float(accuracy or 0.0)


def compute_validation_score(
    val_recall: Optional[float],
    val_prediction_area_ratio: Optional[float],
) -> Optional[float]:
    """Compute the new validation score used by the results analysis view."""

    recall = _safe_optional_float(val_recall)
    area_ratio = _safe_optional_float(val_prediction_area_ratio)
    if recall is None or area_ratio is None:
        return None
    return (
        VALIDATION_RECALL_WEIGHT * recall
        + VALIDATION_AREA_WEIGHT * (1.0 - area_ratio)
    )


def compute_enrichment_index(
    detection_rate: Optional[float],
    area_ratio: Optional[float],
    *,
    epsilon: float = 1e-6,
) -> Optional[float]:
    """Compute EI as detection rate normalized by predicted area ratio."""

    detection = _safe_optional_float(detection_rate)
    area = _safe_optional_float(area_ratio)
    if detection is None or area is None:
        return None
    return float(detection / max(area, float(epsilon)))


def compute_prediction_area_ratio(
    probability_map,
    threshold: Optional[float] = None,
) -> Optional[float]:
    """Return the fraction of positive pixels after thresholding a probability map."""

    if probability_map is None:
        return None

    active_threshold = float(DETECTION_THRESHOLD if threshold is None else threshold)
    mask = np.asarray(probability_map) >= active_threshold
    total_pixels = int(mask.size)
    if total_pixels == 0:
        return 0.0
    return float(mask.sum()) / float(total_pixels)


def resolve_composite_formula(
    val_detection_rate: Optional[float],
    val_f1: Optional[float] = None,
) -> str:
    """Return a human-readable formula string for the active composite scoring strategy."""

    if _safe_optional_float(val_detection_rate) is not None:
        return (
            f"{COMPOSITE_ACCURACY_WEIGHT:.1f} × 验证准确率 + "
            f"{COMPOSITE_DETECTION_WEIGHT:.1f} × 内部验证矿点检出率"
        )
    if _safe_optional_float(val_f1) is not None:
        return (
            f"{COMPOSITE_ACCURACY_WEIGHT:.1f} × 验证准确率 + "
            f"{COMPOSITE_F1_WEIGHT:.1f} × 验证F1（无内部验证矿点评估）"
        )
    return "1.0 × 验证准确率"


def get_detection_rate(detection_result: Optional[Dict[str, object]]) -> Optional[float]:
    if not detection_result:
        return None
    return float(detection_result.get("detection_rate", 0.0))


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def format_params(params: Optional[Dict[str, object]]) -> str:
    return json.dumps(to_jsonable(params or {}), ensure_ascii=False, indent=2, sort_keys=True)


def detect_search_boundary_hits(
    param_space: Optional[Dict[str, tuple]],
    best_params: Optional[Dict[str, object]],
) -> List[str]:
    """Return readable descriptions for params that land on search boundaries."""

    if not param_space or not best_params:
        return []

    hits: List[str] = []
    for param_name, value in best_params.items():
        space_def = param_space.get(param_name)
        if not space_def:
            continue

        space_type = space_def[0]
        if space_type not in {"int", "float", "loguniform"}:
            continue

        lower_bound = space_def[1]
        upper_bound = space_def[2]

        if _is_close_to_boundary(value, lower_bound):
            hits.append(f"{param_name} 命中搜索下边界 {lower_bound}")
        elif _is_close_to_boundary(value, upper_bound):
            hits.append(f"{param_name} 命中搜索上边界 {upper_bound}")

    return hits


def build_improvement_advice(
    result: Dict[str, object],
    *,
    param_space: Optional[Dict[str, tuple]] = None,
    best_params: Optional[Dict[str, object]] = None,
    optimization_history: Optional[Iterable[Dict[str, object]]] = None,
    n_trials: Optional[int] = None,
) -> List[str]:
    """Generate actionable advice from the current metrics and search outcome."""

    advice: List[str] = []
    metrics = result.get("results", {})
    train_accuracy = _safe_float(metrics.get("train_acc"))
    val_accuracy = _safe_float(
        result.get("val_accuracy", metrics.get("val_acc"))
    )
    val_detection_rate = _safe_optional_float(
        result.get("val_mineral_detection_rate")
        or get_detection_rate(result.get("val_mineral_detection"))
    )

    boundary_hits = detect_search_boundary_hits(param_space, best_params)
    if boundary_hits:
        advice.append(
            "最优参数命中搜索边界，建议扩大搜索范围后再做一轮优化。"
        )

    if val_accuracy >= 0.8 and val_detection_rate is not None and val_detection_rate < 0.6:
        advice.append(
            "验证准确率较高但矿点检出率偏低，建议提高正类权重、强化召回导向指标或补充矿点附近样本。"
        )

    if train_accuracy - val_accuracy > 0.15:
        advice.append(
            "训练与验证差距偏大，存在过拟合迹象，建议减小模型复杂度、增加正则化或减少训练轮次。"
        )

    if _history_still_improving(optimization_history):
        if n_trials:
            advice.append(
                f"优化历史末段仍在提升，建议把试验次数从 {n_trials} 次继续增加。"
            )
        else:
            advice.append("优化历史末段仍在提升，建议增加试验次数。")

    if not advice:
        advice.append("当前结果较稳定，建议围绕最优参数做小范围精细搜索并结合地质先验进一步验证。")

    return advice


def flatten_advice(advice: Optional[Iterable[str]]) -> str:
    items = [str(item).strip() for item in advice or [] if str(item).strip()]
    return "；".join(items)


def _history_still_improving(
    optimization_history: Optional[Iterable[Dict[str, object]]],
) -> bool:
    values = [
        _safe_float(item.get("value"))
        for item in (optimization_history or [])
        if item.get("value") is not None
    ]
    if len(values) < 5:
        return False

    tail_size = max(3, len(values) // 5)
    previous_values = values[:-tail_size]
    tail_values = values[-tail_size:]
    if not previous_values:
        return False

    return max(tail_values) > max(previous_values) + 0.01


def _is_close_to_boundary(value: object, boundary: object) -> bool:
    try:
        numeric_value = float(value)
        numeric_boundary = float(boundary)
    except (TypeError, ValueError):
        return value == boundary

    tolerance = max(abs(numeric_boundary) * 0.01, 1e-12)
    return abs(numeric_value - numeric_boundary) <= tolerance


def _safe_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _safe_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
