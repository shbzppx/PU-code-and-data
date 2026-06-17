"""Shared scoring and recommendation helpers for model comparison."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional

import numpy as np

from .metric_protocol import (
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_THRESHOLD_STEP,
    METRIC_PROTOCOL,
    PAF_SCOPE,
    THRESHOLD_RULE,
    THRESHOLD_STRATEGY,
    metric_protocol_fields,
)


DETECTION_THRESHOLD = 0.5
COMPOSITE_RECALL_WEIGHT = 0.7
COMPOSITE_PAF_PENALTY_WEIGHT = 0.3
PRIMARY_SELECTION_METRIC = "best_test_ei"
PRIMARY_SELECTION_LABEL = "测试集EI"


def _safe_metric_value(value, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def primary_test_ei_score(result: Optional[Dict[str, object]]) -> float:
    """Return the primary model-selection score, prioritizing independent test EI."""

    if not result:
        return 0.0
    metrics = dict(result.get("results", {}) or {})
    for key in ("best_test_ei", "test_ei", "cv_ei_mean", "val_ei", "composite_score"):
        if key in result:
            value = _safe_metric_value(result.get(key), default=None)
            if value is not None:
                return float(value)
        if key in metrics:
            value = _safe_metric_value(metrics.get(key), default=None)
            if value is not None:
                return float(value)
    return 0.0


def primary_test_ei_sort_key(result: Optional[Dict[str, object]]):
    """Sort key for descending test-EI-first model ranking."""

    result = result or {}
    metrics = dict(result.get("results", {}) or {})
    return (
        -primary_test_ei_score(result),
        -_safe_metric_value(result.get("best_test_sr", result.get("test_sr", metrics.get("test_sr"))), 0.0),
        _safe_metric_value(result.get("best_test_paf", result.get("test_paf", metrics.get("test_paf"))), 1.0),
        abs(_safe_metric_value(result.get("best_test_threshold"), 0.5) - 0.5),
        -_safe_metric_value(result.get("cv_ei_mean"), 0.0),
        -_safe_metric_value(result.get("val_ei", metrics.get("val_ei")), 0.0),
        _safe_metric_value(result.get("training_time"), 0.0),
        str(result.get("model_name") or result.get("model_type") or ""),
    )


def compute_composite_score(
    val_recall: Optional[float],
    val_paf: Optional[float] = None,
    val_f1: Optional[float] = None,
) -> float:
    """Compute the validation score as recall reward plus area penalty."""

    recall = _safe_optional_float(val_recall)
    paf = _safe_optional_float(val_paf)
    f1_score = _safe_optional_float(val_f1)

    if recall is not None and paf is not None:
        return (
            COMPOSITE_RECALL_WEIGHT * recall
            + COMPOSITE_PAF_PENALTY_WEIGHT * (1.0 - paf)
        )

    if recall is not None:
        return float(recall)

    if f1_score is not None:
        return float(f1_score)

    return 0.0


def resolve_composite_formula(
    val_recall: Optional[float],
    val_paf: Optional[float] = None,
    val_f1: Optional[float] = None,
) -> str:
    """Return a human-readable formula string for the active composite scoring strategy."""

    if _safe_optional_float(val_recall) is not None and _safe_optional_float(val_paf) is not None:
        return f"{COMPOSITE_RECALL_WEIGHT:.1f} × 验证集矿点检出率 + {COMPOSITE_PAF_PENALTY_WEIGHT:.1f} × (1 - 验证集PAF)"
    if _safe_optional_float(val_f1) is not None:
        return f"{COMPOSITE_RECALL_WEIGHT:.1f} × 验证集矿点检出率 + {COMPOSITE_PAF_PENALTY_WEIGHT:.1f} × (1 - 验证集PAF)"
    return f"{COMPOSITE_RECALL_WEIGHT:.1f} × 验证集矿点检出率 + {COMPOSITE_PAF_PENALTY_WEIGHT:.1f} × (1 - 验证集PAF)"


def get_detection_rate(detection_result: Optional[Dict[str, object]]) -> Optional[float]:
    if not detection_result:
        return None
    return float(detection_result.get("detection_rate", 0.0))


def compute_sr_paf_ei(
    labels: Iterable[object],
    predictions: Iterable[object],
) -> Dict[str, float]:
    """Compute Success Rate (SR), Predicted Area Fraction (PAF), and EI from binary arrays."""

    y_true = (np.asarray(list(labels)).reshape(-1) > 0).astype(np.int64)
    y_pred = (np.asarray(list(predictions)).reshape(-1) > 0).astype(np.int64)
    if y_true.size == 0 or y_pred.size == 0 or y_true.size != y_pred.size:
        return {"sr": 0.0, "paf": 0.0, "ei": 0.0}

    positive_count = int(np.sum(y_true > 0))
    predicted_positive_count = int(np.sum(y_pred > 0))
    true_positive_count = int(np.sum((y_true > 0) & (y_pred > 0)))
    sample_count = int(y_true.size)

    sr = float(true_positive_count / positive_count) if positive_count > 0 else 0.0
    paf = float(predicted_positive_count / sample_count) if sample_count > 0 else 0.0
    ei = float(sr / paf) if paf > 0 else 0.0
    return {"sr": sr, "paf": paf, "ei": ei}


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
) -> List[str]:
    """Generate short suggestions based on comparison results."""

def get_detection_rate(detection_result: Optional[Dict[str, object]]) -> Optional[float]:
    if not detection_result:
        return None
    return float(detection_result.get("detection_rate", 0.0))


def compute_sr_paf_ei(
    labels: Iterable[object],
    predictions: Iterable[object],
) -> Dict[str, float]:
    """Compute Success Rate (SR), Predicted Area Fraction (PAF), and EI from binary arrays."""

    y_true = (np.asarray(list(labels)).reshape(-1) > 0).astype(np.int64)
    y_pred = (np.asarray(list(predictions)).reshape(-1) > 0).astype(np.int64)
    if y_true.size == 0 or y_pred.size == 0 or y_true.size != y_pred.size:
        return {"sr": 0.0, "paf": 0.0, "ei": 0.0}

    positive_count = int(np.sum(y_true > 0))
    predicted_positive_count = int(np.sum(y_pred > 0))
    true_positive_count = int(np.sum((y_true > 0) & (y_pred > 0)))
    sample_count = int(y_true.size)

    sr = float(true_positive_count / positive_count) if positive_count > 0 else 0.0
    paf = float(predicted_positive_count / sample_count) if sample_count > 0 else 0.0
    ei = float(sr / paf) if paf > 0 else 0.0
    return {"sr": sr, "paf": paf, "ei": ei}


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
            "最佳参数命中搜索边界，建议扩大搜索范围后再进行一轮优化。"
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
                f"优化历史末段仍在提升，建议把试验次数增加到 {n_trials} 次继续搜索。"
            )
        else:
            advice.append("优化历史末段仍在提升，建议增加试验次数。")

    if not advice:
        advice.append("当前结果较稳定，建议围绕最优参数做小范围精细搜索并结合地质先验进一步验证。")

    return advice


def flatten_advice(advice: Optional[Iterable[str]]) -> str:
    items = [str(item).strip() for item in advice or [] if str(item).strip()]
    return "、".join(items)


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
