import os
import json
import copy
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import (
    augment_training_tensor,
    instantiate_model,
    create_loss_function,
    create_data_loaders,
    is_image_model,
    is_non_neural_model,
    save_loss_diagnostics,
    training_loader_diagnostics,
)
from evaluation_metrics import compute_error, compute_pos_recall, evaluate_model
from result_artifact_policy import (
    should_skip_curve_artifacts,
    should_skip_fold_weight_artifacts,
)
from spatial_cv_metrics import (
    _build_max_ei_sensitivity,
    _unique_mineral_positions,
    build_spatial_cv_context,
    evaluate_spatial_cv_fold,
    summarize_fold_series,
    summarize_spatial_cv_metrics,
)
from visualization import (
    plot_cv_curves,
    plot_training_curves,
    save_cv_curve_artifacts,
    save_spatial_cv_metrics_xlsx,
    save_standard_curve_artifacts,
)


def _spatial_metric_fixed_threshold(args):
    strategy = str(getattr(args, "spatial_metric_threshold_strategy", "max_ei") or "max_ei").strip().lower()
    if strategy == "fixed_area":
        return None
    configured = getattr(args, "spatial_metric_fixed_threshold", None)
    if configured is not None:
        return float(configured)
    if strategy == "fixed":
        return 0.5
    return None


def _spatial_metric_area_fractions(args):
    strategy = str(getattr(args, "spatial_metric_threshold_strategy", "max_ei") or "max_ei").strip().lower()
    raw = str(getattr(args, "spatial_metric_area_fractions", "") or "").strip()
    if strategy == "fixed_area" and not raw:
        raw = "0.05,0.10,0.20,0.30"
    if strategy != "fixed_area":
        return None
    return raw


def _spatial_metric_primary_area_fraction(args):
    value = getattr(args, "spatial_metric_primary_area_fraction", 0.05)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.05
    if value > 1.0:
        value = value / 100.0
    return float(value)


def _flatten_tensor_data(data):
    if torch.is_tensor(data):
        if data.ndim > 2:
            return data.reshape(data.size(0), -1).detach().cpu().numpy()
        return data.detach().cpu().numpy()
    data = np.asarray(data)
    if data.ndim > 2:
        return data.reshape(data.shape[0], -1)
    return data


def _tensor_labels_to_numpy(labels):
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    return np.where(np.asarray(labels).reshape(-1) > 0, 1, -1).astype(np.int64)


def _build_non_neural_model(model_name, prior, input_dim, tuning_params=None, random_state=42):
    tuning_params = dict(tuning_params or {})
    if model_name == "rf":
        from model.random_forest import RandomForestBinaryClassifier

        return RandomForestBinaryClassifier(
            prior,
            input_dim,
            n_estimators=int(tuning_params.get("n_estimators", 200)),
            max_depth=tuning_params.get("max_depth"),
            random_state=int(random_state),
        )
    if model_name == "ocsvm":
        from model.one_class_svm import OneClassSVMClassifier

        return OneClassSVMClassifier(
            prior,
            input_dim,
            nu=float(tuning_params.get("nu", 0.1)),
            gamma=tuning_params.get("gamma", "scale"),
        )
    if model_name == "2step":
        from model.two_step_pu import TwoStepPULearning

        return TwoStepPULearning(
            prior=prior,
            dim=input_dim,
            spy_percentage=float(tuning_params.get("spy_percentage", 0.1)),
            threshold_percentile=float(tuning_params.get("threshold_percentile", 10)),
            n_estimators=int(tuning_params.get("n_estimators", 100)),
            random_state=int(random_state),
        )
    if model_name == "purf":
        from model.pu_random_forest import PURandomForestClassifier

        return PURandomForestClassifier(
            prior=prior,
            dim=input_dim,
            n_estimators=int(tuning_params.get("n_estimators", 200)),
            max_depth=tuning_params.get("max_depth", 12),
            random_state=int(random_state),
        )
    raise ValueError(f"Unsupported non-neural model type: {model_name}")


def _predict_non_neural_labels(model, features):
    predictions = model.predict(features)
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    return np.where(np.asarray(predictions).reshape(-1) > 0, 1, -1).astype(np.int64)


def _predict_non_neural_positive_scores(model, features):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.detach().cpu().numpy()
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return np.clip(probabilities[:, 1], 0.0, 1.0)
        if probabilities.ndim >= 1:
            return np.clip(probabilities.reshape(-1), 0.0, 1.0)
    predictions = _predict_non_neural_labels(model, features)
    return np.where(predictions > 0, 1.0, 0.0).astype(np.float64)


def _evaluate_non_neural_predictions(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    y_true = _tensor_labels_to_numpy(y_true)
    y_pred = np.where(np.asarray(y_pred).reshape(-1) > 0, 1, -1).astype(np.int64)
    accuracy = float(accuracy_score(y_true, y_pred))
    error = float(1.0 - accuracy)
    precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    return {
        "accuracy": accuracy,
        "error": error,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _save_pickle_model(model, path):
    import pickle

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def _predict_neural_positive_scores_from_tensor(model, features, batch_size, device):
    labels = torch.zeros(len(features), dtype=torch.long)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(features, dtype=torch.float32), labels),
        batch_size=int(batch_size or 32),
        shuffle=False,
    )
    model.eval()
    scores = []
    score_transform = getattr(model, "positive_scores_from_logits", None)
    with torch.no_grad():
        for data, _ in loader:
            output = model(data.to(device)).view(-1)
            positive_scores = (
                score_transform(output)
                if callable(score_transform)
                else torch.sigmoid(output)
            )
            scores.append(positive_scores.detach().cpu().numpy())
    if not scores:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(scores).astype(np.float64)


def _as_2d_positions(value):
    arr = np.asarray(value if value is not None else [], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float64)
    return arr[:, :2]


def _assign_eval_mineral_ids(positions, mineral_positions, labels):
    positions = _as_2d_positions(positions)
    mineral_positions = _as_2d_positions(mineral_positions)
    mineral_ids = np.full(len(positions), -1, dtype=np.int64)
    if len(positions) == 0 or len(mineral_positions) == 0:
        return mineral_ids

    label_arr = _tensor_labels_to_numpy(labels) if labels is not None and len(labels) == len(positions) else None
    if label_arr is not None:
        positive_indices = np.where(label_arr > 0)[0]
    else:
        positive_indices = np.empty((0,), dtype=np.int64)

    for sample_index in positive_indices:
        distances = np.sum((mineral_positions - positions[sample_index]) ** 2, axis=1)
        mineral_ids[sample_index] = int(np.argmin(distances))
    return mineral_ids


def _single_eval_spatial_context(normalization_params, X_eval, y_eval):
    if not isinstance(normalization_params, dict) or len(X_eval) == 0:
        return {}, None

    candidates = (
        (
            normalization_params.get("test_positions"),
            normalization_params.get("test_mineral_positions"),
            "external_test",
        ),
        (
            normalization_params.get("train_positions"),
            normalization_params.get("train_mineral_positions"),
            "train_or_validation",
        ),
        (
            normalization_params.get("spatial_cv_area_positions"),
            normalization_params.get("train_mineral_positions"),
            "spatial_area",
        ),
    )
    for positions, mineral_positions, scope in candidates:
        positions = _as_2d_positions(positions)
        mineral_positions = _as_2d_positions(mineral_positions)
        if len(positions) != len(X_eval) or len(mineral_positions) == 0:
            continue
        context = {
            "train_positions": positions,
            "train_mineral_ids": _assign_eval_mineral_ids(positions, mineral_positions, y_eval),
            "train_mineral_positions": mineral_positions,
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
        return context, scope
    return {}, None


def save_single_fold_spatial_eval(
    model,
    X_eval,
    y_eval,
    args,
    model_dir,
    device,
    *,
    normalization_params=None,
    history=None,
    metrics=None,
    metric_scope="single_eval",
):
    """Write cv_results.json and 空间CV评估结果.xlsx for non-CV training paths."""
    if model is None or len(X_eval) == 0:
        return ""

    context, resolved_scope = _single_eval_spatial_context(normalization_params, X_eval, y_eval)
    if not context:
        return ""

    if is_non_neural_model(args.model):
        eval_features = _flatten_tensor_data(X_eval)
        scores = _predict_non_neural_positive_scores(model, eval_features)
    else:
        scores = _predict_neural_positive_scores_from_tensor(model, X_eval, args.batchsize, device)

    metric = evaluate_spatial_cv_fold(
        scores,
        np.arange(len(scores), dtype=np.int64),
        context,
        threshold_step=float(getattr(args, "spatial_metric_threshold_step", 0.01) or 0.01),
        fixed_threshold=_spatial_metric_fixed_threshold(args),
        distance_threshold=float(getattr(args, "spatial_metric_distance_threshold", 4.0) or 4.0),
        area_fractions=_spatial_metric_area_fractions(args),
        primary_area_fraction=_spatial_metric_primary_area_fraction(args),
        primary_probability_mode=getattr(
            model, "preferred_spatial_score_mode", "center"
        ),
    )
    if metric is None:
        return ""

    metric = dict(metric)
    metric["fold"] = 1
    metric["metric_scope"] = resolved_scope or metric_scope
    spatial_summary = summarize_spatial_cv_metrics([metric])

    history = dict(history or {})
    metrics = dict(metrics or {})

    def _last_list_value(name):
        values = history.get(name) or []
        return values[-1] if values else None

    cv_results = {
        "train_losses": [history.get("train_loss", [])] if history.get("train_loss") else [],
        "val_losses": [history.get("test_loss", [])] if history.get("test_loss") else [],
        "train_errors": [history.get("train_error", [])] if history.get("train_error") else [],
        "val_errors": [history.get("test_error", [])] if history.get("test_error") else [],
        "test_losses": [] if _last_list_value("test_loss") is None else [float(_last_list_value("test_loss"))],
        "test_errors": [] if _last_list_value("test_error") is None else [float(_last_list_value("test_error"))],
        "train_recalls": [history.get("train_recall", [])] if history.get("train_recall") else [],
        "val_recalls": [history.get("test_recall", [])] if history.get("test_recall") else [],
        "test_recalls": [] if _last_list_value("test_recall") is None else [float(_last_list_value("test_recall"))],
        "avg_test_loss": metrics.get("test_loss", _last_list_value("test_loss")),
        "avg_test_error": metrics.get("test_error", _last_list_value("test_error")),
        "avg_test_recall": metrics.get("test_recall", _last_list_value("test_recall")),
        "fold_spatial_metrics": [metric],
        "cv_metric_scope": resolved_scope or metric_scope,
        "single_fold_eval": True,
        **spatial_summary,
    }
    cv_path = os.path.join(model_dir, "cv_results.json")
    os.makedirs(os.path.dirname(cv_path), exist_ok=True)
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=4, ensure_ascii=False)

    budget_rows = metric.get("area_budget_metrics") or []
    if budget_rows:
        import csv

        budget_csv = os.path.join(model_dir, "area_budget_metrics.csv")
        fieldnames = sorted({k for row in budget_rows for k in row.keys()})
        preferred = [
            "target_paf_pct",
            "target_paf",
            "sr",
            "paf",
            "ei",
            "threshold",
            "val_detected_count",
            "val_mineral_count",
            "high_potential_count",
            "val_area_count",
        ]
        ordered = [c for c in preferred if c in fieldnames] + [c for c in fieldnames if c not in preferred]
        with open(budget_csv, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ordered, extrasaction="ignore")
            writer.writeheader()
            for row in budget_rows:
                writer.writerow(row)
        print(f"固定面积档指标 CSV: {budget_csv}")

    output_path = save_spatial_cv_metrics_xlsx(cv_results, model_dir)
    if output_path:
        print(f"空间CV评估结果 Excel: {output_path}")
    return output_path


def save_outer_test_spatial_eval(
    model,
    X_test,
    y_test,
    args,
    model_dir,
    device,
    *,
    normalization_params=None,
    validation_selected_tau=None,
):
    """Evaluate the inner-selected model once on the outer geological holdout."""
    if model is None or X_test is None or len(X_test) == 0:
        return None
    context, resolved_scope = _single_eval_spatial_context(normalization_params, X_test, y_test)
    if not context or resolved_scope != "external_test":
        raise ValueError("Outer-test spatial evaluation requires external test positions and minerals.")
    if is_non_neural_model(args.model):
        scores = _predict_non_neural_positive_scores(model, _flatten_tensor_data(X_test))
    else:
        scores = _predict_neural_positive_scores_from_tensor(model, X_test, args.batchsize, device)
    metric = evaluate_spatial_cv_fold(
        scores,
        np.arange(len(scores), dtype=np.int64),
        context,
        threshold_step=float(getattr(args, "spatial_metric_threshold_step", 0.01) or 0.01),
        fixed_threshold=_spatial_metric_fixed_threshold(args),
        distance_threshold=float(getattr(args, "spatial_metric_distance_threshold", 4.0) or 4.0),
        area_fractions=_spatial_metric_area_fractions(args),
        primary_area_fraction=_spatial_metric_primary_area_fraction(args),
        primary_probability_mode=getattr(
            model, "preferred_spatial_score_mode", "center"
        ),
    )
    if metric is None:
        raise ValueError("Outer-test spatial evaluation did not produce a metric row.")
    metric = dict(metric)
    metric.update(
        {
            "metric_scope": "outer_expert_group_test_once",
            "model_selection_scope": "inner_validation_only",
            "outer_test_used_for_tuning": False,
        }
    )

    report_tau = bool(getattr(args, "report_validation_tau_sensitivity", False)) and not bool(
        getattr(args, "no_report_validation_tau_sensitivity", False)
    )
    # Default on for reviewer-primary + fixed_area unless explicitly disabled.
    if (
        not report_tau
        and bool(getattr(args, "reviewer_primary_protocol", False))
        and str(getattr(args, "spatial_metric_threshold_strategy", "")).strip().lower() == "fixed_area"
        and not bool(getattr(args, "no_report_validation_tau_sensitivity", False))
    ):
        report_tau = True
    tau = validation_selected_tau
    if tau is None:
        tau = getattr(args, "validation_selected_tau", None)
    try:
        tau = float(tau) if tau is not None else None
    except (TypeError, ValueError):
        tau = None
    if report_tau and tau is not None and np.isfinite(tau):
        tau_metric = evaluate_spatial_cv_fold(
            scores,
            np.arange(len(scores), dtype=np.int64),
            context,
            threshold_step=float(getattr(args, "spatial_metric_threshold_step", 0.01) or 0.01),
            fixed_threshold=float(tau),
            distance_threshold=float(getattr(args, "spatial_metric_distance_threshold", 4.0) or 4.0),
            area_fractions=None,
            primary_area_fraction=None,
            primary_probability_mode=getattr(
                model, "preferred_spatial_score_mode", "center"
            ),
        )
        if tau_metric is not None:
            area_positions = np.asarray(context["train_positions"], dtype=np.float64)[:, :2]
            deposit_positions = _unique_mineral_positions(
                context["train_mineral_positions"],
                context["train_mineral_ids"],
            )
            metric["validation_tau_sensitivity"] = _build_max_ei_sensitivity(
                {
                    "threshold": float(tau),
                    "val_sr": float(tau_metric.get("val_sr", 0.0) or 0.0),
                    "val_paf": float(tau_metric.get("val_paf", 0.0) or 0.0),
                    "val_ei": float(tau_metric.get("val_ei", 0.0) or 0.0),
                    "val_detected_count": int(tau_metric.get("val_detected_count", 0) or 0),
                    "val_mineral_count": int(tau_metric.get("val_mineral_count", 0) or 0),
                },
                scores,
                area_positions,
                deposit_positions,
                distance_threshold=float(
                    getattr(args, "spatial_metric_distance_threshold", 4.0) or 4.0
                ),
                selection_rule="median_inner_validation_max_ei_tau",
                note=(
                    "τ selected on inner validation folds only; applied once to outer test; "
                    "tie-expected center EI/SR reported at the same selected-area fraction"
                ),
            )
    from model_comparison.metric_protocol import wilson_binomial_ci

    for row in metric.get("area_budget_metrics") or []:
        point, low, high = wilson_binomial_ci(
            int(row.get("val_detected_count", 0) or 0),
            int(row.get("val_mineral_count", 0) or 0),
        )
        row.update(
            {
                "capture_rate": point,
                "wilson_95_ci_low": low,
                "wilson_95_ci_high": high,
            }
        )
    point, low, high = wilson_binomial_ci(
        int(metric.get("val_detected_count", 0) or 0),
        int(metric.get("val_mineral_count", 0) or 0),
    )
    metric.update(
        {
            "capture_rate": point,
            "wilson_95_ci_low": low,
            "wilson_95_ci_high": high,
        }
    )

    output_path = os.path.join(model_dir, "outer_test_spatial_metrics.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metric, handle, ensure_ascii=False, indent=2)
    budget_rows = metric.get("area_budget_metrics") or []
    if budget_rows:
        import csv

        csv_path = os.path.join(model_dir, "outer_test_area_budget_metrics.csv")
        fieldnames = sorted({key for row in budget_rows for key in row})
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(budget_rows)
    print(f"外层留出地质折空间指标: {output_path}")
    return metric


def save_train_dev_spatial_eval(
    model,
    X_train,
    y_train,
    args,
    model_dir,
    device,
    *,
    normalization_params=None,
    history=None,
):
    """Evaluate the fitted model on the outer-training partition (for train-metric HP select)."""
    if model is None or X_train is None or len(X_train) == 0:
        return None
    context, resolved_scope = _single_eval_spatial_context(normalization_params, X_train, y_train)
    history = dict(history or {})
    if not context:
        payload = {
            "metric_scope": "train_dev_selection_unavailable",
            "reason": "no_aligned_train_context",
            "train_tensor_n": int(len(X_train)),
            "plateau_best_train_loss": history.get("plateau_best_train_loss"),
            "plateau_best_epoch": history.get("plateau_best_epoch"),
        }
        output_path = os.path.join(model_dir, "train_dev_spatial_metrics.json")
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4, ensure_ascii=False, default=str)
        return payload

    if is_non_neural_model(args.model):
        scores = _predict_non_neural_positive_scores(model, _flatten_tensor_data(X_train))
    else:
        scores = _predict_neural_positive_scores_from_tensor(model, X_train, args.batchsize, device)
    if len(scores) != len(context["train_positions"]):
        payload = {
            "metric_scope": "train_dev_selection_unavailable",
            "reason": "score_len_mismatch",
            "score_n": int(len(scores)),
            "position_n": int(len(context["train_positions"])),
            "plateau_best_train_loss": history.get("plateau_best_train_loss"),
            "plateau_best_epoch": history.get("plateau_best_epoch"),
        }
        output_path = os.path.join(model_dir, "train_dev_spatial_metrics.json")
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4, ensure_ascii=False, default=str)
        return payload

    metric = evaluate_spatial_cv_fold(
        scores,
        np.arange(len(scores), dtype=np.int64),
        context,
        threshold_step=float(getattr(args, "spatial_metric_threshold_step", 0.01) or 0.01),
        fixed_threshold=_spatial_metric_fixed_threshold(args),
        distance_threshold=float(getattr(args, "spatial_metric_distance_threshold", 4.0) or 4.0),
        area_fractions=_spatial_metric_area_fractions(args),
        primary_area_fraction=_spatial_metric_primary_area_fraction(args),
        primary_probability_mode=getattr(
            model, "preferred_spatial_score_mode", "center"
        ),
    )
    if metric is None:
        return None
    metric = dict(metric)
    metric.update(
        {
            "metric_scope": "train_dev_for_hp_selection",
            "resolved_context_scope": resolved_scope,
            "model_selection_scope": "train_partition_only",
            "outer_test_used_for_tuning": False,
            "plateau_best_train_loss": history.get("plateau_best_train_loss"),
            "plateau_best_epoch": history.get("plateau_best_epoch"),
        }
    )
    output_path = os.path.join(model_dir, "train_dev_spatial_metrics.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metric, handle, indent=4, ensure_ascii=False, default=str)
    print(f"训练集空间指标（选参）: {output_path}")
    return metric


def train_non_neural_model_with_cv(
    X_train,
    y_train,
    X_test,
    y_test,
    prior,
    args,
    model_dir,
    device,
    fold_indices,
    stop_token=None,
    progress_callback=None,
    normalization_params=None,
    selection_only=False,
    preselected_inner=None,
    candidate_params=None,
):
    """Cross-validation path for RF, OCSVM, Two-Step PU and PU-RF."""
    del device
    print(f"非神经网络模型使用 {len(fold_indices)} 折交叉验证...")

    X_train_flat = _flatten_tensor_data(X_train)
    y_train_np = _tensor_labels_to_numpy(y_train)
    X_test_flat = _flatten_tensor_data(X_test) if len(X_test) > 0 else np.empty((0, X_train_flat.shape[1]))
    y_test_np = _tensor_labels_to_numpy(y_test) if len(y_test) > 0 else np.empty((0,), dtype=np.int64)
    outer_test_available = len(X_test_flat) > 0
    has_external_test = outer_test_available and not bool(
        getattr(args, "reviewer_primary_protocol", False)
    )
    input_dim = int(X_train_flat.shape[1]) if X_train_flat.ndim > 1 else 1

    cv_train_losses, cv_val_losses = [], []
    cv_train_errors, cv_val_errors = [], []
    cv_test_losses, cv_test_errors = [], []
    cv_train_recalls, cv_val_recalls, cv_test_recalls = [], [], []
    cv_spatial_metrics = []
    spatial_metric_context = build_spatial_cv_context(normalization_params)
    fold_metrics = []
    fold_prior_estimates = []
    best_model = None
    best_val_error = float("inf")
    if isinstance(preselected_inner, dict):
        cv_train_losses = preselected_inner.get("train_losses", [])
        cv_val_losses = preselected_inner.get("val_losses", [])
        cv_train_errors = preselected_inner.get("train_errors", [])
        cv_val_errors = preselected_inner.get("val_errors", [])
        cv_test_losses = preselected_inner.get("test_losses", [])
        cv_test_errors = preselected_inner.get("test_errors", [])
        cv_train_recalls = preselected_inner.get("train_recalls", [])
        cv_val_recalls = preselected_inner.get("val_recalls", [])
        cv_test_recalls = preselected_inner.get("test_recalls", [])
        cv_spatial_metrics = preselected_inner.get("fold_spatial_metrics", [])
        fold_metrics = preselected_inner.get("fold_metrics", [])
        fold_prior_estimates = preselected_inner.get("fold_prior_estimates", [])

    fold_indices_to_run = [] if isinstance(preselected_inner, dict) else fold_indices
    for fold_idx, fold in enumerate(fold_indices_to_run):
        if isinstance(fold, dict):
            train_idx = np.asarray(fold.get("train_indices", []), dtype=np.int64)
            val_idx = np.asarray(fold.get("val_indices", []), dtype=np.int64)
            area_indices = fold.get("val_area_indices")
            area_indices = None if area_indices is None else np.asarray(area_indices, dtype=np.int64)
        else:
            train_idx, val_idx = fold
            train_idx = np.asarray(train_idx, dtype=np.int64)
            val_idx = np.asarray(val_idx, dtype=np.int64)
            area_indices = None
        if stop_token is not None and stop_token.is_set():
            print("检测到停止信号，停止后续非神经网络 CV 折训练。")
            break

        print(f"\n开始训练非神经网络第 {fold_idx + 1}/{len(fold_indices)} 折...")
        fold_X_train = X_train
        fold_X_test = X_test
        if bool(getattr(args, "nested_fault_calibration", False)):
            from nested_fault_calibration import apply_nested_fault_calibration

            fault_path = str(getattr(args, "fault_lines_path", "") or "").strip()
            if not fault_path or not os.path.exists(fault_path):
                raise FileNotFoundError(
                    "已开启 --nested-fault-calibration，但未提供有效 --fault-lines-path。"
                )
            fold_X_train, _, fold_X_test, nested_meta = apply_nested_fault_calibration(
                fold_name=f"fold{fold_idx + 1}",
                output_dir=model_dir,
                fault_path=fault_path,
                X_train=X_train,
                y_train=y_train,
                train_idx=train_idx,
                val_idx=val_idx,
                X_test=None if bool(getattr(args, "reviewer_primary_protocol", False)) else (
                    X_test if outer_test_available else None
                ),
                normalization_params=normalization_params if isinstance(normalization_params, dict) else {},
                quantile=float(getattr(args, "fault_decay_quantile", 0.8) or 0.8),
                fault_ablation=str(getattr(args, "fault_ablation", "") or ""),
            )
            print(
                f"嵌套断裂标定 fold{fold_idx + 1}: "
                f"Q={nested_meta.get('quantile', 0.8):.2f}, "
                f"λ={nested_meta.get('length_m'):.1f} m, "
                f"train矿点={nested_meta.get('n_train_deposits')}"
                + (
                    f"；已丢弃距离通道，训练通道数={nested_meta.get('train_channels_after_drop')}"
                    if nested_meta.get("dropped_distance_after_nested")
                    else ""
                )
            )
            X_train_flat = _flatten_tensor_data(fold_X_train)
            if fold_X_test is not None and len(fold_X_test) > 0:
                X_test_flat = _flatten_tensor_data(fold_X_test)
            spatial_metric_context = build_spatial_cv_context(normalization_params)
            if nested_meta.get("dropped_distance_after_nested") and spatial_metric_context.get("area_features") is not None:
                from nested_fault_calibration import align_area_features_after_distance_drop

                spatial_metric_context["area_features"] = align_area_features_after_distance_drop(
                    spatial_metric_context["area_features"],
                    normalization_params if isinstance(normalization_params, dict) else {},
                )

        train_features = X_train_flat[train_idx]
        train_labels = y_train_np[train_idx]
        val_features = X_train_flat[val_idx]
        val_labels = y_train_np[val_idx]

        if bool(getattr(args, "augmentation_enabled", False)):
            train_tensor = torch.as_tensor(train_features, dtype=torch.float32)
            train_features = augment_training_tensor(
                train_tensor,
                noise_std=float(getattr(args, "augmentation_noise_std", 0.01) or 0.01),
            ).detach().cpu().numpy()

        fold_prior = float(prior)
        mode = str(getattr(args, "prior_mode", "manual") or "manual").strip().lower()
        if mode in {"estimate_en", "elkan_noto", "en", "estimate_km2", "km2"}:
            from prior_estimation import resolve_prior_triple

            groups = None
            weights = None
            if isinstance(normalization_params, dict):
                all_groups = normalization_params.get("prior_estimation_groups")
                if all_groups is not None and len(np.asarray(all_groups).reshape(-1)) == len(y_train_np):
                    groups = np.asarray(all_groups).reshape(-1)[train_idx]
                all_weights = normalization_params.get("prior_sampling_weights")
                if all_weights is not None and len(np.asarray(all_weights).reshape(-1)) == len(y_train_np):
                    weights = np.asarray(all_weights, dtype=np.float64).reshape(-1)[train_idx]
            fold_prior_meta = resolve_prior_triple(
                prior_mode=mode,
                features=train_features,
                labels=train_labels,
                groups=groups,
                sample_weights=weights,
                tuning_prior=getattr(args, "manual_prior", None),
                geological_favorable_area_ratio=getattr(args, "geological_favorable_area_ratio", None),
                random_state=int(getattr(args, "spatial_random_state", 42) or 42) + int(fold_idx),
                compare_estimators=bool(getattr(args, "reviewer_protocol", False)),
            )
            fold_prior = float(fold_prior_meta["resolved_prior"])
            fold_prior_estimates.append({"fold": int(fold_idx + 1), **fold_prior_meta})
            print(f"非神经内层第 {fold_idx + 1} 折先验（仅 fold-train）: π={fold_prior:.6f}")
        model = _build_non_neural_model(
            args.model,
            fold_prior,
            input_dim,
            tuning_params=candidate_params,
            random_state=int(getattr(args, "spatial_random_state", 42) or 42) + int(fold_idx),
        )
        model.fit(train_features, train_labels)

        train_pred = _predict_non_neural_labels(model, train_features)
        val_pred = _predict_non_neural_labels(model, val_features)
        train_metrics = _evaluate_non_neural_predictions(train_labels, train_pred)
        val_metrics = _evaluate_non_neural_predictions(val_labels, val_pred)
        spatial_metrics = None
        if spatial_metric_context:
            area_features = spatial_metric_context.get("area_features")
            if area_indices is not None and area_features is not None:
                metric_features = _flatten_tensor_data(area_features[area_indices])
                val_scores = _predict_non_neural_positive_scores(model, metric_features)
            else:
                area_indices = None
                val_scores = _predict_non_neural_positive_scores(model, val_features)
            spatial_metrics = evaluate_spatial_cv_fold(
                val_scores,
                val_idx,
                spatial_metric_context,
                area_indices=area_indices,
                threshold_step=float(getattr(args, "spatial_metric_threshold_step", 0.01) or 0.01),
                fixed_threshold=_spatial_metric_fixed_threshold(args),
                distance_threshold=float(getattr(args, "spatial_metric_distance_threshold", 4.0) or 4.0),
                area_fractions=_spatial_metric_area_fractions(args),
                primary_area_fraction=_spatial_metric_primary_area_fraction(args),
                primary_probability_mode=getattr(
                    model, "preferred_spatial_score_mode", "center"
                ),
            )
            if spatial_metrics is not None:
                spatial_metrics = dict(spatial_metrics)
                spatial_metrics["fold"] = int(fold_idx + 1)
                cv_spatial_metrics.append(spatial_metrics)

        if has_external_test:
            test_pred = _predict_non_neural_labels(model, X_test_flat)
            test_metrics = _evaluate_non_neural_predictions(y_test_np, test_pred)
        else:
            test_metrics = dict(val_metrics)
            print("全量矿点训练模式: 未保留外部测试集，本折汇总使用验证集指标。")

        train_loss = train_metrics["error"]
        val_loss = val_metrics["error"]
        test_loss = test_metrics["error"]
        cv_train_losses.append([train_loss])
        cv_val_losses.append([val_loss])
        cv_train_errors.append([train_metrics["error"]])
        cv_val_errors.append([val_metrics["error"]])
        cv_test_losses.append(test_loss)
        cv_test_errors.append(test_metrics["error"])
        cv_train_recalls.append([train_metrics["recall"]])
        cv_val_recalls.append([val_metrics["recall"]])
        cv_test_recalls.append(test_metrics["recall"])

        fold_metrics.append(
            {
                "fold": fold_idx + 1,
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "train_error": train_metrics["error"],
                "val_error": val_metrics["error"],
                "test_error": test_metrics["error"],
                "train_precision": train_metrics["precision"],
                "val_precision": val_metrics["precision"],
                "test_precision": test_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "val_recall": val_metrics["recall"],
                "test_recall": test_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "val_f1": val_metrics["f1"],
                "test_f1": test_metrics["f1"],
                "val_sr": None if spatial_metrics is None else spatial_metrics.get("val_sr"),
                "val_paf": None if spatial_metrics is None else spatial_metrics.get("val_paf"),
                "val_ei": None if spatial_metrics is None else spatial_metrics.get("val_ei"),
                "val_spatial_threshold": None if spatial_metrics is None else spatial_metrics.get("threshold"),
                "val_mineral_count": None if spatial_metrics is None else spatial_metrics.get("val_mineral_count"),
                "val_area_count": None if spatial_metrics is None else spatial_metrics.get("val_area_count"),
            }
        )

        if not should_skip_fold_weight_artifacts(args):
            _save_pickle_model(model, os.path.join(model_dir, f"model_fold{fold_idx + 1}.pkl"))
        print(
            f"第 {fold_idx + 1} 折: "
            f"val_error={val_metrics['error']:.4f}, val_recall={val_metrics['recall']:.4f}, "
            f"test_error={test_metrics['error']:.4f}, test_recall={test_metrics['recall']:.4f}"
        )

        if spatial_metrics is not None:
            print(
                f"第 {fold_idx + 1} 折空间验证: "
                f"SR={spatial_metrics['val_sr']:.4f}, "
                f"PAF={spatial_metrics['val_paf']:.4f}, "
                f"EI={spatial_metrics['val_ei']:.4f}, "
                f"threshold={spatial_metrics['threshold']:.2f}"
            )

        if progress_callback is not None:
            progress_callback(
                {
                    "mode": "cv",
                    "fold": fold_idx + 1,
                    "epoch": 1,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "train_error": float(train_metrics["error"]),
                    "val_error": float(val_metrics["error"]),
                    "train_recall": float(train_metrics["recall"]),
                    "val_recall": float(val_metrics["recall"]),
                }
            )

        if val_metrics["error"] < best_val_error:
            best_val_error = val_metrics["error"]
            best_model = model

    if not cv_test_errors:
        print("未完成有效的非神经网络交叉验证训练。")
        return None

    avg_test_loss = float(np.mean(cv_test_losses))
    avg_test_error = float(np.mean(cv_test_errors))
    avg_test_recall = float(np.mean(cv_test_recalls))
    val_summary = {}
    val_summary.update(summarize_fold_series([item["val_error"] for item in fold_metrics], "val_loss"))
    val_summary.update(summarize_fold_series([item["val_error"] for item in fold_metrics], "val_error"))
    val_summary.update(summarize_fold_series([item["val_recall"] for item in fold_metrics], "val_recall"))
    spatial_summary = summarize_spatial_cv_metrics(cv_spatial_metrics)
    if selection_only:
        return {
            "train_losses": cv_train_losses,
            "val_losses": cv_val_losses,
            "train_errors": cv_train_errors,
            "val_errors": cv_val_errors,
            "test_losses": cv_test_losses,
            "test_errors": cv_test_errors,
            "train_recalls": cv_train_recalls,
            "val_recalls": cv_val_recalls,
            "test_recalls": cv_test_recalls,
            "fold_metrics": fold_metrics,
            "fold_spatial_metrics": cv_spatial_metrics,
            "fold_prior_estimates": fold_prior_estimates,
            "mean_val_loss": val_summary.get("mean_val_loss"),
            "tie_aware_center_selection": bool(
                getattr(args, "tie_aware_center_selection", False)
            ),
            **spatial_summary,
        }

    final_train_idx = np.arange(len(X_train_flat), dtype=np.int64)
    final_X_train = X_train
    final_X_test = X_test
    final_fault_meta = None
    if bool(getattr(args, "nested_fault_calibration", False)):
        from nested_fault_calibration import apply_nested_fault_calibration

        fault_path = str(getattr(args, "fault_lines_path", "") or "").strip()
        final_X_train, _, final_X_test, final_fault_meta = apply_nested_fault_calibration(
            fold_name="outer_train_refit",
            output_dir=model_dir,
            fault_path=fault_path,
            X_train=X_train,
            y_train=y_train,
            train_idx=final_train_idx,
            val_idx=np.empty((0,), dtype=np.int64),
            X_test=X_test if outer_test_available else None,
            normalization_params=normalization_params if isinstance(normalization_params, dict) else {},
            quantile=float(getattr(args, "fault_decay_quantile", 0.8) or 0.8),
                fault_ablation=str(getattr(args, "fault_ablation", "") or ""),
        )
    final_features = _flatten_tensor_data(final_X_train)
    final_labels = _tensor_labels_to_numpy(y_train)
    final_prior = float(prior)
    final_prior_meta = None
    mode = str(getattr(args, "prior_mode", "manual") or "manual").strip().lower()
    if mode in {"estimate_en", "elkan_noto", "en", "estimate_km2", "km2"}:
        from prior_estimation import resolve_prior_triple

        groups = None
        weights = None
        if isinstance(normalization_params, dict):
            all_groups = normalization_params.get("prior_estimation_groups")
            if all_groups is not None and len(np.asarray(all_groups).reshape(-1)) == len(final_labels):
                groups = np.asarray(all_groups).reshape(-1)
            all_weights = normalization_params.get("prior_sampling_weights")
            if all_weights is not None and len(np.asarray(all_weights).reshape(-1)) == len(final_labels):
                weights = np.asarray(all_weights, dtype=np.float64).reshape(-1)
        final_prior_meta = resolve_prior_triple(
            prior_mode=mode,
            features=final_features,
            labels=final_labels,
            groups=groups,
            sample_weights=weights,
            tuning_prior=getattr(args, "manual_prior", None),
            geological_favorable_area_ratio=getattr(args, "geological_favorable_area_ratio", None),
            random_state=int(getattr(args, "spatial_random_state", 42) or 42),
            compare_estimators=bool(getattr(args, "reviewer_protocol", False)),
        )
        final_prior = float(final_prior_meta["resolved_prior"])
    best_model = _build_non_neural_model(
        args.model,
        final_prior,
        input_dim,
        tuning_params=candidate_params,
        random_state=int(getattr(args, "spatial_random_state", 42) or 42),
    )
    best_model.fit(final_features, final_labels)
    _save_pickle_model(best_model, os.path.join(model_dir, "best_model.pkl"))
    _save_pickle_model(best_model, os.path.join(model_dir, "model.pkl"))
    with open(os.path.join(model_dir, "nested_selection_refit_protocol.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "selection_scope": "inner_validation_only",
                "outer_test_used_for_selection": False,
                "selection_metric": "mean_inner_validation_error",
                "refit_scope": "complete_outer_training_partition",
                "refit_from_scratch": True,
                "refit_train_size": int(len(final_features)),
                "refit_prior": float(final_prior),
                "refit_prior_metadata": final_prior_meta,
                "refit_fault_calibration": final_fault_meta,
                "outer_evaluation_count": 1 if outer_test_available else 0,
                "selected_hyperparameters": dict(candidate_params or {}),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("\n非神经网络交叉验证汇总")
    print(f"平均测试错误率: {avg_test_error:.4f} ± {np.std(cv_test_errors):.4f}")
    print(f"平均测试召回率: {avg_test_recall:.4f} ± {np.std(cv_test_recalls):.4f}")

    if spatial_summary.get("cv_ei_mean") is not None:
        print(
            "CV 空间验证 EI: "
            f"{spatial_summary.get('cv_ei_mean'):.4f} ± {spatial_summary.get('cv_ei_std'):.4f}"
        )

    cv_results = {
        "train_losses": cv_train_losses,
        "val_losses": cv_val_losses,
        "train_errors": cv_train_errors,
        "val_errors": cv_val_errors,
        "test_losses": cv_test_losses,
        "test_errors": cv_test_errors,
        "avg_test_loss": avg_test_loss,
        "avg_test_error": avg_test_error,
        "train_recalls": cv_train_recalls,
        "val_recalls": cv_val_recalls,
        "test_recalls": cv_test_recalls,
        "avg_test_recall": avg_test_recall,
        "fold_metrics": fold_metrics,
        "fold_spatial_metrics": cv_spatial_metrics,
        "fold_prior_estimates": fold_prior_estimates,
        "final_refit_scope": "complete_outer_training_partition",
        "outer_test_used_for_selection": False,
        "selected_hyperparameters": dict(candidate_params or {}),
        "tie_aware_center_selection": bool(
            getattr(args, "tie_aware_center_selection", False)
        ),
        **val_summary,
        **spatial_summary,
        "loss_note": "Non-neural CV uses classification error as loss for reporting compatibility.",
    }
    with open(os.path.join(model_dir, "cv_results.json"), "w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=4, ensure_ascii=False)

    if not should_skip_curve_artifacts(args):
        plot_cv_curves(
            cv_val_losses,
            cv_val_errors,
            cv_val_recalls,
            cv_test_losses,
            cv_test_errors,
            cv_test_recalls,
            avg_test_loss,
            avg_test_error,
            avg_test_recall,
            os.path.join(model_dir, "cv_learning_curves.png"),
            len(cv_train_losses),
        )
        curve_artifacts = save_cv_curve_artifacts(cv_results, model_dir)
        print(f"非神经网络 CV 曲线数据已保存到: {curve_artifacts['csv']}")
        if curve_artifacts.get("spatial_cv_xlsx"):
            print(f"空间CV SR/PAF/EI Excel 已保存到: {curve_artifacts['spatial_cv_xlsx']}")
    else:
        try:
            from visualization import save_spatial_cv_metrics_xlsx

            spatial_cv_xlsx = save_spatial_cv_metrics_xlsx(cv_results, model_dir)
            if spatial_cv_xlsx:
                print(f"空间CV SR/PAF/EI Excel 已保存到: {spatial_cv_xlsx}")
        except Exception as exc:  # noqa: BLE001
            print(f"空间CV SR/PAF/EI Excel 生成失败: {exc}")
    if outer_test_available:
        save_outer_test_spatial_eval(
            best_model,
            final_X_test,
            y_test,
            args,
            model_dir,
            device="cpu",
            normalization_params=normalization_params,
            validation_selected_tau=spatial_summary.get("cv_validation_tau_median"),
        )
    return best_model

def train_model(model, train_loader, optimizer, loss_func, device, epoch=0):
    """训练模型一个epoch"""
    model.train()
    total_loss = 0
    processed_batches = 0
    requires_multi_sample_batch = any(isinstance(module, nn.BatchNorm1d) for module in model.modules())
    for batch in train_loader:
        if len(batch) == 3:
            data, target, sample_weights = batch
        else:
            data, target = batch
            sample_weights = None
        if requires_multi_sample_batch and int(data.size(0)) < 2:
            continue
        data, target = data.to(device), target.to(device)
        if sample_weights is not None:
            sample_weights = sample_weights.to(device)
        optimizer.zero_grad()
        output = model(data)
        teacher_forward = getattr(model, "teacher_forward", None)
        set_teacher_logits = getattr(loss_func, "set_teacher_logits", None)
        if callable(teacher_forward) and callable(set_teacher_logits):
            set_teacher_logits(teacher_forward(data))
        try:
            loss = loss_func(output, target, sample_weights)
        except TypeError:
            loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        update_teacher = getattr(model, "update_teacher", None)
        if callable(update_teacher):
            update_teacher(getattr(loss_func, "teacher_momentum", None))
        total_loss += loss.item()
        processed_batches += 1
    
    if processed_batches == 0:
        raise ValueError("训练批次全部小于 2，BatchNorm1d 无法工作；请减小批大小或检查数据划分。")

    # Prefer epoch-level adapters (AdaptivePU / RN wrappers); else legacy counter bump.
    end_epoch = getattr(loss_func, "end_epoch", None)
    if callable(end_epoch):
        end_epoch()
    elif hasattr(loss_func, "epoch_counter"):
        loss_func.epoch_counter += 1
    
    return total_loss / processed_batches


def build_adam_optimizer(model, args):
    """Adam with configurable weight decay (default L2 for neural PU/CNN)."""
    lr = float(getattr(args, "stepsize", 1e-4) or 1e-4)
    weight_decay = float(getattr(args, "weight_decay", 1e-4) or 0.0)
    if weight_decay < 0:
        weight_decay = 0.0
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def _subsample_eval_tensors(X, y, max_samples: int, seed: int = 19):
    """Keep epoch monitoring cheap when leave-one/basin test sets are huge."""
    n = int(len(y)) if y is not None else 0
    if n <= 0 or max_samples <= 0 or n <= max_samples:
        return X, y, False, n
    rng = np.random.default_rng(int(seed))
    y_arr = y.detach().cpu().numpy().reshape(-1) if hasattr(y, "detach") else np.asarray(y).reshape(-1)
    pos = np.where(y_arr == 1)[0]
    other = np.where(y_arr != 1)[0]
    n_pos = min(len(pos), max(1, max_samples // 10)) if len(pos) else 0
    n_other = min(len(other), max_samples - n_pos)
    parts = []
    if n_pos:
        parts.append(rng.choice(pos, size=n_pos, replace=False))
    if n_other > 0:
        parts.append(rng.choice(other, size=n_other, replace=False))
    if not parts:
        idx = rng.choice(n, size=max_samples, replace=False)
    else:
        idx = np.concatenate(parts)
        if len(idx) > max_samples:
            idx = rng.choice(idx, size=max_samples, replace=False)
    idx = np.sort(np.asarray(idx, dtype=np.int64))
    if hasattr(X, "index_select"):
        X_s = X.index_select(0, torch.as_tensor(idx, dtype=torch.long))
    else:
        X_s = X[idx]
    if hasattr(y, "index_select"):
        y_s = y.index_select(0, torch.as_tensor(idx, dtype=torch.long))
    else:
        y_s = y[idx]
    return X_s, y_s, True, n


def train_without_cv(
    X_train,
    y_train,
    X_test,
    y_test,
    prior,
    args,
    model_dir,
    device,
    stop_token=None,
    progress_callback=None,
    normalization_params=None,
    final_metric_scope="standard_training_eval",
):
    """不使用交叉验证的训练流程"""
    has_eval_set = X_test is not None and int(len(X_test)) > 0
    if not has_eval_set:
        print("最终制图训练模式: 未保留验证/测试集，训练过程仅记录训练集指标。")
    else:
        print("训练过程将同步记录测试/验证集损失、错误率与召回率曲线。")

    if bool(getattr(args, "nested_fault_calibration", False)):
        from nested_fault_calibration import apply_nested_fault_calibration

        fault_path = str(getattr(args, "fault_lines_path", "") or "").strip()
        if not fault_path or not os.path.exists(fault_path):
            raise FileNotFoundError(
                "已开启 --nested-fault-calibration，但未提供有效 --fault-lines-path。"
            )
        train_idx = np.arange(len(y_train), dtype=np.int64)
        X_train, y_train, X_test, nested_meta = apply_nested_fault_calibration(
            fold_name="holdout_train",
            output_dir=model_dir,
            fault_path=fault_path,
            X_train=X_train,
            y_train=y_train,
            train_idx=train_idx,
            val_idx=None,
            X_test=X_test if has_eval_set else None,
            normalization_params=normalization_params if isinstance(normalization_params, dict) else {},
            forbidden_deposit_xy=(
                np.asarray(normalization_params.get("test_mineral_positions"), dtype=np.float64)
                if isinstance(normalization_params, dict) and normalization_params.get("test_mineral_positions") is not None
                else None
            ),
            quantile=float(getattr(args, "fault_decay_quantile", 0.8) or 0.8),
                fault_ablation=str(getattr(args, "fault_ablation", "") or ""),
        )
        # all_minerals：标定可能返回 X_test=None，恢复为空张量供下游 DataLoader 使用
        if not has_eval_set or X_test is None:
            empty_x_shape = (0,) + tuple(X_train.shape[1:])
            X_test = torch.zeros(empty_x_shape, dtype=X_train.dtype, device=X_train.device if hasattr(X_train, "device") else "cpu")
            y_test = torch.zeros((0,), dtype=y_train.dtype, device=y_train.device if hasattr(y_train, "device") else "cpu")
        print(
            f"嵌套断裂标定 holdout: "
            f"Q={nested_meta.get('quantile', 0.8):.2f}, "
            f"λ={nested_meta.get('length_m'):.1f} m, "
            f"train矿点={nested_meta.get('n_train_deposits')}"
            + (
                f"；已丢弃距离通道，训练通道数={nested_meta.get('train_channels_after_drop')}"
                if nested_meta.get("dropped_distance_after_nested")
                else ""
            )
        )

    # 创建数据加载器
    sample_weights = None
    if isinstance(normalization_params, dict) and normalization_params.get("deposit_loss_weights") is not None:
        sample_weights = np.asarray(normalization_params.get("deposit_loss_weights"), dtype=np.float64)
    # Epoch monitoring on huge leave-one/basin test sets dominates runtime.
    # Subsample for curves only; final fixed-area SR/EI still use full held-out area.
    monitor_max = int(getattr(args, "eval_monitor_max_samples", 2048) or 2048)
    if monitor_max < 0:
        monitor_max = 0
    X_monitor, y_monitor, did_subsample, n_test_full = _subsample_eval_tensors(
        X_test,
        y_test,
        monitor_max if has_eval_set else 0,
        seed=int(getattr(args, "spatial_random_state", 19) or 19),
    )
    if did_subsample:
        print(
            f"留出/测试集较大（n={n_test_full}）：epoch 监控仅抽样 {len(y_monitor)} 条加速；"
            f"不影响最终固定面积 SR/EI（仍基于完整留出区域）。"
        )

    train_dataset, train_loader, test_dataset, test_loader = create_data_loaders(
        X_train,
        y_train,
        X_monitor if has_eval_set else X_test,
        y_monitor if has_eval_set else y_test,
        args.batchsize,
        augmentation_enabled=bool(getattr(args, "augmentation_enabled", False)),
        augmentation_noise_std=float(getattr(args, "augmentation_noise_std", 0.01) or 0.01),
        sample_weights=sample_weights,
        model_name=args.model,
        random_seed=int(getattr(args, "spatial_random_state", 0) or 0),
        positives_per_batch=getattr(args, "nnpucnn_positives_per_batch", None),
        unlabeled_coverage=getattr(args, "nnpucnn_unlabeled_coverage", None),
    )
    
    # 非神经网络模型特殊处理
    if is_non_neural_model(args.model):
        return train_non_neural_model(
            X_train,
            y_train,
            X_test,
            y_test,
            prior,
            args,
            model_dir,
            device,
            normalization_params=normalization_params,
        )
    
    print("不使用交叉验证，直接训练模型...")
    
    # 创建模型
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    input_channels, input_height, input_width = input_shape
    needs_shape_metadata = is_image_model(args.model)
    model = instantiate_model(args.model, prior, input_shape)
    model = model.to(device)
    
    # 创建优化器（默认 weight_decay=1e-4，对齐手稿 L2）
    optimizer = build_adam_optimizer(model, args)
    
    # 创建损失函数
    loss_func = create_loss_function(args, prior, labels=y_train)
    
    # 记录训练历史
    train_losses, train_errors = [], []
    test_losses, test_errors = [], []
    train_recalls, test_recalls = [], []

    use_plateau_es = bool(getattr(args, "train_loss_plateau_es", True))
    plateau_patience = int(getattr(args, "plateau_patience", 10) or 10)
    plateau_min_epochs = int(getattr(args, "plateau_min_epochs", 20) or 20)
    plateau_delta = float(getattr(args, "plateau_delta", 1e-3) or 1e-3)
    if plateau_patience < 1:
        plateau_patience = 1
    if plateau_min_epochs < 1:
        plateau_min_epochs = 1
    if plateau_delta < 0:
        plateau_delta = 0.0

    best_train_loss = float("inf")
    best_epoch = -1
    best_model_state = None
    plateau_wait = 0
    stopped_early = False

    if use_plateau_es:
        print(
            "已启用训练损失 plateau 早停"
            f"（patience={plateau_patience}, min_epochs={plateau_min_epochs}, delta={plateau_delta:g}）；"
            "不使用测试/留出集做早停判定。"
        )
    else:
        print("已关闭训练损失 plateau 早停，将训练满设定轮数（或直至手动停止）。")
    
    # 训练循环
    for epoch in tqdm(range(args.epoch)):
        if stop_token is not None and stop_token.is_set():
            print("检测到停止信号，提前结束训练。")
            break
        
        # 训练阶段
        train_loss = train_model(model, train_loader, optimizer, loss_func, device, epoch)
        train_error = compute_error(model, train_loader, device)
        train_recall = compute_pos_recall(model, train_loader, device)
        
        # 测试/验证阶段；最终制图训练模式下外部测试集为空，跳过评估。
        if has_eval_set:
            test_loss = evaluate_model(model, test_loader, loss_func, device)
            test_error = compute_error(model, test_loader, device)
            test_recall = compute_pos_recall(model, test_loader, device)
        else:
            test_loss = None
            test_error = None
            test_recall = None
        
        # 记录结果
        train_losses.append(train_loss)
        train_errors.append(train_error)
        if test_loss is not None:
            test_losses.append(test_loss)
        if test_error is not None:
            test_errors.append(test_error)
        train_recalls.append(train_recall)
        if test_recall is not None:
            test_recalls.append(test_recall)

        improved = False
        if use_plateau_es:
            improve_thresh = best_train_loss * (1.0 - plateau_delta) if np.isfinite(best_train_loss) else float("inf")
            if float(train_loss) < improve_thresh:
                best_train_loss = float(train_loss)
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                plateau_wait = 0
                improved = True
            else:
                plateau_wait += 1

        if progress_callback is not None:
            event = {
                'mode': 'standard',
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'train_error': float(train_error),
                'train_recall': float(train_recall),
            }
            if test_loss is not None:
                event['test_loss'] = float(test_loss)
            if test_error is not None:
                event['test_error'] = float(test_error)
            if test_recall is not None:
                event['test_recall'] = float(test_recall)
            if use_plateau_es:
                event['plateau_wait'] = int(plateau_wait)
                event['best_train_loss'] = float(best_train_loss) if np.isfinite(best_train_loss) else None
            progress_callback(event)
        
        if (epoch + 1) % 1 == 0:
            print(f'\nEpoch {epoch+1}/{args.epoch}:')
            print(f'训练损失: {train_loss:.4f}, 错误率: {train_error:.4f}')
            if use_plateau_es:
                marker = "↓改善" if improved else f"平台等待 {plateau_wait}/{plateau_patience}"
                print(f'plateau早停: 最佳训练损失 {best_train_loss:.4f} @ epoch {best_epoch + 1}；{marker}')
            if has_eval_set:
                print(
                    f'测试/验证损失: {test_loss:.4f}, 错误率: {test_error:.4f}, '
                    f'召回率: {test_recall:.4f}'
                )
            else:
                print('测试/验证损失: 跳过（最终制图训练模式未保留验证/测试集）')

        if (
            use_plateau_es
            and (epoch + 1) >= plateau_min_epochs
            and plateau_wait >= plateau_patience
        ):
            stopped_early = True
            print(
                f"训练损失 plateau 早停触发: 已训练 {epoch + 1} 轮，"
                f"连续 {plateau_patience} 轮无相对改善（delta={plateau_delta:g}）。"
                f"恢复第 {best_epoch + 1} 轮最佳训练损失权重。"
            )
            break

    if use_plateau_es and best_model_state is not None:
        model.load_state_dict(best_model_state)
        if not stopped_early:
            print(
                f"未触发 plateau 早停；仍加载训练损失最佳轮次 "
                f"(epoch {best_epoch + 1}, loss={best_train_loss:.4f})。"
            )
    
    # 保存模型
    model_path = os.path.join(model_dir, 'model.pth')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model_payload = model.state_dict()
    if needs_shape_metadata:
        model_payload = {
            'model_state': model.state_dict(),
            'input_channels': input_channels,
            'input_height': input_height,
            'input_width': input_width,
        }

    with open(model_path, 'wb') as f:
        torch.save(model_payload, f)
    
    # Always keep history for spatial eval / checkpoint metadata.
    # Curve PNGs/CSVs remain optional (default off via should_skip_curve_artifacts).
    history = {
        'train_loss': train_losses,
        'train_error': train_errors,
        'test_loss': test_losses,
        'test_error': test_errors,
        'train_recall': train_recalls,
        'test_recall': test_recalls,
        'train_loss_plateau_es': bool(use_plateau_es),
        'plateau_patience': int(plateau_patience),
        'plateau_min_epochs': int(plateau_min_epochs),
        'plateau_delta': float(plateau_delta),
        'plateau_best_epoch': int(best_epoch + 1) if best_epoch >= 0 else None,
        'plateau_best_train_loss': float(best_train_loss) if np.isfinite(best_train_loss) else None,
        'plateau_stopped_early': bool(stopped_early),
    }
    if not should_skip_curve_artifacts(args):
        curves_path = os.path.join(model_dir, 'learning_curves.png')
        os.makedirs(os.path.dirname(curves_path), exist_ok=True)
        plot_training_curves(
            train_losses, test_losses,
            train_errors, test_errors,
            train_recalls, test_recalls,
            curves_path
        )
        history_path = os.path.join(model_dir, 'training_history.pth')
        with open(history_path, 'wb') as f:
            torch.save(history, f)
        curve_artifacts = save_standard_curve_artifacts(history, model_dir)
        print(f"训练曲线数据已保存到: {curve_artifacts['csv']}")
        print(
            "Loss/Accuracy/Recall 曲线数据 CSV 已保存到: "
            f"{curve_artifacts['loss_csv']}, {curve_artifacts['accuracy_csv']}, {curve_artifacts['recall_csv']}"
        )
        print(f"Loss/Accuracy/Recall 曲线已保存到: {curve_artifacts['loss_plot']}, {curve_artifacts['accuracy_plot']}, {curve_artifacts['recall_plot']}")
    
    save_loss_diagnostics(
        loss_func,
        os.path.join(
            model_dir,
            (
                "rn_annpu_diagnostics.json"
                if str(args.model).lower().startswith("rn")
                else "loss_diagnostics_training.json"
            ),
        ),
        context={
            "stage": "training_without_cv",
            "loader": training_loader_diagnostics(train_loader),
            "model_key": str(args.model),
            "existing_pucnn_model_replaced": False,
            "existing_pu_loss_replaced": False,
        },
    )

    save_single_fold_spatial_eval(
        model,
        X_test,
        y_test,
        args,
        model_dir,
        device,
        normalization_params=normalization_params,
        history=history,
        metric_scope=str(final_metric_scope or "standard_training_eval"),
    )

    inner_mode = str(getattr(args, "inner_camp_cv_mode", "") or "").strip().lower()
    no_inner = inner_mode in {"all_internal_train_select", "all_internal_outer_select"}
    if has_eval_set and (
        no_inner
        or str(final_metric_scope or "") == "external_test_eval"
        or bool(getattr(args, "reviewer_primary_protocol", False))
    ):
        save_outer_test_spatial_eval(
            model,
            X_test,
            y_test,
            args,
            model_dir,
            device,
            normalization_params=normalization_params,
        )
    if no_inner:
        save_train_dev_spatial_eval(
            model,
            X_train,
            y_train,
            args,
            model_dir,
            device,
            normalization_params=normalization_params,
            history=history,
        )

    return model

def train_non_neural_model(
    X_train,
    y_train,
    X_test,
    y_test,
    prior,
    args,
    model_dir,
    device,
    normalization_params=None,
):
    """训练非神经网络模型（如OC-SVM, 2step, PU-Random Forest）"""
    print("训练非神经网络模型...")
    
    model = None # 初始化 model 变量为 None
    input_dim = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]

    if args.model == "rf":
        from model.random_forest import RandomForestBinaryClassifier
        model = RandomForestBinaryClassifier(prior, input_dim)
        model.fit(X_train.reshape(X_train.size(0), -1).cpu().numpy(), y_train.cpu().numpy())

        import pickle
        model_file = os.path.join(model_dir, 'model.pkl')
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)

    elif args.model == "ocsvm":
        from model.one_class_svm import OneClassSVMClassifier
        model = OneClassSVMClassifier(prior, input_dim)
        model.fit(X_train.reshape(X_train.size(0), -1).cpu().numpy(), y_train.cpu().numpy())
        
        # 保存 OCSVM 模型
        import pickle
        model_file = os.path.join(model_dir, 'model.pkl')
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
            
    elif args.model == "2step": # <-- 在处理 2step 模型的部分
        from model.two_step_pu import TwoStepPULearning # <-- 修改这里的类名
        
        # 注意：TwoStepPULearning 的 __init__ 接受 prior 和 dim
        # 它内部使用了 RandomForestClassifier，不需要外部传入
        model = TwoStepPULearning(prior=prior, dim=input_dim) # <-- 使用正确的类名和参数初始化
        
        # 假设 fit 方法接受 numpy 数组
        model.fit(X_train.reshape(X_train.size(0), -1).cpu().numpy(), y_train.cpu().numpy())
        
        # 保存 2step 模型 (同样使用 pickle)
        import pickle
        model_file = os.path.join(model_dir, 'model.pkl') # 同样保存为 pkl
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
    elif args.model == "purf":
        from model.pu_random_forest import PURandomForestClassifier

        model = PURandomForestClassifier(prior=prior, dim=input_dim)
        model.fit(X_train.reshape(X_train.size(0), -1).cpu().numpy(), y_train.cpu().numpy())

        import pickle
        model_file = os.path.join(model_dir, 'model.pkl')
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
            
    else:
        print(f"错误：不支持的非神经网络模型类型 '{args.model}'")
        return None # 或者抛出异常

    # --- 模型评估部分 --- 
    # 确保 model 已经被成功初始化
    if model is None:
        print("错误：模型未能成功初始化。")
        return None

    # OCSVM 和 2step 模型可能都需要专门的评估方法
    # 这里以 OCSVM 为例，您需要为 2step 添加类似的评估逻辑
    if args.model in {"rf", "ocsvm", "2step", "purf"}: # <-- 修改条件以包含 RF / 2step / PU-Random Forest
        if len(X_test) == 0:
            print("全量矿点训练模式: 未保留外部测试集，非神经网络模型仅保存训练后的模型。")
            return model
        # 预测结果
        X_train_flat = X_train.reshape(X_train.size(0), -1).cpu().numpy()
        X_test_flat = X_test.reshape(X_test.size(0), -1).cpu().numpy()
        
        # 获取预测标签 (假设 predict 方法返回 +1/-1)
        # 注意：OCSVM 的 forward 返回的是决策函数值，需要转换
        # TwoStepPUClassifier 可能直接返回预测标签，需要确认
        if args.model == "rf":
            y_train_pred = model.predict(X_train_flat)
            y_test_pred = model.predict(X_test_flat)
        elif args.model == "ocsvm":
            train_outputs = model.forward(X_train_flat)
            test_outputs = model.forward(X_test_flat)
            if isinstance(train_outputs, torch.Tensor):
                train_outputs = train_outputs.cpu().detach().numpy()
            if isinstance(test_outputs, torch.Tensor):
                test_outputs = test_outputs.cpu().detach().numpy()
            y_train_pred = np.where(train_outputs.reshape(-1) < 0, 1, -1)
            y_test_pred = np.where(test_outputs.reshape(-1) < 0, 1, -1)
        elif args.model == "2step":
            # 假设 TwoStepPULearning 有 predict 方法返回 +1/-1
            # 查看 two_step_pu.py, 它没有 predict 方法，但 forward 返回 +1/-1 的 Tensor
            # 需要调整评估逻辑以使用 forward
            train_outputs = model.forward(X_train_flat) # 使用 forward
            test_outputs = model.forward(X_test_flat)  # 使用 forward
            
            # 转换 Tensor 为 numpy
            if isinstance(train_outputs, torch.Tensor):
                train_outputs = train_outputs.cpu().detach().numpy()
            if isinstance(test_outputs, torch.Tensor):
                test_outputs = test_outputs.cpu().detach().numpy()
                
            # forward 返回的是 +1/-1，可以直接使用
            y_train_pred = train_outputs.reshape(-1)
            y_test_pred = test_outputs.reshape(-1)
        elif args.model == "purf":
            y_train_pred = model.predict(X_train_flat)
            y_test_pred = model.predict(X_test_flat)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # 计算各种指标
        train_acc = accuracy_score(y_train.cpu().numpy(), y_train_pred)
        test_acc = accuracy_score(y_test.cpu().numpy(), y_test_pred)
        train_prec = precision_score(y_train.cpu().numpy(), y_train_pred, pos_label=1, zero_division=0)
        test_prec = precision_score(y_test.cpu().numpy(), y_test_pred, pos_label=1, zero_division=0)
        train_recall = recall_score(y_train.cpu().numpy(), y_train_pred, pos_label=1, zero_division=0)
        test_recall = recall_score(y_test.cpu().numpy(), y_test_pred, pos_label=1, zero_division=0)
        train_f1 = f1_score(y_train.cpu().numpy(), y_train_pred, pos_label=1, zero_division=0)
        test_f1 = f1_score(y_test.cpu().numpy(), y_test_pred, pos_label=1, zero_division=0)

        print('\n使用传统评估指标:')
        print(f'训练集 - 准确率: {train_acc:.4f}, 精确率: {train_prec:.4f}, 召回率: {train_recall:.4f}, F1: {train_f1:.4f}')
        print(f'测试集 - 准确率: {test_acc:.4f}, 精确率: {test_prec:.4f}, 召回率: {test_recall:.4f}, F1: {test_f1:.4f}')
        
        # 保存评估结果
        metrics = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_precision': train_prec,
            'test_precision': test_prec,
            'train_recall': train_recall,
            'test_recall': test_recall,
            'train_f1': train_f1,
            'test_f1': test_f1
        }
        
        # 保存指标到文件
        metrics_path = os.path.join(model_dir, 'metrics.json')
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
            
        save_single_fold_spatial_eval(
            model,
            X_test,
            y_test,
            args,
            model_dir,
            device,
            normalization_params=normalization_params,
            metrics={
                "test_error": float(1.0 - test_acc),
                "test_recall": float(test_recall),
            },
            metric_scope="standard_training_eval",
        )
        inner_mode = str(getattr(args, "inner_camp_cv_mode", "") or "").strip().lower()
        no_inner = inner_mode in {"all_internal_train_select", "all_internal_outer_select"}
        if no_inner or bool(getattr(args, "reviewer_primary_protocol", False)):
            save_outer_test_spatial_eval(
                model,
                X_test,
                y_test,
                args,
                model_dir,
                device,
                normalization_params=normalization_params,
            )
        if no_inner:
            save_train_dev_spatial_eval(
                model,
                X_train,
                y_train,
                args,
                model_dir,
                device,
                normalization_params=normalization_params,
                history=None,
            )

        return model # 返回训练好的模型
    
    # --- 如果不是 OCSVM 或 2step，则执行以下通用评估（可能不适用）---
    # 这部分代码可能需要移除或调整，因为它假设 model 是 PyTorch 模型
    # 并且使用了 create_data_loaders, evaluate_model 等 PyTorch 相关函数
    
    # # 创建数据加载器进行评估
    # train_dataset, train_loader, test_dataset, test_loader = create_data_loaders(
    #     X_train, y_train, X_test, y_test, args.batchsize
    # )
    # 
    # # 创建损失函数
    # loss_func = create_loss_function(args, prior)
    # 
    # # 评估模型
    # train_loss = evaluate_model(model, train_loader, loss_func, device)
    # test_loss = evaluate_model(model, test_loader, loss_func, device)
    # train_error = compute_error(model, train_loader, device)
    # test_error = compute_error(model, test_loader, device)
    # 
    # print('训练损失: {:.3f}, 错误率: {:.3f}'.format(train_loss, train_error))
    # print('测试损失: {:.3f}, 错误率: {:.3f}'.format(test_loss, test_error))
    # 
    # # 保存模型 (这行假设是 PyTorch 模型，对于非神经网络模型应移除)
    # # torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
    
    # return model # 确保在所有路径都有返回值
