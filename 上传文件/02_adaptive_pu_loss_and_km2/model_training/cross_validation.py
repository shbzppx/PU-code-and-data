import json
import os
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from evaluation_metrics import compute_error, compute_pos_recall, evaluate_model
from nested_fault_calibration import apply_nested_fault_calibration
from spatial_cv_metrics import (
    build_spatial_cv_context,
    evaluate_spatial_cv_fold,
    summarize_fold_series,
    summarize_spatial_cv_metrics,
)
from train_utils import build_adam_optimizer, save_outer_test_spatial_eval, train_model, train_non_neural_model_with_cv
from utils import (
    augment_training_tensor,
    create_loss_function,
    create_training_data_loader,
    instantiate_model,
    is_image_model,
    is_non_neural_model,
    save_loss_diagnostics,
    training_loader_diagnostics,
)
from visualization import plot_cv_curves, save_cv_curve_artifacts, save_spatial_cv_metrics_xlsx
from result_artifact_policy import (
    should_skip_curve_artifacts,
    should_skip_fold_weight_artifacts,
    should_skip_inner_search_log,
)


def _save_torch_payload(payload, path):
    """Save through a file handle so PyTorch 1.13 supports Unicode paths on Windows."""
    absolute_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
    with open(absolute_path, "wb") as handle:
        torch.save(payload, handle)


def _resolve_fold_prior(default_prior, args, features, labels, train_idx, normalization_params, fold_idx):
    mode = str(getattr(args, "prior_mode", "manual") or "manual").strip().lower()
    if mode not in {"estimate_en", "elkan_noto", "en", "estimate_km2", "km2"}:
        return float(default_prior), None
    from prior_estimation import resolve_prior_triple

    train_idx = np.asarray(train_idx, dtype=np.int64)
    groups = None
    weights = None
    if isinstance(normalization_params, dict):
        all_groups = normalization_params.get("prior_estimation_groups")
        if all_groups is not None and len(np.asarray(all_groups).reshape(-1)) == len(labels):
            groups = np.asarray(all_groups).reshape(-1)[train_idx]
        all_weights = normalization_params.get("prior_sampling_weights")
        if all_weights is not None and len(np.asarray(all_weights).reshape(-1)) == len(labels):
            weights = np.asarray(all_weights, dtype=np.float64).reshape(-1)[train_idx]
    feature_subset = features[train_idx]
    if hasattr(feature_subset, "detach"):
        feature_subset = feature_subset.detach().cpu().numpy()
    label_subset = labels[train_idx]
    if hasattr(label_subset, "detach"):
        label_subset = label_subset.detach().cpu().numpy()
    estimate = resolve_prior_triple(
        prior_mode=mode,
        features=feature_subset,
        labels=label_subset,
        groups=groups,
        sample_weights=weights,
        tuning_prior=getattr(args, "manual_prior", None),
        geological_favorable_area_ratio=getattr(args, "geological_favorable_area_ratio", None),
        random_state=int(getattr(args, "spatial_random_state", 42) or 42) + int(fold_idx),
        compare_estimators=bool(getattr(args, "reviewer_protocol", False)),
    )
    resolved = estimate.get("resolved_prior")
    if resolved is None:
        raise ValueError(f"Inner fold {fold_idx + 1} prior could not be estimated.")
    return float(resolved), estimate


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


def prepare_cv_folds(X_train, y_train, n_folds, min_batch_size, stratified=True):
    """构建交叉验证折。"""
    indices = np.arange(len(X_train))
    y_arr = y_train.detach().cpu().numpy() if torch.is_tensor(y_train) else np.asarray(y_train)

    if stratified:
        pos_indices = indices[y_arr == 1]
        neg_indices = indices[y_arr == -1]
        pos_splits = np.array_split(pos_indices, n_folds)
        neg_splits = np.array_split(neg_indices, n_folds)

        fold_indices = []
        for fold_idx in range(n_folds):
            val_fold = np.concatenate([pos_splits[fold_idx], neg_splits[fold_idx]])
            np.random.shuffle(val_fold)

            val_remainder = len(val_fold) % min_batch_size
            if val_remainder > 0:
                val_fold = val_fold[:-val_remainder]

            train_parts = []
            for other_idx in range(n_folds):
                if other_idx == fold_idx:
                    continue
                train_parts.append(np.concatenate([pos_splits[other_idx], neg_splits[other_idx]]))
            train_fold = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)

            train_remainder = len(train_fold) % min_batch_size
            if train_remainder > 0:
                train_fold = train_fold[:-train_remainder]

            if len(train_fold) > 0 and len(val_fold) > 0:
                fold_indices.append((train_fold, val_fold))
    else:
        np.random.shuffle(indices)
        fold_size = len(indices) // n_folds
        adjusted_fold_size = (fold_size // min_batch_size) * min_batch_size

        fold_indices = []
        for fold_idx in range(n_folds):
            start_idx = fold_idx * adjusted_fold_size
            end_idx = min((fold_idx + 1) * adjusted_fold_size, len(indices))
            if end_idx - start_idx < min_batch_size:
                continue

            val_fold = indices[start_idx:end_idx]
            train_fold = np.concatenate([indices[:start_idx], indices[end_idx:]])

            train_remainder = len(train_fold) % min_batch_size
            if train_remainder > 0:
                train_fold = train_fold[:-train_remainder]

            if len(train_fold) > 0 and len(val_fold) > 0:
                fold_indices.append((train_fold, val_fold))

    return fold_indices


def _normalize_predefined_folds(predefined_folds):
    """将外部传入折统一转换为 (train_idx, val_idx) 列表。"""
    if not predefined_folds:
        return []
    normalized = []
    for fold in predefined_folds:
        metadata = {}
        if isinstance(fold, dict):
            train_idx = np.asarray(fold.get("train_indices", []), dtype=np.int64)
            val_idx = np.asarray(fold.get("val_indices", []), dtype=np.int64)
            metadata = dict(fold)
        else:
            try:
                train_idx, val_idx = fold
            except Exception:
                continue
            train_idx = np.asarray(train_idx, dtype=np.int64)
            val_idx = np.asarray(val_idx, dtype=np.int64)
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        metadata["train_indices"] = train_idx
        metadata["val_indices"] = val_idx
        if metadata.get("val_area_indices") is not None:
            metadata["val_area_indices"] = np.asarray(metadata.get("val_area_indices"), dtype=np.int64)
        normalized.append(metadata)
    return normalized


def _predict_neural_positive_scores(model, loader, device):
    model.eval()
    scores = []
    score_transform = getattr(model, "positive_scores_from_logits", None)
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            output = model(data).view(-1)
            positive_scores = (
                score_transform(output)
                if callable(score_transform)
                else torch.sigmoid(output)
            )
            scores.append(positive_scores.detach().cpu().numpy())
    if not scores:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(scores).astype(np.float64)


def _predict_neural_positive_scores_from_tensor(model, features, batch_size, device):
    labels = torch.zeros(len(features), dtype=torch.long)
    loader = DataLoader(TensorDataset(torch.as_tensor(features, dtype=torch.float32), labels), batch_size=batch_size, shuffle=False)
    return _predict_neural_positive_scores(model, loader, device)


def train_with_cv(
    X_train,
    y_train,
    X_test,
    y_test,
    prior,
    args,
    model_dir,
    device,
    predefined_folds=None,
    stop_token=None,
    progress_callback=None,
    normalization_params=None,
    _selection_only=False,
    _preselected_inner=None,
    _candidate_params=None,
):
    """使用交叉验证训练。支持外部预定义折（空间分层折）。"""
    print(f"使用 {args.cv_folds} 折交叉验证...")

    reviewer_primary = bool(getattr(args, "reviewer_primary_protocol", False))
    if reviewer_primary and _candidate_params is None and _preselected_inner is None:
        from hyperparameter_protocol import apply_inner_candidate, inner_candidate_score, reviewer_inner_candidates

        candidates = reviewer_inner_candidates(args.model, args)
        candidate_results = []
        # Short names: full "inner_hyperparameter_search/candidate_XX" blows Windows MAX_PATH
        # under M leave-one-camp × model-grid × nested fault calibration.
        tuning_root = os.path.join(model_dir, "ihp")
        os.makedirs(tuning_root, exist_ok=True)
        for candidate_index, candidate in enumerate(candidates, start=1):
            candidate_args = apply_inner_candidate(args, candidate)
            candidate_dir = os.path.join(tuning_root, f"c{candidate_index:02d}")
            os.makedirs(candidate_dir, exist_ok=True)
            candidate_normalization = copy.deepcopy(normalization_params)
            inner_result = train_with_cv(
                X_train,
                y_train,
                X_test,
                y_test,
                prior,
                candidate_args,
                candidate_dir,
                device,
                predefined_folds=predefined_folds,
                stop_token=stop_token,
                progress_callback=progress_callback,
                normalization_params=candidate_normalization,
                _selection_only=True,
                _candidate_params=candidate,
            )
            if not isinstance(inner_result, dict):
                raise RuntimeError(f"Inner candidate {candidate_index} did not complete.")
            score, rule = inner_candidate_score(inner_result)
            candidate_results.append(
                {
                    "candidate_index": candidate_index,
                    "hyperparameters": candidate,
                    "selection_score": score,
                    "selection_rule": rule,
                    "inner_result": inner_result,
                }
            )
        selected = min(candidate_results, key=lambda row: float(row["selection_score"]))
        protocol = {
            "scope": "outer_training_partition_only",
            "outer_test_used": False,
            "candidate_count": len(candidate_results),
            "equal_budget": {
                "inner_fold_count": int(getattr(args, "cv_folds", 0) or 0),
                "maximum_epochs_per_neural_candidate_fold": int(getattr(args, "epoch", 0) or 0),
                "candidate_count_per_model": len(candidate_results),
            },
            "candidates": [
                {key: value for key, value in row.items() if key != "inner_result"}
                for row in candidate_results
            ],
            "selected_candidate_index": selected["candidate_index"],
            "selected_hyperparameters": selected["hyperparameters"],
            "selection_rule": selected["selection_rule"],
        }
        if not should_skip_inner_search_log(args):
            with open(os.path.join(model_dir, "inner_hyperparameter_search.json"), "w", encoding="utf-8") as handle:
                json.dump(protocol, handle, ensure_ascii=False, indent=2)
        else:
            # Slim selection record only (no full candidate dump).
            slim = {
                "scope": protocol.get("scope"),
                "outer_test_used": protocol.get("outer_test_used"),
                "candidate_count": protocol.get("candidate_count"),
                "selected_candidate_index": protocol.get("selected_candidate_index"),
                "selected_hyperparameters": protocol.get("selected_hyperparameters"),
                "selection_rule": protocol.get("selection_rule"),
            }
            with open(os.path.join(model_dir, "inner_hyperparameter_selected.json"), "w", encoding="utf-8") as handle:
                json.dump(slim, handle, ensure_ascii=False, indent=2)
        if bool(getattr(args, "inner_screening_only", False)):
            keep_keys = (
                "mean_val_loss",
                "cv_sr_mean",
                "cv_sr_std",
                "cv_paf_mean",
                "cv_paf_std",
                "cv_ei_mean",
                "cv_ei_std",
                "cv_tie_expected_center_sr_mean",
                "cv_tie_expected_center_sr_std",
                "cv_tie_expected_center_ei_mean",
                "cv_tie_expected_center_ei_std",
                "inner_fold_best_epochs",
                "spatial_cv_metric_fold_count",
            )
            screening_result = {
                key: selected["inner_result"].get(key) for key in keep_keys
            }
            screening_result.update({
                "inner_screening_only": True,
                "outer_test_used": False,
                "selected_candidate_index": selected["candidate_index"],
                "selected_hyperparameters": selected["hyperparameters"],
                "selection_rule": selected["selection_rule"],
            })
            for filename in ("inner_screening_result.json", "cv_results.json"):
                with open(os.path.join(model_dir, filename), "w", encoding="utf-8") as handle:
                    json.dump(screening_result, handle, ensure_ascii=False, indent=2)
            return screening_result
        selected_args = apply_inner_candidate(args, selected["hyperparameters"])
        return train_with_cv(
            X_train,
            y_train,
            X_test,
            y_test,
            prior,
            selected_args,
            model_dir,
            device,
            predefined_folds=predefined_folds,
            stop_token=stop_token,
            progress_callback=progress_callback,
            normalization_params=normalization_params,
            _preselected_inner=selected["inner_result"],
            _candidate_params=selected["hyperparameters"],
        )
    validation_only_cv = reviewer_primary or (
        str(getattr(args, "split_mode", "") or "").strip().lower() == "spatial_cluster_holdout_cv"
        and str(getattr(args, "mineral_training_strategy", "") or "").strip().lower() == "holdout_cv"
    )
    outer_test_available = len(X_test) > 0
    has_external_test = outer_test_available and not validation_only_cv
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=True) if has_external_test else None
    if validation_only_cv and outer_test_available:
        print("外层测试集已隔离；内层CV、调参和早停仅使用各折训练/验证数据。")
    input_channels = X_train.shape[1]
    input_height = X_train.shape[2]
    input_width = X_train.shape[3]
    base_input_shape = (input_channels, input_height, input_width)

    is_non_neural = is_non_neural_model(args.model)
    min_batch_size = 1 if is_non_neural else max(2, args.batchsize)

    cv_train_losses, cv_val_losses = [], []
    cv_train_errors, cv_val_errors = [], []
    cv_test_losses, cv_test_errors = [], []
    cv_train_recalls, cv_val_recalls, cv_test_recalls = [], [], []
    cv_best_val_losses, cv_best_val_errors, cv_best_val_recalls = [], [], []
    cv_spatial_metrics = []
    fold_prior_estimates = []
    spatial_metric_context = build_spatial_cv_context(normalization_params)

    best_epochs = []
    if isinstance(_preselected_inner, dict):
        cv_train_losses = _preselected_inner.get("train_losses", [])
        cv_val_losses = _preselected_inner.get("val_losses", [])
        cv_train_errors = _preselected_inner.get("train_errors", [])
        cv_val_errors = _preselected_inner.get("val_errors", [])
        cv_test_losses = _preselected_inner.get("test_losses", [])
        cv_test_errors = _preselected_inner.get("test_errors", [])
        cv_train_recalls = _preselected_inner.get("train_recalls", [])
        cv_val_recalls = _preselected_inner.get("val_recalls", [])
        cv_test_recalls = _preselected_inner.get("test_recalls", [])
        cv_best_val_losses = _preselected_inner.get("best_val_losses", [])
        cv_best_val_errors = _preselected_inner.get("best_val_errors", [])
        cv_best_val_recalls = _preselected_inner.get("best_val_recalls", [])
        cv_spatial_metrics = _preselected_inner.get("fold_spatial_metrics", [])
        fold_prior_estimates = _preselected_inner.get("fold_prior_estimates", [])
        best_epochs = _preselected_inner.get("inner_fold_best_epochs", [])

    fold_records = _normalize_predefined_folds(predefined_folds)
    if fold_records:
        print(f"使用预定义空间CV折: {len(fold_records)} 折")
    else:
        fold_records = [
            {"train_indices": train_idx, "val_indices": val_idx}
            for train_idx, val_idx in prepare_cv_folds(
            X_train,
            y_train,
            args.cv_folds,
            min_batch_size,
            not args.no_stratified if hasattr(args, "no_stratified") else args.stratified,
            )
        ]

    if not fold_records:
        print("未能构造有效的交叉验证折。")
        return None

    nested_enabled = bool(getattr(args, "nested_fault_calibration", False))
    fault_path = str(getattr(args, "fault_lines_path", "") or "").strip()
    if nested_enabled:
        if not fault_path or not os.path.exists(fault_path):
            raise FileNotFoundError(
                "已开启 --nested-fault-calibration，但未提供有效 --fault-lines-path。"
            )
        print("OGR R1.5 嵌套断裂衰减标定：已启用（每折仅用 train 矿点标定 λ）。")

    if is_non_neural:
        return train_non_neural_model_with_cv(
            X_train,
            y_train,
            X_test,
            y_test,
            prior,
            args,
            model_dir,
            device,
            fold_records,
            stop_token=stop_token,
            progress_callback=progress_callback,
            normalization_params=normalization_params,
            selection_only=_selection_only,
            preselected_inner=_preselected_inner,
            candidate_params=_candidate_params,
        )

    stop_requested = False
    fold_records_to_run = [] if isinstance(_preselected_inner, dict) else fold_records
    for fold_idx, fold in enumerate(fold_records_to_run):
        train_idx = np.asarray(fold.get("train_indices", []), dtype=np.int64)
        val_idx = np.asarray(fold.get("val_indices", []), dtype=np.int64)
        if stop_token is not None and stop_token.is_set():
            print("检测到停止信号，停止后续折训练。")
            stop_requested = True
            break

        print(f"\n开始训练第 {fold_idx + 1}/{len(fold_records)} 折...")

        fold_X_train = X_train
        fold_X_test = X_test
        if nested_enabled:
            fold_X_train, _, fold_X_test, nested_meta = apply_nested_fault_calibration(
                fold_name=f"fold{fold_idx + 1}",
                output_dir=model_dir,
                fault_path=fault_path,
                X_train=X_train,
                y_train=y_train,
                train_idx=train_idx,
                val_idx=val_idx,
                X_test=None if reviewer_primary else (X_test if outer_test_available else None),
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
            spatial_metric_context = build_spatial_cv_context(normalization_params)
            # Inner-fold drop does not mutate shared area features; align locally for scoring.
            if nested_meta.get("dropped_distance_after_nested") and spatial_metric_context.get("area_features") is not None:
                from nested_fault_calibration import align_area_features_after_distance_drop

                spatial_metric_context["area_features"] = align_area_features_after_distance_drop(
                    spatial_metric_context["area_features"],
                    normalization_params if isinstance(normalization_params, dict) else {},
                )

        # Model input channels must follow post-nested tensors (decay_only drops distance).
        fold_input_shape = (
            int(fold_X_train.shape[1]),
            int(fold_X_train.shape[2]),
            int(fold_X_train.shape[3]),
        )
        fold_input_channels = int(fold_X_train.shape[1])

        train_x = fold_X_train[train_idx]
        if bool(getattr(args, "augmentation_enabled", False)):
            train_x = augment_training_tensor(
                train_x,
                noise_std=float(getattr(args, "augmentation_noise_std", 0.01) or 0.01),
            )
        deposit_weights = None
        if isinstance(normalization_params, dict) and normalization_params.get("deposit_loss_weights") is not None:
            all_w = np.asarray(normalization_params["deposit_loss_weights"], dtype=np.float64)
            if len(all_w) == len(y_train):
                deposit_weights = torch.as_tensor(all_w[np.asarray(train_idx, dtype=np.int64)], dtype=torch.float32)
        if deposit_weights is None:
            train_dataset = TensorDataset(train_x, y_train[train_idx])
        else:
            train_dataset = TensorDataset(train_x, y_train[train_idx], deposit_weights)
        train_loader = create_training_data_loader(
            train_dataset,
            args.batchsize,
            args.model,
            seed=int(getattr(args, "spatial_random_state", 0) or 0) + int(fold_idx),
            positives_per_batch=getattr(args, "nnpucnn_positives_per_batch", None),
            unlabeled_coverage=getattr(args, "nnpucnn_unlabeled_coverage", None),
        )
        val_loader = DataLoader(
            TensorDataset(fold_X_train[val_idx], y_train[val_idx]),
            batch_size=args.batchsize,
            shuffle=False,
        )
        if has_external_test and fold_X_test is not None:
            test_loader = DataLoader(
                TensorDataset(fold_X_test, y_test),
                batch_size=args.batchsize,
                shuffle=True,
            )

        print(f"训练加载器: {len(train_loader)} 批次, 验证加载器: {len(val_loader)} 批次")

        fold_prior, fold_prior_meta = _resolve_fold_prior(
            prior, args, fold_X_train, y_train, train_idx, normalization_params, fold_idx
        )
        if fold_prior_meta is not None:
            fold_prior_estimates.append({"fold": int(fold_idx + 1), **fold_prior_meta})
            print(f"内层第 {fold_idx + 1} 折先验（仅 fold-train）: π={fold_prior:.6f}")
        model = instantiate_model(args.model, fold_prior, fold_input_shape).to(device)
        optimizer = build_adam_optimizer(model, args)
        fold_train_labels = y_train[train_idx] if hasattr(y_train, "__getitem__") else y_train
        loss_func = create_loss_function(args, fold_prior, labels=fold_train_labels)

        fold_train_losses, fold_val_losses = [], []
        fold_train_errors, fold_val_errors = [], []
        fold_train_recalls, fold_val_recalls = [], []

        best_epoch = 0
        best_fold_val_loss = float("inf")
        best_fold_spatial_score = float("-inf")
        best_fold_model_state = None
        patience_counter = 0
        use_early_stopping = args.early_stopping and not (
            hasattr(args, "no_early_stopping") and args.no_early_stopping
        )
        spatial_every = max(1, int(getattr(args, "spatial_checkpoint_every", 1) or 1))
        use_spatial_checkpoint = bool(
            getattr(args, "spatial_checkpoint_selection", True)
        ) and bool(spatial_metric_context)
        if use_spatial_checkpoint:
            print(
                "训练–评价对齐：折内 checkpoint/早停优先使用验证集 "
                "fixed-area 并列稳健 EI（不可用时回退验证损失）。"
            )
        minimum_checkpoint_epoch = 0
        if str(args.model).lower() in {"rncapucnn", "rncpucnn", "rnfcapucnn", "rnfcspucnn", "rngapucnn", "rngspucnn", "rnlrapucnn", "rnlrspucnn", "rnscapucnn", "rnscspucnn"}:
            minimum_checkpoint_epoch = max(0, int(getattr(args, "rn_warmup_epochs", 5) or 5))
            print(
                f"RN-v2 checkpoint和早停从第 {minimum_checkpoint_epoch + 1} 轮开始。"
            )

        for epoch in tqdm(range(args.epoch)):
            if stop_token is not None and stop_token.is_set():
                print("检测到停止信号，提前结束当前折。")
                stop_requested = True
                break

            train_loss = train_model(model, train_loader, optimizer, loss_func, device, epoch)
            train_error = compute_error(model, train_loader, device)
            train_recall = compute_pos_recall(model, train_loader, device)

            val_loss = evaluate_model(model, val_loader, loss_func, device)
            val_error = compute_error(model, val_loader, device)
            val_recall = compute_pos_recall(model, val_loader, device)

            fold_train_losses.append(train_loss)
            fold_val_losses.append(val_loss)
            fold_train_errors.append(train_error)
            fold_val_errors.append(val_error)
            fold_train_recalls.append(train_recall)
            fold_val_recalls.append(val_recall)

            spatial_score = None
            spatial_ei = None
            spatial_tie_ei = None
            monitor_spatial = (
                use_spatial_checkpoint
                and ((epoch + 1) % spatial_every == 0 or epoch == 0)
            )
            if monitor_spatial:
                try:
                    area_indices = fold.get("val_area_indices")
                    area_features = spatial_metric_context.get("area_features")
                    if area_indices is not None and area_features is not None:
                        area_indices = np.asarray(area_indices, dtype=np.int64)
                        val_scores = _predict_neural_positive_scores_from_tensor(
                            model,
                            area_features[area_indices],
                            args.batchsize,
                            device,
                        )
                    else:
                        area_indices = None
                        val_scores = _predict_neural_positive_scores(model, val_loader, device)
                    epoch_spatial = evaluate_spatial_cv_fold(
                        val_scores,
                        val_idx,
                        spatial_metric_context,
                        area_indices=area_indices,
                        threshold_step=float(
                            getattr(args, "spatial_metric_threshold_step", 0.01) or 0.01
                        ),
                        fixed_threshold=_spatial_metric_fixed_threshold(args),
                        distance_threshold=float(
                            getattr(args, "spatial_metric_distance_threshold", 4.0) or 4.0
                        ),
                        area_fractions=_spatial_metric_area_fractions(args),
                        primary_area_fraction=_spatial_metric_primary_area_fraction(args),
                        primary_probability_mode=getattr(
                            model, "preferred_spatial_score_mode", "center"
                        ),
                    )
                    if epoch_spatial is not None:
                        spatial_ei = float(epoch_spatial.get("val_ei", 0.0) or 0.0)
                        if epoch_spatial.get("tie_expected_center_ei") is not None:
                            spatial_tie_ei = float(epoch_spatial["tie_expected_center_ei"])
                        spatial_score = (
                            spatial_tie_ei if spatial_tie_ei is not None else spatial_ei
                        )
                except Exception as exc:  # noqa: BLE001
                    print(f"折内空间监控失败（回退验证损失）: {exc}")

            if progress_callback is not None:
                event = {
                    "mode": "cv",
                    "fold": fold_idx + 1,
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "train_error": float(train_error),
                    "val_error": float(val_error),
                    "train_recall": float(train_recall),
                    "val_recall": float(val_recall),
                }
                if spatial_score is not None:
                    event["val_spatial_score"] = float(spatial_score)
                if spatial_tie_ei is not None:
                    event["val_tie_expected_center_ei"] = float(spatial_tie_ei)
                if spatial_ei is not None:
                    event["val_ei"] = float(spatial_ei)
                progress_callback(event)

            spatial_txt = ""
            if spatial_score is not None:
                spatial_txt = (
                    f", 验证并列稳健EI={spatial_tie_ei:.4f}"
                    if spatial_tie_ei is not None
                    else f", 验证EI={spatial_ei:.4f}"
                )
            print(
                f"周期 {epoch + 1}/{args.epoch}: "
                f"训练损失 {train_loss:.4f}, 训练错误率 {train_error:.4f}, 训练召回率 {train_recall:.4f}"
            )
            print(
                f"验证损失 {val_loss:.4f}, 验证错误率 {val_error:.4f}, "
                f"验证召回率 {val_recall:.4f}{spatial_txt}"
            )

            checkpoint_eligible = epoch >= minimum_checkpoint_epoch
            if not checkpoint_eligible:
                patience_counter = 0
            elif use_spatial_checkpoint:
                if spatial_score is None:
                    # Non-monitor epoch: keep patience unchanged.
                    pass
                elif float(spatial_score) > best_fold_spatial_score + 1e-12:
                    best_fold_spatial_score = float(spatial_score)
                    best_fold_val_loss = float(val_loss)
                    best_fold_model_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
            elif val_loss < best_fold_val_loss:
                best_fold_val_loss = val_loss
                best_fold_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if checkpoint_eligible and use_early_stopping and patience_counter >= args.patience:
                if use_spatial_checkpoint and best_fold_spatial_score > float("-inf"):
                    print(
                        f"早停触发: 连续 {args.patience} 个周期空间指标无提升。"
                    )
                    print(
                        f"最佳验证空间分数出现在周期 {best_epoch + 1}: "
                        f"{best_fold_spatial_score:.4f}（对应验证损失 {best_fold_val_loss:.4f}）"
                    )
                else:
                    print(f"早停触发: 连续 {args.patience} 个周期无提升。")
                    print(f"最佳验证损失出现在周期 {best_epoch + 1}: {best_fold_val_loss:.4f}")
                break

        if fold_train_losses and best_fold_model_state is None:
            # Spatial monitoring never produced a checkpoint; keep final weights.
            best_fold_model_state = copy.deepcopy(model.state_dict())
            best_epoch = max(0, len(fold_train_losses) - 1)
            best_fold_val_loss = float(fold_val_losses[-1]) if fold_val_losses else best_fold_val_loss

        if fold_train_losses and best_fold_model_state is not None:
            model.load_state_dict(best_fold_model_state)
            best_val_loss_eval = evaluate_model(model, val_loader, loss_func, device)
            best_val_error_eval = compute_error(model, val_loader, device)
            best_val_recall_eval = compute_pos_recall(model, val_loader, device)
            if spatial_metric_context:
                area_indices = fold.get("val_area_indices")
                area_features = spatial_metric_context.get("area_features")
                if area_indices is not None and area_features is not None:
                    area_indices = np.asarray(area_indices, dtype=np.int64)
                    val_scores = _predict_neural_positive_scores_from_tensor(
                        model,
                        area_features[area_indices],
                        args.batchsize,
                        device,
                    )
                else:
                    area_indices = None
                    val_scores = _predict_neural_positive_scores(model, val_loader, device)
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
                    print(
                        f"第 {fold_idx + 1} 折空间验证: "
                        f"SR={spatial_metrics['val_sr']:.4f}, "
                        f"PAF={spatial_metrics['val_paf']:.4f}, "
                        f"EI={spatial_metrics['val_ei']:.4f}, "
                        f"tieEI={float(spatial_metrics.get('tie_expected_center_ei') or spatial_metrics['val_ei']):.4f}, "
                        f"uniq={spatial_metrics.get('score_unique_count', '')}, "
                        f"threshold={spatial_metrics['threshold']:.2f}"
                    )
            if has_external_test:
                test_loss = evaluate_model(model, test_loader, loss_func, device)
                test_error = compute_error(model, test_loader, device)
                test_recall = compute_pos_recall(model, test_loader, device)
            else:
                test_loss = float(best_val_loss_eval)
                test_error = float(best_val_error_eval)
                test_recall = float(best_val_recall_eval)
                print("本折汇总使用验证集指标。")
            result_label = "测试结果" if has_external_test else "验证汇总"
            print(
                f"第 {fold_idx + 1} 折{result_label}: "
                f"损失 {test_loss:.4f}, 错误率 {test_error:.4f}, 召回率 {test_recall:.4f}"
            )

            cv_train_losses.append(fold_train_losses)
            cv_val_losses.append(fold_val_losses)
            cv_train_errors.append(fold_train_errors)
            cv_val_errors.append(fold_val_errors)
            cv_test_losses.append(test_loss)
            cv_test_errors.append(test_error)
            cv_train_recalls.append(fold_train_recalls)
            cv_val_recalls.append(fold_val_recalls)
            cv_test_recalls.append(test_recall)
            cv_best_val_losses.append(best_val_loss_eval)
            cv_best_val_errors.append(best_val_error_eval)
            cv_best_val_recalls.append(best_val_recall_eval)
            best_epochs.append(int(best_epoch + 1))
            save_loss_diagnostics(
                loss_func,
                os.path.join(
                    model_dir,
                    (
                        f"rn_annpu_inner_fold{fold_idx + 1}_diagnostics.json"
                        if str(args.model).lower().startswith("rn")
                        else f"loss_diagnostics_inner_fold{fold_idx + 1}.json"
                    ),
                ),
                context={
                    "stage": "inner_fold",
                    "fold": int(fold_idx + 1),
                    "best_epoch": int(best_epoch + 1),
                    "loader": training_loader_diagnostics(train_loader),
                    "model_key": str(args.model),
                },
            )

            fold_model_path = os.path.join(model_dir, f"model_fold{fold_idx + 1}.pth")
            fold_payload = model.state_dict()
            if is_image_model(args.model):
                fold_payload = {
                    "model_state": model.state_dict(),
                    "input_channels": fold_input_channels,
                    "input_height": int(fold_X_train.shape[2]),
                    "input_width": int(fold_X_train.shape[3]),
                }
            if not should_skip_fold_weight_artifacts(args):
                _save_torch_payload(fold_payload, fold_model_path)

        if stop_requested:
            break

    if not cv_test_losses:
        print("未完成有效交叉验证训练。")
        return None

    avg_test_loss = float(np.mean(cv_test_losses))
    avg_test_error = float(np.mean(cv_test_errors))
    avg_test_recall = float(np.mean(cv_test_recalls))
    val_summary = {}
    val_summary.update(summarize_fold_series(cv_best_val_losses, "val_loss"))
    val_summary.update(summarize_fold_series(cv_best_val_errors, "val_error"))
    val_summary.update(summarize_fold_series(cv_best_val_recalls, "val_recall"))
    spatial_summary = summarize_spatial_cv_metrics(cv_spatial_metrics)
    if _selection_only:
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
            "best_val_losses": cv_best_val_losses,
            "best_val_errors": cv_best_val_errors,
            "best_val_recalls": cv_best_val_recalls,
            "fold_spatial_metrics": cv_spatial_metrics,
            "fold_prior_estimates": fold_prior_estimates,
            "inner_fold_best_epochs": best_epochs,
            "mean_val_loss": val_summary.get("mean_val_loss"),
            "tie_aware_center_selection": bool(
                getattr(args, "tie_aware_center_selection", False)
            ),
            **spatial_summary,
        }

    print("\n交叉验证汇总:")
    aggregate_label = "平均验证汇总" if validation_only_cv else "平均测试"
    print(f"{aggregate_label}损失: {avg_test_loss:.4f} ± {np.std(cv_test_losses):.4f}")
    print(f"{aggregate_label}错误率: {avg_test_error:.4f} ± {np.std(cv_test_errors):.4f}")
    print(f"{aggregate_label}召回率: {avg_test_recall:.4f} ± {np.std(cv_test_recalls):.4f}")

    if spatial_summary.get("cv_ei_mean") is not None:
        print(
            "CV 空间验证 EI: "
            f"{spatial_summary.get('cv_ei_mean'):.4f} ± {spatial_summary.get('cv_ei_std'):.4f}"
        )

    selected_epoch = max(1, int(round(float(np.median(best_epochs)))))
    final_train_idx = np.arange(len(X_train), dtype=np.int64)
    final_X_train = X_train
    final_X_test = X_test
    final_fault_meta = None
    if nested_enabled:
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
        if final_fault_meta and final_fault_meta.get("dropped_distance_after_nested"):
            print(
                "外层重训嵌套标定：已丢弃距离通道，"
                f"训练通道数={final_fault_meta.get('train_channels_after_drop')}"
            )
    final_input_shape = (
        int(final_X_train.shape[1]),
        int(final_X_train.shape[2]),
        int(final_X_train.shape[3]),
    )
    final_input_channels = int(final_X_train.shape[1])
    final_prior, final_prior_meta = _resolve_fold_prior(
        prior,
        args,
        final_X_train,
        y_train,
        final_train_idx,
        normalization_params,
        len(fold_records),
    )
    final_train_x = final_X_train
    if bool(getattr(args, "augmentation_enabled", False)):
        final_train_x = augment_training_tensor(
            final_train_x,
            noise_std=float(getattr(args, "augmentation_noise_std", 0.01) or 0.01),
        )
    final_weights = None
    if isinstance(normalization_params, dict) and normalization_params.get("deposit_loss_weights") is not None:
        all_weights = np.asarray(normalization_params["deposit_loss_weights"], dtype=np.float64)
        if len(all_weights) == len(y_train):
            final_weights = torch.as_tensor(all_weights, dtype=torch.float32)
    if final_weights is None:
        final_dataset = TensorDataset(final_train_x, y_train)
    else:
        final_dataset = TensorDataset(final_train_x, y_train, final_weights)
    final_loader = create_training_data_loader(
        final_dataset,
        args.batchsize,
        args.model,
        seed=int(getattr(args, "spatial_random_state", 0) or 0) + 10000,
        positives_per_batch=getattr(args, "nnpucnn_positives_per_batch", None),
        unlabeled_coverage=getattr(args, "nnpucnn_unlabeled_coverage", None),
    )
    best_model = instantiate_model(args.model, final_prior, final_input_shape).to(device)
    final_optimizer = build_adam_optimizer(best_model, args)
    final_loss_func = create_loss_function(args, final_prior, labels=y_train)
    final_train_losses = []
    print(
        f"内层验证完成：各折最佳轮次={best_epochs}；"
        f"以中位数 {selected_epoch} 轮在完整外层训练分区从头重训。"
    )
    for epoch in range(selected_epoch):
        if stop_token is not None and stop_token.is_set():
            print("最终外层训练分区重训被停止；不评估外层留出集。")
            return None
        final_train_losses.append(
            float(train_model(best_model, final_loader, final_optimizer, final_loss_func, device, epoch))
        )
    save_loss_diagnostics(
        final_loss_func,
        os.path.join(
            model_dir,
            (
                "rn_annpu_diagnostics.json"
                if str(args.model).lower().startswith("rn")
                else "loss_diagnostics_outer_refit.json"
            ),
        ),
        context={
            "stage": "outer_train_refit",
            "selected_epoch": int(selected_epoch),
            "loader": training_loader_diagnostics(final_loader),
            "model_key": str(args.model),
            "existing_pucnn_model_replaced": False,
            "existing_pu_loss_replaced": False,
        },
    )

    best_model_state = copy.deepcopy(best_model.state_dict())
    best_payload = best_model_state
    if is_image_model(args.model):
        best_payload = {
            "model_state": best_model_state,
            "input_channels": final_input_channels,
            "input_height": int(final_X_train.shape[2]),
            "input_width": int(final_X_train.shape[3]),
        }
    _save_torch_payload(best_payload, os.path.join(model_dir, "best_model.pth"))
    refit_protocol = {
        "selection_scope": "inner_validation_only",
        "outer_test_used_for_selection": False,
        "inner_fold_best_epochs": best_epochs,
        "selected_epoch_rule": "rounded_median_of_inner_fold_best_epochs",
        "selected_epoch": selected_epoch,
        "refit_scope": "complete_outer_training_partition",
        "refit_from_scratch": True,
        "refit_train_size": int(len(X_train)),
        "refit_prior": float(final_prior),
        "refit_prior_metadata": final_prior_meta,
        "refit_fault_calibration": final_fault_meta,
        "refit_train_losses": final_train_losses,
        "selected_hyperparameters": dict(_candidate_params or {}),
        "outer_evaluation_count": 1 if outer_test_available else 0,
    }
    with open(os.path.join(model_dir, "nested_selection_refit_protocol.json"), "w", encoding="utf-8") as handle:
        json.dump(refit_protocol, handle, ensure_ascii=False, indent=2)

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
        "best_val_losses": cv_best_val_losses,
        "best_val_errors": cv_best_val_errors,
        "best_val_recalls": cv_best_val_recalls,
        "fold_spatial_metrics": cv_spatial_metrics,
        "fold_prior_estimates": fold_prior_estimates,
        "inner_fold_best_epochs": best_epochs,
        "selected_refit_epoch": selected_epoch,
        "final_refit_scope": "complete_outer_training_partition",
        "outer_test_used_for_selection": False,
        "selected_hyperparameters": dict(_candidate_params or {}),
        "cv_metric_scope": "validation" if validation_only_cv else "external_test_or_validation",
        "tie_aware_center_selection": bool(
            getattr(args, "tie_aware_center_selection", False)
        ),
        **val_summary,
        **spatial_summary,
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
        print(f"交叉验证曲线数据已保存到: {curve_artifacts['csv']}")
        print(
            "CV Loss/Accuracy/Recall 曲线数据 CSV 已保存到: "
            f"{curve_artifacts['loss_csv']}, {curve_artifacts['accuracy_csv']}, {curve_artifacts['recall_csv']}"
        )
        print(
            f"CV Loss/Accuracy/Recall 曲线已保存到: "
            f"{curve_artifacts['loss_plot']}, {curve_artifacts['accuracy_plot']}, {curve_artifacts['recall_plot']}"
        )
        if curve_artifacts.get("spatial_cv_xlsx"):
            print(f"空间CV SR/PAF/EI Excel 已保存到: {curve_artifacts['spatial_cv_xlsx']}")
    else:
        try:
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
            device,
            normalization_params=normalization_params,
            validation_selected_tau=spatial_summary.get("cv_validation_tau_median"),
        )

    return best_model
