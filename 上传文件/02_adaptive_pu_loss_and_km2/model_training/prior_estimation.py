"""Group-aware PU class-prior estimation for nested spatial validation."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def _as_2d_features(features: np.ndarray) -> np.ndarray:
    """Flatten sample features for classical EN/KM2 estimators.

    For CNN patches ``(N, C, H, W)`` we use spatial mean pooling → ``(N, C)``.
    Full HxW flatten (e.g. 9×9×14=1134 dims) makes GroupKFold logistic
    regression and KM2 distance grids prohibitively slow on every seed run.
    """
    array = np.asarray(features)
    if array.ndim == 4:
        array = np.mean(array, axis=(2, 3))
    elif array.ndim == 1:
        return array.reshape(-1, 1).astype(np.float64)
    return array.reshape(array.shape[0], -1).astype(np.float64)


def _labels(labels: np.ndarray) -> np.ndarray:
    return np.asarray(labels).reshape(-1)


def _weights(sample_weights: Optional[np.ndarray], size: int) -> np.ndarray:
    if sample_weights is None:
        return np.ones(size, dtype=np.float64)
    values = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if len(values) != size:
        raise ValueError("sample_weights length mismatch for prior estimation.")
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("sample_weights must be finite and greater than zero.")
    return values


def _groups(groups: Optional[Sequence[object]], size: int) -> np.ndarray:
    if groups is None:
        return np.arange(size, dtype=np.int64)
    values = np.asarray(groups).reshape(-1)
    if len(values) != size:
        raise ValueError("groups length mismatch for prior estimation.")
    return values


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def _percentile_ci(values: Sequence[float], alpha: float = 0.05) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan")
    return (
        float(np.quantile(array, float(alpha) / 2.0)),
        float(np.quantile(array, 1.0 - float(alpha) / 2.0)),
    )


def _group_splits(groups: np.ndarray, labels: np.ndarray, max_splits: int = 5):
    from sklearn.model_selection import GroupKFold

    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("Group-aware prior estimation needs at least two spatial groups.")
    split_count = min(max(int(max_splits), 2), int(len(unique)))
    splitter = GroupKFold(n_splits=split_count)
    dummy = np.zeros((len(labels), 1), dtype=np.float64)
    return list(splitter.split(dummy, labels, groups))


def _cluster_bootstrap_indices(groups: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    unique = np.unique(groups)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    index_parts = []
    boot_groups = []
    for boot_id, group in enumerate(sampled):
        indices = np.where(groups == group)[0]
        if len(indices):
            index_parts.append(indices)
            boot_groups.extend([boot_id] * len(indices))
    if not index_parts:
        return np.arange(len(groups), dtype=np.int64), np.arange(len(groups), dtype=np.int64)
    return np.concatenate(index_parts).astype(np.int64), np.asarray(boot_groups, dtype=np.int64)


def _en_point(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    sample_weights: np.ndarray,
    *,
    random_state: int,
) -> Tuple[float, Dict[str, object]]:
    from sklearn.linear_model import LogisticRegression

    positive = labels == 1
    unlabeled = (labels == -1) | (labels == 0)
    keep = positive | unlabeled
    features = features[keep]
    labels = labels[keep]
    groups = groups[keep]
    sample_weights = sample_weights[keep]
    surrogate = (labels == 1).astype(np.int64)
    if not np.any(surrogate == 1) or not np.any(surrogate == 0):
        raise ValueError("Elkan-Noto needs both labeled-positive and unlabeled samples.")

    p_s1 = _weighted_mean(surrogate, sample_weights)
    held_positive_probabilities = []
    held_positive_weights = []
    splits = _group_splits(groups, surrogate)
    for split_id, (train_index, holdout_index) in enumerate(splits):
        train_labels = surrogate[train_index]
        if len(np.unique(train_labels)) < 2:
            continue
        classifier = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=int(random_state + split_id))
        classifier.fit(features[train_index], train_labels, sample_weight=sample_weights[train_index])
        held_positive = holdout_index[surrogate[holdout_index] == 1]
        if len(held_positive) == 0:
            continue
        held_positive_probabilities.extend(classifier.predict_proba(features[held_positive])[:, 1].tolist())
        held_positive_weights.extend(sample_weights[held_positive].tolist())
    if not held_positive_probabilities:
        raise ValueError("No held-out positive predictions were available for group cross-fitting.")
    c_hat = float(
        np.clip(
            _weighted_mean(
                np.asarray(held_positive_probabilities),
                np.asarray(held_positive_weights),
            ),
            1e-6,
            1.0,
        )
    )
    point = float(np.clip(p_s1 / c_hat, 1e-4, 0.999))
    return point, {
        "p_s1_weighted": p_s1,
        "c_hat_group_crossfit": c_hat,
        "crossfit_fold_count": int(len(splits)),
        "held_out_positive_prediction_count": int(len(held_positive_probabilities)),
        "feature_dim": int(features.shape[1]),
        "feature_reduction": "spatial_mean_pool_if_patch",
    }


def estimate_prior_elkan_noto(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    groups: Optional[Sequence[object]] = None,
    sample_weights: Optional[np.ndarray] = None,
    n_bootstrap: int = 30,
    random_state: int = 42,
    verbose: bool = False,
) -> Dict[str, object]:
    """Elkan-Noto with spatial-group cross-fitting and optional cluster bootstrap."""
    feature_array = _as_2d_features(features)
    label_array = _labels(labels)
    if len(feature_array) != len(label_array):
        raise ValueError("features/labels length mismatch for Elkan-Noto.")
    group_array = _groups(groups, len(label_array))
    weight_array = _weights(sample_weights, len(label_array))
    if verbose:
        print(
            f"先验估计 EN：n={len(label_array)}, dim={feature_array.shape[1]}, "
            f"groups={len(np.unique(group_array))}, bootstrap={max(int(n_bootstrap), 0)}"
        )
    point, diagnostics = _en_point(
        feature_array,
        label_array,
        group_array,
        weight_array,
        random_state=int(random_state),
    )
    bootstrap_points = []
    n_boot = max(int(n_bootstrap), 0)
    if n_boot > 0:
        rng = np.random.default_rng(int(random_state))
        for bootstrap_index in range(n_boot):
            sampled_indices, bootstrap_groups = _cluster_bootstrap_indices(group_array, rng)
            try:
                value, _ = _en_point(
                    feature_array[sampled_indices],
                    label_array[sampled_indices],
                    bootstrap_groups,
                    weight_array[sampled_indices],
                    random_state=int(random_state + 1000 + bootstrap_index),
                )
                bootstrap_points.append(value)
            except ValueError:
                continue
            if verbose and (bootstrap_index + 1) % 10 == 0:
                print(f"  EN bootstrap {bootstrap_index + 1}/{n_boot}")
    ci_low, ci_high = _percentile_ci(bootstrap_points) if bootstrap_points else (float("nan"), float("nan"))
    return {
        "method": "elkan_noto_group_crossfit",
        "point": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "status": "estimable" if bootstrap_points else "point_only",
        "assumptions": [
            "SCAR: labeled positives are selected independently of features conditional on the positive class",
            "spatial groups are the resampling and cross-fitting units",
            "sample weights restore the intended outer-training population after sampling",
            "patch features use spatial mean pooling before EN (not full HxW flatten)",
        ],
        "diagnostics": {
            **diagnostics,
            "n_positive": int(np.sum(label_array == 1)),
            "n_unlabeled": int(np.sum((label_array == -1) | (label_array == 0))),
            "group_count": int(len(np.unique(group_array))),
            "bootstrap_requested": int(n_boot),
            "bootstrap_successful": int(len(bootstrap_points)),
            "bootstrap_mean": float(np.mean(bootstrap_points)) if bootstrap_points else float("nan"),
            "bootstrap_std": float(np.std(bootstrap_points, ddof=1)) if len(bootstrap_points) > 1 else 0.0,
            "random_state": int(random_state),
        },
    }


def _km2_point(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    *,
    random_state: int,
    n_gamma: int,
) -> Tuple[float, Dict[str, object]]:
    positive = labels == 1
    unlabeled = (labels == -1) | (labels == 0)
    if not np.any(positive) or not np.any(unlabeled):
        raise ValueError("KM2 needs both labeled-positive and unlabeled samples.")
    rng = np.random.default_rng(int(random_state))
    positive_features = features[positive]
    unlabeled_features = features[unlabeled]
    positive_weights = sample_weights[positive]
    unlabeled_weights = sample_weights[unlabeled]
    max_size = 600
    if len(positive_features) > max_size:
        selected = rng.choice(len(positive_features), size=max_size, replace=False, p=positive_weights / positive_weights.sum())
        positive_features, positive_weights = positive_features[selected], positive_weights[selected]
    if len(unlabeled_features) > max_size:
        selected = rng.choice(len(unlabeled_features), size=max_size, replace=False, p=unlabeled_weights / unlabeled_weights.sum())
        unlabeled_features, unlabeled_weights = unlabeled_features[selected], unlabeled_weights[selected]
    sample = np.concatenate([positive_features[:150], unlabeled_features[:150]], axis=0)
    differences = sample[:, None, :] - sample[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    median_distance = float(np.median(distances[distances > 0])) if np.any(distances > 0) else 1.0
    median_distance = max(median_distance, 1e-6)
    landmarks = np.concatenate([positive_features[:80], unlabeled_features[:80]], axis=0)
    gammas = np.geomspace(0.1 / median_distance**2, 10.0 / median_distance**2, num=max(int(n_gamma), 4))
    best = None
    for gamma in gammas:
        phi_p = np.exp(-gamma * np.sum((positive_features[:, None, :] - landmarks[None, :, :]) ** 2, axis=2))
        phi_u = np.exp(-gamma * np.sum((unlabeled_features[:, None, :] - landmarks[None, :, :]) ** 2, axis=2))
        mean_p = np.average(phi_p, axis=0, weights=positive_weights)
        mean_u = np.average(phi_u, axis=0, weights=unlabeled_weights)
        # KM2-style ratio of RKHS means (heuristic sensitivity estimator)
        ratio = float(np.clip(np.sum(mean_u * mean_p) / max(np.sum(mean_p * mean_p), 1e-12), 1e-4, 0.999))
        score = abs(ratio - 0.5)
        if best is None or score < best[0]:
            best = (score, ratio, float(gamma))
    assert best is not None
    return float(best[1]), {
        "best_gamma": best[2],
        "median_distance": median_distance,
        "feature_dim": int(features.shape[1]),
        "n_gamma": int(len(gammas)),
    }


def estimate_prior_km2(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    groups: Optional[Sequence[object]] = None,
    sample_weights: Optional[np.ndarray] = None,
    n_bootstrap: int = 20,
    n_gamma: int = 8,
    random_state: int = 42,
    verbose: bool = False,
) -> Dict[str, object]:
    """KM-family mixture-proportion heuristic with optional cluster bootstrap."""
    feature_array = _as_2d_features(features)
    label_array = _labels(labels)
    if len(feature_array) != len(label_array):
        raise ValueError("features/labels length mismatch for KM2.")
    group_array = _groups(groups, len(label_array))
    weight_array = _weights(sample_weights, len(label_array))
    if verbose:
        print(
            f"先验估计 KM2：n={len(label_array)}, dim={feature_array.shape[1]}, "
            f"bootstrap={max(int(n_bootstrap), 0)}"
        )
    point, diagnostics = _km2_point(
        feature_array,
        label_array,
        weight_array,
        random_state=int(random_state),
        n_gamma=int(n_gamma),
    )
    bootstrap_points = []
    n_boot = max(int(n_bootstrap), 0)
    if n_boot > 0:
        rng = np.random.default_rng(int(random_state))
        for bootstrap_index in range(n_boot):
            sampled_indices, _ = _cluster_bootstrap_indices(group_array, rng)
            try:
                value, _ = _km2_point(
                    feature_array[sampled_indices],
                    label_array[sampled_indices],
                    weight_array[sampled_indices],
                    random_state=int(random_state + 2000 + bootstrap_index),
                    n_gamma=int(n_gamma),
                )
                bootstrap_points.append(value)
            except ValueError:
                continue
            if verbose and (bootstrap_index + 1) % 10 == 0:
                print(f"  KM2 bootstrap {bootstrap_index + 1}/{n_boot}")
    ci_low, ci_high = _percentile_ci(bootstrap_points) if bootstrap_points else (float("nan"), float("nan"))
    return {
        "method": "km2_group_bootstrap_heuristic",
        "point": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "status": "sensitivity_estimator",
        "assumptions": [
            "positive and unlabeled feature distributions admit a mixture-proportion interpretation",
            "spatial groups are the bootstrap unit",
            "implementation is a KM-family heuristic and is reported as sensitivity evidence",
            "patch features use spatial mean pooling before KM2",
        ],
        "diagnostics": {
            **diagnostics,
            "n_positive": int(np.sum(label_array == 1)),
            "n_unlabeled": int(np.sum((label_array == -1) | (label_array == 0))),
            "group_count": int(len(np.unique(group_array))),
            "bootstrap_requested": int(n_boot),
            "bootstrap_successful": int(len(bootstrap_points)),
            "bootstrap_mean": float(np.mean(bootstrap_points)) if bootstrap_points else float("nan"),
            "bootstrap_std": float(np.std(bootstrap_points, ddof=1)) if len(bootstrap_points) > 1 else 0.0,
            "random_state": int(random_state),
        },
    }


def resolve_prior_triple(
    *,
    prior_mode: str,
    features: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    groups: Optional[Sequence[object]] = None,
    sample_weights: Optional[np.ndarray] = None,
    tuning_prior: Optional[float] = None,
    geological_favorable_area_ratio: Optional[float] = None,
    random_state: int = 42,
    compare_estimators: bool = False,
    disagreement_absolute: float = 0.05,
    n_bootstrap: int = 0,
    verbose: bool = True,
) -> Dict[str, object]:
    """Resolve statistical, tuning and geological priors without conflating them.

    ``n_bootstrap`` defaults to 0: training only needs the point estimate.
    Pass ``n_bootstrap>0`` for CI / dedicated prior-sensitivity reports.
    """
    mode = str(prior_mode or "manual").strip().lower()
    estimates: Dict[str, Dict[str, object]] = {}
    can_estimate = features is not None and labels is not None
    n_boot = max(int(n_bootstrap), 0)
    if can_estimate and (compare_estimators or mode in {"estimate_en", "elkan_noto", "en"}):
        if verbose:
            print(
                f"开始估计先验（Elkan-Noto"
                f"{' + KM2对照' if compare_estimators or mode in {'estimate_km2', 'km2'} else ''}"
                f"；点估计"
                f"{'' if n_boot <= 0 else f' + bootstrap×{n_boot}'}"
                f"）..."
            )
        estimates["elkan_noto"] = estimate_prior_elkan_noto(
            features,
            labels,
            groups=groups,
            sample_weights=sample_weights,
            random_state=random_state,
            n_bootstrap=n_boot,
            verbose=verbose,
        )
    if can_estimate and (compare_estimators or mode in {"estimate_km2", "km2"}):
        estimates["km2"] = estimate_prior_km2(
            features,
            labels,
            groups=groups,
            sample_weights=sample_weights,
            random_state=random_state,
            n_bootstrap=n_boot,
            verbose=verbose,
        )

    selected = None
    if mode in {"estimate_en", "elkan_noto", "en"}:
        selected = estimates.get("elkan_noto")
    elif mode in {"estimate_km2", "km2"}:
        selected = estimates.get("km2")
    elif mode == "auto" and can_estimate:
        raise ValueError("Automatic window-ratio prior is disabled; choose EN, KM2, manual or grid explicitly.")

    disagreement = None
    if "elkan_noto" in estimates and "km2" in estimates:
        en_point = float(estimates["elkan_noto"]["point"])
        km2_point = float(estimates["km2"]["point"])
        absolute = abs(en_point - km2_point)
        disagreement = {
            "absolute_difference": absolute,
            "threshold": float(disagreement_absolute),
            "material": bool(absolute > float(disagreement_absolute)),
            "action": "report_pi_sensitivity" if absolute > float(disagreement_absolute) else "none",
        }

    if selected is not None:
        resolved = float(selected["point"])
        tuning_used = resolved
    else:
        tuning_used = None if tuning_prior is None else float(tuning_prior)
        resolved = tuning_used
    return {
        "prior_mode": mode,
        "statistical_prior_estimate": selected,
        "statistical_prior_estimates": estimates,
        "estimator_disagreement": disagreement,
        "sensitivity_required": bool(disagreement and disagreement.get("material")),
        "tuning_prior": tuning_used,
        "geological_favorable_area_ratio": (
            None if geological_favorable_area_ratio is None else float(geological_favorable_area_ratio)
        ),
        "resolved_prior": resolved,
        "manual_prior_ignored_for_resolve": bool(selected is not None and tuning_prior is not None),
        "estimation_scope": "current_training_partition_only",
        "random_state": int(random_state),
        "n_bootstrap": int(n_boot),
    }
