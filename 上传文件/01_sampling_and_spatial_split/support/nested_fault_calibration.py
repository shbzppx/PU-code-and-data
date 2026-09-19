"""Nested fault-decay calibration for CV folds (OGR R1.5).

Calibrate exponential decay length using train deposits only, rematerialize the
decay channel from the (raw) Euclidean-distance channel, and persist fold params.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _windows_long_path(path: str) -> str:
    path = os.path.abspath(os.path.normpath(str(path)))
    if os.name != "nt":
        return path
    if path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path.lstrip("\\")
    return "\\\\?\\" + path


def _ensure_dir(path: str) -> str:
    path = os.path.abspath(os.path.normpath(str(path)))
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except OSError:
        if os.name != "nt":
            raise
        os.makedirs(_windows_long_path(path), exist_ok=True)
        return path


def _open_path(path: str, mode: str = "r", **kwargs):
    try:
        return open(path, mode, **kwargs)
    except OSError:
        if os.name != "nt":
            raise
        return open(_windows_long_path(path), mode, **kwargs)

DIST_ALIASES = (
    "fault_euclidean_distance_m",
    "euclidean_distance_m",
    "fault_euclidean_distance",
    "euclidean_distance",
)
DECAY_ALIASES = (
    "fault_exponential_decay",
    "exponential_decay",
    "fault_decay",
)
INTER_ALIASES = (
    "fault_intersection_density",
    "intersection_density",
    "fault_intersection",
)

FAULT_ABLATION_MODES = (
    "fault_full",
    "fault_decay_only",
    "fault_distance_only",
    "fault_intersection_only",
    "fault_none",
)


def _norm_name(name: object) -> str:
    return str(name or "").strip().lower().replace("-", "_").replace(" ", "_")


def _match_alias(name: str, aliases: Sequence[str]) -> bool:
    n = _norm_name(name)
    for alias in aliases:
        a = _norm_name(alias)
        if n == a or a in n or n.endswith(a):
            return True
    return False


def find_fault_channel_indices(channel_names: Optional[Sequence[str]]) -> Dict[str, Optional[int]]:
    names = list(channel_names or [])
    dist_idx = decay_idx = inter_idx = None
    for i, name in enumerate(names):
        if dist_idx is None and _match_alias(name, DIST_ALIASES):
            dist_idx = i
        elif decay_idx is None and _match_alias(name, DECAY_ALIASES):
            decay_idx = i
        elif inter_idx is None and _match_alias(name, INTER_ALIASES):
            inter_idx = i
    return {"distance": dist_idx, "decay": decay_idx, "intersection": inter_idx}


def is_fault_channel_name(name: object) -> bool:
    return (
        _match_alias(str(name), DIST_ALIASES)
        or _match_alias(str(name), DECAY_ALIASES)
        or _match_alias(str(name), INTER_ALIASES)
        or "fault_" in _norm_name(name)
    )


def resolve_fault_ablation_indices(
    channel_names: Sequence[str],
    mode: str,
    *,
    keep_distance_for_nested: bool = False,
) -> List[int]:
    """Return 0-based channel indices to keep for a fault ablation mode."""
    names = list(channel_names or [])
    mode_key = str(mode or "").strip().lower()
    idxs = find_fault_channel_indices(names)
    fault_idxs = {i for i in idxs.values() if i is not None}
    non_fault = [i for i in range(len(names)) if i not in fault_idxs]

    if mode_key in {"", "fault_full", "full", "full_embedding"}:
        keep_fault = sorted(fault_idxs)
    elif mode_key in {"fault_decay_only", "decay_only", "fault_decay"}:
        keep_fault = [idxs["decay"]] if idxs["decay"] is not None else []
        if keep_distance_for_nested and idxs["distance"] is not None and idxs["distance"] not in keep_fault:
            keep_fault = [idxs["distance"]] + keep_fault
    elif mode_key in {"fault_distance_only", "distance_only", "fault_distance"}:
        keep_fault = [idxs["distance"]] if idxs["distance"] is not None else []
    elif mode_key in {"fault_intersection_only", "intersection_only", "fault_intersection"}:
        keep_fault = [idxs["intersection"]] if idxs["intersection"] is not None else []
    elif mode_key in {"fault_none", "no_fault", "none"}:
        keep_fault = []
    else:
        keep_fault = sorted(fault_idxs)

    return sorted(non_fault + [i for i in keep_fault if i is not None])


def unique_positive_deposit_xy(
    labels,
    positions: Optional[np.ndarray],
    mineral_ids: Optional[np.ndarray] = None,
    indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Deduplicate positive sample coordinates (prefer unique mineral ids)."""
    if positions is None:
        return np.empty((0, 2), dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] < 2 or len(pos) == 0:
        return np.empty((0, 2), dtype=np.float64)

    y = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)
    y = np.asarray(y).reshape(-1)
    if indices is not None:
        idx = np.asarray(indices, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < len(y)) & (idx < len(pos))]
    else:
        idx = np.arange(min(len(y), len(pos)), dtype=np.int64)

    pos_mask = y[idx] == 1
    idx = idx[pos_mask]
    if len(idx) == 0:
        return np.empty((0, 2), dtype=np.float64)

    coords = pos[idx][:, :2]
    if mineral_ids is not None:
        mids = np.asarray(mineral_ids, dtype=np.int64).reshape(-1)
        if len(mids) >= int(np.max(idx)) + 1:
            mid_sub = mids[idx]
            kept = []
            seen = set()
            for row, mid in zip(coords, mid_sub):
                key = int(mid) if int(mid) >= 0 else (round(float(row[0]), 3), round(float(row[1]), 3))
                if key in seen:
                    continue
                seen.add(key)
                kept.append(row)
            return np.asarray(kept, dtype=np.float64) if kept else np.empty((0, 2), dtype=np.float64)

    # fallback: round-coordinate dedupe
    keys = np.round(coords, 3)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    return coords[np.sort(unique_idx)]


def _clamp_quantile(quantile: float) -> float:
    q = float(quantile)
    if not np.isfinite(q):
        return 0.8
    return float(np.clip(q, 1e-6, 1.0))


def nested_calibrate_fault_fold(
    *,
    fold_name: str,
    output_dir: str,
    train_deposit_xy: np.ndarray,
    forbidden_deposit_xy: Optional[np.ndarray] = None,
    fault_path: str,
    quantile: float = 0.8,
    meters_per_coordinate_unit: float = 1.0,
) -> Dict[str, object]:
    """Calibrate exponential decay length using train deposits only and persist fold params."""
    import sys
    from pathlib import Path

    feature_dir = Path(__file__).resolve().parents[1] / "feature"
    if str(feature_dir) not in sys.path:
        sys.path.append(str(feature_dir))
    from fault_features import (
        build_fault_feature_maps,
        calibrate_decay_length_from_deposits,
        fault_vertices_from_lines,
        _read_xy_frame,
        _ensure_dir,
    )

    _ensure_dir(output_dir)
    q = _clamp_quantile(quantile)
    train_xy = np.asarray(train_deposit_xy, dtype=np.float64).reshape(-1, 2) if len(train_deposit_xy) else np.empty((0, 2))
    if forbidden_deposit_xy is not None and len(forbidden_deposit_xy) and len(train_xy):
        forbid = np.asarray(forbidden_deposit_xy, dtype=np.float64).reshape(-1, 2)
        for i, xy in enumerate(train_xy):
            if np.any(np.all(np.isclose(forbid, xy, atol=1e-6), axis=1)):
                raise AssertionError(f"Fault calibration leaked val/test deposit at index {i}: {xy.tolist()}")

    fault_df = _read_xy_frame(fault_path)
    verts, _ = fault_vertices_from_lines(fault_df)
    calib = calibrate_decay_length_from_deposits(
        train_xy,
        verts,
        quantile=q,
        meters_per_coordinate_unit=meters_per_coordinate_unit,
    )
    dummy_grid = train_xy[:1] if len(train_xy) else np.zeros((1, 2), dtype=np.float64)
    result = build_fault_feature_maps(
        dummy_grid,
        fault_path,
        deposit_xy_train=train_xy,
        decay_length_m=float(calib["length_m"]),
        output_dir=output_dir,
        fold_name=fold_name,
        meters_per_coordinate_unit=meters_per_coordinate_unit,
    )
    result["fault_params"] = calib
    result["fold_name"] = fold_name
    result["quantile"] = q
    result["n_train_deposits"] = int(len(train_xy))
    result["n_forbidden_deposits"] = int(len(forbidden_deposit_xy) if forbidden_deposit_xy is not None else 0)
    return result


def rematerialize_decay_channel(
    features,
    *,
    raw_distance: np.ndarray,
    decay_idx: int,
    length_m: float,
    decay_mean: float,
    decay_std: float,
    amplitude: float = 1.0,
):
    """Rewrite normalized decay channel from raw distance and calibrated length."""
    import torch

    length = float(max(length_m, 1e-6))
    dist = np.asarray(raw_distance, dtype=np.float64)
    if dist.ndim == 4:
        dist = dist[:, 0]
    if dist.shape[0] != (features.shape[0] if hasattr(features, "shape") else len(features)):
        raise ValueError(
            f"raw_distance batch {dist.shape[0]} != features batch {features.shape[0]}"
        )
    decay_raw = float(amplitude) * np.exp(-dist / length)
    decay_norm = ((decay_raw - float(decay_mean)) / (float(decay_std) + 1e-8)).astype(np.float32)

    if torch.is_tensor(features):
        out = features.clone()
        out[:, int(decay_idx), :, :] = torch.as_tensor(decay_norm, dtype=out.dtype, device=out.device)
        return out
    out = np.array(features, copy=True)
    out[:, int(decay_idx), :, :] = decay_norm
    return out


def capture_raw_fault_distance(
    x_train: np.ndarray,
    x_test: np.ndarray,
    channel_names: Sequence[str],
    *,
    area_features: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Snapshot raw distance maps before z-score normalization."""
    idxs = find_fault_channel_indices(channel_names)
    dist_idx = idxs.get("distance")
    payload: Dict[str, object] = {
        "fault_channel_indices": idxs,
        "fault_distance_raw_train": None,
        "fault_distance_raw_test": None,
        "fault_distance_raw_area": None,
    }
    if dist_idx is None:
        return payload
    payload["fault_distance_raw_train"] = np.asarray(x_train[:, dist_idx], dtype=np.float32).copy()
    payload["fault_distance_raw_test"] = np.asarray(x_test[:, dist_idx], dtype=np.float32).copy()
    if area_features is not None and area_features.ndim == 4 and area_features.shape[1] > dist_idx:
        payload["fault_distance_raw_area"] = np.asarray(area_features[:, dist_idx], dtype=np.float32).copy()
    return payload


def find_encoding_matched_faultfull(pack: str) -> Optional[str]:
    """Find a same-dim faultfull H5 sibling that contains distance + decay channels."""
    pack_abs = os.path.abspath(str(pack or ""))
    directory = os.path.dirname(pack_abs)
    if not directory or not os.path.isdir(directory):
        return None
    stem = os.path.splitext(os.path.basename(pack_abs))[0].lower()
    dim = None
    for candidate_dim in (5, 10, 20, 30, 50):
        if f"dim{candidate_dim}" in stem or f"dim_{candidate_dim}" in stem:
            dim = candidate_dim
            break

    candidates: List[str] = []
    if dim is not None:
        candidates.extend(
            [
                os.path.join(directory, f"A2K_main_dim{dim}_seed42_faultfull.h5"),
                os.path.join(directory, f"A2K_dim{dim}_seed42_faultfull.h5"),
            ]
        )
    try:
        for name in sorted(os.listdir(directory)):
            lower = name.lower()
            if not lower.endswith(".h5"):
                continue
            if "faultfull" not in lower and "fault_full" not in lower:
                continue
            if dim is not None and f"dim{dim}" not in lower and f"dim_{dim}" not in lower:
                continue
            candidates.append(os.path.join(directory, name))
    except OSError:
        return None

    seen = set()
    for path in candidates:
        path = os.path.abspath(path)
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        try:
            import h5py

            with h5py.File(path, "r") as handle:
                names = [
                    x.decode() if hasattr(x, "decode") else str(x)
                    for x in handle.attrs.get("file_names", [])
                ]
        except Exception:
            continue
        idxs = find_fault_channel_indices(names)
        if idxs.get("distance") is not None and idxs.get("decay") is not None:
            return path
    return None


def drop_auxiliary_fault_distance_channel(
    features,
    *,
    normalization_params: Optional[dict] = None,
    secondary=None,
    mutate_shared_meta: bool = False,
):
    """Drop Euclidean-distance channel after nested λ rewrite so train/predict keep decay only.

    Distance is only an auxiliary input for rematerializing the decay map. Once decay has been
    rewritten fold-wise, keeping distance would leak a second fault channel into the model.

    By default this only slices returned tensors and does NOT mutate shared normalization
    metadata (so multi-fold CV can keep rematerializing from the original 13-channel tensors).
    """
    import torch

    params = normalization_params if isinstance(normalization_params, dict) else {}
    source_idxs = params.get("fault_channel_indices_source") or params.get("fault_channel_indices") or {}
    if not isinstance(source_idxs, dict):
        source_idxs = {}
    dist_idx = source_idxs.get("distance")
    if dist_idx is None:
        # Fall back to names on the current feature tensor width.
        names = list(
            params.get("selected_channel_names_source")
            or params.get("selected_channel_names")
            or []
        )
        dist_idx = find_fault_channel_indices(names).get("distance")
    if dist_idx is None:
        return features, secondary, params

    dist_idx = int(dist_idx)
    n_channels = None
    if hasattr(features, "shape") and len(getattr(features, "shape", ())) >= 2:
        n_channels = int(features.shape[1])
    if n_channels is None or dist_idx < 0 or dist_idx >= int(n_channels):
        return features, secondary, params

    def _drop_channel(tensor):
        if tensor is None:
            return None
        if torch.is_tensor(tensor):
            keep = [i for i in range(int(tensor.shape[1])) if i != dist_idx]
            return tensor[:, keep, ...]
        arr = np.asarray(tensor)
        if arr.ndim < 2:
            return arr
        keep = [i for i in range(int(arr.shape[1])) if i != dist_idx]
        return arr[:, keep, ...]

    features_out = _drop_channel(features)
    secondary_out = _drop_channel(secondary)

    if mutate_shared_meta:
        area = params.get("spatial_cv_area_features")
        if area is not None and hasattr(area, "shape") and int(area.shape[1]) == n_channels:
            params["spatial_cv_area_features"] = _drop_channel(area)
        names = list(params.get("selected_channel_names") or [])
        if names and len(names) == n_channels:
            params["selected_channel_names"] = [n for i, n in enumerate(names) if i != dist_idx]
        mean = params.get("mean")
        std = params.get("std")
        if isinstance(mean, (list, tuple)) and len(mean) == n_channels:
            params["mean"] = [v for i, v in enumerate(mean) if i != dist_idx]
        if isinstance(std, (list, tuple)) and len(std) == n_channels:
            params["std"] = [v for i, v in enumerate(std) if i != dist_idx]
        rebuilt_names = list(params.get("selected_channel_names") or [])
        if rebuilt_names:
            params["fault_channel_indices"] = find_fault_channel_indices(rebuilt_names)
        params["fault_distance_dropped_after_nested"] = True

    return features_out, secondary_out, params


def align_area_features_after_distance_drop(area_features, normalization_params: Optional[dict]):
    """Return area features with auxiliary distance channel removed (local copy, no shared mutate).

    Used so spatial-CV scoring tensors match decay_only models after nested λ rewrite.
    """
    if area_features is None:
        return None
    dropped, _, _ = drop_auxiliary_fault_distance_channel(
        area_features,
        normalization_params=normalization_params if isinstance(normalization_params, dict) else {},
        mutate_shared_meta=False,
    )
    return dropped


def should_drop_distance_after_nested(fault_ablation: object) -> bool:
    """True when nested calibration is only scaffolding for decay_only training."""
    mode = str(fault_ablation or "").strip().lower()
    return mode in {"fault_decay_only", "decay_only", "fault_decay"}


def apply_nested_fault_calibration(
    *,
    fold_name: str,
    output_dir: str,
    fault_path: str,
    X_train,
    y_train,
    train_idx: Sequence[int],
    val_idx: Optional[Sequence[int]] = None,
    X_test=None,
    normalization_params: Optional[dict] = None,
    forbidden_deposit_xy: Optional[np.ndarray] = None,
    quantile: float = 0.8,
    fault_ablation: str = "",
    drop_distance_after: Optional[bool] = None,
) -> Tuple[object, object, Optional[object], Dict[str, object]]:
    """Calibrate on fold-train deposits and rematerialize decay on train/val/test tensors.

    When ``fault_ablation`` is decay-only (or ``drop_distance_after=True``), the Euclidean
    distance channel is removed after rematerialization so train/predict see only decay.
    """
    params = normalization_params if isinstance(normalization_params, dict) else {}
    # Prefer immutable source indices so multi-fold CV can rematerialize repeatedly.
    source_names = list(
        params.get("selected_channel_names_source")
        or params.get("selected_channel_names")
        or params.get("available_channel_names")
        or []
    )
    idxs = params.get("fault_channel_indices_source") or params.get("fault_channel_indices")
    if not isinstance(idxs, dict) or idxs.get("distance") is None or idxs.get("decay") is None:
        idxs = find_fault_channel_indices(source_names)
    dist_idx = idxs.get("distance") if isinstance(idxs, dict) else None
    decay_idx = idxs.get("decay") if isinstance(idxs, dict) else None
    if dist_idx is None or decay_idx is None:
        raise ValueError(
            "嵌套断裂标定需要同时存在距离通道与衰减通道"
            "（fault_euclidean_distance_m / fault_exponential_decay）。"
        )
    # Cache source snapshot once for subsequent folds.
    if "fault_channel_indices_source" not in params:
        params["fault_channel_indices_source"] = {
            "distance": int(dist_idx),
            "decay": int(decay_idx),
            "intersection": (
                int(idxs["intersection"]) if idxs.get("intersection") is not None else None
            ),
        }
    if source_names and "selected_channel_names_source" not in params:
        params["selected_channel_names_source"] = list(source_names)
    if not fault_path or not os.path.exists(fault_path):
        raise FileNotFoundError(f"嵌套断裂标定需要有效断裂线文件: {fault_path}")

    q = _clamp_quantile(quantile)
    distance_protocol = params.get("distance_protocol") or {}
    meters_per_coordinate_unit = distance_protocol.get("meters_per_coordinate_unit")
    if meters_per_coordinate_unit is None:
        meters_per_coordinate_unit = params.get("meters_per_coordinate_unit")
    if meters_per_coordinate_unit is None:
        raise ValueError(
            "嵌套断裂标定缺少 meters_per_coordinate_unit；"
            "不能把坐标距离静默当作米。"
        )
    meters_per_coordinate_unit = float(meters_per_coordinate_unit)
    if not np.isfinite(meters_per_coordinate_unit) or meters_per_coordinate_unit <= 0:
        raise ValueError("meters_per_coordinate_unit must be finite and greater than zero.")
    train_positions = params.get("train_positions")
    train_mineral_ids = params.get("train_mineral_ids")
    train_xy = unique_positive_deposit_xy(y_train, train_positions, train_mineral_ids, train_idx)
    if train_xy.size == 0 and params.get("train_mineral_positions") is not None:
        # Fallback: all train minerals (for holdout without per-sample positives in idx)
        train_xy = np.asarray(params.get("train_mineral_positions"), dtype=np.float64).reshape(-1, 2)

    forbid = forbidden_deposit_xy
    if forbid is None:
        parts = []
        if val_idx is not None:
            parts.append(unique_positive_deposit_xy(y_train, train_positions, train_mineral_ids, val_idx))
        test_minerals = params.get("test_mineral_positions")
        if X_test is not None and test_minerals is not None and len(test_minerals):
            parts.append(np.asarray(test_minerals, dtype=np.float64).reshape(-1, 2))
        forbid = np.vstack(parts) if parts else np.empty((0, 2), dtype=np.float64)

    # Short dirname keeps leave-one-camp grid paths under Windows MAX_PATH.
    calib_dir = _ensure_dir(os.path.join(output_dir, "nfc"))
    calib = nested_calibrate_fault_fold(
        fold_name=fold_name,
        output_dir=calib_dir,
        train_deposit_xy=train_xy,
        forbidden_deposit_xy=forbid,
        fault_path=fault_path,
        quantile=q,
        meters_per_coordinate_unit=meters_per_coordinate_unit,
    )
    length_m = float(calib.get("fault_params", {}).get("length_m", calib.get("length_m", 500.0)))
    amplitude = float(calib.get("fault_params", {}).get("amplitude", 1.0))

    raw_train = params.get("fault_distance_raw_train")
    if raw_train is None:
        raise ValueError("缺少 fault_distance_raw_train；请在数据加载归一化前捕获原始距离通道。")

    raw_train_m = np.asarray(raw_train, dtype=np.float64) * meters_per_coordinate_unit
    if raw_train_m.ndim == 4:
        raw_train_m = raw_train_m[:, 0]
    fold_decay_train = amplitude * np.exp(-raw_train_m[np.asarray(train_idx, dtype=np.int64)] / length_m)
    decay_mean = float(np.mean(fold_decay_train))
    decay_std = float(np.std(fold_decay_train))
    if not np.isfinite(decay_mean) or not np.isfinite(decay_std):
        raise ValueError("Fold-train fault-decay normalization statistics are not finite.")

    X_train_new = rematerialize_decay_channel(
        X_train,
        raw_distance=raw_train_m,
        decay_idx=int(decay_idx),
        length_m=length_m,
        decay_mean=decay_mean,
        decay_std=decay_std,
        amplitude=amplitude,
    )
    X_test_new = X_test
    raw_test = params.get("fault_distance_raw_test")
    if X_test is not None and raw_test is not None and len(X_test) == len(raw_test):
        X_test_new = rematerialize_decay_channel(
            X_test,
            raw_distance=np.asarray(raw_test, dtype=np.float64) * meters_per_coordinate_unit,
            decay_idx=int(decay_idx),
            length_m=length_m,
            decay_mean=decay_mean,
            decay_std=decay_std,
            amplitude=amplitude,
        )

    area = params.get("spatial_cv_area_features")
    raw_area = params.get("fault_distance_raw_area")
    if area is not None and raw_area is not None and len(area) == len(raw_area):
        params["spatial_cv_area_features"] = rematerialize_decay_channel(
            area,
            raw_distance=np.asarray(raw_area, dtype=np.float64) * meters_per_coordinate_unit,
            decay_idx=int(decay_idx),
            length_m=length_m,
            decay_mean=decay_mean,
            decay_std=decay_std,
            amplitude=amplitude,
        )

    meta = {
        "fold_name": fold_name,
        "length_m": length_m,
        "amplitude": amplitude,
        "quantile": q,
        "n_train_deposits": int(len(train_xy)),
        "n_forbidden_deposits": int(len(forbid)) if forbid is not None else 0,
        "params_path": calib.get("params_path"),
        "nested": True,
        "source": "train_deposits_only",
        "meters_per_coordinate_unit": meters_per_coordinate_unit,
        "decay_normalization_scope": "inner_fold_train_only",
        "decay_mean": decay_mean,
        "decay_std": decay_std,
    }
    fold_meta_path = os.path.join(calib_dir, f"nested_meta_{fold_name}.json")
    with _open_path(fold_meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    meta["meta_path"] = fold_meta_path

    history = list(params.get("nested_fault_fold_records") or [])
    history.append(meta)
    params["nested_fault_fold_records"] = history
    params["nested_fault_calibration_active"] = True
    params["last_nested_fault_params"] = meta

    if drop_distance_after is None:
        drop_distance_after = should_drop_distance_after_nested(fault_ablation)
    if drop_distance_after:
        # Inner folds: only slice returned tensors so later folds can rematerialize from 13-ch raw.
        # Final single-shot paths (outer refit / holdout) may mutate shared meta including area features.
        fname = str(fold_name or "")
        mutate_shared = fname.startswith("outer_train_refit") or fname.startswith("holdout")
        X_train_new, X_test_new, params = drop_auxiliary_fault_distance_channel(
            X_train_new,
            normalization_params=params,
            secondary=X_test_new,
            mutate_shared_meta=mutate_shared,
        )
        meta["dropped_distance_after_nested"] = True
        meta["train_channels_after_drop"] = int(
            getattr(X_train_new, "shape", [0, 0])[1]
        ) if hasattr(X_train_new, "shape") else None

    return X_train_new, y_train, X_test_new, meta
