"""Utilities for converting shared dataset bundles into model-specific views."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - 运行环境可能未安装 torch
    class _TorchFallback:
        @staticmethod
        def as_tensor(*args, **kwargs):
            raise ImportError("torch is required for data views")

    class TensorDataset:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class DataLoader:  # type: ignore[override]
        pass

    torch = _TorchFallback()


def _should_drop_last_for_training(sample_count, batch_size) -> bool:
    batch_size = int(batch_size or 0)
    sample_count = int(sample_count or 0)
    return batch_size > 1 and sample_count > 1 and (sample_count % batch_size) == 1


def _as_numpy_pair(data) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if data is None:
        return None
    if isinstance(data, tuple):
        if len(data) != 2:
            raise TypeError("Expected a (features, labels) tuple.")
        features, labels = data
        return np.asarray(features), np.asarray(labels).reshape(-1)
    if isinstance(data, DataLoader):
        features = []
        labels = []
        for batch_inputs, batch_labels in data:
            features.append(batch_inputs.detach().cpu().numpy())
            labels.append(batch_labels.detach().cpu().numpy())
        if not features:
            return None
        return np.concatenate(features, axis=0), np.concatenate(labels, axis=0).reshape(-1)
    raise TypeError(f"Unsupported data container: {type(data)!r}")


def _balance_binary_pair(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    ratio: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features)
    labels = np.asarray(labels).reshape(-1)
    if len(features) == 0 or len(labels) == 0:
        return features, labels

    ratio = float(ratio)
    if ratio <= 0:
        raise ValueError("ratio must be greater than 0.")

    positive_indices = np.where(labels > 0)[0].astype(np.int64)
    negative_indices = np.where(labels == 0)[0].astype(np.int64)
    if len(positive_indices) == 0 or len(negative_indices) == 0:
        return features, labels

    target_negative_count = int(round(len(positive_indices) / ratio))
    target_negative_count = max(1, min(target_negative_count, len(negative_indices)))

    rng = np.random.default_rng(int(seed))
    if target_negative_count >= len(negative_indices):
        sampled_negative_indices = negative_indices
    else:
        sampled_negative_indices = np.sort(
            rng.choice(negative_indices, size=target_negative_count, replace=False).astype(np.int64)
        )

    selected_indices = np.sort(np.concatenate([positive_indices, sampled_negative_indices]).astype(np.int64))
    return features[selected_indices], labels[selected_indices]


def _to_loader(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: Optional[int] = None,
    shuffle: bool = False,
    template_loader: Optional[DataLoader] = None,
) -> DataLoader:
    tensor_x = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    tensor_y = torch.as_tensor((np.asarray(labels).reshape(-1) > 0).astype(np.int64), dtype=torch.long)
    loader = DataLoader(
        TensorDataset(tensor_x, tensor_y),
        batch_size=int(batch_size or getattr(template_loader, "batch_size", 32) or 32),
        shuffle=bool(shuffle),
        num_workers=0,
        drop_last=_should_drop_last_for_training(
            len(tensor_y),
            batch_size or getattr(template_loader, "batch_size", 32),
        ) if shuffle else False,
    )
    if template_loader is not None:
        loader.normalization_stats = getattr(template_loader, "normalization_stats", None)
        loader.normalization_applied = bool(getattr(template_loader, "normalization_applied", False))
        loader.split_info = getattr(template_loader, "split_info", None)
    else:
        loader.normalization_stats = None
        loader.normalization_applied = False
        loader.split_info = None
    return loader


def _augment_training_features(
    features: np.ndarray,
    *,
    noise_std: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.size == 0:
        return features

    noise_std = float(noise_std)
    if noise_std <= 0:
        return features.copy()

    rng = np.random.default_rng(int(seed))
    feature_std = np.std(features, axis=0, dtype=np.float32)
    feature_scale = np.where(np.abs(feature_std) < 1e-6, 1.0, feature_std).astype(np.float32)
    noise = rng.normal(loc=0.0, scale=noise_std, size=features.shape).astype(np.float32)
    augmented = features + noise * feature_scale
    return augmented.astype(np.float32, copy=False)


def prepare_model_views(
    wrapper,
    train_data,
    val_data=None,
    test_data=None,
    *,
    supervised_train_ratio: float = 1.0,
    seed: int = 42,
    augmentation_enabled: bool = False,
    augmentation_noise_std: float = 0.01,
):
    """Return model-specific training/validation/test views.

    PU models keep the shared data as-is. Supervised models get a balanced
    training view by default so comparison runs can stay at 1:1 while the
    shared dataset still retains enough unlabeled background for PU methods.
    """

    data_mode = str(getattr(wrapper, "data_mode", "array") or "array").strip().lower()
    is_pu_model = bool(getattr(wrapper, "is_pu_model", False)) or str(getattr(wrapper, "training_mode", "")).strip().lower() == "pu"

    train_pair = _as_numpy_pair(train_data)
    val_pair = _as_numpy_pair(val_data) if val_data is not None else None
    test_pair = _as_numpy_pair(test_data) if test_data is not None else None

    if train_pair is None:
        return None, None, None

    train_features, train_labels = train_pair
    if not is_pu_model:
        train_features, train_labels = _balance_binary_pair(
            train_features,
            train_labels,
            ratio=supervised_train_ratio,
            seed=seed,
        )
    if augmentation_enabled:
        train_features = _augment_training_features(
            train_features,
            noise_std=augmentation_noise_std,
            seed=seed,
        )

    if data_mode == "loader":
        template_loader = train_data if isinstance(train_data, DataLoader) else None
        train_view = _to_loader(
            train_features,
            train_labels,
            batch_size=getattr(template_loader, "batch_size", None),
            shuffle=True,
            template_loader=template_loader,
        )
        val_view = _to_loader(val_pair[0], val_pair[1], batch_size=getattr(template_loader, "batch_size", None), shuffle=False, template_loader=template_loader) if val_pair is not None else None
        test_view = _to_loader(test_pair[0], test_pair[1], batch_size=getattr(template_loader, "batch_size", None), shuffle=False, template_loader=template_loader) if test_pair is not None else None
        return train_view, val_view, test_view

    train_view = (train_features, train_labels)
    val_view = val_pair
    test_view = test_pair
    return train_view, val_view, test_view


def prepare_training_view(
    wrapper,
    train_data,
    *,
    supervised_train_ratio: float = 1.0,
    seed: int = 42,
    augmentation_enabled: bool = False,
    augmentation_noise_std: float = 0.01,
):
    train_view, _, _ = prepare_model_views(
        wrapper,
        train_data,
        None,
        None,
        supervised_train_ratio=supervised_train_ratio,
        seed=seed,
        augmentation_enabled=augmentation_enabled,
        augmentation_noise_std=augmentation_noise_std,
    )
    return train_view
