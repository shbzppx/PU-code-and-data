"""Unified model wrappers for deep learning and classic ML models."""

from abc import ABC, abstractmethod
from collections import OrderedDict
import copy
from pathlib import Path
from typing import Optional
import pickle
import shutil
import tempfile

import numpy as np
try:
    import torch
    from torch.utils.data import DataLoader, Dataset, TensorDataset
except ImportError:  # pragma: no cover - 运行环境可能未安装 torch
    class _TorchFallback:
        class cuda:
            @staticmethod
            def is_available():
                return False

    class Dataset:  # type: ignore[override]
        pass

    class TensorDataset:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class DataLoader:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    torch = _TorchFallback()

from .system_pu_loss import AdaptivePULoss, PULoss


def _should_drop_last_for_training(sample_count, batch_size) -> bool:
    batch_size = int(batch_size or 0)
    sample_count = int(sample_count or 0)
    return batch_size > 1 and sample_count > 1 and (sample_count % batch_size) == 1


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


SYSTEM_MODEL_SPECS = OrderedDict(
    [
        ("线性分类器", {"key": "linear", "family": "neural", "complexity": 0.55}),
        ("三层感知机", {"key": "3lp", "family": "neural", "complexity": 0.75}),
        ("多层感知机", {"key": "mlp", "family": "neural", "complexity": 0.95}),
        ("CNN", {"key": "cnn", "family": "neural", "complexity": 1.25}),
        ("CNN-Transformer", {"key": "cnnt", "family": "neural", "complexity": 1.55}),
        ("CNN-TokenTransformer", {"key": "cntt", "family": "neural", "complexity": 1.65}),
        ("PU-CNN", {"key": "pucnn", "family": "neural", "complexity": 1.6}),
        ("PU-CNN-Transformer", {"key": "pucnnt", "family": "neural", "complexity": 1.7}),
        ("PU-CNN-TokenTransformer", {"key": "pucnntransformer", "family": "neural", "complexity": 1.8}),
        ("RF", {"key": "rf", "family": "non_neural", "complexity": 0.8}),
        ("PU-Random Forest", {"key": "purf", "family": "non_neural", "complexity": 1.0}),
        ("One-Class SVM", {"key": "ocsvm", "family": "non_neural", "complexity": 0.85}),
        ("Two-Step PU", {"key": "2step", "family": "non_neural", "complexity": 0.9}),
    ]
)
SYSTEM_NEURAL_MODEL_NAMES = tuple(
    label for label, spec in SYSTEM_MODEL_SPECS.items() if spec["family"] == "neural"
)


def resolve_system_model_name(model_name):
    if model_name in SYSTEM_MODEL_SPECS:
        return model_name
    for label, spec in SYSTEM_MODEL_SPECS.items():
        if model_name == spec["key"]:
            return label
    raise ValueError(f"Unknown system model type: {model_name}")


def apply_model_metadata(wrapper, model_name: str):
    resolved_name = resolve_system_model_name(model_name)
    spec = dict(SYSTEM_MODEL_SPECS.get(resolved_name, {}))
    if "input_kind" not in spec:
        spec["input_kind"] = "window" if resolved_name in {
            "CNN",
            "CNN-Transformer",
            "CNN-TokenTransformer",
            "PU-CNN",
            "PU-CNN-Transformer",
            "PU-CNN-TokenTransformer",
        } else "vector"
    is_pu_family = resolved_name in {"PU-CNN", "PU-CNN-Transformer", "PU-CNN-TokenTransformer", "PU-Random Forest", "Two-Step PU"}
    if "training_mode" not in spec:
        spec["training_mode"] = "pu" if is_pu_family else "supervised"
    if "label_mode" not in spec:
        spec["label_mode"] = "pu" if is_pu_family else "binary"
    wrapper.model_name = resolved_name
    wrapper.model_family = str(spec.get("family", "unknown"))
    wrapper.input_kind = str(spec.get("input_kind", "vector"))
    wrapper.label_mode = str(spec.get("label_mode", "binary"))
    wrapper.training_mode = str(spec.get("training_mode", "supervised"))
    wrapper.is_pu_model = wrapper.training_mode == "pu"
    wrapper.model_spec = spec
    return wrapper


def create_system_model_wrapper(
    model_name,
    dataset_meta=None,
    profile="comparison",
    *,
    fixed_epochs=None,
    persist_artifacts=False,
    artifact_dir=None,
    training_config=None,
    automl_config=None,
):
    resolved_name = resolve_system_model_name(model_name)
    if resolved_name == "PU-CNN":
        from .system_model_adapters import build_pu_cnn_wrapper

        return build_pu_cnn_wrapper(
            dataset_meta=dataset_meta,
            profile=profile,
            fixed_epochs=fixed_epochs,
            persist_artifacts=persist_artifacts,
            artifact_dir=artifact_dir,
            training_config=training_config,
            automl_config=automl_config,
        )
    if resolved_name == "PU-CNN-Transformer":
        from .system_model_adapters import build_pu_cnnt_wrapper

        return build_pu_cnnt_wrapper(
            dataset_meta=dataset_meta,
            profile=profile,
            fixed_epochs=fixed_epochs,
            persist_artifacts=persist_artifacts,
            artifact_dir=artifact_dir,
            training_config=training_config,
            automl_config=automl_config,
        )
    if resolved_name == "PU-CNN-TokenTransformer":
        from .system_model_adapters import build_pu_cnn_transformer_wrapper

        return build_pu_cnn_transformer_wrapper(
            dataset_meta=dataset_meta,
            profile=profile,
            fixed_epochs=fixed_epochs,
            persist_artifacts=persist_artifacts,
            artifact_dir=artifact_dir,
            training_config=training_config,
            automl_config=automl_config,
        )
    if resolved_name in SYSTEM_NEURAL_MODEL_NAMES:
        from .system_model_adapters import build_system_model_wrapper

        return build_system_model_wrapper(
            resolved_name,
            dataset_meta=dataset_meta,
            profile=profile,
            fixed_epochs=fixed_epochs,
            persist_artifacts=persist_artifacts,
            artifact_dir=artifact_dir,
            training_config=training_config,
            automl_config=automl_config,
        )
    if resolved_name == "One-Class SVM":
        return apply_model_metadata(OneClassSVMWrapper(), resolved_name)
    if resolved_name == "RF":
        return apply_model_metadata(RandomForestWrapper(), resolved_name)
    if resolved_name == "PU-Random Forest":
        return apply_model_metadata(PURandomForestWrapper(), resolved_name)
    if resolved_name == "Two-Step PU":
        return apply_model_metadata(TwoStepPUWrapper(), resolved_name)
    raise ValueError(f"Unknown system model type: {model_name}")


class BaseModelWrapper(ABC):
    """Base contract shared by all AutoML model wrappers."""

    data_mode = "array"

    def __init__(self, model_type):
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.model_name = model_type
        self.model_family = "unknown"
        self.input_kind = "vector"
        self.label_mode = "binary"
        self.training_mode = "supervised"
        self.is_pu_model = False
        self.model_spec = {}

    @abstractmethod
    def train(self, train_data, val_data=None, **params):
        """Train the model and return a history dictionary."""

    @abstractmethod
    def predict(self, data):
        """Predict class labels."""

    @abstractmethod
    def predict_proba(self, data):
        """Predict class probabilities."""

    @abstractmethod
    def get_default_params(self):
        """Return default training parameters."""

    @abstractmethod
    def get_param_space(self):
        """Return the Optuna search space."""

    def save_model(self, path):
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError

    def stop(self):
        """Optional stop hook for long-running models."""


class DeepLearningWrapper(BaseModelWrapper):
    """Wrapper for the system's neural models used by comparison and AutoML."""

    data_mode = "loader"

    def __init__(
        self,
        model_class,
        num_classes,
        input_channels,
        image_size,
        *,
        model_name: Optional[str] = None,
        fixed_epochs: Optional[int] = None,
        persist_artifacts: bool = False,
        artifact_dir: Optional[str] = None,
    ):
        super().__init__(model_name or model_class.__name__)
        self.model_class = model_class
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.image_size = image_size
        self.fixed_epochs = fixed_epochs
        self.persist_artifacts = persist_artifacts
        self.artifact_dir = artifact_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.history = {}
        self.current_trainer = None
        self.normalization_stats = None
        self.split_info = None
        self.training_config = {}
        self._stop_requested = False

    def train(self, train_data, val_data=None, **params):
        from ..cnn.trainer import Trainer

        self._stop_requested = False
        train_loader, val_loader = self._prepare_loaders(train_data, val_data, params)
        scheduler_name = params.get("scheduler")
        scheduler_enabled = bool(scheduler_name) and str(scheduler_name).lower() not in {"none", "off", "disabled"}
        requested_save_path = params.get("save_path")
        persist_artifacts = bool(params.get("persist_artifacts", self.persist_artifacts))
        artifact_dir = params.get("artifact_dir") or self.artifact_dir
        temp_dir_path = None

        if requested_save_path:
            save_path = str(requested_save_path)
        elif persist_artifacts:
            save_path = str(artifact_dir or (Path.cwd() / "models" / "comparison"))
        else:
            # AutoML trials should not leave per-run checkpoints behind.
            temp_dir_path = tempfile.mkdtemp(prefix=f"modelcmp_{self.model_type.lower()}_")
            save_path = temp_dir_path

        self.model = self.model_class(
            num_classes=self.num_classes,
            input_channels=self.input_channels,
            image_size=self.image_size,
        )

        trainer = Trainer(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=int(params.get("epochs", 20)),
            lr=float(params.get("learning_rate", 0.001)),
            device=self.device,
            image_size=self.image_size,
            save_path=save_path,
            optimizer_name=params.get("optimizer", "AdamW"),
            scheduler_enabled=scheduler_enabled,
            scheduler_type=scheduler_name or "CosineAnnealingLR",
            lr_step_size=int(params.get("lr_step_size", 10)),
            lr_gamma=float(params.get("lr_gamma", 0.1)),
            early_stopping_patience=int(params.get("early_stopping_patience", 15)),
        )
        try:
            self.current_trainer = trainer
            trainer.train()
            self.history = dict(trainer.history)
            self.is_trained = True
            self.normalization_stats = getattr(train_loader, "normalization_stats", None)
            self.split_info = getattr(train_loader, "split_info", None)
            self.training_config = {
                "epochs": int(params.get("epochs", 20)),
                "learning_rate": float(params.get("learning_rate", 0.001)),
                "batch_size": int(params.get("batch_size", getattr(train_loader, "batch_size", 32) or 32)),
                "optimizer": params.get("optimizer", "AdamW"),
                "scheduler": scheduler_name or "none",
                "lr_step_size": int(params.get("lr_step_size", 10)),
                "lr_gamma": float(params.get("lr_gamma", 0.1)),
                "early_stopping_patience": int(params.get("early_stopping_patience", 15)),
            }
            self.model.normalization_stats = self.normalization_stats

            train_acc = max(self.history.get("train_acc", [0.0])) if self.history.get("train_acc") else 0.0
            val_acc = max(self.history.get("val_acc", [0.0])) if self.history.get("val_acc") else 0.0
            self.history["train_acc"] = train_acc
            self.history["val_acc"] = val_acc
            return self.history
        finally:
            self.current_trainer = None
            if temp_dir_path is not None:
                shutil.rmtree(temp_dir_path, ignore_errors=True)

    def _as_loader(self, data, *, batch_size, shuffle=False):
        if data is None:
            return None
        if isinstance(data, DataLoader):
            return self._clone_loader(data, batch_size=batch_size)
        if isinstance(data, tuple):
            if len(data) != 2:
                raise TypeError("Tuple inputs must be (X, y).")
            features, labels = data
            features = torch.as_tensor(np.asarray(features), dtype=torch.float32)
            labels = torch.as_tensor((np.asarray(labels).reshape(-1) > 0).astype(np.int64), dtype=torch.long)
            dataset = TensorDataset(features, labels)
            loader = DataLoader(
                dataset,
                batch_size=int(batch_size),
                shuffle=bool(shuffle),
                num_workers=0,
                drop_last=_should_drop_last_for_training(len(dataset), batch_size) if shuffle else False,
            )
            loader.normalization_stats = None
            loader.normalization_applied = False
            loader.split_info = None
            return loader
        raise TypeError("DeepLearningWrapper expects DataLoader or (X, y) tuple inputs.")

    def _clone_loader(self, loader, *, batch_size=None):
        target_batch_size = int(batch_size or loader.batch_size or 32)
        kwargs = {
            "dataset": loader.dataset,
            "batch_size": target_batch_size,
            "num_workers": 0,
            "pin_memory": bool(getattr(loader, "pin_memory", False)),
            "drop_last": bool(getattr(loader, "drop_last", False)),
        }

        sampler_name = loader.sampler.__class__.__name__ if getattr(loader, "sampler", None) is not None else ""
        if sampler_name == "RandomSampler":
            kwargs["shuffle"] = True
        elif sampler_name == "SequentialSampler":
            kwargs["shuffle"] = False
        elif getattr(loader, "sampler", None) is not None:
            kwargs["sampler"] = loader.sampler
        else:
            kwargs["shuffle"] = False

        if bool(kwargs.get("shuffle")):
            kwargs["drop_last"] = _should_drop_last_for_training(len(loader.dataset), target_batch_size)

        cloned_loader = DataLoader(**kwargs)
        cloned_loader.normalization_stats = getattr(loader, "normalization_stats", None)
        cloned_loader.normalization_applied = bool(getattr(loader, "normalization_applied", False))
        cloned_loader.split_info = getattr(loader, "split_info", None)
        return cloned_loader

    def _estimate_normalization_from_loader(self, loader):
        dataset = getattr(loader, "dataset", None)
        dataset_stats = getattr(dataset, "normalization_stats", None)
        if dataset_stats:
            return dataset_stats

        loader_stats = getattr(loader, "normalization_stats", None)
        if loader_stats:
            return loader_stats

        channel_sum = None
        channel_sum_sq = None
        total_pixels = 0
        for batch_inputs, _ in loader:
            batch = batch_inputs.detach().float().cpu()
            if batch.ndim != 4:
                continue
            batch_channel_sum = batch.sum(dim=(0, 2, 3))
            batch_channel_sum_sq = (batch * batch).sum(dim=(0, 2, 3))
            if channel_sum is None:
                channel_sum = batch_channel_sum
                channel_sum_sq = batch_channel_sum_sq
            else:
                channel_sum += batch_channel_sum
                channel_sum_sq += batch_channel_sum_sq
            total_pixels += int(batch.shape[0] * batch.shape[2] * batch.shape[3])

        if channel_sum is None or total_pixels == 0:
            return None

        mean_values = (channel_sum / total_pixels).numpy().astype(np.float32)
        variance = np.maximum((channel_sum_sq / total_pixels).numpy().astype(np.float32) - np.square(mean_values), 0.0)
        std_values = np.sqrt(variance).astype(np.float32)
        std_values = np.where(np.abs(std_values) < 1e-6, 1.0, std_values)
        return {
            "mean": mean_values.tolist(),
            "std": std_values.tolist(),
        }

    def _apply_dataset_normalization(self, dataset, normalization_stats):
        if dataset is None or not normalization_stats:
            return
        if hasattr(dataset, "normalization_stats"):
            dataset.normalization_stats = normalization_stats
        if hasattr(dataset, "normalization_mean") and hasattr(dataset, "normalization_std"):
            dataset.normalization_mean = torch.as_tensor(normalization_stats["mean"], dtype=torch.float32).view(-1, 1, 1)
            dataset.normalization_std = torch.as_tensor(normalization_stats["std"], dtype=torch.float32).view(-1, 1, 1)

    def _prepare_loaders(self, train_loader, val_loader, params):
        requested_batch_size = int(params.get("batch_size", getattr(train_loader, "batch_size", 32) or 32))
        train_loader = self._as_loader(train_loader, batch_size=requested_batch_size, shuffle=True)
        val_loader = self._as_loader(val_loader, batch_size=requested_batch_size, shuffle=False) if val_loader is not None else None
        normalization_stats = getattr(train_loader, "normalization_stats", None) or self._estimate_normalization_from_loader(train_loader)

        if normalization_stats:
            self._apply_dataset_normalization(getattr(train_loader, "dataset", None), normalization_stats)
            train_loader.normalization_stats = normalization_stats
            train_loader.normalization_applied = True
            if val_loader is not None:
                self._apply_dataset_normalization(getattr(val_loader, "dataset", None), normalization_stats)
                val_loader.normalization_stats = normalization_stats
                val_loader.normalization_applied = True

        effective_train_loader = self._clone_loader(train_loader, batch_size=requested_batch_size)
        effective_val_loader = self._clone_loader(val_loader, batch_size=requested_batch_size) if val_loader is not None else None
        return effective_train_loader, effective_val_loader

    def _ensure_model(self):
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained yet.")
        self.model.eval()
        self.model.to(self.device)

    def _apply_normalization(self, batch):
        if not self.normalization_stats:
            return batch
        mean_tensor = torch.as_tensor(self.normalization_stats["mean"], dtype=torch.float32, device=batch.device).view(-1, 1, 1)
        std_tensor = torch.as_tensor(self.normalization_stats["std"], dtype=torch.float32, device=batch.device).view(-1, 1, 1)
        return (batch - mean_tensor) / std_tensor

    def _predict_tensor_batches(self, tensor_data, *, apply_normalization=False):
        predictions = []
        probabilities = []
        self._ensure_model()
        with torch.no_grad():
            for start in range(0, tensor_data.shape[0], 128):
                batch = tensor_data[start:start + 128].to(self.device)
                if apply_normalization:
                    batch = self._apply_normalization(batch)
                outputs = self.model(batch)
                probabilities.append(torch.softmax(outputs, dim=1).cpu().numpy())
                predictions.append(torch.argmax(outputs, dim=1).cpu().numpy())
        return np.concatenate(predictions, axis=0), np.concatenate(probabilities, axis=0)

    def _to_tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data.float()
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        raise TypeError("Unsupported tensor input type for deep learning prediction.")

    def _extract_inputs(self, data):
        if isinstance(data, tuple):
            return data[0]
        return data

    def predict(self, data):
        inputs = self._extract_inputs(data)
        if isinstance(inputs, DataLoader):
            preds = []
            self._ensure_model()
            apply_normalization = not bool(getattr(inputs, "normalization_applied", False))
            with torch.no_grad():
                for batch_inputs, _ in inputs:
                    batch_preds, _ = self._predict_tensor_batches(batch_inputs.float(), apply_normalization=apply_normalization)
                    preds.append(batch_preds)
            return np.concatenate(preds, axis=0) if preds else np.array([], dtype=np.int64)

        tensor_data = self._to_tensor(inputs)
        predictions, _ = self._predict_tensor_batches(tensor_data, apply_normalization=True)
        return predictions

    def predict_proba(self, data):
        inputs = self._extract_inputs(data)
        if isinstance(inputs, DataLoader):
            probs = []
            self._ensure_model()
            apply_normalization = not bool(getattr(inputs, "normalization_applied", False))
            with torch.no_grad():
                for batch_inputs, _ in inputs:
                    _, batch_probs = self._predict_tensor_batches(batch_inputs.float(), apply_normalization=apply_normalization)
                    probs.append(batch_probs)
            return np.concatenate(probs, axis=0) if probs else np.empty((0, self.num_classes))

        tensor_data = self._to_tensor(inputs)
        _, probabilities = self._predict_tensor_batches(tensor_data, apply_normalization=True)
        return probabilities

    def get_default_params(self):
        return {
            "epochs": 20,
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "lr_step_size": 10,
            "lr_gamma": 0.1,
            "early_stopping_patience": 15,
        }

    def get_param_space(self):
        if self.fixed_epochs is not None:
            epoch_space = ("categorical", [int(self.fixed_epochs)])
        else:
            epoch_space = ("int", 10, 50)
        return {
            "epochs": epoch_space,
            "learning_rate": ("loguniform", 1e-5, 1e-2),
            "batch_size": ("categorical", [16, 32, 64, 128]),
            "optimizer": ("categorical", ["Adam", "AdamW", "SGD", "RMSprop"]),
            "scheduler": ("categorical", ["none", "CosineAnnealingLR", "StepLR", "ReduceLROnPlateau"]),
            "lr_step_size": ("int", 3, 15),
            "lr_gamma": ("float", 0.1, 0.9),
        }

    def save_model(self, path):
        if self.model is None:
            raise RuntimeError("No trained model is available to save.")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_name": self.model_type,
                "num_classes": self.num_classes,
                "input_channels": self.input_channels,
                "image_size": self.image_size,
                "normalization_stats": self.normalization_stats,
                "split_info": self.split_info,
                "training_config": self.training_config,
            },
            path,
        )

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model = self.model_class(
            num_classes=int(checkpoint.get("num_classes", self.num_classes)) if isinstance(checkpoint, dict) else self.num_classes,
            input_channels=int(checkpoint.get("input_channels", self.input_channels)) if isinstance(checkpoint, dict) else self.input_channels,
            image_size=int(checkpoint.get("image_size", self.image_size)) if isinstance(checkpoint, dict) else self.image_size,
        )
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.normalization_stats = checkpoint.get("normalization_stats") if isinstance(checkpoint, dict) else None
        self.split_info = checkpoint.get("split_info") if isinstance(checkpoint, dict) else None
        self.training_config = checkpoint.get("training_config", {}) if isinstance(checkpoint, dict) else {}
        self.model.normalization_stats = self.normalization_stats
        self.model.to(self.device)
        self.model.eval()
        self.is_trained = True

    def stop(self):
        self._stop_requested = True
        if self.current_trainer is not None and hasattr(self.current_trainer, "stop"):
            self.current_trainer.stop()


class PUCNNTWrapper(DeepLearningWrapper):
    """PU-CNN-Transformer wrapper using scalar score output and PU loss."""

    data_mode = "loader"

    class _PULabelViewDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.normalization_stats = getattr(dataset, "normalization_stats", None)
            self.normalization_mean = getattr(dataset, "normalization_mean", None)
            self.normalization_std = getattr(dataset, "normalization_std", None)

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            sample, label = self.dataset[index]
            if hasattr(label, "item"):
                label = label.item()
            return sample, 1 if float(label) > 0 else -1

    def __init__(
        self,
        input_channels,
        image_size,
        *,
        model_name: Optional[str] = None,
        fixed_epochs: Optional[int] = None,
        persist_artifacts: bool = False,
        artifact_dir: Optional[str] = None,
        training_config: Optional[dict] = None,
        automl_config: Optional[dict] = None,
    ):
        from ..model.cnn_transformer import CNNTransformer

        super().__init__(
            CNNTransformer,
            2,
            input_channels,
            image_size,
            model_name=model_name or "PU-CNN-Transformer",
            fixed_epochs=fixed_epochs,
            persist_artifacts=persist_artifacts,
            artifact_dir=artifact_dir,
        )
        self.training_defaults = dict(training_config or {})
        self.automl_overrides = dict(automl_config or {})
        apply_model_metadata(self, model_name or "PU-CNN-Transformer")
        self.training_mode = "pu"
        self.label_mode = "pu"
        self.is_pu_model = True
        self.model_family = "neural"
        self.input_kind = "window"

    @staticmethod
    def _normalize_image_shape(image_size):
        if isinstance(image_size, (tuple, list)):
            if len(image_size) >= 2:
                return int(image_size[0]), int(image_size[1])
            if len(image_size) == 1:
                side = int(image_size[0])
                return side, side
        side = int(image_size or 1)
        return side, side

    def _as_loader(self, data, *, batch_size, shuffle=False):
        if data is None:
            return None
        if isinstance(data, DataLoader):
            return self._clone_loader(data, batch_size=batch_size)
        if isinstance(data, tuple):
            if len(data) != 2:
                raise TypeError("Tuple inputs must be (X, y).")
            features, labels = data
            features = torch.as_tensor(np.asarray(features), dtype=torch.float32)
            labels = torch.as_tensor(np.where(np.asarray(labels).reshape(-1) > 0, 1, -1).astype(np.int64), dtype=torch.long)
            dataset = TensorDataset(features, labels)
            loader = DataLoader(
                dataset,
                batch_size=int(batch_size),
                shuffle=bool(shuffle),
                num_workers=0,
                drop_last=_should_drop_last_for_training(len(dataset), batch_size) if shuffle else False,
            )
            loader.normalization_stats = None
            loader.normalization_applied = False
            loader.split_info = None
            return loader
        raise TypeError("PUCNNTWrapper expects DataLoader or (X, y) tuple inputs.")

    def _clone_loader(self, loader, *, batch_size=None):
        target_batch_size = int(batch_size or loader.batch_size or 32)
        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            return super()._clone_loader(loader, batch_size=target_batch_size)

        kwargs = {
            "dataset": self._PULabelViewDataset(dataset),
            "batch_size": target_batch_size,
            "num_workers": 0,
            "pin_memory": bool(getattr(loader, "pin_memory", False)),
            "drop_last": bool(getattr(loader, "drop_last", False)),
        }

        sampler_name = loader.sampler.__class__.__name__ if getattr(loader, "sampler", None) is not None else ""
        if sampler_name == "RandomSampler":
            kwargs["shuffle"] = True
        elif sampler_name == "SequentialSampler":
            kwargs["shuffle"] = False
        elif getattr(loader, "sampler", None) is not None:
            kwargs["sampler"] = loader.sampler
        else:
            kwargs["shuffle"] = False

        if bool(kwargs.get("shuffle")):
            kwargs["drop_last"] = _should_drop_last_for_training(len(dataset), target_batch_size)

        cloned_loader = DataLoader(**kwargs)
        cloned_loader.normalization_stats = getattr(loader, "normalization_stats", None)
        cloned_loader.normalization_applied = bool(getattr(loader, "normalization_applied", False))
        cloned_loader.split_info = getattr(loader, "split_info", None)
        return cloned_loader

    def _resolve_prior_value(self, train_loader, params):
        prior_mode = str(
            params.get("prior_mode")
            or self.training_defaults.get("prior_mode")
            or "auto"
        ).strip().lower()
        prior_value = params.get("prior", self.training_defaults.get("prior"))
        if prior_mode == "manual" and prior_value not in (None, ""):
            try:
                return float(prior_value)
            except (TypeError, ValueError):
                pass

        labels = []
        dataset = getattr(train_loader, "dataset", None)
        dataset_labels = getattr(dataset, "labels", None)
        if dataset_labels is not None:
            labels = np.asarray(dataset_labels).reshape(-1)
        elif hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "__len__"):
            try:
                labels = np.asarray([train_loader.dataset[index][1] for index in range(len(train_loader.dataset))]).reshape(-1)
            except Exception:
                labels = []
        if len(labels) == 0:
            fallback = prior_value if prior_value not in (None, "") else 0.5
            return float(fallback)
        prior = float(np.mean(np.asarray(labels).reshape(-1) > 0))
        return float(np.clip(prior, 1e-3, 0.999))

    def _build_optimizer(self, params):
        name = str(params.get("optimizer", self.training_defaults.get("optimizer", "AdamW")) or "AdamW").lower()
        lr = float(params.get("learning_rate", self.training_defaults.get("learning_rate", 0.001)))
        if name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        if name == "rmsprop":
            return torch.optim.RMSprop(self.model.parameters(), lr=lr)
        if name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        return torch.optim.AdamW(self.model.parameters(), lr=lr)

    def _build_scheduler(self, optimizer, params):
        scheduler_name = str(params.get("scheduler", self.training_defaults.get("scheduler", "none")) or "none").lower()
        if scheduler_name in {"", "none", "off", "disabled"}:
            return None
        lr_step_size = int(params.get("lr_step_size", self.training_defaults.get("lr_step_size", 10)))
        lr_gamma = float(params.get("lr_gamma", self.training_defaults.get("lr_gamma", 0.1)))
        if scheduler_name == "steplr":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(lr_step_size, 1),
                gamma=lr_gamma,
            )
        if scheduler_name == "reducelronplateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=lr_gamma,
                patience=3,
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(params.get("epochs", self.fixed_epochs or 20)), 1))

    def _build_loss(self, prior, params, *, adaptive=False):
        beta = float(params.get("beta", self.training_defaults.get("beta", 0.0)))
        gamma = float(params.get("gamma", self.training_defaults.get("gamma", 1.0)))
        adaptive_window = int(params.get("adaptive_window", self.training_defaults.get("adaptive_window", 10)))
        adaptive_lambda = float(params.get("adaptive_lambda", self.training_defaults.get("adaptive_lambda", 1.0)))
        gamma_min = params.get("adaptive_gamma_min", self.training_defaults.get("adaptive_gamma_min"))
        gamma_max = params.get("adaptive_gamma_max", self.training_defaults.get("adaptive_gamma_max"))
        if adaptive:
            return AdaptivePULoss(
                prior=prior,
                gamma=gamma,
                beta=beta,
                adaptive_window=max(adaptive_window, 1),
                nnpu=True,
                adaptive_lambda=adaptive_lambda,
                gamma_min=gamma_min,
                gamma_max=gamma_max,
            )
        return PULoss(
            prior=prior,
            gamma=gamma,
            beta=beta,
            nnpu=True,
        )

    @staticmethod
    def _binary_targets(labels, device):
        labels = labels.to(device=device, dtype=torch.long).view(-1)
        return torch.where(labels > 0, torch.ones_like(labels), -torch.ones_like(labels))

    def _run_epoch(self, loader, *, optimizer=None, loss_func=None):
        training = optimizer is not None
        self.model.train(training)
        loss_total = 0.0
        accuracy_total = 0.0
        batch_count = 0
        positive_count = 0
        true_positive_count = 0
        loss_func = loss_func or self._build_loss(0.5, {}, adaptive=False)

        for batch in loader:
            if self._stop_requested:
                break
            inputs, labels = batch
            inputs = inputs.to(self.device, dtype=torch.float32)
            # BatchNorm layers in PU-CNN-Transformer require at least 2 samples while training.
            # The spatial split + balancing pipeline can occasionally produce a tail batch of size 1.
            if training and int(inputs.shape[0]) < 2:
                continue
            targets = self._binary_targets(labels, self.device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            scores = self.model(inputs)
            if scores.ndim > 1:
                scores = scores.view(scores.shape[0], -1)
                if scores.shape[1] != 1:
                    scores = scores.mean(dim=1, keepdim=True)
            scores = scores.view(-1)
            loss = loss_func(scores, targets)

            if training:
                loss.backward()
                optimizer.step()

            probabilities = torch.sigmoid(scores.detach())
            predictions = (probabilities >= 0.5).long()
            binary_targets = (targets > 0).long()
            batch_size = int(binary_targets.shape[0])
            loss_total += float(loss.item()) * batch_size
            accuracy_total += float((predictions == binary_targets).float().mean().item()) * batch_size
            positive_count += int(binary_targets.sum().item())
            true_positive_count += int(((predictions == 1) & (binary_targets == 1)).sum().item())
            batch_count += batch_size

        if batch_count == 0:
            return 0.0, 0.0, 0.0
        recall = float(true_positive_count / positive_count) if positive_count > 0 else 0.0
        return loss_total / batch_count, accuracy_total / batch_count, recall

    def _predict_scores(self, tensor_data, *, apply_normalization=False):
        scores = []
        self._ensure_model()
        with torch.no_grad():
            for start in range(0, tensor_data.shape[0], 128):
                batch = tensor_data[start : start + 128].to(self.device)
                if apply_normalization:
                    batch = self._apply_normalization(batch)
                outputs = self.model(batch)
                if outputs.ndim > 1:
                    outputs = outputs.view(outputs.shape[0], -1)
                    if outputs.shape[1] != 1:
                        outputs = outputs.mean(dim=1, keepdim=True)
                scores.append(outputs.view(-1).cpu().numpy())
        if not scores:
            return np.array([], dtype=np.float32)
        return np.concatenate(scores, axis=0)

    def train(self, train_data, val_data=None, **params):
        train_loader, val_loader = self._prepare_loaders(train_data, val_data, params)
        requested_save_path = params.get("save_path")
        persist_artifacts = bool(params.get("persist_artifacts", self.persist_artifacts))
        artifact_dir = params.get("artifact_dir") or self.artifact_dir
        temp_dir_path = None

        def _resolve_checkpoint_path(path_like, default_filename: str = "best_model.pth") -> str:
            candidate = Path(str(path_like))
            if candidate.suffix:
                candidate.parent.mkdir(parents=True, exist_ok=True)
                return str(candidate)
            candidate.mkdir(parents=True, exist_ok=True)
            return str(candidate / default_filename)

        if requested_save_path:
            save_path = _resolve_checkpoint_path(requested_save_path, default_filename="best_model.pth")
        elif persist_artifacts:
            save_path = _resolve_checkpoint_path(
                artifact_dir or (Path.cwd() / "models" / "comparison"),
                default_filename=f"{str(self.model_type).lower()}_best.pth",
            )
        else:
            temp_dir_path = tempfile.mkdtemp(prefix=f"modelcmp_{self.model_type.lower()}_")
            save_path = _resolve_checkpoint_path(temp_dir_path, default_filename="best_model.pth")

        input_channels = int(params.get("input_channels", self.input_channels or 1))
        image_size = params.get("image_size", self.image_size or 1)
        height, width = self._normalize_image_shape(image_size)
        if min(height, width) < 3:
            raise ValueError(f"{self.model_name} requires a window-based H5 with patch size at least 3x3.")

        prior = self._resolve_prior_value(train_loader, params)
        self.model = self.model_class(
            prior=prior,
            input_dim=int(input_channels * height * width),
            input_shape=(input_channels, height, width),
        )
        self.model.to(self.device)

        loss_type = str(params.get("loss_type", self.training_defaults.get("loss_type", "standard")) or "standard").strip().lower()
        adaptive_loss = loss_type in {"adaptive", "adaptive_pu", "adaptive-pu"}
        train_loss_func = self._build_loss(prior, params, adaptive=adaptive_loss)
        val_loss_func = self._build_loss(prior, params, adaptive=False)
        optimizer = self._build_optimizer(params)
        scheduler = self._build_scheduler(optimizer, params)
        epochs = int(params.get("epochs", self.fixed_epochs if self.fixed_epochs is not None else 20))
        patience = int(params.get("early_stopping_patience", self.training_defaults.get("early_stopping_patience", 15)))
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_recall": [],
            "val_recall": [],
        }
        self.best_state_dict = None
        self.best_metric = None
        patience_counter = 0

        try:
            for epoch in range(max(epochs, 1)):
                if self._stop_requested:
                    break

                train_loss, train_acc, train_recall = self._run_epoch(train_loader, optimizer=optimizer, loss_func=train_loss_func)
                if val_loader is not None:
                    val_loss, val_acc, val_recall = self._run_epoch(val_loader, optimizer=None, loss_func=val_loss_func)
                else:
                    val_loss, val_acc, val_recall = train_loss, train_acc, train_recall

                self.history["train_loss"].append(float(train_loss))
                self.history["train_acc"].append(float(train_acc))
                self.history["val_loss"].append(float(val_loss))
                self.history["val_acc"].append(float(val_acc))
                self.history["train_recall"].append(float(train_recall))
                self.history["val_recall"].append(float(val_recall))

                monitor_recall = float(val_recall if val_loader is not None else train_recall)
                monitor_loss = float(val_loss if val_loader is not None else train_loss)
                monitor_value = (monitor_recall, -monitor_loss)
                is_better = False
                if self.best_metric is None:
                    is_better = True
                else:
                    best_recall, best_neg_loss = self.best_metric
                    if monitor_recall > best_recall + 1e-6:
                        is_better = True
                    elif abs(monitor_recall - best_recall) <= 1e-6 and (-monitor_loss) > best_neg_loss + 1e-6:
                        is_better = True

                if is_better:
                    self.best_metric = monitor_value
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                    if save_path is not None:
                        torch.save(
                            {
                                "model_state_dict": self.best_state_dict,
                                "history": self.history,
                                "best_metric": None if self.best_metric is None else float(self.best_metric[0]),
                                "best_metric_recall": None if self.best_metric is None else float(self.best_metric[0]),
                                "best_metric_val_loss": float(monitor_loss),
                                "best_epoch": epoch,
                                "model_name": self.model_type,
                                "input_channels": self.input_channels,
                                "image_size": self.image_size,
                                "prior": prior,
                                "training_config": {
                                    **self.training_defaults,
                                    **params,
                                    "loss_type": loss_type,
                                    "prior": prior,
                                    "prior_mode": str(params.get("prior_mode", self.training_defaults.get("prior_mode", "auto"))),
                                    "beta": float(params.get("beta", self.training_defaults.get("beta", 0.0))),
                                    "gamma": float(params.get("gamma", self.training_defaults.get("gamma", 1.0))),
                                    "adaptive_window": int(params.get("adaptive_window", self.training_defaults.get("adaptive_window", 10))),
                                    "adaptive_lambda": float(params.get("adaptive_lambda", self.training_defaults.get("adaptive_lambda", 1.0))),
                                    "adaptive_gamma_min": params.get("adaptive_gamma_min", self.training_defaults.get("adaptive_gamma_min")),
                                    "adaptive_gamma_max": params.get("adaptive_gamma_max", self.training_defaults.get("adaptive_gamma_max")),
                                    "label_mode": "pu",
                                    "input_kind": "window",
                                },
                            },
                            save_path,
                        )
                else:
                    patience_counter += 1

                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(monitor_recall)
                    else:
                        scheduler.step()

                if patience > 0 and patience_counter >= patience:
                    break

            if self.best_state_dict is not None:
                self.model.load_state_dict(self.best_state_dict)

            self.is_trained = True
            self.normalization_stats = getattr(train_loader, "normalization_stats", None)
            self.split_info = getattr(train_loader, "split_info", None)
            self.training_config = {
                **self.training_defaults,
                **params,
                "loss_type": loss_type,
                "prior_mode": str(params.get("prior_mode", self.training_defaults.get("prior_mode", "auto"))),
                "prior": prior,
                "beta": float(params.get("beta", self.training_defaults.get("beta", 0.0))),
                "gamma": float(params.get("gamma", self.training_defaults.get("gamma", 1.0))),
                "adaptive_window": int(params.get("adaptive_window", self.training_defaults.get("adaptive_window", 10))),
                "adaptive_lambda": float(params.get("adaptive_lambda", self.training_defaults.get("adaptive_lambda", 1.0))),
                "adaptive_gamma_min": params.get("adaptive_gamma_min", self.training_defaults.get("adaptive_gamma_min")),
                "adaptive_gamma_max": params.get("adaptive_gamma_max", self.training_defaults.get("adaptive_gamma_max")),
                "learning_rate": float(params.get("learning_rate", self.training_defaults.get("learning_rate", 0.001))),
                "batch_size": int(params.get("batch_size", getattr(train_loader, "batch_size", 32) or 32)),
                "optimizer": params.get("optimizer", self.training_defaults.get("optimizer", "AdamW")),
                "scheduler": params.get("scheduler", self.training_defaults.get("scheduler", "none")),
                "lr_step_size": int(params.get("lr_step_size", self.training_defaults.get("lr_step_size", 10))),
                "lr_gamma": float(params.get("lr_gamma", self.training_defaults.get("lr_gamma", 0.1))),
                "early_stopping_patience": patience,
                "label_mode": "pu",
                "input_kind": "window",
                "prior_estimated": float(prior),
            }
            self.model.normalization_stats = self.normalization_stats
            train_acc = max(self.history.get("train_acc", [0.0])) if self.history.get("train_acc") else 0.0
            val_acc = max(self.history.get("val_acc", [0.0])) if self.history.get("val_acc") else 0.0
            train_recall = max(self.history.get("train_recall", [0.0])) if self.history.get("train_recall") else 0.0
            val_recall = max(self.history.get("val_recall", [0.0])) if self.history.get("val_recall") else 0.0
            self.history["train_acc"] = train_acc
            self.history["val_acc"] = val_acc
            self.history["train_recall"] = train_recall
            self.history["val_recall"] = val_recall
            return self.history
        finally:
            self.current_trainer = None
            if temp_dir_path is not None:
                shutil.rmtree(temp_dir_path, ignore_errors=True)

    def predict(self, data):
        inputs = self._extract_inputs(data)
        if isinstance(inputs, DataLoader):
            preds = []
            self._ensure_model()
            apply_normalization = not bool(getattr(inputs, "normalization_applied", False))
            with torch.no_grad():
                for batch_inputs, _ in inputs:
                    batch_scores = self._predict_scores(batch_inputs.float(), apply_normalization=apply_normalization)
                    batch_probs = 1.0 / (1.0 + np.exp(-np.clip(batch_scores, -50, 50)))
                    preds.append((batch_probs >= 0.5).astype(np.int64))
            return np.concatenate(preds, axis=0) if preds else np.array([], dtype=np.int64)

        tensor_data = self._to_tensor(inputs)
        scores = self._predict_scores(tensor_data, apply_normalization=True)
        probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -50, 50)))
        return (probs >= 0.5).astype(np.int64)

    def predict_proba(self, data):
        inputs = self._extract_inputs(data)
        if isinstance(inputs, DataLoader):
            probs = []
            self._ensure_model()
            apply_normalization = not bool(getattr(inputs, "normalization_applied", False))
            with torch.no_grad():
                for batch_inputs, _ in inputs:
                    batch_scores = self._predict_scores(batch_inputs.float(), apply_normalization=apply_normalization)
                    batch_probs = 1.0 / (1.0 + np.exp(-np.clip(batch_scores, -50, 50)))
                    probs.append(np.column_stack([1.0 - batch_probs, batch_probs]))
            return np.concatenate(probs, axis=0) if probs else np.empty((0, 2), dtype=np.float32)

        tensor_data = self._to_tensor(inputs)
        scores = self._predict_scores(tensor_data, apply_normalization=True)
        batch_probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -50, 50)))
        return np.column_stack([1.0 - batch_probs, batch_probs])

    def get_default_params(self):
        defaults = dict(super().get_default_params())
        defaults.update(
            {
                "loss_type": self.training_defaults.get("loss_type", "standard"),
                "prior_mode": self.training_defaults.get("prior_mode", "auto"),
                "prior": self.training_defaults.get("prior", 0.5),
                "beta": self.training_defaults.get("beta", 0.0),
                "gamma": self.training_defaults.get("gamma", 1.0),
                "adaptive_window": self.training_defaults.get("adaptive_window", 10),
                "adaptive_lambda": self.training_defaults.get("adaptive_lambda", 1.0),
                "adaptive_gamma_min": self.training_defaults.get("adaptive_gamma_min"),
                "adaptive_gamma_max": self.training_defaults.get("adaptive_gamma_max"),
            }
        )
        return defaults

    def get_param_space(self):
        base_space = dict(super().get_param_space())
        if self.fixed_epochs is not None:
            base_space["epochs"] = ("categorical", [int(self.fixed_epochs)])
        base_space.update(
            {
                "learning_rate": ("loguniform", 1e-5, 1e-4),
                "batch_size": ("categorical", [16, 32, 64]),
                "optimizer": ("categorical", ["Adam", "AdamW", "SGD", "RMSprop"]),
                "scheduler": ("categorical", ["none", "CosineAnnealingLR", "StepLR", "ReduceLROnPlateau"]),
                "lr_step_size": ("int", 3, 15),
                "lr_gamma": ("float", 0.1, 0.9),
                "loss_type": ("categorical", ["standard", "adaptive"]),
                "prior_mode": ("categorical", ["auto", "manual"]),
                "prior": ("float", 0.05, 0.5),
                "beta": ("float", 0.0, 0.5),
                "gamma": ("float", 0.5, 5.0),
                "adaptive_window": ("int", 3, 30),
                "adaptive_lambda": ("float", 0.5, 3.0),
                "adaptive_gamma_max": ("float", 5.0, 20.0),
            }
        )
        if self.automl_overrides:
            for key, value in self.automl_overrides.items():
                if not value:
                    continue
                if isinstance(value, tuple) and len(value) >= 2:
                    base_space[key] = tuple(value)
                elif isinstance(value, list):
                    base_space[key] = ("categorical", list(value))
        return base_space

    def save_model(self, path):
        if self.model is None:
            raise RuntimeError("No trained model is available to save.")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_name": self.model_type,
                "num_classes": self.num_classes,
                "input_channels": self.input_channels,
                "image_size": self.image_size,
                "normalization_stats": self.normalization_stats,
                "split_info": self.split_info,
                "training_config": self.training_config,
                "prior": self.training_config.get("prior", self.training_defaults.get("prior", 0.5)),
            },
            path,
        )

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        training_config = checkpoint.get("training_config", {}) if isinstance(checkpoint, dict) else {}
        prior = float(checkpoint.get("prior", training_config.get("prior", self.training_defaults.get("prior", 0.5)))) if isinstance(checkpoint, dict) else float(self.training_defaults.get("prior", 0.5))
        self.model = self.model_class(
            prior=prior,
            input_dim=int(self.input_channels * self._normalize_image_shape(self.image_size)[0] * self._normalize_image_shape(self.image_size)[1]),
            input_shape=(int(self.input_channels), *self._normalize_image_shape(self.image_size)),
        )
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.normalization_stats = checkpoint.get("normalization_stats") if isinstance(checkpoint, dict) else None
        self.split_info = checkpoint.get("split_info") if isinstance(checkpoint, dict) else None
        self.training_config = training_config if isinstance(checkpoint, dict) else {}
        self.model.normalization_stats = self.normalization_stats
        self.model.to(self.device)
        self.model.eval()
        self.is_trained = True


class PUCNNWrapper(PUCNNTWrapper):
    """PU-CNN wrapper using scalar score output and PU loss."""

    def __init__(
        self,
        input_channels,
        image_size,
        *,
        model_name: Optional[str] = None,
        fixed_epochs: Optional[int] = None,
        persist_artifacts: bool = False,
        artifact_dir: Optional[str] = None,
        training_config: Optional[dict] = None,
        automl_config: Optional[dict] = None,
    ):
        from ..model.cnn import CNN

        DeepLearningWrapper.__init__(
            self,
            CNN,
            2,
            input_channels,
            image_size,
            model_name=model_name or "PU-CNN",
            fixed_epochs=fixed_epochs,
            persist_artifacts=persist_artifacts,
            artifact_dir=artifact_dir,
        )
        self.training_defaults = dict(training_config or {})
        self.automl_overrides = dict(automl_config or {})
        apply_model_metadata(self, model_name or "PU-CNN")
        self.training_mode = "pu"
        self.label_mode = "pu"
        self.is_pu_model = True
        self.model_family = "neural"
        self.input_kind = "window"


class PUCNNTransformerWrapper(PUCNNTWrapper):
    """PU-CNN-TokenTransformer wrapper with tokenized CNN features and PU loss."""

    def __init__(
        self,
        input_channels,
        image_size,
        *,
        model_name: Optional[str] = None,
        fixed_epochs: Optional[int] = None,
        persist_artifacts: bool = False,
        artifact_dir: Optional[str] = None,
        training_config: Optional[dict] = None,
        automl_config: Optional[dict] = None,
    ):
        from ..model.cnn_token_transformer import CNNTokenTransformer

        DeepLearningWrapper.__init__(
            self,
            CNNTokenTransformer,
            2,
            input_channels,
            image_size,
            model_name=model_name or "PU-CNN-TokenTransformer",
            fixed_epochs=fixed_epochs,
            persist_artifacts=persist_artifacts,
            artifact_dir=artifact_dir,
        )
        self.training_defaults = dict(training_config or {})
        self.automl_overrides = dict(automl_config or {})
        apply_model_metadata(self, model_name or "PU-CNN-TokenTransformer")
        self.training_mode = "pu"
        self.label_mode = "pu"
        self.is_pu_model = True
        self.model_family = "neural"
        self.input_kind = "window"


class _ArrayWrapper(BaseModelWrapper):
    data_mode = "array"

    def _prepare_features(self, features):
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        elif isinstance(features, DataLoader):
            collected = []
            for batch in features:
                batch_inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
                if isinstance(batch_inputs, torch.Tensor):
                    batch_inputs = batch_inputs.detach().cpu().numpy()
                else:
                    batch_inputs = np.asarray(batch_inputs)
                if batch_inputs.ndim == 4:
                    batch_inputs = batch_inputs.reshape(batch_inputs.shape[0], -1)
                collected.append(np.asarray(batch_inputs))
            if not collected:
                return np.empty((0, 0), dtype=np.float32)
            features = np.concatenate(collected, axis=0)
        features = np.asarray(features)
        # Classic ML models expect tabular inputs, so flatten image patches to vectors.
        if features.ndim > 2:
            return features.reshape(features.shape[0], -1)
        if features.ndim == 1:
            return features.reshape(1, -1)
        if features.ndim == 0:
            return features.reshape(1, 1)
        return features

    def _unpack_array_data(self, data):
        if not isinstance(data, tuple) or len(data) != 2:
            raise TypeError("Classic ML wrappers expect data as (X, y).")
        X, y = data
        return self._prepare_features(X), np.asarray(y).reshape(-1)

    def _extract_inputs(self, data):
        if isinstance(data, tuple):
            data = data[0]
        return self._prepare_features(data)

    def predict(self, data):
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        return self.model.predict(self._extract_inputs(data))

    def predict_proba(self, data):
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        return self.model.predict_proba(self._extract_inputs(data))


class RandomForestWrapper(_ArrayWrapper):
    def __init__(self):
        super().__init__("RandomForest")
        from sklearn.ensemble import RandomForestClassifier

        self.RandomForestClassifier = RandomForestClassifier

    def train(self, train_data, val_data=None, **params):
        X_train, y_train = self._unpack_array_data(train_data)
        if val_data is None:
            X_val, y_val = X_train, y_train
        else:
            X_val, y_val = self._unpack_array_data(val_data)
        self.model = self.RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=params.get("max_depth"),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.num_classes = int(len(np.unique(np.concatenate([y_train, y_val])))) if val_data is not None else int(len(np.unique(y_train)))
        return {
            "train_acc": float(self.model.score(X_train, y_train)),
            "val_acc": float(self.model.score(X_val, y_val)) if val_data is not None else None,
        }

    def get_default_params(self):
        return {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        }

    def get_param_space(self):
        return {
            "n_estimators": ("int", 50, 300),
            "max_depth": ("int", 5, 30),
            "min_samples_split": ("int", 2, 10),
            "min_samples_leaf": ("int", 1, 5),
        }

    def save_model(self, path):
        import joblib

        joblib.dump(
            {
                "model": self.model,
                "model_type": self.model_type,
                "num_classes": getattr(self, "num_classes", None),
            },
            path,
        )

    def load_model(self, path):
        import joblib

        loaded = joblib.load(path)
        if isinstance(loaded, dict) and "model" in loaded:
            self.model = loaded["model"]
            self.num_classes = loaded.get("num_classes")
        else:
            self.model = loaded
        self.is_trained = True


class PURandomForestWrapper(_ArrayWrapper):
    """Bagging-style PU Random Forest using repeated unlabeled resampling."""

    def __init__(self):
        super().__init__("PU-Random Forest")
        from sklearn.ensemble import RandomForestClassifier

        self.RandomForestClassifier = RandomForestClassifier
        self.models = []
        self.ensemble_params = {}

    @staticmethod
    def _binary_labels(labels):
        return np.where(np.asarray(labels).reshape(-1) > 0, 1, 0).astype(np.int64)

    def _build_submodel(self, params, random_state):
        max_depth = params.get("max_depth")
        if max_depth in ("", None):
            max_depth = None
        else:
            max_depth = int(max_depth)
        return self.RandomForestClassifier(
            n_estimators=_safe_int(params.get("n_estimators", 200), 200),
            max_depth=max_depth,
            min_samples_split=_safe_int(params.get("min_samples_split", 2), 2),
            min_samples_leaf=_safe_int(params.get("min_samples_leaf", 1), 1),
            max_features=params.get("max_features", "sqrt"),
            class_weight=params.get("class_weight", "balanced_subsample"),
            bootstrap=bool(params.get("bootstrap", True)),
            random_state=int(random_state),
            n_jobs=-1,
        )

    def _positive_proba(self, estimator, features):
        proba = np.asarray(estimator.predict_proba(features), dtype=np.float64)
        classes = np.asarray(getattr(estimator, "classes_", []))
        if proba.ndim == 1:
            positive_scores = proba.reshape(-1)
        elif classes.size == 0:
            positive_scores = proba[:, -1]
        elif 1 in classes:
            positive_scores = proba[:, int(np.where(classes == 1)[0][0])]
        else:
            positive_scores = np.zeros(features.shape[0], dtype=np.float64)
        positive_scores = np.clip(positive_scores, 0.0, 1.0)
        return np.column_stack([1.0 - positive_scores, positive_scores])

    def train(self, train_data, val_data=None, **params):
        X_train, y_train = self._unpack_array_data(train_data)
        y_train_binary = self._binary_labels(y_train)
        if val_data is None:
            X_val, y_val_binary = X_train, y_train_binary
        else:
            X_val, y_val = self._unpack_array_data(val_data)
            y_val_binary = self._binary_labels(y_val)

        positive_indices = np.where(y_train_binary > 0)[0]
        unlabeled_indices = np.where(y_train_binary == 0)[0]
        if len(positive_indices) == 0:
            raise ValueError("PU-Random Forest requires at least one positive sample.")
        if len(unlabeled_indices) == 0:
            raise ValueError("PU-Random Forest requires unlabeled samples in the training set.")

        n_submodels = max(1, _safe_int(params.get("n_submodels", 15), 15))
        unlabeled_ratio = max(0.25, _safe_float(params.get("unlabeled_ratio", 1.0), 1.0))
        random_state = _safe_int(params.get("random_state", 42), 42)
        sample_size = max(1, int(round(len(positive_indices) * unlabeled_ratio)))
        rng = np.random.default_rng(int(random_state))

        self.models = []
        for model_index in range(n_submodels):
            sampled_unlabeled = rng.choice(
                unlabeled_indices,
                size=sample_size,
                replace=sample_size > len(unlabeled_indices),
            )
            subset_indices = np.concatenate([positive_indices, sampled_unlabeled])
            subset_labels = np.concatenate(
                [
                    np.ones(len(positive_indices), dtype=np.int64),
                    np.zeros(len(sampled_unlabeled), dtype=np.int64),
                ]
            )
            order = rng.permutation(len(subset_indices))
            estimator = self._build_submodel(params, random_state + model_index)
            estimator.fit(X_train[subset_indices][order], subset_labels[order])
            self.models.append(estimator)

        self.model = list(self.models)
        self.ensemble_params = {
            "n_submodels": int(n_submodels),
            "unlabeled_ratio": float(unlabeled_ratio),
            "random_state": int(random_state),
        }
        self.is_trained = True
        self.num_classes = 2

        train_pred = self.predict((X_train, y_train_binary))
        val_pred = self.predict((X_val, y_val_binary))
        return {
            "train_acc": float(np.mean(train_pred == y_train_binary)),
            "val_acc": float(np.mean(val_pred == y_val_binary)) if val_data is not None else None,
        }

    def get_default_params(self):
        return {
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "n_submodels": 15,
            "unlabeled_ratio": 1.0,
            "bootstrap": True,
            "class_weight": "balanced_subsample",
            "random_state": 42,
        }

    def get_param_space(self):
        return {
            "n_estimators": ("int", 80, 320),
            "max_depth": ("int", 4, 24),
            "min_samples_split": ("int", 2, 10),
            "min_samples_leaf": ("int", 1, 5),
            "max_features": ("categorical", ["sqrt", "log2", None]),
            "n_submodels": ("int", 5, 30),
            "unlabeled_ratio": ("float", 0.5, 2.5),
            "bootstrap": ("categorical", [True, False]),
            "class_weight": ("categorical", ["balanced", "balanced_subsample", None]),
        }

    def predict(self, data):
        probabilities = self.predict_proba(data)
        return (probabilities[:, 1] >= 0.5).astype(np.int64)

    def predict_proba(self, data):
        if not self.is_trained or not self.models:
            raise RuntimeError("Model is not trained yet.")
        features = self._extract_inputs(data)
        proba_stack = np.stack(
            [self._positive_proba(estimator, features) for estimator in self.models],
            axis=0,
        )
        return np.mean(proba_stack, axis=0)

    def save_model(self, path):
        import joblib

        joblib.dump(
            {
                "models": self.models,
                "model_type": self.model_type,
                "num_classes": getattr(self, "num_classes", None),
                "ensemble_params": dict(self.ensemble_params or {}),
            },
            path,
        )

    def load_model(self, path):
        import joblib

        loaded = joblib.load(path)
        if isinstance(loaded, dict):
            models = loaded.get("models")
            if models is None and "model" in loaded:
                models = loaded.get("model")
            self.models = list(models or [])
            self.model = list(self.models)
            self.num_classes = loaded.get("num_classes")
            self.ensemble_params = dict(loaded.get("ensemble_params") or {})
        else:
            self.models = list(loaded or [])
            self.model = list(self.models)
        self.is_trained = bool(self.models)


class SVMWrapper(_ArrayWrapper):
    def __init__(self):
        super().__init__("SVM")
        from sklearn.svm import SVC

        self.SVC = SVC

    def train(self, train_data, val_data=None, **params):
        X_train, y_train = self._unpack_array_data(train_data)
        if val_data is None:
            X_val, y_val = X_train, y_train
        else:
            X_val, y_val = self._unpack_array_data(val_data)
        self.model = self.SVC(
            C=float(params.get("C", 1.0)),
            kernel=params.get("kernel", "rbf"),
            gamma=params.get("gamma", "scale"),
            probability=True,
            random_state=42,
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.num_classes = int(len(np.unique(np.concatenate([y_train, y_val])))) if val_data is not None else int(len(np.unique(y_train)))
        return {
            "train_acc": float(self.model.score(X_train, y_train)),
            "val_acc": float(self.model.score(X_val, y_val)) if val_data is not None else None,
        }

    def get_default_params(self):
        return {"C": 1.0, "kernel": "rbf", "gamma": "scale"}

    def get_param_space(self):
        return {
            "C": ("loguniform", 0.01, 100),
            "kernel": ("categorical", ["rbf", "linear", "poly"]),
            "gamma": ("categorical", ["scale", "auto"]),
        }

    def save_model(self, path):
        import joblib

        joblib.dump(
            {
                "model": self.model,
                "model_type": self.model_type,
                "num_classes": getattr(self, "num_classes", None),
            },
            path,
        )

    def load_model(self, path):
        import joblib

        loaded = joblib.load(path)
        if isinstance(loaded, dict) and "model" in loaded:
            self.model = loaded["model"]
            self.num_classes = loaded.get("num_classes")
        else:
            self.model = loaded
        self.is_trained = True


class DecisionTreeWrapper(_ArrayWrapper):
    def __init__(self):
        super().__init__("DecisionTree")
        from sklearn.tree import DecisionTreeClassifier

        self.DecisionTreeClassifier = DecisionTreeClassifier

    def train(self, train_data, val_data=None, **params):
        X_train, y_train = self._unpack_array_data(train_data)
        if val_data is None:
            X_val, y_val = X_train, y_train
        else:
            X_val, y_val = self._unpack_array_data(val_data)
        self.model = self.DecisionTreeClassifier(
            max_depth=params.get("max_depth"),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=42,
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.num_classes = int(len(np.unique(np.concatenate([y_train, y_val])))) if val_data is not None else int(len(np.unique(y_train)))
        return {
            "train_acc": float(self.model.score(X_train, y_train)),
            "val_acc": float(self.model.score(X_val, y_val)) if val_data is not None else None,
        }

    def get_default_params(self):
        return {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1}

    def get_param_space(self):
        return {
            "max_depth": ("int", 3, 20),
            "min_samples_split": ("int", 2, 10),
            "min_samples_leaf": ("int", 1, 5),
        }

    def save_model(self, path):
        import joblib

        joblib.dump(
            {
                "model": self.model,
                "model_type": self.model_type,
                "num_classes": getattr(self, "num_classes", None),
            },
            path,
        )

    def load_model(self, path):
        import joblib

        loaded = joblib.load(path)
        if isinstance(loaded, dict) and "model" in loaded:
            self.model = loaded["model"]
            self.num_classes = loaded.get("num_classes")
        else:
            self.model = loaded
        self.is_trained = True


class OneClassSVMWrapper(_ArrayWrapper):
    def __init__(self):
        super().__init__("One-Class SVM")
        from ..model.one_class_svm import OneClassSVMClassifier

        self.OneClassSVMClassifier = OneClassSVMClassifier

    @staticmethod
    def _binary_labels(labels):
        return np.where(np.asarray(labels).reshape(-1) > 0, 1, -1).astype(np.int64)

    def train(self, train_data, val_data=None, **params):
        X_train, y_train = self._unpack_array_data(train_data)
        y_train_binary = self._binary_labels(y_train)
        if val_data is None:
            X_val, y_val = X_train, y_train_binary
        else:
            X_val, y_val = self._unpack_array_data(val_data)
            y_val = self._binary_labels(y_val)

        self.model = self.OneClassSVMClassifier(
            prior=_safe_float(params.get("prior", 0.0), 0.0),
            dim=_safe_int(X_train.shape[1], 1),
            kernel=params.get("kernel", "rbf"),
            nu=_safe_float(params.get("nu", 0.1), 0.1),
            gamma=params.get("gamma", "scale"),
        )
        self.model.fit(X_train, y_train_binary)
        self.is_trained = True
        self.num_classes = 2

        train_pred = np.where(np.asarray(self.model.predict(X_train)).reshape(-1) > 0, 1, 0)
        if val_data is None:
            val_pred = train_pred
        else:
            val_pred = np.where(np.asarray(self.model.predict(X_val)).reshape(-1) > 0, 1, 0)

        train_acc = float(np.mean(train_pred == (y_train_binary > 0).astype(np.int64)))
        val_acc = float(np.mean(val_pred == (y_val > 0).astype(np.int64))) if val_data is not None else None
        return {
            "train_acc": train_acc,
            "val_acc": val_acc,
        }

    def get_default_params(self):
        return {
            "kernel": "rbf",
            "nu": 0.1,
            "gamma": "scale",
        }

    def get_param_space(self):
        return {
            "kernel": ("categorical", ["rbf", "linear", "poly"]),
            "nu": ("float", 0.01, 0.5),
            "gamma": ("categorical", ["scale", "auto", 0.001, 0.01, 0.1, 1.0]),
        }

    def predict(self, data):
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        predictions = np.asarray(self.model.predict(self._extract_inputs(data))).reshape(-1)
        return np.where(predictions > 0, 1, 0).astype(np.int64)

    def predict_proba(self, data):
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        return self.model.predict_proba(self._extract_inputs(data))

    def save_model(self, path):
        import joblib

        joblib.dump(
            {
                "model": self.model,
                "model_type": self.model_type,
                "num_classes": getattr(self, "num_classes", None),
            },
            path,
        )

    def load_model(self, path):
        import joblib

        loaded = joblib.load(path)
        if isinstance(loaded, dict) and "model" in loaded:
            self.model = loaded["model"]
            self.num_classes = loaded.get("num_classes")
        else:
            self.model = loaded
        self.is_trained = True


class TwoStepPUWrapper(_ArrayWrapper):
    def __init__(self):
        super().__init__("Two-Step PU")
        from ..model.two_step_pu import TwoStepPULearning

        self.TwoStepPULearning = TwoStepPULearning

    @staticmethod
    def _binary_labels(labels):
        return np.where(np.asarray(labels).reshape(-1) > 0, 1, -1).astype(np.int64)

    def train(self, train_data, val_data=None, **params):
        X_train, y_train = self._unpack_array_data(train_data)
        y_train_binary = self._binary_labels(y_train)
        if val_data is None:
            X_val, y_val = X_train, y_train_binary
        else:
            X_val, y_val = self._unpack_array_data(val_data)
            y_val = self._binary_labels(y_val)

        self.model = self.TwoStepPULearning(
            prior=_safe_float(params.get("prior", 0.0), 0.0),
            dim=_safe_int(X_train.shape[1], 1),
            spy_percentage=_safe_float(params.get("spy_percentage", 0.1), 0.1),
            threshold_percentile=_safe_float(params.get("threshold_percentile", 10), 10),
            n_estimators=_safe_int(params.get("n_estimators", 100), 100),
            random_state=_safe_int(params.get("random_state", 42), 42),
        )
        self.model.fit(X_train, y_train_binary)
        self.is_trained = True
        self.num_classes = 2

        train_pred = np.where(np.asarray(self.model.predict(X_train)).reshape(-1) > 0, 1, 0)
        if val_data is None:
            val_pred = train_pred
        else:
            val_pred = np.where(np.asarray(self.model.predict(X_val)).reshape(-1) > 0, 1, 0)

        train_acc = float(np.mean(train_pred == (y_train_binary > 0).astype(np.int64)))
        val_acc = float(np.mean(val_pred == (y_val > 0).astype(np.int64))) if val_data is not None else None
        return {
            "train_acc": train_acc,
            "val_acc": val_acc,
        }

    def get_default_params(self):
        return {
            "spy_percentage": 0.1,
            "threshold_percentile": 10,
            "n_estimators": 100,
        }

    def get_param_space(self):
        return {
            "spy_percentage": ("float", 0.05, 0.3),
            "threshold_percentile": ("int", 5, 30),
            "n_estimators": ("int", 50, 300),
        }

    def predict(self, data):
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        predictions = np.asarray(self.model.predict(self._extract_inputs(data))).reshape(-1)
        return np.where(predictions > 0, 1, 0).astype(np.int64)

    def predict_proba(self, data):
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        return self.model.predict_proba(self._extract_inputs(data))

    def save_model(self, path):
        import joblib

        joblib.dump(
            {
                "model": self.model,
                "model_type": self.model_type,
                "num_classes": getattr(self, "num_classes", None),
            },
            path,
        )

    def load_model(self, path):
        import joblib

        loaded = joblib.load(path)
        if isinstance(loaded, dict) and "model" in loaded:
            self.model = loaded["model"]
            self.num_classes = loaded.get("num_classes")
        else:
            self.model = loaded
        self.is_trained = True
