"""Unified model wrappers for deep learning and classic ML models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import shutil
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader


def _should_drop_last_for_training(sample_count, batch_size) -> bool:
    batch_size = int(batch_size or 0)
    sample_count = int(sample_count or 0)
    return batch_size > 1 and sample_count > 1 and (sample_count % batch_size) == 1


class BaseModelWrapper(ABC):
    """Base contract shared by all AutoML model wrappers."""

    data_mode = "array"

    def __init__(self, model_type):
        self.model_type = model_type
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(self, train_data, val_data, **params):
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
    """Wrapper for CNN and ResNet models used by AutoML."""

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

    def train(self, train_data, val_data, **params):
        from ..cnn.trainer import Trainer

        if not isinstance(train_data, DataLoader) or not isinstance(val_data, DataLoader):
            raise TypeError("DeepLearningWrapper expects DataLoader inputs.")

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
        if self.current_trainer is not None and hasattr(self.current_trainer, "stop"):
            self.current_trainer.stop()


class _ArrayWrapper(BaseModelWrapper):
    data_mode = "array"

    def _prepare_features(self, features):
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
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

    def train(self, train_data, val_data, **params):
        X_train, y_train = self._unpack_array_data(train_data)
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
        self.num_classes = int(len(np.unique(np.concatenate([y_train, y_val]))))
        return {
            "train_acc": float(self.model.score(X_train, y_train)),
            "val_acc": float(self.model.score(X_val, y_val)),
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


class SVMWrapper(_ArrayWrapper):
    def __init__(self):
        super().__init__("SVM")
        from sklearn.svm import SVC

        self.SVC = SVC

    def train(self, train_data, val_data, **params):
        X_train, y_train = self._unpack_array_data(train_data)
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
        self.num_classes = int(len(np.unique(np.concatenate([y_train, y_val]))))
        return {
            "train_acc": float(self.model.score(X_train, y_train)),
            "val_acc": float(self.model.score(X_val, y_val)),
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

    def train(self, train_data, val_data, **params):
        X_train, y_train = self._unpack_array_data(train_data)
        X_val, y_val = self._unpack_array_data(val_data)
        self.model = self.DecisionTreeClassifier(
            max_depth=params.get("max_depth"),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=42,
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.num_classes = int(len(np.unique(np.concatenate([y_train, y_val]))))
        return {
            "train_acc": float(self.model.score(X_train, y_train)),
            "val_acc": float(self.model.score(X_val, y_val)),
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
