import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

try:
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover
    DataLoader = None

from .base import MyClassifier


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


class RandomForestBinaryClassifier(MyClassifier):
    """Supervised Random Forest wrapper using 1/-1 labels."""

    def __init__(
        self,
        prior,
        dim,
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        class_weight="balanced",
        random_state=42,
    ):
        super(RandomForestBinaryClassifier, self).__init__(_safe_float(prior, 0.0))
        self.input_dim = _safe_int(dim, 1)
        self.n_estimators = _safe_int(n_estimators, 200)
        self.max_depth = None if max_depth in ("", None) else _safe_int(max_depth, 12)
        self.min_samples_split = _safe_int(min_samples_split, 2)
        self.min_samples_leaf = _safe_int(min_samples_leaf, 1)
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.class_weight = class_weight
        self.random_state = _safe_int(random_state, 42)
        self.model = None
        self.is_fitted = False
        self.device = None
        self.feature_mean = None
        self.feature_std = None

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            self.device = x.device
            if len(x.shape) == 4:
                x = x.view(x.size(0), -1)
            x = x.detach().cpu().numpy()
        elif DataLoader is not None and isinstance(x, DataLoader):
            features = []
            for batch in x:
                batch_inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
                if isinstance(batch_inputs, torch.Tensor):
                    batch_inputs = batch_inputs.detach().cpu().numpy()
                else:
                    batch_inputs = np.asarray(batch_inputs)
                if batch_inputs.ndim == 4:
                    batch_inputs = batch_inputs.reshape(batch_inputs.shape[0], -1)
                features.append(np.asarray(batch_inputs))
            if not features:
                return np.empty((0, self.input_dim), dtype=np.float32)
            x = np.concatenate(features, axis=0)
        else:
            x = np.asarray(x)
            if x.ndim == 4:
                x = x.reshape(x.shape[0], -1)
        return np.asarray(x, dtype=np.float32)

    def _normalize(self, x):
        if self.feature_mean is None or self.feature_std is None:
            return x
        return (x - self.feature_mean) / self.feature_std

    def _build_binary_labels(self, y):
        y = np.asarray(y).reshape(-1)
        return np.where(y > 0, 1, 0).astype(np.int64)

    def fit(self, x, y):
        x = self._to_numpy(x)
        y = self._build_binary_labels(y)
        if len(np.unique(y)) < 2:
            raise ValueError("RF requires both positive and negative/unlabeled samples.")

        self.feature_mean = x.mean(axis=0)
        self.feature_std = x.std(axis=0) + 1e-10
        x = self._normalize(x)

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.model.fit(x, y)
        self.is_fitted = True
        return self

    def predict_proba(self, x):
        x = self._to_numpy(x)
        if not self.is_fitted or self.model is None:
            return np.full((len(x), 2), 0.5, dtype=np.float32)

        x = self._normalize(x)
        probabilities = self.model.predict_proba(x)
        classes = np.asarray(getattr(self.model, "classes_", []))
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2 and 1 in classes:
            positive_index = int(np.where(classes == 1)[0][0])
            positive_prob = probabilities[:, positive_index]
        elif probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            positive_prob = probabilities[:, -1]
        else:
            positive_prob = np.asarray(probabilities).reshape(-1)
        positive_prob = np.clip(np.asarray(positive_prob, dtype=np.float32), 0.0, 1.0)
        negative_prob = 1.0 - positive_prob
        return np.column_stack([negative_prob, positive_prob]).astype(np.float32)

    def forward(self, x):
        probabilities = self.predict_proba(x)
        logits = probabilities[:, 1] - 0.5
        outputs = torch.FloatTensor(logits).view(-1, 1)
        if self.device is not None:
            outputs = outputs.to(self.device)
        return outputs

    def predict(self, x):
        probabilities = self.predict_proba(x)
        return np.where(probabilities[:, 1] >= 0.5, 1, -1).astype(np.int64)
