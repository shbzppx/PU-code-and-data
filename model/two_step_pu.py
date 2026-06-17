import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
try:
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover - torch may be unavailable in some envs
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


class TwoStepPULearning(MyClassifier):
    def __init__(
        self,
        prior,
        dim,
        spy_percentage=0.1,
        threshold_percentile=10,
        n_estimators=100,
        random_state=42,
    ):
        super(TwoStepPULearning, self).__init__(_safe_float(prior, 0.0))
        self.input_dim = _safe_int(dim, 1)
        self.spy_percentage = _safe_float(spy_percentage, 0.1)
        self.threshold_percentile = _safe_float(threshold_percentile, 10)
        self.n_estimators = _safe_int(n_estimators, 100)
        self.random_state = _safe_int(random_state, 42)
        self.base_estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.reliable_negatives_estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.is_fitted = False
        self.device = None
        self.threshold = None
        self.feature_mean = None
        self.feature_std = None

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            self.device = x.device
            if len(x.shape) == 4:
                x = x.view(x.size(0), -1)
            x = x.cpu().numpy()
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
        return np.where(y > 0, 1, -1).astype(np.int64)

    def fit(self, x, y):
        x = self._to_numpy(x)
        y = self._build_binary_labels(y)

        positive_samples = x[y == 1]
        unlabeled_samples = x[y == -1]

        if len(positive_samples) == 0:
            raise ValueError("Two-Step PU requires at least one positive sample.")

        if len(unlabeled_samples) == 0 or len(positive_samples) < 2:
            # Fallback to a simple supervised classifier when PU splitting is not viable.
            y_fallback = (y > 0).astype(np.int64)
            self.feature_mean = x.mean(axis=0)
            self.feature_std = x.std(axis=0) + 1e-10
            x_norm = self._normalize(x)
            self.reliable_negatives_estimator = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=1,
            )
            self.reliable_negatives_estimator.fit(x_norm, y_fallback)
            self.threshold = 0.5
            self.is_fitted = True
            return

        self.feature_mean = x.mean(axis=0)
        self.feature_std = x.std(axis=0) + 1e-10
        positive_samples = self._normalize(positive_samples)
        unlabeled_samples = self._normalize(unlabeled_samples)

        n_spies = max(int(len(positive_samples) * self.spy_percentage), 1)
        n_spies = min(n_spies, max(len(positive_samples) - 1, 1))
        rng = np.random.default_rng(self.random_state)
        spy_indices = rng.choice(len(positive_samples), n_spies, replace=False)

        spy_samples = positive_samples[spy_indices]
        remaining_positive = np.delete(positive_samples, spy_indices, axis=0)

        X_train = np.vstack([remaining_positive, unlabeled_samples])
        y_train = np.hstack(
            [
                np.ones(len(remaining_positive), dtype=np.int64),
                np.zeros(len(unlabeled_samples), dtype=np.int64),
            ]
        )

        self.base_estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.base_estimator.fit(X_train, y_train)

        spy_scores = self.base_estimator.predict_proba(spy_samples)[:, 1]
        self.threshold = float(np.percentile(spy_scores, self.threshold_percentile))

        unlabeled_scores = self.base_estimator.predict_proba(unlabeled_samples)[:, 1]
        reliable_negative_indices = unlabeled_scores < self.threshold
        reliable_negatives = unlabeled_samples[reliable_negative_indices]

        X_final = np.vstack([positive_samples, reliable_negatives])
        y_final = np.hstack(
            [
                np.ones(len(positive_samples), dtype=np.int64),
                np.zeros(len(reliable_negatives), dtype=np.int64),
            ]
        )

        self.reliable_negatives_estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.reliable_negatives_estimator.fit(X_final, y_final)
        self.is_fitted = True

    def forward(self, x):
        x = self._to_numpy(x)
        if not self.is_fitted:
            return torch.zeros(len(x), 1).to(self.device or "cpu")

        x = self._normalize(x)
        predictions = self.reliable_negatives_estimator.predict(x)
        predictions = predictions * 2 - 1
        predictions = torch.FloatTensor(predictions).view(-1, 1)
        if self.device is not None:
            predictions = predictions.to(self.device)
        return predictions

    def predict(self, x):
        outputs = self.forward(x)
        if isinstance(outputs, torch.Tensor):
            predictions = outputs.cpu().detach().numpy()
        else:
            predictions = np.asarray(outputs)
        if len(predictions.shape) > 1 and predictions.shape[1] == 1:
            predictions = predictions.reshape(-1)
        return np.where(predictions > 0, 1, -1)

    def predict_proba(self, x):
        x = self._to_numpy(x)
        if not self.is_fitted:
            return np.full((len(x), 2), 0.5, dtype=np.float32)

        x = self._normalize(x)
        if hasattr(self.reliable_negatives_estimator, "predict_proba"):
            probabilities = self.reliable_negatives_estimator.predict_proba(x)
            if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
                return probabilities[:, :2].astype(np.float32)

        predictions = self.reliable_negatives_estimator.predict(x)
        positive_prob = np.asarray(predictions, dtype=np.float32).reshape(-1)
        positive_prob = np.clip(positive_prob, 0.0, 1.0)
        negative_prob = 1.0 - positive_prob
        return np.column_stack([negative_prob, positive_prob]).astype(np.float32)
