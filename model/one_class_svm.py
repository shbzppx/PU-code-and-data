import numpy as np
import torch
from sklearn.svm import OneClassSVM
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


class OneClassSVMClassifier(MyClassifier):
    def __init__(self, prior, dim, kernel='rbf', nu=0.1, gamma='scale'):
        super(OneClassSVMClassifier, self).__init__(_safe_float(prior, 0.0))
        self.input_dim = int(dim)
        self.kernel = kernel
        self.nu = _safe_float(nu, 0.1)
        self.gamma = gamma
        self.feature_mean = None
        self.feature_std = None
        self.svm = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        self.is_fitted = False
        self.device = None

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

    def fit(self, x, y):
        x = self._to_numpy(x)
        y = np.asarray(y).reshape(-1)
        positive_samples = x[y == 1]
        if len(positive_samples) == 0:
            raise ValueError("One-Class SVM requires at least one positive sample.")

        self.feature_mean = positive_samples.mean(axis=0)
        self.feature_std = positive_samples.std(axis=0) + 1e-10
        positive_samples = self._normalize(positive_samples)

        self.svm = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        self.svm.fit(positive_samples)
        self.is_fitted = True

    def forward(self, x):
        x = self._to_numpy(x)
        if not self.is_fitted:
            return torch.zeros(len(x), 1).to(self.device or "cpu")

        x = self._normalize(x)
        predictions = self.svm.predict(x)
        predictions = torch.FloatTensor(-predictions).view(-1, 1)
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
        return np.where(predictions < 0, 1, -1)

    def predict_proba(self, x):
        x = self._to_numpy(x)
        if not self.is_fitted:
            return np.full((len(x), 2), 0.5, dtype=np.float32)

        x = self._normalize(x)
        scores = self.svm.decision_function(x)
        positive_prob = 1.0 / (1.0 + np.exp(-scores))
        negative_prob = 1.0 - positive_prob
        return np.column_stack([negative_prob, positive_prob]).astype(np.float32)
