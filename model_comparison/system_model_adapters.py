"""Adapters that expose the system's native models as 2-class classifiers."""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - 运行环境可能未安装 torch
    class _TorchFallback:
        @staticmethod
        def is_tensor(value):
            return False

        @staticmethod
        def as_tensor(*args, **kwargs):
            raise ImportError("torch is required for neural adapters")

        @staticmethod
        def cat(*args, **kwargs):
            raise ImportError("torch is required for neural adapters")

    class nn:  # type: ignore[override]
        class Module:
            pass

    torch = _TorchFallback()

from ..model.cnn import CNN
from ..model.cnn_token_transformer import CNNTokenTransformer
from ..model.cnn_transformer import CNNTransformer
from ..model.linear import LinearClassifier
from ..model.mlp import MultiLayerPerceptron, ThreeLayerPerceptron
from .model_wrappers import SYSTEM_MODEL_SPECS, resolve_system_model_name


def _resolve_neural_spec(model_name: str):
    resolved_name = resolve_system_model_name(model_name)
    spec = dict(SYSTEM_MODEL_SPECS.get(resolved_name, {}))
    if spec.get("family") != "neural":
        raise ValueError(f"Model '{resolved_name}' is not a neural model and cannot be wrapped here.")
    return resolved_name, spec


def _normalize_image_shape(image_size) -> Tuple[int, int]:
    if isinstance(image_size, (tuple, list)):
        if len(image_size) >= 2:
            return int(image_size[0]), int(image_size[1])
        if len(image_size) == 1:
            side = int(image_size[0])
            return side, side
    side = int(image_size or 1)
    return side, side


class SystemBinaryAdapter(nn.Module):
    """Wrap a system-native score model as a 2-logit classifier."""

    def __init__(
        self,
        model_key: str,
        *,
        num_classes: int = 2,
        input_channels: int = 1,
        image_size=64,
        prior: float = 0.5,
    ):
        super().__init__()
        self.model_key = str(model_key)
        self.num_classes = int(num_classes or 2)
        self.input_channels = int(input_channels or 1)
        self.image_size = _normalize_image_shape(image_size)
        self.prior = float(prior)
        self.backbone = self._build_backbone()

    @property
    def input_dim(self) -> int:
        height, width = self.image_size
        return int(self.input_channels * height * width)

    def _build_backbone(self):
        key = self.model_key.lower()
        input_shape = (self.input_channels, self.image_size[0], self.image_size[1])
        if key == "linear":
            return LinearClassifier(prior=self.prior, dim=self.input_dim)
        if key == "3lp":
            return ThreeLayerPerceptron(prior=self.prior, dim=self.input_dim)
        if key == "mlp":
            return MultiLayerPerceptron(prior=self.prior, dim=self.input_dim)
        if key == "cnn":
            return CNN(prior=self.prior, input_dim=self.input_dim, input_shape=input_shape)
        if key == "cnnt":
            return CNNTransformer(prior=self.prior, input_dim=self.input_dim, input_shape=input_shape)
        if key == "cntt":
            return CNNTokenTransformer(prior=self.prior, input_dim=self.input_dim, input_shape=input_shape)
        raise ValueError(f"Unsupported system neural model key: {self.model_key}")

    def forward(self, x):
        score = self.backbone(x)
        if not torch.is_tensor(score):
            score = torch.as_tensor(score, dtype=torch.float32, device=x.device)
        else:
            score = score.float()

        if score.ndim == 1:
            score = score.unsqueeze(1)
        elif score.ndim > 2:
            score = score.reshape(score.shape[0], -1)

        if score.shape[1] != 1:
            score = score.mean(dim=1, keepdim=True)

        return torch.cat([-score, score], dim=1)


@lru_cache(maxsize=None)
def build_system_model_class(model_name: str):
    resolved_name, spec = _resolve_neural_spec(model_name)
    model_key = spec["key"]
    class_name = f"System{model_key.upper()}Adapter"

    class _SystemModel(SystemBinaryAdapter):
        def __init__(self, num_classes=2, input_channels=1, image_size=64, prior=0.5, **kwargs):
            del kwargs
            super().__init__(
                model_key,
                num_classes=num_classes,
                input_channels=input_channels,
                image_size=image_size,
                prior=prior,
            )

    _SystemModel.__name__ = class_name
    _SystemModel.__qualname__ = class_name
    _SystemModel.__module__ = __name__
    _SystemModel.display_name = resolved_name
    return _SystemModel


def build_system_model_wrapper(
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
    del profile
    resolved_name, _spec = _resolve_neural_spec(model_name)
    model_class = build_system_model_class(resolved_name)
    meta = dict(dataset_meta or {})
    num_classes = 2
    input_channels = int(meta.get("input_channels") or 1)
    image_size = meta.get("image_size", 64)

    from .model_wrappers import DeepLearningWrapper, apply_model_metadata

    wrapper = DeepLearningWrapper(
        model_class,
        num_classes,
        input_channels,
        image_size,
        model_name=resolved_name,
        fixed_epochs=fixed_epochs,
        persist_artifacts=persist_artifacts,
        artifact_dir=artifact_dir,
    )
    return apply_model_metadata(wrapper, resolved_name)


def build_pu_cnnt_wrapper(
    dataset_meta=None,
    profile="comparison",
    *,
    fixed_epochs=None,
    persist_artifacts=False,
    artifact_dir=None,
    training_config=None,
    automl_config=None,
):
    del profile
    meta = dict(dataset_meta or {})
    input_channels = int(meta.get("input_channels") or 1)
    image_size = meta.get("image_size", 64)

    from .model_wrappers import PUCNNTWrapper

    return PUCNNTWrapper(
        input_channels=input_channels,
        image_size=image_size,
        model_name="PU-CNN-Transformer",
        fixed_epochs=fixed_epochs,
        persist_artifacts=persist_artifacts,
        artifact_dir=artifact_dir,
        training_config=training_config,
        automl_config=automl_config,
    )


def build_pu_cnn_wrapper(
    dataset_meta=None,
    profile="comparison",
    *,
    fixed_epochs=None,
    persist_artifacts=False,
    artifact_dir=None,
    training_config=None,
    automl_config=None,
):
    del profile
    meta = dict(dataset_meta or {})
    input_channels = int(meta.get("input_channels") or 1)
    image_size = meta.get("image_size", 64)

    from .model_wrappers import PUCNNWrapper

    return PUCNNWrapper(
        input_channels=input_channels,
        image_size=image_size,
        model_name="PU-CNN",
        fixed_epochs=fixed_epochs,
        persist_artifacts=persist_artifacts,
        artifact_dir=artifact_dir,
        training_config=training_config,
        automl_config=automl_config,
    )


def build_pu_cnn_transformer_wrapper(
    dataset_meta=None,
    profile="comparison",
    *,
    fixed_epochs=None,
    persist_artifacts=False,
    artifact_dir=None,
    training_config=None,
    automl_config=None,
):
    del profile
    meta = dict(dataset_meta or {})
    input_channels = int(meta.get("input_channels") or 1)
    image_size = meta.get("image_size", 64)

    from .model_wrappers import PUCNNTransformerWrapper

    return PUCNNTransformerWrapper(
        input_channels=input_channels,
        image_size=image_size,
        model_name="PU-CNN-TokenTransformer",
        fixed_epochs=fixed_epochs,
        persist_artifacts=persist_artifacts,
        artifact_dir=artifact_dir,
        training_config=training_config,
        automl_config=automl_config,
    )
