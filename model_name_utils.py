from __future__ import annotations

from collections import OrderedDict


WORKFLOW_MODEL_DISPLAY_NAMES = OrderedDict(
    [
        ("linear", "线性分类器"),
        ("3lp", "三层感知机"),
        ("mlp", "多层感知机"),
        ("cnn", "CNN"),
        ("cnnt", "CNN-Transformer"),
        ("cntt", "CNN-TokenTransformer"),
        ("pucnn", "PU-CNN"),
        ("pucnnt", "PU-CNN-Transformer"),
        ("pucnntransformer", "PU-CNN-TokenTransformer"),
        ("rf", "RF"),
        ("purf", "PU-Random Forest"),
        ("ocsvm", "One-Class SVM"),
        ("2step", "Two-Step PU"),
    ]
)

WORKFLOW_MODEL_KEYS = tuple(WORKFLOW_MODEL_DISPLAY_NAMES.keys())
SHAP_MODEL_KEYS = ("cnnt", "cnn", "cntt", "pucnn", "pucnnt", "pucnntransformer")

_DISPLAY_TO_KEY = {
    display_name: model_key for model_key, display_name in WORKFLOW_MODEL_DISPLAY_NAMES.items()
}

_ALIAS_TO_KEY = {
    "linear": "linear",
    "线性分类器": "linear",
    "线性分类器 (pu)": "linear",
    "线性分类器（pu）": "linear",
    "3lp": "3lp",
    "三层感知机": "3lp",
    "三层感知机 (pu)": "3lp",
    "三层感知机（pu）": "3lp",
    "mlp": "mlp",
    "多层感知机": "mlp",
    "多层感知机 (pu)": "mlp",
    "多层感知机（pu）": "mlp",
    "cnn": "cnn",
    "pu-cnn": "pucnn",
    "cnnt": "cnnt",
    "cnn-transformer": "cnnt",
    "cnn-tokentransformer": "cntt",
    "cntt": "cntt",
    "pucnn": "pucnn",
    "pucnnt": "pucnnt",
    "pu-cnnt": "pucnnt",
    "pu-cnn-transformer": "pucnnt",
    "pucnntransformer": "pucnntransformer",
    "pu-cnntransformer": "pucnntransformer",
    "pu-cnn-tokentransformer": "pucnntransformer",
    "rf": "rf",
    "random forest": "rf",
    "ocsvm": "ocsvm",
    "one-class svm": "ocsvm",
    "one class svm": "ocsvm",
    "2step": "2step",
    "two-step pu": "2step",
    "two step pu": "2step",
    "purf": "purf",
    "pu-random forest": "purf",
    "pu random forest": "purf",
}


def normalize_model_key(value) -> str:
    """Return the internal model key for a known display name or alias."""
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    if text in WORKFLOW_MODEL_DISPLAY_NAMES:
        return text
    if text in _DISPLAY_TO_KEY:
        return _DISPLAY_TO_KEY[text]

    return _ALIAS_TO_KEY.get(text.lower(), text)


def get_model_display_name(value) -> str:
    """Return the user-facing display name for an internal model key."""
    model_key = normalize_model_key(value)
    if model_key in WORKFLOW_MODEL_DISPLAY_NAMES:
        return WORKFLOW_MODEL_DISPLAY_NAMES[model_key]
    return str(value)


def get_model_display_names(model_keys=None) -> list[str]:
    keys = WORKFLOW_MODEL_KEYS if model_keys is None else tuple(model_keys)
    return [get_model_display_name(model_key) for model_key in keys]
