"""Compatibility helpers for model comparison workflows."""

from .cnn import CNNModel
from .data_loader import H5Dataset, MyDataset, _patch_indices_to_geo, _split_relative_indices
from .trainer import Trainer

__all__ = [
    "CNNModel",
    "H5Dataset",
    "MyDataset",
    "Trainer",
    "_patch_indices_to_geo",
    "_split_relative_indices",
]
