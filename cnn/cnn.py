from __future__ import annotations

import torch
from torch import nn


class CNNModel(nn.Module):
    """Lightweight CNN used as a compatibility bridge for model_comparison."""

    def __init__(
        self,
        num_classes=2,
        input_channels=1,
        image_size=None,
        hidden_channels=32,
        *args,
        **kwargs,
    ):
        super().__init__()
        del args, kwargs
        self.num_classes = max(int(num_classes), 2)
        self.input_channels = int(input_channels)
        self.image_size = image_size
        self.normalization_stats = None

        mid_channels = int(hidden_channels)
        high_channels = mid_channels * 2
        top_channels = mid_channels * 4

        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, high_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(high_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(high_channels, top_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(top_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(top_channels, self.num_classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D tensor, got shape {tuple(x.shape)}")
        features = self.features(x).flatten(1)
        return self.classifier(features)
