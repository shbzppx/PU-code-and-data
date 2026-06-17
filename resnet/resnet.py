from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class _BaseResNet(nn.Module):
    def __init__(self, num_classes=2, input_channels=1, width=32, blocks=2, *args, **kwargs):
        super().__init__()
        del args, kwargs
        self.num_classes = max(int(num_classes), 2)
        self.input_channels = int(input_channels)
        self.normalization_stats = None

        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.residual_layers = nn.Sequential(*[ResidualBlock(width) for _ in range(max(int(blocks), 1))])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(width, self.num_classes),
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D tensor, got shape {tuple(x.shape)}")
        x = self.stem(x)
        x = self.residual_layers(x)
        return self.head(x)


class ResNet18(_BaseResNet):
    def __init__(self, num_classes=2, input_channels=1, image_size=None, *args, **kwargs):
        del image_size
        super().__init__(num_classes=num_classes, input_channels=input_channels, width=32, blocks=2, *args, **kwargs)


class ResNet34(_BaseResNet):
    def __init__(self, num_classes=2, input_channels=1, image_size=None, *args, **kwargs):
        del image_size
        super().__init__(num_classes=num_classes, input_channels=input_channels, width=48, blocks=3, *args, **kwargs)


class ResNet50(_BaseResNet):
    def __init__(self, num_classes=2, input_channels=1, image_size=None, *args, **kwargs):
        del image_size
        super().__init__(num_classes=num_classes, input_channels=input_channels, width=64, blocks=4, *args, **kwargs)
