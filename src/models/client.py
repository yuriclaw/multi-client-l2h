"""Client models for the L2H framework.

Each client has a lightweight local model that handles 'easy' inputs.
Models return logits and support freezing for deployment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional


class LeNetClient(nn.Module):
    """LeNet-5 adapted for CIFAR-10/100 (32×32 RGB inputs)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SmallCNNClient(nn.Module):
    """Minimal 2-layer CNN client."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class MobileNetClient(nn.Module):
    """MobileNet-v2 wrapper via timm (pretrained optional)."""

    def __init__(self, num_classes: int = 10, pretrained: bool = False) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv2_100", pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_client(
    name: str, num_classes: int = 10, pretrained: bool = False
) -> nn.Module:
    """Factory function to create a client model by name."""
    registry = {
        "lenet": lambda: LeNetClient(num_classes),
        "smallcnn": lambda: SmallCNNClient(num_classes),
        "mobilenet": lambda: MobileNetClient(num_classes, pretrained),
    }
    if name not in registry:
        raise ValueError(f"Unknown client model: {name}. Choose from {list(registry)}")
    return registry[name]()


def freeze_model(model: nn.Module) -> None:
    """Freeze all parameters of a model."""
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
