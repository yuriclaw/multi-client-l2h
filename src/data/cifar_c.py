"""CIFAR-10-C / CIFAR-100-C data loading utilities.

Downloads corrupted CIFAR benchmarks and creates per-client datasets
where each client sees a different corruption type, simulating
heterogeneous distribution shift.

Reference: Hendrycks & Dietterich, "Benchmarking Neural Network Robustness
to Common Corruptions and Perturbations", ICLR 2019.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T

# All 15 CIFAR-C corruption types
ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

# URLs for downloading (zenodo mirrors)
CIFAR10C_URL = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
CIFAR100C_URL = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"


class CIFARCorruptedDataset(Dataset):
    """Dataset for a single corruption type + severity from CIFAR-C.

    Expects the standard layout:
        data_dir/CIFAR-{10,100}-C/{corruption}.npy   — (50000, 32, 32, 3)
        data_dir/CIFAR-{10,100}-C/labels.npy          — (50000,)
    Severity levels 1-5 are stacked in chunks of 10000.
    """

    def __init__(
        self,
        data_dir: str,
        corruption: str,
        severity: int = 3,
        cifar_version: int = 10,
        transform: Optional[T.Compose] = None,
    ) -> None:
        assert severity in range(1, 6), "Severity must be 1-5"
        assert corruption in ALL_CORRUPTIONS, f"Unknown corruption: {corruption}"

        folder = f"CIFAR-{cifar_version}-C"
        images_path = os.path.join(data_dir, folder, f"{corruption}.npy")
        labels_path = os.path.join(data_dir, folder, "labels.npy")

        if not os.path.exists(images_path):
            raise FileNotFoundError(
                f"{images_path} not found. Download CIFAR-{cifar_version}-C to {data_dir}."
            )

        all_images = np.load(images_path)       # (50000, 32, 32, 3)
        all_labels = np.load(labels_path)        # (50000,)

        # Each severity has 10k images
        start = (severity - 1) * 10_000
        end = severity * 10_000
        self.images = all_images[start:end]      # (10000, 32, 32, 3)
        self.labels = all_labels[start:end].astype(np.int64)

        self.transform = transform or T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]                   # (32, 32, 3) uint8
        label = int(self.labels[idx])
        img = self.transform(img)
        return img, label


def build_client_datasets(
    data_dir: str,
    corruptions: List[str],
    severity: int = 3,
    cifar_version: int = 10,
    transform: Optional[T.Compose] = None,
) -> dict[str, CIFARCorruptedDataset]:
    """Create one dataset per client, each with a different corruption.

    Returns:
        Dict mapping client_id → dataset.
    """
    datasets: dict[str, CIFARCorruptedDataset] = {}
    for i, corruption in enumerate(corruptions):
        client_id = f"client_{i}"
        datasets[client_id] = CIFARCorruptedDataset(
            data_dir=data_dir,
            corruption=corruption,
            severity=severity,
            cifar_version=cifar_version,
            transform=transform,
        )
    return datasets


def build_mixed_test_set(
    data_dir: str,
    corruptions: List[str],
    severity: int = 3,
    cifar_version: int = 10,
    samples_per_corruption: int = 1000,
) -> Dataset:
    """Build a mixed test set sampling from multiple corruptions."""
    all_images, all_labels = [], []
    for corruption in corruptions:
        ds = CIFARCorruptedDataset(data_dir, corruption, severity, cifar_version)
        indices = np.random.choice(len(ds), samples_per_corruption, replace=False)
        for idx in indices:
            img, label = ds[idx]
            all_images.append(img)
            all_labels.append(label)

    class _MixedDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]

    return _MixedDataset(all_images, all_labels)
