"""PyTorch datasets and loaders backed by immutable raw-image manifests."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


CLASS_LABELS = {"Healthy": 0, "Tumor": 1}


def resolve_device(preference: str = "auto") -> torch.device:
    """Select CUDA when available; otherwise use CPU."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def build_transforms(image_size: int, training: bool, mean: list[float], std: list[float]):
    # BrainTumorDataset converts every source image to RGB before applying this transform.
    # Keeping transforms free of local callables also makes them safe for Windows DataLoader workers.
    steps = [transforms.Resize((image_size, image_size))]
    if training:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.10, contrast=0.10),
            ]
        )
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(steps)


class BrainTumorDataset(Dataset):
    """Returns (image_tensor, class_label, source_path, modality)."""

    def __init__(self, manifest_path: str | Path, transform=None):
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = row["source_path"]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image) if self.transform else image
        return tensor, int(row["label"]), image_path, row["modality"]

    @property
    def labels(self) -> list[int]:
        return [int(row["label"]) for row in self.rows]


def class_weights(dataset: BrainTumorDataset) -> torch.Tensor:
    counts = Counter(dataset.labels)
    total = len(dataset)
    return torch.tensor([total / (2 * counts[label]) for label in (0, 1)], dtype=torch.float32)


def weighted_sampler(dataset: BrainTumorDataset) -> WeightedRandomSampler:
    weights = class_weights(dataset)
    sample_weights = [weights[label].item() for label in dataset.labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def create_dataloader(
    manifest_path: str | Path,
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    mean: list[float],
    std: list[float],
    training: bool,
    use_weighted_sampling: bool = False,
) -> DataLoader:
    dataset = BrainTumorDataset(
        manifest_path,
        transform=build_transforms(image_size, training=training, mean=mean, std=std),
    )
    sampler = weighted_sampler(dataset) if training and use_weighted_sampling else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
