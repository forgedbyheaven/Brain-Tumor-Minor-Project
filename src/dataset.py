"""PyTorch datasets and loaders backed by immutable raw-image manifests."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Mapping

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


def build_transforms(
    image_size: int,
    training: bool,
    mean: list[float],
    std: list[float],
    augmentation: Mapping[str, object] | None = None,
):
    """Build transforms for one split without ever changing source image files.

    Random augmentation is deliberately limited to training samples. Applying it
    after the split prevents augmented variants of an image from leaking into
    validation or test data, whose transforms remain fully deterministic.
    """
    # BrainTumorDataset converts every source image to RGB before applying this transform.
    # Keeping transforms free of local callables also makes them safe for Windows DataLoader workers.
    steps = []
    settings = augmentation or {}
    if training and settings.get("enabled", True):
        horizontal_flip_probability = float(settings.get("horizontal_flip_probability", 0.0))
        rotation_degrees = float(settings.get("rotation_degrees", 0.0))
        translate = tuple(settings.get("translate", (0.0, 0.0)))
        scale = tuple(settings.get("scale", (1.0, 1.0)))
        brightness = float(settings.get("brightness", 0.0))
        contrast = float(settings.get("contrast", 0.0))

        # Conservative spatial and intensity changes improve robustness to normal
        # acquisition/position variation in both CT and MRI without altering labels.
        if horizontal_flip_probability > 0:
            steps.append(transforms.RandomHorizontalFlip(p=horizontal_flip_probability))
        if rotation_degrees > 0:
            steps.append(transforms.RandomRotation(degrees=rotation_degrees))
        if translate != (0.0, 0.0) or scale != (1.0, 1.0):
            steps.append(transforms.RandomAffine(degrees=0, translate=translate, scale=scale))
        if brightness > 0 or contrast > 0:
            steps.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))

    # Resize follows augmentation so all augmented images retain the model's 224x224 input size.
    steps.extend([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
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
    augmentation: Mapping[str, object] | None = None,
) -> DataLoader:
    dataset = BrainTumorDataset(
        manifest_path,
        transform=build_transforms(
            image_size,
            training=training,
            mean=mean,
            std=std,
            augmentation=augmentation,
        ),
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
