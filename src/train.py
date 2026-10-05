"""Train the custom baseline CNN on one modality at a time."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

try:
    from .dataset import BrainTumorDataset, class_weights, create_dataloader, resolve_device
    from .model import BaselineCNN
    from .preprocessing import load_config, project_root
except ImportError:  # Supports direct execution as well as `python -m src.train`.
    from dataset import BrainTumorDataset, class_weights, create_dataloader, resolve_device
    from model import BaselineCNN
    from preprocessing import load_config, project_root


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def manifest_path(config: dict, modality: str, split: str) -> Path:
    return project_root() / config["paths"]["processed_dir"] / modality.lower() / f"{split}.csv"


def make_loader(config: dict, modality: str, split: str):
    return create_dataloader(
        manifest_path(config, modality, split),
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        mean=config["normalization"]["mean"],
        std=config["normalization"]["std"],
        training=split == "train",
        use_weighted_sampling=(
            modality == "MRI"
            and split == "train"
            and config["mri"].get("weighted_sampling", False)
        ),
    )


def criterion_for_modality(config: dict, modality: str, train_loader, device: torch.device) -> nn.CrossEntropyLoss:
    if modality == "MRI" and config["mri"].get("imbalance_strategy") == "weighted_loss":
        weights = class_weights(train_loader.dataset).to(device)
        return nn.CrossEntropyLoss(weight=weights)
    return nn.CrossEntropyLoss()


def run_epoch(model, loader, criterion, device: torch.device, optimizer=None) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels, _, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            samples += labels.size(0)
    return EpochMetrics(loss=total_loss / samples, accuracy=correct / samples)


def save_training_artifacts(history: dict, modality: str, config: dict) -> None:
    root = project_root()
    metrics_dir = root / config["paths"]["outputs_dir"] / "metrics"
    plots_dir = root / config["paths"]["outputs_dir"] / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{modality.lower()}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["validation_loss"], label="Validation")
    axes[0].set(title=f"{modality} loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["validation_accuracy"], label="Validation")
    axes[1].set(title=f"{modality} accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(plots_dir / f"{modality.lower()}_training_curves.png", dpi=150)
    plt.close(figure)


def train(config: dict, modality: str) -> dict:
    seed_everything(config["random_seed"])
    device = resolve_device(config["device"])
    train_loader = make_loader(config, modality, "train")
    validation_loader = make_loader(config, modality, "validation")
    model = BaselineCNN(num_classes=2).to(device)
    criterion = criterion_for_modality(config, modality, train_loader, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    history = {"train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": []}
    best_validation_loss = float("inf")
    patience = 0
    models_dir = project_root() / config["paths"]["outputs_dir"] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / f"{modality.lower()}_baseline_cnn_best.pt"

    for epoch in range(1, config["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_metrics = run_epoch(model, validation_loader, criterion, device)
        history["train_loss"].append(train_metrics.loss)
        history["validation_loss"].append(validation_metrics.loss)
        history["train_accuracy"].append(train_metrics.accuracy)
        history["validation_accuracy"].append(validation_metrics.accuracy)
        print(
            f"Epoch {epoch:02d}: train loss={train_metrics.loss:.4f}, acc={train_metrics.accuracy:.3%}; "
            f"val loss={validation_metrics.loss:.4f}, acc={validation_metrics.accuracy:.3%}"
        )

        if validation_metrics.loss < best_validation_loss - config["early_stopping_min_delta"]:
            best_validation_loss = validation_metrics.loss
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "modality": modality,
                    "num_classes": 2,
                    "config": config,
                    "best_validation_loss": best_validation_loss,
                },
                checkpoint_path,
            )
        else:
            patience += 1
            if patience >= config["early_stopping_patience"]:
                print(f"Early stopping after {epoch} epochs.")
                break

    save_training_artifacts(history, modality, config)
    return history


def smoke_test(config: dict, modality: str) -> dict:
    """Run one mini-batch and one optimizer step without training an epoch."""
    seed_everything(config["random_seed"])
    device = resolve_device(config["device"])
    loader = make_loader(config, modality, "train")
    model = BaselineCNN(num_classes=2).to(device)
    criterion = criterion_for_modality(config, modality, loader, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    last_loss = None
    input_shape = None
    output_shape = None
    images, labels, _, _ = next(iter(loader))
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    input_shape = tuple(images.shape)
    output_shape = tuple(logits.shape)
    last_loss = loss.item()
    steps = 1

    if output_shape is None or output_shape[1] != 2:
        raise RuntimeError(f"Expected two output classes, received output shape {output_shape}")
    result = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "loss": last_loss,
        "optimizer_steps": steps,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or smoke-test the custom CNN baseline.")
    parser.add_argument("--modality", choices=("CT", "MRI"), required=True)
    parser.add_argument("--smoke-test", action="store_true", help="Run two mini-batches only; do not train an epoch.")
    parser.add_argument("--train", action="store_true", help="Start the full training loop.")
    args = parser.parse_args()
    if args.smoke_test == args.train:
        parser.error("Choose exactly one of --smoke-test or --train.")
    configuration = load_config()
    if args.smoke_test:
        smoke_test(configuration, args.modality)
    else:
        train(configuration, args.modality)
