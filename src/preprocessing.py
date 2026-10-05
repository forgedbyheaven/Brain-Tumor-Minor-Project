"""Build reproducible, duplicate-safe manifests without altering raw images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_LABELS = {"Healthy": 0, "Tumor": 1}
MODALITY_DIRECTORIES = {
    "CT": "Brain Tumor CT scan Images",
    "MRI": "Brain Tumor MRI images",
}
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class ImageRecord:
    source_path: Path
    relative_path: str
    modality: str
    class_name: str
    label: int
    sha256: str
    duplicate_group_size: int = 1


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: Path | None = None) -> dict:
    """Load config.yaml. JSON is valid YAML, so this has no runtime YAML dependency."""
    path = config_path or project_root() / "config.yaml"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_records(raw_root: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for modality, directory_name in MODALITY_DIRECTORIES.items():
        for class_name, label in CLASS_LABELS.items():
            class_dir = raw_root / directory_name / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Expected class directory not found: {class_dir}")
            for path in sorted(class_dir.iterdir(), key=lambda item: item.name.lower()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    records.append(
                        ImageRecord(
                            source_path=path.resolve(),
                            relative_path=path.relative_to(raw_root).as_posix(),
                            modality=modality,
                            class_name=class_name,
                            label=label,
                            sha256=sha256_file(path),
                        )
                    )
    return records


def deduplicate(records: Iterable[ImageRecord]) -> tuple[list[ImageRecord], dict]:
    """Keep one stable representative per exact SHA-256 group.

    Raw data is never changed. A duplicate group must not span labels or modalities;
    this guard prevents accidental label/modality leakage in a future dataset update.
    """
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[record.sha256].append(record)

    representatives: list[ImageRecord] = []
    duplicate_groups = 0
    redundant_files = 0
    removed_by_modality_class: Counter[str] = Counter()

    for digest, group in sorted(groups.items()):
        group = sorted(group, key=lambda item: item.relative_path.lower())
        modalities = {item.modality for item in group}
        labels = {item.label for item in group}
        if len(modalities) != 1 or len(labels) != 1:
            raise ValueError(
                f"Exact duplicate group {digest} spans modality or class labels: "
                f"{[item.relative_path for item in group]}"
            )
        if len(group) > 1:
            duplicate_groups += 1
            redundant_files += len(group) - 1
            removed_by_modality_class[f"{group[0].modality}/{group[0].class_name}"] += len(group) - 1
        representative = group[0]
        representatives.append(
            ImageRecord(
                **{
                    **representative.__dict__,
                    "duplicate_group_size": len(group),
                }
            )
        )

    report = {
        "raw_images": sum(len(group) for group in groups.values()),
        "unique_images": len(representatives),
        "exact_duplicate_groups": duplicate_groups,
        "redundant_duplicate_files_removed": redundant_files,
        "removed_by_modality_and_class": dict(sorted(removed_by_modality_class.items())),
    }
    return representatives, report


def split_class_records(records: list[ImageRecord], seed: int, ratios: dict[str, float]) -> dict[str, list[ImageRecord]]:
    if round(sum(ratios.values()), 8) != 1.0:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    shuffled = sorted(records, key=lambda item: item.relative_path.lower())
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    validation_count = round(count * ratios["validation"])
    test_count = round(count * ratios["test"])
    train_count = count - validation_count - test_count
    return {
        "train": shuffled[:train_count],
        "validation": shuffled[train_count : train_count + validation_count],
        "test": shuffled[train_count + validation_count :],
    }


def make_stratified_splits(records: Iterable[ImageRecord], seed: int, ratios: dict[str, float]) -> dict[str, dict[str, list[ImageRecord]]]:
    buckets: dict[str, dict[str, list[ImageRecord]]] = {
        modality: {split: [] for split in SPLITS} for modality in MODALITY_DIRECTORIES
    }
    grouped: dict[tuple[str, int], list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.modality, record.label)].append(record)

    for modality_index, modality in enumerate(MODALITY_DIRECTORIES):
        for label in sorted(CLASS_LABELS.values()):
            class_split = split_class_records(
                grouped[(modality, label)], seed + modality_index * 100 + label, ratios
            )
            for split in SPLITS:
                buckets[modality][split].extend(class_split[split])
                buckets[modality][split].sort(key=lambda item: item.relative_path.lower())

    assigned_hashes: set[str] = set()
    for modality in buckets:
        for split in SPLITS:
            for record in buckets[modality][split]:
                if record.sha256 in assigned_hashes:
                    raise RuntimeError(f"Duplicate hash assigned to more than one split: {record.sha256}")
                assigned_hashes.add(record.sha256)
    return buckets


def write_manifest(path: Path, records: Iterable[ImageRecord]) -> None:
    fields = ["source_path", "relative_path", "label", "class_name", "modality", "sha256", "duplicate_group_size"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "source_path": str(record.source_path),
                    "relative_path": record.relative_path,
                    "label": record.label,
                    "class_name": record.class_name,
                    "modality": record.modality,
                    "sha256": record.sha256,
                    "duplicate_group_size": record.duplicate_group_size,
                }
            )


def build_processed_dataset(config_path: Path | None = None) -> dict:
    config = load_config(config_path)
    root = project_root()
    raw_root = root / config["paths"]["raw_dataset_dir"]
    processed_root = root / config["paths"]["processed_dir"]
    processed_root.mkdir(parents=True, exist_ok=True)

    raw_records = discover_records(raw_root)
    unique_records, duplicate_report = deduplicate(raw_records)
    splits = make_stratified_splits(unique_records, config["random_seed"], config["split"])

    split_summary: dict[str, dict[str, dict[str, int]]] = {}
    for modality, modality_splits in splits.items():
        modality_dir = processed_root / modality.lower()
        modality_dir.mkdir(parents=True, exist_ok=True)
        split_summary[modality] = {}
        for split, records in modality_splits.items():
            write_manifest(modality_dir / f"{split}.csv", records)
            labels = Counter(record.class_name for record in records)
            split_summary[modality][split] = {
                "total": len(records),
                "Healthy": labels["Healthy"],
                "Tumor": labels["Tumor"],
            }

    report = {**duplicate_report, "splits": split_summary}
    with (processed_root / "duplicate_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def print_report(report: dict) -> None:
    print(
        "Exact duplicate handling: "
        f"{report['exact_duplicate_groups']} groups; "
        f"{report['redundant_duplicate_files_removed']} redundant files excluded; "
        f"{report['unique_images']} unique images retained."
    )
    for modality, splits in report["splits"].items():
        for split, counts in splits.items():
            print(
                f"{modality:3} {split:10} total={counts['total']:4} "
                f"Healthy={counts['Healthy']:4} Tumor={counts['Tumor']:4}"
            )


def verify_processed_dataset(config_path: Path | None = None) -> dict:
    """Verify generated manifests and report their split/class composition."""
    config = load_config(config_path)
    processed_root = project_root() / config["paths"]["processed_dir"]
    with (processed_root / "duplicate_report.json").open(encoding="utf-8") as handle:
        report = json.load(handle)

    seen_hashes: set[str] = set()
    for modality in MODALITY_DIRECTORIES:
        for split in SPLITS:
            manifest = processed_root / modality.lower() / f"{split}.csv"
            with manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            hashes = {row["sha256"] for row in rows}
            if len(hashes) != len(rows) or seen_hashes.intersection(hashes):
                raise RuntimeError(f"Duplicate image hash detected across processed manifests: {manifest}")
            seen_hashes.update(hashes)
            counts = Counter(row["class_name"] for row in rows)
            expected = report["splits"][modality][split]
            if len(rows) != expected["total"] or counts["Healthy"] != expected["Healthy"] or counts["Tumor"] != expected["Tumor"]:
                raise RuntimeError(f"Manifest count does not match duplicate report: {manifest}")

    if len(seen_hashes) != report["unique_images"]:
        raise RuntimeError("Processed unique-image count does not match duplicate report")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create duplicate-safe CT and MRI split manifests.")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument("--verify", action="store_true", help="Verify existing manifests without rebuilding them")
    args = parser.parse_args()
    print_report(verify_processed_dataset(args.config) if args.verify else build_processed_dataset(args.config))
