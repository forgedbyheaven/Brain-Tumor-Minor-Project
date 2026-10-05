# Brain Tumor Minor Project

This project focuses on preparing reproducible CT and MRI brain-tumor image datasets and providing a baseline CNN pipeline for classification.

## Project Overview

The project uses two imaging modalities:

* CT scans
* MRI scans

The dataset contains Healthy and Tumor images for each modality. The preprocessing pipeline is designed to preserve the original dataset while creating duplicate-safe, stratified train, validation, and test manifests.

## Dataset Structure

The original dataset is kept unchanged in:

```text
dataset/
├── Brain Tumor CT scan Images/
│   ├── Healthy/
│   └── Tumor/
│
└── Brain Tumor MRI images/
    ├── Healthy/
    └── Tumor/
```

The raw images are not uploaded to GitHub because of their size.

## Data Preparation

The preprocessing pipeline:

1. Reads the original CT and MRI images.
2. Computes SHA-256 hashes to identify exact duplicate images.
3. Keeps one deterministic representative from each exact duplicate group.
4. Rejects duplicate groups that span different modalities or class labels.
5. Creates stratified 70/15/15 train, validation, and test splits.
6. Generates processed CSV manifests pointing to the original image files.

Run preprocessing with:

```powershell
python -m src.preprocessing
```

To verify existing manifests without rebuilding them:

```powershell
python -m src.preprocessing --verify
```

## Dataset Loading

`src.dataset.BrainTumorDataset` loads the processed image manifests and returns:

```text
(image_tensor, class_label, image_path, modality)
```

Class labels are:

```text
Healthy = 0
Tumor   = 1
```

Images are:

* converted to RGB
* resized to 224 × 224 pixels
* normalized

Training data uses configurable, on-the-fly augmentation before resizing: optional horizontal flipping,
rotation up to ±10°, translation up to 5%, scaling from 95–105%, and mild brightness/contrast jitter.
This improves robustness to normal CT/MRI acquisition and positioning variation without modifying raw images.
Augmentation is applied only after the duplicate-safe 70/15/15 split and only to the training manifest;
validation and test data use deterministic resize, tensor conversion, and normalization. Settings are in
the root-level `config.yaml` under `augmentation` (set `horizontal_flip_probability` to `0` when laterality
is clinically important).

MRI class imbalance is handled through class weighting. Optional weighted sampling is also supported.

## Baseline CNN

The project includes a non-pretrained `BaselineCNN` model in:

```text
src/model.py
```

A short smoke test can be performed using:

```powershell
python -m src.train --modality CT --smoke-test
```

For MRI:

```powershell
python -m src.train --modality MRI --smoke-test
```

Full training is intentionally opt-in using:

```powershell
python -m src.train --modality CT --train
```

or:

```powershell
python -m src.train --modality MRI --train
```

The training pipeline saves the best validation-loss checkpoint, training history, and metric plots.

## Project Structure

```text
Brain-Tumor-Minor-Project/
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── config.yaml
│
├── data/
│   └── processed/
│
├── reports/
│
├── dataset/
│   └── Original CT and MRI images
│
├── README.md
├── requirements.txt
└── .gitignore
```

The `dataset/` directory is excluded from GitHub using `.gitignore`.

## Current Status

The preprocessing and dataset-loading pipeline has been tested successfully. Smoke tests have also been performed for the CT and MRI pipelines.

Full model training and evaluation are performed separately after the data-preparation stage.

## Reproducibility

The project uses a fixed random seed defined in the configuration file to make dataset splitting and experiments reproducible.
