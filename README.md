# Brain Tumor Minor Project

This prototype currently prepares reproducible CT and MRI datasets only. It does not define, train, evaluate, or serve a neural network.

## Raw-data safety

The original dataset remains in `dataset/` and is never moved, renamed, copied, deleted, or modified. `data/raw/` is reserved for a future explicitly-authorized raw-data location; it is intentionally empty. Processed CSV manifests in `data/processed/` point back to the immutable original files.

## Prepare duplicate-safe splits

Run:

```powershell
python -m src.preprocessing
```

The command hashes every original image with SHA-256, keeps one deterministic representative of each exact duplicate group, rejects any duplicate group that spans a modality or class label, and creates separate stratified 70/15/15 CT and MRI manifests. The fixed seed is in `config.yaml`.

Verify existing manifests without rebuilding them:

```powershell
python -m src.preprocessing --verify
```

## Loading data later

`src.dataset.BrainTumorDataset` returns `(image_tensor, class_label, image_path, modality)`, where Healthy is `0` and Tumor is `1`. Images are converted to RGB, resized to 224×224, and normalized. Training transforms add only modest augmentation; validation and test transforms are deterministic.

MRI class imbalance is supported by `class_weights()` for weighted loss and `weighted_sampler()` for optional balanced sampling. The initial configuration selects weighted loss and leaves sampling disabled.

## Baseline CNN

The non-pretrained `BaselineCNN` is available in `src/model.py`. Run only a short smoke test with:

```powershell
python -m src.train --modality CT --smoke-test
```

Use `--modality MRI` for the separate MRI experiment. Full training is deliberately opt-in through `--train` and saves the best validation-loss checkpoint, history JSON, and metric plots.
