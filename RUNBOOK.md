# ADNET Runbook

## 1) Environment setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install torch torchvision albumentations==1.4.0 opencv-python scikit-learn pyyaml matplotlib
```

## 2) Expected dataset layout

The code expects:

```text
data/Alzheimers-ADNI/
  train/
    Final AD JPEG/
    Final CN JPEG/
    Final EMCI JPEG/
    Final LMCI JPEG/
    Final MCI JPEG/
  test/
    Final AD JPEG/
    Final CN JPEG/
    Final EMCI JPEG/
    Final LMCI JPEG/
    Final MCI JPEG/
```

Notes:
- Class labels are inferred from folder names.
- Subject-level split is attempted for train/val if filenames contain parseable IDs (for ADNI-style names).
- If subject IDs cannot be parsed reliably, code falls back to slice-level split and logs `leakage_risk=true`.

## 3) Dry-run (sanity)

```bash
python scripts/run.py --mode dry-run --config configs/adnet_dualstream.yaml
```

## 4) Train ADNET dual-stream

```bash
python scripts/run.py --mode train --config configs/adnet_dualstream.yaml
```

## 5) Run baselines

```bash
python scripts/run.py --mode train --config configs/baseline_resnet50.yaml
python scripts/run.py --mode train --config configs/baseline_efficientnetb0.yaml
```

## 6) Run ablations

```bash
python scripts/run.py --mode train --config configs/ablation_no_class_weights.yaml
python scripts/run.py --mode train --config configs/ablation_focal_loss.yaml
python scripts/run.py --mode train --config configs/ablation_no_aug.yaml
python scripts/run.py --mode train --config configs/ablation_light_aug.yaml
```

## 7) Sweep + aggregation

```bash
python scripts/sweep.py --mode train --configs ^
  configs/adnet_dualstream.yaml ^
  configs/baseline_resnet50.yaml ^
  configs/baseline_efficientnetb0.yaml ^
  configs/ablation_no_class_weights.yaml ^
  configs/ablation_focal_loss.yaml ^
  configs/ablation_no_aug.yaml ^
  configs/ablation_light_aug.yaml
```

Or aggregate only:

```bash
python scripts/aggregate_results.py --outputs-dir outputs
```

## 8) Grad-CAM panels

Use a trained run checkpoint (typically `outputs/<run_id>/best_model.pt`):

```bash
python scripts/run.py --mode gradcam --config configs/adnet_dualstream.yaml --checkpoint outputs/<run_id>/best_model.pt --k 8
```

## 9) Smoke tests

```bash
python scripts/smoke_test.py
```

## 10) Output locations

Each run writes to:

```text
outputs/<run_id>/
  config_resolved.yaml
  metadata.json
  logs.txt
  metrics.json
  confusion_matrix.csv
  per_class_metrics.csv
  best_model.pt            (train mode)
  predictions.csv          (predict mode)
  figures/                 (gradcam mode)
```

Aggregated tables:

```text
outputs/tables/
  architecture_comparison.csv
  imbalance_ablation.csv
  augmentation_ablation.csv
  *.tex
```

