# ADNET — Dual-Stream CNN for Five-Stage Alzheimer's Disease Staging

Reference implementation for the paper
**"ADNET: A Dual-Stream Convolutional Network for Five-Stage Alzheimer's Disease Classification with Integrated Grad-CAM Explainability."**

ADNET classifies a single axial brain-MRI slice into one of five stages — cognitively
normal (CN), early MCI (EMCI), MCI, late MCI (LMCI), and Alzheimer's disease (AD) — by
routing it through parallel ImageNet-pretrained **ResNet-50** and **EfficientNet-B0**
backbones, concatenating their globally pooled features (2048-d + 1280-d = 3328-d), and
classifying the fused vector with a shared head. Grad-CAM on the EfficientNet-B0 stream
produces class-discriminative saliency maps.

## Repository layout

```
adnet/                 core package
  config.py            config loading / merging (base + per-experiment overrides)
  data.py              dataset, subject-/slice-level splitting, class weights, augmentation
  dual_backbone.py     dual-stream ResNet-50 + EfficientNet-B0 model
  models.py            model factory and Grad-CAM target-layer selection
  losses.py            class-weighted cross-entropy / focal loss
  metrics.py           accuracy, weighted P/R/F1, macro ROC-AUC, confusion matrix
  train_eval.py        training / evaluation / prediction loops
  gradcam.py           Grad-CAM implementation
  utils.py             run-directory, logging, and seeding helpers
scripts/
  run.py               train / evaluate / predict / gradcam for a single config
  sweep.py             run several configs in sequence
  aggregate_results.py collate per-run metrics into comparison tables
  gradcam_panels.py    render Grad-CAM figure panels
  make_adnet_diagram.py architecture diagram
  smoke_test.py        fast end-to-end sanity check
configs/               base config plus ADNET, single-stream baselines, and ablations
RUNBOOK.md             full command reference
```

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     |     Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The MRI images are from the Alzheimer's Disease Neuroimaging Initiative (ADNI,
<https://adni.loni.usc.edu>) and are **not** included in this repository: the ADNI Data
Use Agreement prohibits redistribution. Obtain access by registering with the Laboratory
of Neuro Imaging (LONI) and executing the DUA, then arrange the slices as:

```
data/Alzheimers-ADNI/
  train/{Final AD JPEG, Final CN JPEG, Final EMCI JPEG, Final LMCI JPEG, Final MCI JPEG}/
  test/  (same five class folders)
```

Train/validation splits are produced deterministically from a fixed seed
(`configs/base.yaml`, `adnet/data.py`); no images are required to read the code.

## Usage

```bash
# Train the dual-stream model and the single-stream baselines
python scripts/run.py --mode train --config configs/adnet_dualstream.yaml
python scripts/run.py --mode train --config configs/baseline_resnet50.yaml
python scripts/run.py --mode train --config configs/baseline_efficientnetb0.yaml

# Grad-CAM panels from a trained checkpoint
python scripts/run.py --mode gradcam --config configs/adnet_dualstream.yaml \
    --checkpoint outputs/<run_id>/best_model.pt

# Collate results into comparison tables
python scripts/aggregate_results.py --outputs-dir outputs
```

See [`RUNBOOK.md`](RUNBOOK.md) for the complete command list, including the ablation configs.

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). The ADNI imaging data are
governed separately by the ADNI Data Use Agreement.
