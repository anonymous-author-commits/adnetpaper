from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError as e:
    raise ImportError("Albumentations is required: pip install albumentations==1.4.0") from e


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
ADNI_SUBJECT_RE = re.compile(r"(ADNI_\d{3}_S_\d{4}|\d{3}_S_\d{4})", re.IGNORECASE)


@dataclass
class Sample:
    path: str
    label_idx: int
    label_name: str
    subject_id: str | None


class MRISliceDataset(Dataset):
    def __init__(self, samples: list[Sample], transform: A.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = cv2.imread(s.path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Unable to load image: {s.path}")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        t = self.transform(image=image)["image"]
        return t, s.label_idx


def parse_subject_id(file_path: str) -> str | None:
    stem = Path(file_path).stem
    m = ADNI_SUBJECT_RE.search(stem)
    if m:
        return m.group(1).upper()
    return None


def discover_samples(root_dir: str, split: str) -> tuple[list[Sample], dict[str, int]]:
    split_dir = Path(root_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")
    classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    samples: list[Sample] = []
    for cls in classes:
        cls_dir = split_dir / cls
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(IMG_EXTENSIONS):
                p = str(cls_dir / fname)
                samples.append(
                    Sample(
                        path=p,
                        label_idx=class_to_idx[cls],
                        label_name=cls,
                        subject_id=parse_subject_id(p),
                    )
                )
    if not samples:
        raise RuntimeError(f"No images found in {split_dir}")
    return samples, class_to_idx


def build_transform(
    *,
    img_size: int,
    split: str,
    augmentation: str,
    clahe: bool,
    mean: list[float],
    std: list[float],
) -> A.Compose:
    tfs: list[Any] = []
    if split == "train":
        if augmentation == "baseline":
            tfs += [
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                A.OneOf(
                    [
                        A.GaussNoise(p=0.5),
                        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.5),
                    ],
                    p=0.3,
                ),
            ]
        elif augmentation == "light":
            tfs += [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.3),
            ]
        elif augmentation != "none":
            raise ValueError(f"Unknown augmentation policy: {augmentation}")

    tfs.append(A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC))
    if clahe:
        tfs.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0))
    tfs += [A.Normalize(mean=mean, std=std), ToTensorV2()]
    return A.Compose(tfs)


def _split_by_subject(samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    rng = np.random.default_rng(seed)
    by_label_subjects: dict[int, dict[str, list[Sample]]] = defaultdict(lambda: defaultdict(list))
    for s in samples:
        by_label_subjects[s.label_idx][s.subject_id].append(s)

    train_out: list[Sample] = []
    val_out: list[Sample] = []
    for label_idx, subj_map in by_label_subjects.items():
        subjects = list(subj_map.keys())
        rng.shuffle(subjects)
        n_val = max(1, int(round(len(subjects) * val_ratio)))
        n_val = min(n_val, max(len(subjects) - 1, 1))
        val_subjects = set(subjects[:n_val])
        for sid, ss in subj_map.items():
            (val_out if sid in val_subjects else train_out).extend(ss)

        if not any(s.label_idx == label_idx for s in train_out):
            for i, s in enumerate(val_out):
                if s.label_idx == label_idx:
                    train_out.append(s)
                    del val_out[i]
                    break
    return train_out, val_out


def _split_slice_level(samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        by_label[s.label_idx].append(s)
    train_out: list[Sample] = []
    val_out: list[Sample] = []
    for _, ss in by_label.items():
        idx = np.arange(len(ss))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(ss) * val_ratio)))
        n_val = min(n_val, max(len(ss) - 1, 1))
        val_set = set(idx[:n_val].tolist())
        for i, s in enumerate(ss):
            (val_out if i in val_set else train_out).append(s)
    return train_out, val_out


def make_splits(
    samples_train_all: list[Sample],
    split_cfg: dict[str, Any],
    seed: int,
) -> tuple[list[Sample], list[Sample], dict[str, Any]]:
    strategy = split_cfg.get("strategy", "auto")
    val_ratio = float(split_cfg.get("val_ratio", 0.2))
    with_subject = [s for s in samples_train_all if s.subject_id]
    parse_rate = len(with_subject) / max(1, len(samples_train_all))

    leakage_risk = False
    used_strategy = strategy
    if strategy == "auto":
        strategy = "subject" if parse_rate >= 0.9 else "slice"
        used_strategy = strategy

    if strategy == "subject":
        has_subject_everywhere = parse_rate >= 0.9
        if has_subject_everywhere:
            train_samples, val_samples = _split_by_subject(samples_train_all, val_ratio, seed)
        else:
            train_samples, val_samples = _split_slice_level(samples_train_all, val_ratio, seed)
            leakage_risk = True
            used_strategy = "slice_fallback"
    elif strategy == "slice":
        train_samples, val_samples = _split_slice_level(samples_train_all, val_ratio, seed)
        leakage_risk = True
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    meta = {
        "split_strategy_requested": split_cfg.get("strategy", "auto"),
        "split_strategy_used": used_strategy,
        "subject_parse_rate": parse_rate,
        "leakage_risk": leakage_risk,
        "val_ratio": val_ratio,
    }
    return train_samples, val_samples, meta


def class_weights_from_samples(samples: list[Sample], num_classes: int) -> torch.Tensor:
    counts = Counter([s.label_idx for s in samples])
    total = sum(counts.values())
    weights = [total / (num_classes * max(1, counts.get(i, 0))) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def _count_by_class(samples: list[Sample], idx_to_class: dict[int, str]) -> dict[str, int]:
    c = Counter([s.label_idx for s in samples])
    return {idx_to_class[i]: int(c.get(i, 0)) for i in sorted(idx_to_class.keys())}


def run_dataset_sanity_checks(
    train_samples: list[Sample],
    val_samples: list[Sample],
    test_samples: list[Sample],
    transforms: dict[str, A.Compose],
    idx_to_class: dict[int, str],
    num_shape_checks: int = 8,
    num_tensor_checks: int = 8,
) -> dict[str, Any]:
    checks: dict[str, Any] = {"warnings": []}
    checks["class_counts"] = {
        "train": _count_by_class(train_samples, idx_to_class),
        "val": _count_by_class(val_samples, idx_to_class),
        "test": _count_by_class(test_samples, idx_to_class),
    }
    checks["split_sizes"] = {
        "train": len(train_samples),
        "val": len(val_samples),
        "test": len(test_samples),
    }
    if min(checks["split_sizes"].values()) <= 0:
        checks["warnings"].append("One or more splits are empty.")

    shape_issues: list[str] = []
    for split_name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        for s in samples[:num_shape_checks]:
            img = cv2.imread(s.path, cv2.IMREAD_GRAYSCALE)
            if img is None or img.ndim != 2 or min(img.shape) < 8:
                shape_issues.append(f"{split_name}:{s.path}")
    checks["shape_issues"] = shape_issues

    numeric_issues: list[str] = []
    for split_name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        tf = transforms[split_name]
        for s in samples[:num_tensor_checks]:
            img = cv2.imread(s.path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                numeric_issues.append(f"{split_name}:{s.path}:unreadable")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            t = tf(image=img)["image"]
            if not torch.isfinite(t).all().item():
                numeric_issues.append(f"{split_name}:{s.path}:non_finite")
    checks["numeric_issues"] = numeric_issues
    if shape_issues:
        checks["warnings"].append("Found problematic image shapes.")
    if numeric_issues:
        checks["warnings"].append("Found NaN/inf values after transforms.")
    return checks


def build_dataloaders_from_config(cfg: dict[str, Any]) -> tuple[dict[str, DataLoader], dict[str, int], dict[str, Any]]:
    data_cfg = cfg["data"]
    train_all, class_to_idx = discover_samples(data_cfg["root_dir"], "train")
    test_samples, _ = discover_samples(data_cfg["root_dir"], data_cfg.get("eval_split", "test"))
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    train_samples, val_samples, split_meta = make_splits(
        train_all,
        split_cfg=data_cfg.get("split", {}),
        seed=int(cfg["seed"]),
    )

    norm_cfg = data_cfg.get("normalization", {})
    mean = norm_cfg.get("mean", [0.5, 0.5, 0.5])
    std = norm_cfg.get("std", [0.5, 0.5, 0.5])
    clahe = bool(norm_cfg.get("clahe", True))
    aug = data_cfg.get("augmentation", "baseline")
    img_size = int(data_cfg.get("img_size", 224))

    tf_train = build_transform(img_size=img_size, split="train", augmentation=aug, clahe=clahe, mean=mean, std=std)
    tf_eval = build_transform(img_size=img_size, split="eval", augmentation="none", clahe=clahe, mean=mean, std=std)

    ds_train = MRISliceDataset(train_samples, tf_train)
    ds_val = MRISliceDataset(val_samples, tf_eval)
    ds_test = MRISliceDataset(test_samples, tf_eval)

    nw = int(data_cfg.get("num_workers", 0))
    bs = int(cfg["training"]["batch_size"])
    loaders = {
        "train": DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True),
        "val": DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True),
        "test": DataLoader(ds_test, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True),
    }

    sanity = run_dataset_sanity_checks(
        train_samples,
        val_samples,
        test_samples,
        transforms={"train": tf_train, "val": tf_eval, "test": tf_eval},
        idx_to_class=idx_to_class,
    )

    metadata = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "num_classes": len(class_to_idx),
        "split": split_meta,
        "sanity_checks": sanity,
        "train_samples": train_samples,
    }
    return loaders, class_to_idx, metadata
