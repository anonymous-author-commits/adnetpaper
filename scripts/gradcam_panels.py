#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adnet.data import build_dataloaders_from_config
from adnet.gradcam import GradCAM
from adnet.models import build_model, get_model_target_layer


def _denorm(img: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    m = np.array(mean).reshape(1, 1, 3)
    s = np.array(std).reshape(1, 1, 3)
    return np.clip((img * s + m), 0.0, 1.0)


def _save_contact_sheet(fig_paths: list[Path], out_path: Path) -> None:
    images = [plt.imread(str(p)) for p in fig_paths if p.exists()]
    if not images:
        return
    cols = min(3, len(images))
    rows = math.ceil(len(images) / cols)
    fig = plt.figure(figsize=(cols * 5, rows * 4))
    for i, im in enumerate(images, start=1):
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(im)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def generate_gradcam_panels(
    cfg: dict[str, Any],
    checkpoint_path: str,
    run_dir: Path,
    k: int,
    logger,
) -> dict[str, Any]:
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    loaders, class_to_idx, data_meta = build_dataloaders_from_config(cfg)
    data_meta.pop("train_samples", None)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    model = build_model(cfg["model"], num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    target_layer = get_model_target_layer(model, cfg["model"]["type"])
    cam = GradCAM(model, target_layer)

    split = cfg["evaluation"].get("split", "test")
    norm = cfg["data"].get("normalization", {})
    mean = norm.get("mean", [0.5, 0.5, 0.5])
    std = norm.get("std", [0.5, 0.5, 0.5])
    figures_dir = Path(run_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    correct, incorrect = [], []
    with torch.no_grad():
        for x, y in loaders[split]:
            x = x.to(device)
            probs = torch.softmax(model(x), dim=1)
            preds = probs.argmax(dim=1)
            for i in range(x.size(0)):
                rec = (x[i].detach().cpu(), int(y[i].item()), int(preds[i].item()))
                if rec[1] == rec[2] and len(correct) < k:
                    correct.append(rec)
                if rec[1] != rec[2] and len(incorrect) < k:
                    incorrect.append(rec)
            if len(correct) >= k and len(incorrect) >= k:
                break

    def render(records: list[tuple[torch.Tensor, int, int]], prefix: str) -> list[Path]:
        paths: list[Path] = []
        for i, rec in enumerate(records):
            xi, yi, pi = rec
            x_batch = xi.unsqueeze(0).to(device)
            heat = cam.generate(x_batch, [pi])[0]
            rgb = xi.permute(1, 2, 0).numpy()
            rgb = _denorm(rgb, mean=mean, std=std)

            fig = plt.figure(figsize=(8, 4))
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.imshow(rgb[..., 0], cmap="gray")
            ax1.axis("off")
            ax1.set_title("Original")
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.imshow(rgb[..., 0], cmap="gray")
            ax2.imshow(heat, cmap="jet", alpha=0.45)
            ax2.axis("off")
            ax2.set_title(f"GT: {idx_to_class[yi]} | Pred: {idx_to_class[pi]}")
            fig.tight_layout()
            out = figures_dir / f"{prefix}_{i:02d}.png"
            fig.savefig(out, dpi=300)
            plt.close(fig)
            paths.append(out)
        return paths

    c_paths = render(correct, "gradcam_correct")
    ic_paths = render(incorrect, "gradcam_incorrect")
    _save_contact_sheet(c_paths + ic_paths, figures_dir / "contact_sheet.png")
    logger.info("Saved Grad-CAM figures in %s", figures_dir)
    return {"gradcam": {"correct_count": len(c_paths), "incorrect_count": len(ic_paths)}, "data": data_meta}


def parse_args():
    p = argparse.ArgumentParser(description="Generate Grad-CAM comparison panels.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--k", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    from adnet.config import load_config
    from adnet.utils import setup_logging

    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logging(Path(args.run_dir))
    generate_gradcam_panels(cfg, args.checkpoint, Path(args.run_dir), args.k, logger)
