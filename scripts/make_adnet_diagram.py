#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def box(ax, xy, w, h, text, fc, ec="#1d2b44", fs=10):
    p = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.07",
        linewidth=1.8,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(p)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=ec)
    return p


def arrow(ax, p0, p1, c="#1d2b44"):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", lw=2, color=c))


def main():
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    b_in = box(ax, (0.8, 6.9), 2.2, 1.0, "Input MRI Slice\nB x 3 x 224 x 224", "#e6f2ff")
    b_pre = box(ax, (0.8, 5.5), 2.2, 1.0, "Resize + CLAHE\nNormalize", "#e6f2ff")
    b_aug = box(ax, (0.8, 4.1), 2.2, 1.0, "Train Augmentation\n(configurable)", "#e6f2ff")

    b_eff = box(ax, (4.1, 6.1), 3.0, 1.2, "EfficientNet-B0\nfeature dim: 1280", "#ece9ff")
    b_res = box(ax, (4.1, 3.7), 3.0, 1.2, "ResNet-50\nfeature dim: 2048", "#ece9ff")

    b_pool_e = box(ax, (8.0, 6.1), 2.2, 1.2, "GAP\nB x 1280", "#e8f7e8")
    b_pool_r = box(ax, (8.0, 3.7), 2.2, 1.2, "GAP\nB x 2048", "#e8f7e8")

    b_cat = box(ax, (11.1, 5.2), 3.2, 1.0, "Concatenate\nB x 3328", "#fff0e0")
    b_h1 = box(ax, (11.1, 4.0), 3.2, 0.9, "Dropout(p)\nLinear 3328 -> 832", "#fff0e0")
    b_h2 = box(ax, (11.1, 2.9), 3.2, 0.9, "ReLU + Dropout(p)\nLinear 832 -> 5", "#fff0e0")
    b_out = box(ax, (11.1, 1.7), 3.2, 0.9, "Logits B x 5\nSoftmax (eval)", "#ffe6d9")

    arrow(ax, (1.9, 6.9), (1.9, 6.5))
    arrow(ax, (1.9, 5.5), (1.9, 5.1))
    arrow(ax, (3.0, 4.6), (4.1, 6.7))
    arrow(ax, (3.0, 4.6), (4.1, 4.3))
    arrow(ax, (7.1, 6.7), (8.0, 6.7))
    arrow(ax, (7.1, 4.3), (8.0, 4.3))
    arrow(ax, (10.2, 6.7), (11.1, 5.9))
    arrow(ax, (10.2, 4.3), (11.1, 5.5))
    arrow(ax, (12.7, 5.2), (12.7, 4.9))
    arrow(ax, (12.7, 4.0), (12.7, 3.8))
    arrow(ax, (12.7, 2.9), (12.7, 2.6))

    ax.text(
        0.8,
        0.5,
        "ADNET dual-stream classifier (conservative implementation): single input routed into both pretrained backbones,\n"
        "feature fusion by concatenation (1280 + 2048), then lightweight MLP head for 5-way diagnosis.",
        fontsize=10,
        color="#1d2b44",
    )

    base = out_dir / "adnet_architecture_nature_ready"
    fig.savefig(str(base) + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".svg", bbox_inches="tight")
    print(f"Saved: {base}.png/.pdf/.svg")


if __name__ == "__main__":
    main()

