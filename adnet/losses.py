from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_loss(loss_cfg: dict[str, Any], class_weights: torch.Tensor | None = None) -> nn.Module:
    loss_type = loss_cfg.get("type", "cross_entropy")
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_type == "focal":
        gamma = float(loss_cfg.get("focal_gamma", 2.0))
        return FocalLoss(gamma=gamma, alpha=class_weights)
    raise ValueError(f"Unsupported loss type: {loss_type}")

