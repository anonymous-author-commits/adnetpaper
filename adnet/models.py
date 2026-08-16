from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch.nn as nn
from torchvision import models

from .dual_backbone import DualBackboneModel


def _build_backbone_with_head(name: str, num_classes: int, dropout_p: float) -> nn.Module:
    if name == "resnet50":
        base = models.resnet50(weights="DEFAULT")
        in_features = base.fc.in_features
        base.fc = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(in_features, num_classes))
        return base
    if name == "efficientnet_b0":
        base = models.efficientnet_b0(weights="DEFAULT")
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(in_features, num_classes))
        return base
    raise ValueError(f"Unsupported model type: {name}")


def build_model(model_cfg: dict[str, Any], num_classes: int) -> nn.Module:
    model_type = model_cfg["type"]
    dropout_p = float(model_cfg.get("dropout_p", 0.1))
    if model_type == "adnet_dualstream":
        return DualBackboneModel(num_classes=num_classes, dropout_p=dropout_p)
    if model_type in {"resnet50", "efficientnet_b0"}:
        return _build_backbone_with_head(model_type, num_classes, dropout_p)
    raise ValueError(f"Unknown model type: {model_type}")


def get_model_target_layer(model: nn.Module, model_type: str):
    if model_type == "adnet_dualstream":
        return model.eff_features.features[-1]
    if model_type == "resnet50":
        return model.layer4[-1]
    if model_type == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"No target layer configured for {model_type}")

