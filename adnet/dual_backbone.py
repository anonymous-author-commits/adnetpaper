"""
dual_model_gradcam.py
======================
Defines a dual-backbone CNN for five-class ADNI MRI classification,
combining EfficientNet-B0 and ResNet-50 feature extractors via concatenation.
Includes a Grad-CAM implementation for post-hoc explainability on the fused model.

Usage:
    model = DualBackboneModel(num_classes=5, dropout_p=0.5)
    logits = model(images)  # images: Tensor (B,3,224,224)
    cam = GradCAM(model, target_layer="features")
    heatmaps = cam.generate(images, class_idx)

Dependencies:
    torch>=1.10, torchvision>=0.11, numpy, opencv-python
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DualBackboneModel(nn.Module):
    """
    Dual-stream CNN combining EfficientNet-B0 and ResNet-50.
    Extracts 1280-dim features from EfficientNet-B0 and 2048-dim from ResNet-50,
    concatenates them, and applies a classification head.
    """
    def __init__(self, num_classes: int = 5, dropout_p: float = 0.5):
        super().__init__()
        # EfficientNet-B0
        eff = models.efficientnet_b0(weights="DEFAULT")
        eff.classifier = nn.Identity()  # drop original head
        self.eff_features = eff
        self.eff_dim = 1280

        # ResNet-50
        res = models.resnet50(weights="DEFAULT")
        res.fc = nn.Identity()  # drop original head
        self.res_features = res
        self.res_dim = 2048

        # Fusion head
        fused_dim = self.eff_dim + self.res_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(fused_dim, fused_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(fused_dim // 4, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,224,224)
        # EfficientNet: feature maps -> avg pool -> flatten
        eff_feats = self.eff_features(x)  # (B,1280)
        # ResNet: conv features -> avg pool -> flatten
        # ResNet expects (B,3,224,224)
        res_feats = self.res_features(x)  # (B,2048)
        # Concatenate
        fused = torch.cat([eff_feats, res_feats], dim=1)  # (B,3328)
        logits = self.classifier(fused)
        return logits

class GradCAM:
    """
    Simple Grad-CAM for models with a single target convolutional layer.
    Capture gradients and activations via hooks.
    """
    def __init__(self, model: nn.Module, target_layer: str = "features"):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        # Register hooks
        # Try EfficientNet B0 last conv: model.eff_features.features[-1]
        # And ResNet-50 last conv: model.res_features.layer4
        eff_block = dict(self.model.named_modules())[f"eff_features.features"]
        res_block = dict(self.model.named_modules())[f"res_features.layer4"]
        # We use fused classifier; apply grads on both streams separately
        eff_block.register_forward_hook(self._save_activation)
        eff_block.register_backward_hook(self._save_gradient)
        res_block.register_forward_hook(self._save_activation)
        res_block.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        # output: Tensor or tuple
        self.activations = output.detach()  # (B,C,H,W)

    def _save_gradient(self, module, grad_in, grad_out):
        # grad_out is tuple
        self.gradients = grad_out[0].detach()  # (B,C,H,W)

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> torch.Tensor:
        """
        Returns heatmap for the given class index.

        Args:
            input_tensor: (B,3,224,224) images
            class_idx: if None, use predicted class per image
        Returns:
            heatmap: (B,1,H,W) normalized [0,1]
        """
        # Forward
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        # Zero grads
        self.model.zero_grad()
        # Backward per-sample
        heatmaps = []
        for i in range(logits.shape[0]):
            score = logits[i, class_idx[i]]
            score.backward(retain_graph=True)
            # GAP over gradients
            grads = self.gradients[i]              # (C,H,W)
            acts = self.activations[i]             # (C,H,W)
            weights = grads.mean(dim=(1,2), keepdim=True)  # (C,1,1)
            cam = (weights * acts).sum(dim=0)      # (H,W)
            cam = F.relu(cam)
            # Normalize
            cam -= cam.min()
            cam /= (cam.max() + 1e-6)
            heatmaps.append(cam.unsqueeze(0))      # (1,H,W)
            # Clear gradients
            self.model.zero_grad()
        return torch.stack(heatmaps, dim=0)         # (B,1,H,W)

# Example usage (not executed at import)
if __name__ == "__main__":
    model = DualBackboneModel(num_classes=5, dropout_p=0.5).cuda()
    dummy = torch.randn(2,3,224,224).cuda()
    logits = model(dummy)
    preds = logits.argmax(dim=1)
    cam = GradCAM(model)
    heatmaps = cam.generate(dummy, class_idx=preds)
    print("Logits shape:", logits.shape)
    print("Heatmaps shape:", heatmaps.shape)
