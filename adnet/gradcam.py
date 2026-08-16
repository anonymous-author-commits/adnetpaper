from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_module: torch.nn.Module):
        self.model = model
        self.target_module = target_module
        self.activations = None
        self.gradients = None
        self.target_module.register_forward_hook(self._save_activation)
        self.target_module.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, outp):
        self.activations = outp

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, x: torch.Tensor, class_idx: list[int]) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        one_hot = torch.zeros_like(logits)
        for i, c in enumerate(class_idx):
            one_hot[i, c] = 1.0
        logits.backward(gradient=one_hot, retain_graph=False)

        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1))
        n = cam.shape[0]
        cam = cam.view(n, -1)
        cam_min = cam.min(dim=1, keepdim=True).values
        cam_max = cam.max(dim=1, keepdim=True).values
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)
        return cam.view(n, acts.shape[2], acts.shape[3]).detach().cpu().numpy()

