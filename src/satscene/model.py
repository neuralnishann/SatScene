"""Model factory and checkpoint helpers."""
from __future__ import annotations

import torch
import timm


def build_model(num_classes: int, name: str = "convnext_tiny.fb_in22k_ft_in1k", pretrained: bool = True):
    """ConvNeXt-Tiny (ImageNet-22k pre-trained, 1k fine-tuned) with a fresh head."""
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def count_params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def load_checkpoint(model: torch.nn.Module, path: str, device) -> torch.nn.Module:
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()
