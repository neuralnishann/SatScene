"""Model factory and checkpoint helpers."""
from __future__ import annotations

import torch
import timm


def build_model(num_classes: int, name: str = "convnext_tiny.fb_in22k_ft_in1k", pretrained: bool = True,
                img_size: int | None = None):
    """Create a pretrained feature extractor with a fresh scene-class head.

    For an image x, the backbone produces a feature vector h = f_theta(x).
    The replacement classifier maps it to K logits z = W h + b, where
    K=num_classes. Inference converts logits to class probabilities using
    softmax(z)_c = exp(z_c) / sum_j exp(z_j). ``timm.create_model`` handles the
    architecture-specific pooling and head construction; this factory does
    not implement alternate blocks or a custom mathematical approximation.
    See docs/MATHEMATICAL_METHODS.md for model-specific formulations.
    """
    kwargs = {}
    if img_size is not None and name.startswith("swinv2_"):
        # Four Swin stages reduce the input by 32. A window of 8 must divide
        # the final feature map, so window-8 models need a multiple of 256.
        if img_size % 256:
            raise ValueError(
                f"{name} requires img_size divisible by 256; got {img_size}. "
                "Use 256 for swinv2_tiny_window8_256."
            )
        kwargs["img_size"] = img_size
    elif img_size is not None and name.startswith("vit_small_patch14_dinov2"):
        kwargs["img_size"] = img_size
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes, **kwargs)


def count_params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def load_checkpoint(model: torch.nn.Module, path: str, device) -> torch.nn.Module:
    state = torch.load(path, map_location=device, weights_only=True)
    # New checkpoints include metadata; accept the original raw state dict too.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model.to(device).eval()


def save_checkpoint(model: torch.nn.Module, path: str, class_names, model_name: str,
                    img_size: int) -> None:
    """Save portable weights together with the class order required at inference."""
    torch.save({
        "state_dict": model.state_dict(),
        "class_names": list(class_names),
        "model_name": model_name,
        "img_size": int(img_size),
    }, path)


def checkpoint_metadata(path: str) -> dict:
    """Read inference metadata from a checkpoint, or return empty for legacy files."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        return {k: obj[k] for k in ("class_names", "model_name", "img_size") if k in obj}
    return {}
