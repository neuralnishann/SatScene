"""Runtime device selection shared by the CLI and local GUI."""
from __future__ import annotations

import torch


def best_device() -> torch.device:
    """Prefer CUDA, then Apple Silicon MPS, and finally the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
