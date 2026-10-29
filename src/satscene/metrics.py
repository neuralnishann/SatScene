"""Small, dependency-free statistics helpers."""
from __future__ import annotations

import math
from typing import Tuple


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95 %).

    With observed accuracy ``p_hat = k / n``, this inverts the score test. Let
    ``d = 1 + z²/n``. The centre is
    ``(p_hat + z²/(2n)) / d`` and the half-width is
    ``z * sqrt(p_hat*(1-p_hat)/n + z²/(4n²)) / d``.
    Thus the interval is centre ± half-width. ``z=1.96`` gives the usual
    approximate 95% interval. It quantifies binomial sampling uncertainty,
    not variation from retraining; see docs/MATHEMATICAL_METHODS.md.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def calibration_summary(probabilities, targets, bins: int = 10) -> dict:
    """Return top-label expected calibration error and reliability-bin values.

    ``probabilities`` must have shape (samples, classes); bins use equal-width
    confidence intervals. Empty bins are omitted from the returned curve.

    For each example i, ``c_i = max_j p_ij`` and
    ``a_i = 1[argmax_j p_ij = y_i]``. For equal-width confidence bins B_m,
    ECE = sum_m (|B_m| / N) * |mean(a_i in B_m) - mean(c_i in B_m)|.
    ECE depends on the binning convention and is a descriptive statistic, not
    a proper scoring rule. This implementation uses ten equal-width bins by
    default, with the upper endpoint included in the final bin.
    """
    import numpy as np

    probs = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets, dtype=int)
    if probs.ndim != 2 or len(probs) != len(targets):
        raise ValueError("probabilities must be (n_samples, n_classes) and match targets")
    if len(targets) == 0:
        return {"ece": 0.0, "bins": []}
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == targets
    edges = np.linspace(0.0, 1.0, bins + 1)
    curve, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence >= lo) & ((confidence <= hi) if hi == 1 else (confidence < hi))
        if in_bin.any():
            mean_confidence = float(confidence[in_bin].mean())
            accuracy = float(correct[in_bin].mean())
            count = int(in_bin.sum())
            ece += count / len(targets) * abs(accuracy - mean_confidence)
            curve.append({"confidence": mean_confidence, "accuracy": accuracy, "count": count})
    return {"ece": float(ece), "bins": curve}


def multiclass_brier_score(probabilities, targets, num_classes: int) -> float:
    """Mean multiclass Brier score (lower is better).

    Let ``Y_ic = 1[y_i = c]`` be the one-hot target. This function returns
    ``(1/N) * sum_i sum_c (p_ic - Y_ic)^2``. It is a strictly proper
    probability score for the multiclass outcome; the unnormalized sum over
    classes has range [0, 2]. See docs/MATHEMATICAL_METHODS.md.
    """
    import numpy as np

    probs = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets, dtype=int)
    if len(targets) == 0:
        return 0.0
    one_hot = np.eye(num_classes, dtype=float)[targets]
    return float(np.square(probs - one_hot).sum(axis=1).mean())
