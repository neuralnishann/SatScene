"""Pure-Python label utilities (no torch / cv2 import, so they are cheap to test).

The Kaggle dataset ``umeradnaan/remote-sensing-satellite-images`` is a Roboflow
export in YOLOv8 *detection* format::

    <root>/data.yaml
    <root>/{train,valid,test}/images/*.jpg
    <root>/{train,valid,test}/labels/*.txt   # one "class cx cy w h" row per box

We turn it into a single-label *scene classification* problem by assigning each
image the class id that occurs most often in its label file (ties are broken in
favour of the class that appears first in the file).  This is an explicit,
documented assumption -- see README, "Known limitations".
"""
from __future__ import annotations

import glob
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def read_class_names(yaml_path: str) -> List[str]:
    """Read the ordered class list from a YOLO ``data.yaml``.

    Supports both the list form (``names: [a, b]``) and the dict form
    (``names: {0: a, 1: b}``).
    """
    import yaml  # local import keeps module import light

    with open(yaml_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names)


def parse_yolo_label(label_path: str) -> Optional[int]:
    """Return the dominant class id in a YOLO label file, or ``None`` if empty."""
    if not os.path.isfile(label_path):
        return None
    ids: List[int] = []
    with open(label_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if parts:
                ids.append(int(float(parts[0])))
    if not ids:
        return None
    counts = Counter(ids)
    best = max(counts.values())
    for cid in ids:  # first-appearing class among the most frequent
        if counts[cid] == best:
            return cid
    return None  # pragma: no cover


@dataclass
class SplitIndex:
    """Image paths and integer labels of one split, plus bookkeeping."""

    paths: List[str]
    labels: List[int]
    skipped: List[str]  # images with a missing/empty label file


def collect_split(root: str, split: str) -> SplitIndex:
    """Collect ``(image_path, class_id)`` pairs for ``split`` under ``root``."""
    img_dir = os.path.join(root, split, "images")
    lbl_dir = os.path.join(root, split, "labels")
    paths: List[str] = []
    labels: List[int] = []
    skipped: List[str] = []
    for p in sorted(glob.glob(os.path.join(img_dir, "*.*"))):
        if not p.lower().endswith(IMAGE_EXTS):
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        cid = parse_yolo_label(os.path.join(lbl_dir, stem + ".txt"))
        if cid is None:
            skipped.append(p)
            continue
        paths.append(p)
        labels.append(cid)
    return SplitIndex(paths, labels, skipped)


def class_distribution(labels: Sequence[int], class_names: Sequence[str]) -> Dict[str, int]:
    """Counts per class name, in ``class_names`` order (zeros included)."""
    counts = Counter(labels)
    return {name: counts.get(i, 0) for i, name in enumerate(class_names)}


def inverse_frequency_weights(labels: Sequence[int], num_classes: int) -> List[float]:
    """Optional class weights ``N / (K * n_c)`` for a weighted loss."""
    counts = Counter(labels)
    n = len(labels)
    return [n / (num_classes * counts[c]) if counts.get(c) else 0.0 for c in range(num_classes)]
