"""Torch dataset, Albumentations pipelines and DataLoaders."""
from __future__ import annotations

import os
import random
from typing import Sequence, Tuple

# This package update check makes an optional network request during import.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int = 224, train: bool = False, augmentation: dict | None = None) -> A.Compose:
    """Flip/rot90 augmentation for training; resize + normalise only otherwise."""
    augmentation = augmentation or {}
    aug = []
    if train:
        aug += [
            A.HorizontalFlip(p=float(augmentation.get("horizontal_flip_p", 0.5))),
            A.VerticalFlip(p=float(augmentation.get("vertical_flip_p", 0.5))),
            A.RandomRotate90(p=float(augmentation.get("rotate90_p", 0.5))),
        ]
        if augmentation.get("color_jitter_p", 0.0):
            aug.append(A.ColorJitter(p=float(augmentation["color_jitter_p"])))
        if augmentation.get("blur_p", 0.0):
            aug.append(A.GaussianBlur(p=float(augmentation["blur_p"])))
    aug += [A.Resize(img_size, img_size), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(aug)


def read_rgb(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class SceneDataset(Dataset):
    """Single-label scene classification dataset: ``(image_tensor, class_id)``."""

    def __init__(self, paths: Sequence[str], labels: Sequence[int], transforms: A.Compose):
        assert len(paths) == len(labels), "paths/labels length mismatch"
        self.paths, self.labels, self.transforms = list(paths), list(labels), transforms

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple:
        image = self.transforms(image=read_rgb(self.paths[idx]))["image"]
        return image, self.labels[idx]


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    __import__("numpy").random.seed(worker_seed)


def make_loader(
    paths,
    labels,
    img_size,
    batch_size,
    train,
    num_workers=2,
    pin_memory=None,
    augmentation: dict | None = None,
    balanced_sampling: bool = False,
    seed: int | None = None,
) -> DataLoader:
    if pin_memory is None:
        # Pinned host memory speeds CUDA transfers; MPS does not support it.
        pin_memory = torch.cuda.is_available()
    ds = SceneDataset(paths, labels, build_transforms(img_size, train=train, augmentation=augmentation))
    if balanced_sampling and not train:
        raise ValueError("balanced_sampling is only valid for training loaders")
    sampler = None
    if balanced_sampling:
        counts = torch.bincount(torch.as_tensor(labels, dtype=torch.long))
        weights = [1.0 / counts[label].item() for label in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
