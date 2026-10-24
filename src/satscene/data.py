"""Torch dataset, Albumentations pipelines and DataLoaders."""
from __future__ import annotations

from typing import Sequence, Tuple

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int = 224, train: bool = False) -> A.Compose:
    """Flip/rot90 augmentation for training; resize + normalise only otherwise."""
    aug = []
    if train:
        aug += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)]
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


def make_loader(paths, labels, img_size, batch_size, train, num_workers=2) -> DataLoader:
    ds = SceneDataset(paths, labels, build_transforms(img_size, train=train))
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)
