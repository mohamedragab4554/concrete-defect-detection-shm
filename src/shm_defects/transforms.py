"""Albumentations pipelines (as used in the dissertation experiments)."""
from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def unet_transforms(size: int = 640, train: bool = True) -> A.Compose:
    aug = [A.Resize(size, size)]
    if train:
        aug += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2)]
    return A.Compose(aug + [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


def multiclass_transforms(size: int = 384, train: bool = True) -> A.Compose:
    aug = [A.Resize(size, size)]
    if train:
        aug += [
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.2),
            A.Affine(translate_percent=0.02, scale=(0.9, 1.1), rotate=(-10, 10), p=0.3,
                     border_mode=cv2.BORDER_REFLECT),
            A.RandomBrightnessContrast(p=0.25),
        ]
    return A.Compose(aug + [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])
