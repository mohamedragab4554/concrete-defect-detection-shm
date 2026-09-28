"""Evaluation metrics.

Binary segmentation metrics follow the dissertation notebook:
* per-image IoU / Dice / pixel accuracy, averaged over images ("mean IoU"),
* pixel-level precision / recall / F1 from confusion counts accumulated over the split.
Multi-label IoU is dataset-level per class (intersection and union summed over the split).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        pred, true = pred.astype(bool), true.astype(bool)
        self.tp += int(np.sum(pred & true))
        self.fp += int(np.sum(pred & ~true))
        self.fn += int(np.sum(~pred & true))
        self.tn += int(np.sum(~pred & ~true))

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        n = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / n if n else 0.0


def binary_image_scores(pred: np.ndarray, true: np.ndarray, eps: float = 1e-7) -> tuple[float, float, float]:
    """IoU, Dice and pixel accuracy for one image. Empty-vs-empty scores ~0 IoU (as in the notebook)."""
    pred, true = pred.astype(bool), true.astype(bool)
    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    iou = inter / (union + eps)
    dice = 2 * inter / (pred.sum() + true.sum() + eps)
    acc = (pred == true).sum() / pred.size
    return float(iou), float(dice), float(acc)


def multilabel_iou(inter: np.ndarray, union: np.ndarray) -> tuple[float, np.ndarray]:
    """Per-class IoU from accumulated intersection/union; returns (mIoU over all classes, per-class)."""
    iou = inter / np.clip(union, 1e-6, None)
    return float(np.mean(iou)), iou
