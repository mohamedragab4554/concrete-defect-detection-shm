"""Losses: soft Dice and class-balanced BCE + Dice for multi-label segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """1 - mean per-channel soft Dice over (N, C, H, W) logits."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        num = 2.0 * (probs * targets).sum(dim=(0, 2, 3))
        den = (probs + targets).sum(dim=(0, 2, 3)).clamp_min(self.eps)
        return 1.0 - (num / den).mean()


class BCEDiceLoss(nn.Module):
    """BCEWithLogits (per-class ``pos_weight``) + soft Dice."""

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("pos_weight", None if pos_weight is None else pos_weight.view(1, -1, 1, 1).float())
        self.dice = SoftDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(f"shape mismatch: logits {tuple(logits.shape)} vs targets {tuple(targets.shape)}")
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)
        return bce + self.dice(logits, targets)


@torch.no_grad()
def estimate_pos_weight(loader, num_classes: int, max_batches: int = 256, clamp_max: float = 1e4) -> torch.Tensor:
    """neg/pos pixel ratio per class, clamped so rare classes cannot destabilise BCE."""
    pos = torch.zeros(num_classes, dtype=torch.float64)
    tot = torch.zeros(num_classes, dtype=torch.float64)
    for i, (_, masks) in enumerate(loader):
        b, _, h, w = masks.shape
        pos += masks.sum(dim=(0, 2, 3)).double()
        tot += b * h * w
        if i + 1 >= max_batches:
            break
    pos = pos.clamp(min=1.0)
    return ((tot - pos).clamp(min=1.0) / pos).float().clamp(max=clamp_max)
