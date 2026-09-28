"""Model factories (segmentation-models-pytorch)."""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_unet(encoder: str = "resnet50", pretrained: bool = True) -> nn.Module:
    """Binary crack U-Net (1 output channel, logits)."""
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=1, activation=None)


def build_fpn(num_classes: int = 18, encoder: str = "efficientnet-b4", pretrained: bool = True) -> nn.Module:
    """Multi-label FPN (one sigmoid channel per defect class, no background channel)."""
    return smp.FPN(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                   in_channels=3, classes=num_classes)
