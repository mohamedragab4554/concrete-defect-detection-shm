"""Evaluate a trained crack U-Net on a split: mean per-image IoU/Dice, pixel P/R/F1.

    python -m shm_defects.evaluate_unet --config configs/unet_resnet50.yaml --weights runs/unet/best.pth --split test
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .data import YoloPolygonMaskDataset
from .metrics import Counts, binary_image_scores
from .models import build_unet
from .transforms import unet_transforms


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5) -> dict:
    model.eval()
    ious, dices, accs, counts = [], [], [], Counts()
    for images, masks in tqdm(loader, desc="eval"):
        preds = torch.sigmoid(model(images.to(device))) > threshold
        for p, t in zip(preds.cpu().numpy(), masks.numpy(), strict=True):
            iou, dice, acc = binary_image_scores(p[0], t[0] > 0.5)
            ious.append(iou); dices.append(dice); accs.append(acc)
            counts.update(p[0], t[0] > 0.5)
    return {"images": len(ious), "mean_iou": float(np.mean(ious)), "mean_dice": float(np.mean(dices)),
            "pixel_accuracy": float(np.mean(accs)), "precision": counts.precision,
            "recall": counts.recall, "f1": counts.f1,
            "tp": counts.tp, "fp": counts.fp, "fn": counts.fn, "tn": counts.tn}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test", help="use a split that was NOT used for checkpoint selection")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = YoloPolygonMaskDataset(cfg["data_dir"], args.split, unet_transforms(cfg.get("image_size", 640), False))
    model = build_unet(cfg.get("encoder", "resnet50"), pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print(json.dumps(evaluate(model.to(device), DataLoader(ds, batch_size=1), device), indent=2))


if __name__ == "__main__":
    main()
