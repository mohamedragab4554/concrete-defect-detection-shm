"""Predict a crack mask for one image with a trained U-Net and save mask + overlay.

    python -m shm_defects.predict --weights runs/unet/best.pth --image path/to/photo.jpg --out out/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .models import build_unet
from .transforms import unet_transforms


@torch.no_grad()
def predict_mask(model, image_rgb: np.ndarray, size: int = 640, threshold: float = 0.5) -> np.ndarray:
    tensor = unet_transforms(size, train=False)(image=image_rgb)["image"].unsqueeze(0)
    device = next(model.parameters()).device
    prob = torch.sigmoid(model.eval()(tensor.to(device)))[0, 0].cpu().numpy()
    prob = cv2.resize(prob, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return (prob > threshold).astype(np.uint8)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet(pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    mask = predict_mask(model.to(device), rgb, threshold=args.threshold)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    overlay = rgb.copy()
    overlay[mask > 0] = (0.45 * overlay[mask > 0] + 0.55 * np.array([255, 60, 60])).astype(np.uint8)
    stem = Path(args.image).stem
    cv2.imwrite(str(out / f"{stem}_mask.png"), mask * 255)
    cv2.imwrite(str(out / f"{stem}_overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"crack pixels: {int(mask.sum())} ({mask.mean():.2%} of image) -> {out}")


if __name__ == "__main__":
    main()
