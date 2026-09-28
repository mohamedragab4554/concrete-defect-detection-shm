"""Train the binary crack U-Net (ResNet-50 encoder) on YOLO-seg polygon labels.

    python -m shm_defects.train_unet --config configs/unet_resnet50.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .data import YoloPolygonMaskDataset
from .losses import SoftDiceLoss
from .models import build_unet
from .transforms import unet_transforms


def run_epoch(model, loader, criterion, device, optimizer=None) -> float:
    train = optimizer is not None
    model.train(train)
    total = 0.0
    with torch.set_grad_enabled(train):
        for images, masks in tqdm(loader, desc="train" if train else "val", leave=False):
            images, masks = images.to(device), masks.to(device)
            loss = criterion(model(images), masks)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += loss.item()
    return total / max(1, len(loader))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size = cfg.get("image_size", 640)
    train_ds = YoloPolygonMaskDataset(cfg["data_dir"], cfg.get("train_split", "train"), unet_transforms(size, True))
    val_ds = YoloPolygonMaskDataset(cfg["data_dir"], cfg.get("val_split", "val"), unet_transforms(size, False))
    print(f"train images: {len(train_ds)} | val images: {len(val_ds)} | device: {device}")
    bs = cfg.get("batch_size", 4)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=cfg.get("num_workers", 0))
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=cfg.get("num_workers", 0))

    model = build_unet(cfg.get("encoder", "resnet50"), pretrained=True).to(device)
    criterion = SoftDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    out = Path(cfg.get("output_dir", "runs/unet"))
    out.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    for epoch in range(cfg.get("epochs", 50)):
        tr = run_epoch(model, train_dl, criterion, device, optimizer)
        va = run_epoch(model, val_dl, criterion, device)
        scheduler.step(va)
        history.append({"epoch": epoch + 1, "train_loss": tr, "val_loss": va})
        print(f"epoch {epoch + 1:3d}  train {tr:.4f}  val {va:.4f}")
        if va < best:  # NOTE: checkpoint selection uses the validation split
            best = va
            torch.save(model.state_dict(), out / "best.pth")
    (out / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
