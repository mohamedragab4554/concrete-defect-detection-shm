"""Train FPN-EfficientNet-B4 for 18-class multi-label defect segmentation (dacl10k).

Stability measures from the research notebook: clamped per-class pos_weight, BCE + Dice,
AdamW with linear warm-up then cosine decay, AMP enabled only after a warm-up, NaN/Inf
guards, gradient clipping, best-checkpoint + early stopping.

    python -m shm_defects.train_multiclass --config configs/fpn_effb4_dacl10k.yaml
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .data import DACL10K_CLASSES, LabelMeMultiLabelDataset
from .losses import BCEDiceLoss, estimate_pos_weight
from .metrics import multilabel_iou
from .models import build_fpn
from .transforms import multiclass_transforms


def lr_at(epoch: int, base_lr: float, warmup: int, total: int) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    t = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, threshold: float = 0.5):
    model.eval()
    inter = np.zeros(num_classes)
    union = np.zeros(num_classes)
    for images, masks in tqdm(loader, desc="val", leave=False):
        preds = (torch.sigmoid(model(images.to(device))) > threshold).float().cpu()
        inter += (preds * masks).sum(dim=(0, 2, 3)).numpy()
        union += ((preds + masks) > 0).float().sum(dim=(0, 2, 3)).numpy()
    return multilabel_iou(inter, union)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = cfg.get("classes", DACL10K_CLASSES)
    size = cfg.get("image_size", 384)

    train_ds = LabelMeMultiLabelDataset(cfg["train_img_dir"], cfg["train_ann_dir"],
                                        multiclass_transforms(size, True), classes)
    val_ds = LabelMeMultiLabelDataset(cfg["val_img_dir"], cfg["val_ann_dir"],
                                      multiclass_transforms(size, False), classes)
    pos_weight = estimate_pos_weight(DataLoader(train_ds, batch_size=2, shuffle=True), len(classes),
                                     clamp_max=cfg.get("pos_weight_max", 1e4))
    train_dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 4), shuffle=True, drop_last=True,
                          num_workers=cfg.get("num_workers", 0))
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = build_fpn(len(classes), cfg.get("encoder", "efficientnet-b4"), pretrained=True).to(device)
    criterion = BCEDiceLoss(pos_weight.to(device)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4),
                                  weight_decay=cfg.get("weight_decay", 1e-4))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    epochs, patience = cfg.get("epochs", 30), cfg.get("patience", 6)
    out = Path(cfg.get("output_dir", "runs/fpn"))
    out.mkdir(parents=True, exist_ok=True)

    best, stale, log = 0.0, 0, []
    for epoch in range(epochs):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(epoch, cfg.get("lr", 3e-4), cfg.get("warmup_epochs", 2), epochs)
        use_amp = device.type == "cuda" and epoch >= cfg.get("amp_warmup_epochs", 3)
        model.train()
        running = 0.0
        for images, masks in tqdm(train_dl, desc=f"epoch {epoch + 1}/{epochs}", leave=False):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)
            if not torch.isfinite(loss):
                continue  # skip unstable batches instead of corrupting the weights
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip_norm", 1.0))
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        miou, per_class = evaluate(model, val_dl, device, len(classes))
        log.append({"epoch": epoch + 1, "loss": running / max(1, len(train_dl)), "val_miou": miou})
        print(f"epoch {epoch + 1}: loss {log[-1]['loss']:.4f}  val mIoU {miou:.4f}")
        if miou > best:  # NOTE: checkpoint selection uses the validation split
            best, stale = miou, 0
            torch.save(model.state_dict(), out / "best.pth")
            per_class_iou = dict(zip(classes, map(float, per_class), strict=True))
            (out / "per_class_iou.json").write_text(json.dumps(per_class_iou, indent=2))
        else:
            stale += 1
            if stale >= patience:
                print("early stopping")
                break
    (out / "history.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
