"""Rebuild every chart in docs/images from the CSV/JSON files in results/.

    python scripts/make_figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "docs" / "images"
BG, PANEL, GRID, INK, MUTED = "#0b1220", "#101a2b", "#22324a", "#e6edf5", "#8aa0b8"
TEAL, AMBER, CORAL, BLUE = "#199e70", "#c98500", "#d95926", "#3987e5"  # validated categorical palette (dark surface)

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID, "axes.grid": True,
    "grid.linewidth": 0.6, "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.titleweight": "bold", "axes.axisbelow": True, "axes.titlesize": 12.5,
    "axes.spines.top": False, "axes.spines.right": False,
})


def rows(name: str) -> list[dict]:
    with open(RES / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def note(fig, text: str) -> None:
    fig.text(0.01, 0.015, text, color=MUTED, fontsize=8.5, ha="left")


def yolo_comparison() -> None:
    r = rows("crack_seg_yolo.csv")
    names = [x["model"] for x in r]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    w, xs = 0.2, range(len(r))
    series = [("box_map50", "Box mAP50", TEAL), ("mask_map50", "Mask mAP50", BLUE),
              ("box_map50_95", "Box mAP50-95", AMBER), ("mask_map50_95", "Mask mAP50-95", CORAL)]
    for k, (key, label, col) in enumerate(series):
        vals = [float(x[key]) for x in r]
        bars = ax.bar([i + (k - 1.5) * w for i in xs], vals, w, label=label, color=col, edgecolor=BG, linewidth=1.5)
        ax.bar_label(bars, fmt="%.3f", fontsize=7.5, color=INK, padding=2)
    ax.set_xticks(list(xs), names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Crack-Seg: YOLO-seg variants (200 images, 249 instances)", loc="left")
    ax.legend(ncol=4, frameon=False, loc="upper right", fontsize=8.5)
    note(fig, "Source: results/crack_seg_yolo.csv (dissertation Table 6). "
              "Box scores >> mask scores: thin-crack masks are hard.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "yolo_comparison.png", dpi=150)
    plt.close(fig)


def field_validation() -> None:
    r = rows("field_validation.csv")
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    metrics = []
    for x in r:
        tp, fn, tn, fp = (int(x[k]) for k in ("tp", "fn", "tn", "fp"))
        metrics.append((x["model"], (tp + tn) / (tp + tn + fp + fn), tp / (tp + fp), tp / (tp + fn), tn / (tn + fp)))
    w = 0.2
    for k, (label, col) in enumerate([("Accuracy", TEAL), ("Precision", BLUE), ("Recall", AMBER),
                                       ("Specificity", CORAL)]):
        vals = [m[k + 1] for m in metrics]
        xs_k = [i + (k - 1.5) * w for i in range(len(metrics))]
        bars = ax.bar(xs_k, vals, w, label=label, color=col, edgecolor=BG, linewidth=1.5)
        ax.bar_label(bars, fmt="%.2f", fontsize=7.5, color=INK, padding=2)
    ax.set_xticks(range(len(metrics)), [m[0] for m in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_title("Field validation: 44 real inspection images, manually verified", loc="left")
    ax.legend(ncol=4, frameon=False, loc="upper center", fontsize=8.5)
    note(fig, "Image-level confusion counts in results/field_validation.csv. High recall, low specificity: "
              "textured surfaces are read as cracks.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "field_validation.png", dpi=150)
    plt.close(fig)


def dacl10k_per_class() -> None:
    r = [x for x in rows("dacl10k_fpn_effb4_per_class.csv") if x["class"] != "mIoU"]
    miou = float(next(x["IoU"] for x in rows("dacl10k_fpn_effb4_per_class.csv") if x["class"] == "mIoU"))
    r.sort(key=lambda x: float(x["IoU"]))
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    iou = [float(x["IoU"]) for x in r]
    cols = [CORAL if v < 0.2 else (AMBER if v < 0.4 else TEAL) for v in iou]
    ax.barh([x["class"] for x in r], iou, color=cols, height=0.7)
    for i, v in enumerate(iou):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.axvline(miou, color=INK, lw=1.2, ls="--")
    ax.text(miou + 0.005, len(r) - 0.4, f"mIoU {miou:.3f}", fontsize=9)
    ax.axvline(0.414, color=MUTED, lw=1, ls=":")
    ax.text(0.419, 0.2, "best mIoU reported in\ndacl10k paper: 0.414", fontsize=8, color=MUTED)
    ax.set_xlim(0, 0.72)
    ax.set_xlabel("IoU (validation split)")
    ax.set_title("dacl10k, 18 classes: FPN-EfficientNet-B4 per-class IoU", loc="left")
    note(fig, "Source: results/dacl10k_fpn_effb4_per_class.csv. Thin/rare classes (Crack, Rockpocket, "
              "Climbingplants) are weakest.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "dacl10k_per_class_iou.png", dpi=150)
    plt.close(fig)


def training_curves() -> None:
    r = rows("yolo11x_seg_training_log.csv")
    key = {k.strip(): k for k in r[0]}
    ep = [int(x[key["epoch"]]) for x in r]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    curves = [("metrics/mAP50(B)", "box mAP50", TEAL), ("metrics/mAP50(M)", "mask mAP50", BLUE),
              ("metrics/mAP50-95(B)", "box mAP50-95", AMBER), ("metrics/mAP50-95(M)", "mask mAP50-95", CORAL)]
    for col, lab, c in curves:
        a.plot(ep, [float(x[key[col]]) for x in r], label=lab, color=c, lw=1.8)
    a.set_title("YOLO11x-seg: validation metrics per epoch", loc="left")
    a.set_xlabel("epoch"); a.set_ylim(0, 1); a.legend(frameon=False, fontsize=8)
    for col, lab, c in [("train/seg_loss", "train seg loss", TEAL), ("val/seg_loss", "val seg loss", AMBER),
                        ("train/box_loss", "train box loss", BLUE), ("val/box_loss", "val box loss", CORAL)]:
        b.plot(ep, [float(x[key[col]]) for x in r], label=lab, color=c, lw=1.8)
    b.set_title("Losses", loc="left"); b.set_xlabel("epoch"); b.legend(frameon=False, fontsize=8)
    hours = float(r[-1][key["time"]]) / 3600
    note(fig, f"Source: results/yolo11x_seg_training_log.csv: 50 epochs, {hours:.1f} h on one RTX 4060 (8 GB), "
              "batch 4, 640 px.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "yolo11x_training_curves.png", dpi=150)
    plt.close(fig)


def unet_summary() -> None:
    u = json.loads((RES / "unet_resnet50_crack_seg.json").read_text())
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    labels = ["mean IoU", "mean Dice", "pixel precision", "pixel recall", "pixel F1"]
    vals = [u["mean_iou"], u["mean_dice"], u["pixel_precision"], u["pixel_recall"], u["pixel_f1"]]
    bars = ax.barh(labels[::-1], vals[::-1], color=[TEAL, BLUE, AMBER, BLUE, TEAL][::-1], height=0.6)
    ax.bar_label(bars, fmt="%.3f", fontsize=8.5, color=INK, padding=3)
    ax.set_xlim(0, 1)
    ax.set_title("U-Net ResNet-50 crack segmentation (208-image validation split)", loc="left")
    note(fig, "Source: results/unet_resnet50_crack_seg.json (notebook outputs). Same split selected the checkpoint.")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "unet_metrics.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    yolo_comparison(); field_validation(); dacl10k_per_class(); training_curves(); unet_summary()
    print("figures written to", OUT)
