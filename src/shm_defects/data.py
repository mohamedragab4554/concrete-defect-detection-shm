"""Datasets.

* :class:`YoloPolygonMaskDataset` - binary crack masks rasterised from YOLO-seg polygon
  labels (used for the U-Net on the Ultralytics Crack-Seg dataset).
* :class:`LabelMeMultiLabelDataset` - 18-channel multi-label masks from LabelMe-style JSON
  polygons (used for FPN-EfficientNet-B4 on dacl10k).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# dacl10k class order used in training (18 classes, no background channel)
DACL10K_CLASSES: list[str] = [
    "Crack", "ACrack", "Wetspot", "Efflorescence", "Rust", "Rockpocket", "Hollowareas",
    "Climbingplants", "Weathering", "Restformwork", "Spalling", "Graffiti",
    "ExposedRebars", "Bearing", "EJoint", "Drainage", "PEquipment", "JTape",
]

# Raw label spellings seen in the JSON files -> canonical class names
LABEL_NORMALIZE: dict[str, str] = {
    "crack": "Crack", "acrack": "ACrack", "wetspot": "Wetspot", "efflorescence": "Efflorescence",
    "rust": "Rust", "rockpocket": "Rockpocket", "hollowareas": "Hollowareas",
    "hollowarea": "Hollowareas", "climbingplants": "Climbingplants",
    "climbingplant": "Climbingplants", "weathering": "Weathering",
    "restformwork": "Restformwork", "spalling": "Spalling", "graffiti": "Graffiti",
    "exposedrebars": "ExposedRebars", "exposed_rebars": "ExposedRebars", "bearing": "Bearing",
    "ejoint": "EJoint", "expansionjoint": "EJoint", "drainage": "Drainage",
    "pequipment": "PEquipment", "protectiveequipment": "PEquipment", "jtape": "JTape",
    "jointtape": "JTape",
}


def yolo_polygons_to_mask(label_path: str | Path, height: int, width: int) -> np.ndarray:
    """Rasterise a YOLO-seg label file (``cls x1 y1 x2 y2 ...`` normalised) to a {0,1} mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    label_path = Path(label_path)
    if not label_path.exists():
        return mask
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:  # class id + at least 3 points
            continue
        coords = np.asarray(parts[1:], dtype=np.float32)
        if coords.size % 2:
            coords = coords[:-1]
        coords = coords.reshape(-1, 2)
        coords[:, 0] *= width
        coords[:, 1] *= height
        cv2.fillPoly(mask, [np.round(coords).astype(np.int32)], 1)
    return mask


def labelme_shapes_to_multilabel(
    shapes: Sequence[dict], height: int, width: int, classes: Sequence[str] = DACL10K_CLASSES
) -> np.ndarray:
    """Return an ``(H, W, C)`` float32 multi-label mask; unknown labels are ignored."""
    name_to_idx = {n: i for i, n in enumerate(classes)}
    mask = np.zeros((height, width, len(classes)), dtype=np.float32)
    for shape in shapes:
        raw = (shape.get("label", "") or "").replace(" ", "").lower()
        name = LABEL_NORMALIZE.get(raw)
        if name is None or name not in name_to_idx:
            continue
        pts = shape.get("points") or []
        poly = [(int(x), int(y)) for x, y in pts]
        if len(poly) < 3:
            continue
        layer = Image.new("L", (width, height), 0)
        ImageDraw.Draw(layer).polygon(poly, outline=1, fill=1)
        ci = name_to_idx[name]
        mask[:, :, ci] = np.maximum(mask[:, :, ci], np.asarray(layer, dtype=np.float32))
    return mask


class YoloPolygonMaskDataset(Dataset):
    """``<root>/images/<split>/*.jpg`` + ``<root>/labels/<split>/*.txt`` -> (image, mask[1,H,W])."""

    def __init__(self, root: str | Path, split: str = "train", transform: Callable | None = None):
        self.root = Path(root)
        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        self.images = sorted(p for p in self.image_dir.glob("*") if p.suffix in IMAGE_EXTS)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = yolo_polygons_to_mask(self.label_dir / f"{img_path.stem}.txt", *image.shape[:2])
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        mask = torch.as_tensor(np.asarray(mask), dtype=torch.float32)
        return image, mask.unsqueeze(0)


class LabelMeMultiLabelDataset(Dataset):
    """Image folder + folder of LabelMe-style JSON files -> (image, mask[C,H,W])."""

    def __init__(self, img_dir: str | Path, ann_dir: str | Path, transform: Callable | None = None,
                 classes: Sequence[str] = DACL10K_CLASSES):
        self.img_dir, self.ann_dir = str(img_dir), str(ann_dir)
        self.files = sorted(f for f in os.listdir(self.ann_dir) if f.endswith(".json"))
        self.transform = transform
        self.classes = list(classes)

    def __len__(self) -> int:
        return len(self.files)

    def _find_image(self, ann_file: str, image_path: str | None) -> str:
        base = os.path.splitext(os.path.basename(image_path or ann_file))[0]
        for ext in IMAGE_EXTS:
            p = os.path.join(self.img_dir, base + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"No image found for {ann_file}")

    def __getitem__(self, idx: int):
        ann_file = self.files[idx]
        with open(os.path.join(self.ann_dir, ann_file), "r", encoding="utf-8") as f:
            data = json.load(f)
        rgb = cv2.cvtColor(cv2.imread(self._find_image(ann_file, data.get("imagePath"))), cv2.COLOR_BGR2RGB)
        h = int(data.get("imageHeight") or rgb.shape[0])
        w = int(data.get("imageWidth") or rgb.shape[1])
        mask = labelme_shapes_to_multilabel(data.get("shapes", []), h, w, self.classes)
        if self.transform is not None:
            out = self.transform(image=rgb, mask=mask)
            rgb, mask = out["image"], out["mask"]
        mask_t = torch.as_tensor(np.asarray(mask), dtype=torch.float32).permute(2, 0, 1)
        return rgb, (mask_t > 0.5).float()
