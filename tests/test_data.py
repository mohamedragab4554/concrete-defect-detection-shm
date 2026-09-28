import json

import numpy as np

from shm_defects.data import (
    DACL10K_CLASSES,
    LabelMeMultiLabelDataset,
    YoloPolygonMaskDataset,
    labelme_shapes_to_multilabel,
    yolo_polygons_to_mask,
)
from shm_defects.transforms import multiclass_transforms, unet_transforms


def test_yolo_polygon_square(tmp_path):
    lbl = tmp_path / "a.txt"
    lbl.write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n")
    m = yolo_polygons_to_mask(lbl, 100, 100)
    assert m.dtype == np.uint8 and set(np.unique(m)) == {0, 1}
    assert abs(m.sum() - 51 * 51) <= 101  # ~50x50 filled square (inclusive edges)


def test_yolo_missing_label_is_empty(tmp_path):
    assert yolo_polygons_to_mask(tmp_path / "none.txt", 8, 8).sum() == 0


def test_labelme_normalises_label_names():
    shapes = [{"label": "Exposed Rebars", "points": [[0, 0], [9, 0], [9, 9]]},
              {"label": "unknown", "points": [[0, 0], [5, 0], [5, 5]]}]
    m = labelme_shapes_to_multilabel(shapes, 10, 10)
    assert m.shape == (10, 10, len(DACL10K_CLASSES))
    assert m[..., DACL10K_CLASSES.index("ExposedRebars")].sum() > 0
    assert m.sum() == m[..., DACL10K_CLASSES.index("ExposedRebars")].sum()


def _write_img(path, h=64, w=64):
    import cv2
    cv2.imwrite(str(path), np.full((h, w, 3), 128, np.uint8))


def test_yolo_dataset_shapes(tmp_path):
    (tmp_path / "images/val").mkdir(parents=True)
    (tmp_path / "labels/val").mkdir(parents=True)
    _write_img(tmp_path / "images/val/x.jpg")
    (tmp_path / "labels/val/x.txt").write_text("0 0.1 0.1 0.9 0.1 0.9 0.9\n")
    ds = YoloPolygonMaskDataset(tmp_path, "val", unet_transforms(32, train=False))
    img, mask = ds[0]
    assert tuple(img.shape) == (3, 32, 32) and tuple(mask.shape) == (1, 32, 32)
    assert mask.max() == 1


def test_labelme_dataset_shapes(tmp_path):
    (tmp_path / "img").mkdir()
    (tmp_path / "ann").mkdir()
    _write_img(tmp_path / "img/p.jpg")
    ann = {"imagePath": "p.jpg", "imageHeight": 64, "imageWidth": 64,
           "shapes": [{"label": "spalling", "points": [[2, 2], [40, 2], [40, 40]]}]}
    (tmp_path / "ann/p.json").write_text(json.dumps(ann))
    ds = LabelMeMultiLabelDataset(tmp_path / "img", tmp_path / "ann", multiclass_transforms(32, train=False))
    img, mask = ds[0]
    assert tuple(mask.shape) == (18, 32, 32)
    assert mask[DACL10K_CLASSES.index("Spalling")].sum() > 0
