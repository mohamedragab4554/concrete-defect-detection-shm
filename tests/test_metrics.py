import numpy as np
import pytest

from shm_defects.field_validation import summarise
from shm_defects.metrics import Counts, binary_image_scores, multilabel_iou


def test_binary_scores_perfect_and_disjoint():
    a = np.zeros((4, 4), bool); a[:2] = True
    iou, dice, acc = binary_image_scores(a, a)
    assert iou == pytest.approx(1.0) and dice == pytest.approx(1.0) and acc == 1.0
    iou, dice, _ = binary_image_scores(a, ~a)
    assert iou == pytest.approx(0.0) and dice == pytest.approx(0.0)


def test_counts_reproduce_unet_pixel_metrics():
    # pixel counts from the U-Net evaluation cell (results/unet_resnet50_crack_seg.json)
    c = Counts(tp=1354063, fn=364674, fp=421254, tn=83056809)
    assert c.recall == pytest.approx(0.7878, abs=1e-4)
    assert c.precision == pytest.approx(0.7627, abs=1e-4)
    assert c.f1 == pytest.approx(0.7751, abs=1e-4)


def test_field_validation_reproduces_table7():
    rows = {r["model"]: r for r in summarise("results/field_validation.csv")}
    assert rows["YOLO11x-seg"]["accuracy"] == pytest.approx(0.6136, abs=1e-4)
    assert rows["YOLO11x-seg"]["precision"] == pytest.approx(0.467, abs=1e-3)
    assert rows["YOLO11x-seg"]["recall"] == pytest.approx(0.933, abs=1e-3)
    assert rows["YOLO11n-seg"]["accuracy"] == pytest.approx(0.4318, abs=1e-4)
    assert rows["YOLOv8x-seg"]["accuracy"] == pytest.approx(0.3409, abs=1e-4)
    assert all(r["images"] == 44 for r in rows.values())


def test_multilabel_iou_mean_matches_csv():
    import csv
    with open("results/dacl10k_fpn_effb4_per_class.csv") as f:
        rows = list(csv.DictReader(f))
    per_class = np.array([float(r["IoU"]) for r in rows if r["class"] != "mIoU"])
    reported = float(next(r["IoU"] for r in rows if r["class"] == "mIoU"))
    assert len(per_class) == 18
    assert per_class.mean() == pytest.approx(reported, abs=1e-5)
    miou, iou = multilabel_iou(np.array([1.0, 0.0]), np.array([2.0, 0.0]))
    assert iou.tolist() == [0.5, 0.0] and miou == 0.25
