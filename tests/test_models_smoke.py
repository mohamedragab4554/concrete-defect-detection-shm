import torch

from shm_defects.losses import BCEDiceLoss, SoftDiceLoss, estimate_pos_weight
from shm_defects.models import build_fpn, build_unet
from shm_defects.train_multiclass import lr_at


def test_unet_forward_cpu():
    m = build_unet("resnet18", pretrained=False).eval()
    with torch.no_grad():
        y = m(torch.zeros(1, 3, 64, 64))
    assert tuple(y.shape) == (1, 1, 64, 64)


def test_fpn_forward_cpu():
    m = build_fpn(18, "resnet18", pretrained=False).eval()
    with torch.no_grad():
        y = m(torch.zeros(1, 3, 64, 64))
    assert tuple(y.shape) == (1, 18, 64, 64)


def test_losses_finite_and_ordered():
    t = torch.zeros(2, 3, 8, 8); t[:, :, :4] = 1
    good = (t * 20 - 10)
    bad = -good
    for loss in (SoftDiceLoss(), BCEDiceLoss(torch.ones(3))):
        assert torch.isfinite(loss(good, t)) and loss(good, t) < loss(bad, t)


def test_pos_weight_and_schedule():
    loader = [(None, torch.cat([torch.ones(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)], 1))]
    pw = estimate_pos_weight(loader, 2, clamp_max=10)
    assert pw[0] == 1 / 16  # all-positive class: neg clamped to 1 pixel
    assert pw[1] == 10  # empty class: ratio 15/1 clamped to clamp_max
    assert lr_at(0, 1.0, 2, 10) == 0.5 and lr_at(2, 1.0, 2, 10) == 1.0 and lr_at(10, 1.0, 2, 10) < 1e-9
