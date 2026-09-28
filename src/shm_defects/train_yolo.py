"""Fine-tune / validate an Ultralytics YOLO-seg model from a YAML config.

    python -m shm_defects.train_yolo --config configs/yolo11x_seg.yaml
    python -m shm_defects.train_yolo --config configs/yolo11x_seg.yaml \
        --validate-only --weights runs/.../best.pt --split test
"""
from __future__ import annotations

import argparse

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--split", default="test", help="Ultralytics split to validate on")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    from ultralytics import YOLO  # imported lazily: optional heavy dependency

    if args.validate_only:
        metrics = YOLO(args.weights or cfg["model"]).val(data=cfg["data_yaml"], split=args.split,
                                                          imgsz=cfg.get("imgsz", 640))
        print(metrics.results_dict)
        return
    YOLO(cfg["model"]).train(data=cfg["data_yaml"], epochs=cfg.get("epochs", 50), imgsz=cfg.get("imgsz", 640),
                             batch=cfg.get("batch", 4), project=cfg.get("project", "runs/yolo"),
                             name=cfg.get("name", "exp"), pretrained=True, seed=cfg.get("seed", 0))


if __name__ == "__main__":
    main()
