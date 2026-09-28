# Original research notebooks

These are the notebooks that produced the dissertation results. They are kept as a record of the runs, and their **text outputs are preserved** (training logs, metrics).

Changes from the originals:
- local Windows paths replaced with `data/...` or `<local>/`;
- image outputs and `pip` logs removed;
- very long logs truncated to their last 400 lines.

| Notebook | Produces |
|---|---|
| `YOLOv8x-ML.ipynb`, `YOLO11x-ML.ipynb`, `YOLO11n-ML-2.ipynb` | YOLO-seg training and validation on Crack-Seg (Table 6) |
| `UNET_WITH_RESNET.ipynb` | U-Net ResNet-50 training, then mean IoU 0.6303 / Dice 0.7576 and pixel P/R/F1 |
| `Multi_class_model.ipynb` | FPN EfficientNet-B4 on dacl10k; best validation mIoU 0.3169 |

For new work, use the package in `src/shm_defects`, which refactors the same logic with configs and tests. The notebooks assume the Windows / CUDA environment described in the main README.
