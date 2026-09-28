# Model card: concrete defect segmentation models (research)

| | |
|---|---|
| **Models** | YOLOv8x-seg, YOLO11x-seg, YOLO11n-seg (crack instance segmentation); U-Net ResNet-50 (binary crack masks); FPN EfficientNet-B4 (18-class multi-label defects) |
| **Developer** | Mohamed Ragab, MSc dissertation, Ulster University (2025) |
| **Type** | Research prototypes. Not validated for safety-critical or regulatory decisions |
| **Weights** | **Not published.** YOLO weights derive from Ultralytics models (AGPL-3.0). FPN weights are trained on dacl10k (CC BY-NC 4.0). Field data belongs to an industry partner |

## Intended use

- **Primary:** decision support for trained inspectors. The models flag images and regions that may contain cracks or other defects, for review by a competent engineer.
- **Users:** structural and asset-management engineers, and researchers studying computer vision for inspection.
- **Out of scope:** automated condition rating, structural capacity assessment, crack-width measurement in mm, compliance sign-off, or any use without human review. **The dacl10k-trained FPN must not be used commercially** (CC BY-NC 4.0) without the dataset owners' permission.

## Training data

| Model | Data | Split used | Licence |
|---|---|---|---|
| YOLO-seg ×3 | Ultralytics **Crack-Seg**: 3,717 train, 200 `val`, 112 `test` images at 416×416, upscaled to 640 | train / `val` | Public domain (via Roboflow Universe) |
| U-Net | Crack-Seg converted to images/labels folders: 3,812 train, 208 `val` | train / `val` | Public domain |
| FPN | **dacl10k** v2 (bridge inspection images, 18 classes) | train / validation | CC BY-NC 4.0 |
| Field test only | 44 real inspection photographs from a Belfast industry partner | test only | Private; never used in training |

## Evaluation, splits and leakage controls

| Model | Metric | Value | Split | Leakage note |
|---|---|---:|---|---|
| YOLOv8x-seg | box / mask mAP50 | 0.807 / 0.674 | Crack-Seg `val` (200 images) | Split also used by Ultralytics to select `best.pt` |
| YOLO11x-seg | box / mask mAP50 | 0.804 / 0.639 | same | same |
| YOLO11n-seg | box / mask mAP50 | 0.792 / 0.658 | same | same |
| U-Net ResNet-50 | mean IoU / pixel F1 | 0.630 / 0.775 | converted `val` (208 images) | Same split selected the checkpoint; optimistic |
| FPN EfficientNet-B4 | mIoU (18 classes) | 0.317 | dacl10k validation | Same split selected the checkpoint |
| YOLO11x-seg | image-level recall / specificity | 0.933 / 0.448 | 44 field images | **Independent**: never seen in training or tuning |

**Metric definitions** are in `src/shm_defects/metrics.py`:
- mean IoU is averaged per image;
- pixel precision, recall and F1 come from confusion counts summed over the split;
- multi-label IoU is dataset-level per class;
- field metrics are image-level: an image is positive if the model predicts any crack.

**Known gaps:**
- No run on the untouched Crack-Seg `test` split was saved. The CLIs support it: `--split test`.
- No confidence intervals were reported. With only 15 cracked and 29 crack-free field images, the Wilson 95% intervals are wide: recall 0.70–0.99, specificity 0.28–0.62.

## Limitations

- **Domain shift:**
  - Specificity falls sharply on site photographs. Service pipes, cables, shadows, coatings and textured surfaces cause most false positives.
  - YOLOv8x-seg marked 28 of 29 crack-free field images as cracked.
- **Thin structures:** mask mAP50-95 is about 0.22 for every YOLO model. dacl10k "Crack" IoU is 0.066. Pixel-accurate crack geometry is not reliable.
- **Class imbalance:** rare dacl10k classes are close to zero ("Climbingplants" 0.000, "Rockpocket" 0.098). The FPN over-segments: recall exceeds precision for most classes.
- **No physical scale:** outputs are in pixels. Width or length in mm needs a calibration target or known geometry.
- **Small field sample:** 44 images from one partner and a limited range of structures.

## Hardware and software

One NVIDIA RTX 4060 (8 GB) on Windows. PyTorch 2.5.1, segmentation-models-pytorch 0.5.0, Ultralytics 8.3.169.
- YOLO11x-seg: 50 epochs in 4.0 h at batch 4.
- Inference: 0.53 s/image for the U-Net, 0.77 s/image for YOLO, 2.05 s/image for the FPN.

## Ethical, safety and engineering-use boundaries

- A model output is **evidence for an engineer, not a decision**. Condition ratings, repair priorities and safety actions stay with a competent person under the relevant standards and the asset owner's procedures.
- **Missing defects is the main safety risk.** Recall on 15 cracked field images is 0.933 (one miss). Do not use any model here to *clear* a structure.
- **Privacy:** site photographs can contain people, vehicle plates and addresses. Blur or crop before sharing, and follow the data owner's terms.
- **Licences:** respect them. No dacl10k-derived model in commercial products; Ultralytics models are AGPL-3.0.
