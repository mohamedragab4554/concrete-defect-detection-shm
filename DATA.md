# Data sources and licences

**No dataset, image set or weight file is redistributed in this repository**, apart from the small figures in `docs/images` (attributed below).

| Dataset | Used for | Where to get it | Licence | Cite |
|---|---|---|---|---|
| **Crack-Seg** (Ultralytics packaging of Roboflow "crack-bphdr") | YOLO-seg and U-Net | Downloads automatically when a YOLO config uses `crack-seg.yaml`; see the [Ultralytics docs](https://docs.ultralytics.com/datasets/segment/crack-seg/) | Public domain (per the Ultralytics dataset page) | Roboflow Universe, *crack Dataset* (University), 2022 |
| **dacl10k** v2 | FPN multi-class | [dacl10k toolkit / download](https://github.com/phiyodr/dacl10k-toolkit) | **CC BY-NC 4.0** | Flotzinger, J., Rösch, P. J., & Braml, T. (2024). *dacl10k: Benchmark for semantic bridge damage segmentation.* WACV 2024. arXiv:2309.00460 |
| Field images (44) | External validation only | Not available (industry partner) | Private | — |

## Expected layout

```
data/
  crack-seg/
    crack-seg.yaml
    images/{train,val,test}/*.jpg
    labels/{train,val,test}/*.txt      # YOLO-seg polygons
  dacl10k/
    images/{train,validation}/*.jpg
    annotations/{train,validation}/*.json   # LabelMe-style polygons
```

`data/` is in `.gitignore`. Point the configs elsewhere by editing the `*_dir` / `*_yaml` keys: relative paths resolve from the repository root, and absolute paths are used as given.
