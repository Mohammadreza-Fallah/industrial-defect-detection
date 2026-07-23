# Dataset — NEU Surface Defect Database

## Download

The dataset is publicly available on Kaggle — search "NEU Surface Defect
Database" (or "NEU-DET").

## Actual folder layout (as downloaded)

This distribution ships as an object-detection dataset (images +
Pascal-VOC-style XML annotations), pre-split into `train/` and
`validation/`:

```
data/raw/NEU-DET/
├── train/
│   ├── images/
│   │   ├── crazing/*.jpg
│   │   ├── inclusion/*.jpg
│   │   ├── patches/*.jpg
│   │   ├── pitted_surface/*.jpg
│   │   ├── rolled-in_scale/*.jpg
│   │   └── scratches/*.jpg
│   └── annotations/
│       ├── crazing_1.xml
│       ├── patches_35.xml
│       └── ...              (flat, one XML per image)
└── validation/
    ├── images/...
    └── annotations/...
```

`src/data/dataset.py` points at `train/` or `validation/` directly (the
folder that contains `images/`), e.g.:

```python
DATA_ROOT = "../data/raw/NEU-DET/train"
```

For Phase 1/2 (classification), only `images/` is used — the class label
comes from the subfolder name. The `annotations/` XML files (bounding
boxes) are reserved for Phase 3 (detection).

## Class summary

| Class | Folder name | Description |
|---|---|---|
| Crazing | `crazing` | Fine network of surface cracks |
| Inclusion | `inclusion` | Foreign material embedded in the steel |
| Patches | `patches` | Irregular surface patches |
| Pitted Surface | `pitted_surface` | Small pits/craters on the surface |
| Rolled-in Scale | `rolled-in_scale` | Oxide scale rolled into the surface |
| Scratches | `scratches` | Linear surface scratches |

~300 images per class per split, 200x200 px grayscale-ish JPGs.

## Notes for preprocessing

- Images are already fairly clean (lab-controlled acquisition), so heavy
  denoising is not required — focus augmentation on rotation/flip/contrast
  jitter to improve generalization.
- Contour-extraction techniques used here (thresholding, Canny edges,
  connected components) are directly reused from the thesis pipeline for
  isolating the keyhole boundary in X-ray projections.
- Annotation XML files follow a Pascal-VOC-like schema (filename, size,
  object/bndbox per defect instance) — useful later for a detection model
  instead of plain classification.
