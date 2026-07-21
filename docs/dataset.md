# Dataset — NEU Surface Defect Database

## Download

The dataset is publicly available. Common sources:

1. Kaggle: search "NEU Surface Defect Database" (or "NEU-DET")
2. Original source: Northeastern University (NEU), China — Song & Yan
   surface-defect database

Download and unzip so the images land under:

```
data/raw/NEU-DET/
├── Crazing/
├── Inclusion/
├── Patches/
├── Pitted_Surface/
├── Rolled-in_Scale/
└── Scratches/
```

(Some distributions ship it already split into `train/` and `validation/`
subfolders with the same 6 class folders inside each — `src/data/dataset.py`
handles both layouts; see the `NEUDataset` docstring.)

## Class summary

| Class | Code | Description |
|---|---|---|
| Crazing | Cr | Fine network of surface cracks |
| Inclusion | In | Foreign material embedded in the steel |
| Patches | Pa | Irregular surface patches |
| Pitted Surface | PS | Small pits/craters on the surface |
| Rolled-in Scale | RS | Oxide scale rolled into the surface |
| Scratches | Sc | Linear surface scratches |

300 images per class, 1800 total, 200x200 px grayscale.

## Notes for preprocessing

- Images are already fairly clean (lab-controlled acquisition), so heavy
  denoising is not required — focus augmentation on rotation/flip/contrast
  jitter to improve generalization.
- Contour-extraction techniques used here (thresholding, Canny edges,
  connected components) are directly reused from the thesis pipeline for
  isolating the keyhole boundary in X-ray projections.
