# Industrial Defect Detection

Computer vision pipeline for detecting and classifying surface defects on
steel, using classical image processing (OpenCV) and deep learning (PyTorch),
with an interactive Streamlit demo.

## Why this project

This repository extends image-analysis skills directly relevant to my
M.Sc. thesis at FAU Erlangen-Nürnberg (Institute of Photonic Technologies),
*"Development of a Dynamic 3D Model of the Keyhole"*, where I reconstruct
the 3D geometry of the vapor capillary (keyhole) during laser beam welding
from synchrotron X-ray images. Both projects share the same core toolkit:

| Thesis (X-ray keyhole reconstruction) | This project (defect detection) |
|---|---|
| Contour extraction from X-ray projections | Contour / boundary extraction from surface images |
| Automated feature extraction (depth, width, fluctuations) | Automated defect localization and classification |
| Python + image processing pipeline | Python + OpenCV + PyTorch pipeline |
| Distinguishing pores/cracks from background | Distinguishing defect types from clean surface |

This project is a portfolio piece for Data Scientist / ML Engineer roles
in Germany's industrial sector (e.g. automotive, manufacturing, process
quality control).

## Dataset

**NEU Surface Defect Database** — steel-strip surface images across 6
defect classes: crazing, inclusion, patches, pitted surface, rolled-in
scale, scratches. Not committed to this repo — see `docs/dataset.md` for
download instructions and the expected folder layout.

## Project structure

```
industrial-defect-detection/
├── data/
│   ├── raw/            # original NEU-DET images (not tracked)
│   └── processed/      # preprocessed/augmented images (not tracked)
├── notebooks/
│   ├── 01_eda.ipynb                        # exploratory data analysis
│   ├── 02_baseline_model.ipynb             # baseline CNN training
│   ├── 03_transfer_learning_comparison.ipynb  # ResNet18 fine-tuning
│   └── 04_data_leakage_check.ipynb         # train/val duplicate analysis
├── src/
│   ├── data/            # dataset loading & splitting
│   ├── models/          # model architectures & training
│   └── utils/           # OpenCV preprocessing, leakage-detection helpers
├── streamlit_app/
│   └── app.py           # interactive defect-detection demo
├── tests/                # unit tests
├── docs/                 # dataset notes, model card, results
├── models/               # saved model weights (not tracked)
├── requirements.txt
└── README.md
```

## Results

| Model | Val Accuracy | Val Loss | Weighted F1 |
|---|---|---|---|
| Baseline CNN (from scratch) | 90.3% | 0.262 | 0.90 |
| ResNet18 (fine-tuned, ImageNet-pretrained) | 100%* | 0.008 | 1.00* |

\* See **Known Data Limitations** below — the ResNet18 figure is inflated
by train/validation leakage in this dataset distribution and should not
be read as production-ready accuracy.

The baseline CNN was trained from scratch (3 conv blocks + 2 FC layers,
~15 epochs, Adam, lr=1e-3). ResNet18 was fine-tuned end-to-end from
ImageNet-pretrained weights (10 epochs, Adam, lr=1e-4). Both use the same
reusable training loop (`src/models/train.py`), so the comparison isolates
the effect of the architecture/pretraining, not the training procedure.

## Known Data Limitations

Investigation via perceptual hashing (`notebooks/04_data_leakage_check.ipynb`,
`src/utils/leakage_check.py`) found that **50 of 360 validation images
(~14%)** are near-duplicates (Hamming distance ≤ 5) of a training image —
a known characteristic of this NEU-DET distribution, where many crops
originate from a small number of source micrographs.

Leakage is heavily concentrated in two classes:

| Class | Near-duplicates | Share of val set |
|---|---|---|
| inclusion | 26/60 | 43% |
| pitted_surface | 16/60 | 27% |
| crazing | 6/60 | 10% |
| scratches | 2/60 | 3% |
| patches | 0/60 | 0% |
| rolled-in_scale | 0/60 | 0% |

This largely explains the 100% ResNet18 validation accuracy — the model
effectively saw near-identical images during training for the two
highest-leakage classes. Notably, `patches` and `rolled-in_scale` had
**zero** detected duplicates and still scored a perfect F1, suggesting
these two defect types are genuinely highly separable in this dataset.
The overall 100% figure should be read as an artifact of the dataset
split, not a claim of real-world production accuracy — a caveat any
production deployment of this dataset should account for (e.g. by
re-splitting at the source-micrograph level rather than per-crop).

## Roadmap

- [x] Phase 1 — Project scaffolding, EDA, OpenCV preprocessing pipeline
- [x] Phase 2 — Baseline CNN classifier + ResNet18 transfer-learning comparison
- [x] Phase 2b — Data-quality audit (train/val leakage check)
- [ ] Phase 3 — Segmentation/detection model using the XML bounding-box annotations
- [ ] Phase 4 — Streamlit demo wired up to the trained model
- [ ] Phase 5 — Polish: tests, CI, final documentation

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Status

🚧 Phase 2 complete — baseline and transfer-learning models trained and
evaluated, with a documented data-quality caveat. Phase 3 (detection)
next.

## Author

Mohammadreza Fallah — Data Scientist / ML Engineer, M.Sc. Computational
Engineering (FAU Erlangen-Nürnberg)
[LinkedIn](https://linkedin.com/in/mohammadrezafallah) ·
[GitHub](https://github.com/Mohammadreza-Fallah)
