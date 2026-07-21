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

**NEU Surface Defect Database** — 1,800 grayscale images (200x200 px) of
hot-rolled steel strip surfaces, across 6 defect classes:

- Crazing (Cr)
- Inclusion (In)
- Patches (Pa)
- Pitted Surface (PS)
- Rolled-in Scale (RS)
- Scratches (Sc)

Not committed to this repo (see `.gitignore`). Download it and place it
under `data/raw/NEU-DET/` — see `docs/dataset.md` for the exact folder
layout expected by `src/data/dataset.py`.

## Project structure

```
industrial-defect-detection/
├── data/
│   ├── raw/            # original NEU-DET images (not tracked)
│   └── processed/      # preprocessed/augmented images (not tracked)
├── notebooks/
│   └── 01_eda.ipynb    # exploratory data analysis
├── src/
│   ├── data/            # dataset loading & splitting
│   ├── models/          # model architectures & training
│   └── utils/           # OpenCV preprocessing helpers
├── streamlit_app/
│   └── app.py           # interactive defect-detection demo
├── tests/                # unit tests
├── docs/                 # dataset notes, model card, results
├── models/               # saved model weights (not tracked)
├── requirements.txt
└── README.md
```

## Roadmap

- [x] Phase 1 — Project scaffolding, EDA, OpenCV preprocessing pipeline
- [ ] Phase 2 — Baseline classifier (defect type)
- [ ] Phase 3 — Segmentation/detection model (PyTorch)
- [ ] Phase 4 — Streamlit demo (upload image → detected defect)
- [ ] Phase 5 — Polish: README results, tests, CI

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Status

🚧 Week 1 — scaffolding and EDA in progress.

## Author

Mohammadreza Fallah — Data Scientist / ML Engineer, M.Sc. Computational
Engineering (FAU Erlangen-Nürnberg)
[LinkedIn](https://linkedin.com/in/mohammadrezafallah) ·
[GitHub](https://github.com/Mohammadreza-Fallah)
