"""
Streamlit demo — upload a steel-surface image and get defect detections
(bounding boxes + class + confidence) from the fine-tuned Faster R-CNN.

Run from the project root:

    streamlit run streamlit_app/app.py
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw

# Make `src` importable when Streamlit runs this file directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.detection_dataset import CLASS_NAMES
from src.models.detection_model import build_fasterrcnn

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "fasterrcnn.pt"

# One distinct colour per class so overlapping detections stay readable
CLASS_COLORS = {
    "crazing": "#e6194b",
    "inclusion": "#3cb44b",
    "patches": "#ffe119",
    "pitted_surface": "#4363d8",
    "rolled-in_scale": "#f58231",
    "scratches": "#911eb4",
}


@st.cache_resource
def load_model():
    """Loads the trained detector once and keeps it in memory across
    reruns (Streamlit re-executes the whole script on every interaction,
    so without caching the model would reload on every slider move)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_fasterrcnn(num_classes=len(CLASS_NAMES) + 1, backbone="mobilenet")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model, device


def preprocess(image: Image.Image) -> torch.Tensor:
    """PIL image -> (3, H, W) float tensor in [0, 1], matching how
    DetectionDataset feeds images to the model during training."""
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(gray).unsqueeze(0)  # (1, H, W)
    return tensor.repeat(3, 1, 1)  # (3, H, W)


def draw_detections(image: Image.Image, boxes, labels, scores) -> Image.Image:
    """Draws boxes with class name and confidence onto a copy of the image."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [float(v) for v in box]
        class_name = CLASS_NAMES[int(label) - 1]
        color = CLASS_COLORS.get(class_name, "#ffffff")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        caption = f"{class_name} {score:.2f}"
        text_box = draw.textbbox((0, 0), caption)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        # Keep the label inside the image when the box touches the top edge
        label_y = max(0, y1 - text_h - 4)
        draw.rectangle([x1, label_y, x1 + text_w + 6, label_y + text_h + 4], fill=color)
        draw.text((x1 + 3, label_y + 2), caption, fill="black")

    return annotated


st.set_page_config(page_title="Industrial Defect Detection", layout="wide")
st.title("Industrial Defect Detection")
st.caption(
    "Faster R-CNN (MobileNetV3-FPN backbone) fine-tuned on the NEU Surface "
    "Defect Database. Upload a steel-surface image to locate and classify "
    "defects. Validation mAP@0.5: 0.74."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model weights not found at `{MODEL_PATH}`. Train the detector first "
        "by running `notebooks/05_detection_model.ipynb`."
    )
    st.stop()

model, device = load_model()

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Only detections scoring above this are shown. Lower it to "
             "catch more defects at the cost of more false positives.",
    )
    st.caption(f"Running on: `{device}`")

uploaded = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded is None:
    st.info(
        "Upload a steel-surface image to begin. Images from "
        "`data/raw/NEU-DET/validation/images/` work well for testing."
    )
    st.stop()

image = Image.open(uploaded)
tensor = preprocess(image).to(device)

with torch.no_grad():
    prediction = model([tensor])[0]

keep = prediction["scores"] > threshold
boxes = prediction["boxes"][keep].cpu()
labels = prediction["labels"][keep].cpu()
scores = prediction["scores"][keep].cpu()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Original")
    st.image(image, use_container_width=True)
with col2:
    st.subheader("Detections")
    st.image(draw_detections(image, boxes, labels, scores), use_container_width=True)

st.subheader("Results")
if len(boxes) == 0:
    st.warning(
        f"No defects detected above a confidence of {threshold:.2f}. "
        "Try lowering the threshold in the sidebar."
    )
else:
    counts = {}
    for label in labels:
        name = CLASS_NAMES[int(label) - 1]
        counts[name] = counts.get(name, 0) + 1

    summary = ", ".join(f"{n}x {name}" for name, n in sorted(counts.items()))
    st.success(f"Found {len(boxes)} defect(s): {summary}")

    rows = []
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [round(float(v)) for v in box]
        rows.append(
            {
                "Class": CLASS_NAMES[int(label) - 1],
                "Confidence": f"{float(score):.3f}",
                "Box (x1, y1, x2, y2)": f"({x1}, {y1}, {x2}, {y2})",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
