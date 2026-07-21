"""
Streamlit demo — upload a steel-surface image and get a defect prediction.

NOTE: this is a Phase-4 placeholder. Model inference will be wired up
once a trained model exists under models/. Run with:

    streamlit run streamlit_app/app.py
"""

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.preprocessing import extract_edges, extract_contours

st.set_page_config(page_title="Industrial Defect Detection", layout="centered")
st.title("Industrial Defect Detection")
st.caption(
    "Upload a steel-surface image to preview the preprocessing pipeline. "
    "Model-based classification will be added in Phase 4."
)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    import numpy as np

    image = Image.open(uploaded).convert("L")
    img_array = np.array(image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        edges = extract_edges(img_array)
        st.image(edges, caption="Canny edges (preview)", use_container_width=True)

    contours = extract_contours(img_array)
    st.write(f"Detected {len(contours)} contour(s) in the image.")
    st.info("Trained-model prediction coming in Phase 4.")
else:
    st.write("Waiting for an image upload...")
