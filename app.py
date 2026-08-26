"""Small Streamlit deployment app for SiteGuard Vision."""
from pathlib import Path
import tempfile
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="SiteGuard Vision", layout="wide")
st.title("SiteGuard Vision — PPE Compliance Inspector")
st.write("Upload a construction-site image to detect PPE and missing-PPE classes with the fine-tuned YOLO model.")

weights = st.sidebar.text_input("Weights", "runs/siteguard/ppe_train/weights/best.pt")
conf = st.sidebar.slider("Confidence", 0.05, 0.95, 0.35, 0.05)
upload = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if upload:
    if not Path(weights).exists():
        st.error("Trained weights not found. Run src/train.py first or provide a valid best.pt path.")
    else:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(upload.getbuffer())
            temp_path = f.name
        model = YOLO(weights)
        result = model.predict(temp_path, conf=conf, verbose=False)[0]
        plotted = result.plot()[:, :, ::-1]
        st.image(Image.fromarray(plotted), caption=f"Detections: {len(result.boxes) if result.boxes is not None else 0}", use_container_width=True)
