# app.py
import os

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"


import streamlit as st
from ultralytics import YOLO

from PIL import Image
import tempfile
import io
import cv2
import numpy as np

# -----------------
# Page config
# -----------------
st.set_page_config(page_title="Mineral Detection System", page_icon="🪨", layout="wide")

# -----------------
# Model download config (Google Drive)
# -----------------
MODEL_PATH = "best.pt"
# 🔹 Replace with your Google Drive file id
GDRIVE_FILE_ID = "1bArWKdAZt6xHJUSnpLAHZ2IFPzrUz-8J"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

def download_model_from_gdrive(url: str, target_path: str) -> bool:
    """Download using gdown. Returns True if success."""
    try:
        import gdown
    except Exception:
        st.error("⚠️ gdown is not installed. Add `gdown` to requirements.txt.")
        return False

    try:
        gdown.download(url, target_path, quiet=False)
        return os.path.exists(target_path)
    except Exception as e:
        st.error(f"❌ Model download error: {e}")
        return False

# -----------------
# Ensure model exists (download if missing)
# -----------------
if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Model not found locally — downloading from Google Drive..."):
        ok = download_model_from_gdrive(GDRIVE_URL, MODEL_PATH)
    if ok:
        st.success("✅ Model downloaded successfully.")
    else:
        st.error("❌ Failed to download model. Check Drive sharing settings and file id.")
        st.stop()

# -----------------
# Load YOLOv8 Model
# -----------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

with st.spinner("🧠 Loading YOLOv8 model..."):
    model = load_model()

# -----------------
# Sidebar Info
# -----------------
st.sidebar.title("📘 About Us")
st.sidebar.markdown("""
- Custom mineral dataset (Roboflow) with 4 classes:  
  🟤 **Baryte (BaSO₄)**  
  ⚪ **Calcite (CaCO₃)**  
  🟣 **Fluorite (CaF₂)**  
  🟡 **Pyrite (FeS₂)**  

Data was split into train/val/test with augmentations (mosaic, mixup, HSV, rotations).
""")

st.sidebar.title("ℹ️ About This App")
st.sidebar.markdown("""
This **AI-powered Mineral Detection System** uses a YOLOv8 deep learning model  
to identify and classify minerals in uploaded images.
""")

st.sidebar.markdown("---")
st.sidebar.subheader("👨‍💻 Developer")
st.sidebar.markdown("""
**Dileep Kushwaha**  
B.Tech (Honors) CSE (AI)  
CSVTU Bhilai
""")

# Social Links with Icons
st.sidebar.markdown("""
[![Gmail](https://img.icons8.com/color/30/gmail-new.png)](mailto:dileep.300012722032@csvtu.ac.in)
[![LinkedIn](https://img.icons8.com/color/30/linkedin.png)](https://www.linkedin.com/in/dileep-kushwaha-b70a82263/)
[![GitHub](https://img.icons8.com/material-outlined/30/github.png)](https://github.com/DileepKushwah)
""")

st.sidebar.markdown("---")
st.sidebar.info("📌 Purpose: AI-powered mineral classification for geology, mining, and education.\n"
"📂 Dataset Source: [Roboflow](https://roboflow.com/)")

# -----------------
# Title & Instructions
# -----------------
st.title("🔎 Mineral Detection and Classification System")
st.markdown("Upload a mineral image, and the model will **detect and classify** minerals.")

# Mineral descriptions
mineral_info = {
    "Baryte": "Baryte (BaSO₄) — used in drilling fluids due to its high density.",
    "Calcite": "Calcite (CaCO₃) — forms limestone and marble; reacts with acids.",
    "Fluorite": "Fluorite (CaF₂) — famous for fluorescence; used in optics and metallurgy.",
    "Pyrite": "Pyrite (FeS₂) — known as 'Fool’s Gold'; metallic luster and brassy yellow."
}

# -----------------
# File Upload
# -----------------
uploaded_file = st.file_uploader("📤 Upload a mineral image", type=["jpg", "jpeg", "png"])

import cv2
import numpy as np

if uploaded_file is not None:
    # Convert uploaded file → bytes → numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_np = cv2.imdecode(file_bytes, 1)  # Decode as BGR image

    # Run YOLO detection
    with st.spinner("🔎 Detecting minerals..."):
        results = model.predict(img_np, conf=0.25)
        annotated_img = results[0].plot()

    # Show Uploaded & Detection Results
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_np[..., ::-1], caption="📷 Uploaded Image", use_container_width=True)  # convert BGR→RGB
    with col2:
        st.image(annotated_img, caption="✅ Detection Results", use_container_width=True)


    # Show Detected Minerals
    detected_classes = results[0].boxes.cls.cpu().numpy()
    detected_conf = results[0].boxes.conf.cpu().numpy()
    class_names = [model.names[int(c)] for c in detected_classes]

    if class_names:
        st.subheader("📊 Detected Minerals:")
        seen = set()
        for mineral, conf in zip(class_names, detected_conf):
            if mineral not in seen:
                st.success(f"🔹 {mineral} — {conf*100:.2f}% confidence")
                if mineral in mineral_info:
                    st.caption(mineral_info[mineral])
                seen.add(mineral)
    else:
        st.warning("⚠️ No minerals detected. Try another image.")

    # Download annotated image
    buf = io.BytesIO()
    Image.fromarray(annotated_img[..., ::-1]).save(buf, format="JPEG")
    buf.seek(0)
    st.download_button("⬇️ Download annotated image", data=buf,
                       file_name="detection.jpg", mime="image/jpeg")




