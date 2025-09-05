# app.py
import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import io

# -----------------
# App Configuration
# -----------------
st.set_page_config(
    page_title="Mineral Detection System",
    page_icon="🪨",
    layout="wide"
)

# -----------------
# Load YOLOv8 Model
# -----------------
MODEL_PATH = "best.pt"  # Make sure 'best.pt' is available
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
**Dataset**  
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

if uploaded_file is not None:
    # Save uploaded file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(uploaded_file.read())
    img_path = temp_file.name

    # Run YOLO detection
    with st.spinner("🔎 Detecting minerals..."):
        results = model(img_path, conf=0.25)
        annotated_img = results[0].plot()

    # Show Uploaded & Detection Results
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_path, caption="📷 Uploaded Image", use_container_width=True)
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

# -----------------
# Expandable Project Details
# -----------------
with st.expander("📘 About Project & Training Details"):
    st.markdown("""
    ### 🛠 Training Setup  
    - **Model**: YOLOv8m (pretrained on COCO, fine-tuned on custom dataset)  
    - **Epochs**: 50–200  
    - **Image size**: 640×640  
    - **Batch size**: 32  
    - **Optimizer**: AdamW  
    - **Learning rate**: 0.001 → cosine decay (lrf=0.1)  
    - **Augmentations**: Mosaic, Mixup, HSV shift, rotation, scaling  

    ### 📊 Evaluation Metrics  
    - Precision / Recall  
    - mAP@50 and mAP@50-95  
    - Confusion matrix & training curves  

    ✅ The exported `best.pt` is used for real-time inference in this app.
    """)
