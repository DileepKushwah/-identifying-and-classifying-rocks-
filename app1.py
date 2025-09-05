import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# 🔹 Load YOLOv8 Model
MODEL_PATH = "best.pt"  # ensure best.pt is in the same folder
model = YOLO(MODEL_PATH)

# 🔹 App Configuration
st.set_page_config(
    page_title="Mineral Detection System",
    page_icon="🪨",
    layout="wide"
)

# 🔹 Sidebar Info
st.sidebar.title("ℹ️ About This App")
st.sidebar.markdown("""
This **Mineral Detection System** uses a trained **YOLOv8 deep learning model**  
to identify and classify minerals in images.  

**Supported Minerals:**
- 🟤 **Baryte (BaSO₄)** → Used in drilling fluids.  
- ⚪ **Calcite (CaCO₃)** → Forms limestone & marble.  
- 🟣 **Fluorite (CaF₂)** → Famous for fluorescence.  
- 🟡 **Pyrite (FeS₂)** → Known as "Fool’s Gold".  
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

# 🔹 Title & Instructions
st.title("🔎 Mineral Detection and Classification System")
st.markdown("""
Upload a mineral image, and the model will **detect and classify** minerals.  
""")

# 🔹 File Upload
uploaded_file = st.file_uploader("📤 Upload a mineral image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file to a temp location
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(uploaded_file.read())
    img_path = temp_file.name

    # Run YOLO detection
    with st.spinner("🔎 Detecting minerals..."):
        results = model(img_path, conf=0.25)
        annotated_img = results[0].plot()

    # 🔹 Show Uploaded & Detection Results Side by Side
    col1, col2 = st.columns(2)

    with col1:
        st.image(img_path, caption="📷 Uploaded Image", use_container_width=True)

    with col2:
        st.image(annotated_img, caption="✅ Detection Results", use_container_width=True)

    # 🔹 Show Detected Minerals in Text
    detected_classes = results[0].boxes.cls.cpu().numpy()  # class IDs
    detected_conf = results[0].boxes.conf.cpu().numpy()   # confidence scores
    class_names = [model.names[int(c)] for c in detected_classes]

    if class_names:
        st.subheader("📊 Detected Minerals:")
        for mineral, conf in zip(class_names, detected_conf):
            st.success(f"🔹 {mineral} — ({conf*100:.2f} % confidence)")
    else:
        st.warning("⚠️ No minerals detected. Try another image.")
