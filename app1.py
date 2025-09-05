# app.py
import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import io
import cv2
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # avoid video backend errors

import cv2  # test import early
print("✅ OpenCV version:", cv2.__version__)

# -----------------
# Page config
# -----------------
st.set_page_config(page_title="Mineral Detection System", page_icon="🪨", layout="wide")

# -----------------
# Model download config (Google Drive)
# -----------------
MODEL_PATH = "best.pt"
# Your Google Drive file id (from the shared link)
GDRIVE_FILE_ID = "1bArWKdAZt6xHJUSnpLAHZ2IFPzrUz-8J"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

def download_model_from_gdrive(url: str, target_path: str) -> bool:
    """Download using gdown. Returns True if success."""
    try:
        import gdown
    except Exception as e:
        st.error("gdown is not installed. Add 'gdown' to requirements.txt.")
        return False

    try:
        # gdown will handle large file download from Google Drive
        gdown.download(url, target_path, quiet=False)
        return os.path.exists(target_path)
    except Exception as e:
        st.error(f"Model download error: {e}")
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
        st.error("❌ Failed to download model. Check drive sharing settings and file id.")
        st.stop()

# -----------------
# Load model and cache it so reloads are fast
# -----------------
@st.cache_resource
def load_yolo_model(path: str):
    return YOLO(path)

with st.spinner("🧠 Loading model (this may take a while on first run)..."):
    model = load_yolo_model(MODEL_PATH)

# -----------------
# Sidebar content
# -----------------
st.sidebar.title("📘 About Us")
st.sidebar.markdown("""
**Dataset**  
Custom mineral dataset with 4 classes: Baryte, Calcite, Fluorite, Pyrite.  
Train/Val/Test splits, with augmentations (mosaic, mixup, HSV, rotation).

**Model**  
YOLOv8m fine-tuned on the dataset. Exports: `best.pt` used for inference.
""")

st.sidebar.title("ℹ️ About This App")
st.sidebar.markdown("""
Upload an image and the app will detect minerals and display results.
""")

st.sidebar.markdown("---")
st.sidebar.subheader("👨‍💻 Developer")
st.sidebar.markdown("**Dileep Kushwaha**  \nB.Tech (Honors) CSE (AI), CSVTU Bhilai")

st.sidebar.markdown(
    "[![GitHub](https://img.icons8.com/material-outlined/24/github.png)](https://github.com/DileepKushwah) "
    "[![LinkedIn](https://img.icons8.com/color/24/linkedin.png)](https://www.linkedin.com/in/dileep-kushwaha-b70a82263/) "
    "[![Gmail](https://img.icons8.com/color/24/gmail-new.png)](mailto:dileep.300012722032@csvtu.ac.in)"
)

# -----------------
# Main UI
# -----------------
st.title("🔎 Mineral Detection and Classification System")
st.markdown("Upload a mineral image and the model will detect Baryte, Calcite, Fluorite, and Pyrite.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

# optional descriptions dict (you can expand)
mineral_info = {
    "Baryte": "Baryte (BaSO₄) — a barium sulfate mineral used in drilling fluids.",
    "Calcite": "Calcite (CaCO₃) — common carbonate mineral; reacts with acid.",
    "Fluorite": "Fluorite (CaF₂) — often fluorescent; used in metallurgy and optics.",
    "Pyrite": "Pyrite (FeS₂) — 'Fool's Gold', metallic luster; used in sulfuric acid production."
}

if uploaded_file is not None:
    # Save uploaded file to disk (preserve extension)
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    img_path = tmp.name

    # run inference
    with st.spinner("🔎 Detecting minerals..."):
        results = model(img_path, conf=0.25)  # returns list of Result
        annotated = results[0].plot()  # numpy image (BGR by ultralytics)

    # show side-by-side original and annotated
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_path, caption="📷 Uploaded Image", use_container_width=True)
    with col2:
        # convert BGR->RGB before displaying via PIL
        if isinstance(annotated, (list, tuple)):
            # handle possible multiple outputs (rare)
            annotated = annotated[0]
        im_rgb = Image.fromarray(annotated[..., ::-1])
        st.image(im_rgb, caption="✅ Detection Results", use_container_width=True)

    # Show text results and descriptions
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        # class ids and confidences
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        st.subheader("📊 Detected Minerals")
        # unique order-preserving
        seen = set()
        for cid, conf in zip(cls_ids, confs):
            name = model.names[int(cid)]
            if name in seen:
                # still print occurrences if you want (comment out if not)
                st.info(f"{name} — {conf*100:.2f}%")
            else:
                seen.add(name)
                st.success(f"{name} — {conf*100:.2f}%")
                # show description if available
                desc = mineral_info.get(name, "")
                if desc:
                    st.caption(desc)
    else:
        st.warning("⚠️ No minerals detected. Try another image.")

    # Provide a download button for the annotated image
    buf = io.BytesIO()
    im_rgb.save(buf, format="JPEG")
    buf.seek(0)
    st.download_button("⬇️ Download annotated image", data=buf, file_name="detection.jpg", mime="image/jpeg")

    # cleanup temp file
    try:
        os.remove(img_path)
    except Exception:
        pass

# Optional: About Project expandable
with st.expander("📘 About Project & Training Details"):
    st.markdown("""
    **Dataset**: Custom mineral dataset (Baryte, Calcite, Fluorite, Pyrite) with bbox annotations.  
    **Model**: YOLOv8m pretrained on COCO, fine-tuned on our dataset.  
    **Training settings (example)**:
    - epochs = 50 (or 200 for full run)
    - imgsz = 640
    - batch = 32
    - optimizer = AdamW
    - lr0 = 0.001, lrf = 0.1
    - augmentations: mosaic, mixup, hsv, rotate, translate, scale
    **Evaluation**: Precision, Recall, mAP@50, mAP@50-95, confusion matrix, training curves.
    """)


