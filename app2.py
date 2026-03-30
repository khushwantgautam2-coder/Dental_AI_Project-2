import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from datetime import datetime
import torch
import os

# 1. Page Config
st.set_page_config(page_title="PAI Nano Diagnostic Suite", layout="wide")

# 2. Session State
if 'history' not in st.session_state:
    st.session_state.history = []

# 3. Load Model SAFELY
@st.cache_resource
def load_nano_model():
    model_path = "weights/weight 11n.pt"
    
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Check weights path.")
        return None
    
    try:
        model = YOLO(model_path)
        model.fuse()  # faster inference
        return model
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

model_n = load_nano_model()

# Detect device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 4. Sidebar
st.sidebar.header("⚙️ System Controls")
use_tta = st.sidebar.toggle("Enable TTA (Consensus Mode)", value=True)
heatmap_val = st.sidebar.slider("Heatmap Opacity", 0.0, 1.0, 0.5)
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.40)

st.sidebar.divider()
st.sidebar.subheader("👨‍💻 Developer Insight")
st.sidebar.info(f"""
**Engine:** YOLOv11 Nano  
**Inference:** 640px (Optimized)  
**TTA Status:** {'Active' if use_tta else 'Inactive'}  
**Hardware:** {device.upper()}  
""")

# 5. Clinical Definitions
PAI_LOGIC = {
    "PAI_1": "Healthy: Normal periapical bone structure.",
    "PAI_2": "Monitor: Slight widening of periodontal space.",
    "PAI_3": "Pathological: Early diffuse mineral loss.",
    "PAI_4": "Infection: Well-defined bone lesion (Radiolucency).",
    "PAI_5": "Severe: Large lesion with extensive bone destruction."
}

# 6. Heatmap SAFE
def get_heatmap(img_array, results, opacity):
    if results is None or len(results) == 0 or results[0].boxes is None:
        return img_array
    
    mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.float32)
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        mask[y1:y2, x1:x2] += float(box.conf[0])
    
    if np.max(mask) == 0:
        return img_array

    mask = cv2.GaussianBlur(mask, (71, 71), 0)
    mask = mask / np.max(mask)

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(img_array, 1 - opacity, heatmap, opacity, 0)

# 7. UI
st.title("🦷 PAI Diagnostic Assistant")
st.write("YOLOv11 Nano powered real-time periapical analysis.")

uploaded_file = st.file_uploader("Upload Patient X-ray", type=["jpg", "png", "jpeg"])

if uploaded_file and model_n:

    try:
        img = Image.open(uploaded_file).convert("RGB")
        img_arr = np.array(img)
        timestamp = datetime.now().strftime("%H:%M:%S")

        st.divider()
        st.header("📍 Nano Engine (v11n) Clinical View")

        # Prediction (SAFE)
        results = model_n.predict(
            source=img_arr,
            imgsz=416,   # faster
            conf=conf_thresh,
            augment=use_tta,
            device=device
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.image(img, use_container_width=True)

        with c2:
            st.image(results[0].plot(), use_container_width=True)

        with c3:
            st.image(get_heatmap(img_arr, results, heatmap_val), use_container_width=True)

        # Diagnosis SAFE
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            label = model_n.names[int(results[0].boxes[0].cls[0])]
            conf = float(results[0].boxes[0].conf[0])

            st.success(f"**Diagnosis:** {label} ({conf:.1%}) — {PAI_LOGIC.get(label, 'No description')}")

            st.session_state.history.append({
                "Time": timestamp,
                "File": uploaded_file.name,
                "Result": label,
                "Confidence": f"{conf:.1%}",
                "TTA": "On" if use_tta else "Off"
            })
        else:
            st.warning("No PAI markers detected.")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# 8. History
if st.session_state.history:
    st.divider()
    st.subheader("📋 Patient Diagnostic History")

    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📩 Export CSV", csv, "PAI_Log.csv")

# 9. Footer
st.divider()
st.caption("Dental AI Project | Stable Build ✅")