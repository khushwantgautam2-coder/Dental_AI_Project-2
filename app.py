import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from datetime import datetime

# 1. Page Configuration
st.set_page_config(page_title="PAI Nano Diagnostic Suite", layout="wide")

# 2. Session State for Clinical History
if 'history' not in st.session_state:
    st.session_state.history = []

# 3. Load Model (Nano Engine)
@st.cache_resource
def load_nano_model():
    # Ensure 'weight 11n.pt' is in your /weights folder
    return YOLO("weights/weight 11n.pt")

model_n = load_nano_model()

# 4. Sidebar: System Controls & Developer Insight
st.sidebar.header("⚙️ System Controls")
use_tta = st.sidebar.toggle("Enable TTA (Consensus Mode)", value=True)
heatmap_val = st.sidebar.slider("Heatmap Opacity", 0.0, 1.0, 0.5)
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.40)

st.sidebar.divider()
st.sidebar.subheader("👨‍💻 Developer Insight")
st.sidebar.info(f"""
**Engine:** YOLOv11 Nano
**Inference:** 640px (Mobile Optimized)
**TTA Status:** {'Active' if use_tta else 'Inactive'}
**Hardware:** GPU Accelerated
""")

# 5. Clinical Definitions
PAI_LOGIC = {
    "PAI_1": "Healthy: Normal periapical bone structure.",
    "PAI_2": "Monitor: Slight widening of periodontal space.",
    "PAI_3": "Pathological: Early diffuse mineral loss.",
    "PAI_4": "Infection: Well-defined bone lesion (Radiolucency).",
    "PAI_5": "Severe: Large lesion with extensive bone destruction."
}

def get_heatmap(img_array, results, opacity):
    mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.float32)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        mask[y1:y2, x1:x2] += box.conf[0].item()
    mask = cv2.GaussianBlur(mask, (71, 71), 0)
    if np.max(mask) > 0: mask = mask / np.max(mask)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_array, 1 - opacity, heatmap, opacity, 0)

# 6. Main Dashboard UI
st.title("🦷 PAI Diagnostic Assistant")
st.write("Diagnostic engine powered by **YOLOv11 Nano** for real-time periapical analysis.")

uploaded_file = st.file_uploader("Upload Patient X-ray", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_arr = np.array(img)
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # --- ANALYSIS SECTION ---
    st.divider()
    st.header("📍 Nano Engine (v11n) Clinical View")
    
    # Run Prediction
    res_n = model_n.predict(source=img, imgsz=640, conf=conf_thresh, augment=use_tta, device=0)
    
    # Triple-View Columns
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(img, use_container_width=True, caption="Original Scan")
    with c2:
        st.image(res_n[0].plot(), use_container_width=True, caption="Standard Detection (YOLO)")
    with c3:
        st.image(get_heatmap(img_arr, res_n, heatmap_val), use_container_width=True, caption="Explainable AI (Reasoning)")
    
    # Clinical Reasoning Text
    if len(res_n[0].boxes) > 0:
        label = model_n.names[int(res_n[0].boxes[0].cls[0])]
        conf = res_n[0].boxes[0].conf[0]
        st.success(f"**Diagnosis:** {label} ({conf:.1%}) — {PAI_LOGIC.get(label)}")
        
        # Add to history
        st.session_state.history.append({
            "Time": timestamp, 
            "File": uploaded_file.name, 
            "Result": label, 
            "Confidence": f"{conf:.1%}",
            "TTA": "On" if use_tta else "Off"
        })
    else:
        st.warning("No PAI markers detected at this threshold. Consider adjusting confidence or TTA settings.")

# 7. Clinical History Log Table
if st.session_state.history:
    st.divider()
    st.subheader("📋 Patient Diagnostic History")
    log_df = pd.DataFrame(st.session_state.history)
    st.dataframe(log_df, use_container_width=True, hide_index=True)
    
    # Download Button
    csv = log_df.to_csv(index=False).encode('utf-8')
    st.download_button("📩 Export Diagnostic Report (CSV)", csv, "PAI_Session_Log.csv", "text/csv")

# 8. Portfolio Footer
st.divider()
st.caption("Dental AI Project | Optimization: Edge-Ready (v11n)")
