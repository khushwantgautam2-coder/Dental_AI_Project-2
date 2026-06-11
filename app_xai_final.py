import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from datetime import datetime

# 1. Page Configuration
st.set_page_config(page_title="PAI Professional Diagnostic Suite", layout="wide")

# 2. Session State for Clinical History
if 'history' not in st.session_state:
    st.session_state.history = []

# 3. Load Models (Optimized for Office PC/Laptop)
@st.cache_resource
def load_dual_models():
    # Ensure these .pt files are in your /weights folder
    model_n = YOLO("weights/weight 11n.pt")
    model_m = YOLO("weights/weight 26m.pt")
    return model_n, model_m

model_n, model_m = load_dual_models()

# 4. Sidebar: Diagnostic & System Controls
st.sidebar.header("⚙️ System Controls")
use_tta = st.sidebar.toggle("Enable TTA (Consensus Mode)", value=True)
heatmap_val = st.sidebar.slider("Heatmap Opacity", 0.0, 1.0, 0.5)
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.40)

st.sidebar.divider()
st.sidebar.subheader("👨‍💻 Developer Insight")
st.sidebar.info(f"""
**Active Hardware:** GPU Accelerated
**TTA Status:** {'Active' if use_tta else 'Inactive'}
**Model N:** 640px Inference
**Model M:** 1280px Inference
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
st.title("🦷 PAI Advanced Multi-Model Analysis")
st.write("Professional comparison of **Nano (Edge)** vs **Medium (Precision)** AI engines.")

uploaded_file = st.file_uploader("Upload Patient X-ray", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_arr = np.array(img)
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # --- ROW 1: NANO MODEL (11n) ---
    st.divider()
    st.header("📍 Nano Engine (v11n) Analysis")
    res_n = model_n.predict(source=img, imgsz=640, conf=conf_thresh, augment=use_tta, device=0)
    
    c1, c2, c3 = st.columns(3)
    with c1: st.image(img, use_container_width=True, caption="Original")
    with c2: st.image(res_n[0].plot(), use_container_width=True, caption="Standard Detection")
    with c3: st.image(get_heatmap(img_arr, res_n, heatmap_val), use_container_width=True, caption="AI Reasoning")
    
    if len(res_n[0].boxes) > 0:
        lbl_n = model_n.names[int(res_n[0].boxes[0].cls[0])]
        conf_n = res_n[0].boxes[0].conf[0]
        st.success(f"**Nano Diagnosis:** {lbl_n} ({conf_n:.1%}) — {PAI_LOGIC.get(lbl_n)}")
        st.session_state.history.append({"Time": timestamp, "Model": "Nano", "File": uploaded_file.name, "Result": lbl_n, "Conf": f"{conf_n:.1%}"})

    # --- ROW 2: MEDIUM MODEL (26m) ---
    st.divider()
    st.header("🔬 Medium Engine (v26m) Analysis")
    res_m = model_m.predict(source=img, imgsz=1280, conf=conf_thresh, augment=use_tta, device=0)
    
    c4, c5, c6 = st.columns(3)
    with c4: st.image(img, use_container_width=True, caption="Original")
    with c5: st.image(res_m[0].plot(), use_container_width=True, caption="Standard Detection")
    with c6: st.image(get_heatmap(img_arr, res_m, heatmap_val), use_container_width=True, caption="AI Reasoning")

    if len(res_m[0].boxes) > 0:
        lbl_m = model_m.names[int(res_m[0].boxes[0].cls[0])]
        conf_m = res_m[0].boxes[0].conf[0]
        st.success(f"**Medium Diagnosis:** {lbl_m} ({conf_m:.1%}) — {PAI_LOGIC.get(lbl_m)}")
        st.session_state.history.append({"Time": timestamp, "Model": "Medium", "File": uploaded_file.name, "Result": lbl_m, "Conf": f"{conf_m:.1%}"})

# 7. Clinical History Log Table
if st.session_state.history:
    st.divider()
    st.subheader("📋 Clinical History Log")
    log_df = pd.DataFrame(st.session_state.history)
    st.dataframe(log_df, use_container_width=True, hide_index=True)
    
    # Download Button
    csv = log_df.to_csv(index=False).encode('utf-8')
    st.download_button("📩 Download Diagnostic Report (CSV)", csv, "PAI_Clinical_Report.csv", "text/csv")