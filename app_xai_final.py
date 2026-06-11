import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import torch
from datetime import datetime

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="PAI Professional Diagnostic Suite",
    layout="wide"
)

# =====================================================
# DEVICE SELECTION
# =====================================================
DEVICE = 0 if torch.cuda.is_available() else "cpu"
HARDWARE = (
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU"
)

# =====================================================
# SESSION STATE
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_dual_models():
    model_n = YOLO("weights/weight 11n.pt")
    model_m = YOLO("weights/weight 26m.pt")
    return model_n, model_m

model_n, model_m = load_dual_models()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ System Controls")

use_tta = st.sidebar.toggle(
    "Enable TTA (Consensus Mode)",
    value=True
)

heatmap_val = st.sidebar.slider(
    "Heatmap Opacity",
    0.0,
    1.0,
    0.5
)

conf_thresh = st.sidebar.slider(
    "Confidence Threshold",
    0.0,
    1.0,
    0.40
)

st.sidebar.divider()

st.sidebar.subheader("👨‍💻 Developer Insight")

st.sidebar.info(
    f"""
**Active Hardware:** {HARDWARE}

**Inference Device:** {DEVICE}

**TTA Status:** {'Active' if use_tta else 'Inactive'}

**Model N:** 640px Inference

**Model M:** 1280px Inference
"""
)

# =====================================================
# DIAGNOSTICS
# =====================================================
with st.sidebar.expander("🛠 System Diagnostics"):
    st.write("CUDA Available:", torch.cuda.is_available())
    st.write("CUDA Device Count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        st.write(
            "GPU Name:",
            torch.cuda.get_device_name(0)
        )

# =====================================================
# PAI DEFINITIONS
# =====================================================
PAI_LOGIC = {
    "PAI_1": "Healthy: Normal periapical bone structure.",
    "PAI_2": "Monitor: Slight widening of periodontal space.",
    "PAI_3": "Pathological: Early diffuse mineral loss.",
    "PAI_4": "Infection: Well-defined bone lesion (Radiolucency).",
    "PAI_5": "Severe: Large lesion with extensive bone destruction."
}

# =====================================================
# HEATMAP FUNCTION
# =====================================================
def get_heatmap(img_array, results, opacity):

    mask = np.zeros(
        (img_array.shape[0], img_array.shape[1]),
        dtype=np.float32
    )

    if len(results[0].boxes) == 0:
        return img_array

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        mask[y1:y2, x1:x2] += (
            box.conf[0].item()
        )

    mask = cv2.GaussianBlur(
        mask,
        (71, 71),
        0
    )

    if np.max(mask) > 0:
        mask = mask / np.max(mask)

    heatmap = cv2.applyColorMap(
        np.uint8(255 * mask),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    return cv2.addWeighted(
        img_array,
        1 - opacity,
        heatmap,
        opacity,
        0
    )

# =====================================================
# MAIN UI
# =====================================================
st.title("🦷 PAI Advanced Multi-Model Analysis")

st.write(
    "Professional comparison of "
    "**Nano (Edge)** vs "
    "**Medium (Precision)** AI engines."
)

uploaded_file = st.file_uploader(
    "Upload Patient X-ray",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# INFERENCE
# =====================================================
if uploaded_file:

    img = Image.open(
        uploaded_file
    ).convert("RGB")

    img_arr = np.array(img)

    timestamp = datetime.now().strftime(
        "%H:%M:%S"
    )

    # ==========================================
    # NANO MODEL
    # ==========================================
    st.divider()

    st.header("📍 Nano Engine (v11n) Analysis")

    with st.spinner(
        "Running Nano Engine..."
    ):

        res_n = model_n.predict(
            source=img,
            imgsz=640,
            conf=conf_thresh,
            augment=use_tta,
            device=DEVICE,
            verbose=False
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.image(
            img,
            use_container_width=True,
            caption="Original"
        )

    with c2:
        st.image(
            res_n[0].plot(),
            use_container_width=True,
            caption="Detection"
        )

    with c3:
        st.image(
            get_heatmap(
                img_arr,
                res_n,
                heatmap_val
            ),
            use_container_width=True,
            caption="AI Reasoning"
        )

    if len(res_n[0].boxes) > 0:

        best_idx = np.argmax(
            res_n[0].boxes.conf.cpu().numpy()
        )

        lbl_n = model_n.names[
            int(
                res_n[0].boxes.cls[
                    best_idx
                ]
            )
        ]

        conf_n = float(
            res_n[0].boxes.conf[
                best_idx
            ]
        )

        st.success(
            f"**Nano Diagnosis:** "
            f"{lbl_n} "
            f"({conf_n:.1%}) "
            f"— {PAI_LOGIC.get(lbl_n,'N/A')}"
        )

        st.session_state.history.append({
            "Time": timestamp,
            "Model": "Nano",
            "File": uploaded_file.name,
            "Result": lbl_n,
            "Confidence": f"{conf_n:.1%}"
        })

    else:
        st.warning(
            "No pathology detected by Nano model."
        )

    # ==========================================
    # MEDIUM MODEL
    # ==========================================
    st.divider()

    st.header("🔬 Medium Engine (v26m) Analysis")

    with st.spinner(
        "Running Medium Engine..."
    ):

        res_m = model_m.predict(
            source=img,
            imgsz=1280,
            conf=conf_thresh,
            augment=use_tta,
            device=DEVICE,
            verbose=False
        )

    c4, c5, c6 = st.columns(3)

    with c4:
        st.image(
            img,
            use_container_width=True,
            caption="Original"
        )

    with c5:
        st.image(
            res_m[0].plot(),
            use_container_width=True,
            caption="Detection"
        )

    with c6:
        st.image(
            get_heatmap(
                img_arr,
                res_m,
                heatmap_val
            ),
            use_container_width=True,
            caption="AI Reasoning"
        )

    if len(res_m[0].boxes) > 0:

        best_idx = np.argmax(
            res_m[0].boxes.conf.cpu().numpy()
        )

        lbl_m = model_m.names[
            int(
                res_m[0].boxes.cls[
                    best_idx
                ]
            )
        ]

        conf_m = float(
            res_m[0].boxes.conf[
                best_idx
            ]
        )

        st.success(
            f"**Medium Diagnosis:** "
            f"{lbl_m} "
            f"({conf_m:.1%}) "
            f"— {PAI_LOGIC.get(lbl_m,'N/A')}"
        )

        st.session_state.history.append({
            "Time": timestamp,
            "Model": "Medium",
            "File": uploaded_file.name,
            "Result": lbl_m,
            "Confidence": f"{conf_m:.1%}"
        })

    else:
        st.warning(
            "No pathology detected by Medium model."
        )

# =====================================================
# HISTORY TABLE
# =====================================================
if st.session_state.history:

    st.divider()

    st.subheader(
        "📋 Clinical History Log"
    )

    log_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        log_df,
        use_container_width=True,
        hide_index=True
    )

    csv = log_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "📩 Download Diagnostic Report (CSV)",
        csv,
        "PAI_Clinical_Report.csv",
        "text/csv"
    )
