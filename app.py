from taipy.gui import Gui, notify
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# --- CONFIG & MODEL ---
model = YOLO("weights/weight 11n.pt")
content = None  # Holds the uploaded image
results_img = None
heatmap_img = None
opacity = 0.5
status = "Waiting for X-ray upload..."

# --- PAI LOGIC ---
def process_image(state):
    state.status = "Analyzing Periapical Region..."
    img = Image.open(state.content).convert("RGB")
    res = model.predict(img, imgsz=640, conf=0.4)
    
    # 1. Standard Plot
    state.results_img = res[0].plot()
    
    # 2. Heatmap Generation
    img_arr = np.array(img)
    mask = np.zeros(img_arr.shape[:2], dtype=np.float32)
    for box in res[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        mask[y1:y2, x1:x2] += box.conf[0].item()
    
    mask = cv2.GaussianBlur(mask, (71, 71), 0)
    if np.max(mask) > 0: mask /= np.max(mask)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    state.heatmap_img = cv2.addWeighted(img_arr, 1 - state.opacity, cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), state.opacity, 0)
    
    state.status = "Analysis Complete."
    notify(state, "s", "Diagnosis Generated!")

# --- UI LAYOUT (Markdown) ---
page = """
# 🦷 **PAI Diagnostic Assistant** (Taipy Edition)

<|layout|columns=1 1|
<|{content}|file_selector|label=Upload X-ray|on_action=process_image|extensions=.jpg,.jpeg,.png|>
<|{status}|text|class_name=status-text|>
|>

---

<|layout|columns=1 1 1|
### Original Scan
<|{content}|image|width=100%|>

### AI Detection
<|{results_img}|image|width=100%|>

### XAI Reasoning
<|{heatmap_img}|image|width=100%|>
|>

**Heatmap Opacity:** <|{opacity}|slider|min=0|max=1|on_change=process_image|>
"""

if __name__ == "__main__":
    Gui(page=page).run(dark_mode=True, port=5001)
