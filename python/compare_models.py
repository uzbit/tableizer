import torch
import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path

# --- Configuration ---
img_path = "../../../data/Photos-1-001/P_20250718_205820.jpg"  # change to your image
pt_model_path = "best.pt"
onnx_model_path = "best.onnx"
imgsz = 800

# --- Load and preprocess image ---
def load_image(path, size):
    img = cv2.imread(path)
    h0, w0 = img.shape[:2]
    r = min(size / w0, size / h0)
    new_w, new_h = int(w0 * r), int(h0 * r)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_w, pad_h = (size - new_w) // 2, (size - new_h) // 2
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = img_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0  # CHW
    return tensor[None], r, pad_w, pad_h, (h0, w0)

img_tensor, r, pad_w, pad_h, original_shape = load_image(img_path, imgsz)

# --- Run PyTorch model ---
pt_model = torch.hub.load('ultralytics/yolov9', 'custom', path=pt_model_path)
pt_results = pt_model(img_path, size=imgsz)
pt_boxes = pt_results.pandas().xyxy[0][['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']].values

# --- Run ONNX model ---
ort_sess = ort.InferenceSession(onnx_model_path)
input_name = ort_sess.get_inputs()[0].name
onnx_outputs = ort_sess.run(None, {input_name: img_tensor.astype(np.float32)})
onnx_raw = onnx_outputs[0]  # shape: [1, 8, 13125]
onnx_raw = onnx_raw[0]  # remove batch dim

# Convert ONNX output to readable detections
num_preds = onnx_raw.shape[1]
boxes = []
for i in range(num_preds):
    cx, cy, w, h = onnx_raw[0, i], onnx_raw[1, i], onnx_raw[2, i], onnx_raw[3, i]
    obj = 1 / (1 + np.exp(-onnx_raw[4, i]))
    class_scores = 1 / (1 + np.exp(-onnx_raw[5:, i]))
    cls_id = np.argmax(class_scores)
    cls_score = class_scores[cls_id]
    conf = obj * cls_score
    if conf < 0.25:
        continue
    x1 = (cx - w / 2 - pad_w) / r
    y1 = (cy - h / 2 - pad_h) / r
    x2 = (cx + w / 2 - pad_w) / r
    y2 = (cy + h / 2 - pad_h) / r
    boxes.append([x1, y1, x2, y2, conf, cls_id])

# --- Return both sets of boxes for display ---
import pandas as pd
pt_df = pd.DataFrame(pt_boxes, columns=["x1", "y1", "x2", "y2", "conf", "class"])
onnx_df = pd.DataFrame(boxes, columns=["x1", "y1", "x2", "y2", "conf", "class"])

import ace_tools as tools; tools.display_dataframe_to_user(name="ONNX Model Detections", dataframe=onnx_df)
pt_df.head()  # return PyTorch results inline