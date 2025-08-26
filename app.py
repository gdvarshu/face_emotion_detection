# app.py (improved + debug info)
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from mtcnn import MTCNN
import os
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="Face Emotion Detection", layout="centered")
st.title("😊 Face Emotion Detection (Debug-Friendly)")

# ----------------------------
# Helpers
# ----------------------------
def load_class_names():
    # 1) prefer saved mapping
    if os.path.exists("class_names.json"):
        with open("class_names.json", "r") as f:
            return json.load(f)
    # 2) else derive from 'train' folder (same sorting Keras uses)
    train_dir = "train"
    if os.path.isdir(train_dir):
        items = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        items.sort()
        if items:
            return items
    # 3) fallback (common FER classes)
    return ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def model_has_rescaling(m):
    for layer in m.layers:
        # direct Rescaling class or layer name contains 'rescaling'
        if isinstance(layer, tf.keras.layers.Rescaling) or 'rescaling' in layer.name.lower():
            return True
    return False

def clamp_box(x, y, w, h, img_w, img_h):
    x = int(max(0, x))
    y = int(max(0, y))
    w = int(max(0, w))
    h = int(max(0, h))
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    return x, y, x2, y2

# ----------------------------
# Load model & class names
# ----------------------------
st.sidebar.info("Make sure emotion_model.h5 exists in this folder.")
try:
    model = tf.keras.models.load_model("emotion_model.h5")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

class_names = load_class_names()
st.sidebar.write("Using class order:")
st.sidebar.write(class_names)

rescaling_present = model_has_rescaling(model)
st.sidebar.write("Model contains Rescaling layer:", rescaling_present)

# MTCNN detector
detector = MTCNN()

# ----------------------------
# Uploader
# ----------------------------
uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.info("Upload an image to run emotion detection.")
    st.stop()

# Read & prepare image
image = Image.open(uploaded_file).convert("RGB")
img_array = np.array(image)
img_h, img_w = img_array.shape[:2]

# Detect faces (MTCNN)
results = detector.detect_faces(img_array)

if len(results) == 0:
    st.warning("No face detected in the image. Try a clear frontal face or a different photo.")
else:
    annotated = img_array.copy()
    all_predictions = []  # for summary chart

    for i, res in enumerate(results, start=1):
        # MTCNN result 'box' = [x, y, width, height] (may be negative)
        x, y, w, h = res.get('box', [0,0,0,0])
        x1, y1, x2, y2 = clamp_box(x, y, w, h, img_w, img_h)

        # if the region is too small, skip
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue

        # extract face ROI and convert to grayscale
        face_rgb = annotated[y1:y2, x1:x2]
        face_gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        face_resized = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
        face_input = face_resized.astype(np.float32)

        # scaling: only scale if model does NOT already include Rescaling
        if not rescaling_present:
            face_input = face_input / 255.0

        # shape to (1, 48, 48, 1)
        face_input = np.expand_dims(face_input, axis=-1)  # channel
        face_input = np.expand_dims(face_input, axis=0)   # batch

        # predict
        preds = model.predict(face_input, verbose=0)[0]  # shape (num_classes,)
        top_idx = int(np.argmax(preds))
        label = class_names[top_idx] if top_idx < len(class_names) else f"class_{top_idx}"
        confidence = float(preds[top_idx])

        # annotate image
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{label} ({confidence:.2f})", (x1, max(y1-8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # save per-face probs for chart
        prob_map = {class_names[j] if j < len(class_names) else f"class_{j}": float(preds[j]) for j in range(len(preds))}
        all_predictions.append({"face": i, "label": label, "confidence": confidence, "probs": prob_map})

    # Show annotated image
    st.image(annotated, caption="Detected faces (click to enlarge)", use_container_width=True)

    # Show detailed predictions
    st.subheader("🔎 Predictions (per face)")
    for p in all_predictions:
        st.write(f"**Face {p['face']}:** {p['label']} (confidence {p['confidence']:.2f})")
        # show top-3
        sorted_probs = sorted(p['probs'].items(), key=lambda x: x[1], reverse=True)
        st.write("Top 3:", sorted_probs[:3])

        # draw a bar chart using matplotlib (no extra packages needed)
        names = [k for k, _ in sorted_probs]
        vals = [v for _, v in sorted_probs]
        plt.figure(figsize=(6,2))
        plt.bar(names, vals)
        plt.ylim(0,1)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Probability")
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()

    # Summary: aggregate counts (optional)
    st.subheader("Summary")
    counts = {}
    for p in all_predictions:
        counts[p['label']] = counts.get(p['label'], 0) + 1
    st.write(counts)
