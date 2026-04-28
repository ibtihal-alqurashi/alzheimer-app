import os
import numpy as np
import streamlit as st
import requests
from PIL import Image
import tensorflow as tf

from auth import login, check_login
from utils import save_result

# =====================
# UI CONFIG
# =====================
st.set_page_config(page_title="🧠 Alzheimer AI Pro", layout="wide")

# =====================
# LOGIN
# =====================
login()

if not check_login():
    st.warning("Please login to continue")
    st.stop()

# =====================
# MODEL CONFIG
# =====================
CLASSES = ["Mild", "Moderate", "Non Demented", "Very Mild"]
IMG_SIZE = (224, 224)

MODEL_URL = "https://drive.google.com/uc?export=download&id=1s6bLOL2xDY3OQmi0x9SzWVNUanu6O8_H"
MODEL_PATH = "model.h5"

# =====================
# LOAD MODEL
# =====================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)

    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =====================
# SIDEBAR
# =====================
st.sidebar.title("📊 Dashboard")

if os.path.exists("results.csv"):
    st.sidebar.success("Results saved ✔")
else:
    st.sidebar.warning("No results yet")

# =====================
# IMAGE PREPROCESS (FIXED)
# =====================
def preprocess_img(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img)

    if img.shape[-1] == 4:
        img = img[..., :3]

    # normalization instead of VGG preprocess
    img = img.astype(np.float32) / 255.0

    return np.expand_dims(img, axis=0)

# =====================
# MAIN UI
# =====================
st.title("🧠 Alzheimer MRI Classifier (Pro Version)")
st.write("Upload MRI scan to detect Alzheimer stage")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:

    image = Image.open(uploaded)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, use_container_width=True)

    x = preprocess_img(image)

    with st.spinner("Analyzing..."):
        probs = model.predict(x, verbose=0)[0]

    idx = np.argmax(probs)
    label = CLASSES[idx]
    conf = float(probs[idx]) * 100

    with col2:
        st.subheader("🧾 Result")
        st.success(label)
        st.metric("Confidence", f"{conf:.2f}%")

    # =====================
    # INSIGHT
    # =====================
    st.write("### 🧠 Model Insight")

    if conf > 80:
        st.info("High confidence prediction → Model is very sure.")
    elif conf > 50:
        st.warning("Medium confidence → Consider re-checking scan.")
    else:
        st.error("Low confidence → Prediction uncertain.")

    st.bar_chart({c: float(p) for c, p in zip(CLASSES, probs)})

    # =====================
    # SAVE RESULT
    # =====================
    save_result(label, conf)

    st.success("Result saved 📁")