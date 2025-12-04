# app.py
import os
import json
import time
from datetime import datetime
from collections import Counter

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Load environment (.env) ----
from dotenv import load_dotenv
load_dotenv()

# ---- NEW OpenAI SDK ----
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="LeafGuard", page_icon="🌿", layout="wide")
st.set_option("client.showErrorDetails", False)

# ---------------------------
# LOAD CSS
# ---------------------------
def load_css(file_name="style.css"):
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ---------------------------
# PATHS & CONSTANTS
# ---------------------------
MODEL_PATH = "models/best_custom_cnn_model.keras"
DRIFT_DIR = "drift_data"
TRAIN_DIST_PATH = os.path.join(DRIFT_DIR, "train_distribution.json")
DRIFT_HISTORY_PATH = os.path.join(DRIFT_DIR, "drift_history.json")
os.makedirs(DRIFT_DIR, exist_ok=True)

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# fallback disease info
DISEASE_INFO = {
    "Default": {
        "description": "Detailed information not available.",
        "cure": "Consult agricultural experts or local guidelines."
    }
}

# ---------------------------
# MODEL LOADING
# ---------------------------
@st.cache_resource
def load_model_from_path(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model_from_path()

def preprocess_image(img, size=(64, 64)):
    img = img.resize(size)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr

# ---------------------------
# DRIFT LOGGING
# ---------------------------
if not os.path.exists(DRIFT_HISTORY_PATH):
    with open(DRIFT_HISTORY_PATH, "w") as f:
        json.dump({"predictions": []}, f)

def log_prediction(pred_class):
    with open(DRIFT_HISTORY_PATH, "r") as f:
        h = json.load(f)
    h["predictions"].append({"class": pred_class, "timestamp": time.time()})
    with open(DRIFT_HISTORY_PATH, "w") as f:
        json.dump(h, f)

# load train dist if exists
train_distribution = {}
if os.path.exists(TRAIN_DIST_PATH):
    with open(TRAIN_DIST_PATH) as f:
        train_distribution = json.load(f)

# ---------------------------
# LLM Cure Generator (OpenAI)
# ---------------------------
def get_llm_cure(plant, disease):
    """Ask OpenAI for cure recommendations."""
    if not OPENAI_API_KEY:
        return None

    prompt = (
        f"Give a short, practical agricultural cure for the plant '{plant}' with disease '{disease}'. "
        "Return only 2–3 sentences, simple and actionable."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None

def get_cure_info(pred_name):
    """Return cure/description using LLM or fallback."""
    if "___" in pred_name:
        plant, disease = pred_name.split("___", 1)
    else:
        plant, disease = pred_name, ""

    plant_clean = plant.replace("_", " ")
    disease_clean = disease.replace("_", " ")

    # Try LLM
    answer = get_llm_cure(plant_clean, disease_clean)
    if answer:
        return "Generated by AI", answer

    # fallback
    return DISEASE_INFO["Default"]["description"], DISEASE_INFO["Default"]["cure"]

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("LeafGuard")
page = st.sidebar.selectbox("Choose page", ["Disease Detection", "Drift Dashboard", "About"])

# ---------------------------
# DISEASE DETECTION PAGE
# ---------------------------
if page == "Disease Detection":
    st.title("🌿 LeafGuard — Plant Disease Detector")
    uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        left, right = st.columns([1, 1.2])
        with left:
            st.image(image, caption="Uploaded Image", width=300)

        with right:
            if model is None:
                st.error("Model not found.")
            else:
                arr = preprocess_image(image)
                with st.spinner("Analyzing..."):
                    preds = model.predict(arr, verbose=0)

                idx = int(np.argmax(preds))
                conf = float(np.max(preds)) * 100
                pred_name = CLASS_NAMES[idx]

                st.subheader("Prediction")
                st.markdown(f"**Class:** `{pred_name}`")
                st.metric("Confidence", f"{conf:.2f}%")

                plant, disease = (pred_name.split("___") + [""])[:2]
                status = "Healthy" if "healthy" in disease.lower() else "Diseased"

                if status == "Healthy":
                    st.success("Status: Healthy")
                else:
                    st.error("Status: Diseased")

                desc, cure = get_cure_info(pred_name)
                st.markdown("### Disease Information")
                st.write(f"**Description:** {desc}")
                st.write(f"**Cure / Treatment:** {cure}")

                log_prediction(pred_name)

# ---------------------------
# DRIFT DASHBOARD
# ---------------------------
elif page == "Drift Dashboard":
    st.title("📈 Drift Monitoring Dashboard")

    with open(DRIFT_HISTORY_PATH) as f:
        history = json.load(f)

    preds = [p["class"] for p in history["predictions"]]
    total = len(preds)
    st.markdown(f"**Total predictions logged:** {total}")

    if total == 0:
        st.warning("No predictions yet.")
    else:
        counts = Counter(preds)

        # Bar Chart — Smaller
        fig, ax = plt.subplots(figsize=(7, 3))
        top = counts.most_common(8)
        labels = [l.replace("___", " — ") for l, _ in top]
        values = [v for _, v in top]
        sns.barplot(x=values, y=labels, palette="rocket", ax=ax)
        st.pyplot(fig)

        # Pie Chart — Smaller
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.pie(values, labels=[l.split(" — ")[-1] for l in labels], autopct="%1.1f%%")
        st.pyplot(fig2)

        # Time Trend Chart
        times = [
            datetime.fromtimestamp(p.get("timestamp", time.time()))
            for p in history.get("predictions", [])
            if isinstance(p, dict)
        ]

        day_counts = Counter([t.date().isoformat() for t in times])
        srt = sorted(day_counts.items())
        if srt:
            x = [d for d, _ in srt]
            y = [c for _, c in srt]
            fig3, ax3 = plt.subplots(figsize=(7, 3))
            ax3.plot(x, y, marker="o")
            plt.xticks(rotation=45)
            st.pyplot(fig3)

# ---------------------------
# ABOUT
# ---------------------------
# ---------------------------
# ABOUT PAGE
# ---------------------------
else:
    st.title("About LeafGuard")
    st.markdown(
        """
        LeafGuard is an intelligent plant disease detection system built using a
        **Custom Convolutional Neural Network (CNN)** trained on the **PlantVillage dataset**.

        ### 🔬 What LeafGuard Does
        - Identifies **38+ plant diseases** across fruits, vegetables, and crops  
        - Uses a trained **TensorFlow CNN model** for image classification  
        - Provides **AI-generated cures** using an LLM (OpenAI API, if available)  
        - Tracks prediction patterns using a **Drift Monitoring Dashboard**  
        - Detects changes in data trends using **distribution comparison & drift score**

        ### 🧠 Technologies Used
        - **TensorFlow / Keras** (Custom CNN Image Classifier)  
        - **PlantVillage Dataset** (Training)  
        - **Streamlit** (Frontend UI + Dashboard)  
        - **Matplotlib & Seaborn** (Charts & visualizations)  
        - **OpenAI API** (Optional — to generate tailored disease treatments)

        ### 📁 App File Structure
        - `models/best_custom_cnn_model.keras` — trained leaf disease model  
        - `drift_data/train_distribution.json` — class distribution from training  
        - `drift_data/drift_history.json` — logs all predictions  
        - `style.css` — custom UI styling  

        ---
        Project by **Iqra, Sameen & Laiba** 🌿  
        """
    )

