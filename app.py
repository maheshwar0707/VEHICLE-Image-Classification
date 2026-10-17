"""
AI Vehicle Classifier - Streamlit Web Application
=================================================
Interactive web app for real-time fine-grained vehicle classification
using the trained MobileNetV3Large transfer learning model (99.02% accuracy).
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="AI Vehicle Classifier",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Sleek Glassmorphism UI
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 12px;
        background-color: #f8f9fa;
        border-left: 6px solid #1f77b4;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .prediction-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #111;
        margin-bottom: 0.3rem;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        background-color: #28a745;
        color: white;
        font-weight: 600;
        border-radius: 20px;
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
    }
    .taxonomy-text {
        font-size: 0.95rem;
        color: #444;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Taxonomy & Class Metadata
# ==============================================================================
CLASS_METADATA: Dict[str, Dict[str, str]] = {
    "Auto Rickshaws": {
        "emoji": "🛺",
        "taxonomy": "Three-Wheeler Commercial Transit (Category L5M/L5N)",
        "description": "Lightweight 3-wheeled passenger/cargo urban transport vehicle.",
    },
    "Bikes": {
        "emoji": "🚲",
        "taxonomy": "Human/Light Mobility (Pedal Cycle)",
        "description": "Two-wheeled pedal-driven or lightweight pedal-assist mobility vehicle.",
    },
    "Cars": {
        "emoji": "🚗",
        "taxonomy": "Passenger Four-Wheeler (Category M1)",
        "description": "Four-wheeled passenger motor vehicle (Sedans, SUVs, Hatchbacks).",
    },
    "Motorcycles": {
        "emoji": "🏍️",
        "taxonomy": "Motorized Two-Wheeler (Category L3)",
        "description": "Motorized 2-wheeled high-speed vehicle (Sport, Cruiser, Commuter).",
    },
    "Planes": {
        "emoji": "✈️",
        "taxonomy": "Aviation Craft (Heavier-than-Air)",
        "description": "Fixed-wing aerodynamic powered aircraft for civil, commercial, or cargo transport.",
    },
    "Ships": {
        "emoji": "🚢",
        "taxonomy": "Marine Watercraft",
        "description": "Large waterborne vessel navigating oceans, seas, and deep waterways.",
    },
    "Trains": {
        "emoji": "🚆",
        "taxonomy": "Rail & Track Transit (Locomotive)",
        "description": "Series of connected vehicles operating on guided railway tracks.",
    },
}

MODEL_PATH = "best_vehicle_classifier.keras"
CLASSES_PATH = "vehicle_classes.json"


# ==============================================================================
# Cached Model Loader
# ==============================================================================
@st.cache_resource(show_spinner=False)
def get_model():
    """Loads and caches the compiled Keras model and class names list."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found. Please train the model first.")
        st.stop()

    model = tf.keras.models.load_model(MODEL_PATH)

    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, "r") as f:
            classes = json.load(f)
    else:
        classes = list(CLASS_METADATA.keys())

    return model, classes


# ==============================================================================
# Prediction Function
# ==============================================================================
def predict_image(image: Image.Image, model: tf.keras.Model, classes: List[str]):
    """Preprocesses an image and returns predictions with probability distribution."""
    # Resize to exact expected MobileNetV3 input shape
    rgb_img = image.convert("RGB").resize((224, 224))
    img_array = np.array(rgb_img, dtype=np.float32)
    input_tensor = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

    # Forward inference
    probs = model.predict(input_tensor, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    top_class = classes[top_idx]
    top_prob = float(probs[top_idx])

    # Sorted list of (class_name, probability)
    distribution = [
        (classes[i], float(probs[i]))
        for i in np.argsort(probs)[::-1]
    ]

    return top_class, top_prob, distribution


# ==============================================================================
# Sidebar & App Controls
# ==============================================================================
with st.sidebar:
    st.image(
        "https://img.icons8.com/clouds/200/car.png",
        width=120,
    )
    st.title("Vehicle Classifier AI")
    st.markdown("**Powered by TensorFlow 2.x & MobileNetV3Large**")
    st.markdown("---")

    st.subheader("📊 Model Performance")
    st.metric(label="Validation Accuracy", value="99.02%")
    st.metric(label="Validation Loss", value="0.0431")
    st.metric(label="Backbone Architecture", value="MobileNetV3-L")
    st.metric(label="Trained Classes", value="7 Categories")

    st.markdown("---")
    st.subheader("📚 Categories Supported")
    for cls_name, meta in CLASS_METADATA.items():
        st.markdown(f"**{meta['emoji']} {cls_name}**")
        st.caption(meta["taxonomy"])

    st.markdown("---")
    st.caption("Developed with TensorFlow & Streamlit")


# ==============================================================================
# Main User Interface
# ==============================================================================
st.markdown('<div class="main-header">🚗 AI Fine-Grained Vehicle Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload any vehicle photograph or choose a pre-loaded sample. The deep learning model will analyze visual contours and predict the vehicle category with high-confidence probability estimates.</div>',
    unsafe_allow_html=True,
)

model, classes = get_model()

# Input Mode Tabs
input_tab1, input_tab2, input_tab3 = st.tabs(["📁 Upload Image", "🖼️ Try Sample Image", "📸 Webcam Snapshot"])

selected_image = None
image_caption = ""

with input_tab1:
    uploaded_file = st.file_uploader(
        "Choose an image from your device (JPG, PNG, JPEG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        key="file_uploader",
    )
    if uploaded_file is not None:
        try:
            selected_image = Image.open(uploaded_file)
            image_caption = f"Uploaded File: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Error reading image: {e}")

with input_tab2:
    # Discover available sample images from the local dataset directory
    vehicles_dir = Path("Vehicles")
    sample_options = {}

    if vehicles_dir.exists():
        for category in sorted([d for d in vehicles_dir.iterdir() if d.is_dir()]):
            sample_files = list(category.glob("*.jpg")) + list(category.glob("*.png"))
            if sample_files:
                sample_options[f"{category.name} (Sample 1)"] = sample_files[0]
                if len(sample_files) > 1:
                    sample_options[f"{category.name} (Sample 2)"] = sample_files[1]

    if sample_options:
        chosen_sample_name = st.selectbox("Select a benchmark sample to test:", list(sample_options.keys()))
        if st.button("Load Selected Sample Image", type="secondary"):
            sample_path = sample_options[chosen_sample_name]
            selected_image = Image.open(sample_path)
            image_caption = f"Dataset Sample: {chosen_sample_name}"
    else:
        st.info("No samples found in the 'Vehicles' folder.")

with input_tab3:
    camera_file = st.camera_input("Take a photo using your camera")
    if camera_file is not None:
        selected_image = Image.open(camera_file)
        image_caption = "Webcam Snapshot"


# ==============================================================================
# Inference & Results Presentation
# ==============================================================================
st.markdown("---")

if selected_image is not None:
    col_img, col_pred = st.columns([1, 1.2], gap="large")

    with col_img:
        st.subheader("📷 Input Image")
        st.image(selected_image, caption=image_caption, use_container_width=True)

    with col_pred:
        st.subheader("🎯 AI Prediction")

        with st.spinner("Analyzing image features..."):
            top_class, top_prob, distribution = predict_image(selected_image, model, classes)

        meta = CLASS_METADATA.get(top_class, {"emoji": "🚗", "taxonomy": "Vehicle", "description": ""})
        confidence_percent = top_prob * 100.0

        # Prediction Highlight Card
        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-title">{meta['emoji']} {top_class}</div>
                <div class="confidence-badge">{confidence_percent:.2f}% Confidence</div>
                <div class="taxonomy-text"><b>Domain Taxonomy:</b> {meta['taxonomy']}</div>
                <div style="font-size: 0.9rem; color: #666; margin-top: 0.4rem;">{meta['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📊 Class Probability Breakdown")
        for cls_name, prob in distribution:
            c_meta = CLASS_METADATA.get(cls_name, {"emoji": "🚗"})
            pct = prob * 100.0
            st.markdown(f"**{c_meta['emoji']} {cls_name}**: `{pct:5.2f}%`")
            st.progress(min(max(prob, 0.0), 1.0))

else:
    st.info("👆 Please upload an image, select a preloaded sample, or use the webcam above to see the AI classification in action.")

# ==============================================================================
# Historical Training Curves Expander
# ==============================================================================
metrics_plot = Path("vehicle_classifier_training_metrics.png")
if metrics_plot.exists():
    with st.expander("📈 View Model Training & Validation Performance Curves"):
        st.image(str(metrics_plot), caption="Accuracy and Loss across Transfer Learning (Phase 1) & Fine-Tuning (Phase 2)", use_container_width=True)
