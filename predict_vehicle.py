"""
Vehicle Classifier Inference Engine
===================================
Load the trained fine-grained vehicle classification model and run prediction
on an arbitrary vehicle image, returning predicted category, confidence, and taxonomy.

Usage:
------
python predict_vehicle.py --image "Vehicles/Cars/sample_001.jpg"
"""

import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger("VehiclePredictor")

# Vehicle Domain Taxonomy Mapping
TAXONOMY_MAP = {
    "Auto Rickshaws": "Three-Wheeler Commercial Transit (L5M/L5N)",
    "Bikes": "Human-Powered Two-Wheeler (Pedal Cycle)",
    "Cars": "Passenger Four-Wheeler (M1)",
    "Motorcycles": "Motorized Two-Wheeler (L3)",
    "Planes": "Aviation Craft (Heavier-than-Air)",
    "Ships": "Marine Watercraft",
    "Trains": "Rail & Track Transit",
}


def predict(image_path: str, model_path: str = "best_vehicle_classifier.keras", classes_path: str = "vehicle_classes.json"):
    import tensorflow as tf

    img_file = Path(image_path)
    if not img_file.exists():
        logger.error("Image not found: %s", image_path)
        sys.exit(1)

    model_file = Path(model_path)
    if not model_file.exists():
        logger.error("Model weights not found at: %s. Train the model first!", model_path)
        sys.exit(1)

    # Load classes
    classes_file = Path(classes_path)
    if classes_file.exists():
        with open(classes_file, "r") as f:
            class_names = json.load(f)
    else:
        class_names = [
            "Auto Rickshaws", "Bikes", "Cars", "Motorcycles", "Planes", "Ships", "Trains"
        ]

    logger.info("Loading trained model from %s...", model_path)
    model = tf.keras.models.load_model(model_path)

    # Preprocess image
    logger.info("Loading image: %s", img_file.name)
    img = Image.open(img_file).convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    input_tensor = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

    # Run forward inference pass
    predictions = model.predict(input_tensor, verbose=0)[0]
    top_idx = int(np.argmax(predictions))
    top_class = class_names[top_idx]
    top_confidence = float(predictions[top_idx] * 100.0)
    taxonomy = TAXONOMY_MAP.get(top_class, "Vehicle")

    print("\n" + "=" * 50)
    print(f"  IMAGE:            {img_file.name}")
    print(f"  PREDICTED CLASS:  {top_class}")
    print(f"  CONFIDENCE:       {top_confidence:.2f}%")
    print(f"  DOMAIN TAXONOMY:  {taxonomy}")
    print("=" * 50)

    print("\nClass Probabilities:")
    for idx, prob in sorted(enumerate(predictions), key=lambda x: x[1], reverse=True):
        print(f"  - {class_names[idx]:<18}: {prob * 100.0:6.2f}%")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify vehicle image using trained model.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file.")
    parser.add_argument("--model", type=str, default="best_vehicle_classifier.keras", help="Path to .keras model.")
    parser.add_argument("--classes", type=str, default="vehicle_classes.json", help="Path to class labels json.")
    args = parser.parse_args()

    predict(args.image, args.model, args.classes)
