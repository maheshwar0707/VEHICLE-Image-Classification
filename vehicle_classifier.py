"""
Production-Ready Fine-Grained Vehicle Classifier
================================================
Architecture: MobileNetV3Large (ImageNet) Transfer Learning with Two-Stage Fine-Tuning
Framework:    TensorFlow 2.x / tf.keras
Dataset:      Vehicles (Auto Rickshaws, Bikes, Cars, Motorcycles, Planes, Ships, Trains, etc.)

Supports both:
  1. Pre-split datasets:
     dataset_root/
       ├── train/
       └── validation/
  2. Single-directory datasets (automatically split 80/20 into train/validation):
     dataset_root/
       ├── Class_A/
       ├── Class_B/
       └── ...
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==============================================================================
# 0. System Configuration & Logging Setup
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("VehicleClassifier")

# Deterministic reproducibility
tf.keras.utils.set_random_seed(42)

# GPU Memory Growth Configuration (Prevents OOM on NVIDIA GPUs)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("GPU detected and memory growth enabled: %s", gpus)
    except RuntimeError as e:
        logger.warning("Could not set GPU memory growth: %s", e)
else:
    logger.info("No GPU detected, running on CPU.")

# Default Hyperparameters
IMAGE_DIMS: Tuple[int, int] = (224, 224)
INPUT_SHAPE: Tuple[int, int, int] = (224, 224, 3)
BATCH_SIZE: int = 32

PHASE1_EPOCHS: int = 5
PHASE2_EPOCHS: int = 7
TOTAL_EPOCHS: int = PHASE1_EPOCHS + PHASE2_EPOCHS

PHASE1_LR: float = 1e-3
PHASE2_LR: float = 1e-5


# ==============================================================================
# 1. High-Throughput Data Ingestion Pipeline (tf.data)
# ==============================================================================
def load_datasets(
    dataset_root: Path,
    image_size: Tuple[int, int] = IMAGE_DIMS,
    batch_size: int = BATCH_SIZE,
    val_split: float = 0.2,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Constructs high-throughput tf.data pipelines for training and validation.
    Dynamically supports both pre-split (train/val) directories and single-folder
    class-structured directories with automated stratified splitting.
    """
    dataset_root = Path(dataset_root).resolve()
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "validation"

    autotune = tf.data.AUTOTUNE

    if train_dir.exists() and val_dir.exists():
        logger.info("Detected pre-split dataset directories: %s and %s", train_dir, val_dir)
        train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=str(train_dir),
            labels="inferred",
            label_mode="int",
            color_mode="rgb",
            batch_size=batch_size,
            image_size=image_size,
            shuffle=True,
            seed=42,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            directory=str(val_dir),
            labels="inferred",
            label_mode="int",
            color_mode="rgb",
            batch_size=batch_size,
            image_size=image_size,
            shuffle=False,
        )
    else:
        logger.info("Dataset at '%s' has class directories. Splitting into Train (80%%) and Val (20%%)...", dataset_root)
        train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=str(dataset_root),
            labels="inferred",
            label_mode="int",
            validation_split=val_split,
            subset="training",
            seed=42,
            color_mode="rgb",
            batch_size=batch_size,
            image_size=image_size,
            shuffle=True,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            directory=str(dataset_root),
            labels="inferred",
            label_mode="int",
            validation_split=val_split,
            subset="validation",
            seed=42,
            color_mode="rgb",
            batch_size=batch_size,
            image_size=image_size,
            shuffle=True,
        )

    class_names = train_ds.class_names
    logger.info("Identified %d classes: %s", len(class_names), class_names)

    # High-Throughput Hardware Pipeline Acceleration
    train_ds = (
        train_ds.cache()
        .shuffle(buffer_size=1000)
        .prefetch(buffer_size=autotune)
    )

    val_ds = (
        val_ds.cache()
        .prefetch(buffer_size=autotune)
    )

    return train_ds, val_ds, class_names


# ==============================================================================
# 2. Deep Transfer Learning Network Architecture
# ==============================================================================
def build_vehicle_classifier(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = 7,
) -> Tuple[keras.Model, keras.Model]:
    """
    Constructs the Fine-Grained Vehicle Classifier with explicit tensor flows.

    Tensor Pipeline Flow:
    ---------------------
    1. Input Tensor:              (None, 224, 224, 3)
    2. Data Augmentation Stage:   (None, 224, 224, 3) [Flip, Rotation 0.15, Zoom 0.1]
    3. MobileNetV3 Preprocessing: (None, 224, 224, 3) [Scales to [-1, 1]]
    4. MobileNetV3Large Base:     (None, 7, 7, 960)   [include_top=False, ImageNet weights]
    5. GlobalAveragePooling2D:    (None, 960)         [Spatial reduction over 7x7 grid]
    6. Regularization Dropout:    (None, 960)         [Rate 0.3]
    7. Classification Head:       (None, num_classes) [Dense with Softmax activation]
    """
    logger.info("Constructing Fine-Grained Vehicle Classifier Architecture for %d classes...", num_classes)

    # [1] Input Layer
    inputs = layers.Input(shape=input_shape, name="input_tensor")

    # [2] Data Augmentation Pipeline
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal", name="aug_random_flip"),
            layers.RandomRotation(factor=0.15, name="aug_random_rotation"),
            layers.RandomZoom(height_factor=0.1, width_factor=0.1, name="aug_random_zoom"),
        ],
        name="data_augmentation_pipeline",
    )
    x = data_augmentation(inputs)

    # [3] Preprocessing Stage
    x = layers.Lambda(
        tf.keras.applications.mobilenet_v3.preprocess_input,
        name="mobilenet_v3_preprocessing",
    )(x)

    # [4] Feature Extraction Base (MobileNetV3Large)
    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling=None,
    )
    base_model.trainable = False

    # Freeze BatchNormalization running mean and variance stats
    x = base_model(x, training=False)

    # [5] Bottleneck Flattening
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)

    # [6] Regularization Layer
    x = layers.Dropout(rate=0.3, name="bottleneck_dropout")(x)

    # [7] Output Classifier Head
    outputs = layers.Dense(
        units=num_classes,
        activation="softmax",
        dtype="float32",
        name="vehicle_classifier_softmax",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Vehicle_Classifier_MobileNetV3")

    logger.info("Model Summary:")
    model.summary(print_fn=logger.info)

    return model, base_model


# ==============================================================================
# 3. Production Training Callbacks Factory
# ==============================================================================
def create_callbacks(checkpoint_path: str = "best_vehicle_classifier.keras") -> List[keras.callbacks.Callback]:
    """Configures EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint."""
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
    ]


# ==============================================================================
# 4. Two-Stage Training Protocol
# ==============================================================================
def train_two_stage(
    model: keras.Model,
    base_model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    phase1_epochs: int = PHASE1_EPOCHS,
    phase2_epochs: int = PHASE2_EPOCHS,
    phase1_lr: float = PHASE1_LR,
    phase2_lr: float = PHASE2_LR,
) -> Tuple[keras.callbacks.History, keras.callbacks.History]:
    """
    Executes:
      - Phase 1: Feature Extraction (Frozen Base) for `phase1_epochs`.
      - Phase 2: Fine-Tuning Top 20 Layers of MobileNetV3 for `phase2_epochs`.
    """
    callbacks = create_callbacks()

    # -------------------------------------------------------------------------
    # PHASE 1: Train Top Classifier Head
    # -------------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("PHASE 1: Feature Extraction (Frozen Base) | Epochs: 1 to %d", phase1_epochs)
    logger.info("Learning Rate: %e | Loss: SparseCategoricalCrossentropy", phase1_lr)
    logger.info("=" * 70)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=phase1_lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=callbacks,
    )

    # -------------------------------------------------------------------------
    # PHASE 2: Fine-Tuning Top 20 Layers of Base Model
    # -------------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("PHASE 2: Fine-Tuning Base Model (Top 20 Layers)")
    logger.info("=" * 70)

    base_model.trainable = True
    num_base_layers = len(base_model.layers)
    unfreeze_cutoff = max(0, num_base_layers - 20)

    for idx, layer in enumerate(base_model.layers):
        if idx < unfreeze_cutoff:
            layer.trainable = False
        else:
            layer.trainable = True

    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_count = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    logger.info("Total Layers: %d | Unfrozen Top Layers: 20", num_base_layers)
    logger.info("Trainable Parameters: %s | Non-Trainable Parameters: %s", f"{trainable_count:,}", f"{non_trainable_count:,}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=phase2_lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    total_target_epochs = phase1_epochs + phase2_epochs
    last_phase1_epoch = history_phase1.epoch[-1] if history_phase1.epoch else (phase1_epochs - 1)
    initial_phase2_epoch = last_phase1_epoch + 1

    logger.info("Resuming Fine-Tuning: Epoch %d -> %d | Learning Rate: %e", initial_phase2_epoch + 1, total_target_epochs, phase2_lr)

    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_target_epochs,
        initial_epoch=initial_phase2_epoch,
        callbacks=callbacks,
    )

    return history_phase1, history_phase2


# ==============================================================================
# 5. Visualization & Metric Plotting
# ==============================================================================
def plot_training_history(
    history_phase1: keras.callbacks.History,
    history_phase2: keras.callbacks.History,
    output_filepath: str = "vehicle_classifier_training_metrics.png",
) -> None:
    """Concatenates metrics across phases and saves publication-quality plots."""
    logger.info("Plotting evaluation metrics...")

    acc = history_phase1.history.get("accuracy", []) + history_phase2.history.get("accuracy", [])
    val_acc = history_phase1.history.get("val_accuracy", []) + history_phase2.history.get("val_accuracy", [])
    loss = history_phase1.history.get("loss", []) + history_phase2.history.get("loss", [])
    val_loss = history_phase1.history.get("val_loss", []) + history_phase2.history.get("val_loss", [])

    total_epochs_run = len(acc)
    epochs_range = range(1, total_epochs_run + 1)
    phase1_end_epoch = len(history_phase1.history.get("loss", []))

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Subplot 1: Accuracy
    ax1 = axes[0]
    ax1.plot(epochs_range, acc, label="Train Accuracy", color="#1f77b4", linewidth=2.5, marker="o")
    ax1.plot(epochs_range, val_acc, label="Val Accuracy", color="#ff7f0e", linewidth=2.5, marker="s")
    if phase1_end_epoch < total_epochs_run:
        ax1.axvline(x=phase1_end_epoch + 0.5, color="#d62728", linestyle="--", linewidth=1.8, label="Start Fine-Tuning")
    ax1.set_title("Vehicle Classifier: Model Accuracy", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Accuracy", fontsize=11, fontweight="semibold")
    ax1.set_ylim([0.0, 1.02])
    ax1.legend(loc="lower right", frameon=True, fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Loss
    ax2 = axes[1]
    ax2.plot(epochs_range, loss, label="Train Loss", color="#1f77b4", linewidth=2.5, marker="o")
    ax2.plot(epochs_range, val_loss, label="Val Loss", color="#ff7f0e", linewidth=2.5, marker="s")
    if phase1_end_epoch < total_epochs_run:
        ax2.axvline(x=phase1_end_epoch + 0.5, color="#d62728", linestyle="--", linewidth=1.8, label="Start Fine-Tuning")
    ax2.set_title("Vehicle Classifier: Sparse Categorical Cross-Entropy Loss", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Loss", fontsize=11, fontweight="semibold")
    ax2.legend(loc="upper right", frameon=True, fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_filepath, bbox_inches="tight")
    logger.info("Evaluation plots saved to: %s", output_filepath)
    plt.close(fig)


# ==============================================================================
# 6. Sample Inference Validation
# ==============================================================================
def run_sample_inference(model: keras.Model, val_ds: tf.data.Dataset, class_names: List[str], num_samples: int = 5) -> None:
    """Demonstrates inference on validation samples with prediction confidences."""
    logger.info("Running sample inference validation...")
    for images, labels in val_ds.take(1):
        preds = model.predict(images[:num_samples], verbose=0)
        for i in range(min(num_samples, len(images))):
            true_label = class_names[labels[i].numpy()]
            pred_idx = np.argmax(preds[i])
            pred_label = class_names[pred_idx]
            confidence = preds[i][pred_idx] * 100.0
            logger.info("Sample #%d -> True: %-15s | Predicted: %-15s | Confidence: %.2f%%", i + 1, true_label, pred_label, confidence)


# ==============================================================================
# 7. Main Execution Flow
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Production Fine-Grained Vehicle Classifier.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Root path containing vehicle images or train/val splits.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        default=PHASE1_EPOCHS,
        help="Epochs for top classifier head training.",
    )
    parser.add_argument(
        "--phase2-epochs",
        type=int,
        default=PHASE2_EPOCHS,
        help="Epochs for backbone fine-tuning.",
    )

    args = parser.parse_args()

    # Automatically resolve dataset directory if not provided
    if args.data_dir:
        dataset_path = Path(args.data_dir).resolve()
    else:
        for candidate in ["Vehicles", "vehichle", "dataset"]:
            p = Path(candidate).resolve()
            if p.exists() and p.is_dir():
                dataset_path = p
                break
        else:
            raise FileNotFoundError("Could not find dataset directory. Please specify with --data-dir.")

    logger.info("Using dataset directory: %s", dataset_path)

    # Ingest Dataset
    train_ds, val_ds, class_names = load_datasets(
        dataset_root=dataset_path,
        image_size=IMAGE_DIMS,
        batch_size=args.batch_size,
    )

    # Save class names mapping for production deployment
    with open("vehicle_classes.json", "w") as f:
        json.dump(class_names, f, indent=2)
    logger.info("Exported class labels to 'vehicle_classes.json'.")

    # Build Architecture
    model, base_model = build_vehicle_classifier(
        input_shape=INPUT_SHAPE,
        num_classes=len(class_names),
    )

    # Train Model
    history_phase1, history_phase2 = train_two_stage(
        model=model,
        base_model=base_model,
        train_ds=train_ds,
        val_ds=val_ds,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase1_lr=PHASE1_LR,
        phase2_lr=PHASE2_LR,
    )

    # Plot Combined Curves
    plot_training_history(
        history_phase1=history_phase1,
        history_phase2=history_phase2,
        output_filepath="vehicle_classifier_training_metrics.png",
    )

    # Run Sample Predictions
    run_sample_inference(model, val_ds, class_names, num_samples=5)

    logger.info("Complete pipeline finished successfully.")


if __name__ == "__main__":
    main()
