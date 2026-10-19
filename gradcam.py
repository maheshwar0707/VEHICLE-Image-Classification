"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Module
============================================================
Provides visual feature attribution and interpretability heatmaps for MobileNetV3-Large,
revealing the discriminative spatial regions that govern classification decisions.

Mathematical Formulation:
-------------------------
1. Activation Map: A^k in R^(U x V) for channel k in the final conv layer.
2. Target Class Score: y^c for vehicle category c.
3. Feature Importance Weights:
       alpha_k^c = (1 / Z) * sum_i sum_j ( d(y^c) / d(A^k_{i,j}) )
4. Grad-CAM Heatmap:
       L_{Grad-CAM}^c = ReLU( sum_k alpha_k^c * A^k )
"""

from typing import Tuple, Optional
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf


def create_gradcam_subgraph(
    model: tf.keras.Model,
    base_layer_name: str = "MobileNetV3Large",
    conv_layer_name: str = "conv_1",
) -> Tuple[tf.keras.Model, tf.keras.layers.Layer, tf.keras.layers.Layer, tf.keras.layers.Layer]:
    """
    Constructs an intermediate gradient extraction model targeting the final convolutional
    layer of the MobileNetV3-Large feature extractor.

    Args:
        model: Top-level trained Keras model.
        base_layer_name: Layer name of the nested MobileNetV3Large backbone.
        conv_layer_name: Final Conv2D layer name inside MobileNetV3Large ('conv_1').

    Returns:
        sub_base_model: Multi-output Model mapping input tensor to [conv_outputs, backbone_output].
        gap_layer: GlobalAveragePooling2D layer.
        dropout_layer: Dropout layer.
        classifier_layer: Output Dense layer with Softmax.
    """
    base_model = model.get_layer(base_layer_name)
    last_conv_layer = base_model.get_layer(conv_layer_name)

    sub_base_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output],
        name="GradCAM_Backbone_Subgraph",
    )

    gap_layer = model.get_layer("global_average_pool")
    dropout_layer = model.get_layer("bottleneck_dropout")
    classifier_layer = model.get_layer("vehicle_classifier_softmax")

    return sub_base_model, gap_layer, dropout_layer, classifier_layer


def compute_gradcam_heatmap(
    preprocessed_tensor: tf.Tensor,
    sub_base_model: tf.keras.Model,
    gap_layer: tf.keras.layers.Layer,
    dropout_layer: tf.keras.layers.Layer,
    classifier_layer: tf.keras.layers.Layer,
    target_class_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Computes normalized 2D Grad-CAM heatmap via automatic differentiation.

    Args:
        preprocessed_tensor: Input tensor shaped (1, 224, 224, 3) scaled to [-1.0, 1.0].
        sub_base_model: Multi-output backbone model.
        gap_layer: GlobalAveragePooling2D layer.
        dropout_layer: Dropout layer.
        classifier_layer: Output Dense layer.
        target_class_idx: Class index for which gradients are computed. If None, top prediction is used.

    Returns:
        2D numpy array representing the normalized heatmap in range [0.0, 1.0].
    """
    with tf.GradientTape() as tape:
        # Forward pass through backbone
        conv_outputs, base_output = sub_base_model(preprocessed_tensor)
        tape.watch(conv_outputs)

        # Forward pass through top classifier head
        x = gap_layer(base_output)
        x = dropout_layer(x, training=False)
        predictions = classifier_layer(x)

        if target_class_idx is None:
            target_class_idx = tf.argmax(predictions[0])

        class_channel = predictions[:, target_class_idx]

    # Compute gradient of class score with respect to feature maps
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling over spatial dimensions (H, W) -> alpha_k weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Linear combination of channels weighted by alpha_k
    # conv_outputs shape: (1, 7, 7, 960); pooled_grads shape: (960,)
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU activation to retain features positively contributing to the decision
    heatmap = tf.maximum(heatmap, 0.0)

    # Normalize heatmap into [0.0, 1.0]
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    else:
        heatmap = tf.zeros_like(heatmap)

    return heatmap.numpy()


def overlay_gradcam(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet",
) -> Tuple[Image.Image, Image.Image]:
    """
    Resizes, colorizes, and superimposes the 2D activation heatmap over the original RGB image.

    Args:
        original_image: PIL Image in RGB mode (any arbitrary resolution).
        heatmap: 2D numpy array of shape (7, 7) with values in [0, 1].
        alpha: Blending weight for heatmap (0.0 = original image, 1.0 = raw heatmap).
        colormap_name: Matplotlib colormap ('jet', 'turbo', 'inferno', 'viridis').

    Returns:
        superimposed_image: Blended RGB image with Grad-CAM heatmap overlay.
        colored_heatmap_image: Standalone colorized heatmap image.
    """
    orig_rgb = original_image.convert("RGB")
    orig_width, orig_height = orig_rgb.size

    # Resize coarse (7x7) heatmap to original image resolution with bilinear interpolation
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_pil = Image.fromarray(heatmap_uint8).resize((orig_width, orig_height), resample=Image.Resampling.BILINEAR)
    heatmap_resized = np.array(heatmap_pil, dtype=np.float32) / 255.0

    # Colorize using selected colormap
    cmap = plt.get_cmap(colormap_name)
    colored_heatmap = cmap(heatmap_resized)[:, :, :3]  # Strip alpha channel -> (H, W, 3) in [0, 1]
    colored_heatmap_uint8 = np.uint8(255 * colored_heatmap)

    # Alpha blending with original image
    orig_array = np.array(orig_rgb, dtype=np.float32)
    blended = np.uint8(alpha * colored_heatmap_uint8 + (1.0 - alpha) * orig_array)

    return Image.fromarray(blended), Image.fromarray(colored_heatmap_uint8)
