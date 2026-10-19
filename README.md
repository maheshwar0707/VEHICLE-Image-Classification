# Fine-Grained Vehicle Classification Using Deep Convolutional Neural Networks & Explainable Grad-CAM

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Validation Accuracy](https://img.shields.io/badge/Val%20Accuracy-99.02%25-059669?style=flat-square)](#benchmark-performance)
[![Inference Latency](https://img.shields.io/badge/Latency-~50ms-2563EB?style=flat-square)](#benchmark-performance)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end, production-grade **Machine Learning and Computer Vision** system designed to perform fine-grained vehicle classification across 7 target transportation categories. Powered by a two-stage fine-tuned **MobileNetV3-Large** convolutional neural network, the system integrates **Gradient-weighted Class Activation Mapping (Grad-CAM)** for transparent visual feature attribution and runs locally in real time via an interactive **Streamlit** dashboard. 
Here is the live streamlit app link : https://vehicle-image-classification.streamlit.app/    CHECK IT OUT

---

## 📑 Table of Contents
- [Project Overview](#project-overview)
- [Benchmark Performance](#benchmark-performance)
- [End-to-End System Architecture](#end-to-end-system-architecture)
- [Mathematical Formulation](#mathematical-formulation)
  - [Categorical Cross-Entropy Loss](#1-categorical-cross-entropy-loss)
  - [Grad-CAM Feature Attribution](#2-grad-cam-feature-attribution)
- [Dataset Specifications](#dataset-specifications)
- [Two-Stage Fine-Tuning Methodology](#two-stage-fine-tuning-methodology)
- [Repository Structure](#repository-structure)
- [Installation & Quickstart](#installation--quickstart)
  - [Prerequisites](#prerequisites)
  - [Setup Environment](#setup-environment)
  - [Launch Interactive Dashboard](#launch-interactive-dashboard)
  - [Run Command-Line Inference](#run-command-line-inference)
  - [Train Model from Scratch](#train-model-from-scratch)
- [Explainability & Interpretability (Grad-CAM)](#explainability--interpretability-grad-cam)
- [Technical Stack](#technical-stack)
- [License](#license)

---

## 🔬 Project Overview

In transportation analytics, traffic surveillance, and intelligent routing systems, recognizing specific vehicle types under variable lighting, perspective distortion, and background clutter is a challenging computer vision task. 

This project implements a complete deep learning workflow:
1. **High-Throughput Data Pipeline (`tf.data`):** Automated stratification, cached pipeline execution, and GPU prefetching.
2. **Inverted Residual Backbone (MobileNetV3-Large):** Utilizing depthwise separable convolutions, Squeeze-and-Excitation (SE) attention blocks, and Hard-Swish activations to deliver high representational capacity with minimal parameter footprint (~4.2M parameters).
3. **Two-Stage Transfer Learning:** Linear probing of the classification head followed by selective backbone fine-tuning.
4. **Visual Interpretability (Grad-CAM):** Extracting backward gradients from target class activations to convolutional feature maps (`conv_1`) to verify that classification decisions are anchored on salient structural features (e.g., wheel assemblies, pantographs, wings, hydrodynamic hulls).
5. **Local Edge-Ready Deployment:** Completely self-contained `.keras` model executing on commodity hardware with zero external cloud dependencies.

---

## 📊 Benchmark Performance

The model was evaluated on a dedicated, stratified hold-out validation dataset (20% split) consisting of 1,118 unseen vehicle images:

| Evaluation Metric | Score | Detail / Significance |
| :--- | :--- | :--- |
| **Validation Accuracy** | **99.02%** | Correctly classified 1,107 out of 1,118 hold-out validation samples |
| **Validation Loss** | **0.0431** | Sparse Categorical Cross-Entropy loss |
| **Training Accuracy** | **98.86%** | Final epoch training accuracy across 4,472 samples |
| **Training Loss** | **0.0382** | Monotonically converged cross-entropy objective |
| **Inference Latency** | **40–60 ms** | Forward-pass latency on standard quad-core CPU |
| **Model Disk Footprint** | **20.4 MB** | Checkpointed `.keras` weights |
| **Parameter Count** | **~4.22 Million** | High parameter efficiency via depthwise separable convolutions |

### Convergence Trajectory
```
Phase 1 (Epochs 1–5: Top-Head Warmup | lr = 1e-3)
Epoch 1: Val Accuracy = 97.31% | Val Loss = 0.1084
Epoch 2: Val Accuracy = 98.21% | Val Loss = 0.0695
Epoch 3: Val Accuracy = 98.48% | Val Loss = 0.0641
Epoch 4: Val Accuracy = 98.84% | Val Loss = 0.0534
Epoch 5: Val Accuracy = 98.84% | Val Loss = 0.0497

Phase 2 (Epochs 6–12: Backbone Fine-Tuning | lr = 1e-5)
Epoch 6:  Val Accuracy = 99.10% | Val Loss = 0.0453
Epoch 7:  Val Accuracy = 99.10% | Val Loss = 0.0441
Epoch 8:  Val Accuracy = 99.10% | Val Loss = 0.0437
Epoch 9:  Val Accuracy = 99.02% | Val Loss = 0.0439
Epoch 10: Val Accuracy = 99.02% | Val Loss = 0.0436
Epoch 11: Val Accuracy = 99.02% | Val Loss = 0.0435
Epoch 12: Val Accuracy = 99.02% | Val Loss = 0.0431 (★ Optimal Checkpoint Saved)
```

---

## 🏛️ End-to-End System Architecture

```
                              RAW INPUT IMAGE
                        (Arbitrary Dimensions / Formats)
                                      │
                                      ▼
                       PREPROCESSING & STANDARDIZATION
                        • Bilinear Resizing: (224 × 224 × 3)
                        • Range Normalization: [-1.0, 1.0]
                                      │
                                      ▼
                      DATA AUGMENTATION PIPELINE (Training)
                        • Random Horizontal Flip
                        • Random Rotation (±10%)
                        • Random Zoom (±10%)
                                      │
                                      ▼
                 FEATURE EXTRACTION BACKBONE (MobileNetV3-Large)
                        • Inverted Bottleneck Residual Blocks
                        • Depthwise Separable Convolutions (3×3 & 5×5)
                        • Squeeze-and-Excitation (SE) Channel Attention
                        • Non-Linearity: Hard-Swish (h-swish)
                        • Final Conv Layer: conv_1 ──┐ (Grad-CAM Subgraph)
                                      │              │
                                      ▼              │
                          SPATIAL FEATURE MAP TENSOR │
                               (7 × 7 × 960)         │
                                      │              │
                                      ▼              │
                         GLOBAL AVERAGE POOLING (GAP)│
                                (960-d Vector)       │
                                      │              │
                                      ▼              │
                           REGULARIZATION LAYER      │
                            • Dropout (p = 0.20)     │
                                      │              │
                                      ▼              │
                           DENSE CLASSIFICATION HEAD │
                            • Dense Units: 7         │
                            • Activation: Softmax    │
                                      │              │
                                      ▼              ▼
                          PREDICTED CLASS & PROBABILITY   GRAD-CAM ATTRIBUTION
                            [ p_1, p_2, ..., p_7 ]        Backpropagated Gradients
                                      │                   Layer conv_1 -> Heatmap
                                      ▼                              │
                               INTERACTIVE UI ◄──────────────────────┘
                          (Streamlit Dashboard)
```

---

## 📐 Mathematical Formulation

### 1. Categorical Cross-Entropy Loss
For a mini-batch with true class labels $y \in \{1, \dots, C\}$ and predicted softmax class probabilities $\hat{y}_c$:

$$\mathcal{L}_{\text{CE}} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c) = -\log(\hat{y}_{c^*})$$

where $c^*$ represents the ground-truth vehicle category index, and predicted probabilities are obtained via the normalized exponential function:

$$\hat{y}_c = \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}}$$

### 2. Grad-CAM Feature Attribution
To explain the spatial regions influencing the prediction score $y^c$ for class $c$, gradients are computed with respect to activation map $A^k$ of channel $k$ in the final convolutional layer `conv_1` ($U \times V = 7 \times 7$):

$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{U} \sum_{j=1}^{V} \frac{\partial y^c}{\partial A_{i,j}^k}$$

where $Z = U \times V$ represents the spatial dimensions of the feature map. The final discriminative localization heatmap is calculated as a rectified linear combination of forward activation maps:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k=1}^{K} \alpha_k^c A^k \right)$$

The $\text{ReLU}$ operator suppresses features contributing negatively to class $c$, isolating exclusively the visual cues that support the classification.

---

## 📦 Dataset Specifications

The dataset comprises **5,590 high-resolution vehicle images** partitioned into 7 distinct ground-truth categories:

| Class Name | Domain Description | Visual Distinguishing Features | Sample Count |
| :--- | :--- | :--- | :--- |
| **Auto Rickshaws** | Urban 3-Wheeled Commercial Transit | 3-wheel delta geometry, open cabin frame, front handlebars | ~800 |
| **Bikes** | Human-Powered 2-Wheel Cycles | Tubular diamond frame, thin wheel spokes, pedals, drive chain | ~800 |
| **Cars** | 4-Wheeled Passenger Automobiles | Front radiator grille, dual headlamps, cabin windows, 4 wheels | ~800 |
| **Motorcycles** | Motorized 2-Wheel Vehicles | Exposed engine block, fuel tank contours, exhaust assembly | ~800 |
| **Planes** | Aviation Aircraft | Airfoil wing spans, jet turbofans/propellers, empennage tail | ~800 |
| **Ships** | Marine Watercraft | Hydrodynamic hull, container stacks, deck superstructure | ~800 |
| **Trains** | Rail Locomotives & Transit Cars | Parallel steel rail tracks, articulated cars, locomotive nose | ~790 |

*Data Partitioning:* Stratified 80% training split (4,472 images) and 20% validation split (1,118 images).

---

## ⚙️ Two-Stage Fine-Tuning Methodology

Transfer learning was executed using an intentional two-stage schedule to mitigate catastrophic forgetting:

```
Stage 1: Classification Head Warmup (Epochs 1–5)
├── MobileNetV3-Large Feature Extractor: FROZEN (weights non-trainable)
├── Dense Head + Dropout: TRAINABLE
├── Learning Rate: 1e-3 (Adam Optimizer)
└── Objective: Stabilize randomly initialized top weights without perturbing backbone features.

Stage 2: Selective Backbone Fine-Tuning (Epochs 6–12)
├── MobileNetV3-Large Upper Layers: UNFROZEN (trainable)
├── BatchNormalization Layers: FROZEN (prevents running mean/variance drift)
├── Learning Rate: 1e-5 (Small step size for gentle adaptation)
└── Objective: Fine-tune deep spatial filters to vehicle domain textures and silhouettes.
```

---

## 📂 Repository Structure

```
image-classification/
├── app.py                                  # Main Streamlit web application & interactive dashboard
├── best_vehicle_classifier.keras           # Trained deep neural network weights checkpoint (20.4 MB)
├── gradcam.py                              # Grad-CAM automatic differentiation & heatmap generator
├── predict_vehicle.py                      # Standalone CLI inference script
├── vehicle_classifier.py                   # Complete training pipeline & data ingestion script
├── vehicle_classes.json                    # Class labels JSON dictionary (7 categories)
├── vehicle_classifier_training_metrics.png # Matplotlib validation accuracy & loss curves
├── run_app.bat                             # One-click Windows batch launcher
├── README.md                               # Project documentation & architectural manual
├── .gitignore                              # Git version control ignore rules
└── Vehicles/                               # Image dataset directory
    ├── Auto Rickshaws/
    ├── Bikes/
    ├── Cars/
    ├── Motorcycles/
    ├── Planes/
    ├── Ships/
    └── Trains/
```

---

## 🚀 Installation & Quickstart

### Prerequisites
* Python 3.10, 3.11, 3.12, 3.13, or 3.14
* Recommended: Virtual environment (`venv` or `conda`)

### Setup Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/vehicle-image-classification.git
   cd vehicle-image-classification
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install tensorflow numpy pillow streamlit matplotlib plotly
   ```

---

### Launch Interactive Dashboard

You can launch the web application with a single command:

```bash
# Using Streamlit
streamlit run app.py
```

*Or on Windows, simply double-click:*  
`run_app.bat`

Once started, navigate to `http://localhost:8501` in your web browser.

---

### Run Command-Line Inference

To run predictions directly from the command line without launching the UI:

```bash
# Predict on an arbitrary image
python predict_vehicle.py --image "Vehicles/Cars/sample_001.jpg"
```

**Sample Output:**
```
==================================================
  IMAGE:            Auto Rickshaw (12).jpg
  PREDICTED CLASS:  Auto Rickshaws
  CONFIDENCE:       99.98%
  DOMAIN TAXONOMY:  Three-Wheeler Commercial Transit (L5M/L5N)
==================================================

Class Probabilities:
  - Auto Rickshaws    :  99.98%
  - Motorcycles       :   0.01%
  - Cars              :   0.01%
  - Bikes             :   0.00%
  - Trains            :   0.00%
  - Planes            :   0.00%
  - Ships             :   0.00%
==================================================
```

---

### Train Model from Scratch

To retrain the complete two-stage transfer learning pipeline:

```bash
python vehicle_classifier.py --data-dir "Vehicles" --batch-size 32 --phase1-epochs 5 --phase2-epochs 7
```

This will automatically:
1. Detect and stratify the dataset (80/20 train/val).
2. Train Phase 1 (top head) and Phase 2 (fine-tuning).
3. Save the optimal weights checkpoint to `best_vehicle_classifier.keras`.
4. Generate and save training convergence curves to `vehicle_classifier_training_metrics.png`.

---

## 🔍 Explainability & Interpretability (Grad-CAM)

Standard deep convolutional networks are often critiqued as opaque black-box models. To provide full transparency for academic and industrial verification, this project embeds **Grad-CAM**:

* **Feature Targeting:** Targets layer `conv_1` of the MobileNetV3-Large sub-graph, outputting 960 channels of spatial activation maps.
* **Heatmap Calibration:** Normalizes gradients to $[0.0, 1.0]$ and overlays via perceptual colormaps (`turbo`, `jet`, `inferno`, `viridis`).
* **Interactive Adjustments:** The dashboard provides real-time blending opacity sliders ($\alpha \in [0.1, 0.8]$) to inspect exact vehicle structural boundaries.

---

## 🛠️ Technical Stack

* **Core Framework:** [TensorFlow 2.x](https://tensorflow.org) & [Keras](https://keras.io)
* **Backbone Network:** MobileNetV3-Large (ImageNet Pre-trained)
* **Explainability Module:** Grad-CAM via TensorFlow Automatic Differentiation (`tf.GradientTape`)
* **Visualization Engine:** [Plotly](https://plotly.com/) & [Matplotlib](https://matplotlib.org/)
* **Application Framework:** [Streamlit](https://streamlit.io/)
* **Image Processing:** [Pillow (PIL)](https://pillow.readthedocs.io/) & [NumPy](https://numpy.org/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
