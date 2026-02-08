# 👁️ APEX VISION - Autonomous Traffic Recognition System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-99.7%25-green)

**Apex Vision** is a high-performance computer vision system designed for autonomous vehicle logic. It utilizes a custom Convolutional Neural Network (CNN) trained on the GTSRB dataset to classify 43 distinct traffic sign classes with **99.73% accuracy**, outperforming standard human benchmarks.

## 🚀 Key Features
- **Neural Engine:** Custom CNN architecture trained on 60x60 augmented tensors (Rotation/Zoom/Shear).
- **Command Center:** "Glassmorphism" UI built with Streamlit for real-time inference and telemetry.
- **Decision Logic:** Maps visual classification to actionable vehicle control commands (e.g., "Stop" -> "CRITICAL BRAKE").
- **Live Feed:** Supports real-time webcam inference and static file uploads.

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow, Keras, NumPy
- **Interface:** Streamlit, HTML5/CSS3 (Custom CSS), JavaScript (Canvas Animation)
- **Deployment:** FastAPI Backend with Asynchronous Inference

## 📸 Usage
1. **Backend:** Serves the model via API (`uvicorn backend.main:app --reload`)
2. **Frontend:** Connects to the API for visual feedback (`streamlit run frontend/dashboard.py`)

---
*Developed by Rajwant Ramasubramanian Sarma*