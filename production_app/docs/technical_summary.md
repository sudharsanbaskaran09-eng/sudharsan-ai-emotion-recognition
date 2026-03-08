# Technical System Documentation: DCNN-BiLSTM-DAM

## 1. Overview
The Facial Expression Recognition (FER) system is a high-performance hybrid deep learning solution designed to classify 7 human emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) with real-time feedback.

## 2. Neural Architecture Specs
The system adheres to a specific multi-stage pipeline as required by the architectural constraints:

### A. Feature Extraction (HOG + DCNN)
- **Input:** 64x64 Grayscale images.
- **Preprocessing:** Histogram of Oriented Gradients (HOG) is used to isolate geometric facial contours, making the system resistant to lighting noise.
- **Deep CNN:** 3 Convolutional layers (5x5 kernels) followed by 2 MaxPool layers to extract deep spatial features.

### B. Attention Mechanism (DAM)
- **Spatial Attention:** Identifies "Where to look" (Eyes, Mouth, Eyebrows).
- **Channel Attention:** Identifies "What to look for" (Specific feature relationships).
- **Function:** The Dual Attention Mechanism (DAM) weights important facial regions higher than background artifacts.

### C. Sequential Memory (Bi-LSTM)
- **Logic:** Features are converted into a temporal sequence.
- **Bidirectional Flow:** Processes data in forward and backward directions to capture the full context of facial muscle movement longitudinal transitions.

## 3. Real-Time Integration
- **Backend:** Python FastAPI handles sub-100ms inference.
- **Fallback Logic:** If standard face detection fails, the system utilizes YOLOv8 person-tracking bounding boxes to estimate face locations.
- **Frontend:** A dynamic Javascript-powered dashboard with live bar charts and age estimation (ViT Integration).

## 4. Dataset & Performance
- **Training Source:** FER-2013 (35,887 images).
- **Optimization:** AdamW optimizer with Cosine Annealing learning rate scheduling.
- **Hardware:** Utilizes Apple Silicon (MPS) / NVIDIA (CUDA) for accelerated matrix calculations.
