# AI Emotion Recognition System

A deep learning based **Facial Expression Recognition System** built using a hybrid neural architecture combining **DCNN, BiLSTM, and Dual Attention Mechanism**. The system analyzes facial features in real time and predicts human emotions using a trained deep learning model.

This project also includes a **web-based interactive presentation dashboard** explaining the architecture and workflow of the system.

---

## Project Overview

Facial Expression Recognition (FER) is a key component of modern **Human-Computer Interaction (HCI)** systems. This project implements a custom deep learning pipeline capable of identifying human emotions from facial images.

The model integrates multiple deep learning techniques to improve recognition performance under varying conditions such as lighting changes and facial orientation.

---

## Key Features

- Real-time facial expression detection
- Deep learning architecture using **DCNN + BiLSTM + Dual Attention**
- YOLOv8 based face detection
- Interactive web presentation dashboard
- Visualization of system architecture
- Modular pipeline for preprocessing and inference
- Docker-ready for cloud deployment

---

## System Architecture

The pipeline follows this flow:


Input Image / Video
↓
Face Detection (YOLOv8)
↓
Preprocessing (HOG Feature Extraction)
↓
Deep Convolutional Neural Network
↓
Dual Attention Mechanism
↓
Bidirectional LSTM
↓
Softmax Classifier
↓
Emotion Prediction


---

## Technology Stack

### Machine Learning
- Python
- PyTorch
- OpenCV
- NumPy
- Scikit-image

### Deep Learning Models
- DCNN (Deep Convolutional Neural Network)
- BiLSTM (Bidirectional Long Short-Term Memory)
- Dual Attention Mechanism
- YOLOv8 for face detection

### Web Framework
- FastAPI
- HTML / CSS / JavaScript

### Deployment (Planned)
- Docker
- AWS ECS Fargate
- AWS ECR
- Application Load Balancer
- CloudFront
- CloudWatch

---

## Project Structure


AIIO_Expression_Analyzer_Project
│
├── production_app
│ ├── main.py
│ ├── core_model.py
│ ├── requirements.txt
│ ├── models
│ │ ├── best_model_dcnn_dam.pth
│ │ └── dcnn_dam_opt.onnx
│ ├── static
│ │ ├── index.html
│ │ ├── architecture.html
│ │ └── dashboard.html
│
├── Dockerfile
└── README.md


---

## Installation

Clone the repository:


git clone https://github.com/YOUR_USERNAME/REPO_NAME.git

cd REPO_NAME


Create a virtual environment:


python -m venv venv


Activate environment:

Windows:


venv\Scripts\activate


Install dependencies:


pip install -r requirements.txt


---

## Running the Application

Start the FastAPI server:


uvicorn main:app --reload


Open browser:


http://127.0.0.1:8000


---

## Docker Setup

Build Docker image:


docker build -t emotion-ai .


Run container:


docker run -p 8000:8000 emotion-ai


---

## Model Information

Dataset used for training:

**FER-2013**

Total images: **35,887**

Emotion classes:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## Future Improvements

- Deploy model inference using AWS ECS Fargate
- Add scalable architecture using AWS ALB
- Integrate CloudFront CDN
- Improve model accuracy with larger datasets
- Implement real-time video streaming inference

---

## Author

**Sudharsan B**

Computer Science Engineering  
Cloud Computing & AI Enthusiast

---

## License

This project is for educational and research purposes.
