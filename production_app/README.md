# Facial Expression Recognition System (DCNN-BiLSTM-DAM)

This is a state-of-the-art implementation of the **DCNN-BiLSTM-DAM** architecture for Facial Expression Recognition (FER), as specified in the project requirements.

## 📁 Project Structure
- `main.py`: The core FastAPI backend server and real-time inference engine.
- `core_model.py`: The DCNN-BiLSTM-DAM Neural Network architecture (PyTorch).
- `train.py`: The automated training script that fetches the FER-2013 dataset and fine-tunes the model.
- `/static`: Premium web-based dashboard and presentation assets.
- `/models`: Contains the trained weights (`.pth`) and exported (`.onnx`) versions.
- `/docs`: Project paperwork, base paper, and requirement forms.
- `/tools`: Auxiliary scripts for testing and exporting models.

## 🚀 Getting Started

### 1. Training the Model
The system is pre-configured to automatically download and train on the **FER-2013** dataset (35,000+ images) via KaggleHub.
```bash
python3 train.py
```
*The model will automatically save the best weights to `models/best_model_dcnn_dam.pth`.*

### 2. Running the Live Dashboard
Access the application at: **http://127.0.0.1:8005**

#### 🐧 macOS / Linux
Starts the production server:
```bash
python3 -m uvicorn main:app --port 8005 --reload
```

#### 💻 Windows (One-Click)
Simply double-click the `run_windows.bat` file. 
- It will automatically create a virtual environment (`venv`).
- Installs all dependencies from `requirements.txt`.
- Launches the server and dashboard immediately.

### 3. Viewing the Presentation
Open the interactive landing page at: **http://127.0.0.1:8005**


## 💡 Tech Stack
- **Backend:** FastAPI, PyTorch (DCNN-BiLSTM-DAM Architecture).
- **Computer Vision:** YOLOv8 (Tracking), HOG (Feature Extraction), OpenCV.
- **Frontend:** Vanilla HTML5, CSS3 (Glassmorphism), JavaScript (Particle Engine).
- **Environment:** Compatible with macOS (MPS) and Windows (CUDA) acceleration.
