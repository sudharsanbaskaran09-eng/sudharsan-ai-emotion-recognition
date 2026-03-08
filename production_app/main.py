import os
import io
import cv2
import torch
import base64
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from skimage.feature import hog

# Core Inference Engine
from core_model import DCNN_BiLSTM_DAM

# Next-Gen AI Integration (YOLOv8 + Transformers)
from ultralytics import YOLO
from transformers import pipeline

# Initialize Production App
app = FastAPI(title="Facial Expression Analyzer AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# =======================================================================
# 1. LOAD MODELS (EMOTION, OBJECT DETECTION, AGE, AND PROFILE TRACKING)
# =======================================================================

# 1. EMOTION TRACKER (DCNN-BiLSTM-DAM with HOG Preprocessing)
print("Loading DCNN-BiLSTM-DAM for Facial Expression Recognition...")
model = DCNN_BiLSTM_DAM(num_classes=7)
weights_path = "./models/best_model_dcnn_dam.pth"
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    print("✅ Loaded Custom Emotion Weights!")
else:
    print("Warning: No runtime PyTorch emotion weights found. Generating initials.")
model.to(DEVICE)
model.eval()

# Load YOLOv8 for Crowd/Object/Animal Tracking (Downloads nano weights automatically!)
yolo_model = YOLO('yolov8n.pt') 

# Load Age Prediction ViT Transformer natively onto GPU/MPS Memory
age_classifier = pipeline("image-classification", model="nateraw/vit-age-classifier", device=DEVICE)

# Load OpenCV Cascades (Frontal + Side Profile Tracking)
face_cascade_front = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# FER-2013 Output labels from ImageFolder layout (Alphabetical sorting)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# =======================================================================
# 2. GRAD-CAM (EXPLAINABLE AI HEATMAP GENERATOR)
# =======================================================================
# Grad-CAM Heatmap has been removed as per user request.

# =======================================================================
# 3. CORE AI INFERENCE PIPELINE
# =======================================================================
def get_faces(gray_image):
    # Pass 1: Scan Frontal
    faces_front = list(face_cascade_front.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48)))
    # Pass 2: Scan Side Profile
    faces_profile = list(face_cascade_profile.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48)))
    
    # Merge and simple deduplicate boxes (Overlap check)
    all_faces = faces_front + faces_profile
    return all_faces

def predict_single_face(gray_crop, color_crop):
    # Process for ViT Age Prediction first (Transformers require PIL format)
    # We do this before HOG modifies our color crop sizes
    pil_img = Image.fromarray(cv2.cvtColor(color_crop, cv2.COLOR_BGR2RGB))
    age_results = age_classifier(pil_img)
    predicted_age = age_results[0]['label'] # e.g. "20-29"

    # Convert face to highly accurate HOG Geometry for DCNN Architecture
    resized_gray = cv2.resize(gray_crop, (64, 64)) 
    
    # Extract HOG structures matching training pipeline exactly
    _, hog_img = hog(resized_gray, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    hog_normalized = hog_img / (np.max(hog_img) + 1e-5)
    
    tensor = torch.tensor(hog_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze().tolist()
    
    if len(np.array(probabilities).shape) > 1:
        probabilities = probabilities[0]
        
    max_idx = np.argmax(probabilities)
    emotion = EMOTIONS[max_idx]
    confidence = probabilities[max_idx] * 100
    scores = {EMOTIONS[i]: round(probabilities[i]*100, 2) for i in range(len(EMOTIONS))}
    
    return emotion, confidence, scores, predicted_age

@app.post("/api/predict")
async def analyze_crowd_and_objects(file: UploadFile = File(...)):
    """ Main Upload Inference Endpoint (YOLO + Emotion + Age + GradCAM) """
    try:
        img_bytes = await file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        results = {"faces": [], "objects": [], "counts": {"people": 0, "animals": 0, "other": 0}}
        
        # 1. YOLOv8 Pass (Detect everything)
        yolo_res = yolo_model(img_cv, verbose=False)[0]
        for box in yolo_res.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            class_name = yolo_model.names[int(box.cls[0])]
            conf = float(box.conf[0]) * 100
            
            # YOLO distinguishes people and animals
            if class_name == 'person':
                results["counts"]["people"] += 1
            elif class_name in ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'bear']:
                results["counts"]["animals"] += 1
            else:
                results["counts"]["other"] += 1
                
            results["objects"].append({
                "box": [int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])],
                "label": class_name,
                "confidence": conf
            })
            
        # 2. OpenCV Face + Age + Emotion Pass
        faces = get_faces(gray)
        
        # Fallback: If Haar Cascades fail but YOLO found people, use the top 35% of those boxes as face candidates
        if not faces:
            for box in yolo_res.boxes:
                b = box.xyxy[0].cpu().numpy().astype(int)
                if yolo_model.names[int(box.cls[0])] == 'person':
                    # Estimate face from person box (Top 35%)
                    w = b[2] - b[0]
                    h_full = b[3] - b[1]
                    h = int(h_full * 0.35)
                    faces.append([int(b[0]), int(b[1]), int(w), int(h)])

        for (x, y, w, h) in faces:
            # Ensure coordinates are within image boundaries
            x, y = max(0, x), max(0, y)
            w = min(w, img_cv.shape[1] - x)
            h = min(h, img_cv.shape[0] - y)
            
            if w < 10 or h < 10: continue

            color_crop = img_cv[y:y+h, x:x+w]
            gray_crop = gray[y:y+h, x:x+w]
            
            emo, conf, scores, age = predict_single_face(gray_crop, color_crop)
            
            results["faces"].append({
               "box": [int(x), int(y), int(w), int(h)],
               "prediction": emo,
               "confidence": round(conf, 2),
               "age": age,
               "all_scores": scores
            })
            
        return {"success": True, "data": results}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"success": False, "error": str(e)}

@app.post("/api/predict_frame")
def predict_webcam_frame(image_base64: str = Form(...)):
    """ Live Accelerated Webcam Streaming Endpoint """
    try:
        encoded_data = image_base64.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        results = {"faces": [], "objects": [], "counts": {"people": 0}}
        
        # Fast YOLO tracking
        yolo_res = yolo_model.predict(img_cv, classes=[0], verbose=False)[0] # Class 0 = Person only for speed
        results["counts"]["people"] = len(yolo_res.boxes)
        for box in yolo_res.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            results["objects"].append({
                "box": [int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])],
                "label": "Person",
                "confidence": float(box.conf[0]) * 100
            })
        
        # Face Extraction
        faces = get_faces(gray)
        
        # Fallback: If Haar Cascades fail but YOLO found people, use the top portion of those boxes
        if not faces:
            for box in yolo_res.boxes:
                b = box.xyxy[0].cpu().numpy().astype(int)
                # Estimate face from person box
                w = b[2]-b[0]
                h_full = b[3]-b[1]
                h = int(h_full * 0.35)
                faces.append([int(b[0]), int(b[1]), int(w), int(h)])

        for (x, y, w, h) in faces:
            # Safeguard coordinates
            x, y = max(0, x), max(0, y)
            w = min(w, img_cv.shape[1] - x)
            h = min(h, img_cv.shape[0] - y)
            
            if w < 10 or h < 10: continue

            color_crop = img_cv[y:y+h, x:x+w]
            gray_crop = gray[y:y+h, x:x+w]
            emo, conf, scores, age = predict_single_face(gray_crop, color_crop)
            results["faces"].append({
               "box": [int(x), int(y), int(w), int(h)],
               "prediction": emo,
               "confidence": round(conf, 2),
               "age": age,
               "all_scores": scores
            })
            
        return {"success": True, "data": results}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"success": False, "error": str(e)}

# Serve static frontend files last!
app.mount("/", StaticFiles(directory="static", html=True), name="static")
