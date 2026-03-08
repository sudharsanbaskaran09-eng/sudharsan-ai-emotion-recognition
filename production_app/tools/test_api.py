import os
import random
import requests
import time

def test_api():
    base_dir = os.path.expanduser("~/.cache/kagglehub/datasets/msambare/fer2013/versions/1/test")
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} not found.")
        return

    emotions = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not emotions:
        print("No emotion folders found.")
        return

    emotion = random.choice(emotions)
    emotion_dir = os.path.join(base_dir, emotion)
    
    images = [f for f in os.listdir(emotion_dir) if f.endswith('.jpg') or f.endswith('.png')]
    if not images:
        print(f"No images found in {emotion_dir}")
        return

    image_name = random.choice(images)
    image_path = os.path.join(emotion_dir, image_name)

    print(f"Testing API with randomly selected image:")
    print(f"Path: {image_path}")
    print(f"True Label: {emotion.upper()}")

    url = "http://127.0.0.1:8005/api/predict"
    print(f"\nSending POST request to {url} ...")
    
    start_time = time.time()
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_name, f, "image/jpeg")}
            response = requests.post(url, files=files)
            
        elapsed = time.time() - start_time
        print(f"Response Time: {elapsed:.2f} seconds")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Response JSON:")
            print(response.json())
        else:
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"Failed to connect to API: {e}")

if __name__ == "__main__":
    test_api()
