import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import cv2
from skimage.feature import hog
import kagglehub

from core_model import DCNN_BiLSTM_DAM

# Dataset wrapper to apply HOG on the fly exactly as required
class FER2013HOGDataset(Dataset):
    def __init__(self, image_folder_dataset, apply_augmentations=False):
        self.dataset = image_folder_dataset
        self.apply_augmentations = apply_augmentations
        
        # We need transforms for augmentation before HOG
        if self.apply_augmentations:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        else:
            self.aug_transform = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pil_image, label = self.dataset[idx]
        
        if self.aug_transform:
            pil_image = self.aug_transform(pil_image)
            
        # Convert to numpy array (grayscale)
        gray_image = np.array(pil_image.convert('L'))
        resized = cv2.resize(gray_image, (64, 64))
        
        # Apply HOG (Histogram of Oriented Gradients)
        _, hog_img = hog(resized, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        
        # Normalize
        hog_normalized = hog_img / (np.max(hog_img) + 1e-5)
        
        # Create tensor [1, 64, 64]
        image_tensor = torch.tensor(hog_normalized, dtype=torch.float32).unsqueeze(0)
        
        return image_tensor, label

def run_fer2013_training():
    print("\n--- Initializing FER-2013 HOG DCNN-BiLSTM-DAM Engine ---")
    
    # 1. Download/Locate FER2013 via KaggleHub
    print("Fetching FER-2013 Dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download("msambare/fer2013")
    print(f"Dataset located at: {dataset_path}")
    
    train_dir = os.path.join(dataset_path, "train")
    test_dir = os.path.join(dataset_path, "test")
    
    # 'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6
    raw_train = datasets.ImageFolder(train_dir)
    raw_test = datasets.ImageFolder(test_dir)
    
    print(f"Classes correctly mapped: {raw_train.classes}")
    
    # Create HOG-wrapped datasets
    train_dataset = FER2013HOGDataset(raw_train, apply_augmentations=True)
    val_dataset = FER2013HOGDataset(raw_test, apply_augmentations=False)
    
    epochs = 12 
    batch_size = 64
    learning_rate = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Hardware Backbone: {device}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
    
    model = DCNN_BiLSTM_DAM(num_classes=7).to(device)
    
    # Label smoothing & AdamW for robustness
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) 
    
    best_val_acc = 0.0
    
    print(f"Training on {len(train_dataset)} images. Validating on {len(val_dataset)} images.")
    
    # Create models directory
    os.makedirs("./models", exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 100 == 0:
                print(f"  Batch {i}/{len(train_loader)} Loss: {loss.item():.4f}")
                
        # Validation Loop
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        print(f"--> Epoch [{epoch}/{epochs}] | LR: {current_lr:.5f} | Avg Loss: {epoch_loss:.4f} | Validation Acc: {val_acc:.2f}%")
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), './models/best_model_dcnn_dam.pth')
            print(f"🌟 New best model saved! Validation Acc: {best_val_acc:.2f}%")

    print(f"\n✅ Full FER-2013 Training Complete. Max Acc Reached: {best_val_acc:.2f}%")

if __name__ == "__main__":
    run_fer2013_training()
