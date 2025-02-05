import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import json
import numpy as np
import csv

class RoomDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg") or f.endswith(".png")]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".jpg", ".json").replace(".png", ".json"))
        
        image = Image.open(img_path).convert("RGB")
        
        with open(label_path, 'r') as f:
            label_data = json.load(f)
        
        room_counts = self.count_room_types(label_data)        
        score = self.calculate_score(room_counts)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor([score], dtype=torch.float32)

    def count_room_types(self, label_data):
        room_counts = { "거실": 0, "침실": 0, "주방": 0, "화장실": 0, "발코니": 0 }
        for annotation in label_data['annotations']:
            category_id = annotation['category_id']
            if category_id == 13:
                room_counts["거실"] += 1
            elif category_id == 14:
                room_counts["침실"] += 1
            elif category_id == 15:
                room_counts["주방"] += 1
            elif category_id == 18:
                room_counts["화장실"] += 1
            elif category_id == 17:
                room_counts["발코니"] += 1
        return room_counts

    def calculate_score(self, room_counts):
        score = (room_counts["거실"] * 2 + room_counts["침실"] * 1.5 +
                 room_counts["주방"] * 1.5 + room_counts["화장실"] * 1 +
                 room_counts["발코니"] * 0.5) / 10
        return min(score, 1)

class RoomQualityModel(nn.Module):
    def __init__(self):
        super(RoomQualityModel, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.ReLU()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        score = self.fc(features)
        return torch.clamp(score, 0, 1)

class ModelTrainer:
    def __init__(self, dataset_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = 0.0001
        self.BATCH_SIZE = 16
        self.epochs = 10
        self.early_stop_patience = 0

        self.model = RoomQualityModel().to(self.device)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        
        self.data_transforms = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(5),  # 회전 범위
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.dataloaders = self._prepare_dataloaders(dataset_path)

    def _prepare_dataloaders(self, dataset_path):
        data_dirs = {
            "train": (os.path.join(dataset_path, "image/train"), os.path.join(dataset_path, "label/train")),
            "valid": (os.path.join(dataset_path, "image/valid"), os.path.join(dataset_path, "label/valid")),
            "test":  (os.path.join(dataset_path, "image/test"), os.path.join(dataset_path, "label/test"))
        }

        dataloaders = {}
        for phase in ["train", "valid", "test"]:
            img_dir, label_dir = data_dirs[phase]
            dataset = RoomDataset(img_dir, label_dir, transform=self.data_transforms)
            dataloaders[phase] = DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=(phase == "train"))
        return dataloaders

    def train(self, path, patience=5):
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            for phase in ["train", "valid"]:
                self.model.train() if phase == "train" else self.model.eval()

                running_loss = 0.0
                for images, scores in self.dataloaders[phase]:
                    images, scores = images.to(self.device), scores.to(self.device)

                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, scores)
                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()
                    running_loss += loss.item()

                epoch_loss = running_loss / len(self.dataloaders[phase])
                print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}")

                if phase == "valid":
                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        torch.save(self.model.state_dict(), path)
                        self.early_stop_patience = 0
                    else:
                        self.early_stop_patience += 1

                    if self.early_stop_patience >= patience:
                        print(f"Early stopping : {epoch+1}")
                        return

    def evaluate(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        test_loss = 0.0
        predictions = []
        true_scores = []
        with torch.no_grad():
            for images, scores in self.dataloaders["test"]:
                images, scores = images.to(self.device), scores.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, scores)
                test_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                true_scores.extend(scores.cpu().numpy())

        print(f"Test Loss: {test_loss / len(self.dataloaders['test']):.4f}")

        predictions = np.array(predictions).flatten()
        true_scores = np.array(true_scores).flatten()
        mse = np.mean((predictions - true_scores) ** 2)
        mae = np.mean(np.abs(predictions - true_scores))

        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        
        with open('test_results.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Predicted Score", "True Score"])
            for pred, true in zip(predictions, true_scores):
                writer.writerow([pred, true])
        print("Results saved")

if __name__ == "__main__":
    dataset_path = os.path.abspath(os.path.join(os.getcwd(), "..", "datasets"))
    models_path = os.path.abspath(os.path.join(os.getcwd(), "best_model.pth"))
    
    trainer = ModelTrainer(dataset_path)
    trainer.train(models_path)
    trainer.evaluate(models_path)