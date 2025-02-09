import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json
import numpy as np
import csv
import networkx as nx
import gc

class RoomDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]

        self.room_weights = {
            13: 800,   # 거실
            14: 1000,  # 침실
            15: 800,   # 주방
            16: 500,   # 현관
            17: 800,   # 발코니
            18: 500,   # 화장실
            19: 800,   # 실외기실
            20: 1000    # 드레스룸
        }

        self.object_weights = {
            4: 500,    # 변기
            5: 500,    # 세면대
            6: 500,    # 싱크대
            7: 1000,   # 욕조
            8: 700     # 가스레인지
        }


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".jpg", ".json").replace(".png", ".json"))

        image = Image.open(img_path).convert("RGB")
        image = image.resize((640, 640), Image.Resampling.LANCZOS)

        with open(label_path, 'r') as f:
            label_data = json.load(f)

        score = self.calculate_score(label_data)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([score], dtype=torch.float32), img_name

    def calculate_score(self, label_data):
        room_score = 0.0
        object_score = 0.0
        
        # 이미지 전체 면적 계산
        image_area = float(label_data['images'][0]['width']) * float(label_data['images'][0]['height'])
        
        for a in label_data['annotations']:
            category_id = a['category_id']
            
            if category_id in self.room_weights:
                room_score += self.room_weights[category_id] * (a['area'] / image_area)
            
            if category_id in self.object_weights:
                object_score += self.object_weights[category_id] * a.get("count", 1)
        
        final_score = room_score * 0.9 + object_score * 0.1
        return final_score

class RoomQualityModel(nn.Module):
    def __init__(self):
        super(RoomQualityModel, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.backbone(x)

class Model:
    def __init__(self, dataset_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = 0.001
        self.batch_size = 16
        self.epochs = 5
        self.patience = 2

        self.model = RoomQualityModel().to(self.device)
        self.criterion = nn.HuberLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2)

        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(640),
                transforms.CenterCrop(640),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(10),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(640),
                transforms.CenterCrop(640),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(640),
                transforms.CenterCrop(640),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        self.dataloaders = self._prepare_dataloaders(dataset_path)

    def _prepare_dataloaders(self, dataset_path):
        phases = ["train", "valid", "test"]
        dataloaders = {}
        for phase in phases:
            img_dir = os.path.join(dataset_path, f"image/{phase}")
            label_dir = os.path.join(dataset_path, f"label/{phase}")
            dataset = RoomDataset(img_dir, label_dir, transform=self.data_transforms[phase])
            dataloaders[phase] = DataLoader(dataset, batch_size=self.batch_size, shuffle=(phase == "train"))
        return dataloaders

    def train(self, save_path):
        best_val_loss = float('inf')
        early_stop_counter = 0

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            for phase in ["train", "valid"]:
                self.model.train() if phase == "train" else self.model.eval()
                running_loss = 0.0

                for images, scores, _ in self.dataloaders[phase]:
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
                    self.scheduler.step(epoch_loss)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Current Learning Rate: {current_lr:.6f}")
                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        torch.save(self.model.state_dict(), save_path)
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= self.patience:
                            print("Early stopping!")
                            return
        gc.collect()
        torch.cuda.empty_cache()

    def evaluate(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        test_loss = 0.0
        predictions = []
        true_scores = []
        file_names = []

        with torch.no_grad():
            for batch in self.dataloaders["test"]:
                images, scores, filenames = batch
                images, scores = images.to(self.device), scores.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, scores)
                test_loss += loss.item()

                predictions.extend(outputs.cpu().numpy())
                true_scores.extend(scores.cpu().numpy())
                file_names.extend(filenames)

        print(f"Test Loss: {test_loss / len(self.dataloaders['test']):.4f}")

        predictions = np.array(predictions).flatten()
        true_scores = np.array(true_scores).flatten()
        mse = np.mean((predictions - true_scores) ** 2)
        mae = np.mean(np.abs(predictions - true_scores))

        print(f"MSE : {mse:.4f}")
        print(f"MAE : {mae:.4f}")

        with open('struct.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["File Name", "Predicted Score", "True Score"])
            for fname, pred, true in zip(file_names, predictions, true_scores):
                writer.writerow([fname, pred, true])