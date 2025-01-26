import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, X, y_classification, y_regression):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_classification = torch.tensor(y_classification, dtype=torch.long)
        self.y_regression = torch.tensor(y_regression, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_classification[idx], self.y_regression[idx]

class MultiTaskModel(nn.Module):
    def __init__(self, input_dim):
        super(MultiTaskModel, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        self.classification_head = nn.Sequential(
            nn.Linear(1024, 3),
            nn.Softmax(dim=1)
        )
        self.regression_head = nn.Sequential(
            nn.Linear(1024, 1),
            nn.ReLU()
        )

    def forward(self, x):
        shared_output = self.shared(x)
        classification_output = self.classification_head(shared_output)
        regression_output = self.regression_head(shared_output)
        return classification_output, regression_output

# 데이터 로드 및 전처리
path = "./data/data.csv"
df = pd.read_csv(path)
df.dropna(axis=0, inplace=True)
features = ['Supply_type', 'Units', 'Gender', 'Year',
            'Quarter', 'Rate1', 'Rate2', 'Rate3']
X = df[features]
y_classification = df['Cutline_rate']  # 분류 타겟
y_regression = df['Cutline_score']    # 회귀 타겟

# 라벨 인코딩 및 정규화
X['Supply_type'] = LabelEncoder().fit_transform(X['Supply_type'])
X['Gender'] = LabelEncoder().fit_transform(X['Gender'])
X['Year'] = LabelEncoder().fit_transform(X['Year'])
X['Quarter'] = LabelEncoder().fit_transform(X['Quarter'])

X_train, X_test, y_train_classification, y_test_classification, y_train_regression, y_test_regression = train_test_split(
    X, y_classification, y_regression, test_size=0.1, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train_classification = y_train_classification - 1
y_test_classification = y_test_classification - 1

# 데이터셋 및 데이터로더 정의
train_dataset = CustomDataset(X_train, y_train_classification.values, y_train_regression.values)
test_dataset = CustomDataset(X_test, y_test_classification.values, y_test_regression.values)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 모델 초기화
input_dim = X_train.shape[1]
model = MultiTaskModel(input_dim)

# 손실 함수 및 옵티마이저 정의
classification_criterion = nn.CrossEntropyLoss()
regression_criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 100
best_loss = float('inf')
patience = 15
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    train_classification_loss = 0.0
    train_regression_loss = 0.0
    for X_batch, y_class, y_reg in train_loader:
        X_batch, y_class, y_reg = X_batch.to(device), y_class.to(device), y_reg.to(device)
        
        optimizer.zero_grad()
        classification_output, regression_output = model(X_batch)
        
        classification_loss = classification_criterion(classification_output, y_class)
        regression_loss = regression_criterion(regression_output.squeeze(), y_reg)
        loss = classification_loss + regression_loss
        
        loss.backward()
        optimizer.step()
        
        train_classification_loss += classification_loss.item()
        train_regression_loss += regression_loss.item()

    train_classification_loss /= len(train_loader)
    train_regression_loss /= len(train_loader)

    # 검증
    model.eval()
    val_classification_loss = 0.0
    val_regression_loss = 0.0
    with torch.no_grad():
        for X_batch, y_class, y_reg in test_loader:
            X_batch, y_class, y_reg = X_batch.to(device), y_class.to(device), y_reg.to(device)
            classification_output, regression_output = model(X_batch)
            
            classification_loss = classification_criterion(classification_output, y_class)
            regression_loss = regression_criterion(regression_output.squeeze(), y_reg)
            
            val_classification_loss += classification_loss.item()
            val_regression_loss += regression_loss.item()

    val_classification_loss /= len(test_loader)
    val_regression_loss /= len(test_loader)
    val_loss = val_classification_loss + val_regression_loss

    print(f"Epoch {epoch+1}/{num_epochs}, Train Classification Loss: {train_classification_loss:.4f}, "
          f"Train Regression Loss: {train_regression_loss:.4f}, Val Classification Loss: {val_classification_loss:.4f}, "
          f"Val Regression Loss: {val_regression_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print("Model saved!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping.")
            break

# 평가 및 시각화
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
all_classification_outputs = []
all_regression_outputs = []
all_y_class = []
all_y_reg = []

with torch.no_grad():
    for X_batch, y_class, y_reg in test_loader:
        X_batch = X_batch.to(device)
        classification_output, regression_output = model(X_batch)
        
        all_classification_outputs.append(classification_output.cpu().numpy())
        all_regression_outputs.append(regression_output.cpu().numpy())
        all_y_class.append(y_class.numpy())
        all_y_reg.append(y_reg.numpy())

# 결과 출력
print("Evaluation complete.")
