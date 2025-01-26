import torch
import torch.nn as nn
from torch.utils.data import Dataset

class getDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_regression = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_regression[idx]

class MultiTaskModel(nn.Module):
    def __init__(self, input_dim):
        super(MultiTaskModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)