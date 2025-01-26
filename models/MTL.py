import torch
import torch.nn as nn
from torch.utils.data import Dataset

class getDataset(Dataset):
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