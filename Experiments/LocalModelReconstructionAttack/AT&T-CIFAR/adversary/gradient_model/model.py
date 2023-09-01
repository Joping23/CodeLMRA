import torch.nn as nn
import torch


class LinearNet(nn.Module):
    def __init__(self, input_size, num_features):
        super(LinearNet, self).__init__()
        self.linearlayer = torch.nn.Sequential(
            torch.nn.Linear(input_size, num_features),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=num_features),
            torch.nn.Linear(num_features, input_size),
        )

    def forward(self, x):
        out = self.linearlayer(x)
        return out

class LinearNetMulti(nn.Module):
    def __init__(self, input_size):
        super(LinearNetMulti, self).__init__()
        self.linearlayer = torch.nn.Sequential(
            torch.nn.Linear(input_size, 16),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=16),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=16),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(16, input_size),
        )

    def forward(self, x):
        out = self.linearlayer(x)
        return out