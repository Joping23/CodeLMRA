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
            torch.nn.Linear(input_size, 50),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=50),
            torch.nn.Linear(50, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=128),
            torch.nn.Linear(128, input_size),
        )

    def forward(self, x):
        out = self.linearlayer(x)
        return out