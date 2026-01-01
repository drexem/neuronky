import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.bn1 = nn.BatchNorm1d(size)
        self.fc2 = nn.Linear(size, size)
        self.bn2 = nn.BatchNorm1d(size)

    def forward(self, x):
        r = x
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return torch.relu(x + r)

class ResidualMLP(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_blocks=3):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_size) for _ in range(num_blocks)]
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        return self.output(x).squeeze(1)