import torch.nn as nn

class BaseLineModel(nn.Module):
    def __init__(self, input_size, dropout=(0.4, 0.3, 0.2)):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout[0]),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout[1]),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(dropout[2]),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


