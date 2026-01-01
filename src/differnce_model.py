import torch
import torch.nn as nn

class DifferenceModelBN(nn.Module):
    def __init__(self, player_size, match_size, dropout=(0.4, 0.3, 0.2)):
        super().__init__()
        self.player_size = player_size
        self.match_size = match_size

        self.net = nn.Sequential(

            nn.Linear(player_size + match_size, 128),
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
        p1 = x[:, :self.player_size]
        p2 = x[:, self.player_size:2 * self.player_size]
        match = x[:, 2 * self.player_size:2 * self.player_size + self.match_size]

        diff = (p1 - p2) / (p1.abs() + p2.abs() + 1e-6)
        return self.net(torch.cat([diff, match], dim=1)).squeeze(1)


