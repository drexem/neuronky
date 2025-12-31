from src.dataset import ATPMatchesDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import random

# =========================
# Reproducibility
# =========================
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

# =========================
# Models
# =========================

class MLPBatchNorm(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64]):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


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


class DifferenceModel(nn.Module):
    def __init__(self, player_size, match_size):
        super().__init__()
        self.player_size = player_size
        self.match_size = match_size
        self.net = nn.Sequential(
            nn.Linear(player_size + match_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        p1 = x[:, :self.player_size]
        p2 = x[:, self.player_size:2 * self.player_size]
        match = x[:, 2 * self.player_size:2 * self.player_size + self.match_size]

        diff = (p1 - p2) / (p1.abs() + p2.abs() + 1e-6)
        return self.net(torch.cat([diff, match], dim=1)).squeeze(1)

# celkom dobre to funguje nad 66 vsetko
# class DifferenceModelBN(nn.Module):
#     def __init__(self, player_size, match_size, dropout=0.5):
#         super().__init__()
#         self.player_size = player_size
#         self.match_size = match_size
#
#         self.net = nn.Sequential(
#             nn.Linear(player_size + match_size, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#
#             nn.Linear(64, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#
#             nn.Linear(32, 1)
#         )
#
#     def forward(self, x):
#         p1 = x[:, :self.player_size]
#         p2 = x[:, self.player_size:2 * self.player_size]
#         match = x[:, 2 * self.player_size:2 * self.player_size + self.match_size]
#
#         diff = (p1 - p2) / (p1.abs() + p2.abs() + 1e-6)
#         return self.net(torch.cat([diff, match], dim=1)).squeeze(1)

class DifferenceModelBN(nn.Module):
    def __init__(self, player_size, match_size, dropout=0.45):
        super().__init__()
        self.player_size = player_size
        self.match_size = match_size

        self.net = nn.Sequential(
            nn.Linear(player_size + match_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        p1 = x[:, :self.player_size]
        p2 = x[:, self.player_size:2 * self.player_size]
        match = x[:, 2 * self.player_size:2 * self.player_size + self.match_size]

        diff = (p1 - p2) / (p1.abs() + p2.abs() + 1e-6)
        return self.net(torch.cat([diff, match], dim=1)).squeeze(1)
class PlayerEncoder(nn.Module):
    def __init__(self, player_size, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(player_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class SymmetricMatchPredictor(nn.Module):
    def __init__(self, player_size, match_size):
        super().__init__()
        self.player_size = player_size
        self.match_size = match_size
        self.encoder = PlayerEncoder(player_size)
        self.final = nn.Sequential(
            nn.Linear(2 * 64 + match_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        p1 = x[:, :self.player_size]
        p2 = x[:, self.player_size:2 * self.player_size]
        match = x[:, 2 * self.player_size:2 * self.player_size + self.match_size]

        e1 = self.encoder(p1)
        e2 = self.encoder(p2)

        combined = torch.cat([
            torch.abs(e1 - e2),
            e1 * e2,
            match
        ], dim=1)

        return self.final(combined).squeeze(1)


class WideAndDeep(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.wide = nn.Linear(input_size, 1)
        self.deep = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return (self.wide(x) + self.deep(x)).squeeze(1)


# =========================
# Feature size helper
# =========================

def get_feature_block_sizes(dataset):
    ct = dataset.pipeline.named_steps['column_transformer']

    winner_size = (
        len(ct.named_transformers_['winner_cat'].get_feature_names_out()) +
        len(ct.named_transformers_['winner_ioc'].get_feature_names_out()) +
        len(dataset.winner_numerical)
    )

    loser_size = (
        len(ct.named_transformers_['loser_cat'].get_feature_names_out()) +
        len(ct.named_transformers_['loser_ioc'].get_feature_names_out()) +
        len(dataset.loser_numerical)
    )

    match_size = len(ct.named_transformers_['match_cat'].get_feature_names_out())
    total = winner_size + loser_size + match_size

    return winner_size, loser_size, match_size, total


# =========================
# Training loop
# =========================

def train_model(model, train_loader, dev_loader, device, epochs=50):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_acc = 0
    patience, pat = 50, 0

    for epoch in range(epochs):
        model.train()
        loss_sum = 0

        for batch in train_loader:
            x = batch['features'].to(device)
            y = batch['target'].float().to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        model.eval()
        correct = total = 0

        with torch.no_grad():
            for batch in dev_loader:
                x = batch['features'].to(device)
                y = batch['target'].to(device)

                preds = (torch.sigmoid(model(x)) > 0.5).long()
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1:02d} | Loss {loss_sum/len(train_loader):.4f} | Dev Acc {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                print("Early stopping")
                break


if __name__ == "__main__":
    dataset = ATPMatchesDataset("data/processed/atp_matches_2000_2024_final.csv")

    train_loader = DataLoader(dataset.TRAIN, batch_size=32, shuffle=True, num_workers=2)
    dev_loader = DataLoader(dataset.DEV, batch_size=32, shuffle=False, num_workers=2)

    winner_size, loser_size, match_size, input_size = get_feature_block_sizes(dataset)

    assert winner_size == loser_size
    assert winner_size * 2 + match_size == input_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models = {
        "DifferenceModel": DifferenceModelBN(winner_size, match_size),
    }

    for name, model in models.items():
        print(f"\n🔥 Training {name}")
        model.to(device)
        train_model(model, train_loader, dev_loader, device)
