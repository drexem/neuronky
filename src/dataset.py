import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class ATPMatchesDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.matches_df = pd.read_csv(csv_file)
        self.transform = transform

        # Clean data - remove rows with missing critical values
        self.matches_df = self.matches_df.dropna(subset=['winner_rank', 'loser_rank'])

    def __len__(self):
        return len(self.matches_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        match = self.matches_df.iloc[idx]

        # Create features for both players
        winner_features = [
            match['winner_rank'] if pd.notna(match['winner_rank']) else 1000,
            match['winner_age'] if pd.notna(match['winner_age']) else 25,
            match['winner_ht'] if pd.notna(match['winner_ht']) else 180,
            1 if match['winner_hand'] == 'R' else 0 if pd.notna(match['winner_hand']) else 0.5
        ]

        loser_features = [
            match['loser_rank'] if pd.notna(match['loser_rank']) else 1000,
            match['loser_age'] if pd.notna(match['loser_age']) else 25,
            match['loser_ht'] if pd.notna(match['loser_ht']) else 180,
            1 if match['loser_hand'] == 'R' else 0 if pd.notna(match['loser_hand']) else 0.5
        ]

        # Match context features
        surface_encoding = {'Hard': 0, 'Clay': 1, 'Grass': 2}
        surface = surface_encoding.get(match['surface'], 0)

        level_encoding = {'G': 4, 'M': 3, 'A': 2, 'D': 1, 'F': 0}
        level = level_encoding.get(match['tourney_level'], 1)

        # Combine all features
        feature_tensor = torch.tensor(
            winner_features + loser_features + [surface, level],
            dtype=torch.float32
        )

        # Target: 1 if first player (winner) wins, 0 if second player (loser) wins
        # Since we know the winner, we'll randomly assign which player is "player 1"
        # This creates a balanced dataset
        if np.random.random() > 0.5:
            # Winner is player 1
            target = torch.tensor(1, dtype=torch.long)
        else:
            # Swap players - loser becomes player 1
            feature_tensor = torch.tensor(
                loser_features + winner_features + [surface, level],
                dtype=torch.float32
            )
            target = torch.tensor(0, dtype=torch.long)

        sample = {'features': feature_tensor, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Usage example
# dataset = ATPMatchesDataset('tennis_atp/atp_matches_2019.csv')
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)