import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class ATPMatchesDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.matches_df = pd.read_csv(csv_file)
        self.transform = transform

        # we dont want nones here
        self.matches_df = self.matches_df.dropna()

        # Initialize OneHotEncoder for surface
        self.surface_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        # Fit on all possible surfaces
        surfaces = self.matches_df['surface'].values.reshape(-1, 1)
        self.surface_encoder.fit(surfaces)

        # Initialize OneHotEncoder for tourney_level
        self.level_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        # Fit on all possible tourney levels
        levels = self.matches_df['tourney_level'].values.reshape(-1, 1)
        self.level_encoder.fit(levels)

        # Initialize OneHotEncoder for best_of
        self.best_of_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        best_of_values = self.matches_df['best_of'].values.reshape(-1, 1)
        self.best_of_encoder.fit(best_of_values)

        # Initialize OneHotEncoder for round
        self.round_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        rounds = self.matches_df['round'].values.reshape(-1, 1)
        self.round_encoder.fit(rounds)

        # Initialize OneHotEncoder for IOC (both winner and loser)
        self.ioc_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        # Combine all IOC values from both winner and loser columns
        all_iocs = np.concatenate([
            self.matches_df['winner_ioc'].values,
            self.matches_df['loser_ioc'].values
        ]).reshape(-1, 1)
        self.ioc_encoder.fit(all_iocs)

    def __len__(self):
        return len(self.matches_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        match = self.matches_df.iloc[idx]

        # Create features for both players
        # OneHot encode winner_ioc and loser_ioc using unified encoder
        winner_ioc_onehot = self.ioc_encoder.transform([[match['winner_ioc']]])[0]
        loser_ioc_onehot = self.ioc_encoder.transform([[match['loser_ioc']]])[0]

        winner_features = [
            match['winner_rank'],
            match['winner_age'],
            match['winner_ht'],
            1 if match['winner_hand'] == 'R' else 0,
            match['w_ace_avg'],
            match['w_df_avg'],
            match['w_svpt_avg'],
            match['w_1stIn_avg'],
            match['w_1stWon_avg'],
            match['w_2ndWon_avg'],
            match['w_SvGms_avg'],
            match['w_bpSaved_avg'],
            match['w_bpFaced_avg']
        ] + winner_ioc_onehot.tolist()

        loser_features = [
            match['loser_rank'],
            match['loser_age'],
            match['loser_ht'],
            1 if match['loser_hand'] == 'R' else 0,
            match['l_ace_avg'],
            match['l_df_avg'],
            match['l_svpt_avg'],
            match['l_1stIn_avg'],
            match['l_1stWon_avg'],
            match['l_2ndWon_avg'],
            match['l_SvGms_avg'],
            match['l_bpSaved_avg'],
            match['l_bpFaced_avg']
        ] + loser_ioc_onehot.tolist()

        # Match context features
        # OneHot encode surface
        surface_onehot = self.surface_encoder.transform([[match['surface']]])[0]

        # OneHot encode tourney_level
        level_onehot = self.level_encoder.transform([[match['tourney_level']]])[0]

        # OneHot encode best_of
        best_of_onehot = self.best_of_encoder.transform([[match['best_of']]])[0]

        # OneHot encode round
        round_onehot = self.round_encoder.transform([[match['round']]])[0]

        # Combine all features
        feature_tensor = torch.tensor(
            winner_features + loser_features + surface_onehot.tolist() + level_onehot.tolist() + best_of_onehot.tolist() + round_onehot.tolist(),
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
                loser_features + winner_features + surface_onehot.tolist() + level_onehot.tolist() + best_of_onehot.tolist() + round_onehot.tolist(),
                dtype=torch.float32
            )
            target = torch.tensor(0, dtype=torch.long)

        sample = {'features': feature_tensor, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Usage example
# dataset = ATPMatchesDataset('../data/processed/atp_matches_2000_2024_final.csv')
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#
# for batch in dataloader:
#     features = batch['features']
#     targets = batch['target']
#     print(features.size(), targets.size())