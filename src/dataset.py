import sklearn
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from src.subset_dataset import ATPMatchesSubsetDataset

class ATPMatchesDataset:
    def __init__(self, csv_file, transform=None, seed=42):

        self.matches_df = pd.read_csv(csv_file)
        self.transform = transform
        self.matches_df = self.matches_df.dropna()
        self.seed = seed

        self.winner_numerical = [
            'winner_rank', 'winner_age', 'winner_ht', 'winner_seed',
            'w_ace_avg', 'w_df_avg', 'w_svpt_avg', 'w_1stIn_avg', 'w_1stWon_avg',
            'w_2ndWon_avg', 'w_SvGms_avg', 'w_bpSaved_avg', 'w_bpFaced_avg'
            , 'w_win_pct_avg'
        ]

        self.loser_numerical = [
            'loser_rank', 'loser_age', 'loser_ht', 'loser_seed',
            'l_ace_avg', 'l_df_avg', 'l_svpt_avg', 'l_1stIn_avg', 'l_1stWon_avg',
            'l_2ndWon_avg', 'l_SvGms_avg', 'l_bpSaved_avg', 'l_bpFaced_avg'
            , 'l_win_pct_avg'
        ]
        self.match_categorical = ['surface', 'tourney_level', 'best_of', 'round']


        self.TRAIN, self.DEV, self.TEST = self.create_splits()


    def create_splits(self, train_ratio=0.75, dev_ratio=0.125):
        """Create train/dev/test splits and fit pipeline only on training data"""
        n_samples = len(self.matches_df)

        # Calculate split indices
        train_end = int(train_ratio * n_samples)
        dev_end = train_end + int(dev_ratio * n_samples)

        # Create indices and shuffle
        indices = np.arange(n_samples)
        np.random.seed(self.seed)
        np.random.shuffle(indices)

        # Split indices
        train_indices = indices[:train_end]
        dev_indices = indices[train_end:dev_end]
        test_indices = indices[dev_end:]

        # Get training data for fitting pipeline
        train_df = self.matches_df.iloc[train_indices]

        winner_hand_unique = np.unique(self.matches_df['winner_hand'].dropna())
        loser_hand_unique = np.unique(self.matches_df['loser_hand'].dropna())
        hand_unique = np.unique(np.concatenate([winner_hand_unique, loser_hand_unique]))

        surface_unique = np.unique(self.matches_df['surface'].dropna())
        tourney_level_unique = np.unique(self.matches_df['tourney_level'].dropna())
        best_of_unique = np.unique(self.matches_df['best_of'].dropna())
        round_unique = np.unique(self.matches_df['round'].dropna())

        winner_num_cols = self.winner_numerical
        loser_num_cols = self.loser_numerical

        combined_num_data = np.vstack([
            train_df[winner_num_cols].values,
            train_df[loser_num_cols].values
        ])

        self.shared_num_scaler = MinMaxScaler()
        self.shared_num_scaler.fit(combined_num_data)

        ct = sklearn.compose.ColumnTransformer([
            ('winner_num', 'passthrough', self.winner_numerical),
            ('loser_num', 'passthrough', self.loser_numerical),
            ('match_cat', OneHotEncoder(categories=[surface_unique, tourney_level_unique, best_of_unique, round_unique],
                                        sparse_output=False, handle_unknown='ignore'), self.match_categorical),
        ])

        self.pipeline = sklearn.pipeline.Pipeline([
            ('column_transformer', ct),
        ])

        feature_columns = self.winner_numerical + self.loser_numerical + self.match_categorical

        self.pipeline.fit(train_df[feature_columns])

        # Transform all data using the fitted pipeline
        all_transformed_data = self.pipeline.transform(self.matches_df[feature_columns])

        # Now apply shared scaling to the numerical columns
        # Find the indices of numerical features in the transformed data
        winner_num_size = len(self.winner_numerical)
        loser_num_size = len(self.loser_numerical)

        # Calculate start indices for numerical features
        winner_num_start = 0
        winner_num_end = winner_num_start + winner_num_size
        loser_num_start = winner_num_end + 0
        loser_num_end = loser_num_start + loser_num_size

        # Apply shared scaler to both winner and loser numerical features
        all_transformed_data[:, winner_num_start:winner_num_end] = \
            self.shared_num_scaler.transform(all_transformed_data[:, winner_num_start:winner_num_end])
        all_transformed_data[:, loser_num_start:loser_num_end] = \
            self.shared_num_scaler.transform(all_transformed_data[:, loser_num_start:loser_num_end])

        # Prepare targets + implement swapping logic
        np.random.seed(self.seed)
        n_samples_total = len(self.matches_df)

        # Create random swap mask - True means swap winner/loser
        swap_mask = np.random.random(n_samples_total) < 0.5

        # Initialize targets array
        self.targets = np.ones(n_samples_total, dtype=np.int64)  # 1 = player 1 wins

        # For swapped matches, target becomes 0 (player 1 loses, original loser wins)
        self.targets[swap_mask] = 0

        # Create swapped version of transformed data
        swapped_data = all_transformed_data.copy()
        for i in range(n_samples_total):
            if swap_mask[i]:
                # Swap winner and loser features
                winner_features = all_transformed_data[i, :winner_total].copy()
                loser_features = all_transformed_data[i, winner_total:winner_total + loser_total].copy()

                swapped_data[i, :winner_total] = loser_features
                swapped_data[i, winner_total:winner_total + loser_total] = winner_features
                # Match features remain unchanged

        all_transformed_data = swapped_data


        # Create subset datasets
        train_dataset = ATPMatchesSubsetDataset(
            all_transformed_data[train_indices],
            self.targets[train_indices],
            self.transform
        )

        dev_dataset = ATPMatchesSubsetDataset(
            all_transformed_data[dev_indices],
            self.targets[dev_indices],
            self.transform
        )

        test_dataset = ATPMatchesSubsetDataset(
            all_transformed_data[test_indices],
            self.targets[test_indices],
            self.transform
        )

        return train_dataset, dev_dataset, test_dataset

    def __len__(self):
        return len(self.matches_df)