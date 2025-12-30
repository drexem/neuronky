import sklearn
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


class ATPMatchesSubsetDataset(Dataset):
    def __init__(self, transformed_data, targets, transform=None):
        self.transformed_data = transformed_data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.transformed_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = torch.tensor(self.transformed_data[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.long)

        sample = {'features': features, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ATPMatchesDataset:
    def __init__(self, csv_file, transform=None, seed=42):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.matches_df = pd.read_csv(csv_file)
        self.transform = transform
        self.matches_df = self.matches_df.dropna()
        self.seed = seed

        # Define feature columns
        self.winner_categorical = ['winner_hand']
        self.winner_numerical = [
            'winner_rank', 'winner_age', 'winner_ht', 'winner_seed',
            'w_ace_avg', 'w_df_avg', 'w_svpt_avg', 'w_1stIn_avg', 'w_1stWon_avg',
            'w_2ndWon_avg', 'w_SvGms_avg', 'w_bpSaved_avg', 'w_bpFaced_avg'
        ]
        self.loser_categorical = ['loser_hand']
        self.loser_numerical = [
            'loser_rank', 'loser_age', 'loser_ht', 'loser_seed',
            'l_ace_avg', 'l_df_avg', 'l_svpt_avg', 'l_1stIn_avg', 'l_1stWon_avg',
            'l_2ndWon_avg', 'l_SvGms_avg', 'l_bpSaved_avg', 'l_bpFaced_avg'
        ]
        self.match_categorical = ['surface', 'tourney_level', 'best_of', 'round']


        self.TRAIN, self.DEV, self.TEST = self.create_splits()


    def create_splits(self, train_ratio=0.9, dev_ratio=0.05, test_ratio=0.05):
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

        ioc_unique = np.unique(np.concatenate([
            self.matches_df['winner_ioc'].dropna().values,
            self.matches_df['loser_ioc'].dropna().values
        ]))

        # Create and fit pipeline on training data only
        ct = sklearn.compose.ColumnTransformer([
            ('winner_cat', OneHotEncoder(categories=[hand_unique], sparse_output=False, handle_unknown='ignore'),
             self.winner_categorical),
            ('winner_ioc', OneHotEncoder(categories=[ioc_unique], sparse_output=False, handle_unknown='ignore'),
             ['winner_ioc']),
            ('winner_num', MinMaxScaler(), self.winner_numerical),
            ('loser_cat', OneHotEncoder(categories=[hand_unique], sparse_output=False, handle_unknown='ignore'),
             self.loser_categorical),
            ('loser_ioc', OneHotEncoder(categories=[ioc_unique], sparse_output=False, handle_unknown='ignore'),
             ['loser_ioc']),
            ('loser_num', MinMaxScaler(), self.loser_numerical),
            ('match_cat', OneHotEncoder(categories=[surface_unique, tourney_level_unique, best_of_unique, round_unique],
                                        sparse_output=False, handle_unknown='ignore'), self.match_categorical),
        ])

        self.pipeline = sklearn.pipeline.Pipeline([
            ('column_transformer', ct),
        ])

        feature_columns = (self.winner_categorical + ['winner_ioc'] + self.winner_numerical +
                          self.loser_categorical + ['loser_ioc'] + self.loser_numerical +
                          self.match_categorical)

        self.pipeline.fit(train_df[feature_columns])

        # Transform all data using the fitted pipeline
        all_transformed_data = self.pipeline.transform(self.matches_df[feature_columns])

        # Prepare targets + implement swapping logic
        np.random.seed(self.seed)
        n_samples = len(self.matches_df)

        # Create random swap mask - True means swap winner/loser
        swap_mask = np.random.random(n_samples) < 0.5

        # Initialize targets array
        self.targets = np.ones(n_samples, dtype=np.int64)  # 1 = player 1 wins

        # For swapped matches, target becomes 0 (player 1 loses, original loser wins)
        self.targets[swap_mask] = 0

        winner_cat_size = len(self.pipeline.named_steps['column_transformer']
                              .named_transformers_['winner_cat']
                              .get_feature_names_out())
        winner_ioc_size = len(self.pipeline.named_steps['column_transformer']
                              .named_transformers_['winner_ioc']
                              .get_feature_names_out())
        winner_num_size = len(self.winner_numerical)
        winner_total = winner_cat_size + winner_ioc_size + winner_num_size

        loser_cat_size = len(self.pipeline.named_steps['column_transformer']
                             .named_transformers_['loser_cat']
                             .get_feature_names_out())
        loser_ioc_size = len(self.pipeline.named_steps['column_transformer']
                             .named_transformers_['loser_ioc']
                             .get_feature_names_out())
        loser_num_size = len(self.loser_numerical)
        loser_total = loser_cat_size + loser_ioc_size + loser_num_size

        # Create swapped version of transformed data
        swapped_data = all_transformed_data.copy()
        for i in range(n_samples):
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

# Usage example
# dataset = ATPMatchesDataset('../data/processed/atp_matches_2000_2024_final.csv')
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#
# for batch in dataloader:
#     features = batch['features']
#     targets = batch['target']
#     print(features.size(), targets.size())