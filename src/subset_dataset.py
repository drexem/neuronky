import torch
from torch.utils.data import Dataset


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