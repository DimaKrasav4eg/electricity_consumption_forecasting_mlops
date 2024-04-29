import random

import numpy as np
import torch
from torch.utils.data import Dataset


class ElForecastDataset(Dataset):
    def __init__(self, data: np.array, seq_length, step=1, augment=1):
        self.augment = augment
        self.seq_length = seq_length
        self.data = data
        self.nfeatures = self.data.shape[1]

        self.step = step

    def __len__(self):
        return ((len(self.data) - self.seq_length) // self.step) * self.augment

    def __getitem__(self, idx):
        if self.augment == 1:
            start_idx = idx * self.step
            end_idx = start_idx + self.seq_length

            features = self.data[start_idx:end_idx, :-1]
            target = self.data[start_idx:end_idx, -1]
        else:
            start_idx = random.randint(0, len(self))
            end_idx = start_idx + self.seq_length
            features = self.data[start_idx:end_idx, :-1]
            target = self.data[start_idx:end_idx, -1]

        features_tensor = torch.tensor(features)
        target_tensor = torch.tensor(target)

        return features_tensor, target_tensor
