import random
from typing import List

import numpy as np
import torch
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class ElForecastDataset(Dataset):
    def __init__(
        self,
        df_list: List[DataFrame],
        seq_length: int,
        step: int = 1,
        augment: int = 1,
        feat_scaler: StandardScaler = None,
        target_scaler: StandardScaler = None,
    ):
        self.augment = augment
        self.lseq = seq_length
        self.df_list = df_list
        self.nrows = [df.shape[0] for df in self.df_list]
        self.step = step

        self.feat_scaler = feat_scaler
        self.target_scaler = target_scaler

        Xs = []
        ys = []
        for df in df_list:
            vals = df.values.astype(np.float32)
            Xs.append(vals[:, :-1])
            ys.append(vals[:, -1])

        self.X, self.y = [], []

        step_subseq = self.step
        if self.augment > 1:
            step_subseq = 1

        for X, y in zip(Xs, ys):
            for i in range(0, len(X), step_subseq):
                if i + self.lseq > len(X):
                    break
                self.X.append(X[i : i + self.lseq, :])
                self.y.append(y[i : i + self.lseq])

    def __len__(self):
        if self.augment == 1:
            return len(self.X)
        return (len(self.X) - self.lseq) // self.step * self.augment

    def __getitem__(self, idx):
        if self.augment > 1:
            idx = random.randint(0, len(self) - 1)
        features = self.feat_scaler.transform(self.X[idx])
        target = self.target_scaler.transform(self.y[idx].reshape(-1, 1))
        features_tensor = torch.tensor(features)
        target_tensor = torch.tensor(target)

        return features_tensor, target_tensor
