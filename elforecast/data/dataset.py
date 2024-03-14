import pandas as pd
import numpy as np
import torch

from elforecast.data.tools import data_preprocessing
from torch.utils.data import Dataset


class ElForecastDataset(Dataset):

    def __init__(self, path, seq_length, step=1):

        df = pd.read_csv(path)

        df =  data_preprocessing(df)
        self.seq_length = seq_length
        self.data = df.values[:, 1:].astype(np.float32)
        self.nfeatures = self.data.shape[1]

        self.mean = np.mean(self.data, axis=0)
        self.std = np.std(self.data, axis=0)

        self.step = step
    def __len__(self):
        return (len(self.data) - self.seq_length) // self.step

    def __getitem__(self, idx):
        
        start_idx = idx * self.step
        end_idx = start_idx + self.seq_length


        features = self.data[start_idx : end_idx]
        target = self.data[end_idx]

        features = (features - self.mean) / self.std

        features_tensor = torch.tensor(features)
        target_tensor = torch.tensor(target)

        return features_tensor, target_tensor
    
    def get_nfeatures(self):
        return self.nfeatures