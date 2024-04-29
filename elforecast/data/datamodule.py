from typing import Optional

import lightning.pytorch as pl
import numpy as np
import pandas as pd
from elforecast.data.dataset import ElForecastDataset
from elforecast.data.tools import data_preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


class ElForecastDataModule(pl.LightningDataModule):
    def __init__(self, cfg: dict):
        super(ElForecastDataModule, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None):
        df = pd.read_csv(self.cfg.data.train_path)
        df_test = pd.read_csv(self.cfg.data.test_path)
        df = data_preprocessing(
            df, self.cfg.data.date, self.cfg.data.target, self.cfg.data.used_features
        )
        df_test = data_preprocessing(
            df_test,
            self.cfg.data.date,
            self.cfg.data.target,
            self.cfg.data.used_features,
            fill_target=True,
        )
        data = df.values.astype(np.float32)
        train_ratio = self.cfg["data"]["ratio"]["train"]
        train_data, val_data = train_test_split(
            data, train_size=train_ratio, shuffle=False
        )
        test_data = df_test.values.astype(np.float32)
        scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(train_data[:, -1].reshape(-1, 1))
        train_data = scaler.fit_transform(train_data)
        val_data = scaler.transform(val_data)
        test_data = scaler.transform(test_data)

        self.train_dataset = ElForecastDataset(
            train_data,
            self.cfg.data.lseq,
            step=self.cfg.data.step,
            augment=self.cfg.data.augment,
        )
        self.val_dataset = ElForecastDataset(
            val_data,
            self.cfg["data"]["lseq"],
            step=self.cfg["data"]["step"],
            augment=self.cfg.data.augment,
        )
        self.test_dataset = ElForecastDataset(test_data, self.cfg.data.lseq, step=1)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg["training"]["batch_size"],
            num_workers=self.cfg.training.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg["training"]["batch_size"],
            num_workers=self.cfg.training.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers,
            shuffle=False,
        )

    def get_target_scaler(self) -> StandardScaler:
        return self.target_scaler
