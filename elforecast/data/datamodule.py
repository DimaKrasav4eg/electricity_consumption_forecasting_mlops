import random
from typing import Optional

import lightning.pytorch as pl
import numpy as np
import pandas as pd
from elforecast.data.dataset import ElForecastDataset
from elforecast.data.tools import data_preprocessing
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
        drop_if_less = "less" in self.cfg.data.drop
        dl = None
        if drop_if_less:
            dl = self.cfg.data.drop.less
        df_list = data_preprocessing(
            df,
            self.cfg.data.date,
            self.cfg.data.target,
            self.cfg.data.used_features,
            max_miss=self.cfg.data.max_miss,
            min_true=self.cfg.data.min_true,
            drop_if_less=drop_if_less,
            dl=dl,
        )
        df_list_test = data_preprocessing(
            df_test,
            self.cfg.data.date,
            self.cfg.data.target,
            self.cfg.data.used_features,
            fill_target=True,
        )

        nrows = [df.shape[0] for df in df_list]
        sum_nrows = sum(nrows)
        train_ratio = self.cfg.data.ratio.train
        val_nrows = sum_nrows * (1 - train_ratio)

        val_indxs = dict()
        while sum(val_indxs.values()) < val_nrows:
            ind = random.randint(0, len(nrows) - 1)
            while ind in val_indxs.keys():
                ind = random.randint(0, len(nrows) - 1)
            val_indxs[ind] = nrows[ind]
        val_indxs = sorted(list(val_indxs.keys()))
        val_indxs = [i for i in range(1, len(nrows), 4)]

        train_indxs = []
        for i in range(len(nrows)):
            if i not in val_indxs:
                train_indxs.append(i)

        train_df_list = [df_list[ind] for ind in train_indxs]
        valid_df_list = [df_list[ind] for ind in val_indxs]
        df_train = pd.concat(train_df_list, ignore_index=True)

        train_data = df_train.values.astype(np.float32)

        test_data = df_list_test[0].values.astype(np.float32)
        X_train, y_train = train_data[:, :-1], train_data[:, -1]

        X_test, y_test = test_data[:, :-1], test_data[:, -1]

        scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        X_train = scaler.fit(X_train)

        X_test = scaler.transform(X_test)

        y_train = self.target_scaler.fit(y_train.reshape(-1, 1))
        y_test = self.target_scaler.transform(y_test.reshape(-1, 1))

        self.train_dataset = ElForecastDataset(
            train_df_list,
            self.cfg.data.lseq,
            step=self.cfg.data.step,
            augment=self.cfg.data.augment,
            feat_scaler=scaler,
            target_scaler=self.target_scaler,
        )
        self.val_dataset = ElForecastDataset(
            valid_df_list,
            self.cfg["data"]["lseq"],
            step=self.cfg["data"]["step"],
            augment=self.cfg.data.augment,
            feat_scaler=scaler,
            target_scaler=self.target_scaler,
        )
        self.test_dataset = ElForecastDataset(
            df_list_test,
            self.cfg.data.lseq,
            step=1,
            feat_scaler=scaler,
            target_scaler=self.target_scaler,
        )

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
