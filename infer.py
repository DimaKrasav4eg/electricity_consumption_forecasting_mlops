import os
import warnings

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from elforecast.data import ElForecastDataModule, data_preprocessing
from elforecast.models import Selector
from omegaconf import DictConfig
from pandas import DataFrame
from torch.utils.data import DataLoader
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    R2Score,
)


warnings.filterwarnings("ignore", message="Failed to load image Python extension")


def pred(model: nn.Module, loader: DataLoader) -> np.array:
    preds = []
    model.eval()
    with torch.no_grad():
        for i, (X, _) in enumerate(loader):
            pred = model(X)
            # print(pred.shape, 'preddd')
            if i == 0:
                preds.append(pred[0, :].reshape(-1).cpu().numpy())
                preds.append(pred[1:, -1].reshape(-1).cpu().numpy())
                continue
            preds.append(pred[:, -1].reshape(-1).cpu().numpy())
    return np.concatenate(preds)


def save_predictions(
    date_name: str,
    df_test_original: DataFrame,
    predictions: DataFrame,
    preds_name: str,
    path: str,
    name: str,
) -> None:
    path_exist = os.path.isdir(path)
    if not path_exist:
        os.mkdir(path)
    df_test_original[date_name] = df_test_original.index.strftime("%Y-%m-%dT%H:%M:%SZ")
    df_test_original = df_test_original[date_name]
    df_pred = pd.merge(
        df_test_original, predictions, left_index=True, right_index=True, how="inner"
    )
    df_pred.to_csv(os.path.join(path, name), index=False)
    predictions.to_csv(os.path.join(path, "full_" + name), index=False)


def test(path_pred: str, path_ans: str, cfg: DictConfig):
    df_preds = pd.read_csv(path_pred)
    df_ans = pd.read_csv(path_ans)
    # print(df_preds.shape, df_ans.shape)
    preds = torch.tensor(df_preds.values[:, 1].astype(np.float32), dtype=torch.float32)
    ans = torch.tensor(df_ans.values[:, 1].astype(np.float32), dtype=torch.float32)
    mae = MeanAbsoluteError()
    r2 = R2Score()
    mape = MeanAbsolutePercentageError()

    mae_list = []
    r2_list = []
    mape_list = []
    hours = sorted(cfg.data.hours.values())
    hours.append(ans.shape[0])
    for h in hours:
        mae_list.append(mae(preds[:h], ans[:h]))
        r2_list.append(r2(preds[:h], ans[:h]))
        mape_list.append(mape(preds[:h], ans[:h]))
    return (("mae", mae_list), ("r2", r2_list), ("mape", mape_list))


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    model = Selector.load_from_checkpoint(
        os.path.join(
            cfg.artifacts.checkpoint.dirpath,
            cfg.model.name,
            "last",
        )
        + cfg.artifacts.checkpoint.file_ext
    )
    dm = ElForecastDataModule(cfg)
    dm.setup()

    df_test = pd.read_csv(cfg.data.test_path)
    df_test_original = df_test.copy()
    # print('before', df_test_original.columns)
    df_test_original = data_preprocessing(
        df_test_original,
        cfg.data.date,
        cfg.data.target,
        [],
        fill_target=True,
        fill_date=False,
    )
    df_test_original = df_test_original[0]
    df_test = data_preprocessing(
        df_test, cfg.data.date, cfg.data.target, cfg.data.used_features, fill_target=True
    )
    df_test = df_test[0]
    loader = dm.predict_dataloader()
    scaler = dm.get_target_scaler()
    preds = pred(model, loader)
    preds = scaler.inverse_transform(preds.reshape(-1, 1))
    preds = np.sum(preds, axis=1)
    df_test[cfg.data.target] = preds
    df_test = df_test[cfg.data.target]
    save_predictions(
        cfg.data.date,
        df_test_original,
        df_test,
        cfg.data.target,
        cfg.data.submit_path,
        cfg.data.submit_name,
    )

    print(
        test(
            os.path.join(cfg.data.submit_path, cfg.data.submit_name),
            cfg.data.ans_path,
            cfg,
        )
    )


if __name__ == "__main__":
    main()
