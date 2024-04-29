import os
import warnings

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from elforecast.data import ElForecastDataModule, data_preprocessing
from elforecast.models import ConvLin
from omegaconf import DictConfig
from pandas import DataFrame
from torch.utils.data import DataLoader
from torchmetrics.regression import MeanAbsoluteError, R2Score


warnings.filterwarnings("ignore", message="Failed to load image Python extension")


def pred(model: nn.Module, loader: DataLoader) -> np.array:
    preds = []
    model.eval()
    with torch.no_grad():
        for i, (X, _) in enumerate(loader):
            pred = model(X)
            if i == 0:
                preds.append(pred[0, :].reshape(-1).cpu().numpy())
                preds.append(pred[:, -1].reshape(-1).cpu().numpy())
                continue
            preds.append(pred[:, -1].reshape(-1).cpu().numpy())
    return np.concatenate(preds)


def save_predictions(
    date_col: DataFrame,
    date_name: str,
    predictions: DataFrame,
    preds_name: str,
    path: str,
    name: str,
) -> None:
    path_exist = os.path.isdir(path)
    if not path_exist:
        os.mkdir(path)
    df_pred = pd.DataFrame({preds_name: predictions})
    df_pred[date_name] = date_col
    df_pred = df_pred[[date_name, preds_name]]
    df_pred.to_csv(os.path.join(path, name), index=False)


def test(path_pred: str, path_ans: str, cfg: DictConfig):
    df_preds = pd.read_csv(path_pred)
    df_ans = pd.read_csv(path_ans)
    preds = torch.tensor(df_preds.values[:, 1].astype(np.float32), dtype=torch.float32)
    ans = torch.tensor(df_ans.values[:, 1].astype(np.float32), dtype=torch.float32)
    mae = MeanAbsoluteError()
    r2 = R2Score()
    preds_day, ans_day = preds[: cfg.data.hours_in_day], ans[: cfg.data.hours_in_day]
    preds_month, ans_month = (
        preds[: cfg.data.hours_in_month],
        ans[: cfg.data.hours_in_month],
    )
    loss_mae_day = mae(preds_day, ans_day)
    loss_mae_month = mae(preds_month, ans_month)
    loss_mae_all = mae(preds, ans)
    loss_r2_day = r2(preds_day, ans_day)
    loss_r2_month = r2(preds_month, ans_month)
    loss_r2_all = r2(preds, ans)
    return (
        loss_mae_day,
        loss_mae_month,
        loss_mae_all,
        loss_r2_day,
        loss_r2_month,
        loss_r2_all,
    )


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    model = ConvLin.load_from_checkpoint(
        os.path.join(
            cfg.artifacts.checkpoint.dirpath,
            cfg.artifacts.experiment_name,
            cfg.artifacts.checkpoint.filename,
        )
        + cfg.artifacts.checkpoint.file_ext
    )
    dm = ElForecastDataModule(cfg)
    dm.setup()

    df_test = pd.read_csv(cfg.data.test_path)
    date_col = df_test[cfg.data.date]
    df_test = data_preprocessing(
        df_test, cfg.data.date, cfg.data.target, cfg.data.used_features, fill_target=True
    )
    loader = dm.predict_dataloader()
    scaler = dm.get_target_scaler()
    preds = pred(model, loader)
    preds = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(-1)
    save_predictions(
        date_col,
        cfg.data.date,
        preds,
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
