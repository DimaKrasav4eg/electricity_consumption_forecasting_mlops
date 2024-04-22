import os
import warnings

import hydra
import numpy as np
import pandas as pd
import torch
from elforecast.data import ElForecastDataset, data_preprocessing, normalize_data
from elforecast.models import ConvLSTM
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


warnings.filterwarnings("ignore", message="Failed to load image Python extension")


def prediction(model, old_data, new_data, mean, std, pred_len):
    model.eval()
    with torch.no_grad():
        all_pred = np.zeros(pred_len, dtype=np.float32)
        old_data, _ = normalize_data(old_data, mean, std)
        for i in tqdm(range(pred_len)):
            pred = model(old_data)
            all_pred[i] = pred[0, -1]
            next_data = torch.tensor(new_data[i, :-1])
            next_data = torch.cat((next_data, pred[0, -1].view(-1)))
            next_data, _ = normalize_data(next_data, mean, std)
            old_data = torch.cat(
                (old_data[:, 1:, :], next_data.unsqueeze(0).unsqueeze(0)), dim=1
            )
    return all_pred


def unscale(data, mean, std):
    return data * std + mean


def save_predictions(date_col, predictions, path, name):
    path_exist = os.path.isdir(path)
    if not path_exist:
        os.mkdir(path)
    df_pred = pd.DataFrame({"ST": predictions})
    df_pred["Date"] = date_col
    df_pred = df_pred[["Date", "ST"]]
    df_pred.to_csv(os.path.join(path, name), index=False)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    dataset = ElForecastDataset(
        cfg["data"]["train_path"], cfg["data"]["lseq"], step=cfg["data"]["step"]
    )

    nfeats = dataset.get_nfeatures()
    model_params = OmegaConf.to_container(cfg["model"]["params"])
    model_params.update({"lseq": cfg["data"]["lseq"], "nfeats": nfeats})
    model = ConvLSTM(model_params)
    model.load_state_dict(
        torch.load(os.path.join(cfg["training"]["save_path"], cfg["model"]["name"]))
    )

    old_data_pd = pd.read_csv(cfg["data"]["train_path"])[-cfg["data"]["lseq"] :]
    old_data = torch.tensor(
        data_preprocessing(old_data_pd).values[:, 1:].astype(np.float32)
    ).unsqueeze(0)

    new_data_df = pd.read_csv(cfg["data"]["test_path"])
    date_col = new_data_df["Date"]
    new_data_df = data_preprocessing(new_data_df, fill_target=True, target_name="ST")
    new_data = new_data_df.values[:, 1:].astype(np.float32)

    mean = dataset.get_mean()
    std = dataset.get_std()
    preds = prediction(
        model, old_data, new_data, torch.tensor(mean), torch.tensor(std), len(new_data)
    )
    preds = unscale(preds, mean[-1], std[-1])
    save_predictions(date_col, preds, cfg["data"]["submit_path"], "submission.csv")


if __name__ == "__main__":
    main()
