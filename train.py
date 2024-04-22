import os
import warnings

import hydra
import torch
import torch.nn as nn
from elforecast.data import ElForecastDataset
from elforecast.models import ConvLSTM
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from torchmetrics.regression import MeanAbsoluteError, R2Score
from tqdm import tqdm


warnings.filterwarnings("ignore", message="Failed to load image Python extension")


def get_tvt_dataloader(dataset: ElForecastDataset, ratio: dict, batch_size: int):
    assert len(ratio) == 3

    ldataset = len(dataset)

    train_size = int(ratio["train"] * ldataset)
    val_size = int(ratio["val"] * ldataset)
    test_size = ldataset - train_size - val_size

    indices = list(range(len(dataset)))

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[-test_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train(
    model,
    train_loader,
    val_loader,
    opt,
    criterion,
    n_epochs=100,
    lr=1e-3,
    train_history=None,
    val_history=None,
):
    if train_history is None:
        train_history = []
    if val_history is None:
        val_history = []

    for epoch in tqdm(range(n_epochs)):
        model.train()

        train_batch_history = 0
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred.view(-1), y_batch.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_batch_history += loss.data.numpy()
        train_batch_history /= len(train_loader)
        train_history.append(train_batch_history)

        model.eval()
        with torch.no_grad():
            mae, r2 = test(model, val_loader)
        tqdm.write(
            f"Epoch {epoch+1}/{n_epochs}, \
                     Train Loss: {train_batch_history:.3f}, \
                     Val MAE: {mae:.3f}, \
                     Val R2: {r2:.3f}"
        )

    return model, (train_history, val_history)


def test(model, test_loader: ElForecastDataset):
    mae = 0
    r2 = 0
    mean_absolute_error = MeanAbsoluteError()
    r2score = R2Score()
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        mae += mean_absolute_error(y_pred.view(-1), y_batch.view(-1))
        r2 += r2score(y_pred.view(-1), y_batch.view(-1))
    return mae / len(test_loader), r2 / len(test_loader)


def save_model(model, path, name):
    path_exist = os.path.isdir(path)
    if not path_exist:
        os.mkdir(path)
    torch.save(model.state_dict(), os.path.join(path, name))


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    dataset = ElForecastDataset(
        cfg["data"]["train_path"], cfg["data"]["lseq"], step=cfg["data"]["step"]
    )

    ratios = OmegaConf.to_container(cfg["data"]["ratio"])
    train_loader, val_loader, test_loader = get_tvt_dataloader(
        dataset, ratios, cfg["training"]["batch_size"]
    )

    nfeats = dataset.get_nfeatures()
    model_params = OmegaConf.to_container(cfg["model"]["params"])
    model_params.update({"lseq": cfg["data"]["lseq"], "nfeats": nfeats})
    model = ConvLSTM(model_params)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    criterion = nn.L1Loss()

    model, _ = train(
        model,
        train_loader,
        val_loader,
        opt,
        criterion,
        n_epochs=cfg["training"]["n_epochs"],
    )
    mae, r2 = test(model, test_loader)
    print(mae.item(), r2.item())
    save_model(model, cfg["training"]["save_path"], cfg["model"]["name"])


if __name__ == "__main__":
    main()
