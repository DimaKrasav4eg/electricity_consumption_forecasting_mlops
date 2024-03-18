import os
import warnings

warnings.filterwarnings("ignore", \
                        message="Failed to load image Python extension")

import torch, torch.nn as nn
from tqdm import tqdm
from elforecast.data import ElForecastDataset
from elforecast.models import ConvLSTM
from torch.utils.data import DataLoader, Subset
from torchmetrics.regression import MeanAbsoluteError, R2Score

def get_tvt_dataloader(dataset:ElForecastDataset, ratio:list, batch_size:int):
    assert len(ratio) == 3

    ldataset = len(dataset)
    train_ratio = ratio[0]
    val_ratio   = ratio[1]
    test_ratio  = ratio[2]

    train_size = int(train_ratio * ldataset)
    val_size = int(val_ratio * ldataset)
    test_size = ldataset - train_size - val_size

    indices = list(range(len(dataset)))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[-test_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset   = Subset(dataset, val_indices)
    test_dataset  = Subset(dataset, test_indices)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train(model, train_loader, val_loader, opt, criterion, \
            n_epochs=100, lr=1e-3, train_history=[], val_history=[]):
    for epoch in tqdm(range(n_epochs)):
        model.train()

        train_batch_history = 0
        val_batch_history = 0
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
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred.view(-1), y_batch.view(-1))
                val_batch_history += loss.data.numpy()

        val_batch_history /= len(val_loader)
        val_history.append(val_batch_history)

        tqdm.write(f"Epoch {epoch+1}/{n_epochs}, \
                     Train Loss: {train_batch_history:.3f}, \
                     Validation Loss: {val_batch_history:.3f}")

    return model, (train_history, val_history)

def test(model, test_loader:ElForecastDataset):
    mae = 0
    r2 = 0
    mean_absolute_error = MeanAbsoluteError()
    r2score = R2Score()
    for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            mae += mean_absolute_error(y_pred.view(-1), y_batch.view(-1))
            r2  += r2score(y_pred.view(-1), y_batch.view(-1))
    return mae / len(test_loader), r2 / len(test_loader)

def save_model(model, path, name):
    path_exist = os.path.isdir(path)
    if not path_exist:
        os.mkdir(path)
    torch.save(model.state_dict(), os.path.join(path, name))


DATA_PATH = '.data/train.csv'
SEQ_LEN = 24
NFEATURES = 11
OUT_FUTURES = 64
OUT_CHANNEL_1 = 128
OUT_CHANNEL_2 = 256
HIDDEN_SIZE = 1024
NUM_LAYERS = 3
BATCH_SIZE = 50
MODEL_SAVE_PATH = 'checkpoints'
DATASET = ElForecastDataset(DATA_PATH, SEQ_LEN, step=SEQ_LEN//2)

if __name__ == "__main__":
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    lr = 1e-3
    n_epochs = 50

    dataset = DATASET
    nfeatures = dataset.get_nfeatures()

    train_loader, val_loader, test_loader = \
        get_tvt_dataloader(dataset, [train_ratio, val_ratio, test_ratio], BATCH_SIZE)

    model = ConvLSTM(NFEATURES, SEQ_LEN, OUT_FUTURES, 
                                OUT_CHANNEL_1, 
                                OUT_CHANNEL_2,
                                HIDDEN_SIZE, 
                                NUM_LAYERS)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    model, _ = train(model, train_loader, 
                     val_loader, opt, criterion, n_epochs=n_epochs)
    mae, r2 = test(model, test_loader)
    print(mae.item(), r2.item())
    save_model(model, MODEL_SAVE_PATH, 'cnn_lstm.pt')