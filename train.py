import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", \
                        message="Failed to load image Python extension")

import torch, torch.nn as nn
from tqdm import tqdm
from elforecast.data import ElForecastDataset
from elforecast.models import ConvLSTM
from torch.utils.data import DataLoader
from torch.utils.data import random_split



def train(model, train_dataloader, val_data_loader, opt, criterion, \
            n_epochs=100, lr=1e-3, train_history=[], val_history=[]):
    for epoch in tqdm(range(n_epochs)):
        model.train()

        train_batch_history = 0
        val_batch_history = 0
        for X_batch, y_batch in train_dataloader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_batch_history += loss.data.numpy()
        train_batch_history /= train_dataloader.batch_size
        train_history.append(train_batch_history)

        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_dataloader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_batch_history += loss.data.numpy()

        val_batch_history /= val_data_loader.batch_size
        val_history.append(val_batch_history)

        tqdm.write(f"Epoch {epoch+1}/{n_epochs}, \
                     Train Loss: {train_batch_history:.3f}, \
                     Validation Loss: {val_batch_history:.3f}")

    return model, (train_history, val_history)

def save_model(model, path, name):
    path_exist = os.path.isdir(path)
    if not path_exist:
        os.mkdir(path)
    torch.save(model.state_dict(), os.path.join(path, name))


DATA_PATH = '.data/train.csv'
SEQ_LEN = 24
OUT_CHANNEL_1 = 128
OUT_CHANNEL_2 = 256
HIDDEN_SIZE = 1024
NUM_LAYERS = 3
BATCH_SIZE = 50
MODEL_SAVE_PATH = 'checkpoints'

train_size = 0.8
val_size = 0.1
test_size = 0.1

lr = 1e-3
n_epochs = 1

dataset = ElForecastDataset(DATA_PATH, SEQ_LEN, step=SEQ_LEN)
nfeatures = dataset.get_nfeatures()

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ConvLSTM(nfeatures, OUT_CHANNEL_1, OUT_CHANNEL_2, HIDDEN_SIZE, NUM_LAYERS)

opt = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.L1Loss()

model, _ = train(model, train_dataloader, val_dataloader, opt, criterion, n_epochs=n_epochs)
save_model(model, MODEL_SAVE_PATH, 'cnn_lstm.pt')