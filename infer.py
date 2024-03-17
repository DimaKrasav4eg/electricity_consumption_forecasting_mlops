import pandas as pd
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore", \
            message="Failed to load image Python extension")

import torch, torch.nn as nn
from torch.nn.functional import normalize
from train import DATA_PATH, OUT_CHANNEL_1, OUT_CHANNEL_2, \
                  HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN, DATASET
from tqdm import tqdm
from elforecast.data import ElForecastDataset, \
                            data_preprocessing, \
                            normalize_data
from elforecast.models import ConvLSTM

MODEL_PATH = 'checkpoints'
MODEL_NAME = 'cnn_lstm.pt'
SUBMIT_PATH = '.data'
NEW_DATA_PATH = '.data/test.csv'

def prediction(model, old_data, new_data, mean, std, pred_len):
    all_pred = np.zeros(pred_len, dtype=np.float32)
    old_data, _ = normalize_data(old_data, mean, std)
    for i in (range(pred_len)):
        pred = model(old_data)
        all_pred[i] = pred[0, -1]
        next_data = torch.tensor(new_data[i, :-1])
        next_data = torch.cat((next_data, pred[0, -1].view(-1)))
        next_data, _ = normalize_data(next_data, mean, std)
        old_data  = torch.cat((old_data[:, 1:, :], 
                               next_data.unsqueeze(0).unsqueeze(0)), 
                               dim=1)
    return all_pred
    
def unscale(data, mean, std):
    return data * std + mean

def save_predictions(date_col, predictions, path, name):
    path_exist = os.path.isdir(path)
    if not path_exist:
        os.mkdir(path)
    df_pred = pd.DataFrame({'ST':predictions})
    df_pred['Date'] = date_col
    df_pred = df_pred[['Date', 'ST']]
    df_pred.to_csv(os.path.join(path, name), index=False)

if __name__ == '__main__':
    model = ConvLSTM(SEQ_LEN, OUT_CHANNEL_1, OUT_CHANNEL_2, HIDDEN_SIZE, NUM_LAYERS)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, MODEL_NAME)))

    old_data_pd = pd.read_csv(DATA_PATH)[-SEQ_LEN:]
    old_data = torch.tensor(data_preprocessing(old_data_pd).values[:, 1:].astype(np.float32)).unsqueeze(0)

    new_data_df = pd.read_csv(NEW_DATA_PATH)
    date_col = new_data_df['Date']
    new_data_df = data_preprocessing(new_data_df, fill_target=True, target_name='ST')
    new_data = new_data_df.values[:, 1:].astype(np.float32)

    mean = DATASET.get_mean()
    std  = DATASET.get_std()
    preds = prediction(model, old_data, new_data, torch.tensor(mean), torch.tensor(std), len(new_data))
    preds = unscale(preds, mean[-1], std[-1])
    save_predictions(date_col, preds, SUBMIT_PATH, 'submission.csv')