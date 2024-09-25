import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        params = cfg.model.params
        self.lstm = nn.LSTM(
            cfg.data.nfeats, params.hidden_size, params.nlayers, dropout=params.dropout
        )
        self.fc = nn.Linear(params.hidden_size, cfg.data.lseq)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


# (('mae', [tensor(67961.6641), tensor(49246.4805), tensor(57537.9336), tensor(80255.5156), tensor(79987.6328)]),
# ('r2', [tensor(-0.7240), tensor(-0.0769), tensor(-0.3784), tensor(-0.1982), tensor(-0.1915)]),
# ('mape', [tensor(0.2955), tensor(0.2657), tensor(0.6326), tensor(0.5022), tensor(0.4691)]))
