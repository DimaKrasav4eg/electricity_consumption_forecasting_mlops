import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        params = cfg.model.params
        self.gru = nn.GRU(cfg.data.nfeats, params.hidden_size, params.nlayers)
        self.fc = nn.Linear(params.hidden_size, cfg.data.lseq)

    def forward(self, x):
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


# (('mae', [tensor(44429.5664), tensor(33696.0430), tensor(48623.8281), tensor(80979.7188)]),
# ('r2', [tensor(0.2244), tensor(0.5255), tensor(-0.0938), tensor(-0.2948)]),
# ('mape', [tensor(0.1937), tensor(0.2009), tensor(0.5290), tensor(0.4668)]))
