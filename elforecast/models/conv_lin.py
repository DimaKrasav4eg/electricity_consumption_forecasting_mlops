import torch.nn as nn
from omegaconf.dictconfig import DictConfig


class ConvLin(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        params = cfg.model.params
        self.cnn = nn.Sequential(
            ConvBlock(
                cfg.data.nfeats,
                params.conv.n_channels[0],
                params.conv.kernels[0],
                params.conv.padding[0],
                params.conv.maxpool,
                params.conv.dropout,
            ),
            ConvBlock(
                params.conv.n_channels[0],
                params.conv.n_channels[1],
                params.conv.kernels[1],
                params.conv.padding[1],
                params.conv.maxpool,
                params.conv.dropout,
            ),
            ConvBlock(
                params.conv.n_channels[1],
                params.conv.n_channels[2],
                params.conv.kernels[2],
                params.conv.padding[2],
                params.conv.maxpool,
                params.conv.dropout,
            ),
            ConvBlock(
                params.conv.n_channels[2],
                params.conv.n_channels[3],
                params.conv.kernels[3],
                params.conv.padding[3],
                params.conv.maxpool,
                params.conv.dropout,
            ),
        )

        self.lin = nn.Sequential(
            nn.Linear(params.conv.n_channels[3], params.lin.size[0]),
            nn.BatchNorm1d(params.lin.size[0]),
            nn.ReLU(),
            nn.Dropout(params.lin.dropout),
            nn.Linear(params.lin.size[0], params.lin.size[1]),
            nn.BatchNorm1d(params.lin.size[1]),
            nn.ReLU(),
            nn.Dropout(params.lin.dropout),
            nn.Linear(params.lin.size[1], cfg.data.lseq),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.cnn(x)
        out = out.squeeze(2)
        out = self.lin(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, input_size, out_size, kernel_size, padding, pool, dropout):
        super().__init__()
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=out_size,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(out_size),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=pool[0],
                stride=pool[1],
            ),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.conv(x)


class LinBlock(nn.Module):
    def __init__(self, input_size, output_size, lin_size, dropout):
        super().__init__()

        self.lin = nn.Sequential(
            nn.Linear(input_size, lin_size[0]),
            nn.BatchNorm1d(lin_size[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lin_size[0], lin_size[1]),
            nn.BatchNorm1d(lin_size[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lin_size[1], output_size),
        )

    def forward(self, x):
        return self.lin(x)
