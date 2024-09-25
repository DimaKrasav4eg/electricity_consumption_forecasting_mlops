import torch.nn as nn
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig


class ConvResLike(nn.Module):
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
        )

        self.lin = nn.Linear(params.conv.n_channels[-1], cfg.data.lseq)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.cnn(x)
        out = out.squeeze(2)
        out = self.lin(out)
        return out


class ConvBlock(nn.Module):
    def __init__(
        self, inp_size, out_size, kernel_size, padding, pool, dropout, convstride=1
    ):
        super().__init__()
        self.inp_size = inp_size
        self.seq = nn.Sequential(
            nn.Conv1d(
                in_channels=inp_size,
                out_channels=out_size,
                kernel_size=kernel_size,
                padding=padding,
                stride=convstride,
            ),
            nn.BatchNorm1d(out_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.conv = nn.Conv1d(inp_size, out_size, 1, stride=convstride)
        self.maxpool = nn.AvgPool1d(pool[0], pool[1])

    def forward(self, x):
        out = self.seq(x)
        out = out + F.relu(self.conv(x))
        out = out.transpose(1, 2)
        out = self.maxpool(out)
        out = out.transpose(1, 2)
        return out
