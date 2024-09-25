import torch
import torch.nn as nn
from omegaconf.dictconfig import DictConfig


class ConvGRU(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        params = cfg.model.params
        self.conv1 = ConvBlock(
            cfg.data.nfeats,
            params.conv.n_channels[0],
            params.conv.kernels[0],
            params.conv.padding[0],
            params.conv.maxpool,
            params.conv.dropout,
            1,
        )
        self.conv2 = ConvBlock(
            cfg.data.nfeats,
            params.conv.n_channels[1],
            params.conv.kernels[1],
            params.conv.padding[1],
            params.conv.maxpool,
            params.conv.dropout,
            2,
        )
        self.conv3 = ConvBlock(
            cfg.data.nfeats,
            params.conv.n_channels[2],
            params.conv.kernels[2],
            params.conv.padding[2],
            params.conv.maxpool,
            params.conv.dropout,
            3,
        )
        self.gru = nn.GRU(96, 128, 3, dropout=0.2)
        self.fc = nn.Linear(128, 12)
        self.lin = nn.Sequential(
            nn.Linear(
                params.conv.n_channels[-1],
                params.lin.size[0],
            ),
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
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        print(out1.shape, out2.shape, out3.shape)
        out = torch.cat((out1, out2, out3), dim=1)
        out = out.transpose(1, 2)
        print(out.shape)
        out, _ = self.gru(out)
        out = out[:, -1, :]

        # out = out.squeeze(2)
        # out = self.lin(out)
        return self.fc(out)


class ConvBlock(nn.Module):
    def __init__(self, input_size, out_size, kernel_size, padding, pool, dropout, stride):
        super().__init__()
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=out_size,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.BatchNorm1d(out_size),
            nn.ReLU(),
            # nn.MaxPool1d(
            #     kernel_size=pool[0],
            #     stride=pool[1],
            # ),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.conv(x)


# (('mae', [tensor(43169.3477), tensor(100955.0469), tensor(87041.0312), tensor(87792.8672), tensor(82735.9531)]),
# ('r2', [tensor(0.3459), tensor(-2.1910), tensor(-1.5298), tensor(-0.1943), tensor(-0.0940)]),
# ('mape', [tensor(0.2098), tensor(0.7119), tensor(0.9499), tensor(0.6462), tensor(0.5729)]))

# class ConvBlock(nn.Module):
#     def __init__(self, inp_size, out_size, kernel_size, padding, pool_ker, stride, dropout, convstride=1):
#         super().__init__()
#         self.seq = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=inp_size,
#                 out_channels=out_size,
#                 kernel_size=kernel_size,
#                 padding=padding,
#                 stride=convstride
#             ),
#             nn.BatchNorm1d(out_size
#             ),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )
#         self.conv = nn.Conv1d(inp_size, out_size, 1, stride=convstride)
#         self.maxpool = nn.AvgPool1d(pool_ker, stride)
#     def forward(self, x):
#         out = self.seq(x)
#         out = out + F.relu(self.conv(x))
#         print(out.shape, 'convBlock')
#         out = out.transpose(1, 2)
#         out = self.maxpool(out)
#         out = out.transpose(1, 2)
#         return out
