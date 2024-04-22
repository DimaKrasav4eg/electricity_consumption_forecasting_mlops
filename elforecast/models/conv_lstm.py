import torch.nn as nn


class ConvLSTM(nn.Module):
    def __init__(self, params: dict):
        super(ConvLSTM, self).__init__()
        self.cnn = nn.ModuleList()
        self.conv_layers = params["conv"]["nlayers"]
        for i in range(self.conv_layers):
            input_size = params["nfeats"]
            if i > 0:
                input_size = params["conv"]["out_channels"][i - 1]
            out_size = params["conv"]["out_channels"][i]
            kernel = params["conv"]["kernels"][i]
            conv_layer = nn.Sequential(
                nn.Conv1d(
                    input_size,
                    out_size,
                    kernel,
                    stride=params["conv"]["strides"][i],
                    padding=params["conv"]["paddings"][i],
                ),
                nn.BatchNorm1d(out_size),
                nn.ReLU(),
                nn.MaxPool1d(
                    params["conv"]["max_pool"]["kernel_size"][i],
                    params["conv"]["max_pool"]["stride"][i],
                ),
                nn.Dropout(params["conv"]["dropout"]),
            )

            self.cnn.append(conv_layer)
        self.lstm = nn.LSTM(
            params["conv"]["out_channels"][-1],
            params["rnn"]["hidden_size"],
            params["rnn"]["nlayers"],
            batch_first=True,
            dropout=params["rnn"]["dropout"],
        )
        self.fc = nn.ModuleList()
        for i in range(params["lin"]["nlayers"]):
            input_size = params["rnn"]["hidden_size"]
            if i > 0:
                input_size = params["lin"]["out_sizes"][i - 1]
            output_size = params["lseq"]
            if i < params["lin"]["nlayers"] - 1:
                output_size = params["lin"]["out_sizes"][i]
            lin_layer = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.BatchNorm1d(output_size),
                nn.ReLU(),
                nn.Dropout(params["lin"]["dropout"]),
            )
            self.fc.append(lin_layer)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for conv_layer in self.cnn:
            x = conv_layer(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        i = 0
        for lin_layer in self.fc:
            x = lin_layer(x)
            i += 1
        return x
