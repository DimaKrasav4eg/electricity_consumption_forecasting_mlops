import torch, torch.nn as nn

class ConvLSTM(nn.Module):
    def __init__(self, input_size, out_channels1, out_channels2, hidden_size, num_layers):
        super(ConvLSTM, self).__init__()
        out_channels1 = out_channels1
        out_channels2 = out_channels2
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_channels1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm1d(out_channels2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.1)
        )
        self.lstm = nn.LSTM(input_size=out_channels2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.cnn(x)
        out = out.permute(0, 2, 1)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out