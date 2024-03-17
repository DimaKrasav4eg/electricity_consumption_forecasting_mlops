import torch, torch.nn as nn

class ConvLSTM(nn.Module):
    def __init__(self, input_size:int, out_channels1:int, out_channels2:int, hidden_size:int, num_layers:int):
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
        self.fc1 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out = self.cnn(x)
        out = out.permute(0, 2, 1)
        out, _ = self.lstm(out)
        out = out[:, -1, :]
        out = self.fc1(out)
        return out