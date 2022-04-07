import torch.nn as nn


class ConvLSTM(nn.Module):
    def __init__(self, num_classes):
        super(ConvLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.bn3 = nn.BatchNorm1d(num_features=32)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(num_features=32)

        self.relu = nn.ReLU()

        self.LSTM = nn.LSTM(input_size=32, hidden_size=20, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc = nn.Linear(in_features=40, out_features=num_classes)

    def forward(self, inputs):
        x = self.pool1(self.relu(self.bn1(self.conv1(inputs))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout1(self.relu(self.bn4(self.conv4(x))))
        x = x.transpose(1, 2)  # LSTM expects input [batch, sequence, features]
        x, (h_n, c_n) = self.LSTM(x)
        x = x[:, -1, :]
        x = self.fc(self.dropout2(x))
        return x
