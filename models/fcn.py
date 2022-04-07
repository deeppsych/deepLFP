import torch.nn as nn
from models.model_utils import Conv1dSame


class Features(nn.Module):
    """I separate the feature and classification part of my model so I can easily
    use the feature extraction separately from the rest for vizualisation"""
    def __init__(self):
        super(Features, self).__init__()
        self.conv1 = Conv1dSame(in_channels=2, out_channels=256, kernel_size=8, bias=False)
        self.conv1_bn = nn.BatchNorm1d(256)
        self.conv2 = Conv1dSame(in_channels=256, out_channels=128, kernel_size=5, bias=False)
        self.conv2_bn = nn.BatchNorm1d(128)
        self.conv3 = Conv1dSame(in_channels=128, out_channels=256, kernel_size=3, bias=False)
        self.conv3_bn = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        x = self.relu(self.conv1_bn(self.conv1(x)))
        x = self.relu(self.conv2_bn(self.conv2(x)))
        x = self.relu(self.conv3_bn(self.conv3(x)))
        return x


class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        self.features = Features()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x).squeeze(2)
        x = self.fc(x)
        return x


class LSTM_FCN(nn.Module):
    def __init__(self, num_classes):
        super(LSTM_FCN, self).__init__()
        self.features = Features()
        self.lstm = nn.LSTM(input_size=256, hidden_size=8, num_layers=1, batch_first=True,
                            bidirectional=False)
        self.dropout = nn.Dropout(p=0.8)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, input):
        x = self.features(input)
        x = x.transpose(1, 2)  # LSTM expects input [batch, sequence, features]
        x, (h_n, c_n) = self.lstm(x)
        h_n = h_n.view(-1, 8)
        x = self.fc(self.dropout(h_n))
        return x
