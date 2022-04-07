import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self, input_shape, dropout):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=9, stride=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1)
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=dropout)
        n_size = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(n_size, 2040)
        self.fc2 = nn.Linear(2040, 9)

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.rand(bs, *shape)
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc2(self.dropout(self.relu(self.fc1(x))))
        return x

