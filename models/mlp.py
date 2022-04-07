import torch.nn as nn
import torch
from models.model_utils import Conv1dSame


class MLP(nn.Module):
    def __init__(self, timeseries_length, n_classes):
        super(MLP, self).__init__()
        self.bottleneck = Conv1dSame(2, 1, 1)
        self.fc1 = nn.Linear(in_features=timeseries_length, out_features=500)
        self.fc1_bn = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(in_features=500, out_features=50)
        self.fc2_bn = nn.BatchNorm1d(50)
        self.fc3 = nn.Linear(in_features=50, out_features=n_classes)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        x = self.relu(self.fc1_bn(self.fc1(torch.squeeze(self.bottleneck(x)))))
        x = self.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x



