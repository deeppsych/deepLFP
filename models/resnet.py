import torch.nn as nn
from models.model_utils import Conv1dSame


class ResNet(nn.Module):
    def __init__(self, num_classes=9):
        super(ResNet, self).__init__()
        self.res_block1 = ResidualBlock(2, 64, 64)
        self.res_block2 = ResidualBlock(64, 128, 128)
        self.res_block3 = ResidualBlock(128, 128, 128)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((128, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.res_block1(x)
        out = self.res_block2(out)
        out = self.res_block3(out)
        out = self.global_avg_pool(out).squeeze(2)
        out = self.fc(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv1dSame(in_channels, middle_channels, 8, bias=False)
        self.conv1_bn = nn.BatchNorm1d(middle_channels)
        self.relu = nn.ReLU()
        self.conv2 = Conv1dSame(middle_channels, middle_channels, 5, bias=False)
        self.conv2_bn = nn.BatchNorm1d(middle_channels)
        self.conv3 = Conv1dSame(middle_channels, out_channels, 3, bias=False)
        self.conv3_bn = nn.BatchNorm1d(out_channels)
        self.projection = nn.Sequential(Conv1dSame(in_channels, out_channels, 1, bias=False),
                                        nn.BatchNorm1d(out_channels))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.conv2_bn(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.conv3_bn(out)
        out += self.projection(residual)
        out = self.relu(out)

        return out
