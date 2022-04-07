import torch
import torch.nn as nn
from models.model_utils import Conv1dSame


class ConvTimeNet(nn.Module):

    def __init__(self, num_classes=9):
        super(ConvTimeNet, self).__init__()
        self.kernel_sizes = (4, 8, 16, 32, 64)
        self.num_filters = 32  #33
        self.conv_block1 = ConvBlock(input_size=2, kernel_sizes=self.kernel_sizes, num_filters=self.num_filters)
        self.conv_block2 = ConvBlock(input_size=5*self.num_filters, kernel_sizes=self.kernel_sizes,
                                     num_filters=self.num_filters)
        self.conv_block3 = ConvBlock(input_size=5*self.num_filters, kernel_sizes=self.kernel_sizes,
                                     num_filters=self.num_filters)
        self.conv_block4 = ConvBlock(input_size=5*self.num_filters, kernel_sizes=self.kernel_sizes,
                                     num_filters=self.num_filters)
        self.residual_connection1 = nn.Sequential(Conv1dSame(in_channels=2, out_channels=5*self.num_filters,
                                                             kernel_size=1, bias=False),
                                                  nn.BatchNorm1d(num_features=5*self.num_filters))
        self.residual_connection2 = nn.Sequential(Conv1dSame(in_channels=5*self.num_filters,
                                                             out_channels=5 * self.num_filters,
                                                  kernel_size=1, bias=False),
                                                  nn.BatchNorm1d(num_features=5*self.num_filters))
        self.activation = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((self.num_filters, 1))
        self.fc = nn.Linear(in_features=self.num_filters, out_features=num_classes, bias=True)

    def forward(self, my_input):
        out_conv1 = self.activation(self.conv_block1(my_input))
        out_conv2 = self.conv_block2(out_conv1)
        out_res1 = self.residual_connection1(my_input)
        out_conv2 += out_res1
        out = self.activation(out_conv2)
        out_conv3 = self.activation(self.conv_block3(out))
        out_conv4 = self.conv_block4(out_conv3)
        out_res2 = self.residual_connection2(out)
        out_conv4 += out_res2
        out = self.activation(out_conv4)
        out = self.global_avg_pool(out).squeeze(2)
        out = self.fc(out)

        return out


class ConvBlock(nn.Module):

    def __init__(self, input_size=2, kernel_sizes=(4, 8, 16, 32, 64), num_filters=33):
        super(ConvBlock, self).__init__()
        self.conv1 = Conv1dSame(input_size, num_filters, kernel_sizes[0], bias=False)
        self.conv2 = Conv1dSame(input_size, num_filters, kernel_sizes[1], bias=False)
        self.conv3 = Conv1dSame(input_size, num_filters, kernel_sizes[2], bias=False)
        self.conv4 = Conv1dSame(input_size, num_filters, kernel_sizes[3], bias=False)
        self.conv5 = Conv1dSame(input_size, num_filters, kernel_sizes[4], bias=False)
        self.batch_norm = nn.BatchNorm1d(5*num_filters)

    def forward(self, my_input):
        out_conv1 = self.conv1(my_input)
        out_conv2 = self.conv2(my_input)
        out_conv3 = self.conv3(my_input)
        out_conv4 = self.conv4(my_input)
        out_conv5 = self.conv5(my_input)

        out_concat = torch.cat((out_conv1, out_conv2, out_conv3, out_conv4, out_conv5), dim=1)
        out = self.batch_norm(out_concat)

        return out
