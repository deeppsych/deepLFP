import torch
import torch.nn as nn
from models.model_utils import Conv1dSame


class ResidualBlock(nn.Module):
    def __init__(self, input_size=2, kernel_sizes=[40, 20, 10], num_filters=32, is_first=True):
        super(ResidualBlock, self).__init__()
        if is_first:
            self.inceptions_blocks = nn.ModuleList([InceptionBlock(input_size=input_size, kernel_sizes=kernel_sizes,
                                                               num_filters=num_filters)])
            self.inceptions_blocks.extend([InceptionBlock(input_size=4*num_filters, kernel_sizes=kernel_sizes,
                                                          num_filters=num_filters) for i in range(2)])
            self.residual_connection = nn.Sequential(Conv1dSame(input_size, 4 * num_filters, kernel_size=1,
                                                               bias=False), nn.BatchNorm1d(4 * num_filters))

        else:
            self.inceptions_blocks = nn.ModuleList([InceptionBlock(input_size=4*num_filters, kernel_sizes=kernel_sizes,
                                                                   num_filters=num_filters) for i in range(3)])
            self.residual_connection = nn.Sequential(Conv1dSame(4 * num_filters, 4 * num_filters, kernel_size=1,
                                                                bias=False), nn.BatchNorm1d(4 * num_filters))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        input_residual = input

        for layer in self.inceptions_blocks:
            input = layer(input)
        input += self.residual_connection(input_residual)
        input = self.relu(input)

        return input


class InceptionTime(nn.Module):

    def __init__(self, num_classes=9, kernel=40, num_blocks=3, num_filters=32, input_block=True):
        super(InceptionTime, self).__init__()
        self.kernel_sizes = [kernel // (2 ** i) for i in range(3)]
        self.num_filters = num_filters
        self.input_block = input_block

        if input_block:
            self.input_block = nn.Sequential(nn.Conv2d(in_channels=1, kernel_size=[1, 5],
                                                       out_channels=64, bias=False, padding=[0, 2]), nn.BatchNorm2d(num_features=64))
            self.bottleneck = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=1, bias=False,
                                                      kernel_size=1), nn.BatchNorm2d(1))

        self.residual_blocks = nn.ModuleList([ResidualBlock(input_size=2, kernel_sizes=self.kernel_sizes,
                                                            num_filters=self.num_filters, is_first=True)])
        self.residual_blocks.extend([ResidualBlock(input_size=4*self.num_filters, kernel_sizes=self.kernel_sizes,
                                                      num_filters=self.num_filters) for i in range(num_blocks - 1)])
        self.fc = nn.Linear(self.num_filters*4, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, my_input):
        if self.input_block:
            my_input = self.relu(self.input_block(my_input[:, None, ...]))
            my_input = self.relu(self.bottleneck(my_input)).squeeze()

        for block in self.residual_blocks:
            my_input = block(my_input)
        features = my_input
        my_input = features.mean(axis=2)
        my_input = self.fc(my_input).squeeze()

        return my_input, features


class InceptionBlock(nn.Module):

    def __init__(self, input_size=2, kernel_sizes=(40, 20, 10), num_filters=32):
        super(InceptionBlock, self).__init__()
        self.conv1 = Conv1dSame(input_size, num_filters, kernel_sizes[0], bias=False)
        self.conv2 = Conv1dSame(input_size, num_filters, kernel_sizes[1], bias=False)
        self.conv3 = Conv1dSame(input_size, num_filters, kernel_sizes[2], bias=False)
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv1dSame(input_size, num_filters, kernel_size=1, bias=False)
        self.batch_norm = nn.BatchNorm1d(4*num_filters)  # 4x because I will have concatenated the 4 conv layers
        self.activation = nn.ReLU()

    def forward(self, my_input):
        input_inception = my_input
        output1 = self.conv1(input_inception)
        output2 = self.conv2(input_inception)
        output3 = self.conv3(input_inception)
        output4 = self.conv4(self.max_pool(my_input))

        output = torch.cat((output1, output2, output3, output4), dim=1)
        output = self.batch_norm(output)
        output = self.activation(output)

        return output
