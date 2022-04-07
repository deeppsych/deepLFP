import torch
import torch.nn as nn
import numpy as np
from models.model_utils import Conv1dSame


class ChronoNet(nn.Module):

    def __init__(self, num_classes=9):
        super(ChronoNet, self).__init__()
        self.kernel_sizes = (2, 4, 8)  # (2, 4, 8)  # should also try 4,8,16 or 4,8,32
        self.num_filters = 32
        self.num_classes = num_classes
        self.inception_block1 = InceptionBlock(input_size=2, kernel_sizes=self.kernel_sizes,
                                               num_filters=self.num_filters)
        self.inception_block2 = InceptionBlock(input_size=3*self.num_filters, kernel_sizes=self.kernel_sizes,
                                               num_filters=self.num_filters)
        self.inception_block3 = InceptionBlock(input_size=3 * self.num_filters, kernel_sizes=self.kernel_sizes,
                                               num_filters=self.num_filters)
        self.gru1 = nn.GRU(input_size=3*self.num_filters, hidden_size=self.num_filters, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=self.num_filters, hidden_size=self.num_filters, num_layers=1, batch_first=True)
        self.gru3 = nn.GRU(input_size=2*self.num_filters, hidden_size=self.num_filters, num_layers=1, batch_first=True)
        self.gru4 = nn.GRU(input_size=3*self.num_filters, hidden_size=self.num_filters, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=self.num_filters, out_features=self.num_classes)

    def forward(self, my_input):
        out = self.inception_block1(my_input)
        out = self.inception_block3(out)
        out_conv = self.inception_block3(out)

        out = out_conv.transpose(1, 2)  # need to swap dimensions so input is on the form [batch, length, features] for gru
        out_gru1, _ = self.gru1(out)
        out_gru2, _ = self.gru2(out_gru1)
        out_cat1 = torch.cat((out_gru1, out_gru2), dim=2)  #concatenate along feature axis
        out_gru3, _ = self.gru3(out_cat1)
        out_cat2 = torch.cat((out_gru1, out_gru2, out_gru3), dim=2)
        out, _ = self.gru4(out_cat2)
        out = out[:, -1, :]  # take output at last timepoint as input to my classification
        out = self.fc(out)

        return out


class InceptionBlock(nn.Module):

    def __init__(self, input_size=2, kernel_sizes=(2, 4, 8), num_filters=32):
        super(InceptionBlock, self).__init__()
        self.conv1 = Conv1dSame(input_size, num_filters, kernel_sizes[0], bias=False)
        self.conv2 = Conv1dSame(input_size, num_filters, kernel_sizes[1], bias=False)
        self.conv3 = Conv1dSame(input_size, num_filters, kernel_sizes[2], bias=False)
        self.batch_norm = nn.BatchNorm1d(3*num_filters)  # 3x because I will have concatenated the 3 conv layers
        self.activation = nn.ReLU()

    def forward(self, my_input):
        input_inception = my_input
        output1 = self.conv1(input_inception)
        output2 = self.conv2(input_inception)
        output3 = self.conv3(input_inception)

        output = torch.cat((output1, output2, output3), dim=1)
        output = self.batch_norm(output)
        output = self.activation(output)

        return output

