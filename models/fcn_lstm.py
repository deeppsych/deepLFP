import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from models.fcn import Features


class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=True):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        stdv = 1.0/np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs):
        if self.batch_first:
            batch_size = inputs.size()[0]
        else:
            batch_size = inputs.size()[1]

        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(dim=0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1)  # (batch_size, hidden_size, 1)
                            )

        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)

        # apply weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get final fixed vector representations
        representations = weighted.sum(1).squeeze()

        return representations, attentions


class ALSTM(nn.Module):
    def __init__(self, hidden_size=8, dropout=0.8, num_classes=4):
        super(ALSTM, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size)
        self.attention = Attention(hidden_size, batch_first=True)
        
        self.features = Features()
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(256 + hidden_size, num_classes)
        
    def forward(self, inputs):

        x = inputs.transpose(1, 2)
        x, (h_n, c_n) = self.lstm(x)
        x, _ = self.attention(x)
        x = self.dropout(x)
        
        y = self.features(inputs)
        y = self.global_avg_pooling(y).squeeze(2)
        
        out = torch.cat((x, y),dim=1)

        out = self.fc(out)

        return out
        
        
        return x