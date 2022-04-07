import torch.nn as nn
import torch
from tst.encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, num_classes=11, input_size=2, time_steps=633, model_dimension=128, n_heads=16, dropout=0.3,
                 n_encoders=3):
        super(Transformer, self).__init__()
        # first embedding layer
        self.embedding = nn.Linear(input_size, model_dimension)
        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=dropout)

        self.pos_embedding = nn.Embedding(time_steps, model_dimension)

        self.encoder_layer = nn.TransformerEncoderLayer(model_dimension, n_heads, dropout=dropout, dim_feedforward=256,
                                                        activation='gelu')
        self.encoder = nn.TransformerEncoder(self.encoder_layer, n_encoders, norm=nn.LayerNorm(model_dimension))
        self.model_d = model_dimension

        self.avg_pool = nn.AdaptiveAvgPool2d((self.model_d, 1))

        self.fc = nn.Linear(model_dimension*time_steps, num_classes)

    def forward(self, input):
        input = input.transpose(1, 2)  # batch, time, features
        input = self.dropout(self.norm(self.embedding(input)))

        B = input.shape[0]
        T = input.shape[1]
        pos = torch.arange(0, T).unsqueeze(0).repeat(B, 1).to(input.device)

        input.add_(self.pos_embedding(pos))

        encoding = self.encoder(input)

        # concat vectors into size model_dimension * time_steps
        encoding = encoding.reshape((B, -1))

        output = self.fc(self.dropout(encoding))

        return output


class TransformerTST(nn.Module):
    def __init__(self, num_classes=11, input_size=2, time_steps=633, model_dimension=128, n_heads=16, dropout=0.5,
                 n_encoders=3):
        super(TransformerTST, self).__init__()

        self.embedding = nn.Linear(input_size, model_dimension)
        self.norm = nn.LayerNorm(model_dimension)
        self.time_steps = time_steps

        self.positional_embedding = nn.Embedding(time_steps, model_dimension)

        self.layers_encoding = nn.ModuleList([Encoder(model_dimension, q=model_dimension, v=model_dimension,
                                                      h=n_heads, dropout=dropout) for _ in range(n_encoders)])

        self.output = nn.Linear(time_steps * model_dimension, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, pos):
        input = input.transpose(1, 2)
        input = self.norm(self.dropout(self.embedding(input)))

        input.add_(self.positional_embedding(pos))

        for layer in self.layers_encoding:
            input = layer(input)

        input = input.reshape((input.shape[0], -1))

        input = self.output(self.dropout(input))

        return input





