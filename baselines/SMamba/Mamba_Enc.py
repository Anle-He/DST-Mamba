import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
            self, 
            ssm_layers, 
            conv_layers=None, 
            norm_layer=None
        ):
        super(Encoder, self).__init__()

        self.ssm_layers = nn.ModuleList(ssm_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer


    def forward(self, x):
        # x: [batch_size, num_nodes, d_model]
        if self.conv_layers is not None:
            for i, (ssm_layer, conv_layer) in enumerate(zip(self.ssm_layers, self.conv_layers)):
                x = ssm_layer(x)
                x = conv_layer(x)
            x = self.ssm_layers[-1](x)
        else:
            for ssm_layer in self.ssm_layers:
                x = ssm_layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(
            self,
            ssm,
            ssm_r,
            d_model,
            d_ff=None,
            dropout=0.1,
            activation='relu'
        ):
        super(EncoderLayer, self).__init__()

        self.ssm = ssm
        self.ssm_r = ssm_r
        d_ff = d_ff or 4 * d_model

        # FFN
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu


    def forward(self, x):
        new_x = self.ssm(x) + self.ssm_r(x.flip(dims=[1])).flip(dims=[1])

        x = x + new_x
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)