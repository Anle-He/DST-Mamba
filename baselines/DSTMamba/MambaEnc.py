import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, ssm_layers, norm_layer):
        super(Encoder, self).__init__()

        self.ssm_layers = nn.ModuleList(ssm_layers)
        self.norm_layer = norm_layer


    def forward(self, x_emb):
        # x_emb: [batch_size, num_channels, d_model]
        x_enc = x_emb

        for ssm_layer in self.ssm_layers:
            x_enc = ssm_layer(x_enc)
            
        # TODO: Test the effectiveness of _RMSNorm_
        if self.norm_layer is not None:
            enc_out = self.norm_layer(x_enc)

        return enc_out
    

class EncoderLayer(nn.Module):
    def __init__(self, ssm, ssm_r, d_model, d_ff, dropout, activation):
        super(EncoderLayer, self).__init__()

        self.ssm = ssm
        self.ssm_r = ssm_r 

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)  

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)     

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu


    def forward(self, x_enc):
        if self.ssm_r is not None:
            ssm_out = self.ssm(x_enc) + self.ssm_r(x_enc.flip(dims=[1])).flip(dims=[1])    
        else:
            ssm_out = self.ssm(x_enc)

        out = x_enc = self.norm1(ssm_out)
        out = self.dropout(self.activation(self.conv1(out.transpose(-1, 1))))
        out = self.dropout(self.conv2(out).transpose(-1, 1))

        return self.norm2(out + x_enc)