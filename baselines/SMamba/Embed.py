import torch.nn as nn


class DataEmbedding_inverted(nn.Module):
    def __init__(
           self, 
           c_in, 
           d_model,
           dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()

        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    
    def forward(self, x):
        # x_in: [B, T, N]
        x = x.permute(0, 2, 1) # x -> [B, N, T]

        x = self.value_embedding(x)

        # x: [Batch Variate d_model]
        return self.dropout(x)