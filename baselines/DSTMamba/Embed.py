import torch.nn as nn


class DataEmbedding(nn.Module):
    def __init__(self, 
                 history_seq_len, 
                 d_model, 
                 dropout):
        super(DataEmbedding, self).__init__()

        self.ValueEmb = nn.Linear(history_seq_len, d_model)
        self.Dropout = nn.Dropout(p=dropout)

    
    def forward(self, x_in):
        # x_in: (batch_size, history_seq_len <-> num_channels)
        x_in = x_in.permute(0, 2, 1)

        x_emb = self.ValueEmb(x_in) # x_emb: (batch_size, num_channels, d_model)
        x_emb = self.Dropout(x_emb)

        return x_emb