import torch
import torch.nn as nn
import torch.functional as F

from einops import repeat


class SeriesEmbedding(nn.Module):
    def __init__(self,
                 history_seq_len,
                 num_channels,
                 d_model,
                 dropout,
                 add_se,
                 rank,
                 node_dim):
        super(SeriesEmbedding, self).__init__()

        self.add_se = add_se
        if self.add_se:
            
            self.rank = rank
            self.node_dim = node_dim 

            self.adapter = nn.Parameter(torch.empty(num_channels, d_model-self.node_dim, rank))
            # TODO: try other inits
            nn.init.xavier_uniform_(self.adapter)
            self.lora = nn.Linear(rank, self.node_dim, bias=False)

        else:
            self.node_dim = 0

        embed_dim = d_model - self.node_dim

        self.LinearTokenizer = nn.Linear(history_seq_len, embed_dim)
        self.Dropout = nn.Dropout(p=dropout)


    def forward(self, x_in):

        # (batch_size, history_seq_len <-> num_channels)
        x_in = x_in.permute(0, 2, 1)
        # x_emb: (batch_size, num_channels, embed_dim)
        x_emb = self.Dropout(self.LinearTokenizer(x_in))

        if self.add_se:

            B, _, _ = x_emb.shape

            adaptation = []
            adapter = F.relu(self.lora(self.adapter)) # [N, E-D, r] -> [N, E-D, D]
            
            adapter = adapter.permute(1, 2, 0)  # [E-D, D, N]
            adapter = repeat(adapter, 'D d n -> repeat D d n', repeat=B) # [B, E-D, D, N]
            
            x_emb = x_emb.transpose(1, 2) # (B, N, E-D) -> (B, E-D, N)
            adaptation.append(torch.einsum('bDn,bDdn->bdn', [x_emb, adapter]))  # [B, D, N]
            
            x_emb = torch.cat([x_emb] + adaptation, dim=1)  # [B, E, N]
            x_emb = x_emb.transpose(1, 2)  # [B, E, N] -> # [B, N, E]            

        return x_emb