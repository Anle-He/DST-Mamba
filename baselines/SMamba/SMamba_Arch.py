import torch
import torch.nn as nn
from mamba_ssm import Mamba

from .Embed import DataEmbedding_inverted
from .Mamba_Enc import Encoder, EncoderLayer


class S_Mamba(nn.Module):
    def __init__(self, **model_args):
        super(S_Mamba, self).__init__()

        self.history_seq_len = model_args['history_seq_len']
        self.future_seq_len = model_args['future_seq_len']

        self.use_norm = model_args['use_norm']

        self.enc_embedding = DataEmbedding_inverted(model_args['history_seq_len'], model_args['d_model'], model_args['emd_dropout'])

        self.Encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(
                        d_model=model_args['d_model'],
                        d_state=model_args['d_state'],
                        d_conv=model_args['d_conv'],
                        expand=model_args['expand']
                    ),
                    Mamba(
                        d_model=model_args['d_model'],
                        d_state=model_args['d_state'],
                        d_conv=model_args['d_conv'],
                        expand=model_args['expand']
                    ),
                    model_args['d_model'],
                    model_args['d_ff'],
                    dropout=model_args['ffn_dropout'],
                    activation=model_args['ffn_activation']
                ) for layer in range(model_args['e_layers'])
            ],
            norm_layer=nn.LayerNorm(model_args['d_model'])
        )

        self.projector = nn.Linear(model_args['d_model'], model_args['future_seq_len'], bias=True)


    def forward(self, 
                history_data: torch.Tensor) -> torch.Tensor:
        
        x_enc = history_data[..., 0]

        if self.use_norm: 
            # Normalization from Non-Staionary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding: [B, T, N] -> [B, N, E]
        emb_out = self.enc_embedding(x_enc)

        # Encoder: [B, N, E] -> [B, N, E]
        enc_out = self.Encoder(emb_out)

        # Projector: [B, N, E] -> [B, N, T] -> [B, T, N]
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.future_seq_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.future_seq_len, 1))

        prediction = dec_out.unsqueeze(-1)

        return prediction