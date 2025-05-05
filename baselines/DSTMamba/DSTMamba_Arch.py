import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

from .RevIN import RevIN
from .SeriesDec import Temporal_Decomposition
from .SeriesMix import MultiScaleTrendMixing
from .Embed import SeriesEmbedding
from .MambaEnc import Encoder, EncoderLayer


class DSTMamba(nn.Module):
    def __init__(self, **model_args):
        super(DSTMamba, self).__init__()

        self.history_seq_len = model_args['history_seq_len']
        self.future_seq_len = model_args['future_seq_len']
        self.num_channels = model_args['num_channels']
        self.d_model = model_args['d_model']

        self.use_revin = model_args['use_revin']
        #self.norm_type = model_args['norm_type']
        self.emb_dropout = model_args['emb_dropout']
        self.add_se = model_args['add_se'] 
        self.rank = model_args['rank']
        self.node_dim = model_args['node_dim']     

        self.decom_type = model_args['decom_type']
        self.std_kernel = model_args['std_kernel']

        # TODO: Change the sequential mode of Mamba
        # self.mamba_mode = model_args['mamba_mode']
        # assert self.mamba_mode in ['bi-directional', 'uni-directional']

        self.e_layers = model_args['e_layers']
        self.d_state = model_args['d_state']
        self.d_conv = model_args['d_conv']
        self.expand = model_args['expand']

        self.d_ff = model_args['d_ff']
        self.ffn_dropout = model_args['ffn_dropout']
        self.ffn_activation = model_args['ffn_activation']

        self.ds_type = model_args['ds_type']
        assert self.ds_type in ['max', 'avg', 'conv']
        self.ds_layers = model_args['ds_layers']
        self.ds_window = model_args['ds_window']

        self.initial_tre_w = model_args['initial_tre_w']

        self.build()

    
    def build(self):

        self.revin = RevIN(num_features=self.num_channels, eps=1e-5, affine=True, subtract_last=False) if self.use_revin else None

        if self.decom_type == 'STD':
            self.decom = Temporal_Decomposition(self.std_kernel)

        self.embedding = SeriesEmbedding(self.history_seq_len, self.num_channels, self.d_model, self.emb_dropout, self.add_se, self.rank, self.node_dim)      

        self.encoder = Encoder(
            [
                EncoderLayer(
                    ssm     = Mamba(self.d_model, self.d_state, self.d_conv, self.expand),
                    ssm_r   = Mamba(self.d_model, self.d_state, self.d_conv, self.expand),
                    d_model = self.d_model,
                    d_ff    = self.d_ff,
                    dropout = self.ffn_dropout,
                    activation = self.ffn_activation
                ) for layer in range(self.e_layers)
            ],
            # TODO: Test the effectiveness of _RMSNorm_
            norm_layer = nn.LayerNorm(self.d_model)
        )

        self.projector = nn.Linear(self.d_model, self.future_seq_len, bias=True)  

        if self.ds_type == 'max':
            self.down_pool = nn.MaxPool1d(self.ds_window, return_indices=False)
        elif self.ds_type == 'avg':
            self.down_pool = nn.AvgPool1d(self.ds_window)
        elif self.ds_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            self.down_pool = nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels,
                                       kernel_size=3, padding=padding, stride=self.ds_window,
                                       padding_mode='circular', bias=False) 
            
        self.ms_mixing = MultiScaleTrendMixing(self.history_seq_len, self.future_seq_len, self.num_channels, 
                                               self.ds_layers, self.ds_window)
            
        self.linear_mappings = nn.ModuleList(
            [
                nn.Linear(self.history_seq_len//(self.ds_window**(l)), 
                          self.future_seq_len) for l in range(self.ds_layers+1)
            ]
        )

        self.tre_w = nn.Parameter(torch.FloatTensor([self.initial_tre_w]*self.num_channels), requires_grad=True)


    def forward(self,
                history_data: torch.Tensor) -> torch.Tensor:
        
        # x_in: [batch_size, seq_len, num_channels]
        x_in = history_data[..., 0]
        
        if self.use_revin:
            x_in = self.revin_layer(x_in, mode='norm')

        x_sea, _ = self.decom(x_in)

        x_emb = self.embedding(x_sea) # (B, L, N) -> (B, N, d_model)

        # TODO: Add Graph or reduce sequential bias
        # Encoder: [B, N, E] -> [B, N, E]
        enc_out = self.encoder(x_emb)

        x_sea_out = self.projector(enc_out).permute(0, 2, 1) # (B, N, d_model) -> (B, N, T) -> (B, T, N) 

        # Trend part: multi-scale processing
        ms_list = []
        ms_list.append(x_in) # [B, T, N]

        x_ms = x_in.permute(0, 2, 1)
        for layer in range(self.ds_layers):
            x_sampling = self.down_pool(x_ms) # [B, N, t_1/t_2/t_3 ... ]

            ms_list.append(x_sampling.permute(0, 2, 1))
            x_ms = x_sampling

        ms_trend_list = []
        for x in ms_list:
            _, x_tre = self.decom(x)
            ms_trend_list.append(x_tre)
        
        ms_trend_list = self.ms_mixing(ms_trend_list)

        # multi-scale mappings
        out_trend_list = []
        for i, trend in zip(range(len(ms_trend_list)), ms_trend_list):
            trend_out = self.linear_mappings[i](trend.permute(0, 2, 1)).permute(0, 2, 1)
            out_trend_list.append(trend_out)   

        x_tre_out = torch.stack(out_trend_list, dim=-1).sum(-1)         

        # Weighted Sum
        prediction = self.revin_layer(x_sea_out+self.tre_w*x_tre_out, mode='denorm')

        prediction = prediction.unsqueeze(-1)

        return prediction