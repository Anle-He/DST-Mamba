import torch.nn as nn


class MultiScaleTrendMixing(nn.Module):
    def __init__(self, 
                 history_seq_len,
                 future_seq_len,
                 num_channels,
                 ds_layers,
                 ds_window):
        super(MultiScaleTrendMixing, self).__init__()

        self.history_seq_len = history_seq_len
        self.future_seq_len = future_seq_len
        self.num_channels = num_channels
        self.ds_layers = ds_layers        
        self.ds_window = ds_window
        
        # Length Alignment
        self.up_sampling = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        self.history_seq_len//(self.ds_window**(l+1)),
                        self.history_seq_len//(self.ds_window**(l)),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        self.history_seq_len//(self.ds_window**(l)),
                        self.history_seq_len//(self.ds_window**(l)),
                    )
                ) for l in reversed(range(self.ds_layers))
            ]
        )


    def forward(self, ms_trend_list):

        length_list = []
        trend_list = []
        for x in ms_trend_list:
            _, t, _ = x.size()
            length_list.append(t)
            trend_list.append(x.permute(0, 2, 1)) # [B, N, t]

        # Trend mixing
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()

        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]

        out_trend_list = [out_low.permute(0, 2, 1)]
        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()

        out_list = []
        for out_trend, length in zip(out_trend_list, length_list):
            out_list.append(out_trend[:, :length, :]) # list of each element in [B, t, C]

        return out_list