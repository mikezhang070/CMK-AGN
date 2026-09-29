"""BiLSTM with additive attention pooling (Zhou et al. 2016).

@inproceedings{zhou2016attention,
  author    = {Zhou, P. and Shi, W. and Tian, J. and Qi, Z. and Li, B. and
               Hao, H. and Xu, B.},
  title     = {Attention-Based Bidirectional Long Short-Term Memory Networks
               for Relation Classification},
  booktitle = {ACL}, year = {2016}
}
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.dl_models._common import DEFAULT_FEATURE_DIM, BaselineDLModule
from baselines.registry import register


class BiLSTMAttentionBaseline(BaselineDLModule):
    def __init__(self, spec, args, hidden: int = 16):
        super().__init__(spec, args, name="bilstm_attn", feature_dim=DEFAULT_FEATURE_DIM)
        ce, cm, ci = args.eeg_channels, args.emg_channels, args.imu_channels
        in_ch = ce + cm + ci
        self.proj = nn.Linear(in_ch, hidden)
        self.lstm = nn.LSTM(
            input_size=hidden, hidden_size=hidden,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.attn = nn.Linear(2 * hidden, 1)
        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden, self.feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def encode_trial(self, eeg, emg, imu):
        # Stack channels: [N, Ce+Cm+Ci, T] → [N, T, in_ch]
        x = torch.cat([eeg, emg, imu], dim=1).transpose(1, 2)
        x = self.proj(x)                          # [N, T, hidden]
        h, _ = self.lstm(x)                        # [N, T, 2*hidden]
        a = self.attn(h).squeeze(-1)               # [N, T]
        w = F.softmax(a, dim=1).unsqueeze(-1)      # [N, T, 1]
        ctx = (h * w).sum(dim=1)                   # [N, 2*hidden]
        return self.fuse(ctx)


@register("bilstm_attn")
def build(spec, args):
    return BiLSTMAttentionBaseline(spec, args)
