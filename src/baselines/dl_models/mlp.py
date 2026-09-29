"""MLP baseline on flattened tri-modal input.

@book{goodfellow2016dl,
  author = {Goodfellow, I. and Bengio, Y. and Courville, A.},
  title  = {Deep Learning}, year = {2016}, publisher = {MIT Press}
}
"""
from __future__ import annotations

import torch
import torch.nn as nn

from baselines.dl_models._common import DEFAULT_FEATURE_DIM, BaselineDLModule
from baselines.registry import register


class MLPBaseline(BaselineDLModule):
    def __init__(self, spec, args):
        super().__init__(spec, args, name="mlp", feature_dim=DEFAULT_FEATURE_DIM)
        ce, cm, ci, T = args.eeg_channels, args.emg_channels, args.imu_channels, args.seq_len
        in_dim = (ce + cm + ci) * int(T)
        self.encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.feature_dim),
            nn.GELU(),
        )

    def encode_trial(self, eeg, emg, imu):
        # [N, C, T] each → concat along channel axis → [N, (Ce+Cm+Ci)*T] after flatten.
        x = torch.cat([eeg, emg, imu], dim=1)
        return self.encoder(x)


@register("mlp")
def build(spec, args):
    return MLPBaseline(spec, args)
