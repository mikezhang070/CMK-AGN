"""Per-modality 1D-CNN with late fusion.

@article{lecun1998gradient,
  author  = {LeCun, Y. and Bottou, L. and Bengio, Y. and Haffner, P.},
  title   = {Gradient-based learning applied to document recognition},
  journal = {Proc. IEEE}, volume = {86}, number = {11}, year = {1998},
  pages   = {2278--2324}
}
"""
from __future__ import annotations

import torch
import torch.nn as nn

from baselines.dl_models._common import DEFAULT_FEATURE_DIM, BaselineDLModule
from baselines.registry import register


def _conv_stack(in_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, 16, kernel_size=7, padding=3),
        nn.BatchNorm1d(16),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=2),
        nn.Conv1d(16, 24, kernel_size=5, padding=2),
        nn.BatchNorm1d(24),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool1d(1),
    )


class CNN1DBaseline(BaselineDLModule):
    def __init__(self, spec, args):
        super().__init__(spec, args, name="cnn1d", feature_dim=DEFAULT_FEATURE_DIM)
        self.eeg_enc = _conv_stack(args.eeg_channels)
        self.emg_enc = _conv_stack(args.emg_channels)
        self.imu_enc = _conv_stack(args.imu_channels)
        self.fuse = nn.Sequential(
            nn.Linear(24 * 3, self.feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def encode_trial(self, eeg, emg, imu):
        e = self.eeg_enc(eeg).squeeze(-1)
        m = self.emg_enc(emg).squeeze(-1)
        i = self.imu_enc(imu).squeeze(-1)
        return self.fuse(torch.cat([e, m, i], dim=1))


@register("cnn1d")
def build(spec, args):
    return CNN1DBaseline(spec, args)
