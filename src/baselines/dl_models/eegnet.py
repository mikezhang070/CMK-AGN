"""EEGNet-style baseline (EEG only; EMG/IMU intentionally ignored).

Deliberately weaker than the tri-modal main model — uses ONLY the EEG channels
to make the inductive-bias gap (cross-modal fusion vs. EEG-only) explicit.

@article{lawhern2018eegnet,
  author  = {Lawhern, V. J. and Solon, A. J. and Waytowich, N. R. and
             Gordon, S. M. and Hung, C. P. and Lance, B. J.},
  title   = {EEGNet: a compact convolutional neural network for EEG-based
             brain--computer interfaces},
  journal = {Journal of Neural Engineering}, volume = {15}, number = {5},
  year    = {2018}, pages = {056013}
}
"""
from __future__ import annotations

import torch
import torch.nn as nn

from baselines.dl_models._common import DEFAULT_FEATURE_DIM, BaselineDLModule
from baselines.registry import register


class EEGNetBaseline(BaselineDLModule):
    def __init__(self, spec, args, F1: int = 8, D: int = 2, F2: int = 16,
                 kernel_length: int = 64, dropout: float = 0.25):
        super().__init__(spec, args, name="eegnet", feature_dim=DEFAULT_FEATURE_DIM)
        C = int(args.eeg_channels)

        # Block 1 — temporal conv then depthwise spatial conv.
        self.temporal = nn.Conv2d(1, F1, (1, kernel_length),
                                   padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2 — separable conv.
        self.sep_dw = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                                groups=F1 * D, bias=False)
        self.sep_pw = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(F2, self.feature_dim),
            nn.GELU(),
        )

    def encode_trial(self, eeg, emg, imu):  # noqa: ARG002 — emg/imu unused by design
        # eeg: [N, C, T] → [N, 1, C, T]
        x = eeg.unsqueeze(1)
        x = self.bn1(self.temporal(x))
        x = self.elu(self.bn2(self.depthwise(x)))
        x = self.drop1(self.pool1(x))
        x = self.elu(self.bn3(self.sep_pw(self.sep_dw(x))))
        x = self.drop2(self.pool2(x))
        return self.proj(x)


@register("eegnet")
def build(spec, args):
    return EEGNetBaseline(spec, args)
