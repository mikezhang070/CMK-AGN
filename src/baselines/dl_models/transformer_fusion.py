"""Concat-encoder Transformer fusion baseline.

Per-modality temporal mean-pool → 3 tokens (one per modality) → single
TransformerEncoderLayer → mean-pool tokens. Deliberately tiny.

@inproceedings{vaswani2017attention,
  author    = {Vaswani, A. and Shazeer, N. and Parmar, N. and Uszkoreit, J. and
               Jones, L. and Gomez, A. N. and Kaiser, {\\L}. and Polosukhin, I.},
  title     = {Attention Is All You Need},
  booktitle = {NeurIPS}, year = {2017}
}
"""
from __future__ import annotations

import torch
import torch.nn as nn

from baselines.dl_models._common import DEFAULT_FEATURE_DIM, BaselineDLModule
from baselines.registry import register


class TransformerFusionBaseline(BaselineDLModule):
    def __init__(self, spec, args, d_model: int = 24, nhead: int = 2):
        super().__init__(spec, args, name="transformer_fusion", feature_dim=DEFAULT_FEATURE_DIM)
        self.eeg_proj = nn.Linear(args.eeg_channels, d_model)
        self.emg_proj = nn.Linear(args.emg_channels, d_model)
        self.imu_proj = nn.Linear(args.imu_channels, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=48, dropout=0.1,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.fuse = nn.Linear(d_model, self.feature_dim) if d_model != self.feature_dim else nn.Identity()

    def _mod_token(self, x: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        # x: [N, C, T] → mean over T → [N, C] → proj → [N, d_model]
        m = x.mean(dim=2)
        return proj(m)

    def encode_trial(self, eeg, emg, imu):
        toks = torch.stack([
            self._mod_token(eeg, self.eeg_proj),
            self._mod_token(emg, self.emg_proj),
            self._mod_token(imu, self.imu_proj),
        ], dim=1)                            # [N, 3, d_model]
        h = self.encoder(toks)               # [N, 3, d_model]
        ctx = h.mean(dim=1)                  # [N, d_model]
        return self.fuse(ctx)


@register("transformer_fusion")
def build(spec, args):
    return TransformerFusionBaseline(spec, args)
