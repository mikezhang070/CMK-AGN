"""Tri-modal ADK-MDFAN backbone for the BJH dataset (EEG + EMG + IMU).

Re-uses the temporal encoder, masked GAT layer and temporal attention pooling
from the original 2-modal `adk_mdfan` module, so the architectural innovations
remain identical — only the modality count, node count and adjacency change.

Node ordering (must stay consistent with the adjacency builder):
    0 .. eeg_nodes-1                                   -> EEG  (1 channel/node)
    eeg_nodes .. eeg_nodes+emg_nodes-1                 -> EMG  (1 channel/node, one per muscle)
    eeg_nodes+emg_nodes .. eeg_nodes+emg_nodes+imu_nodes-1 -> IMU (imu_node_dim channels/node, one node per muscle)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .adk_mdfan import (
    GraphAttentionLayer,
    MultiScaleNodeTemporalEncoder,
    TemporalAttentionPooling,
)


def build_adk_mdfan_tri_adjacency(
    eeg_nodes: int = 32,
    emg_nodes: int = 4,
    imu_nodes: int = 4,
    cross_modal: bool = True,
    peripheral_anatomical_pairing: bool = True,
) -> torch.Tensor:
    """Initial graph for the tri-modal backbone.

    Within-modality: chain (each node connected to its index neighbour).
    Cross-modality (when `cross_modal=True`):
      - EEG <-> EMG: dense (central system modulates every muscle)
      - EEG <-> IMU: dense (central system modulates every limb segment)
      - EMG <-> IMU: anatomical pairing — same-muscle EMG-IMU node pairs are
        connected when `peripheral_anatomical_pairing=True`. If
        `emg_nodes != imu_nodes` we fall back to dense EMG-IMU connectivity.

    GAT learns the actual edge weights; the adjacency only masks which message
    paths are allowed.
    """
    if min(eeg_nodes, emg_nodes, imu_nodes) <= 0:
        raise ValueError(f"Node counts must be positive: eeg={eeg_nodes}, emg={emg_nodes}, imu={imu_nodes}")
    total = eeg_nodes + emg_nodes + imu_nodes
    A = torch.eye(total, dtype=torch.bool)

    eeg_off, emg_off, imu_off = 0, eeg_nodes, eeg_nodes + emg_nodes

    def _chain(off: int, n: int) -> None:
        for i in range(n - 1):
            A[off + i, off + i + 1] = True
            A[off + i + 1, off + i] = True

    _chain(eeg_off, eeg_nodes)
    _chain(emg_off, emg_nodes)
    _chain(imu_off, imu_nodes)

    if cross_modal:
        # EEG <-> EMG (dense)
        A[eeg_off:eeg_off + eeg_nodes, emg_off:emg_off + emg_nodes] = True
        A[emg_off:emg_off + emg_nodes, eeg_off:eeg_off + eeg_nodes] = True
        # EEG <-> IMU (dense)
        A[eeg_off:eeg_off + eeg_nodes, imu_off:imu_off + imu_nodes] = True
        A[imu_off:imu_off + imu_nodes, eeg_off:eeg_off + eeg_nodes] = True
        # EMG <-> IMU
        if peripheral_anatomical_pairing and emg_nodes == imu_nodes:
            for m in range(emg_nodes):
                A[emg_off + m, imu_off + m] = True
                A[imu_off + m, emg_off + m] = True
        else:
            A[emg_off:emg_off + emg_nodes, imu_off:imu_off + imu_nodes] = True
            A[imu_off:imu_off + imu_nodes, emg_off:emg_off + emg_nodes] = True

    return A


class ADKMDFANTriBackbone(nn.Module):
    """Tri-modal ADK-MDFAN backbone for FMA regression.

    Forward inputs (all batched, channels-major):
        eeg: [B, eeg_channels, T]            — typically 32 channels
        emg: [B, emg_channels, T]            — typically 4 muscle envelopes
        imu: [B, imu_channels, T]            — typically 24 = imu_nodes * imu_node_dim

    Forward outputs:
        Either a tensor `[B, 1]` (FMA), or a richer dict if `return_aux=True`.
    """

    def __init__(
        self,
        eeg_channels: int = 32,
        emg_channels: int = 4,
        imu_channels: int = 24,
        imu_node_dim: int = 6,           # 3 ACC + 3 GYRO per muscle
        node_feature_dim: int = 64,
        gat_heads: int = 4,
        graph_layers: int = 2,
        dropout: float = 0.2,
        cross_modal_graph: bool = True,
        peripheral_anatomical_pairing: bool = True,
        use_attention: bool = True,
        use_graph: bool = True,
        enabled_modalities: tuple = ("eeg", "emg", "imu"),
    ):
        super().__init__()
        if imu_channels % imu_node_dim != 0:
            raise ValueError(f"imu_channels={imu_channels} must be divisible by imu_node_dim={imu_node_dim}")
        if use_graph and graph_layers < 1:
            raise ValueError(f"graph_layers must be >= 1, got {graph_layers}")
        enabled = set(enabled_modalities)
        if not enabled or not enabled.issubset({"eeg", "emg", "imu"}):
            raise ValueError(f"enabled_modalities must be a non-empty subset of {{eeg,emg,imu}}, got {enabled_modalities}")
        self.enabled_modalities = enabled

        self.eeg_channels = eeg_channels
        self.emg_channels = emg_channels
        self.imu_channels = imu_channels
        self.imu_node_dim = imu_node_dim
        self.eeg_nodes = eeg_channels                 # 1 ch / node
        self.emg_nodes = emg_channels                 # 1 ch / node (one per muscle)
        self.imu_nodes = imu_channels // imu_node_dim # imu_node_dim ch / node
        self.total_nodes = self.eeg_nodes + self.emg_nodes + self.imu_nodes

        self.use_attention = use_attention
        self.use_graph = use_graph
        self.supports_aux_output = True

        branch_channels = max(8, node_feature_dim // 4)
        self.eeg_encoder = MultiScaleNodeTemporalEncoder(
            node_channels=1, feature_dim=node_feature_dim, branch_channels=branch_channels,
        )
        self.emg_encoder = MultiScaleNodeTemporalEncoder(
            node_channels=1, feature_dim=node_feature_dim, branch_channels=branch_channels,
        )
        self.imu_encoder = MultiScaleNodeTemporalEncoder(
            node_channels=imu_node_dim, feature_dim=node_feature_dim, branch_channels=branch_channels,
        )

        # One learnable embedding per modality (3 instead of the original 2).
        self.modality_embedding = nn.Parameter(torch.zeros(3, node_feature_dim))

        self.graph_layers = nn.ModuleList()
        if self.use_graph:
            self.graph_layers.extend(
                [
                    GraphAttentionLayer(
                        in_features=node_feature_dim,
                        out_features=node_feature_dim,
                        heads=gat_heads,
                        dropout=dropout,
                    )
                    for _ in range(graph_layers)
                ]
            )

        self.temporal_attention = (
            TemporalAttentionPooling(node_feature_dim, hidden_dim=node_feature_dim, dropout=dropout)
            if self.use_attention
            else None
        )

        self.feature_pyramid = nn.Sequential(
            nn.LayerNorm(node_feature_dim * 3),
            nn.Linear(node_feature_dim * 3, node_feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(node_feature_dim),
            nn.Linear(node_feature_dim, node_feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_feature_dim, 1),
        )

        adjacency = build_adk_mdfan_tri_adjacency(
            self.eeg_nodes,
            self.emg_nodes,
            self.imu_nodes,
            cross_modal=cross_modal_graph,
            peripheral_anatomical_pairing=peripheral_anatomical_pairing,
        )
        self.register_buffer("adjacency", adjacency, persistent=False)

    # --------------------------------------------------------------------- #
    def _check(self, eeg: torch.Tensor, emg: torch.Tensor, imu: torch.Tensor) -> None:
        for name, x, expected in (
            ("eeg", eeg, self.eeg_channels),
            ("emg", emg, self.emg_channels),
            ("imu", imu, self.imu_channels),
        ):
            if x.ndim != 3:
                raise ValueError(f"{name} must be [B, C, T], got {tuple(x.shape)}")
            if x.shape[1] != expected:
                raise ValueError(f"{name} expected {expected} channels, got {x.shape[1]}")
        if not (eeg.shape[0] == emg.shape[0] == imu.shape[0]):
            raise ValueError(f"Batch mismatch: eeg={eeg.shape[0]}, emg={emg.shape[0]}, imu={imu.shape[0]}")
        if not (eeg.shape[-1] == emg.shape[-1] == imu.shape[-1]):
            raise ValueError(
                f"Time mismatch: eeg={eeg.shape[-1]}, emg={emg.shape[-1]}, imu={imu.shape[-1]}. "
                "Run align_by_strategy_tri so all modalities share `output_length`."
            )

    def forward(
        self,
        eeg: torch.Tensor,
        emg: torch.Tensor,
        imu: torch.Tensor,
        return_attention: bool = False,
        return_features: bool = False,
        return_aux: bool = False,
    ):
        self._check(eeg, emg, imu)
        if "eeg" not in self.enabled_modalities:
            eeg = torch.zeros_like(eeg)
        if "emg" not in self.enabled_modalities:
            emg = torch.zeros_like(emg)
        if "imu" not in self.enabled_modalities:
            imu = torch.zeros_like(imu)
        batch, _, time_steps = eeg.shape

        # Reshape to [B, nodes, node_channels, T]
        eeg_in = eeg.unsqueeze(2)                                              # [B, 32, 1, T]
        emg_in = emg.unsqueeze(2)                                              # [B,  4, 1, T]
        imu_in = imu.reshape(batch, self.imu_nodes, self.imu_node_dim, time_steps)  # [B, 4, 6, T]

        eeg_feat = self.eeg_encoder(eeg_in) + self.modality_embedding[0].view(1, 1, -1, 1)
        emg_feat = self.emg_encoder(emg_in) + self.modality_embedding[1].view(1, 1, -1, 1)
        imu_feat = self.imu_encoder(imu_in) + self.modality_embedding[2].view(1, 1, -1, 1)

        node_features = torch.cat([eeg_feat, emg_feat, imu_feat], dim=1)       # [B, total_nodes, F, T]
        graph_features = node_features.permute(0, 3, 1, 2)                     # [B, T, nodes, F]
        if self.use_graph:
            for layer in self.graph_layers:
                graph_features = layer(graph_features, self.adjacency)

        temporal_features = graph_features.mean(dim=2)                         # [B, T, F]
        if self.temporal_attention is None:
            pooled = temporal_features.mean(dim=1)
            attention = torch.full(
                (temporal_features.shape[0], temporal_features.shape[1]),
                1.0 / float(temporal_features.shape[1]),
                device=temporal_features.device,
                dtype=temporal_features.dtype,
            )
        else:
            pooled, attention = self.temporal_attention(temporal_features)

        mean_feature = temporal_features.mean(dim=1)
        max_feature = temporal_features.amax(dim=1)
        pyramid_feature = self.feature_pyramid(torch.cat([pooled, mean_feature, max_feature], dim=1))
        prediction = self.regressor(pyramid_feature)

        # Per-modality global summaries (useful for ablation / interpretability).
        eeg_end = self.eeg_nodes
        emg_end = self.eeg_nodes + self.emg_nodes
        eeg_global = graph_features[:, :, :eeg_end, :].mean(dim=(1, 2))
        emg_global = graph_features[:, :, eeg_end:emg_end, :].mean(dim=(1, 2))
        imu_global = graph_features[:, :, emg_end:, :].mean(dim=(1, 2))

        if return_aux:
            return {
                "fma": prediction,
                "features": pyramid_feature,
                "temporal_attention": attention,
                "eeg_global": eeg_global,
                "emg_global": emg_global,
                "imu_global": imu_global,
            }
        if return_attention and return_features:
            return prediction, attention, pyramid_feature
        if return_attention:
            return prediction, attention
        if return_features:
            return prediction, pyramid_feature
        return prediction
