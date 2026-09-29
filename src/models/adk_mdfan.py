from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_adk_mdfan_adjacency(
    emg_nodes: int = 12,
    kin_nodes: int = 21,
    cross_modal: bool = True,
) -> torch.Tensor:
    """Return a first-pass ADK-MDFAN graph adjacency shaped [nodes, nodes].

    Node order is EMG nodes first, then KIN marker nodes. The initial graph uses
    self loops, within-modality local chains, and optional dense cross-modal
    links. GAT learns the actual edge weights during training.
    """
    if emg_nodes <= 0 or kin_nodes <= 0:
        raise ValueError(f"emg_nodes and kin_nodes must be positive, got {emg_nodes}, {kin_nodes}")
    total_nodes = emg_nodes + kin_nodes
    adjacency = torch.eye(total_nodes, dtype=torch.bool)

    for node in range(emg_nodes - 1):
        adjacency[node, node + 1] = True
        adjacency[node + 1, node] = True

    kin_offset = emg_nodes
    for node in range(kin_nodes - 1):
        left = kin_offset + node
        right = kin_offset + node + 1
        adjacency[left, right] = True
        adjacency[right, left] = True

    if cross_modal:
        adjacency[:emg_nodes, kin_offset:] = True
        adjacency[kin_offset:, :emg_nodes] = True

    return adjacency


class MultiScaleNodeTemporalEncoder(nn.Module):
    """Multi-scale 1D temporal encoder for per-node signals.

    Input shape: [batch, nodes, node_channels, time].
    Output shape: [batch, nodes, feature_dim, time].
    """

    def __init__(
        self,
        node_channels: int,
        feature_dim: int = 64,
        branch_channels: int = 16,
        kernel_sizes: tuple[int, ...] = (3, 5, 9),
    ):
        super().__init__()
        if node_channels <= 0:
            raise ValueError(f"node_channels must be positive, got {node_channels}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if not kernel_sizes:
            raise ValueError("kernel_sizes must not be empty")

        self.branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            if kernel_size % 2 == 0:
                raise ValueError(f"kernel_size must be odd to keep time length, got {kernel_size}")
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(node_channels, branch_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(branch_channels),
                    nn.GELU(),
                )
            )
        self.projection = nn.Sequential(
            nn.Conv1d(branch_channels * len(kernel_sizes), feature_dim, kernel_size=1),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [batch, nodes, node_channels, time], got {tuple(x.shape)}")
        batch, nodes, channels, time_steps = x.shape
        if time_steps < 2:
            raise ValueError(f"Expected at least 2 time steps, got {time_steps}")
        flat = x.reshape(batch * nodes, channels, time_steps)
        encoded = torch.cat([branch(flat) for branch in self.branches], dim=1)
        encoded = self.projection(encoded)
        return encoded.reshape(batch, nodes, encoded.shape[1], time_steps)


class GraphAttentionLayer(nn.Module):
    """Masked multi-head graph attention over modality nodes.

    Input shape: [batch, time, nodes, in_features].
    Output shape: [batch, time, nodes, out_features].
    """

    def __init__(self, in_features: int, out_features: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        if out_features % heads != 0:
            raise ValueError(f"out_features={out_features} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = out_features // heads
        self.proj = nn.Linear(in_features, out_features, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        if isinstance(self.residual, nn.Linear):
            nn.init.xavier_uniform_(self.residual.weight)
            nn.init.zeros_(self.residual.bias)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [batch, time, nodes, features], got {tuple(x.shape)}")
        batch, time_steps, nodes, _ = x.shape
        if adjacency.shape != (nodes, nodes):
            raise ValueError(f"Adjacency shape {tuple(adjacency.shape)} does not match node count {nodes}")

        h = self.proj(x).reshape(batch * time_steps, nodes, self.heads, self.head_dim)
        h = h.permute(0, 2, 1, 3)

        src_score = (h * self.attn_src.view(1, self.heads, 1, self.head_dim)).sum(dim=-1)
        dst_score = (h * self.attn_dst.view(1, self.heads, 1, self.head_dim)).sum(dim=-1)
        logits = F.leaky_relu(src_score.unsqueeze(-1) + dst_score.unsqueeze(-2), negative_slope=0.2)
        mask = adjacency.to(device=x.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

        attention = torch.softmax(logits, dim=-1)
        attention = self.dropout(attention)
        out = torch.matmul(attention, h)
        out = out.permute(0, 2, 1, 3).reshape(batch, time_steps, nodes, self.heads * self.head_dim)
        out = self.dropout(out)
        return self.norm(out + self.residual(x))


class TemporalAttentionPooling(nn.Module):
    """Attention pooling over the aligned time axis.

    Input shape: [batch, time, features].
    Output shape: [batch, features].
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, features], got {tuple(x.shape)}")
        scores = self.scorer(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class ADKMDFANBackbone(nn.Module):
    """First trainable ADK-MDFAN EMG/KIN fusion model for FMA regression.

    Forward input:
    - emg: [batch, 12, time], ADK fixed-length EMG sequence.
    - kin: [batch, 63, time], ADK fixed-length KIN sequence.

    Forward output:
    - fma: [batch, 1], single continuous FMA prediction.
    """

    def __init__(
        self,
        emg_channels: int = 12,
        kin_channels: int = 63,
        kin_node_dim: int = 3,
        node_feature_dim: int = 64,
        gat_heads: int = 4,
        graph_layers: int = 2,
        dropout: float = 0.2,
        cross_modal_graph: bool = True,
        use_attention: bool = True,
        use_graph: bool = True,
    ):
        super().__init__()
        if kin_channels % kin_node_dim != 0:
            raise ValueError(f"kin_channels={kin_channels} must be divisible by kin_node_dim={kin_node_dim}")
        if use_graph and graph_layers < 1:
            raise ValueError(f"graph_layers must be >= 1, got {graph_layers}")

        self.emg_channels = emg_channels
        self.kin_channels = kin_channels
        self.kin_node_dim = kin_node_dim
        self.kin_nodes = kin_channels // kin_node_dim
        self.total_nodes = emg_channels + self.kin_nodes
        self.use_attention = use_attention
        self.use_graph = use_graph
        self.supports_aux_output = True

        self.emg_encoder = MultiScaleNodeTemporalEncoder(
            node_channels=1,
            feature_dim=node_feature_dim,
            branch_channels=max(8, node_feature_dim // 4),
        )
        self.kin_encoder = MultiScaleNodeTemporalEncoder(
            node_channels=kin_node_dim,
            feature_dim=node_feature_dim,
            branch_channels=max(8, node_feature_dim // 4),
        )
        self.modality_embedding = nn.Parameter(torch.zeros(2, node_feature_dim))
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
        adjacency = build_adk_mdfan_adjacency(emg_channels, self.kin_nodes, cross_modal=cross_modal_graph)
        self.register_buffer("adjacency", adjacency, persistent=False)

    def forward(
        self,
        emg: torch.Tensor,
        kin: torch.Tensor,
        return_attention: bool = False,
        return_features: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if emg.ndim != 3 or kin.ndim != 3:
            raise ValueError(
                "Expected EMG and KIN tensors shaped [batch, channels, time], "
                f"got emg={tuple(emg.shape)}, kin={tuple(kin.shape)}"
            )
        if emg.shape[1] != self.emg_channels:
            raise ValueError(f"Expected {self.emg_channels} EMG channels, got {emg.shape[1]}")
        if kin.shape[1] != self.kin_channels:
            raise ValueError(f"Expected {self.kin_channels} KIN channels, got {kin.shape[1]}")
        if emg.shape[0] != kin.shape[0] or emg.shape[-1] != kin.shape[-1]:
            raise ValueError(f"Batch/time mismatch: emg={tuple(emg.shape)}, kin={tuple(kin.shape)}")

        batch, _, time_steps = emg.shape
        emg_nodes = emg.unsqueeze(2)
        kin_nodes = kin.reshape(batch, self.kin_nodes, self.kin_node_dim, time_steps)

        emg_features = self.emg_encoder(emg_nodes)
        kin_features = self.kin_encoder(kin_nodes)
        emg_features = emg_features + self.modality_embedding[0].view(1, 1, -1, 1)
        kin_features = kin_features + self.modality_embedding[1].view(1, 1, -1, 1)

        node_features = torch.cat([emg_features, kin_features], dim=1)
        graph_features = node_features.permute(0, 3, 1, 2)
        if self.use_graph:
            for layer in self.graph_layers:
                graph_features = layer(graph_features, self.adjacency)

        temporal_features = graph_features.mean(dim=2)
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
        emg_global = graph_features[:, :, : self.emg_channels, :].mean(dim=(1, 2))
        kin_global = graph_features[:, :, self.emg_channels :, :].mean(dim=(1, 2))
        if return_aux:
            return {
                "fma": prediction,
                "features": pyramid_feature,
                "temporal_attention": attention,
                "emg_global": emg_global,
                "kin_global": kin_global,
            }
        if return_attention and return_features:
            return prediction, attention, pyramid_feature
        if return_attention:
            return prediction, attention
        if return_features:
            return prediction, pyramid_feature
        return prediction
