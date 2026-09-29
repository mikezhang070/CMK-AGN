"""ClinicalPredictionModel — single-task model used by every one of the 5 tasks.

Design choice:
    Re-use the existing `ADKMDFANTriBackbone` as the shared encoder / fusion
    block. Only the final head changes between tasks:
      - regression  → 1-d scalar output (clipped at inference time)
      - classification → C-class logits

Each task instantiates its OWN model (own weights, own checkpoint). There is
no joint loss and no loss weighting — every model trains its single task
loss only.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from models.adk_mdfan_tri import ADKMDFANTriBackbone


class HybridRegressionHead(nn.Module):
    """Discrete-distribution expectation + tanh residual for bounded regression.

    The classifier produces logits over `num_bins` evenly-spaced bin centers
    spanning ``[score_min, score_max]`` with stride ``bin_step``. The expected
    value of softmax(logits) gives a coarse continuous prediction; a tanh-bounded
    residual (scaled by ``bin_step``) refines it.

    Forward output is a dict so the trainer can apply Huber on ``pred`` AND
    cross-entropy on ``logits`` against the rounded bin index of the target.
    At inference, only ``pred`` is needed.
    """

    def __init__(
        self,
        in_features: int,
        score_min: float,
        score_max: float,
        bin_step: float,
    ) -> None:
        super().__init__()
        if bin_step <= 0.0:
            raise ValueError(f"bin_step must be > 0, got {bin_step}")
        if score_max <= score_min:
            raise ValueError(f"score_max ({score_max}) must exceed score_min ({score_min})")
        num_bins = int(round((score_max - score_min) / bin_step)) + 1
        if num_bins < 2:
            raise ValueError(f"Need at least 2 bins, got num_bins={num_bins}")

        self.score_min = float(score_min)
        self.score_max = float(score_max)
        self.bin_step = float(bin_step)
        self.num_bins = int(num_bins)

        bin_centers = torch.linspace(self.score_min, self.score_max, self.num_bins)
        self.register_buffer("bin_centers", bin_centers, persistent=False)

        self.classifier = nn.Linear(in_features, self.num_bins)
        self.residual = nn.Linear(in_features, 1)

    def target_to_bin_index(self, target: torch.Tensor) -> torch.Tensor:
        """Round a continuous target to its nearest bin index, clamped in range."""
        idx = torch.round((target - self.score_min) / self.bin_step).long()
        return torch.clamp(idx, 0, self.num_bins - 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.classifier(x)
        probs = torch.softmax(logits, dim=-1)
        expected = (probs * self.bin_centers).sum(dim=-1)
        residual = self.bin_step * torch.tanh(self.residual(x).squeeze(-1))
        return {
            "pred": expected + residual,
            "logits": logits,
            "expected": expected,
            "residual": residual,
        }


class CORNOrdinalHead(nn.Module):
    """CORN ordinal head (Cao et al. 2020).

    Outputs K-1 *conditional* logits z_k = logit P(y>k | y>k-1).
    Decoding: cumulative product of sigmoid(z_k) yields P(y>k); the predicted
    rank is the count of k for which P(y>k) > 0.5, which is automatically
    monotonic in k. A small MLP body adds capacity over the prior plain Linear
    head while staying lightweight.
    """

    def __init__(self, in_features: int, num_classes: int, hidden: int, p: float):
        super().__init__()
        if int(num_classes) < 2:
            raise ValueError(f"CORN head needs num_classes >= 2, got {num_classes}")
        self.num_classes = int(num_classes)
        self.body = nn.Sequential(
            nn.LayerNorm(int(in_features)),
            nn.Linear(int(in_features), int(hidden)),
            nn.GELU(),
            nn.Dropout(float(p)),
            nn.Linear(int(hidden), self.num_classes - 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class ClinicalPredictionModel(nn.Module):
    """Tri-modal model for one of the 5 clinical tasks.

    Args:
        task_type:  "regression" or "classification".
        num_classes: required when task_type == "classification".
        eeg_channels / emg_channels / imu_channels: tri-modal input shapes.
        f / dropout / num_tasks / num_trials: feature width, dropout, embedding sizes
            (kept as args so this can be swapped into the existing trainer with the
            same hyperparameter surface).
        head_kind: classification head variant — "ce" (plain Linear → CE logits,
            legacy default) or "corn" (CORN ordinal head → K-1 conditional
            logits, trained with CORN loss). Ignored for regression.

    Forward inputs (batched bag-of-trials, matching the existing trainer):
        eeg:   [B, S, C_eeg, T]
        emg:   [B, S, C_emg, T]
        imu:   [B, S, C_imu, T]
        task_id:      [B, S]   (long)
        trial_number: [B, S]   (long)

    Forward output:
        regression (legacy)   → 1-d Tensor [B] (scalar prediction, NOT clipped).
        regression (hybrid)   → dict with keys "pred"[B], "logits"[B,K],
                                "expected"[B], "residual"[B]. Use "pred" at
                                inference; train with Huber on "pred" + CE on
                                "logits" against the rounded bin index.
        classification (ce)   → 2-d Tensor [B, num_classes] (raw CE logits).
        classification (corn) → 2-d Tensor [B, num_classes - 1] (CORN
                                conditional logits). Decode in the trainer.
    """

    def __init__(
        self,
        task_type: str,
        num_classes: Optional[int] = None,
        eeg_channels: int = 30,
        emg_channels: int = 4,
        imu_channels: int = 24,
        f: int = 48,
        te: int = 12,
        p: float = 0.15,
        num_tasks: int = 30,
        num_trials: int = 8,
        score_min: float = 0.0,
        score_max: float = 0.0,
        bin_step: float = 0.0,
        head_kind: str = "ce",
        use_graph: bool = True,
        use_attention: bool = True,
        enabled_modalities: tuple = ("eeg", "emg", "imu"),
    ):
        super().__init__()
        if task_type not in ("regression", "classification"):
            raise ValueError(f"task_type must be 'regression' or 'classification', got {task_type!r}")
        if task_type == "classification":
            if num_classes is None or int(num_classes) < 2:
                raise ValueError("classification task requires num_classes >= 2")
        if head_kind not in ("ce", "corn"):
            raise ValueError(f"head_kind must be 'ce' or 'corn', got {head_kind!r}")
        self.task_type = task_type
        self.num_classes = int(num_classes) if num_classes is not None else 0
        self.score_min = float(score_min)
        self.score_max = float(score_max)
        self.bin_step = float(bin_step)
        self.is_hybrid_regression = task_type == "regression" and self.bin_step > 0.0
        self.head_kind = head_kind if task_type == "classification" else "ce"

        gat_heads = int(os.environ.get("ADK_MDFAN_GAT_HEADS", "4"))
        graph_layers = int(os.environ.get("ADK_MDFAN_GRAPH_LAYERS", "2"))

        self.eeg_channels = int(eeg_channels)
        self.emg_channels = int(emg_channels)
        self.imu_channels = int(imu_channels)

        self.trial_encoder = ADKMDFANTriBackbone(
            eeg_channels=self.eeg_channels,
            emg_channels=self.emg_channels,
            imu_channels=self.imu_channels,
            imu_node_dim=6,
            node_feature_dim=int(f),
            gat_heads=gat_heads,
            graph_layers=graph_layers,
            dropout=float(p),
            cross_modal_graph=True,
            peripheral_anatomical_pairing=True,
            use_attention=use_attention,
            use_graph=use_graph,
            enabled_modalities=tuple(enabled_modalities),
        )

        self.task_embedding = nn.Embedding(int(num_tasks) + 1, int(te))
        self.trial_embedding = nn.Embedding(int(num_trials) + 1, max(4, int(te) // 2))
        meta_dim = int(te) + max(4, int(te) // 2)

        self.meta = nn.Sequential(
            nn.LayerNorm(meta_dim),
            nn.Linear(meta_dim, int(f)),
            nn.GELU(),
            nn.Dropout(p),
        )
        self.trial_fusion = nn.Sequential(
            nn.LayerNorm(int(f) * 2),
            nn.Linear(int(f) * 2, int(f)),
            nn.GELU(),
            nn.Dropout(p),
        )
        self.trial_attention = nn.Sequential(
            nn.LayerNorm(int(f)),
            nn.Linear(int(f), max(16, int(f) // 2)),
            nn.Tanh(),
            nn.Dropout(p),
            nn.Linear(max(16, int(f) // 2), 1),
        )
        self.subject_head = nn.Sequential(
            nn.LayerNorm(int(f) * 3),
            nn.Linear(int(f) * 3, int(f)),
            nn.GELU(),
            nn.Dropout(p),
        )

        if self.task_type == "regression":
            if self.is_hybrid_regression:
                self.output_head = HybridRegressionHead(
                    in_features=int(f),
                    score_min=self.score_min,
                    score_max=self.score_max,
                    bin_step=self.bin_step,
                )
            else:
                self.output_head = nn.Linear(int(f), 1)
        else:
            if self.head_kind == "corn":
                self.output_head = CORNOrdinalHead(
                    in_features=int(f),
                    num_classes=self.num_classes,
                    hidden=int(f),
                    p=float(p),
                )
            else:
                self.output_head = nn.Linear(int(f), self.num_classes)

    def _encode_subject(self, eeg, emg, imu, task_id, trial_number):
        b, s = eeg.shape[:2]
        eeg_flat = eeg.reshape(b * s, eeg.shape[2], eeg.shape[3])
        emg_flat = emg.reshape(b * s, emg.shape[2], emg.shape[3])
        imu_flat = imu.reshape(b * s, imu.shape[2], imu.shape[3])

        encoded = self.trial_encoder(eeg_flat, emg_flat, imu_flat, return_features=True)
        if isinstance(encoded, tuple):
            _, trial_feature = encoded
        elif isinstance(encoded, dict):
            trial_feature = encoded.get("features")
        else:
            trial_feature = encoded

        task_flat = torch.clamp(task_id.reshape(-1), 0, self.task_embedding.num_embeddings - 1)
        trial_flat = torch.clamp(trial_number.reshape(-1), 0, self.trial_embedding.num_embeddings - 1)
        meta_feature = self.meta(
            torch.cat([self.task_embedding(task_flat), self.trial_embedding(trial_flat)], dim=1)
        )

        trial_feature = self.trial_fusion(torch.cat([trial_feature, meta_feature], dim=1))
        trial_feature = trial_feature.reshape(b, s, -1)

        weights = torch.softmax(self.trial_attention(trial_feature).squeeze(-1), dim=1)
        weighted = torch.sum(trial_feature * weights.unsqueeze(-1), dim=1)
        return self.subject_head(
            torch.cat([weighted, trial_feature.mean(dim=1), trial_feature.amax(dim=1)], dim=1)
        )

    def forward(self, eeg, emg, imu, task_id, trial_number):
        subject_feature = self._encode_subject(eeg, emg, imu, task_id, trial_number)
        out = self.output_head(subject_feature)
        if self.task_type == "regression":
            if self.is_hybrid_regression:
                # `out` is the dict returned by HybridRegressionHead.
                return out
            return out.squeeze(-1)
        return out


class MultiTaskClinicalPredictionModel(nn.Module):
    """Shared ADK--MDFan encoder with four task-specific clinical heads.

    This is deliberately an experimental alternative to ``ClinicalPredictionModel``:
    it shares only the signal representation.  FMA/BI retain bounded hybrid
    regression heads and the two ordered assessments retain CORN heads.
    """

    def __init__(
        self,
        eeg_channels: int = 30,
        emg_channels: int = 4,
        imu_channels: int = 24,
        f: int = 48,
        te: int = 12,
        p: float = 0.15,
        use_graph: bool = True,
        use_attention: bool = True,
        enabled_modalities: tuple = ("eeg", "emg", "imu"),
    ) -> None:
        super().__init__()
        self.encoder = ClinicalPredictionModel(
            task_type="regression",
            eeg_channels=eeg_channels,
            emg_channels=emg_channels,
            imu_channels=imu_channels,
            f=f,
            te=te,
            p=p,
            bin_step=0.0,
            use_graph=use_graph,
            use_attention=use_attention,
            enabled_modalities=enabled_modalities,
        )
        # The single-task output head is unused in MTL; retain only the shared encoder.
        self.encoder.output_head = nn.Identity()
        self.fma_head = HybridRegressionHead(f, 0.0, 20.0, 1.0)
        self.bi_head = HybridRegressionHead(f, 0.0, 100.0, 7.0)
        self.tone_head = CORNOrdinalHead(f, num_classes=6, hidden=f, p=p)
        self.hand_function_head = CORNOrdinalHead(f, num_classes=6, hidden=f, p=p)
        self.masked_representation_decoder = nn.Sequential(
            nn.LayerNorm(f), nn.Linear(f, f), nn.GELU(), nn.Linear(f, f),
        )

    def encode(self, eeg, emg, imu, task_id, trial_number):
        return self.encoder._encode_subject(eeg, emg, imu, task_id, trial_number)

    def forward(self, eeg, emg, imu, task_id, trial_number):
        feature = self.encode(eeg, emg, imu, task_id, trial_number)
        return {
            "feature": feature,
            "FMA_UE": self.fma_head(feature),
            "BI": self.bi_head(feature),
            "hand_tone": self.tone_head(feature),
            "hand_function": self.hand_function_head(feature),
        }
