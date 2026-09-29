"""Common building blocks shared by all DL baselines.

Design constraints (enforced here so individual baseline files stay tiny):
  * All baselines accept the same forward signature as the main model:
        forward(eeg[B,S,Ce,T], emg[B,S,Cm,T], imu[B,S,Ci,T], task[B,S], trial[B,S])
    (``task`` and ``trial`` are deliberately ignored — capacity reduction.)
  * Per-trial encoder returns a [B*S, F] feature; ``pool_bag`` reduces to
    [B, F] by *mean* (no attention) — another capacity reduction vs. the
    main model's learned trial attention.
  * Output: a Tensor [B] for regression (legacy single-Linear head, NOT the
    main model's hybrid bin head) or logits [B, C] for classification. This
    matches the two branches that :func:`train.evaluate` already handles.

The lightweight trainer at the bottom is intentionally plainer than
``train.train_one_task``: AdamW, fixed LR, no warmup, no cosine, no class
weights, no minority oversampling, no ordinal aux loss. Default capacity
``feature_dim=24`` is ~50% of the main model's 48.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from task_config import LabelEncoder, TaskSpec

DEFAULT_FEATURE_DIM = 24


class SmallHead(nn.Module):
    """Single ``nn.Linear`` head — regression scalar or classification logits."""

    def __init__(self, in_features: int, task_type: str, num_classes: int | None):
        super().__init__()
        self.task_type = task_type
        if task_type == "regression":
            self.head = nn.Linear(in_features, 1)
        elif task_type == "classification":
            assert num_classes is not None and num_classes > 0
            self.head = nn.Linear(in_features, int(num_classes))
        else:
            raise ValueError(f"Unsupported task_type={task_type!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(x)
        if self.task_type == "regression":
            return out.squeeze(-1)
        return out


def pool_bag(trial_feats: torch.Tensor, bag_shape: Tuple[int, int]) -> torch.Tensor:
    """Mean-pool per-trial features back to per-bag (subject) features."""
    B, S = bag_shape
    return trial_feats.view(B, S, -1).mean(dim=1)


class BaselineDLModule(nn.Module):
    """Base class wiring trial encoder + bag pool + head + tiny trainer."""

    family = "dl"

    def __init__(self, spec: TaskSpec, args, name: str, feature_dim: int = DEFAULT_FEATURE_DIM):
        super().__init__()
        self.spec = spec
        self.args = args
        self.name = name
        self.feature_dim = int(feature_dim)
        self.head = SmallHead(
            self.feature_dim,
            spec.task_type,
            spec.num_classes if spec.task_type == "classification" else None,
        )

    # Subclasses implement: returns trial-level features [B*S, F].
    def encode_trial(self, eeg: torch.Tensor, emg: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        eeg: torch.Tensor,
        emg: torch.Tensor,
        imu: torch.Tensor,
        task: torch.Tensor,  # noqa: ARG002 — deliberately ignored
        trial: torch.Tensor,  # noqa: ARG002 — deliberately ignored
    ) -> torch.Tensor:
        B, S = eeg.shape[0], eeg.shape[1]
        eeg_flat = eeg.reshape(B * S, *eeg.shape[2:])
        emg_flat = emg.reshape(B * S, *emg.shape[2:])
        imu_flat = imu.reshape(B * S, *imu.shape[2:])
        trial_feats = self.encode_trial(eeg_flat, emg_flat, imu_flat)
        bag_feats = pool_bag(trial_feats, (B, S))
        return self.head(bag_feats)


def make_loss_fn(spec: TaskSpec) -> nn.Module:
    """Plain loss — no class weighting, no label smoothing — by design."""
    if spec.task_type == "regression":
        return nn.SmoothL1Loss(beta=1.0)
    return nn.CrossEntropyLoss()


def train_dl_baseline(
    model: BaselineDLModule,
    tr_loader: DataLoader,
    ev_loader: DataLoader,
    spec: TaskSpec,
    encoder: LabelEncoder | None,
    device: torch.device,
    epochs: int,
    lr: float,
    grad_clip: float,
    rounded_tol: float,
    score_tolerance: float,
    evaluate_fn,
) -> Tuple[Dict[str, float], List[Dict[str, float]], dict]:
    """Stripped trainer. Returns (best_metrics, history, best_state_dict)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = make_loss_fn(spec).to(device)

    history: List[Dict[str, float]] = []
    best: Dict[str, float] | None = None
    best_state: dict | None = None
    best_metric_val = float("inf")

    for ep in range(1, int(epochs) + 1):
        model.train()
        total_loss, n_loss = 0.0, 0
        for batch in tr_loader:
            out = model(
                batch["eeg"].to(device),
                batch["emg"].to(device),
                batch["imu"].to(device),
                batch["task"].to(device),
                batch["trial"].to(device),
            )
            target = batch["target"].to(device)
            if spec.task_type == "regression":
                loss = loss_fn(out, target.float())
            else:
                loss = loss_fn(out, target.long())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total_loss += float(loss.item()) * int(target.numel())
            n_loss += int(target.numel())

        train_loss = total_loss / max(n_loss, 1)
        metrics, _ = evaluate_fn(
            model, ev_loader, device, spec, encoder,
            rounded_tol, score_tolerance, loss_fn=loss_fn,
        )

        # Selection criterion (kept simple): minimize val loss for both task types.
        cur = float(metrics.get("loss", float("inf")))
        is_best = cur < best_metric_val - 1e-12
        if is_best:
            best_metric_val = cur
            best = metrics
            best_state = deepcopy(model.state_dict())

        history.append({
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": float(metrics.get("loss", float("nan"))),
            **{f"val_{k}": v for k, v in metrics.items()
               if k not in ("confusion_matrix", "loss") and not isinstance(v, (list, dict))},
            "is_best": bool(is_best),
        })
        print(
            f"  ep {ep:>3}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={cur:.4f}"
            + (f"  acc={metrics.get('accuracy', float('nan'))*100:.1f}%"
               if spec.task_type == "classification" else
               f"  MAE={metrics.get('mae', float('nan')):.3f}")
            + ("  *" if is_best else ""),
            flush=True,
        )

    assert best is not None and best_state is not None, "No epoch ran."
    model.load_state_dict(best_state)
    # Re-evaluate once with best weights to get the final metrics+predictions df.
    final_m, final_sf = evaluate_fn(
        model, ev_loader, device, spec, encoder,
        rounded_tol, score_tolerance, loss_fn=loss_fn,
    )
    return final_m, history, best_state
