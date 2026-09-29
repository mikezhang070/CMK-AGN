"""Single-task trainer for one of the 4 clinical prediction tasks.

Each task is trained INDEPENDENTLY:
    python train.py --task FMA_UE
    python train.py --task BI
    python train.py --task hand_tone
    python train.py --task hand_function

There is no joint loss and no loss-weight tuning. Every task gets its own
checkpoint at checkpoints/<task>_model.pth.

The data pipeline (manifest reading, EEG/EMG/IMU loading, tri-modal alignment,
bag construction) is reused from the existing tri-modal trainer modules; we
swap the label column and the model head only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Local imports — keep `src/` on sys.path when running from project root.
import sys
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alignment.wby_dtw import WBYDTWConfig  # noqa: E402
from alignment.tri_strategies import align_by_strategy_tri  # noqa: E402
from bjh_io.bjh_loader import EEG_FS_DEFAULT, load_bjh_trial  # noqa: E402
from clinical_model import ClinicalPredictionModel  # noqa: E402
from task_config import (  # noqa: E402
    LabelEncoder,
    TaskSpec,
    get_encoder,
    get_task,
)


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_project_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


_CACHE: Dict[Tuple, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _fstamp(p: Path) -> Tuple[str, int, int]:
    s = p.stat()
    return (str(p.resolve()), int(s.st_mtime_ns), int(s.st_size))


def _cfgkey(c: WBYDTWConfig) -> Tuple:
    return tuple(sorted((str(k), repr(v)) for k, v in c.__dict__.items()))


def _disk_key(key: tuple) -> str:
    return hashlib.sha256(repr(key).encode()).hexdigest()[:32]


def load_aligned(
    root: Path,
    eeg_path: Path,
    emg_path: Path,
    seq_len: int,
    cfg: WBYDTWConfig,
    mode: str,
    eeg_fs: float,
    preprocess: bool,
    cache_dir: Optional[Path] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (
        _fstamp(eeg_path),
        _fstamp(emg_path),
        int(seq_len),
        str(mode),
        bool(preprocess),
        float(eeg_fs),
        _cfgkey(cfg),
    )
    if key not in _CACHE:
        disk_path = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            disk_path = cache_dir / f"{_disk_key(key)}.npz"
            if disk_path.exists():
                d = np.load(disk_path)
                _CACHE[key] = (d["eeg"], d["emg"], d["imu"])

        if key not in _CACHE:
            sig = load_bjh_trial(eeg_path, emg_path, eeg_fs=eeg_fs, preprocess=preprocess)
            a = align_by_strategy_tri(sig.eeg, sig.emg, sig.imu, mode, cfg)
            _CACHE[key] = (
                np.ascontiguousarray(a.eeg_aligned, dtype=np.float32),
                np.ascontiguousarray(a.emg_aligned, dtype=np.float32),
                np.ascontiguousarray(a.imu_aligned, dtype=np.float32),
            )
            if disk_path is not None:
                np.savez_compressed(
                    disk_path,
                    eeg=_CACHE[key][0],
                    emg=_CACHE[key][1],
                    imu=_CACHE[key][2],
                )
    eeg_arr, emg_arr, imu_arr = _CACHE[key]
    return (
        torch.from_numpy(eeg_arr),
        torch.from_numpy(emg_arr),
        torch.from_numpy(imu_arr),
    )


# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #
class TaskStore:
    """Per-subject store of preprocessed trials for ONE task."""

    def __init__(
        self,
        df: pd.DataFrame,
        root: Path,
        spec: TaskSpec,
        encoder: Optional[LabelEncoder],
        seq_len: int,
        cfg: WBYDTWConfig,
        mode: str,
        eeg_fs: float,
        preprocess: bool,
        cache_dir: Optional[Path] = None,
    ):
        self.subjects: Dict[str, Dict[str, object]] = {}
        for _, r in df.iterrows():
            sid = str(r["subject_id"])
            eeg_path = _as_project_path(root, str(r["eeg_path"]))
            emg_path = _as_project_path(root, str(r["emg_path"]))
            eeg, emg, imu = load_aligned(
                root, eeg_path, emg_path, seq_len, cfg, mode, eeg_fs, preprocess,
                cache_dir,
            )
            label_value = r[spec.manifest_col]
            if pd.isna(label_value):
                raise ValueError(
                    f"Subject {sid} row missing label for task {spec.name} "
                    f"(column '{spec.manifest_col}')."
                )
            if spec.task_type == "regression":
                target_value = float(label_value)
            else:
                assert encoder is not None
                target_value = float(encoder.encode(label_value))
            entry = self.subjects.setdefault(sid, {"target": target_value, "trials": []})
            # Sanity: every trial of the same subject should carry the same label.
            if abs(float(entry["target"]) - target_value) > 1e-9:
                raise ValueError(
                    f"Inconsistent {spec.name} label for subject {sid}: "
                    f"{entry['target']} vs {target_value}"
                )
            entry["trials"].append(
                {
                    "eeg": eeg,
                    "emg": emg,
                    "imu": imu,
                    "task": max(int(r["task_id"]) - 1, 0),
                    "trial": max(int(r["trial_number"]) - 1, 0),
                    "trial_id": str(r["trial_id"]),
                    "key": f"{sid}:{r['trial_id']}",
                }
            )


class BagDS(Dataset):
    """Sample bags of trials per subject, single-task aware."""

    def __init__(
        self,
        store: TaskStore,
        bag_size: int,
        bags_per_subject: int,
        seed: int,
        deterministic: bool,
    ):
        self.store = store
        self.sids = list(store.subjects.keys())
        self.bag = int(bag_size)
        self.bps = int(bags_per_subject)
        self.seed = int(seed)
        self.det = bool(deterministic)

    def __len__(self) -> int:
        return len(self.sids) * self.bps

    def subject_target_per_bag(self) -> np.ndarray:
        """Encoded target (int) for each bag index — enables class-aware sampling."""
        targets = np.empty(len(self), dtype=np.int64)
        for i in range(len(self)):
            so = i // self.bps
            targets[i] = int(self.store.subjects[self.sids[so]]["target"])
        return targets

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        so, bo = int(idx) // self.bps, int(idx) % self.bps
        sid = self.sids[so]
        entry = self.store.subjects[sid]
        trials = entry["trials"]
        if self.det and self.bag == 1:
            ids = np.array([bo % len(trials)], dtype=np.int64)
        else:
            rng = (
                np.random.default_rng(self.seed + 1009 * so + 9176 * bo)
                if self.det
                else np.random.default_rng(np.random.randint(0, 2**31 - 1))
            )
            ids = rng.choice(len(trials), size=self.bag, replace=len(trials) < self.bag)
        picked = [trials[int(i)] for i in ids]
        return {
            "eeg": torch.stack([x["eeg"] for x in picked]),
            "emg": torch.stack([x["emg"] for x in picked]),
            "imu": torch.stack([x["imu"] for x in picked]),
            "task": torch.tensor([x["task"] for x in picked], dtype=torch.long),
            "trial": torch.tensor([x["trial"] for x in picked], dtype=torch.long),
            "target": torch.tensor(float(entry["target"]), dtype=torch.float32),
            "sid": sid,
        }


# --------------------------------------------------------------------------- #
# Train / val split                                                            #
# --------------------------------------------------------------------------- #
def _subject_sort_key(sid: str) -> Tuple[int, object]:
    """Sort numeric subject ids numerically, and non-numeric ids lexically."""
    s = str(sid)
    return (0, int(s)) if s.isdigit() else (1, s)


def _format_subjects(subjects: List[str], head: int = 5, tail: int = 3) -> List[str]:
    """Compact subject list for console printing."""
    subjects = [str(s) for s in subjects]
    if len(subjects) <= head + tail + 2:
        return subjects
    return subjects[:head] + ["..."] + subjects[-tail:]


def _resolve_path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _load_fold_file(root: Path, split_json: Path) -> Tuple[Path, Dict[str, Any]]:
    split_path = _resolve_path(root, split_json)
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}. Expected a JSON file like "
            "`splits/split_patient_3fold.json`."
        )
    data = json.loads(split_path.read_text(encoding="utf-8"))
    if data.get("split_unit") != "subject_id":
        raise ValueError(
            f"Unsupported split_unit={data.get('split_unit')!r}; this trainer expects subject_id splits."
        )
    folds = data.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"Split file {split_path} does not contain a non-empty `folds` list.")
    for fold in folds:
        if "fold" not in fold or "train_subjects" not in fold:
            raise ValueError("Each fold must contain `fold` and `train_subjects`.")
        if "val_test_subjects" not in fold and "val_subjects" not in fold:
            raise ValueError("Each fold must contain `val_test_subjects` or `val_subjects`.")
    return split_path, data


def _select_folds(split_data: Dict[str, Any], fold_arg: int) -> List[Dict[str, Any]]:
    folds = sorted(list(split_data["folds"]), key=lambda f: int(f["fold"]))
    if fold_arg in (0, -1):
        return folds
    selected = [f for f in folds if int(f["fold"]) == int(fold_arg)]
    if not selected:
        available = [int(f["fold"]) for f in folds]
        raise ValueError(f"Requested --fold {fold_arg}, but available folds are {available}.")
    return selected


def _fold_subjects(fold_info: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    train = [str(s) for s in fold_info["train_subjects"]]
    val = [str(s) for s in fold_info.get("val_subjects", fold_info.get("val_test_subjects", []))]
    return train, val


def _validate_subject_split(
    df: pd.DataFrame,
    train_subjects: List[str],
    val_subjects: List[str],
    fold: Optional[int],
) -> None:
    available = set(df["subject_id"].astype(str).unique().tolist())
    train_set = set(str(s) for s in train_subjects)
    val_set = set(str(s) for s in val_subjects)

    overlap = sorted(train_set & val_set, key=_subject_sort_key)
    if overlap:
        raise ValueError(f"Fold {fold} has overlapping train/val subjects: {overlap}")

    missing_train = sorted(train_set - available, key=_subject_sort_key)
    missing_val = sorted(val_set - available, key=_subject_sort_key)
    if missing_train or missing_val:
        raise ValueError(
            f"Fold {fold} contains subjects not present in the manifest. "
            f"missing_train={missing_train}, missing_val={missing_val}"
        )

    if not train_subjects:
        raise ValueError(f"Fold {fold} has no training subjects.")
    if not val_subjects:
        raise ValueError(f"Fold {fold} has no validation subjects.")

    unused = sorted(available - train_set - val_set, key=_subject_sort_key)
    if unused:
        print(f"  Warning: manifest subjects not used by fold {fold}: {unused}")


def _with_fold_suffix(path: Path, fold: Optional[int]) -> Path:
    if fold is None:
        return path
    return path.with_name(f"{path.stem}_fold{int(fold)}{path.suffix}")


def _get_out_dir(args: argparse.Namespace, spec: TaskSpec) -> Path:
    """Canonical output directory for this task.

    Priority:
      1. ``--out-dir`` if explicitly provided
      2. ``<root>/results/<task>/baseline/`` otherwise

    All artefacts (checkpoint, logs, summary) land here unless
    ``--checkpoint`` overrides the .pth location specifically.
    """
    if getattr(args, "out_dir", None) is not None:
        return args.out_dir.resolve()
    tag = getattr(args, "ablation_tag", None)
    if tag:
        kind = "module" if getattr(args, "modalities", "eeg+emg+imu") == "eeg+emg+imu" else "modality"
        return (args.root / "results" / "ablation" / kind / tag / spec.name).resolve()
    return (args.root / "results" / spec.name / "baseline").resolve()


def _checkpoint_path_for_fold(args: argparse.Namespace, spec: TaskSpec) -> Path:
    """Return the .pth save path for the current fold.

    ``--checkpoint`` (supports ``{fold}`` placeholder) takes priority over
    ``--out-dir``; useful when you need an exact file location.
    Otherwise checkpoints are saved as ``<out-dir>/<task>_fold<n>.pth``.
    """
    fold = getattr(args, "fold", None)
    if args.checkpoint:
        tmpl = str(args.checkpoint)
        if fold is not None and "{fold}" in tmpl:
            return Path(tmpl.format(fold=int(fold)))
        return _with_fold_suffix(Path(args.checkpoint), fold)
    out_dir = _get_out_dir(args, spec)
    stem = f"{spec.name}_fold{int(fold)}" if fold is not None else spec.name
    return out_dir / f"{stem}.pth"


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def _fmt_tol(v: float) -> str:
    """Format a tolerance value for use in a metric key (e.g. 1.5 → '1p5')."""
    return f"{v:g}".replace(".", "p")


def rounded_acc_key(rounded_tol: float) -> str:
    return f"rounded_acc_{_fmt_tol(rounded_tol)}"


def tolerance_acc_key(score_tolerance: float) -> str:
    return f"tolerance_acc_{_fmt_tol(score_tolerance)}"


def _regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rounded_tol: float,
    score_tolerance: float,
    score_range: float = 1.0,
) -> Dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    mae = float(abs_err.mean()) if abs_err.size else 0.0
    rmse = float(np.sqrt(np.mean(err ** 2))) if err.size else 0.0
    if y_true.size > 1 and float(np.var(y_true)) > 1e-9:
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else float("nan")
    else:
        r2 = float("nan")

    # Pearson / Spearman correlation
    if y_true.size >= 3 and float(np.var(y_true)) > 1e-9 and float(np.var(y_pred)) > 1e-9:
        pearson_r = float(pearsonr(y_true, y_pred)[0])
        spearman_r = float(spearmanr(y_true, y_pred)[0])
    else:
        pearson_r, spearman_r = float("nan"), float("nan")

    # Normalized MAE
    nmae = mae / score_range if score_range > 0 else float("nan")

    # Bland-Altman limits of agreement
    ba_mean_diff = float(np.mean(err))
    ba_std = float(np.std(err))
    ba_loa_upper = ba_mean_diff + 1.96 * ba_std
    ba_loa_lower = ba_mean_diff - 1.96 * ba_std

    rounded_true = np.floor(y_true + 0.5)
    rounded_pred = np.floor(y_pred + 0.5)
    rounded_errors = rounded_pred - rounded_true
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "nmae": nmae,
        "ba_mean_diff": ba_mean_diff,
        "ba_loa_upper": ba_loa_upper,
        "ba_loa_lower": ba_loa_lower,
        rounded_acc_key(rounded_tol): float(np.mean(np.abs(rounded_errors) <= rounded_tol)),
        tolerance_acc_key(score_tolerance): float(np.mean(np.abs(err) <= score_tolerance)),
    }


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    if y_true.size == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "cohen_kappa": float("nan"), "weighted_kappa": float("nan")}
    acc = float(np.mean(y_true == y_pred))

    # Per-class precision / recall / F1 via sklearn
    labels = list(range(num_classes))
    prec_arr, rec_arr, f1_arr, supp_arr = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0, average=None
    )
    macro_f1 = float(np.mean(f1_arr))

    per_class: Dict[str, float] = {}
    for c, (p, r, f, s) in enumerate(zip(prec_arr, rec_arr, f1_arr, supp_arr)):
        per_class[f"precision_c{c}"] = float(p)
        per_class[f"recall_c{c}"] = float(r)
        per_class[f"f1_c{c}"] = float(f)
        per_class[f"support_c{c}"] = int(s)

    # Cohen's kappa + linear weighted kappa (captures ordinal proximity)
    try:
        ck = float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        ck = float("nan")
    try:
        wk = float(cohen_kappa_score(y_true, y_pred, weights="linear"))
    except Exception:
        wk = float("nan")

    return {"accuracy": acc, "macro_f1": macro_f1, "cohen_kappa": ck, "weighted_kappa": wk, **per_class}


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


# --------------------------------------------------------------------------- #
# Train / eval loops                                                          #
# --------------------------------------------------------------------------- #
def _ordinal_soft_label_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Label-smoothed CE where smoothing mass decays exponentially with ordinal distance."""
    K = logits.shape[1]
    idx = torch.arange(K, device=logits.device, dtype=torch.float32)
    dist = torch.abs(idx.unsqueeze(0) - targets.float().unsqueeze(1))  # [B, K]
    soft = torch.exp(-dist / temperature)
    soft = soft / soft.sum(dim=1, keepdim=True)
    log_probs = torch.log_softmax(logits, dim=1)  # [B, K]
    loss = -(soft * log_probs).sum(dim=1)         # [B]
    if class_weights is not None:
        loss = loss * class_weights[targets.long()]
    return loss.mean()


def _corn_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CORN loss (Cao et al. 2020) on [B, K-1] conditional logits.

    For task k, the loss only considers samples with y > k-1, predicting the
    binary indicator (y > k). This factorisation guarantees the trained
    cumulative probabilities are monotonic at decode time. Per-sample weights
    are taken from the target class so the existing inverse-frequency
    `class_weights` vector keeps minority subjects in the gradient.
    """
    K = int(num_classes)
    if logits.shape[1] != K - 1:
        raise ValueError(
            f"CORN logits should have shape [B, K-1]={K - 1}, got {tuple(logits.shape)}"
        )
    losses: List[torch.Tensor] = []
    for k in range(K - 1):
        if k == 0:
            mask = torch.ones_like(targets, dtype=torch.bool)
        else:
            mask = targets > (k - 1)
        if int(mask.sum()) == 0:
            continue
        y_k = (targets[mask] > k).float()
        per_sample = F.binary_cross_entropy_with_logits(
            logits[mask, k], y_k, reduction="none"
        )
        if class_weights is not None:
            per_sample = per_sample * class_weights[targets[mask].long()]
        losses.append(per_sample.mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _corn_decode(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode CORN conditional logits.

    Returns:
        preds: [B] long, predicted rank in 0..K-1.
        probs: [B, K] per-class probabilities derived from the cumulative
               survival function, so downstream majority-vote + mean-prob
               aggregation works exactly like the CE head's softmax output.
    """
    cond = torch.sigmoid(logits)                       # P(y>k | y>k-1)
    cum = torch.cumprod(cond, dim=1)                   # P(y>k)
    preds = (cum > 0.5).long().sum(dim=1)              # 0..K-1
    ones = torch.ones_like(cum[:, :1])
    zeros = torch.zeros_like(cum[:, :1])
    cum_padded = torch.cat([ones, cum, zeros], dim=1)  # [B, K+1]
    probs = (cum_padded[:, :-1] - cum_padded[:, 1:]).clamp_min(1e-6)
    return preds, probs


def evaluate(
    model: ClinicalPredictionModel,
    loader: DataLoader,
    device: torch.device,
    spec: TaskSpec,
    encoder: Optional[LabelEncoder],
    rounded_tol: float,
    score_tolerance: float,
    loss_fn=None,
    head_kind: str = "ce",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    rows: List[Dict[str, object]] = []
    total_loss, n_loss = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["eeg"].to(device),
                batch["emg"].to(device),
                batch["imu"].to(device),
                batch["task"].to(device),
                batch["trial"].to(device),
            )
            if loss_fn is not None:
                _t = batch["target"].to(device)
                _inp = (out["pred"] if isinstance(out, dict) else out) if spec.task_type == "regression" else out
                _l = loss_fn(_inp, _t if spec.task_type == "regression" else _t.long())
                total_loss += float(_l.item()) * int(_t.numel())
                n_loss += int(_t.numel())
            if spec.task_type == "regression":
                pred_t = out["pred"] if isinstance(out, dict) else out
                pred = torch.clamp(pred_t, spec.score_min, spec.score_max).cpu().numpy()
                tgt = batch["target"].numpy().astype(float)
                for sid, y, p in zip(batch["sid"], tgt, pred.astype(float)):
                    rows.append({"subject_id": str(sid), "y_true": float(y), "y_pred": float(p)})
            else:
                if head_kind == "corn":
                    pred_t, probs_t = _corn_decode(out)
                    probs = probs_t.cpu().numpy()
                    pred = pred_t.cpu().numpy()
                else:
                    probs = torch.softmax(out, dim=1).cpu().numpy()  # [B, C]
                    pred = out.argmax(dim=1).cpu().numpy()
                tgt = batch["target"].numpy().astype(int)
                for i, (sid, y, p) in enumerate(zip(batch["sid"], tgt, pred.astype(int))):
                    row: Dict[str, object] = {"subject_id": str(sid), "y_true": int(y), "y_pred": int(p)}
                    for c in range(spec.num_classes):
                        row[f"prob_c{c}"] = float(probs[i, c])
                    rows.append(row)

    bf = pd.DataFrame(rows)
    if spec.task_type == "regression":
        sf = bf.groupby("subject_id", as_index=False).agg(
            y_true=("y_true", "first"), y_pred=("y_pred", "median")
        )
        sf["error"] = sf["y_pred"] - sf["y_true"]
        sf["abs_error"] = sf["error"].abs()
        sf["y_pred_rounded"] = sf["y_pred"].round().astype(float)
        sf["within_rounded_tol"] = (sf["abs_error"] <= rounded_tol + 0.5).astype(int)
        sf["within_score_tol"] = (sf["abs_error"] <= score_tolerance).astype(int)
        m = _regression_metrics(
            sf["y_true"].to_numpy(),
            sf["y_pred"].to_numpy(),
            rounded_tol,
            score_tolerance,
            score_range=float(spec.score_max - spec.score_min),
        )
        if loss_fn is not None:
            m["loss"] = total_loss / max(n_loss, 1)
        return m, sf
    else:
        # # Majority vote + mean class probabilities across bags per subject.
        # prob_cols = [f"prob_c{c}" for c in range(spec.num_classes)]
        # agg_dict: Dict[str, Any] = {
        #     "y_true": ("y_true", "first"),
        #     "y_pred": ("y_pred", lambda s: int(np.bincount(s.to_numpy().astype(int)).argmax())),
        # }
        # for pc in prob_cols:
        #     agg_dict[pc] = (pc, "mean")
        # sf = bf.groupby("subject_id", as_index=False).agg(**agg_dict)
        # sf["is_correct"] = (sf["y_true"] == sf["y_pred"]).astype(int)
        
        # Mean class probabilities across bags per subject, then predict by averaged probability.
        prob_cols = [f"prob_c{c}" for c in range(spec.num_classes)]

        agg_dict: Dict[str, Any] = {
            "y_true": ("y_true", "first"),
        }
        for pc in prob_cols:
            agg_dict[pc] = (pc, "mean")

        sf = bf.groupby("subject_id", as_index=False).agg(**agg_dict)

        sf["y_pred"] = sf[prob_cols].to_numpy().argmax(axis=1).astype(int)
        sf["is_correct"] = (sf["y_true"] == sf["y_pred"]).astype(int)

        # Save numeric y_true/y_pred before decoding (needed for metrics).
        y_true_int = sf["y_true"].to_numpy().astype(int)
        y_pred_int = sf["y_pred"].to_numpy().astype(int)

        m = _classification_metrics(y_true_int, y_pred_int, spec.num_classes)
        cm = _confusion_matrix(y_true_int, y_pred_int, spec.num_classes)
        if encoder is not None:
            sf["y_true"] = sf["y_true"].apply(lambda v: encoder.decode(int(v)))
            sf["y_pred"] = sf["y_pred"].apply(lambda v: encoder.decode(int(v)))
        m["confusion_matrix"] = cm.tolist()
        if loss_fn is not None:
            m["loss"] = total_loss / max(n_loss, 1)
        return m, sf


def _filter_classification_manifest(
    df: pd.DataFrame,
    spec: TaskSpec,
    subset_name: str,
) -> pd.DataFrame:
    if spec.task_type != "classification":
        return df
    valid = {str(c) for c in spec.classes}
    labels = df[spec.manifest_col]
    label_str = labels.astype(str)
    out_of_spec_mask = labels.notna() & (label_str != "nan") & ~label_str.isin(valid)
    if not out_of_spec_mask.any():
        return df
    dropped = label_str[out_of_spec_mask].value_counts().to_dict()
    print(
        f"  Warning: dropped {int(out_of_spec_mask.sum())} {subset_name} rows with "
        f"out-of-spec {spec.name} labels: {dropped}. spec.classes={spec.classes}"
    )
    return df.loc[~out_of_spec_mask].copy()


def _warn_rare_classes(
    df: pd.DataFrame,
    spec: TaskSpec,
    subset_name: str,
    min_subjects: int = 2,
) -> None:
    if spec.task_type != "classification" or df.empty:
        return
    per_subject = df.drop_duplicates("subject_id")[spec.manifest_col].astype(str)
    counts = {str(c): 0 for c in spec.classes}
    for v in per_subject:
        if v in counts:
            counts[v] += 1
    rare = {c: n for c, n in counts.items() if n < min_subjects}
    if rare:
        print(
            f"  Warning: {spec.name} {subset_name} has rare class(es) "
            f"(<{min_subjects} subjects): {rare}. Class weight / oversampling "
            f"will be down-weighted to 0 for absent classes."
        )


def train_one_task(args: argparse.Namespace) -> Dict[str, float]:
    spec = get_task(args.task)
    encoder = get_encoder(args.task) if spec.task_type == "classification" else None

    args.root = args.root.resolve()
    manifest_path = (
        args.manifest if args.manifest.is_absolute() else args.root / args.manifest
    )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Check data/samples_manifest_real_archive.csv."
        )

    df = pd.read_csv(manifest_path, dtype={"subject_id": str, "trial_id": str})
    if spec.manifest_col not in df.columns:
        raise ValueError(
            f"Manifest is missing column '{spec.manifest_col}' required for task {args.task}. "
            f"Rebuild the real-case manifest with scripts/build_real_archive_manifest.py."
        )
    df = df.sort_values(["subject_id", "task_id", "trial_number"], key=lambda c: c.astype(int))

    seed_all(args.seed)
    train_subj = [str(s) for s in getattr(args, "train_subjects", [])]
    val_subj = [str(s) for s in getattr(args, "val_subjects", [])]
    fold = getattr(args, "fold", None)
    _validate_subject_split(df, train_subj, val_subj, fold)
    train_subj = sorted(train_subj, key=_subject_sort_key)
    val_subj = sorted(val_subj, key=_subject_sort_key)
    print(f"  Fold: {fold}")
    print(f"  Train subjects ({len(train_subj)}): {_format_subjects(train_subj)}")
    print(f"  Val subjects   ({len(val_subj)}): {_format_subjects(val_subj)}")

    df_train = df[df["subject_id"].astype(str).isin(train_subj)].copy()
    df_val = df[df["subject_id"].astype(str).isin(val_subj)].copy()
    if spec.task_type == "classification":
        df_train = _filter_classification_manifest(df_train, spec, "train")
        df_val = _filter_classification_manifest(df_val, spec, "val")
        _warn_rare_classes(df_train, spec, "train", min_subjects=2)
        _warn_rare_classes(df_val, spec, "val", min_subjects=2)
    print(f"  Train samples: {len(df_train)} | Val samples: {len(df_val)}")

    cfg = WBYDTWConfig(
        output_length=args.seq_len,
        dtw_length=args.dtw_length,
        band_radius=0.15,
        alpha=0.7,
        beta=0.3,
    )
    cache_dir = getattr(args, "cache_dir", None)
    tr_store = TaskStore(df_train, args.root, spec, encoder, args.seq_len, cfg,
                         args.alignment_mode, args.eeg_fs, not args.no_preprocess,
                         cache_dir)
    ev_store = TaskStore(df_val, args.root, spec, encoder, args.seq_len, cfg,
                         args.alignment_mode, args.eeg_fs, not args.no_preprocess,
                         cache_dir)

    _train_bag_ds = BagDS(tr_store, args.bag_size, args.train_bags, args.seed, deterministic=False)
    nw = int(getattr(args, "num_workers", 0))
    pin = nw > 0 and torch.cuda.is_available()
    persistent = nw > 0
    _dl_kwargs = dict(num_workers=nw, pin_memory=pin, persistent_workers=persistent)

    if spec.task_type == "classification" and getattr(args, "minority_oversample", False):
        _bag_targets = _train_bag_ds.subject_target_per_bag()
        _K_s = spec.num_classes
        _counts_s = np.bincount(_bag_targets, minlength=_K_s).astype(float)
        _per_class_w = np.where(
            _counts_s > 0,
            1.0 / np.sqrt(np.maximum(_counts_s, 1.0)),
            0.0,
        )
        _sample_w = _per_class_w[_bag_targets]
        print(f"  Minority oversample (sqrt-inv-freq) sampler weights per class: "
              f"{ {c: round(float(_per_class_w[c]), 3) for c in range(_K_s)} }")
        _sampler = WeightedRandomSampler(
            weights=torch.as_tensor(_sample_w, dtype=torch.double),
            num_samples=len(_train_bag_ds),
            replacement=True,
        )
        tr_loader = DataLoader(_train_bag_ds, batch_size=args.batch_size, sampler=_sampler, **_dl_kwargs)
    else:
        tr_loader = DataLoader(_train_bag_ds, batch_size=args.batch_size, shuffle=True, **_dl_kwargs)
    ev_loader = DataLoader(
        BagDS(ev_store, args.eval_bag_size, args.eval_bags, args.seed + 7, deterministic=True),
        batch_size=args.batch_size, shuffle=False, **_dl_kwargs,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if spec.task_type == "classification":
        head_kind = getattr(spec, "default_head", "ce") if args.head == "auto" else args.head
    else:
        head_kind = "ce"
    print(f"  Classification head: {head_kind if spec.task_type == 'classification' else 'n/a (regression)'}")
    model = ClinicalPredictionModel(
        task_type=spec.task_type,
        num_classes=spec.num_classes if spec.task_type == "classification" else None,
        eeg_channels=args.eeg_channels,
        emg_channels=args.emg_channels,
        imu_channels=args.imu_channels,
        f=args.feature,
        te=args.task_emb,
        p=args.dropout,
        score_min=spec.score_min,
        score_max=spec.score_max,
        bin_step=spec.bin_step,
        head_kind=head_kind,
        use_graph=not getattr(args, "no_mdfan", False),
        use_attention=not getattr(args, "no_attention", False),
        enabled_modalities=tuple(getattr(args, "modalities", "eeg+emg+imu").split("+")),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # LR scheduler: linear warmup → cosine annealing.
    if args.warmup_epochs > 0 and args.epochs > args.warmup_epochs:
        scheduler = SequentialLR(
            opt,
            schedulers=[
                LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs),
                CosineAnnealingLR(
                    opt,
                    T_max=max(1, args.epochs - args.warmup_epochs),
                    eta_min=args.lr * args.cosine_min_lr_ratio,
                ),
            ],
            milestones=[args.warmup_epochs],
        )
    else:
        scheduler = None

    # Class weights for classification (computed per-fold from training subjects only).
    if spec.task_type == "classification" and getattr(args, "class_weight", True):
        _targets_train = np.array(
            [entry["target"] for entry in tr_store.subjects.values()], dtype=int
        )
        _n = len(_targets_train)
        _K = spec.num_classes
        _counts = np.bincount(_targets_train, minlength=_K).astype(float)
        _weights = np.where(
            _counts > 0,
            _n / (_K * np.maximum(_counts, 1.0)),
            0.0,
        )
        _class_weights_tensor: Optional[torch.Tensor] = torch.tensor(
            _weights, dtype=torch.float32, device=device
        )
        print(f"  Class weights ({spec.name}): { {c: round(float(w), 3) for c, w in enumerate(_weights)} }")
    else:
        _class_weights_tensor = None

    if spec.task_type == "regression":
        huber_loss_fn = nn.SmoothL1Loss(beta=args.huber_delta) if spec.loss == "SmoothL1Loss" else nn.MSELoss()
        loss_fn = huber_loss_fn  # legacy single-head path
    else:
        loss_fn = nn.CrossEntropyLoss(
            weight=_class_weights_tensor,
            label_smoothing=args.label_smoothing,
        )
    if spec.task_type == "regression":
        _eval_loss_fn = huber_loss_fn
    elif head_kind == "corn":
        _K_corn = spec.num_classes
        _eval_loss_fn = lambda _logits, _tgt: _corn_loss(
            _logits, _tgt, _K_corn, _class_weights_tensor
        )
    else:
        _eval_loss_fn = loss_fn

    # Task-driven regression tolerances. --tolerance overrides only the raw-error tolerance.
    rounded_tol = spec.rounded_tol
    score_tolerance = args.tolerance if args.tolerance is not None else spec.score_tolerance
    rounded_key = rounded_acc_key(rounded_tol)
    tolerance_key = tolerance_acc_key(score_tolerance)

    best, best_state, best_ep, stale, hist = None, None, 0, 0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        tr_subj_preds: Dict[str, List[float]] = {}
        tr_subj_targets: Dict[str, float] = {}
        for batch in tr_loader:
            out = model(
                batch["eeg"].to(device), batch["emg"].to(device), batch["imu"].to(device),
                batch["task"].to(device), batch["trial"].to(device),
            )
            if spec.task_type == "regression":
                target = batch["target"].to(device)
                if isinstance(out, dict):
                    # Hybrid head: Huber on the continuous prediction +
                    # cross-entropy on the bin logits against the rounded target.
                    huber = huber_loss_fn(out["pred"], target)
                    bin_idx = model.output_head.target_to_bin_index(target)
                    ce = F.cross_entropy(
                        out["logits"], bin_idx, label_smoothing=args.label_smoothing
                    )
                    loss = huber + args.ce_weight * ce

                    # Ordinal auxiliary BCE: cumulative survival probabilities.
                    # Derives ordinal structure from the existing bin logits without
                    # adding parameters; thresholds are bin-center midpoints.
                    if args.ordinal_weight > 0.0:
                        _probs = torch.softmax(out["logits"], dim=1)
                        _cdf = torch.cumsum(_probs, dim=1)[:, :-1].clamp(1e-6, 1.0 - 1e-6)
                        _cdf_logits = torch.log(_cdf) - torch.log(1.0 - _cdf)

                        _centers = model.output_head.bin_centers
                        _thresholds = (_centers[:-1] + _centers[1:]) / 2.0
                        _cdf_tgt = (target.unsqueeze(1) <= _thresholds.unsqueeze(0)).float()

                        ordinal_loss = F.binary_cross_entropy_with_logits(_cdf_logits, _cdf_tgt)
                        loss = loss + args.ordinal_weight * ordinal_loss
                else:
                    loss = loss_fn(out, target)
            else:
                _tgt_long = batch["target"].to(device).long()
                if head_kind == "corn":
                    loss = _corn_loss(
                        out, _tgt_long, spec.num_classes, _class_weights_tensor
                    )
                elif args.ordinal_cls_temp > 0.0:
                    loss = _ordinal_soft_label_ce(
                        out, _tgt_long, args.ordinal_cls_temp, _class_weights_tensor
                    )
                else:
                    loss = loss_fn(out, _tgt_long)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total += float(loss.item()) * batch["target"].numel()
            n += batch["target"].numel()
            with torch.no_grad():
                if spec.task_type == "regression":
                    _pv = torch.clamp(
                        (out["pred"] if isinstance(out, dict) else out).detach(),
                        spec.score_min, spec.score_max,
                    ).cpu().numpy()
                else:
                    if head_kind == "corn":
                        _pv_t, _ = _corn_decode(out.detach())
                        _pv = _pv_t.cpu().numpy()
                    else:
                        _pv = out.argmax(dim=1).detach().cpu().numpy()
                for _sid, _t, _p in zip(batch["sid"], batch["target"].numpy(), _pv):
                    tr_subj_preds.setdefault(str(_sid), []).append(float(_p))
                    tr_subj_targets[str(_sid)] = float(_t)
        if scheduler is not None:
            scheduler.step()
        current_lr = float(opt.param_groups[0]["lr"])

        # Compute subject-aggregated train metrics from predictions collected during training.
        _sids = sorted(tr_subj_preds.keys())
        if spec.task_type == "regression":
            _ty = np.array([tr_subj_targets[s] for s in _sids])
            _tp = np.array([float(np.median(tr_subj_preds[s])) for s in _sids])
            train_m = _regression_metrics(
                _ty, _tp, rounded_tol, score_tolerance,
                float(spec.score_max - spec.score_min),
            )
        else:
            _ty = np.array([int(tr_subj_targets[s]) for s in _sids])
            _tp = np.array([
                int(np.bincount(np.array(tr_subj_preds[s], dtype=int)).argmax())
                for s in _sids
            ])
            train_m = _classification_metrics(_ty, _tp, spec.num_classes)
        train_m["loss"] = total / max(n, 1)

        m, _ = evaluate(model, ev_loader, device, spec, encoder, rounded_tol, score_tolerance,
                        loss_fn=_eval_loss_fn, head_kind=head_kind)
        if spec.task_type == "regression":
            ckpt_metric = getattr(args, "checkpoint_metric", "rounded_acc")
            if ckpt_metric == "mae":
                primary_value = float(m.get("mae", math.inf))
                better = best is None or primary_value < float(best.get("mae", math.inf)) - 1e-12
            else:  # rounded_acc (default)
                primary_value = float(m.get(rounded_key, 0.0))
                better = (
                    best is None
                    or primary_value > float(best.get(rounded_key, 0.0)) + 1e-12
                    or (
                        abs(primary_value - float(best.get(rounded_key, 0.0))) <= 1e-12
                        and m.get("mae", math.inf) < best.get("mae", math.inf)
                    )
                )
        else:
            ckpt_metric_cls = getattr(args, "checkpoint_metric", "rounded_acc")
            if ckpt_metric_cls == "weighted_kappa":
                primary_value = float(m.get("weighted_kappa", float("-inf")))
                if not math.isfinite(primary_value):
                    primary_value = float("-inf")
                best_value = float(best.get("weighted_kappa", float("-inf"))) if best is not None else float("-inf")
                if not math.isfinite(best_value):
                    best_value = float("-inf")
                better = (
                    best is None
                    or primary_value > best_value + 1e-12
                    or (
                        abs(primary_value - best_value) <= 1e-12
                        and m.get("macro_f1", 0.0) > best.get("macro_f1", 0.0) + 1e-12
                    )
                )
            elif ckpt_metric_cls == "macro_f1":
                primary_value = float(m.get("macro_f1", 0.0))
                better = best is None or primary_value > float(best.get("macro_f1", 0.0)) + 1e-12
            else:
                primary_value = float(m.get("accuracy", 0.0))
                better = best is None or primary_value > float(best.get("accuracy", 0.0)) + 1e-12
        if better:
            best, best_state, best_ep, stale = m, deepcopy(model.state_dict()), ep, 0
        else:
            stale += 1
        _train_hist = {f"train_{k}": v for k, v in train_m.items() if k != "confusion_matrix"}
        _val_hist = {f"val_{k}": v for k, v in m.items() if k != "confusion_matrix"}
        hist.append({"epoch": ep, "lr": current_lr, **_train_hist, **_val_hist, "is_best": better})

        if spec.task_type == "regression":
            print(
                f"Epoch {ep}/{args.epochs} | "
                f"[Train] loss={train_m['loss']:.4f} MAE={train_m['mae']:.3f} "
                f"round-acc@{rounded_tol:g}={train_m[rounded_key]*100:.1f}% | "
                f"[Val] loss={m.get('loss', float('nan')):.4f} "
                f"MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} "
                f"R2={m['r2']:.3f} round-acc@{rounded_tol:g}={m[rounded_key]*100:.1f}% "
                f"tol@{score_tolerance:g}={m[tolerance_key]*100:.1f}% {'*' if better else ''}",
                flush=True,
            )
        else:
            print(
                f"Epoch {ep}/{args.epochs} | "
                f"[Train] loss={train_m['loss']:.4f} acc={train_m['accuracy']*100:.1f}% "
                f"F1={train_m['macro_f1']:.3f} | "
                f"[Val] loss={m.get('loss', float('nan')):.4f} "
                f"acc={m['accuracy']*100:.1f}% F1={m['macro_f1']:.3f} "
                f"κ={m.get('cohen_kappa', float('nan')):.3f} "
                f"wκ={m.get('weighted_kappa', float('nan')):.3f} {'*' if better else ''}",
                flush=True,
            )
        if args.patience > 0 and stale >= args.patience:
            print(f"  early stopping at epoch {ep}; best epoch={best_ep}")
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    final_m, sf = evaluate(model, ev_loader, device, spec, encoder, rounded_tol, score_tolerance,
                           loss_fn=_eval_loss_fn, head_kind=head_kind)

    # Resolve output paths and announce them before writing anything.
    ckpt_path = _checkpoint_path_for_fold(args, spec)
    fold_suffix = f"_fold{int(getattr(args, 'fold'))}" if getattr(args, "fold", None) is not None else ""
    logs_dir = _get_out_dir(args, spec) / f"{spec.name}{fold_suffix}_logs"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir:  {_get_out_dir(args, spec)}")
    print(f"  Checkpoint:  {ckpt_path}")
    print(f"  Logs dir:    {logs_dir}")

    # Persist the checkpoint.
    payload = {
        "task": spec.name,
        "fold": getattr(args, "fold", None),
        "n_splits": getattr(args, "n_splits", None),
        "split_json": str(getattr(args, "split_json_resolved", "")),
        "train_subjects": train_subj,
        "val_subjects": val_subj,
        "task_type": spec.task_type,
        "num_classes": spec.num_classes,
        "score_min": spec.score_min,
        "score_max": spec.score_max,
        "classes": list(spec.classes),
        "state_dict": best_state,
        "model_config": {
            "eeg_channels": args.eeg_channels,
            "emg_channels": args.emg_channels,
            "imu_channels": args.imu_channels,
            "feature": args.feature,
            "task_emb": args.task_emb,
            "dropout": args.dropout,
            "score_min": spec.score_min,
            "score_max": spec.score_max,
            "bin_step": spec.bin_step,
            "head_kind": head_kind,
        },
        "metrics": {k: v for k, v in final_m.items() if k != "confusion_matrix"},
    }
    torch.save(payload, ckpt_path)
    print(f"  checkpoint saved → {ckpt_path}")

    # Persist history & per-subject predictions.
    pd.DataFrame(hist).to_csv(logs_dir / "training_history.csv", index=False)
    sf.to_csv(logs_dir / "val_predictions.csv", index=False)
    (logs_dir / "metrics.json").write_text(json.dumps(final_m, ensure_ascii=False, indent=2))

    # Extra visualization files.
    if spec.task_type == "regression":
        # Bland-Altman data for agreement analysis.
        ba_df = pd.DataFrame({
            "subject_id": sf["subject_id"],
            "mean": (sf["y_true"] + sf["y_pred"]) / 2,
            "diff": sf["error"],
        })
        ba_df.to_csv(logs_dir / "bland_altman_data.csv", index=False)

        # Calibration data: bucket y_true into quantile bins, compare mean pred vs mean true.
        n_cal_bins = min(5, max(2, len(sf) // 3))
        try:
            sf_sorted = sf.sort_values("y_true").copy()
            sf_sorted["cal_bin"] = pd.qcut(
                sf_sorted["y_true"], q=n_cal_bins, duplicates="drop", labels=False
            )
            cal_df = sf_sorted.groupby("cal_bin", as_index=False).agg(
                mean_true=("y_true", "mean"),
                mean_pred=("y_pred", "mean"),
                count=("y_true", "count"),
                mean_abs_error=("abs_error", "mean"),
            )
            cal_df.to_csv(logs_dir / "calibration_data.csv", index=False)
        except Exception:
            pass
    else:
        # Per-class precision / recall / F1.
        per_class_data: Dict[str, Any] = {}
        for c in range(spec.num_classes):
            label = encoder.decode(c) if encoder is not None else str(c)
            per_class_data[str(label)] = {
                k: final_m.get(f"{k}_c{c}", float("nan"))
                for k in ("precision", "recall", "f1", "support")
            }
        (logs_dir / "per_class_metrics.json").write_text(
            json.dumps(per_class_data, ensure_ascii=False, indent=2)
        )

        # Labeled confusion matrix CSV.
        if encoder is not None:
            cls_labels = [str(encoder.decode(c)) for c in range(spec.num_classes)]
        else:
            cls_labels = [str(c) for c in range(spec.num_classes)]
        cm_arr = np.array(final_m.get("confusion_matrix", []))
        if cm_arr.ndim == 2:
            cm_df = pd.DataFrame(
                cm_arr,
                index=[f"true_{l}" for l in cls_labels],
                columns=[f"pred_{l}" for l in cls_labels],
            )
            cm_df.to_csv(logs_dir / "confusion_matrix.csv")

    return final_m


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Single-task clinical trainer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output layout (default, task=FMA_UE):
  results/FMA_UE/baseline/
  ├── FMA_UE_fold1.pth                  ← best checkpoint (fold 1)
  ├── FMA_UE_fold1_logs/
  │   ├── training_history.csv
  │   ├── val_predictions.csv
  │   ├── metrics.json
  │   └── bland_altman_data.csv / calibration_data.csv
  ├── FMA_UE_3fold_summary.{csv,json}   ← cross-fold aggregation
  └── config.json                       ← experiment config snapshot

Override the root with --out-dir, or the exact .pth path with --checkpoint.
""",
    )
    ap.add_argument("--task", required=True, help="One of FMA_UE, BI, hand_tone, hand_function")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                    help="Project root (used to resolve relative paths). Default: repo root.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Root output directory for checkpoints, logs, and summary files. "
                         "Defaults to results/<task>/baseline/ under --root. "
                         "Structure: <out-dir>/<task>_fold<n>.pth  +  <out-dir>/<task>_fold<n>_logs/")
    ap.add_argument("--manifest", type=Path, default=Path("data/samples_manifest_real_archive.csv"))
    ap.add_argument("--split-json", type=Path, default=Path("splits/split_patient_3fold.json"),
                    help="Patient-level split JSON. Default runs all folds in this file.")
    ap.add_argument("--fold", type=int, default=0,
                    help="Fold id to train. Use 0 or -1 to train all folds from --split-json.")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Explicit checkpoint save path (supports {fold} placeholder). "
                         "Overrides --out-dir for .pth placement only; logs still go to --out-dir.")
    ap.add_argument("--alignment-mode", default="adk",
                    help="Tri-modal alignment strategy. "
                         "'adk' (default) = Tri-ADK-Knot (WBy-DTW on EMG↔IMU + EEG linear resample). "
                         "'adk_no_dtw' = same grid structure but knot_strength=0 (drops WBy-DTW, keeps tri-alignment). "
                         "'resample' = independent linear resample per modality (no tri-alignment, no DTW).")
    # ----- Ablation switches (Section 4.6) -----
    ap.add_argument("--modalities", default="eeg+emg+imu",
                    choices=["eeg+emg+imu", "eeg", "emg", "imu",
                             "eeg+emg", "eeg+imu", "emg+imu"],
                    help="Modality subset for ablation. Default uses all three (Ours). "
                         "Missing modalities are zeroed at the backbone input — the "
                         "tri-modal graph topology is preserved.")
    ap.add_argument("--no-mdfan", action="store_true",
                    help="Disable MDFAN graph attention (sets use_graph=False).")
    ap.add_argument("--no-attention", action="store_true",
                    help="Disable temporal attention pooling.")
    ap.add_argument("--ablation-tag", default=None,
                    help="Optional tag. When set, default --out-dir becomes "
                         "results/ablation/<kind>/<tag>/<task>/ where <kind> is "
                         "'modality' if --modalities != tri else 'module'.")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--dtw-length", type=int, default=32)
    ap.add_argument("--eeg-channels", type=int, default=30)
    ap.add_argument("--emg-channels", type=int, default=4)
    ap.add_argument("--imu-channels", type=int, default=24)
    ap.add_argument("--eeg-fs", type=float, default=EEG_FS_DEFAULT)
    ap.add_argument("--no-preprocess", action="store_true")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="Directory for disk-cached aligned trial arrays (.npz). "
                         "Skips alignment on subsequent runs when data/config is unchanged. "
                         "Suggested: --cache-dir .aligned_cache")
    ap.add_argument("--feature", type=int, default=64)
    ap.add_argument("--task-emb", type=int, default=12)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--bag-size", type=int, default=4)
    ap.add_argument("--eval-bag-size", type=int, default=4)
    ap.add_argument("--train-bags", type=int, default=40)
    ap.add_argument("--eval-bags", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--ce-weight", type=float, default=0.2,
                    help="Weight on the cross-entropy term of the hybrid regression loss "
                         "(Huber + ce_weight * CE). Ignored for tasks without bin_step.")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--tolerance", type=float, default=None,
                    help="Override the task's score_tolerance (raw-error threshold). "
                         "Defaults to the task spec (FMA_UE: 1.5, BI: 10.0).")
    ap.add_argument("--warmup-epochs", type=int, default=8,
                    help="Linear LR warmup epochs before cosine annealing.")
    ap.add_argument("--cosine-min-lr-ratio", type=float, default=0.01,
                    help="Minimum LR = lr * cosine_min_lr_ratio at end of cosine annealing.")
    ap.add_argument("--ordinal-weight", type=float, default=0.15,
                    help="Weight on the ordinal cumulative BCE auxiliary loss for regression "
                         "tasks with hybrid head. Set 0 to disable.")
    ap.add_argument("--minority-oversample", action="store_true",
                    help="Classification only: use WeightedRandomSampler with sqrt-inverse-frequency "
                         "weights so rare classes appear in nearly every batch. Recommend combining "
                         "with --no-class-weight to avoid double-correcting minorities.")
    ap.add_argument("--no-class-weight", dest="class_weight", action="store_false",
                    help="Disable per-fold class-frequency balancing for classification CE loss.")
    ap.set_defaults(class_weight=True)
    ap.add_argument("--ordinal-cls-temp", type=float, default=0.0,
                    help="Temperature for ordinal soft-label CE for classification tasks. "
                         "Set > 0 (e.g. 1.5) to enable distance-penalized label smoothing.")
    ap.add_argument("--head", type=str, default="auto",
                    choices=["auto", "ce", "corn"],
                    help="Classification head type. 'auto' uses TaskSpec.default_head "
                         "(currently 'corn' for hand_tone / hand_function, 'ce' otherwise). "
                         "'ce' = plain Linear → cross-entropy logits (legacy). "
                         "'corn' = CORN ordinal head (K-1 conditional logits trained with "
                         "CORN loss); recommended for ordered clinical scales because it "
                         "natively encodes adjacency and reduces off-by-one errors. "
                         "Ignored for regression tasks.")
    ap.add_argument("--checkpoint-metric", type=str, default="rounded_acc",
                    choices=["rounded_acc", "mae", "macro_f1", "weighted_kappa"],
                    help="Primary metric for checkpoint selection. "
                         "'rounded_acc' (default, regression): maximize rounded accuracy (±rounded_tol); "
                         "  for classification falls back to plain accuracy. "
                         "'mae' (regression only): minimize MAE directly. "
                         "'macro_f1' (classification only): maximize macro-F1 to prevent the "
                         "  best checkpoint from abandoning minority classes. "
                         "'weighted_kappa' (classification only): maximize ordinal agreement.")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--num-workers", type=int, default=4,
                    help="DataLoader worker processes for background data loading. "
                         "Set 0 to disable (single-process). Default: 4.")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    args.root = args.root.resolve()
    spec = get_task(args.task)
    split_path, split_data = _load_fold_file(args.root, args.split_json)
    selected_folds = _select_folds(split_data, args.fold)

    out_dir = _get_out_dir(args, spec)
    print("=" * 60)
    print(f"Training task:  {args.task}  ({spec.task_type})")
    print(f"Split file:     {split_path}")
    print(f"Output dir:     {out_dir}")
    print(f"Folds to train: {[int(f['fold']) for f in selected_folds]}")
    print("=" * 60)

    all_results: List[Dict[str, Any]] = []
    for fold_info in selected_folds:
        fold_args = deepcopy(args)
        fold_args.fold = int(fold_info["fold"])
        fold_args.n_splits = int(split_data.get("n_splits", len(split_data["folds"])))
        fold_args.split_json_resolved = split_path
        fold_args.train_subjects, fold_args.val_subjects = _fold_subjects(fold_info)

        print("")
        print("-" * 60)
        print(f"Starting fold {fold_args.fold}/{fold_args.n_splits}")
        print("-" * 60)
        metrics = train_one_task(fold_args)
        row = {
            "fold": fold_args.fold,
            **{k: v for k, v in metrics.items() if not isinstance(v, (list, dict))},
        }
        all_results.append(row)

    if all_results:
        summary_dir = _get_out_dir(args, spec)
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_stem = f"{spec.name}_{len(selected_folds)}fold_summary"
        summary_csv = summary_dir / f"{summary_stem}.csv"
        summary_json = summary_dir / f"{summary_stem}.json"
        pd.DataFrame(all_results).sort_values("fold").to_csv(summary_csv, index=False)
        summary_json.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

        # Save experiment-level config (fold-independent args only).
        _fold_keys = {"fold", "train_subjects", "val_subjects", "split_json_resolved", "n_splits"}
        exp_cfg = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k not in _fold_keys
        }
        exp_cfg["folds_trained"] = [int(f["fold"]) for f in selected_folds]
        (summary_dir / "config.json").write_text(json.dumps(exp_cfg, ensure_ascii=False, indent=2))

        print("")
        print("=" * 60)
        print(f"Finished {len(selected_folds)} fold(s).")
        print(f"Summary saved → {summary_csv}")
        print(f"Summary saved → {summary_json}")
        print(f"Config  saved → {summary_dir / 'config.json'}")
        print("=" * 60)


if __name__ == "__main__":
    main()
