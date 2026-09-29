"""Experimental shared-encoder trainer for all four CMK-AGN endpoints.

The outer split is strictly patient independent.  Within each outer-training
fold, a deterministic patient-level inner validation set is held out for early
stopping; the outer-test fold is used only once for final metrics.  This script
is intentionally separate from ``train.py`` so the established single-task
tri model remains an untouched comparator.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alignment.wby_dtw import WBYDTWConfig
from clinical_model import MultiTaskClinicalPredictionModel
from task_config import TASK_CONFIGS, get_encoder
from train import (
    EEG_FS_DEFAULT, _classification_metrics, _corn_decode, _corn_loss,
    _fold_subjects, _load_fold_file, _regression_metrics, _select_folds,
    _validate_subject_split, _as_project_path, load_aligned, seed_all,
)


TASKS = ("FMA_UE", "BI", "hand_tone", "hand_function")


class MultiTaskStore:
    def __init__(self, df: pd.DataFrame, root: Path, args: argparse.Namespace, cfg: WBYDTWConfig):
        self.subjects: Dict[str, Dict[str, object]] = {}
        tone_encoder = get_encoder("hand_tone")
        function_encoder = get_encoder("hand_function")
        for _, row in df.iterrows():
            sid = str(row["subject_id"])
            if any(pd.isna(row[TASK_CONFIGS[name].manifest_col]) for name in TASKS):
                raise ValueError(f"Subject {sid} has a missing multi-task label.")
            target = {
                "FMA_UE": float(row["fma_ue"]),
                "BI": float(row["bi"]),
                "hand_tone": int(tone_encoder.encode(row["hand_tone"])),
                "hand_function": int(function_encoder.encode(row["hand_function"])),
            }
            eeg, emg, imu = load_aligned(
                root,
                _as_project_path(root, str(row["eeg_path"])),
                _as_project_path(root, str(row["emg_path"])),
                args.seq_len, cfg, args.alignment_mode, args.eeg_fs,
                not args.no_preprocess, args.cache_dir,
            )
            entry = self.subjects.setdefault(sid, {"targets": target, "trials": []})
            if entry["targets"] != target:
                raise ValueError(f"Inconsistent multi-task labels within subject {sid}.")
            entry["trials"].append({
                "eeg": eeg, "emg": emg, "imu": imu,
                "task": max(int(row["task_id"]) - 1, 0),
                "trial": max(int(row["trial_number"]) - 1, 0),
            })


class MultiTaskBagDataset(Dataset):
    def __init__(self, store: MultiTaskStore, bag_size: int, bags_per_subject: int, seed: int, deterministic: bool):
        self.store, self.sids = store, sorted(store.subjects, key=lambda value: int(value))
        self.bag, self.bags_per_subject, self.seed, self.deterministic = bag_size, bags_per_subject, seed, deterministic

    def __len__(self) -> int:
        return len(self.sids) * self.bags_per_subject

    def subject_sampling_weights(self) -> np.ndarray:
        tone = np.array([self.store.subjects[sid]["targets"]["hand_tone"] for sid in self.sids], dtype=int)
        function = np.array([self.store.subjects[sid]["targets"]["hand_function"] for sid in self.sids], dtype=int)
        def inv_sqrt(values: np.ndarray) -> np.ndarray:
            counts = np.bincount(values, minlength=6).astype(float)
            return np.where(counts > 0, 1.0 / np.sqrt(counts), 0.0)[values]
        per_subject = 0.5 * (inv_sqrt(tone) + inv_sqrt(function))
        return np.repeat(per_subject, self.bags_per_subject)

    def __getitem__(self, index: int) -> Dict[str, object]:
        subject_offset, bag_offset = divmod(index, self.bags_per_subject)
        sid = self.sids[subject_offset]
        entry = self.store.subjects[sid]
        trials = entry["trials"]
        rng = np.random.default_rng(self.seed + 7919 * subject_offset + 173 * bag_offset) if self.deterministic else np.random.default_rng()
        ids = rng.choice(len(trials), size=self.bag, replace=len(trials) < self.bag)
        picked = [trials[int(i)] for i in ids]
        return {
            "eeg": torch.stack([x["eeg"] for x in picked]),
            "emg": torch.stack([x["emg"] for x in picked]),
            "imu": torch.stack([x["imu"] for x in picked]),
            "task": torch.tensor([x["task"] for x in picked], dtype=torch.long),
            "trial": torch.tensor([x["trial"] for x in picked], dtype=torch.long),
            "targets": {name: torch.tensor(value) for name, value in entry["targets"].items()},
            "sid": sid,
        }


def _split_inner(subjects: List[str], fraction: float, seed: int) -> Tuple[List[str], List[str]]:
    if not 0.0 < fraction < 0.5:
        raise ValueError("--inner-val-fraction must be in (0, 0.5).")
    shuffled = list(subjects)
    random.Random(seed).shuffle(shuffled)
    n_val = max(2, round(len(shuffled) * fraction))
    if len(shuffled) - n_val < 2:
        raise ValueError("Not enough outer-training patients after the inner validation split.")
    return sorted(shuffled[n_val:], key=int), sorted(shuffled[:n_val], key=int)


def _device_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        "eeg": batch["eeg"].to(device), "emg": batch["emg"].to(device), "imu": batch["imu"].to(device),
        "task": batch["task"].to(device), "trial": batch["trial"].to(device),
        "targets": {name: target.to(device) for name, target in batch["targets"].items()}, "sid": batch["sid"],
    }


def _supervised_loss(out: Dict[str, object], targets: Dict[str, torch.Tensor], ce_weight: float) -> torch.Tensor:
    fma = F.smooth_l1_loss(out["FMA_UE"]["pred"], targets["FMA_UE"].float()) / 20.0
    fma_target = torch.clamp(torch.round(targets["FMA_UE"]), 0, 20).long()
    fma = fma + ce_weight * F.cross_entropy(out["FMA_UE"]["logits"], fma_target)
    bi_bins = out["BI"]["logits"].shape[1]
    bi_target = torch.clamp(torch.round(targets["BI"] / 7.0), 0, bi_bins - 1).long()
    bi = F.smooth_l1_loss(out["BI"]["pred"], targets["BI"].float()) / 100.0
    bi = bi + ce_weight * F.cross_entropy(out["BI"]["logits"], bi_target)
    tone = _corn_loss(out["hand_tone"], targets["hand_tone"].long(), 6, None)
    function = _corn_loss(out["hand_function"], targets["hand_function"].long(), 6, None)
    return fma + bi + tone + function


def _masked_inputs(batch: Dict[str, object], ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = [batch["eeg"], batch["emg"], batch["imu"]]
    if ratio <= 0.0:
        return tuple(inputs)
    mask = torch.rand(inputs[0].shape[0], inputs[0].shape[1], 1, inputs[0].shape[-1], device=inputs[0].device) < ratio
    return tuple(x.masked_fill(mask, 0.0) for x in inputs)


def _pretraining_loss(model: MultiTaskClinicalPredictionModel, batch: Dict[str, object], mask_ratio: float, consistency_weight: float) -> torch.Tensor:
    with torch.no_grad():
        target_feature = model.encode(batch["eeg"], batch["emg"], batch["imu"], batch["task"], batch["trial"])
    masked = _masked_inputs(batch, mask_ratio)
    masked_feature = model.encode(*masked, batch["task"], batch["trial"])
    reconstruction = F.mse_loss(model.masked_representation_decoder(masked_feature), target_feature)
    # Cross-modal and missing-modality perturbations operate on the same shared encoder.
    zeros = [torch.zeros_like(batch[name]) for name in ("eeg", "emg", "imu")]
    modal_features = []
    for index in range(3):
        selected = [zeros[0], zeros[1], zeros[2]]
        selected[index] = batch[("eeg", "emg", "imu")[index]]
        modal_features.append(model.encode(*selected, batch["task"], batch["trial"]))
    consistency = sum(F.mse_loss(feature, target_feature) for feature in modal_features) / 3.0
    drop_index = int(torch.randint(0, 3, (1,), device=batch["eeg"].device).item())
    dropped = [batch["eeg"], batch["emg"], batch["imu"]]
    dropped[drop_index] = torch.zeros_like(dropped[drop_index])
    missing = F.mse_loss(model.encode(*dropped, batch["task"], batch["trial"]), target_feature)
    return reconstruction + consistency_weight * (consistency + missing)


def _evaluate(model: MultiTaskClinicalPredictionModel, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, Dict[str, float]], Dict[str, pd.DataFrame]]:
    model.eval()
    raw: Dict[str, List[Dict[str, object]]] = {name: [] for name in TASKS}
    with torch.no_grad():
        for source_batch in loader:
            batch = _device_batch(source_batch, device)
            out = model(batch["eeg"], batch["emg"], batch["imu"], batch["task"], batch["trial"])
            for name, maximum in (("FMA_UE", 20.0), ("BI", 100.0)):
                values = torch.clamp(out[name]["pred"], 0.0, maximum).cpu().numpy()
                target = batch["targets"][name].cpu().numpy()
                raw[name].extend({"subject_id": str(sid), "y_true": float(y), "y_pred": float(p)} for sid, y, p in zip(batch["sid"], target, values))
            for name in ("hand_tone", "hand_function"):
                pred, probs = _corn_decode(out[name])
                for i, (sid, y, p) in enumerate(zip(batch["sid"], batch["targets"][name].cpu().numpy(), pred.cpu().numpy())):
                    row = {"subject_id": str(sid), "y_true": int(y), "y_pred": int(p)}
                    row.update({f"prob_c{c}": float(probs[i, c]) for c in range(6)})
                    raw[name].append(row)
    metrics: Dict[str, Dict[str, float]] = {}
    subject_frames: Dict[str, pd.DataFrame] = {}
    for name in ("FMA_UE", "BI"):
        frame = pd.DataFrame(raw[name]).groupby("subject_id", as_index=False).agg(y_true=("y_true", "first"), y_pred=("y_pred", "median"))
        frame["error"] = frame.y_pred - frame.y_true
        frame["abs_error"] = frame.error.abs()
        spec = TASK_CONFIGS[name]
        metrics[name] = _regression_metrics(frame.y_true.to_numpy(), frame.y_pred.to_numpy(), spec.rounded_tol, spec.score_tolerance, spec.score_max - spec.score_min)
        subject_frames[name] = frame
    for name in ("hand_tone", "hand_function"):
        frame = pd.DataFrame(raw[name])
        prob_cols = [f"prob_c{c}" for c in range(6)]
        frame = frame.groupby("subject_id", as_index=False).agg({"y_true": "first", **{col: "mean" for col in prob_cols}})
        frame["y_pred"] = frame[prob_cols].to_numpy().argmax(axis=1)
        metrics[name] = _classification_metrics(frame.y_true.to_numpy().astype(int), frame.y_pred.to_numpy().astype(int), 6)
        encoder = get_encoder(name)
        frame["y_true"] = frame["y_true"].apply(lambda value: encoder.decode(int(value)))
        frame["y_pred"] = frame["y_pred"].apply(lambda value: encoder.decode(int(value)))
        subject_frames[name] = frame
    return metrics, subject_frames


def _selection_objective(metrics: Dict[str, Dict[str, float]]) -> float:
    return float(np.mean([
        metrics["FMA_UE"]["mae"] / 20.0,
        metrics["BI"]["mae"] / 100.0,
        1.0 - metrics["hand_tone"]["macro_f1"],
        1.0 - metrics["hand_function"]["macro_f1"],
    ]))


def _run_fold(args: argparse.Namespace, fold_info: Dict[str, object], df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    fold = int(fold_info["fold"])
    outer_train, outer_test = _fold_subjects(fold_info)
    _validate_subject_split(df, outer_train, outer_test, fold)
    fit_ids, inner_ids = _split_inner(outer_train, args.inner_val_fraction, args.seed + fold)
    print(f"[mtl] fold={fold} fit={len(fit_ids)} inner_val={len(inner_ids)} outer_test={len(outer_test)}")
    cfg = WBYDTWConfig(output_length=args.seq_len, dtw_length=args.dtw_length, band_radius=0.15, alpha=0.7, beta=0.3)
    make_store = lambda ids: MultiTaskStore(df[df.subject_id.astype(str).isin(ids)].copy(), args.root, args, cfg)
    fit_store, inner_store, test_store = make_store(fit_ids), make_store(inner_ids), make_store(outer_test)
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": args.num_workers > 0 and torch.cuda.is_available()}
    train_ds = MultiTaskBagDataset(fit_store, args.bag_size, args.train_bags, args.seed + fold, False)
    if args.minority_oversample:
        sampler = WeightedRandomSampler(torch.as_tensor(train_ds.subject_sampling_weights(), dtype=torch.double), len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    inner_loader = DataLoader(MultiTaskBagDataset(inner_store, args.eval_bag_size, args.eval_bags, args.seed + 100 + fold, True), batch_size=args.batch_size, **loader_kwargs)
    test_loader = DataLoader(MultiTaskBagDataset(test_store, args.eval_bag_size, args.eval_bags, args.seed + 200 + fold, True), batch_size=args.batch_size, **loader_kwargs)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MultiTaskClinicalPredictionModel(eeg_channels=args.eeg_channels, emg_channels=args.emg_channels, imu_channels=args.imu_channels, f=args.feature, te=args.task_emb, p=args.dropout, enabled_modalities=tuple(args.modalities.split("+"))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.pretrain_epochs:
        model.train()
        for epoch in range(args.pretrain_epochs):
            losses = []
            for source_batch in train_loader:
                batch = _device_batch(source_batch, device)
                loss = _pretraining_loss(model, batch, args.mask_ratio, args.consistency_weight)
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); optimizer.step()
                losses.append(float(loss.item()))
            print(f"[mtl] fold={fold} pretrain {epoch + 1}/{args.pretrain_epochs} loss={np.mean(losses):.4f}")
    best_state, best_objective, stale = None, float("inf"), 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for source_batch in train_loader:
            batch = _device_batch(source_batch, device)
            out = model(batch["eeg"], batch["emg"], batch["imu"], batch["task"], batch["trial"])
            loss = _supervised_loss(out, batch["targets"], args.ce_weight)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); optimizer.step()
            losses.append(float(loss.item()))
        inner_metrics, _ = _evaluate(model, inner_loader, device)
        objective = _selection_objective(inner_metrics)
        improved = objective < best_objective - 1e-12
        if improved:
            best_state, best_objective, stale = deepcopy(model.state_dict()), objective, 0
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "inner_objective": objective, "is_best": improved})
        print(f"[mtl] fold={fold} epoch={epoch} loss={np.mean(losses):.4f} inner_objective={objective:.4f}{' *' if improved else ''}")
        if stale >= args.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    metrics, frames = _evaluate(model, test_loader, device)
    fold_root = args.out_dir / f"fold{fold}"
    fold_root.mkdir(parents=True, exist_ok=True)
    torch.save({"fold": fold, "outer_train_subjects": outer_train, "inner_fit_subjects": fit_ids, "inner_val_subjects": inner_ids, "outer_test_subjects": outer_test, "state_dict": best_state, "metrics": metrics}, fold_root / "multitask.pth")
    pd.DataFrame(history).to_csv(fold_root / "training_history.csv", index=False)
    for task, frame in frames.items():
        logs = args.out_dir / task / f"{task}_fold{fold}_logs"; logs.mkdir(parents=True, exist_ok=True)
        frame.to_csv(logs / "val_predictions.csv", index=False)
        (logs / "metrics.json").write_text(json.dumps(metrics[task], ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--manifest", type=Path, default=Path("data/samples_manifest_real_archive.csv"))
    ap.add_argument("--split-json", type=Path, default=Path("splits/split_patient_3fold.json"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("results/mtl/tri_ssl"))
    ap.add_argument("--alignment-mode", default="adk")
    ap.add_argument("--modalities", default="eeg+emg+imu")
    ap.add_argument("--seq-len", type=int, default=256); ap.add_argument("--dtw-length", type=int, default=32)
    ap.add_argument("--eeg-channels", type=int, default=30); ap.add_argument("--emg-channels", type=int, default=4); ap.add_argument("--imu-channels", type=int, default=24); ap.add_argument("--eeg-fs", type=float, default=EEG_FS_DEFAULT)
    ap.add_argument("--no-preprocess", action="store_true"); ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--feature", type=int, default=64); ap.add_argument("--task-emb", type=int, default=12); ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--bag-size", type=int, default=4); ap.add_argument("--eval-bag-size", type=int, default=4); ap.add_argument("--train-bags", type=int, default=40); ap.add_argument("--eval-bags", type=int, default=60); ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=120); ap.add_argument("--patience", type=int, default=25); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--weight-decay", type=float, default=5e-4); ap.add_argument("--ce-weight", type=float, default=0.2); ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--pretrain-epochs", type=int, default=10); ap.add_argument("--mask-ratio", type=float, default=0.15); ap.add_argument("--consistency-weight", type=float, default=0.25)
    ap.add_argument("--inner-val-fraction", type=float, default=0.2); ap.add_argument("--minority-oversample", action="store_true"); ap.add_argument("--num-workers", type=int, default=4); ap.add_argument("--seed", type=int, default=2024); ap.add_argument("--device", default="")
    args = ap.parse_args(); args.root = args.root.resolve(); args.out_dir = args.out_dir if args.out_dir.is_absolute() else args.root / args.out_dir
    manifest = args.manifest if args.manifest.is_absolute() else args.root / args.manifest
    split = args.split_json if args.split_json.is_absolute() else args.root / args.split_json
    df = pd.read_csv(manifest, dtype={"subject_id": str, "trial_id": str}).sort_values(["subject_id", "task_id", "trial_number"], key=lambda series: series.astype(int))
    _, split_data = _load_fold_file(args.root, split)
    seed_all(args.seed)
    rows = []
    for fold_info in _select_folds(split_data, args.fold):
        metrics = _run_fold(args, fold_info, df)
        for task, values in metrics.items(): rows.append({"fold": int(fold_info["fold"]), "task": task, **values})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "multitask_outer_test_metrics.csv", index=False)
    (args.out_dir / "config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[mtl] wrote patient-independent outer-test metrics to {args.out_dir}")


if __name__ == "__main__":
    main()
