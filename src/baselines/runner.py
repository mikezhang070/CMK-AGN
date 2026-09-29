"""Fold-loop orchestrator for baseline experiments.

One ``run_folds(args, builder)`` entry point drives both the ML and DL
branches. Output layout, metric names, file naming, and CSV schema are kept
byte-identical to :mod:`train` so downstream aggregation scripts work
without modification.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from task_config import LabelEncoder, TaskSpec, get_encoder, get_task

from baselines._train_shim import (
    BagDS,
    _filter_classification_manifest,
    _fold_subjects,
    _format_subjects,
    _load_fold_file,
    _select_folds,
    _subject_sort_key,
    _validate_subject_split,
    _warn_rare_classes,
    evaluate as evaluate_dl,
    rounded_acc_key,
    seed_all,
    tolerance_acc_key,
)
from baselines.data import build_stores, stores_to_xy
from baselines.dl_models._common import BaselineDLModule, train_dl_baseline
from baselines.metrics import (
    classification_metrics,
    confusion_matrix as confusion_matrix_fn,
    regression_metrics,
)
from baselines.ml_wrappers import MLBaseline


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def apply_smoke_overrides(args: argparse.Namespace) -> None:
    """In-place: shrink everything so each fold finishes in seconds.

    Also enables ``smoke_filter_existing`` which auto-drops manifest rows
    whose EEG/EMG files are missing on the current machine — lets local
    verification work even on partial data dumps.
    """
    args.epochs = 2
    args.train_bags = 2
    args.eval_bags = 2
    args.batch_size = 2
    args.bag_size = 2
    args.eval_bag_size = 2
    args.smoke_max_train = 4
    args.smoke_max_val = 2
    args.smoke_filter_existing = True


def _filter_manifest_to_existing(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Keep only manifest rows whose EEG + EMG files exist on disk."""
    def _exists(rel: str) -> bool:
        p = Path(rel)
        full = p if p.is_absolute() else root / p
        return full.exists()

    mask = df.apply(lambda r: _exists(str(r["eeg_path"])) and _exists(str(r["emg_path"])), axis=1)
    return df[mask].reset_index(drop=True)


def _get_out_dir(args: argparse.Namespace, spec: TaskSpec, model_name: str) -> Path:
    if getattr(args, "out_dir", None) is not None:
        return Path(args.out_dir).resolve()
    return (args.root / "results" / "baselines" / spec.name / model_name).resolve()


def _fold_artifact_paths(out_dir: Path, model_name: str, task_name: str, fold: int) -> Tuple[Path, Path]:
    stem = f"{model_name}_fold{int(fold)}"
    logs_dir = out_dir / f"{stem}_logs"
    return out_dir, logs_dir


def _write_regression_extras(sf: pd.DataFrame, logs_dir: Path) -> None:
    """Bland-Altman + (optional) calibration data — mirrors train.py."""
    if sf.empty:
        return
    ba = pd.DataFrame({
        "subject_id": sf["subject_id"],
        "mean": (sf["y_true"] + sf["y_pred"]) / 2,
        "diff": sf["y_pred"] - sf["y_true"],
    })
    ba.to_csv(logs_dir / "bland_altman_data.csv", index=False)
    n_bins = min(5, max(2, len(sf) // 3))
    try:
        s = sf.sort_values("y_true").copy()
        s["cal_bin"] = pd.qcut(s["y_true"], q=n_bins, duplicates="drop", labels=False)
        s["abs_error"] = (s["y_pred"] - s["y_true"]).abs()
        cal = s.groupby("cal_bin", as_index=False).agg(
            mean_true=("y_true", "mean"),
            mean_pred=("y_pred", "mean"),
            count=("y_true", "count"),
            mean_abs_error=("abs_error", "mean"),
        )
        cal.to_csv(logs_dir / "calibration_data.csv", index=False)
    except Exception:
        pass


def _write_classification_extras(
    final_m: Dict[str, Any], spec: TaskSpec, encoder: Optional[LabelEncoder], logs_dir: Path
) -> None:
    per_class: Dict[str, Any] = {}
    for c in range(spec.num_classes):
        label = encoder.decode(c) if encoder is not None else str(c)
        per_class[str(label)] = {
            k: final_m.get(f"{k}_c{c}", float("nan"))
            for k in ("precision", "recall", "f1", "support")
        }
    (logs_dir / "per_class_metrics.json").write_text(
        json.dumps(per_class, ensure_ascii=False, indent=2)
    )

    cls_labels = [str(encoder.decode(c)) if encoder is not None else str(c)
                  for c in range(spec.num_classes)]
    cm_arr = np.array(final_m.get("confusion_matrix", []))
    if cm_arr.ndim == 2:
        cm_df = pd.DataFrame(
            cm_arr,
            index=[f"true_{l}" for l in cls_labels],
            columns=[f"pred_{l}" for l in cls_labels],
        )
        cm_df.to_csv(logs_dir / "confusion_matrix.csv")


# --------------------------------------------------------------------------- #
# ML branch                                                                   #
# --------------------------------------------------------------------------- #
def _run_ml_fold(
    args: argparse.Namespace,
    model: MLBaseline,
    spec: TaskSpec,
    encoder: Optional[LabelEncoder],
    tr_store,
    ev_store,
    out_dir: Path,
    logs_dir: Path,
    fold: int,
    rounded_tol: float,
    score_tolerance: float,
) -> Dict[str, float]:
    X_tr, y_tr, _ = stores_to_xy(tr_store, args.eeg_fs)
    X_va, y_va, sids_va = stores_to_xy(ev_store, args.eeg_fs)

    print(f"  ML features: X_train={X_tr.shape}  X_val={X_va.shape}")
    model.fit(X_tr, y_tr)
    pred = model.predict(X_va)
    proba = model.predict_proba(X_va) if spec.task_type == "classification" else None

    # Build val_predictions.csv in the same schema as train.py's evaluate().
    rows: List[Dict[str, Any]] = []
    if spec.task_type == "regression":
        for sid, y, p in zip(sids_va, y_va, pred):
            rows.append({"subject_id": str(sid), "y_true": float(y), "y_pred": float(p)})
        sf = pd.DataFrame(rows)
        sf["error"] = sf["y_pred"] - sf["y_true"]
        sf["abs_error"] = sf["error"].abs()
        sf["y_pred_rounded"] = sf["y_pred"].round().astype(float)
        sf["within_rounded_tol"] = (sf["abs_error"] <= rounded_tol + 0.5).astype(int)
        sf["within_score_tol"] = (sf["abs_error"] <= score_tolerance).astype(int)
        metrics = regression_metrics(
            sf["y_true"].to_numpy(), sf["y_pred"].to_numpy(),
            rounded_tol, score_tolerance,
            float(spec.score_max - spec.score_min),
        )
    else:
        if proba is None:
            proba = np.zeros((len(sids_va), spec.num_classes), dtype=np.float64)
            proba[np.arange(len(sids_va)), pred.astype(int)] = 1.0
        for i, (sid, y, p) in enumerate(zip(sids_va, y_va.astype(int), pred.astype(int))):
            row: Dict[str, Any] = {"subject_id": str(sid), "y_true": int(y), "y_pred": int(p)}
            for c in range(spec.num_classes):
                row[f"prob_c{c}"] = float(proba[i, c])
            rows.append(row)
        sf = pd.DataFrame(rows)
        sf["is_correct"] = (sf["y_true"] == sf["y_pred"]).astype(int)
        y_true_int = sf["y_true"].to_numpy().astype(int)
        y_pred_int = sf["y_pred"].to_numpy().astype(int)
        metrics = classification_metrics(y_true_int, y_pred_int, spec.num_classes)
        cm = confusion_matrix_fn(y_true_int, y_pred_int, spec.num_classes)
        if encoder is not None:
            sf["y_true"] = sf["y_true"].apply(lambda v: encoder.decode(int(v)))
            sf["y_pred"] = sf["y_pred"].apply(lambda v: encoder.decode(int(v)))
        metrics["confusion_matrix"] = cm.tolist()

    # Persist artifacts.
    logs_dir.mkdir(parents=True, exist_ok=True)
    sf.to_csv(logs_dir / "val_predictions.csv", index=False)
    (logs_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    if spec.task_type == "regression":
        _write_regression_extras(sf, logs_dir)
    else:
        _write_classification_extras(metrics, spec, encoder, logs_dir)

    ckpt = out_dir / f"{model.name}_fold{int(fold)}.pkl"
    model.save(ckpt, extra={"fold": int(fold), "metrics":
                            {k: v for k, v in metrics.items() if k != "confusion_matrix"}})
    print(f"  ML checkpoint saved → {ckpt}")
    return metrics


# --------------------------------------------------------------------------- #
# DL branch                                                                   #
# --------------------------------------------------------------------------- #
def _run_dl_fold(
    args: argparse.Namespace,
    model: BaselineDLModule,
    spec: TaskSpec,
    encoder: Optional[LabelEncoder],
    tr_store,
    ev_store,
    out_dir: Path,
    logs_dir: Path,
    fold: int,
    rounded_tol: float,
    score_tolerance: float,
) -> Dict[str, float]:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)

    tr_ds = BagDS(tr_store, args.bag_size, args.train_bags, args.seed, deterministic=False)
    ev_ds = BagDS(ev_store, args.eval_bag_size, args.eval_bags, args.seed + 7, deterministic=True)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    ev_loader = DataLoader(ev_ds, batch_size=args.batch_size, shuffle=False)

    print(f"  DL bags: train={len(tr_ds)} val={len(ev_ds)}  device={device}")
    final_m, history, best_state = train_dl_baseline(
        model, tr_loader, ev_loader, spec, encoder, device,
        epochs=int(args.epochs), lr=float(args.lr), grad_clip=1.0,
        rounded_tol=rounded_tol, score_tolerance=score_tolerance,
        evaluate_fn=evaluate_dl,
    )

    # Re-run evaluate once to also capture the per-subject prediction frame.
    from torch.nn import SmoothL1Loss, CrossEntropyLoss
    loss_fn = SmoothL1Loss(beta=1.0) if spec.task_type == "regression" else CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
    _, sf = evaluate_dl(model, ev_loader, device, spec, encoder,
                        rounded_tol, score_tolerance, loss_fn=loss_fn)

    logs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(logs_dir / "training_history.csv", index=False)
    sf.to_csv(logs_dir / "val_predictions.csv", index=False)
    (logs_dir / "metrics.json").write_text(json.dumps(final_m, ensure_ascii=False, indent=2))
    if spec.task_type == "regression":
        _write_regression_extras(sf, logs_dir)
    else:
        _write_classification_extras(final_m, spec, encoder, logs_dir)

    ckpt = out_dir / f"{model.name}_fold{int(fold)}.pth"
    torch.save({
        "name": model.name,
        "task": spec.name,
        "fold": int(fold),
        "task_type": spec.task_type,
        "num_classes": spec.num_classes,
        "score_min": spec.score_min,
        "score_max": spec.score_max,
        "classes": list(spec.classes),
        "state_dict": best_state,
        "model_config": {
            "feature_dim": int(getattr(model, "feature_dim", 0)),
            "eeg_channels": args.eeg_channels,
            "emg_channels": args.emg_channels,
            "imu_channels": args.imu_channels,
            "seq_len": args.seq_len,
        },
        "metrics": {k: v for k, v in final_m.items() if k != "confusion_matrix"},
    }, ckpt)
    print(f"  DL checkpoint saved → {ckpt}")
    return final_m


# --------------------------------------------------------------------------- #
# Top-level fold driver                                                       #
# --------------------------------------------------------------------------- #
def run_folds(args: argparse.Namespace, builder: Callable[[TaskSpec, argparse.Namespace], Any]) -> List[Dict[str, Any]]:
    spec = get_task(args.task)
    encoder = get_encoder(args.task) if spec.task_type == "classification" else None

    args.root = Path(args.root).resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else args.root / args.manifest
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path, dtype={"subject_id": str, "trial_id": str})
    if spec.manifest_col not in df.columns:
        raise ValueError(
            f"Manifest is missing column {spec.manifest_col!r} required for task {spec.name}."
        )
    df = df.sort_values(["subject_id", "task_id", "trial_number"], key=lambda c: c.astype(int))

    if getattr(args, "smoke_filter_existing", False):
        before = len(df)
        df = _filter_manifest_to_existing(df, args.root)
        print(f"[smoke-test] filtered manifest to existing files: {before} -> {len(df)} rows")
        if df.empty:
            raise FileNotFoundError(
                "No manifest rows have BJH data on disk. Check --root and BJH/ contents."
            )

    split_path, split_data = _load_fold_file(args.root, args.split_json)
    selected_folds = _select_folds(split_data, args.fold)

    model_name = args.model
    out_dir = _get_out_dir(args, spec, model_name)
    print("=" * 60)
    print(f"Baseline:       {model_name}")
    print(f"Training task:  {args.task}  ({spec.task_type})")
    print(f"Split file:     {split_path}")
    print(f"Output dir:     {out_dir}")
    print(f"Folds to train: {[int(f['fold']) for f in selected_folds]}")
    print("=" * 60)

    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []

    for fold_info in selected_folds:
        fold = int(fold_info["fold"])
        train_subj, val_subj = _fold_subjects(fold_info)

        if getattr(args, "smoke_filter_existing", False):
            available = set(df["subject_id"].astype(str).unique().tolist())
            train_subj = [s for s in train_subj if s in available]
            val_subj = [s for s in val_subj if s in available]
            if not train_subj or not val_subj:
                print(f"[smoke-test] fold {fold}: no available subjects on disk, skipping.")
                continue

        if getattr(args, "smoke_max_train", None):
            train_subj = sorted(train_subj, key=_subject_sort_key)[: int(args.smoke_max_train)]
            val_subj = sorted(val_subj, key=_subject_sort_key)[: int(args.smoke_max_val)]

        _validate_subject_split(df, train_subj, val_subj, fold)
        train_subj = sorted(train_subj, key=_subject_sort_key)
        val_subj = sorted(val_subj, key=_subject_sort_key)
        df_train = df[df["subject_id"].astype(str).isin(train_subj)].copy()
        df_val = df[df["subject_id"].astype(str).isin(val_subj)].copy()
        if spec.task_type == "classification":
            df_train = _filter_classification_manifest(df_train, spec, "train")
            df_val = _filter_classification_manifest(df_val, spec, "val")
            _warn_rare_classes(df_train, spec, "train", min_subjects=2)
            _warn_rare_classes(df_val, spec, "val", min_subjects=2)

        print("")
        print("-" * 60)
        print(f"Fold {fold}  (train_subjects={len(train_subj)}  val_subjects={len(val_subj)})")
        print(f"  Train: {_format_subjects(train_subj)}")
        print(f"  Val:   {_format_subjects(val_subj)}")
        print(f"  Train rows: {len(df_train)}  Val rows: {len(df_val)}")
        print("-" * 60)

        seed_all(int(args.seed))

        tr_store, ev_store = build_stores(
            df_train, df_val, args.root, spec, encoder,
            args.seq_len, args.dtw_length, args.alignment_mode,
            args.eeg_fs, not args.no_preprocess,
            getattr(args, "cache_dir", None),
        )

        out_dir_for_fold, logs_dir = _fold_artifact_paths(out_dir, model_name, spec.name, fold)
        out_dir_for_fold.mkdir(parents=True, exist_ok=True)

        rounded_tol = spec.rounded_tol
        score_tolerance = getattr(args, "tolerance", None) or spec.score_tolerance

        model = builder(spec, args)

        if getattr(model, "family", None) == "ml":
            metrics = _run_ml_fold(
                args, model, spec, encoder, tr_store, ev_store,
                out_dir_for_fold, logs_dir, fold, rounded_tol, score_tolerance,
            )
        elif getattr(model, "family", None) == "dl":
            metrics = _run_dl_fold(
                args, model, spec, encoder, tr_store, ev_store,
                out_dir_for_fold, logs_dir, fold, rounded_tol, score_tolerance,
            )
        else:
            raise TypeError(f"Builder for {model_name!r} returned a model with no 'family'.")

        row = {
            "fold": fold,
            **{k: v for k, v in metrics.items() if not isinstance(v, (list, dict))},
        }
        all_rows.append(row)

    if all_rows:
        n_folds = len(selected_folds)
        summary_stem = f"{spec.name}_{n_folds}fold_summary"
        summary_csv = out_dir / f"{summary_stem}.csv"
        summary_json = out_dir / f"{summary_stem}.json"
        pd.DataFrame(all_rows).sort_values("fold").to_csv(summary_csv, index=False)
        summary_json.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")

        fold_keys = {"fold", "train_subjects", "val_subjects"}
        exp_cfg = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k not in fold_keys
        }
        exp_cfg["folds_trained"] = [int(f["fold"]) for f in selected_folds]
        (out_dir / "config.json").write_text(json.dumps(exp_cfg, ensure_ascii=False, indent=2))

        print("")
        print("=" * 60)
        print(f"Finished {n_folds} fold(s) for baseline '{model_name}' on task '{spec.name}'.")
        print(f"Summary saved → {summary_csv}")
        print(f"Summary saved → {summary_json}")
        print(f"Config  saved → {out_dir / 'config.json'}")
        print("=" * 60)

    return all_rows
