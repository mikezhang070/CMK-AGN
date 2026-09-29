"""Inference for the 4 clinical tasks.

Single task:
    python predict.py --task FMA_UE --checkpoint checkpoints/FMA_UE_model.pth

All tasks (loads each of the 4 checkpoints in turn and merges results):
    python predict.py --all-tasks

Output is written as JSON / CSV. Each row in the JSON is one patient with the
predicted labels for every task that was run.

Postprocess:
    * regression predictions are clipped to the task's [score_min, score_max].
    * classification predictions are decoded back to the original class labels
      (so hand_tone outputs come out as "0"/"1"/"1+"/... not class indices).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alignment.wby_dtw import WBYDTWConfig  # noqa: E402
from bjh_io.bjh_loader import EEG_FS_DEFAULT  # noqa: E402
from clinical_model import ClinicalPredictionModel  # noqa: E402
from task_config import (  # noqa: E402
    ALL_TASK_NAMES,
    LabelEncoder,
    TaskSpec,
    clip_regression,
    get_encoder,
    get_task,
)
from train import BagDS, TaskStore, _corn_decode  # noqa: E402


def _load_checkpoint(ckpt_path: Path) -> Tuple[Dict, ClinicalPredictionModel, str]:
    """Load a single-task checkpoint and rebuild its model.

    Returns the raw payload, the rebuilt model, and the classification
    ``head_kind`` ("ce" or "corn") so the caller can decode the right output
    shape. ``head_kind`` is always "ce" for regression tasks.
    """
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = payload.get("model_config", {})
    # Fall back to top-level payload fields for older checkpoints that didn't
    # store score_min/score_max in model_config.
    score_min = float(cfg.get("score_min", payload.get("score_min", 0.0)))
    score_max = float(cfg.get("score_max", payload.get("score_max", 0.0)))
    bin_step = float(cfg.get("bin_step", 0.0))
    head_kind = str(cfg.get("head_kind", "ce"))
    model = ClinicalPredictionModel(
        task_type=payload["task_type"],
        num_classes=payload.get("num_classes") or None,
        eeg_channels=cfg.get("eeg_channels", 30),
        emg_channels=cfg.get("emg_channels", 4),
        imu_channels=cfg.get("imu_channels", 24),
        f=cfg.get("feature", 64),
        te=cfg.get("task_emb", 12),
        p=cfg.get("dropout", 0.35),
        score_min=score_min,
        score_max=score_max,
        bin_step=bin_step,
        head_kind=head_kind,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return payload, model, head_kind


def _default_checkpoint(args: argparse.Namespace, spec: TaskSpec) -> Path:
    """Resolve a task's default checkpoint path.

    Primary layout matches train.py's ``<out-dir>/<task>_fold<n>.pth`` convention:
        <root>/<results-dir>/<task>/baseline/<task>_fold<fold>.pth
    Falls back to the legacy ``<root>/checkpoints/<task>_model.pth`` only if the
    fold checkpoint is absent.
    """
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = args.root / results_dir
    fold_ckpt = results_dir / spec.name / "baseline" / f"{spec.name}_fold{int(args.fold)}.pth"
    if fold_ckpt.exists():
        return fold_ckpt
    return args.root / spec.checkpoint


def _predict_with_model(
    model: ClinicalPredictionModel,
    spec: TaskSpec,
    encoder: Optional[LabelEncoder],
    df: pd.DataFrame,
    root: Path,
    args: argparse.Namespace,
    device: torch.device,
    head_kind: str = "ce",
) -> Dict[str, object]:
    """Return {subject_id: predicted_label} for every subject in df.

    Mirrors ``train.evaluate``: regression aggregates by per-subject median;
    classification averages per-class probabilities across bags and argmaxes the
    mean (CORN heads are decoded via ``_corn_decode`` first).
    """
    cfg = WBYDTWConfig(
        output_length=args.seq_len,
        dtw_length=args.dtw_length,
        band_radius=0.15,
        alpha=0.7,
        beta=0.3,
    )
    df = df.sort_values(["subject_id", "task_id", "trial_number"], key=lambda c: c.astype(int))
    store = TaskStore(df, root, spec, encoder, args.seq_len, cfg, args.alignment_mode,
                      args.eeg_fs, not args.no_preprocess,
                      getattr(args, "cache_dir", None))
    loader = DataLoader(
        # train.py builds its eval BagDS with seed=args.seed+7; match it so that
        # deterministic bag sampling reproduces the validation predictions.
        BagDS(store, args.eval_bag_size, args.eval_bags, args.seed + 7, deterministic=True),
        batch_size=args.batch_size, shuffle=False,
    )
    rows: List[Dict[str, object]] = []
    model.to(device)
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["eeg"].to(device), batch["emg"].to(device), batch["imu"].to(device),
                batch["task"].to(device), batch["trial"].to(device),
            )
            if spec.task_type == "regression":
                if isinstance(out, dict):
                    out = out["pred"]
                preds = out.cpu().numpy().astype(float)
                for sid, p in zip(batch["sid"], preds):
                    rows.append({"subject_id": str(sid), "raw": float(p)})
            else:
                if head_kind == "corn":
                    _, probs_t = _corn_decode(out)
                    probs = probs_t.cpu().numpy()
                else:
                    probs = torch.softmax(out, dim=1).cpu().numpy()  # [B, C]
                for i, sid in enumerate(batch["sid"]):
                    row: Dict[str, object] = {"subject_id": str(sid)}
                    for c in range(spec.num_classes):
                        row[f"prob_c{c}"] = float(probs[i, c])
                    rows.append(row)

    bf = pd.DataFrame(rows)
    out_map: Dict[str, object] = {}
    if spec.task_type == "regression":
        agg = bf.groupby("subject_id", as_index=False).agg(raw=("raw", "median"))
        for _, r in agg.iterrows():
            out_map[str(r["subject_id"])] = clip_regression(spec.name, float(r["raw"]))
    else:
        # Mean class probability across bags per subject, then argmax.
        assert encoder is not None
        prob_cols = [f"prob_c{c}" for c in range(spec.num_classes)]
        agg = bf.groupby("subject_id", as_index=False)[prob_cols].mean()
        pred_idx = agg[prob_cols].to_numpy().argmax(axis=1).astype(int)
        for sid, p in zip(agg["subject_id"].astype(str), pred_idx):
            out_map[str(sid)] = encoder.decode(int(p))
    return out_map


def run(args: argparse.Namespace) -> Dict[str, Dict[str, object]]:
    args.root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else args.root / args.manifest
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Check data/samples_manifest_real_archive.csv."
        )
    df = pd.read_csv(manifest_path, dtype={"subject_id": str, "trial_id": str})

    if args.subjects:
        wanted = set(str(s).strip() for s in args.subjects.split(","))
        df = df[df["subject_id"].astype(str).isin(wanted)].copy()
        if df.empty:
            raise ValueError(f"No manifest rows match --subjects={args.subjects!r}")

    if args.all_tasks:
        task_names = list(ALL_TASK_NAMES)
    else:
        task_names = [args.task]

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    per_task_predictions: Dict[str, Dict[str, object]] = {}

    for tname in task_names:
        spec = get_task(tname)
        if args.checkpoint and not args.all_tasks:
            ckpt = Path(args.checkpoint)
        else:
            ckpt = _default_checkpoint(args, spec)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint for task {tname} not found at {ckpt}. "
                f"Train it first with `python train.py --task {tname}`."
            )
        payload, model, head_kind = _load_checkpoint(ckpt)
        encoder = get_encoder(tname) if spec.task_type == "classification" else None
        print(f"  → predicting task {tname} (ckpt={ckpt.name}, head={head_kind})")
        per_task_predictions[tname] = _predict_with_model(
            model, spec, encoder, df, args.root, args, device, head_kind
        )

    # Merge per-task predictions into a per-patient view.
    all_subjects = sorted(set().union(*[set(d.keys()) for d in per_task_predictions.values()]),
                          key=lambda x: int(x) if x.isdigit() else x)
    merged: Dict[str, Dict[str, object]] = {}
    for sid in all_subjects:
        rec: Dict[str, object] = {"patient_id": sid}
        for tname in task_names:
            value = per_task_predictions[tname].get(sid)
            rec[f"{tname}_pred"] = value
        merged[sid] = rec

    return merged


def write_outputs(merged: Dict[str, Dict[str, object]], output: Path) -> None:
    output = output if output.is_absolute() else Path.cwd() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        rows = list(merged.values())
        pd.DataFrame(rows).to_csv(output, index=False)
    else:
        output.write_text(json.dumps(list(merged.values()), ensure_ascii=False, indent=2))
    print(f"  predictions written to {output}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Single / all-task clinical predictor.")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--task", help="One of FMA_UE, BI, hand_tone, hand_function")
    grp.add_argument("--all-tasks", action="store_true",
                     help="Run all 4 tasks; uses each task's default checkpoint path.")
    ap.add_argument("--checkpoint", default="",
                    help="Override checkpoint path (only when --task is used).")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--results-dir", type=Path, default=Path("results"),
                    help="Root results dir holding <task>/baseline/<task>_fold<n>.pth. "
                         "Resolved under --root if relative. Used to find default checkpoints.")
    ap.add_argument("--fold", type=int, default=1,
                    help="Fold whose checkpoint to load by default (1-3 available).")
    ap.add_argument("--manifest", type=Path, default=Path("data/samples_manifest_real_archive.csv"))
    ap.add_argument("--subjects", default="", help="Optional comma list of subject_ids to predict for.")
    ap.add_argument("--output", type=Path, default=Path("predictions.json"))
    # Inference loader knobs (must match train.py's data pipeline / eval defaults).
    ap.add_argument("--alignment-mode", default="adk")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--dtw-length", type=int, default=32)
    ap.add_argument("--eeg-fs", type=float, default=EEG_FS_DEFAULT)
    ap.add_argument("--no-preprocess", action="store_true")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="Directory for disk-cached aligned trial arrays (.npz), shared with train.py.")
    ap.add_argument("--eval-bag-size", type=int, default=4)
    ap.add_argument("--eval-bags", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    merged = run(args)
    print(f"\nMerged predictions for {len(merged)} subject(s):")
    for sid, rec in list(merged.items())[:5]:
        print(f"  {sid}: {rec}")
    if len(merged) > 5:
        print(f"  ... ({len(merged) - 5} more)")

    write_outputs(merged, args.output)


if __name__ == "__main__":
    main()
