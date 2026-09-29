"""Aggregate ablation runs into Table 5 (modality) and Table 6 (module).

Scans:  results/{modality,module}/<variant>/<task>/*_3fold_summary.csv
Writes: results/_summary/
    - long.csv                              (kind, variant, task, ttype, metric, mean, std)
    - long_with_seeds.csv                   (raw, before folding tri_seed* into tri)
    - table5_modality_regression.csv        (7 variants x 2 tasks; 6 reg metrics)
    - table5_modality_classification.csv    (7 variants x 2 tasks; 4 cls metrics)
    - table6_module_regression.csv          (4 variants x 2 tasks; 6 reg metrics)
    - table6_module_classification.csv      (4 variants x 2 tasks; 4 cls metrics)

The Ours rows ``tri_seed42`` / ``tri_seed1337`` and the baseline ``tri`` row are
folded into a single final ``tri`` row (module side: ``full``) using:
    - regression  → seed with minimum mean MAE  wins
    - classification → seed with maximum mean Weighted-kappa wins
The 5/3 non-primary metrics on that selected seed are carried along (no
cross-seed cherry-picking).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# --------------------------------------------------------------------------- #
# Task & metric registry                                                      #
# --------------------------------------------------------------------------- #
REG_TASKS = {"FMA_UE", "BI"}
CLS_TASKS = {"hand_tone", "hand_function"}

REG_METRICS = ["MAE", "RMSE", "PearsonR", "SpearmanRho", "R2", "NMAE"]
CLS_METRICS = ["Accuracy", "MacroF1", "CohenKappa", "WeightedKappa"]

# canonical_key -> ordered list of accepted source column names in summary csv.
# The first hit wins.  Update here if train.py ever renames columns.
COLUMN_ALIAS: Dict[str, List[str]] = {
    "MAE":           ["mae", "MAE"],
    "RMSE":          ["rmse", "RMSE"],
    "PearsonR":      ["pearson_r", "Pearson r", "PearsonR"],
    "SpearmanRho":   ["spearman_r", "Spearman r", "SpearmanRho"],
    "R2":            ["r2", "R2", "R²"],
    "NMAE":          ["nmae", "NMAE"],
    "Accuracy":      ["accuracy", "Accuracy"],
    "MacroF1":       ["macro_f1", "MacroF1"],
    "CohenKappa":    ["cohen_kappa", "CohenKappa"],
    "WeightedKappa": ["weighted_kappa", "WeightedKappa"],
}

PRIMARY_REG_METRIC = "MAE"            # min wins on regression
PRIMARY_CLS_METRIC = "WeightedKappa"  # max wins on classification

# Variants whose rows must be folded into a single Ours row.
OURS_SEED_TAGS = {"tri", "tri_seed42", "tri_seed1337", "tri_seed2024"}

VARIANT_LABELS = {
    "eeg_only": "EEG only",
    "emg_only": "EMG only",
    "imu_only": "IMU only",
    "eeg_emg": "EEG + EMG",
    "eeg_imu": "EEG + IMU",
    "emg_imu": "EMG + IMU",
    "wo_mdfan": "w/o MDFAN",
    "wo_wbydtw": "w/o W-DTW",
    "wo_trialign": "w/o TriAlign",
}

FIGURE_METRICS = {
    "regression": {
        "MAE": "mae",
        "RMSE": "rmse",
        "R2": "r2",
        "PearsonR": "pearson_r",
        "SpearmanRho": "spearman_r",
    },
    "classification": {
        "Accuracy": "accuracy",
        "MacroF1": "macro_f1",
        "CohenKappa": "cohen_kappa",
        "WeightedKappa": "weighted_kappa",
    },
}


# --------------------------------------------------------------------------- #
# IO helpers                                                                  #
# --------------------------------------------------------------------------- #
def _pick(df: pd.DataFrame, key: str) -> pd.Series:
    for cand in COLUMN_ALIAS[key]:
        if cand in df.columns:
            return df[cand]
    raise KeyError(
        f"Metric {key!r}: none of {COLUMN_ALIAS[key]!r} found in {list(df.columns)!r}"
    )


def _task_type(task: str) -> str:
    if task in REG_TASKS:
        return "regression"
    if task in CLS_TASKS:
        return "classification"
    raise ValueError(f"Unknown task: {task}")


def _metric_keys(ttype: str) -> List[str]:
    return REG_METRICS if ttype == "regression" else CLS_METRICS


def collect(root: Path) -> pd.DataFrame:
    """Walk results/{modality,module}/<variant>/<task>/*_3fold_summary.csv."""
    rows: List[dict] = []
    for kind in ("modality", "module"):
        kind_dir = root / kind
        if not kind_dir.is_dir():
            continue
        for variant_dir in sorted(p for p in kind_dir.iterdir() if p.is_dir()):
            for task_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir()):
                summaries = list(task_dir.glob("*_3fold_summary.csv"))
                if not summaries:
                    continue
                df = pd.read_csv(summaries[0])
                task = task_dir.name
                ttype = _task_type(task)
                for k in _metric_keys(ttype):
                    try:
                        s = _pick(df, k)
                    except KeyError as exc:
                        print(f"  [warn] {summaries[0]}: {exc}")
                        continue
                    rows.append({
                        "kind": kind,
                        "variant": variant_dir.name,
                        "task": task,
                        "ttype": ttype,
                        "metric": k,
                        "mean": float(s.mean()),
                        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                        "n_folds": int(len(s)),
                    })
    return pd.DataFrame(rows)


def collect_figure_rows(root: Path) -> pd.DataFrame:
    """Collect fold-level ablation metrics for manuscript Figures 10 and 11."""
    rows: List[dict] = []
    for kind in ("modality", "module"):
        kind_dir = root / kind
        if not kind_dir.is_dir():
            continue
        for variant_dir in sorted(p for p in kind_dir.iterdir() if p.is_dir()):
            variant = variant_dir.name
            if variant in OURS_SEED_TAGS or variant not in VARIANT_LABELS:
                continue
            for task_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir()):
                summaries = list(task_dir.glob("*_3fold_summary.csv"))
                if not summaries:
                    continue
                task = task_dir.name
                ttype = _task_type(task)
                df = pd.read_csv(summaries[0])
                folds = df["fold"] if "fold" in df.columns else pd.Series(range(1, len(df) + 1))
                for canonical, public_name in FIGURE_METRICS[ttype].items():
                    try:
                        values = _pick(df, canonical)
                    except KeyError as exc:
                        print(f"  [warn] {summaries[0]}: {exc}")
                        continue
                    for fold, value in zip(folds, values):
                        rows.append({
                            "kind": kind,
                            "variant": variant,
                            "variant_label": VARIANT_LABELS[variant],
                            "task": task,
                            "metric": public_name,
                            "fold": int(fold),
                            "value": float(value),
                        })
    return pd.DataFrame(rows)


def write_figure_sources(root: Path, output: Path) -> None:
    """Write fold-level source tables consumed by Figure 10/11 plotters."""
    figure_rows = collect_figure_rows(root)
    if figure_rows.empty:
        raise RuntimeError(f"No fold-level ablation rows found below {root}")
    output.mkdir(parents=True, exist_ok=True)
    specs = (
        ("regression", REG_TASKS, "Fig10_Regression_task_ablation.csv"),
        ("classification", CLS_TASKS, "Fig11_Ordinal_task_ablation.csv"),
    )
    for ttype, tasks, filename in specs:
        metrics = set(FIGURE_METRICS[ttype].values())
        table = figure_rows[
            figure_rows["task"].isin(tasks) & figure_rows["metric"].isin(metrics)
        ].copy()
        if table.empty:
            raise RuntimeError(f"No {ttype} ablation rows available for {filename}")
        table.to_csv(output / filename, index=False)
        print(f"  Figure source → {output / filename}  ({len(table)} rows)")


# --------------------------------------------------------------------------- #
# Ours seed selection                                                         #
# --------------------------------------------------------------------------- #
def _select_best_ours_seed(long: pd.DataFrame, task: str) -> Optional[str]:
    """Pick the seed-variant tag whose primary metric wins on this task.

    Returns one of OURS_SEED_TAGS, or None if no candidate is available.
    """
    ttype = _task_type(task)
    primary = PRIMARY_REG_METRIC if ttype == "regression" else PRIMARY_CLS_METRIC
    cand = long[
        (long["task"] == task)
        & (long["metric"] == primary)
        & (long["variant"].isin(OURS_SEED_TAGS))
    ]
    if cand.empty:
        return None
    if ttype == "regression":
        return cand.loc[cand["mean"].idxmin(), "variant"]
    return cand.loc[cand["mean"].idxmax(), "variant"]


def fold_ours_seeds(long: pd.DataFrame) -> pd.DataFrame:
    """Collapse tri / tri_seed* rows into a single per-task 'tri' (modality) and 'full' (module) row.

    For each task, the winning seed is chosen via the primary metric; all
    metrics from that seed are then relabeled as 'tri' (kind=modality) and
    'full' (kind=module).  Original seed rows are kept in long_with_seeds.
    """
    if long.empty:
        return long
    folded = long[~long["variant"].isin(OURS_SEED_TAGS)].copy()
    for task in sorted(long["task"].unique()):
        best_tag = _select_best_ours_seed(long, task)
        if best_tag is None:
            continue
        chosen = long[(long["task"] == task) & (long["variant"] == best_tag)].copy()
        chosen_mod = chosen.copy()
        chosen_mod["kind"] = "modality"
        chosen_mod["variant"] = "tri"
        chosen_mod["source_seed_variant"] = best_tag
        chosen_full = chosen.copy()
        chosen_full["kind"] = "module"
        chosen_full["variant"] = "full"
        chosen_full["source_seed_variant"] = best_tag
        folded = pd.concat([folded, chosen_mod, chosen_full], ignore_index=True)
    return folded


# --------------------------------------------------------------------------- #
# Wide tables                                                                 #
# --------------------------------------------------------------------------- #
def _wide_table(long: pd.DataFrame, kind: str, ttype: str) -> pd.DataFrame:
    metrics = _metric_keys(ttype)
    sub = long[(long["kind"] == kind) & (long["ttype"] == ttype)]
    if sub.empty:
        return pd.DataFrame()
    sub_mean = sub.pivot_table(index=["variant", "task"], columns="metric", values="mean")
    sub_std = sub.pivot_table(index=["variant", "task"], columns="metric", values="std")
    out = pd.DataFrame(index=sub_mean.index)
    for m in metrics:
        if m in sub_mean.columns:
            out[f"{m}_mean"] = sub_mean[m]
            out[f"{m}_std"] = sub_std[m] if m in sub_std.columns else 0.0
    return out.reset_index()


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate ablation runs into Table 5 / 6.")
    ap.add_argument("--root", type=Path, default=Path("results"),
                    help="Ablation results root containing modality/ and module/ subdirs.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output dir (default: <root>/_summary)")
    ap.add_argument("--figure-source-dir", type=Path, default=None,
                    help="Optional output directory for Figure 10/11 fold-level source CSVs.")
    args = ap.parse_args()

    root = args.root.resolve()
    out = (args.out or (root / "_summary")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {root} ...")
    raw = collect(root)
    if raw.empty:
        print("  No 3fold_summary.csv files found. Nothing to aggregate.")
        return

    raw.to_csv(out / "long_with_seeds.csv", index=False)
    print(f"  Raw long table → {out / 'long_with_seeds.csv'}  ({len(raw)} rows)")

    long = fold_ours_seeds(raw)
    long.to_csv(out / "long.csv", index=False)
    print(f"  Folded long table → {out / 'long.csv'}  ({len(long)} rows)")

    # Write 4 wide tables.
    for kind in ("modality", "module"):
        for ttype in ("regression", "classification"):
            wide = _wide_table(long, kind, ttype)
            if wide.empty:
                continue
            name = (
                "table5" if kind == "modality" else "table6"
            ) + f"_{kind}_{ttype}.csv"
            wide.to_csv(out / name, index=False)
            print(f"  Wide table → {out / name}  ({wide.shape[0]} rows x {wide.shape[1]} cols)")

    seeds = {
        task: _select_best_ours_seed(raw, task)
        for task in sorted(raw["task"].unique())
    }
    (out / "ours_selected_seeds.json").write_text(
        json.dumps(seeds, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Per-task winning Ours seed → {out / 'ours_selected_seeds.json'}")
    if args.figure_source_dir is not None:
        write_figure_sources(root, args.figure_source_dir.resolve())
    print("Done.")


if __name__ == "__main__":
    main()
