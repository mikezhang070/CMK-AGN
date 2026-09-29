"""Leakage-safe patient-level metrics and paired bootstrap utilities."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd


def regression_metrics(frame: pd.DataFrame, tolerance: float) -> Dict[str, float]:
    required = {"subject_id", "fold", "y_true", "y_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing {sorted(missing)}")
    if frame.duplicated(["fold", "subject_id"]).any():
        raise ValueError("Prediction frame has duplicate outer-fold patient rows.")
    y_true = frame["y_true"].to_numpy(float)
    y_pred = frame["y_pred"].to_numpy(float)
    error = y_pred - y_true
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "n_patients": int(len(frame)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1 - np.sum(error**2) / denom) if denom > 0 else float("nan"),
        "pearson_r": float(pd.Series(y_true).corr(pd.Series(y_pred), method="pearson")),
        "spearman_r": float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")),
        "mean_signed_error": float(np.mean(error)),
        "median_signed_error": float(np.median(error)),
        "median_absolute_error": float(np.median(np.abs(error))),
        "error_sd": float(np.std(error, ddof=1)) if len(error) > 1 else 0.0,
        "tolerance_accuracy": float(np.mean(np.abs(error) <= tolerance)),
        "tolerance_hits": int(np.sum(np.abs(error) <= tolerance)),
        "tolerance_total": int(len(error)),
    }


def patient_bootstrap_ci(frame: pd.DataFrame, tolerance: float, n_boot: int = 20_000, seed: int = 2026) -> Dict[str, list[float]]:
    if len(frame) < 2:
        raise ValueError("At least two patients are required for bootstrap.")
    rng = np.random.default_rng(seed)
    statistics = ("mae", "rmse", "r2", "pearson_r", "spearman_r", "tolerance_accuracy")
    values = {name: [] for name in statistics}
    for _ in range(n_boot):
        sample = frame.iloc[rng.integers(0, len(frame), len(frame))].copy()
        metrics = regression_metrics(sample.reset_index(drop=True).assign(subject_id=lambda x: np.arange(len(x)).astype(str), fold=1), tolerance)
        for name in statistics:
            if np.isfinite(metrics[name]):
                values[name].append(metrics[name])
    return {
        name: [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
        for name, draws in values.items() if draws
    }


def fold_statistics(frame: pd.DataFrame, tolerance: float) -> tuple[list[Dict[str, float]], Dict[str, float], Dict[str, float]]:
    per_fold = []
    for fold, fold_frame in frame.groupby("fold", sort=True):
        per_fold.append({"fold": int(fold), **regression_metrics(fold_frame, tolerance)})
    table = pd.DataFrame(per_fold)
    numeric = table.drop(columns="fold")
    return per_fold, numeric.mean().to_dict(), numeric.std(ddof=1).to_dict()


def paired_bootstrap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    tolerance: float,
    n_boot: int = 20_000,
    seed: int = 2026,
) -> pd.DataFrame:
    keys = ["fold", "subject_id"]
    base = baseline[keys + ["y_true", "y_pred"]].rename(columns={"y_pred": "baseline_pred"})
    cand = candidate[keys + ["y_true", "y_pred"]].rename(columns={"y_pred": "candidate_pred"})
    merged = base.merge(cand, on=keys + ["y_true"], how="inner", validate="one_to_one")
    if len(merged) != len(baseline) or len(merged) != len(candidate):
        raise ValueError("Baseline/candidate do not have identical outer-test patients.")

    def score(frame: pd.DataFrame) -> Dict[str, float]:
        common = frame[["fold", "subject_id", "y_true"]].copy()
        a = common.assign(y_pred=frame["candidate_pred"].to_numpy())
        b = common.assign(y_pred=frame["baseline_pred"].to_numpy())
        ma = regression_metrics(a, tolerance)
        mb = regression_metrics(b, tolerance)
        return {
            "delta_mae": ma["mae"] - mb["mae"],
            "delta_r2": ma["r2"] - mb["r2"],
            "delta_tolerance_accuracy": ma["tolerance_accuracy"] - mb["tolerance_accuracy"],
        }

    point = score(merged)
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in point}
    for _ in range(n_boot):
        sample = merged.iloc[rng.integers(0, len(merged), len(merged))].copy()
        sample["subject_id"] = np.arange(len(sample)).astype(str)
        sample["fold"] = 1
        values = score(sample)
        for name, value in values.items():
            if np.isfinite(value):
                draws[name].append(value)
    rows = []
    for name, estimate in point.items():
        vector = np.asarray(draws[name], dtype=float)
        rows.append({
            "metric": name,
            "estimate_candidate_minus_baseline": estimate,
            "ci_lower": float(np.quantile(vector, 0.025)),
            "ci_upper": float(np.quantile(vector, 0.975)),
            "n_patients": len(merged),
        })
    return pd.DataFrame(rows)


def transition_table(baseline: pd.DataFrame, candidate: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    keys = ["fold", "subject_id"]
    base = baseline[keys + ["y_true", "y_pred"]].rename(columns={"y_pred": "baseline_pred"})
    cand = candidate[keys + ["y_true", "y_pred"]].rename(columns={"y_pred": "candidate_pred"})
    out = base.merge(cand, on=keys + ["y_true"], how="inner", validate="one_to_one")
    out["baseline_abs_error"] = (out.baseline_pred - out.y_true).abs()
    out["candidate_abs_error"] = (out.candidate_pred - out.y_true).abs()
    before = out.baseline_abs_error <= tolerance
    after = out.candidate_abs_error <= tolerance
    out["transition"] = np.select(
        [~before & after, before & after, ~before & ~after, before & ~after],
        ["improved_to_hit", "remained_hit", "remained_miss", "degraded_to_miss"],
        default="unknown",
    )
    return out.sort_values(["fold", "subject_id"]).reset_index(drop=True)

