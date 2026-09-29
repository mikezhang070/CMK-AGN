from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _safe_minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        return np.ones_like(values, dtype=np.float64)
    return (values - vmin) / (vmax - vmin)


def _replace_nonfinite_with_median(values: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    fill_value = float(np.median(finite)) if finite.size else float(fallback)
    return np.where(np.isfinite(values), values, fill_value)


def add_quality_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "alignment_score" not in result.columns:
        result["alignment_score"] = np.nan
    if "signal_quality" not in result.columns:
        result["signal_quality"] = np.nan
    if "prediction_uncertainty" not in result.columns:
        result["prediction_uncertainty"] = np.nan

    alignment = result["alignment_score"].to_numpy(dtype=np.float64)
    signal = result["signal_quality"].to_numpy(dtype=np.float64)
    uncertainty = result["prediction_uncertainty"].to_numpy(dtype=np.float64)

    alignment = _replace_nonfinite_with_median(alignment)
    signal = _replace_nonfinite_with_median(signal)
    uncertainty = _replace_nonfinite_with_median(uncertainty)

    align_norm = _safe_minmax(alignment)
    signal_norm = _safe_minmax(signal)
    uncertainty_norm = 1.0 - _safe_minmax(uncertainty)
    quality = 0.45 * align_norm + 0.35 * signal_norm + 0.20 * uncertainty_norm
    result["trial_quality"] = quality.astype(np.float64)
    result["quality_weight"] = np.clip(quality + 1e-3, 1e-3, None)
    result["uncertainty_weight"] = np.clip(uncertainty_norm + 1e-3, 1e-3, None)
    result["hybrid_weight"] = np.clip(0.6 * result["quality_weight"] + 0.4 * result["uncertainty_weight"], 1e-3, None)
    return result




def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.clip(np.asarray(weights, dtype=np.float64), 1e-6, None)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    idx = int(np.searchsorted(cdf, 0.5, side="left"))
    idx = min(max(idx, 0), len(values) - 1)
    return float(values[idx])


def _robust_hybrid_aggregate(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.clip(np.asarray(weights, dtype=np.float64), 1e-6, None)
    if values.size <= 2:
        return _weighted_mean(values, weights)
    center = _weighted_median(values, weights)
    abs_dev = np.abs(values - center)
    mad = float(np.median(abs_dev))
    scale = max(0.35, 1.4826 * mad)
    keep = abs_dev <= (1.75 * scale)
    if np.sum(keep) < max(2, values.size // 2):
        keep = abs_dev <= np.quantile(abs_dev, 0.75)
    if np.sum(keep) == 0:
        keep = np.ones_like(values, dtype=bool)
    return _weighted_mean(values[keep], weights[keep])

def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    weights = np.clip(weights, 1e-6, None)
    return float(np.sum(values * weights) / np.sum(weights))


def aggregate_subject_predictions(predictions: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    if predictions.empty:
        raise RuntimeError("No predictions available for subject-level aggregation.")
    method = str(method).lower()
    supported = {"mean", "median", "quality_weighted", "uncertainty_weighted", "hybrid_weighted", "robust_hybrid_weighted"}
    if method not in supported:
        raise ValueError(f"Unknown subject aggregation method {method!r}; expected one of {sorted(supported)}")

    enriched = add_quality_columns(predictions)
    rows: List[Dict[str, object]] = []
    group_cols = [column for column in ["fold", "subject_id"] if column in enriched.columns]
    if not group_cols:
        group_cols = ["subject_id"]
    for group_key, frame in enriched.groupby(group_cols, sort=True):
        if isinstance(group_key, tuple):
            fold_value = int(group_key[0]) if "fold" in group_cols else None
            subject_id = str(group_key[-1])
        else:
            fold_value = int(frame["fold"].iloc[0]) if "fold" in frame.columns else None
            subject_id = str(group_key)
        unique_true = np.unique(frame["fma_true"].to_numpy(dtype=float))
        if unique_true.size != 1:
            raise ValueError(f"Subject {subject_id} has inconsistent FMA labels: {unique_true.tolist()}")
        pred_values = frame["fma_pred"].to_numpy(dtype=float)
        if method == "mean":
            pred_value = float(np.mean(pred_values))
        elif method == "median":
            pred_value = float(np.median(pred_values))
        elif method == "quality_weighted":
            pred_value = _weighted_mean(pred_values, frame["quality_weight"].to_numpy(dtype=float))
        elif method == "uncertainty_weighted":
            pred_value = _weighted_mean(pred_values, frame["uncertainty_weight"].to_numpy(dtype=float))
        else:
            pred_value = _weighted_mean(pred_values, frame["hybrid_weight"].to_numpy(dtype=float))
        true_value = float(unique_true[0])
        error = pred_value - true_value
        row = {
            "subject_id": subject_id,
            "fma_true": true_value,
            "fma_pred": pred_value,
            "error": error,
            "abs_error": abs(error),
            "n_trials": int(len(frame)),
            "aggregation_method": method,
            "trial_quality_mean": float(frame["trial_quality"].mean()),
            "alignment_score_mean": float(frame["alignment_score"].mean()) if np.isfinite(frame["alignment_score"]).any() else float("nan"),
            "signal_quality_mean": float(frame["signal_quality"].mean()) if np.isfinite(frame["signal_quality"]).any() else float("nan"),
            "prediction_uncertainty_mean": float(frame["prediction_uncertainty"].mean()) if np.isfinite(frame["prediction_uncertainty"]).any() else float("nan"),
        }
        if fold_value is not None:
            row["fold"] = fold_value
        rows.append(row)
    ordered_cols = [column for column in ["fold", "subject_id", "fma_true", "fma_pred", "error", "abs_error", "n_trials", "aggregation_method", "trial_quality_mean", "alignment_score_mean", "signal_quality_mean", "prediction_uncertainty_mean"] if any(column in row for row in rows)]
    return pd.DataFrame(rows)[ordered_cols]
