"""Patient-level confirmation and split-conformal utilities.

The functions here do not create calibration predictions themselves.  Callers
must provide predictions from an inner, training-patient-only calibration set;
the utility rejects any overlap with outer-test subjects.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


REQUIRED_PREDICTION_COLUMNS = {"subject_id", "y_true", "y_pred"}


def split_conformal_radius(abs_residuals: Iterable[float], alpha: float = 0.1) -> float:
    """Return the finite-sample split-conformal absolute-residual radius.

    Uses the order statistic at ``ceil((n + 1) * (1 - alpha))``.  This is the
    standard finite-sample conservative quantile, rather than a post-hoc test
    residual quantile.
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    residuals = np.sort(np.asarray(list(abs_residuals), dtype=float))
    if residuals.size == 0 or not np.isfinite(residuals).all():
        raise ValueError("Calibration residuals must be non-empty finite values.")
    rank = min(residuals.size, int(math.ceil((residuals.size + 1) * (1.0 - alpha))))
    return float(residuals[rank - 1])


def conformal_intervals(
    calibration: pd.DataFrame, test: pd.DataFrame, alpha: float = 0.1,
) -> Tuple[pd.DataFrame, Dict[str, float | int]]:
    """Add symmetric split-conformal prediction intervals to held-out patients."""
    for name, frame in (("calibration", calibration), ("test", test)):
        missing = REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{name} predictions are empty.")
        if frame["subject_id"].astype(str).duplicated().any():
            raise ValueError(f"{name} contains duplicate subject_id rows.")
    cal_ids = set(calibration["subject_id"].astype(str))
    test_ids = set(test["subject_id"].astype(str))
    overlap = sorted(cal_ids & test_ids)
    if overlap:
        raise ValueError(f"Calibration/test subject overlap is not allowed: {overlap}")

    cal_error = np.abs(
        calibration["y_true"].astype(float).to_numpy() - calibration["y_pred"].astype(float).to_numpy()
    )
    radius = split_conformal_radius(cal_error, alpha=alpha)
    out = test.copy()
    out["interval_lower"] = out["y_pred"].astype(float) - radius
    out["interval_upper"] = out["y_pred"].astype(float) + radius
    out["interval_width"] = 2.0 * radius
    out["covered"] = (
        (out["y_true"].astype(float) >= out["interval_lower"])
        & (out["y_true"].astype(float) <= out["interval_upper"])
    ).astype(int)
    return out, {
        "alpha": float(alpha),
        "nominal_coverage": float(1.0 - alpha),
        "n_calibration": int(len(calibration)),
        "n_test": int(len(test)),
        "radius": radius,
        "coverage": float(out["covered"].mean()),
        "mean_interval_width": float(out["interval_width"].mean()),
    }


__all__ = ["REQUIRED_PREDICTION_COLUMNS", "conformal_intervals", "split_conformal_radius"]
