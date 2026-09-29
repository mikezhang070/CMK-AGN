"""Mean (regression) / Majority class (classification) — sanity floor.

No primary citation: the constant predictor is the textbook null baseline
referenced in every clinical-ML paper as a sanity check.
"""
from __future__ import annotations

import numpy as np

from baselines.ml_wrappers import MLBaseline
from baselines.registry import register


class _ConstantEstimator:
    """sklearn-compatible constant predictor."""

    def __init__(self, task_type: str, num_classes: int):
        self.task_type = task_type
        self.num_classes = int(num_classes)
        self.value: float = 0.0
        self.proba: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_ConstantEstimator":
        if self.task_type == "regression":
            self.value = float(np.mean(y))
        else:
            counts = np.bincount(np.asarray(y, dtype=int), minlength=self.num_classes)
            self.value = float(int(np.argmax(counts)))
            self.proba = counts.astype(np.float64) / max(int(counts.sum()), 1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if self.task_type == "regression":
            return np.full((n,), self.value, dtype=np.float64)
        return np.full((n,), int(self.value), dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.proba is not None
        return np.tile(self.proba[None, :], (X.shape[0], 1))


class MeanMajorityBaseline(MLBaseline):
    def __init__(self, spec, args):
        super().__init__(spec, args, name="mean_majority")
        self.scaler = None  # constant predictor has no use for standardization
        self.model = _ConstantEstimator(spec.task_type, spec.num_classes or 1)

    def fit(self, X, y):
        if self.spec.task_type == "regression":
            self.model.fit(X, np.asarray(y, dtype=np.float64))
        else:
            self.model.fit(X, np.asarray(y, dtype=int))

    def predict(self, X):
        yhat = self.model.predict(X)
        if self.spec.task_type == "regression":
            return np.clip(yhat, self.spec.score_min, self.spec.score_max)
        return yhat.astype(int)

    def predict_proba(self, X):
        if self.spec.task_type != "classification":
            return None
        return self.model.predict_proba(X)


@register("mean_majority")
def build(spec, args):
    return MeanMajorityBaseline(spec, args)
