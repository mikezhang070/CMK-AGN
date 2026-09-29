"""Uniform wrapper for traditional ML baselines.

Every ML baseline implements the same protocol so the runner only needs to
think in two terms — ``fit(X, y)`` and ``predict(X)``. Probabilities are
exposed for classification baselines that support them (used to write the
``prob_c<i>`` columns of ``val_predictions.csv`` identical to the main model).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from task_config import TaskSpec


class MLBaseline:
    """Base class for traditional ML baselines.

    Subclasses populate ``self.model`` in ``__init__`` with the sklearn /
    xgboost estimator and rely on the default ``fit``/``predict`` here.
    Override only when the estimator does not follow the sklearn API
    (e.g. :class:`MeanMajorityBaseline`).
    """

    family = "ml"

    def __init__(self, spec: TaskSpec, args, name: str):
        self.spec = spec
        self.args = args
        self.name = name
        self.scaler: Optional[StandardScaler] = StandardScaler()
        self.model: Any = None

    # -- sklearn-style API ---------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        assert self.scaler is not None and self.model is not None
        Xs = self.scaler.fit_transform(X)
        if self.spec.task_type == "classification":
            self.model.fit(Xs, y.astype(int))
        else:
            self.model.fit(Xs, y.astype(np.float64))

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.scaler is not None and self.model is not None
        Xs = self.scaler.transform(X)
        if self.spec.task_type == "regression":
            yhat = self.model.predict(Xs).astype(np.float64)
            return np.clip(yhat, self.spec.score_min, self.spec.score_max)
        return self.model.predict(Xs).astype(int)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.spec.task_type != "classification":
            return None
        if self.model is None or self.scaler is None:
            return None
        if not hasattr(self.model, "predict_proba"):
            return None
        Xs = self.scaler.transform(X)
        proba = self.model.predict_proba(Xs).astype(np.float64)
        # Expand sparse class set (sklearn .classes_) back to full spec.num_classes columns
        # so val_predictions.csv always has the same prob_c0..prob_c{K-1} schema as the main model.
        present = getattr(self.model, "classes_", None)
        if present is not None and len(present) != int(self.spec.num_classes):
            full = np.zeros((proba.shape[0], int(self.spec.num_classes)), dtype=np.float64)
            for i, c in enumerate(present):
                full[:, int(c)] = proba[:, i]
            proba = full
        return proba

    # -- persistence ---------------------------------------------------------
    def save(self, path: Path, extra: Optional[dict] = None) -> None:
        payload = {
            "name": self.name,
            "task": self.spec.name,
            "task_type": self.spec.task_type,
            "num_classes": self.spec.num_classes,
            "score_min": self.spec.score_min,
            "score_max": self.spec.score_max,
            "classes": list(self.spec.classes),
            "scaler": self.scaler,
            "model": self.model,
        }
        if extra:
            payload.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)
