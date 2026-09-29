"""XGBoost regressor / classifier.

@inproceedings{chen2016xgboost,
  author = {Chen, T. and Guestrin, C.},
  title  = {XGBoost: A Scalable Tree Boosting System},
  booktitle = {KDD}, year = {2016}, pages = {785--794}
}
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from baselines.ml_wrappers import MLBaseline
from baselines.registry import register


class XGBoostBaseline(MLBaseline):
    def __init__(self, spec, args):
        super().__init__(spec, args, name="xgboost")
        try:
            import xgboost as xgb  # type: ignore
        except ImportError as e:
            raise ImportError(
                "xgboost is not installed. Run `pip install 'xgboost>=1.7'`."
            ) from e

        common = dict(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=int(args.seed),
            n_jobs=1,
            verbosity=0,
            tree_method="hist",
        )
        if spec.task_type == "regression":
            self.model = xgb.XGBRegressor(objective="reg:squarederror", **common)
        else:
            # ``num_class`` is set lazily inside ``fit`` once we know which
            # classes are present in the training fold (XGBoost requires the
            # labels to form a contiguous 0..K-1 set; we map sparse class ids
            # to a dense range and reverse the mapping in predict).
            self._xgb_common = common
            self._xgb_module = xgb
            self.model = None
            self._present_classes: Optional[List[int]] = None

    def fit(self, X, y):
        if self.spec.task_type != "classification":
            return super().fit(X, y)

        assert self.scaler is not None
        Xs = self.scaler.fit_transform(X)
        y_int = np.asarray(y, dtype=int)
        present = sorted(set(int(v) for v in y_int))
        self._present_classes = present
        if len(present) < 2:
            # XGBoost cannot fit on one class; fall back to a constant prediction.
            self.model = _SingleClassFallback(present[0])
            self.model.fit(Xs, y_int)
            return

        remap = {orig: i for i, orig in enumerate(present)}
        y_dense = np.array([remap[int(v)] for v in y_int], dtype=int)
        xgb = self._xgb_module
        self.model = xgb.XGBClassifier(
            objective="multi:softprob" if len(present) > 2 else "binary:logistic",
            num_class=len(present) if len(present) > 2 else None,
            eval_metric="mlogloss" if len(present) > 2 else "logloss",
            **self._xgb_common,
        )
        self.model.fit(Xs, y_dense)

    def predict(self, X):
        if self.spec.task_type != "classification":
            return super().predict(X)
        assert self.scaler is not None and self.model is not None and self._present_classes is not None
        Xs = self.scaler.transform(X)
        y_dense = np.asarray(self.model.predict(Xs)).astype(int)
        return np.array([self._present_classes[int(v)] for v in y_dense], dtype=int)

    def predict_proba(self, X):
        if self.spec.task_type != "classification":
            return None
        assert self.scaler is not None and self.model is not None and self._present_classes is not None
        Xs = self.scaler.transform(X)
        if not hasattr(self.model, "predict_proba"):
            return None
        proba_dense = np.asarray(self.model.predict_proba(Xs), dtype=np.float64)
        if proba_dense.ndim == 1:  # binary fallback: [N] -> [N, 2]
            proba_dense = np.stack([1 - proba_dense, proba_dense], axis=1)
        proba_full = np.zeros((proba_dense.shape[0], int(self.spec.num_classes)), dtype=np.float64)
        for i, c in enumerate(self._present_classes):
            proba_full[:, c] = proba_dense[:, i]
        return proba_full


class _SingleClassFallback:
    """When training data has a single class, just predict that class."""

    def __init__(self, cls: int):
        self.cls = int(cls)

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros((X.shape[0],), dtype=int)  # dense index 0 -> only present class

    def predict_proba(self, X):
        return np.ones((X.shape[0], 1), dtype=np.float64)


@register("xgboost")
def build(spec, args):
    return XGBoostBaseline(spec, args)
