"""Thin re-export of metric helpers from :mod:`train`.

Baselines reuse the main model's exact metric implementations to guarantee
identical numbers in the final comparison tables — no risk of method drift.
"""
from baselines._train_shim import (  # noqa: F401
    _classification_metrics as classification_metrics,
    _confusion_matrix as confusion_matrix,
    _regression_metrics as regression_metrics,
    evaluate as evaluate_dl,
    rounded_acc_key,
    tolerance_acc_key,
)
