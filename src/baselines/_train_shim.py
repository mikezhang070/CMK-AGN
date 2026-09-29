"""Import shim so baselines can reuse :mod:`train`'s helpers without
triggering its (currently incomplete) ``clinical_model`` import chain.

:mod:`train` imports :class:`ClinicalPredictionModel` at module top-level
but only instantiates it inside :func:`train.train_one_task`. Baselines
never invoke that function — they only call the data / metric / fold
helpers (``TaskStore``, ``BagDS``, ``evaluate``, ``_load_fold_file`` etc.).

We register a placeholder ``clinical_model`` module so the top-level
``from clinical_model import ClinicalPredictionModel`` in train.py
resolves without dragging in the missing model files. This keeps the
baseline framework runnable independently of the main-model code state.
"""
from __future__ import annotations

import sys
import types

if "clinical_model" not in sys.modules:
    _stub = types.ModuleType("clinical_model")
    _stub.ClinicalPredictionModel = None  # type: ignore[attr-defined]
    sys.modules["clinical_model"] = _stub

# Re-export all train.py symbols that baselines consume.
from train import (  # noqa: F401, E402
    BagDS,
    TaskStore,
    _classification_metrics,
    _confusion_matrix,
    _filter_classification_manifest,
    _fold_subjects,
    _format_subjects,
    _load_fold_file,
    _regression_metrics,
    _select_folds,
    _subject_sort_key,
    _validate_subject_split,
    _warn_rare_classes,
    evaluate,
    rounded_acc_key,
    seed_all,
    tolerance_acc_key,
)
