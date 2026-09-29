"""Task definitions for the 4 clinical prediction tasks.

Each task is trained as an INDEPENDENT model — there is no joint loss and no
loss weight to tune. This module is the single source of truth used by:
    - scripts/build_real_archive_manifest.py
    - train.py             (single-task trainer)
    - predict.py           (single / all-task inference)
    - smoke tests

Hand-tone classes contain the string "1+", which MUST remain a string and
never be parsed as a number. `LabelEncoder` below preserves it.

NOTE: ``wrist_tone`` was retired — it is no longer trained or evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union


# Manifest column name → in-task semantic label.
# These columns are written by the real-case manifest builder.
MANIFEST_COLS: Dict[str, str] = {
    "FMA_UE": "fma_ue",
    "BI": "bi",
    "hand_tone": "hand_tone",        # 手肌张力
    "hand_function": "hand_function",  # 手功能分级
}


@dataclass(frozen=True)
class TaskSpec:
    name: str                       # Canonical task name (e.g. "FMA_UE")
    label_name: str                 # Chinese / human label name
    task_type: str                  # "regression" or "classification"
    manifest_col: str               # Column in samples_manifest_*.csv

    # Regression-only:
    score_min: float = 0.0
    score_max: float = 0.0
    # Stride between adjacent score bins for the hybrid regression head.
    # Set 0.0 to keep the legacy single-Linear regression head; a positive
    # value enables a discrete-distribution + tanh-residual head trained with
    # Huber + cross-entropy on the bin index of the rounded target.
    bin_step: float = 0.0
    # Tolerance on |round(y_pred) - round(y_true)| for "rounded accuracy".
    rounded_tol: float = 1.0
    # Tolerance on |y_pred - y_true| (raw units) for "tolerance accuracy".
    score_tolerance: float = 1.5

    # Classification-only:
    classes: Tuple[Any, ...] = ()
    # Default classification head: "ce" (plain Linear → CE) or "corn" (CORN
    # ordinal head — recommended for ordered clinical scales). Ignored when
    # task_type == "regression".
    default_head: str = "ce"

    # Loss / checkpoint:
    loss: str = "SmoothL1Loss"
    checkpoint: str = ""

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def num_bins(self) -> int:
        """Number of bin centers for the hybrid regression head (0 if disabled)."""
        if self.bin_step <= 0.0:
            return 0
        return int(round((self.score_max - self.score_min) / self.bin_step)) + 1


TASK_CONFIGS: Dict[str, TaskSpec] = {
    "FMA_UE": TaskSpec(
        name="FMA_UE",
        label_name="FMA手部分数",
        task_type="regression",
        manifest_col="fma_ue",
        score_min=0.0,
        score_max=20.0,
        bin_step=1.0,            # 21 integer bins (FMA-UE hand subscale is integer)
        loss="SmoothL1Loss",
        checkpoint="checkpoints/FMA_UE_model.pth",
    ),
    "BI": TaskSpec(
        name="BI",
        label_name="Barthel_Index",
        task_type="regression",
        manifest_col="bi",
        score_min=0.0,
        score_max=100.0,
        bin_step=7.0,            # Barthel scores in steps of 7 → 15 bins
        rounded_tol=5.0,         # BI is in 5-pt increments; ±5 = 1 bin
        score_tolerance=10.0,    # ±10 raw BI points
        loss="SmoothL1Loss",
        checkpoint="checkpoints/BI_model.pth",
    ),
    "hand_tone": TaskSpec(
        name="hand_tone",
        label_name="手肌张力",
        task_type="classification",
        manifest_col="hand_tone",
        classes=("0", "1", "1+", "2", "3", "4"),
        default_head="corn",
        loss="CrossEntropyLoss",
        checkpoint="checkpoints/hand_tone_model.pth",
    ),
    "hand_function": TaskSpec(
        name="hand_function",
        label_name="手功能分级",
        task_type="classification",
        manifest_col="hand_function",
        # The real 29-case label source contains Brunnstrom stages 1--6.
        # Keep stage 1 instead of silently filtering S24/S26 during training.
        classes=(1, 2, 3, 4, 5, 6),
        default_head="corn",
        loss="CrossEntropyLoss",
        checkpoint="checkpoints/hand_function_model.pth",
    ),
}

ALL_TASK_NAMES: Tuple[str, ...] = tuple(TASK_CONFIGS.keys())


# --------------------------------------------------------------------------- #
# Encoder/decoder for classification tasks.                                   #
# --------------------------------------------------------------------------- #
class LabelEncoder:
    """String-aware encoder/decoder for categorical clinical labels.

    Tone classes contain the literal string "1+" which must NOT be coerced to a
    number. Hand-function classes are integers 1..6 but we still expose decoded
    output as the original int label, not the index.
    """

    def __init__(self, classes: Sequence[Any]):
        self.classes: Tuple[Any, ...] = tuple(classes)
        # Build a string-key map so callers can pass "0" or 0 interchangeably,
        # except for "1+" which has no numeric form.
        self._to_index: Dict[str, int] = {}
        for i, c in enumerate(self.classes):
            self._to_index[str(c)] = i
            # Allow integer keys for numeric classes (for hand_function).
            if isinstance(c, int):
                self._to_index[str(int(c))] = i

    def encode(self, value: Any) -> int:
        """Map a raw label to its class index. Accepts str or int."""
        key = str(value).strip()
        if key not in self._to_index:
            raise KeyError(
                f"Label {value!r} not in classes {self.classes!r}. "
                "Note '1+' must be passed as the string \"1+\"."
            )
        return self._to_index[key]

    def decode(self, index: int) -> Any:
        if not 0 <= int(index) < len(self.classes):
            raise IndexError(f"Class index {index} out of range 0..{len(self.classes) - 1}")
        return self.classes[int(index)]


def get_task(name: str) -> TaskSpec:
    if name not in TASK_CONFIGS:
        raise KeyError(
            f"Unknown task {name!r}. Valid: {sorted(TASK_CONFIGS)}"
        )
    return TASK_CONFIGS[name]


def get_encoder(name: str) -> LabelEncoder:
    spec = get_task(name)
    if spec.task_type != "classification":
        raise ValueError(f"Task {name} is not a classification task.")
    return LabelEncoder(spec.classes)


def clip_regression(name: str, value: float) -> float:
    """Postprocess clip for regression predictions."""
    spec = get_task(name)
    if spec.task_type != "regression":
        raise ValueError(f"Task {name} is not a regression task.")
    return float(min(max(value, spec.score_min), spec.score_max))


def checkpoint_path(name: str, root: Path) -> Path:
    spec = get_task(name)
    p = Path(spec.checkpoint)
    return p if p.is_absolute() else (root / p)


__all__ = [
    "MANIFEST_COLS",
    "TaskSpec",
    "TASK_CONFIGS",
    "ALL_TASK_NAMES",
    "LabelEncoder",
    "get_task",
    "get_encoder",
    "clip_regression",
    "checkpoint_path",
]
