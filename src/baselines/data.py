"""Data utilities shared by ML and DL baselines.

All functions are thin wrappers over the main trainer's primitives so that
folds, subject splits, alignment, and metric semantics are guaranteed
identical to the main model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import after sys.path is configured by train_baseline.py entry point.
from alignment.wby_dtw import WBYDTWConfig
from task_config import LabelEncoder, TaskSpec

from baselines._train_shim import TaskStore  # reuse trial loading + alignment
from baselines.features import subject_vector


def build_stores(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    root: Path,
    spec: TaskSpec,
    encoder: Optional[LabelEncoder],
    seq_len: int,
    dtw_length: int,
    alignment_mode: str,
    eeg_fs: float,
    preprocess: bool,
    cache_dir: Optional[Path] = None,
) -> Tuple[TaskStore, TaskStore]:
    """Build train/val :class:`TaskStore` objects with main-model-identical alignment."""
    cfg = WBYDTWConfig(
        output_length=seq_len,
        dtw_length=dtw_length,
        band_radius=0.15,
        alpha=0.7,
        beta=0.3,
    )
    tr_store = TaskStore(df_train, root, spec, encoder, seq_len, cfg,
                         alignment_mode, eeg_fs, preprocess, cache_dir)
    ev_store = TaskStore(df_val, root, spec, encoder, seq_len, cfg,
                         alignment_mode, eeg_fs, preprocess, cache_dir)
    return tr_store, ev_store


def stores_to_xy(
    store: TaskStore,
    eeg_fs: float,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build a subject-level feature matrix X[N, D] + label vector y[N]."""
    sids = sorted(store.subjects.keys(), key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
    X_rows = []
    y_rows = []
    for sid in sids:
        entry = store.subjects[sid]
        X_rows.append(subject_vector(entry["trials"], eeg_fs))
        y_rows.append(float(entry["target"]))
    X = np.stack(X_rows, axis=0).astype(np.float32)
    y = np.array(y_rows, dtype=np.float64)
    return X, y, sids
