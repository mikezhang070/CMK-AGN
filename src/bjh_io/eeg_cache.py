"""On-disk cache for preprocessed EEG arrays produced from BDF files.

Each `(bdf_path, preprocessing parameters)` pair maps to one `.npz` file under
`<repo_root>/.cache/eeg/`. Cache keys include the source file's mtime and size,
so editing the BDF invalidates the cache automatically.

This cache stores ONLY the single-modality preprocessed EEG (drop ref + filter +
resample). Cross-modality synchronization happens upstream, in `bjh_loader`.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Tuple

import numpy as np


_DEFAULT_CACHE_DIR_ENV = "BJH_EEG_CACHE_DIR"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_cache_dir() -> Path:
    """Resolve the cache directory. Override with $BJH_EEG_CACHE_DIR if set."""
    override = os.environ.get(_DEFAULT_CACHE_DIR_ENV)
    if override:
        return Path(override)
    return _project_root() / ".cache" / "eeg"


def _params_hash(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _file_fingerprint(path: Path) -> Tuple[str, int, int]:
    s = path.stat()
    return (str(path.resolve()), int(s.st_mtime_ns), int(s.st_size))


def _cache_key(path: Path, params: dict) -> str:
    fp = _file_fingerprint(path)
    ph = _params_hash(params)
    digest = hashlib.sha1(f"{fp}|{ph}".encode("utf-8")).hexdigest()
    return f"{path.stem}_{digest[:16]}_{ph}"


def load_or_compute(
    path: Path,
    params: dict,
    compute_fn: Callable[[Path, dict], Tuple[np.ndarray, float]],
) -> Tuple[np.ndarray, float]:
    """Return cached `(eeg, fs)` for `path`+`params`, or compute and store it.

    `compute_fn(path, params)` must return `(eeg [T, C] float32, fs float)`.
    """
    path = Path(path)
    cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(path, params)
    cache_file = cache_dir / f"{key}.npz"

    if cache_file.exists():
        try:
            with np.load(cache_file) as data:
                eeg = data["eeg"].astype(np.float32, copy=False)
                fs = float(data["fs"].item())
            return eeg, fs
        except Exception:
            # Corrupted cache — fall through to recompute.
            try:
                cache_file.unlink()
            except OSError:
                pass

    eeg, fs = compute_fn(path, params)
    eeg = np.ascontiguousarray(eeg, dtype=np.float32)

    # np.savez always appends a `.npz` extension. Build a temp basename without
    # one, let savez attach it, then rename to the final cache path atomically.
    tmp_base = cache_file.with_suffix("")  # drops `.npz`
    tmp_base = tmp_base.with_name(tmp_base.name + ".tmp")
    np.savez(tmp_base, eeg=eeg, fs=np.float64(fs))
    os.replace(str(tmp_base) + ".npz", cache_file)
    return eeg, fs
