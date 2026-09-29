"""Hand-crafted features for traditional ML baselines.

Produces one fixed-length subject-level feature vector by averaging per-trial
vectors. Designed deliberately simple so reviewers can replicate it.

Per trial (after tri-modal alignment to length T=256, eeg 30 ch, emg 4 ch,
imu 24 ch) we concatenate:

  * Time-domain statistics per channel (8 feats x 58 chans = 464):
        mean, std, min, max, skew, kurtosis, RMS, zero-crossing count
  * EEG band-power per channel via Welch (5 bands x 30 EEG chans = 150):
        delta (0.5-4 Hz), theta (4-8), alpha (8-13), beta (13-30), gamma (30-45)
  * EMG Hudgins features per channel (4 feats x 4 EMG chans = 16):
        MAV, WL, ZC, SSC

  Total = 630 floats per trial; subject vector = mean across that subject's trials.

@article{hudgins1993new,
  author  = {Hudgins, B. and Parker, P. and Scott, R. N.},
  title   = {A new strategy for multifunction myoelectric control},
  journal = {IEEE Trans. Biomed. Eng.},
  volume  = {40}, number = {1}, year = {1993}, pages = {82--94}
}
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats

# numpy >= 2.0 renamed np.trapz -> np.trapezoid; keep one compatible name.
_trapz = getattr(np, "trapezoid", None)
if _trapz is None:
    _trapz = getattr(np, "trapz", None)

FEATURE_DIM = 630  # 464 (time) + 150 (EEG band) + 16 (Hudgins)

EEG_BANDS: Sequence[tuple] = (
    (0.5, 4.0),    # delta
    (4.0, 8.0),    # theta
    (8.0, 13.0),   # alpha
    (13.0, 30.0),  # beta
    (30.0, 45.0),  # gamma
)


def _as_numpy(arr) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr, dtype=np.float32)


def _time_stats(x: np.ndarray) -> np.ndarray:
    """Return [C, 8] time-domain statistics, then flatten."""
    # x: [C, T]
    C, T = x.shape
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    sk = scipy_stats.skew(x, axis=1, bias=False, nan_policy="omit")
    kt = scipy_stats.kurtosis(x, axis=1, bias=False, nan_policy="omit")
    rms = np.sqrt(np.mean(x ** 2, axis=1))
    # Zero-crossing count: sign changes per channel.
    sx = np.sign(x)
    zc = (np.diff(sx, axis=1) != 0).sum(axis=1).astype(np.float32)
    feats = np.stack([mean, std, mn, mx, sk, kt, rms, zc], axis=1)  # [C, 8]
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)


def _eeg_bandpower(eeg: np.ndarray, fs: float) -> np.ndarray:
    """Welch PSD integrated over each band, per channel; log1p stabilized.

    Returns flat array of length 30 * 5 = 150.
    """
    C, T = eeg.shape
    nperseg = int(min(T, 256))
    freqs, psd = scipy_signal.welch(eeg, fs=fs, nperseg=nperseg, axis=1)
    out = np.empty((C, len(EEG_BANDS)), dtype=np.float32)
    for bi, (lo, hi) in enumerate(EEG_BANDS):
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.any():
            out[:, bi] = _trapz(psd[:, mask], freqs[mask], axis=1)
        else:
            out[:, bi] = 0.0
    out = np.log1p(np.clip(out, 0.0, None))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)


def _hudgins_emg(emg: np.ndarray, zc_threshold: float = 1e-5) -> np.ndarray:
    """Hudgins (1993) feature set per EMG channel: MAV, WL, ZC, SSC."""
    C, T = emg.shape
    mav = np.mean(np.abs(emg), axis=1)
    wl = np.sum(np.abs(np.diff(emg, axis=1)), axis=1)
    # ZC: sign changes that exceed the threshold magnitude.
    diff_sign = np.diff(np.sign(emg), axis=1) != 0
    big_enough = np.abs(emg[:, :-1]) + np.abs(emg[:, 1:]) >= zc_threshold
    zc = (diff_sign & big_enough).sum(axis=1).astype(np.float32)
    # SSC: slope-sign change count.
    d1 = np.diff(emg, axis=1)
    ssc = (np.diff(np.sign(d1), axis=1) != 0).sum(axis=1).astype(np.float32)
    feats = np.stack([mav, wl, zc, ssc], axis=1)  # [C, 4]
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)


def trial_vector(eeg, emg, imu, eeg_fs: float) -> np.ndarray:
    """Build one 1-D feature vector of length :data:`FEATURE_DIM` for a single trial."""
    eeg_np = _as_numpy(eeg)
    emg_np = _as_numpy(emg)
    imu_np = _as_numpy(imu)
    parts = [
        _time_stats(eeg_np),
        _time_stats(emg_np),
        _time_stats(imu_np),
        _eeg_bandpower(eeg_np, eeg_fs),
        _hudgins_emg(emg_np),
    ]
    vec = np.concatenate(parts).astype(np.float32)
    if vec.size != FEATURE_DIM:
        raise RuntimeError(
            f"Feature-dim mismatch: expected {FEATURE_DIM}, got {vec.size}. "
            f"Component sizes: {[p.size for p in parts]}. "
            "If EEG/EMG/IMU channel counts changed, update features.py."
        )
    return vec


def subject_vector(trials: List[dict], eeg_fs: float) -> np.ndarray:
    """Aggregate per-trial vectors into a single subject-level vector (mean)."""
    if not trials:
        raise ValueError("subject_vector called with no trials.")
    rows = [trial_vector(t["eeg"], t["emg"], t["imu"], eeg_fs) for t in trials]
    return np.mean(np.stack(rows, axis=0), axis=0).astype(np.float32)
