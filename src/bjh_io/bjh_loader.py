from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal as scisig

from .eeg_cache import load_or_compute as _eeg_cache_load_or_compute


EEG_CHANNELS: Tuple[str, ...] = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FT7", "FC3", "FCz", "FC4", "FT8",
    "T3", "C3", "Cz", "C4", "T4",
    "TP7", "CP3", "CPz", "CP4", "TP8",
    "A1", "T5", "P3", "Pz", "P4", "T6", "A2",
    "O1", "Oz", "O2",
)

# Channel layout after dropping the A1/A2 mastoid references in the BDF path.
EEG_CHANNELS_BDF_30: Tuple[str, ...] = tuple(c for c in EEG_CHANNELS if c not in ("A1", "A2"))

# Motor-area channels used as the EEG side of the cross-correlation sync.
MOTOR_CHANNELS_FOR_SYNC: Tuple[str, ...] = ("C3", "Cz", "C4", "CP3", "CPz", "CP4")

EMG_MUSCLES: Tuple[str, ...] = (
    "FLEXOR CARPI RADIALIS",
    "PALMARIS LONGUS",
    "EXTENSOR CARPI ULNARIS",
    "EXTENSOR DIGITORUM",
)

IMU_AXES_PER_MUSCLE: Tuple[str, ...] = (
    "ACC.X", "ACC.Y", "ACC.Z", "GYRO.X", "GYRO.Y", "GYRO.Z",
)

# EEG sample rate:
#   - synthetic CSV (legacy + simulate_data.py) is written at 1000 Hz.
#   - real BDF recordings are loaded at 1000 Hz, then resampled to 500 Hz
#     inside `load_eeg_bdf` (matches the JSON sidecar config used clinically).
EEG_FS_DEFAULT: float = 1000.0
EEG_FS_BDF_OUT: float = 500.0
EMG_FS_DEFAULT: float = 1259.4  # 1 / 0.0007941176 s
IMU_FS_DEFAULT: float = 148.1   # 1 / 0.00675 s

# Defaults for the BDF preprocessing chain (overridable via load_bjh_trial).
EEG_BDF_BANDPASS: Tuple[float, float] = (1.5, 50.0)
EEG_BDF_NOTCH: float = 50.0
EEG_BDF_DROP_CHANNELS: Tuple[str, ...] = ("A1", "A2")
EEG_BDF_REREF: str = "average"

# Tri-modal synchronisation knobs (cross-correlation on a downsampled envelope).
SYNC_ENV_FS: float = 100.0
SYNC_MAX_LAG_S: float = 15.0
SYNC_MIN_PEAK_CORR: float = 0.2


@dataclass
class TriModalSignals:
    """Tri-modal recording for a single trial after de-padding and resampling.

    All arrays are time-major: shape [T, C].
    """
    eeg: np.ndarray              # [T_eeg, 32]
    emg: np.ndarray              # [T_emg, 4]
    imu: np.ndarray              # [T_imu, 24]   (4 muscles * 6 axes, muscle-major)
    eeg_fs: float
    emg_fs: float
    imu_fs: float
    duration: float              # active duration in seconds
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# EEG                                                                         #
# --------------------------------------------------------------------------- #
def _butter_bandpass(low: float, high: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    low_n = max(low / nyq, 1e-4)
    high_n = min(high / nyq, 0.999)
    if not 0 < low_n < high_n < 1.0:
        raise ValueError(f"Invalid bandpass [{low}, {high}] for fs={fs}")
    return scisig.butter(order, [low_n, high_n], btype="band", output="sos")


def _notch(freq: float, fs: float, q: float = 30.0):
    return scisig.iirnotch(w0=freq / (0.5 * fs), Q=q)


def preprocess_eeg(
    eeg: np.ndarray,
    fs: float,
    bandpass: Optional[Tuple[float, float]] = (1.0, 45.0),
    notch_freq: Optional[float] = 50.0,
    detrend: bool = True,
    robust_clip: float = 6.0,
) -> np.ndarray:
    """Standard EEG cleaning: detrend -> notch -> bandpass -> robust z-score.

    Input/output shape: [T, 32].
    """
    if eeg.ndim != 2:
        raise ValueError(f"Expected [T, C], got {eeg.shape}")
    x = eeg.astype(np.float64, copy=True)
    if detrend:
        x = scisig.detrend(x, axis=0, type="linear")
    if notch_freq is not None and 0 < notch_freq < 0.5 * fs:
        b, a = _notch(notch_freq, fs)
        x = scisig.filtfilt(b, a, x, axis=0)
    if bandpass is not None:
        sos = _butter_bandpass(bandpass[0], bandpass[1], fs)
        x = scisig.sosfiltfilt(sos, x, axis=0)
    # Per-channel robust z-score using MAD; clip extreme spikes.
    median = np.median(x, axis=0, keepdims=True)
    mad = np.median(np.abs(x - median), axis=0, keepdims=True)
    scale = np.where(mad > 1e-9, 1.4826 * mad, 1.0)
    x = (x - median) / scale
    if robust_clip > 0:
        x = np.clip(x, -robust_clip, robust_clip)
    return x.astype(np.float32)


def load_eeg(
    path: Path | str,
    columns: Sequence[str] = EEG_CHANNELS_BDF_30,
    expected_fs: float = EEG_FS_DEFAULT,
) -> Tuple[np.ndarray, float]:
    """Load BJH EEG csv (legacy / synthetic path).

    Defaults to the 30-channel layout (A1/A2 mastoid refs dropped) so the
    synthetic CSV pipeline matches the BDF pipeline.

    Returns:
        eeg: [T, len(columns)] float32 raw values (no preprocessing yet).
        fs:  assumed sampling rate (no timestamp column in BJH EEG; uses default).
    """
    path = Path(path)
    df = pd.read_csv(path)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: EEG missing channels {missing}")
    arr = df[list(columns)].to_numpy(dtype=np.float32, copy=False)
    return arr, float(expected_fs)


def load_eeg_csv_bdf_equiv(
    path: Path | str,
    target_fs: float = EEG_FS_BDF_OUT,
    bandpass: Tuple[float, float] = EEG_BDF_BANDPASS,
    notch: Optional[float] = EEG_BDF_NOTCH,
    drop_channels: Sequence[str] = EEG_BDF_DROP_CHANNELS,
    reref: Optional[str] = EEG_BDF_REREF,
    source_fs: float = EEG_FS_DEFAULT,
) -> Tuple[np.ndarray, float, Sequence[str]]:
    """Load a synthetic/legacy CSV EEG through the *same* chain as ``load_eeg_bdf``
    but **without** z-scoring — so its biomarkers are comparable to the real BDF path.

    Chain: read 32 channels → drop A1/A2 → average re-reference → notch → bandpass →
    resample to ``target_fs``. Deliberately omits the robust z-score that
    ``preprocess_eeg`` applies (which would destroy absolute power scale).

    Returns ``(eeg [T, C] float32, fs, channel_names)`` with the exact channel order
    of ``load_eeg_bdf`` (``EEG_CHANNELS`` minus ``drop_channels``).
    """
    path = Path(path)
    df = pd.read_csv(path)
    missing = [c for c in EEG_CHANNELS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: EEG CSV missing channels {missing}")
    drop = set(drop_channels)
    keep = tuple(c for c in EEG_CHANNELS if c not in drop)
    x = df[list(keep)].to_numpy(dtype=np.float64, copy=True)  # [T, C]

    if reref == "average":
        x = x - x.mean(axis=1, keepdims=True)
    elif reref:
        raise ValueError(f"unsupported reref {reref!r} (only 'average' or None)")

    if notch is not None and 0.0 < float(notch) < 0.5 * source_fs:
        b, a = _notch(float(notch), source_fs)
        x = scisig.filtfilt(b, a, x, axis=0)
    if bandpass is not None:
        sos = _butter_bandpass(bandpass[0], bandpass[1], source_fs)
        x = scisig.sosfiltfilt(sos, x, axis=0)

    fs = float(source_fs)
    if float(target_fs) > 0.0 and abs(source_fs - float(target_fs)) > 1e-6:
        cols = [_decimate_to(x[:, i], source_fs, float(target_fs)) for i in range(x.shape[1])]
        x = np.stack(cols, axis=1)
        fs = float(target_fs)

    return x.astype(np.float32), fs, keep


# --------------------------------------------------------------------------- #
# EEG (BDF — real recordings)                                                 #
# --------------------------------------------------------------------------- #
def _compute_eeg_bdf(path: Path, params: dict) -> Tuple[np.ndarray, float]:
    """Read one BDF, drop refs, re-reference, filter, resample — return [T, C]."""
    import mne  # local import keeps mne optional for users on CSV-only paths

    target_fs = float(params["target_fs"])
    bandpass = tuple(params["bandpass"])
    notch = params.get("notch")
    drop_channels = tuple(params.get("drop_channels") or ())
    reref = params.get("reref", "average")

    raw = mne.io.read_raw_bdf(str(path), preload=True, verbose="ERROR")
    to_drop = [c for c in drop_channels if c in raw.ch_names]
    if to_drop:
        raw.drop_channels(to_drop)
    if reref:
        raw.set_eeg_reference(reref, projection=False, verbose="ERROR")
    if notch is not None and float(notch) > 0.0:
        raw.notch_filter(freqs=[float(notch)], verbose="ERROR")
    raw.filter(l_freq=float(bandpass[0]), h_freq=float(bandpass[1]),
               verbose="ERROR")
    if float(target_fs) > 0.0 and abs(raw.info["sfreq"] - float(target_fs)) > 1e-6:
        raw.resample(float(target_fs), npad="auto", verbose="ERROR")

    data = raw.get_data().T.astype(np.float32)  # [T, C]
    fs = float(raw.info["sfreq"])
    return data, fs


def load_eeg_bdf(
    path: Path | str,
    target_fs: float = EEG_FS_BDF_OUT,
    bandpass: Tuple[float, float] = EEG_BDF_BANDPASS,
    notch: Optional[float] = EEG_BDF_NOTCH,
    drop_channels: Sequence[str] = EEG_BDF_DROP_CHANNELS,
    reref: Optional[str] = EEG_BDF_REREF,
    use_cache: bool = True,
) -> Tuple[np.ndarray, float, Sequence[str]]:
    """Load + preprocess one BJH BDF file. Cached to disk on first call.

    Returns `(eeg [T, C] float32, fs, channel_names)`. The filter+resample chain
    runs inside the cache layer; no further filtering should be applied
    downstream (avoids stacking filtfilt passes).
    """
    path = Path(path)
    params = {
        "target_fs": float(target_fs),
        "bandpass": [float(bandpass[0]), float(bandpass[1])],
        "notch": float(notch) if notch is not None else None,
        "drop_channels": list(drop_channels),
        "reref": reref,
    }
    if use_cache:
        eeg, fs = _eeg_cache_load_or_compute(path, params, _compute_eeg_bdf)
    else:
        eeg, fs = _compute_eeg_bdf(path, params)

    expected_channels = tuple(c for c in EEG_CHANNELS if c not in set(drop_channels))
    if eeg.shape[1] != len(expected_channels):
        # Cached payload doesn't carry channel names — fall back to the order in
        # EEG_CHANNELS minus drop_channels.
        return eeg, fs, tuple(f"ch{i}" for i in range(eeg.shape[1]))
    return eeg, fs, expected_channels


# --------------------------------------------------------------------------- #
# Tri-modal synchronisation (cross-correlation on envelopes)                  #
# --------------------------------------------------------------------------- #
def _decimate_to(x: np.ndarray, src_fs: float, dst_fs: float) -> np.ndarray:
    """Linear-interp downsample of a 1-D signal from src_fs to dst_fs."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0 or src_fs <= 0 or dst_fs <= 0 or abs(src_fs - dst_fs) < 1e-6:
        return x.astype(np.float32, copy=False)
    dur = (x.shape[0] - 1) / float(src_fs)
    n_out = max(2, int(round(dur * float(dst_fs))) + 1)
    src_t = np.arange(x.shape[0], dtype=np.float64) / float(src_fs)
    dst_t = np.arange(n_out, dtype=np.float64) / float(dst_fs)
    dst_t = np.clip(dst_t, 0.0, src_t[-1])
    return np.interp(dst_t, src_t, x).astype(np.float32)


def _z_envelope(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = float(x.mean()) if x.size else 0.0
    sd = float(x.std()) if x.size else 0.0
    if sd < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mu) / sd).astype(np.float32)


def _emg_sync_envelope(emg: np.ndarray, fs: float, dst_fs: float = SYNC_ENV_FS) -> np.ndarray:
    """|EMG| → 6 Hz lowpass → mean across muscles → decimate to dst_fs → z-score."""
    if emg.size == 0:
        return np.zeros(0, dtype=np.float32)
    x = np.abs(emg.astype(np.float32))
    nyq = 0.5 * float(fs)
    if nyq > 6.0:
        sos = scisig.butter(4, 6.0 / nyq, btype="low", output="sos")
        x = scisig.sosfiltfilt(sos, x, axis=0)
    env = x.mean(axis=1)
    env = _decimate_to(env, fs, dst_fs)
    return _z_envelope(env)


def _eeg_mu_beta_envelope(
    eeg: np.ndarray,
    fs: float,
    channels: Sequence[str],
    motor_ch: Sequence[str] = MOTOR_CHANNELS_FOR_SYNC,
    dst_fs: float = SYNC_ENV_FS,
) -> np.ndarray:
    """μ/β-band power envelope on motor channels, decimated to dst_fs and z-scored."""
    if eeg.size == 0:
        return np.zeros(0, dtype=np.float32)
    motor_idx = [channels.index(c) for c in motor_ch if c in channels]
    if not motor_idx:
        # Fallback: average all channels.
        motor_idx = list(range(eeg.shape[1]))
    x = eeg[:, motor_idx].astype(np.float32)
    nyq = 0.5 * float(fs)
    low = min(8.0, 0.9 * nyq)
    high = min(30.0, 0.95 * nyq)
    if high > low + 1.0:
        sos = scisig.butter(4, [low / nyq, high / nyq], btype="band", output="sos")
        x = scisig.sosfiltfilt(sos, x, axis=0)
    x = x ** 2
    lowpass = min(5.0, 0.9 * nyq)
    if lowpass > 0.1:
        sos = scisig.butter(4, lowpass / nyq, btype="low", output="sos")
        x = scisig.sosfiltfilt(sos, x, axis=0)
    x = np.sqrt(np.clip(x, 0.0, None))
    env = x.mean(axis=1)
    env = _decimate_to(env, fs, dst_fs)
    return _z_envelope(env)


def estimate_lag_xcorr(
    emg_env: np.ndarray,
    eeg_env: np.ndarray,
    fs_env: float = SYNC_ENV_FS,
    max_lag_s: float = SYNC_MAX_LAG_S,
) -> Tuple[float, float]:
    """Return `(lag_s, peak_corr)`.

    `lag_s > 0` means EMG starts *later* than EEG (EEG lead). The convention is
    chosen so that `t0_emg = max(0, lag_s)` and `t0_eeg = max(0, -lag_s)`
    bring both signals back onto a common time axis.
    """
    emg_env = np.asarray(emg_env, dtype=np.float32)
    eeg_env = np.asarray(eeg_env, dtype=np.float32)
    if emg_env.size < 2 or eeg_env.size < 2:
        return 0.0, float("nan")

    corr = scisig.correlate(emg_env, eeg_env, mode="full", method="auto")
    lags = scisig.correlation_lags(len(emg_env), len(eeg_env), mode="full")

    max_lag_samples = int(round(float(max_lag_s) * float(fs_env)))
    if max_lag_samples > 0:
        mask = np.abs(lags) <= max_lag_samples
        if not mask.any():
            return 0.0, float("nan")
        corr = corr[mask]
        lags = lags[mask]

    peak_idx = int(np.argmax(corr))
    lag_samples = int(lags[peak_idx])
    # Normalise correlation peak by the geometric mean of the energies in the
    # overlapping window, to get something comparable to a Pearson r.
    if lag_samples >= 0:
        a = emg_env[lag_samples:]
        b = eeg_env[: len(a)]
    else:
        b = eeg_env[-lag_samples:]
        a = emg_env[: len(b)]
    n = int(min(len(a), len(b)))
    if n < 4:
        peak_corr = float("nan")
    else:
        a = a[:n] - a[:n].mean()
        b = b[:n] - b[:n].mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        peak_corr = float(np.dot(a, b) / denom) if denom > 1e-12 else float("nan")

    lag_s = float(lag_samples) / float(fs_env)
    return lag_s, peak_corr


def crop_to_common_window(
    eeg: np.ndarray, eeg_fs: float,
    emg: np.ndarray, emg_fs: float,
    imu: np.ndarray, imu_fs: float,
    lag_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Shift EMG/IMU back by `lag_s` and crop all three to a shared window.

    `lag_s > 0` ⇒ EMG/IMU started later than EEG by that many seconds. Returns
    the cropped arrays plus `(t0_eeg, t0_peripheral, common_window_s)`.
    """
    eeg_dur = eeg.shape[0] / float(eeg_fs) if eeg_fs > 0 else 0.0
    emg_dur = emg.shape[0] / float(emg_fs) if emg_fs > 0 else 0.0
    imu_dur = imu.shape[0] / float(imu_fs) if imu_fs > 0 else 0.0

    if lag_s >= 0.0:
        t0_eeg = float(lag_s)
        t0_peripheral = 0.0
    else:
        t0_eeg = 0.0
        t0_peripheral = float(-lag_s)

    common = max(
        0.0,
        min(eeg_dur - t0_eeg, emg_dur - t0_peripheral, imu_dur - t0_peripheral),
    )

    def _slice(arr: np.ndarray, fs: float, t0: float, dur: float) -> np.ndarray:
        if arr.size == 0 or fs <= 0 or dur <= 0:
            return arr[:0]
        i0 = int(round(t0 * float(fs)))
        i1 = i0 + int(round(dur * float(fs)))
        i0 = max(0, min(i0, arr.shape[0]))
        i1 = max(i0, min(i1, arr.shape[0]))
        return arr[i0:i1]

    eeg_c = _slice(eeg, eeg_fs, t0_eeg, common)
    emg_c = _slice(emg, emg_fs, t0_peripheral, common)
    imu_c = _slice(imu, imu_fs, t0_peripheral, common)
    return eeg_c, emg_c, imu_c, t0_eeg, t0_peripheral, common


# --------------------------------------------------------------------------- #
# EMG / IMU                                                                   #
# --------------------------------------------------------------------------- #
def _emg_signal_col(df_columns: Sequence[str], muscle: str) -> str:
    candidates = [c for c in df_columns if muscle in c and ": EMG" in c]
    if not candidates:
        raise KeyError(f"No EMG column for muscle {muscle!r}")
    return candidates[0]


def _imu_signal_col(df_columns: Sequence[str], muscle: str, axis: str) -> str:
    candidates = [c for c in df_columns if muscle in c and f": {axis}" in c]
    if not candidates:
        raise KeyError(f"No {axis} column for muscle {muscle!r}")
    return candidates[0]


def _time_col_for(df_columns: Sequence[str], signal_col: str) -> str:
    """Return the X[s] column immediately preceding `signal_col` in the CSV.

    BJH EMG_new layout: every signal column is preceded by its own X[s] timestamp.
    """
    cols = list(df_columns)
    idx = cols.index(signal_col)
    if idx == 0 or not cols[idx - 1].startswith("X[s]"):
        raise ValueError(f"Cannot locate timestamp column for {signal_col!r}")
    return cols[idx - 1]


def _strip_padding(values: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Drop trailing NaN/zero placeholder rows using the timestamp column.

    A row is "padding" if its timestamp is non-finite or duplicates the previous
    timestamp (Delsys file format pads after each block ends).
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values)
    finite = np.isfinite(times)
    if not finite.any():
        return values[:0], times[:0]
    # Strict monotonic increasing prefix.
    diffs = np.diff(times)
    keep = np.empty_like(times, dtype=bool)
    keep[0] = finite[0]
    keep[1:] = finite[1:] & (diffs > 1e-9)
    return values[keep], times[keep]


def load_emg_imu(
    path: Path | str,
    muscles: Sequence[str] = EMG_MUSCLES,
    axes: Sequence[str] = IMU_AXES_PER_MUSCLE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load BJH EMG_new csv, splitting EMG (high-rate) from IMU (low-rate).

    Returns:
        emg:   [T_emg, len(muscles)]                — float32
        imu:   [T_imu, len(muscles) * len(axes)]    — float32, muscle-major
        t_emg: [T_emg]                              — float64 seconds
        t_imu: [T_imu]                              — float64 seconds
        meta:  {emg_fs, imu_fs, duration_emg, duration_imu, ...}

    The Delsys raw layout repeats the (X[s], signal) pair for every channel; EMG
    rows differ from IMU rows because their sampling rates differ. Padding rows
    (NaN time after a block ends) are stripped per modality.
    """
    path = Path(path)
    df = pd.read_csv(path)
    cols = list(df.columns)

    # ---- EMG ----
    emg_columns = [_emg_signal_col(cols, m) for m in muscles]
    t_emg_col = _time_col_for(cols, emg_columns[0])
    # All EMG signals share the SAME sampling instants (Delsys EMG channels are
    # always sampled together), but each gets its own X[s] copy. We use the
    # first muscle's timestamp as the canonical EMG clock.
    t_emg_raw = df[t_emg_col].to_numpy(dtype=np.float64, copy=False)
    emg_raw = np.stack(
        [df[c].to_numpy(dtype=np.float32, copy=False) for c in emg_columns], axis=1
    )
    emg, t_emg = _strip_padding(emg_raw, t_emg_raw)

    # ---- IMU (ACC + GYRO, also synchronously sampled per Delsys IMU block) ----
    imu_signal_cols = [_imu_signal_col(cols, m, a) for m in muscles for a in axes]
    t_imu_col = _time_col_for(cols, imu_signal_cols[0])
    t_imu_raw = df[t_imu_col].to_numpy(dtype=np.float64, copy=False)
    imu_raw = np.stack(
        [df[c].to_numpy(dtype=np.float32, copy=False) for c in imu_signal_cols], axis=1
    )
    imu, t_imu = _strip_padding(imu_raw, t_imu_raw)

    def _fs(t: np.ndarray) -> float:
        return float(1.0 / np.median(np.diff(t))) if t.size > 1 else 0.0

    meta = {
        "emg_fs": _fs(t_emg),
        "imu_fs": _fs(t_imu),
        "duration_emg": float(t_emg[-1] - t_emg[0]) if t_emg.size else 0.0,
        "duration_imu": float(t_imu[-1] - t_imu[0]) if t_imu.size else 0.0,
        "n_emg_samples": int(emg.shape[0]),
        "n_imu_samples": int(imu.shape[0]),
        "emg_columns": emg_columns,
        "imu_columns": imu_signal_cols,
    }
    return emg, imu, t_emg, t_imu, meta


# --------------------------------------------------------------------------- #
# Modality preprocessing                                                      #
# --------------------------------------------------------------------------- #
def preprocess_emg(
    emg: np.ndarray,
    fs: float,
    bandpass: Optional[Tuple[float, float]] = (20.0, 450.0),
    notch_freq: Optional[float] = 50.0,
    rectify: bool = True,
    envelope_lowpass: Optional[float] = 6.0,
    robust_clip: float = 8.0,
) -> np.ndarray:
    """Standard surface-EMG pipeline. Returns [T, C] float32.

    Steps: notch -> bandpass -> (rectify -> envelope lowpass) -> robust z-score.

    Pass `rectify=False, envelope_lowpass=None` to keep raw filtered EMG (useful
    if you want the temporal encoder to learn its own envelope).
    """
    if emg.ndim != 2:
        raise ValueError(f"Expected [T, C], got {emg.shape}")
    x = emg.astype(np.float64, copy=True)
    nyq = 0.5 * fs
    if notch_freq is not None and 0 < notch_freq < nyq:
        b, a = _notch(notch_freq, fs)
        x = scisig.filtfilt(b, a, x, axis=0)
    if bandpass is not None:
        low = max(bandpass[0], 1.0)
        high = min(bandpass[1], 0.99 * nyq)
        if high > low:
            sos = _butter_bandpass(low, high, fs)
            x = scisig.sosfiltfilt(sos, x, axis=0)
    if rectify:
        x = np.abs(x)
    if envelope_lowpass is not None and 0 < envelope_lowpass < nyq:
        sos = scisig.butter(4, envelope_lowpass / nyq, btype="low", output="sos")
        x = scisig.sosfiltfilt(sos, x, axis=0)
    median = np.median(x, axis=0, keepdims=True)
    mad = np.median(np.abs(x - median), axis=0, keepdims=True)
    scale = np.where(mad > 1e-9, 1.4826 * mad, 1.0)
    x = (x - median) / scale
    if robust_clip > 0:
        x = np.clip(x, -robust_clip, robust_clip)
    return x.astype(np.float32)


def preprocess_imu(
    imu: np.ndarray,
    fs: float,
    lowpass: Optional[float] = 20.0,
    detrend: bool = True,
    robust_clip: float = 8.0,
) -> np.ndarray:
    """IMU cleaning: detrend -> lowpass -> robust z-score. Returns [T, C] float32.

    Each muscle contributes ACC.{X,Y,Z}+GYRO.{X,Y,Z}; we standardize per channel
    independently, leaving inter-axis ratios intact.
    """
    if imu.ndim != 2:
        raise ValueError(f"Expected [T, C], got {imu.shape}")
    x = imu.astype(np.float64, copy=True)
    if detrend:
        x = scisig.detrend(x, axis=0, type="linear")
    nyq = 0.5 * fs
    if lowpass is not None and 0 < lowpass < nyq:
        sos = scisig.butter(4, lowpass / nyq, btype="low", output="sos")
        x = scisig.sosfiltfilt(sos, x, axis=0)
    median = np.median(x, axis=0, keepdims=True)
    mad = np.median(np.abs(x - median), axis=0, keepdims=True)
    scale = np.where(mad > 1e-9, 1.4826 * mad, 1.0)
    x = (x - median) / scale
    if robust_clip > 0:
        x = np.clip(x, -robust_clip, robust_clip)
    return x.astype(np.float32)


# --------------------------------------------------------------------------- #
# Top-level convenience loader                                                #
# --------------------------------------------------------------------------- #
def _truncate_by_duration(arr: np.ndarray, fs: float, duration: float) -> np.ndarray:
    n = int(round(duration * fs))
    if n <= 0:
        return arr[:0]
    return arr[:n] if arr.shape[0] > n else arr


def _zscore_clip(x: np.ndarray, clip: float = 6.0) -> np.ndarray:
    """Robust z-score + symmetric clip (used as final EEG normalisation in the
    BDF path, where filter+resample is already done inside the cache)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.size == 0:
        return x.astype(np.float32, copy=False)
    median = np.median(x, axis=0, keepdims=True)
    mad = np.median(np.abs(x - median), axis=0, keepdims=True)
    scale = np.where(mad > 1e-9, 1.4826 * mad, 1.0)
    x = (x - median) / scale
    if clip > 0:
        x = np.clip(x, -clip, clip)
    return x.astype(np.float32)


def load_bjh_trial(
    eeg_path: Path | str,
    emg_path: Path | str,
    eeg_fs: float = EEG_FS_DEFAULT,
    preprocess: bool = True,
    eeg_kwargs: Optional[dict] = None,
    emg_kwargs: Optional[dict] = None,
    imu_kwargs: Optional[dict] = None,
) -> TriModalSignals:
    """Load + (optionally) preprocess one BJH trial.

    Dispatches on the EEG file extension:

    * ``.bdf`` (real recordings) → MNE bandpass/notch/avg-ref/resample to 500 Hz
      (cached on disk), then cross-correlation sync between the EMG envelope
      and the EEG μ/β envelope, then crop to the common time window.
    * ``.csv`` (legacy / simulated data) → keep the existing pipeline:
      bandpass+robust-z directly on the raw matrix and a simple
      ``min(eeg, emg, imu)`` truncation (no sync — synthetic trials are
      generated aligned at t=0).
    """
    eeg_kwargs = eeg_kwargs or {}
    emg_kwargs = emg_kwargs or {}
    imu_kwargs = imu_kwargs or {}

    eeg_path = Path(eeg_path)
    emg_path = Path(emg_path)
    is_bdf = eeg_path.suffix.lower() == ".bdf"

    emg_raw, imu_raw, t_emg, t_imu, meta = load_emg_imu(emg_path)
    emg_fs = float(meta["emg_fs"])
    imu_fs = float(meta["imu_fs"])

    if is_bdf:
        eeg_proc, eeg_fs_out, eeg_channels = load_eeg_bdf(eeg_path, **eeg_kwargs)
        # EEG is already filtered+resampled inside the cache; we only run
        # robust z-score now, and EMG/IMU still get their full pipeline.
        if preprocess:
            emg_proc = preprocess_emg(emg_raw, emg_fs, **emg_kwargs)
            imu_proc = preprocess_imu(imu_raw, imu_fs, **imu_kwargs)
        else:
            emg_proc = emg_raw.astype(np.float32, copy=False)
            imu_proc = imu_raw.astype(np.float32, copy=False)

        emg_env = _emg_sync_envelope(emg_proc, emg_fs, SYNC_ENV_FS)
        eeg_env = _eeg_mu_beta_envelope(eeg_proc, eeg_fs_out, eeg_channels,
                                        MOTOR_CHANNELS_FOR_SYNC, SYNC_ENV_FS)
        lag_s, peak_corr = estimate_lag_xcorr(emg_env, eeg_env, SYNC_ENV_FS,
                                              SYNC_MAX_LAG_S)
        if not np.isfinite(peak_corr) or peak_corr < SYNC_MIN_PEAK_CORR:
            # Weak correlation — fall back to lag=0 and rely on the common-window
            # crop to absorb the duration mismatch.
            sync_fallback = True
            lag_for_crop = 0.0
        else:
            sync_fallback = False
            lag_for_crop = lag_s

        eeg_c, emg_c, imu_c, t0_eeg, t0_peripheral, common = crop_to_common_window(
            eeg_proc, eeg_fs_out, emg_proc, emg_fs, imu_proc, imu_fs, lag_for_crop,
        )

        if preprocess:
            eeg_out = _zscore_clip(eeg_c)
        else:
            eeg_out = eeg_c.astype(np.float32, copy=False)
        emg_out = emg_c.astype(np.float32, copy=False)
        imu_out = imu_c.astype(np.float32, copy=False)

        return TriModalSignals(
            eeg=eeg_out,
            emg=emg_out,
            imu=imu_out,
            eeg_fs=eeg_fs_out,
            emg_fs=emg_fs,
            imu_fs=imu_fs,
            duration=common,
            metadata={
                "eeg_path": str(eeg_path),
                "emg_path": str(emg_path),
                "eeg_format": "bdf",
                "eeg_channels": list(eeg_channels),
                "raw_eeg_duration": float(eeg_proc.shape[0]) / eeg_fs_out,
                "raw_emg_duration": float(meta["duration_emg"]),
                "raw_imu_duration": float(meta["duration_imu"]),
                "sync_lag_s": float(lag_s),
                "sync_peak_corr": float(peak_corr) if np.isfinite(peak_corr) else float("nan"),
                "sync_fallback": bool(sync_fallback),
                "common_window_s": float(common),
                "t0_eeg_s": float(t0_eeg),
                "t0_peripheral_s": float(t0_peripheral),
                "preprocessed": bool(preprocess),
                **{k: v for k, v in meta.items() if k not in {"emg_columns", "imu_columns"}},
            },
        )

    # ----- Legacy / synthetic CSV path -----
    eeg_raw, eeg_fs_out = load_eeg(eeg_path, expected_fs=eeg_fs)
    eeg_dur = eeg_raw.shape[0] / eeg_fs_out
    imu_dur = float(meta["duration_imu"])
    emg_dur = float(meta["duration_emg"])
    duration = float(min(eeg_dur, imu_dur, emg_dur))

    eeg_clip = _truncate_by_duration(eeg_raw, eeg_fs_out, duration)
    emg_clip = _truncate_by_duration(emg_raw, emg_fs, duration)
    imu_clip = _truncate_by_duration(imu_raw, imu_fs, duration)

    if preprocess:
        eeg_out = preprocess_eeg(eeg_clip, eeg_fs_out, **eeg_kwargs)
        emg_out = preprocess_emg(emg_clip, emg_fs, **emg_kwargs)
        imu_out = preprocess_imu(imu_clip, imu_fs, **imu_kwargs)
    else:
        eeg_out = eeg_clip.astype(np.float32, copy=False)
        emg_out = emg_clip.astype(np.float32, copy=False)
        imu_out = imu_clip.astype(np.float32, copy=False)

    return TriModalSignals(
        eeg=eeg_out,
        emg=emg_out,
        imu=imu_out,
        eeg_fs=eeg_fs_out,
        emg_fs=emg_fs,
        imu_fs=imu_fs,
        duration=duration,
        metadata={
            "eeg_path": str(eeg_path),
            "emg_path": str(emg_path),
            "eeg_format": "csv",
            "raw_eeg_duration": eeg_dur,
            "raw_emg_duration": emg_dur,
            "raw_imu_duration": imu_dur,
            "preprocessed": bool(preprocess),
            **{k: v for k, v in meta.items() if k not in {"emg_columns", "imu_columns"}},
        },
    )
