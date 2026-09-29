"""Tri-modal (EEG / EMG / IMU) alignment for the BJH dataset.

Design notes:
- EEG, EMG and IMU run at different sampling rates and physiologically distinct
  pipelines. Forcing a hard time-warp between EEG (central) and EMG/IMU
  (peripheral) tends to inject phase artefacts. So we keep EEG independent and
  reuse the existing EMG↔IMU adaptive-DTW knot trick (carried over from the
  original ADK-Resample) for the synchronously-recorded peripheral pair only.
- The public output shape mirrors the existing 2-modal `AlignedSignals`
  (channels-major, fixed `output_length`), so downstream code stays identical
  apart from one extra tensor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .wby_dtw import (
    WBYDTWConfig,
    _normalize_channels,
    _resample_time_major,
    _validate_signal,
)


@dataclass
class TriAlignedSignals:
    """Aligned three-modal pair used by training.

    Each tensor is channel-major and fixed-length:
        eeg_aligned shape: [eeg_channels, output_length]
        emg_aligned shape: [emg_channels, output_length]
        imu_aligned shape: [imu_channels, output_length]
    """

    eeg_aligned: np.ndarray
    emg_aligned: np.ndarray
    imu_aligned: np.ndarray
    strategy: str
    metadata: dict[str, float | str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _fit_length_time_major(x: np.ndarray, length: int) -> np.ndarray:
    """Linearly resample [T, C] to [length, C]. Uses linear interpolation."""
    return _resample_time_major(x, length)


def _smooth_1d(x: np.ndarray, window: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size < 3 or window <= 1:
        return x
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(x, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _z1(x: np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    std = float(x.std())
    if std < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - float(x.mean())) / std).astype(np.float32)


def _interp_rows(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.float32)
    src = np.arange(x.shape[0], dtype=np.float32)
    cols = [np.interp(idx, src, x[:, c]).astype(np.float32) for c in range(x.shape[1])]
    return np.stack(cols, axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Strategies                                                                  #
# --------------------------------------------------------------------------- #
def align_tri_resample(
    eeg: np.ndarray,
    emg: np.ndarray,
    imu: np.ndarray,
    config: Optional[WBYDTWConfig] = None,
) -> TriAlignedSignals:
    """Linear resample of each modality to `output_length`, normalized per-channel.

    No cross-modal warping. Cheapest baseline; useful for sanity checks and as a
    fallback when ADK quality filters reject a trial.
    """
    cfg = config or WBYDTWConfig()
    eeg_raw = _validate_signal("eeg", eeg, min_channels=1)
    emg_raw = _validate_signal("emg", emg, min_channels=1)
    imu_raw = _validate_signal("imu", imu, min_channels=1)

    eeg_resampled = _fit_length_time_major(eeg_raw, cfg.output_length)
    emg_resampled = _fit_length_time_major(emg_raw, cfg.output_length)
    imu_resampled = _fit_length_time_major(imu_raw, cfg.output_length)

    eeg_aligned = _normalize_channels(eeg_resampled, cfg.eps, cfg.normalization, cfg.robust_clip).T
    emg_aligned = _normalize_channels(emg_resampled, cfg.eps, cfg.normalization, cfg.robust_clip).T
    imu_aligned = _normalize_channels(imu_resampled, cfg.eps, cfg.normalization, cfg.robust_clip).T

    return TriAlignedSignals(
        eeg_aligned=eeg_aligned.astype(np.float32),
        emg_aligned=emg_aligned.astype(np.float32),
        imu_aligned=imu_aligned.astype(np.float32),
        strategy="tri_resample",
        metadata={"alignment_score": float("nan"), "feature_correlation": float("nan")},
    )


def _band_width(length: int, band_value: float) -> int:
    if band_value < 1:
        return max(1, int(round(band_value * length)))
    return max(1, int(round(band_value)))


def _peripheral_features(
    emg_x: np.ndarray, imu_x: np.ndarray, cfg: WBYDTWConfig
) -> tuple[np.ndarray, np.ndarray]:
    """EMG envelope vs IMU speed feature pair, aligned in length via resample."""
    emg_n = _normalize_channels(emg_x, cfg.eps, cfg.normalization, cfg.robust_clip)
    imu_n = _normalize_channels(imu_x, cfg.eps, cfg.normalization, cfg.robust_clip)
    emg_env = np.mean(np.abs(emg_n), axis=1)
    emg_slope = np.abs(np.diff(emg_env, prepend=emg_env[:1]))
    imu_vel = np.diff(imu_n, axis=0, prepend=imu_n[:1])
    imu_speed = np.linalg.norm(imu_vel, axis=1)
    imu_acc = np.abs(np.diff(imu_speed, prepend=imu_speed[:1]))
    emg_feat = 0.70 * _z1(_smooth_1d(emg_env, 5), cfg.eps) + 0.30 * _z1(_smooth_1d(emg_slope, 5), cfg.eps)
    imu_feat = 0.70 * _z1(_smooth_1d(imu_speed, 5), cfg.eps) + 0.30 * _z1(_smooth_1d(imu_acc, 5), cfg.eps)
    return _z1(emg_feat, cfg.eps), _z1(imu_feat, cfg.eps)


def _dtw_path(a: np.ndarray, b: np.ndarray, cfg: WBYDTWConfig) -> tuple[np.ndarray, float]:
    n = int(len(a))
    m = int(len(b))
    band = _band_width(max(n, m), cfg.band_radius)
    cost = np.full((n, m), np.inf, dtype=np.float32)
    parent = np.full((n, m, 2), -1, dtype=np.int32)
    for i in range(n):
        j0 = max(0, i - band)
        j1 = min(m, i + band + 1)
        for j in range(j0, j1):
            diag_dist = abs(i - j) / float(max(band, 1))
            local = float((a[i] - b[j]) ** 2) + 0.02 * diag_dist * diag_dist
            if i == 0 and j == 0:
                cost[i, j] = local
                continue
            cands = []
            if i > 0 and np.isfinite(cost[i - 1, j]):
                cands.append((cost[i - 1, j] + 0.025, i - 1, j))
            if j > 0 and np.isfinite(cost[i, j - 1]):
                cands.append((cost[i, j - 1] + 0.025, i, j - 1))
            if i > 0 and j > 0 and np.isfinite(cost[i - 1, j - 1]):
                cands.append((cost[i - 1, j - 1], i - 1, j - 1))
            if not cands:
                continue
            prev, pi, pj = min(cands, key=lambda t: t[0])
            cost[i, j] = local + prev
            parent[i, j] = (pi, pj)
    if not np.isfinite(cost[n - 1, m - 1]):
        raise RuntimeError("no valid adaptive DTW-knot path")
    path = [(n - 1, m - 1)]
    i, j = n - 1, m - 1
    while i > 0 or j > 0:
        pi, pj = parent[i, j]
        if pi < 0 or pj < 0:
            raise RuntimeError("broken adaptive DTW-knot path")
        i, j = int(pi), int(pj)
        path.append((i, j))
    path.reverse()
    return np.asarray(path, dtype=np.float32), float(cost[n - 1, m - 1] / max(len(path), 1))


def align_tri_adk_knot(
    eeg: np.ndarray,
    emg: np.ndarray,
    imu: np.ndarray,
    config: Optional[WBYDTWConfig] = None,
    knot_strength: float = 0.01,
    max_shift_frac: float = 0.08,
    min_corr: float = -0.20,
    max_expansion: float = 1.85,
) -> TriAlignedSignals:
    """Adaptive DTW-Knot resample on the EMG↔IMU pair; EEG independently resampled.

    Same idea as the original `align_simple_resampling`, but:
      - DTW is computed only between EMG envelope and IMU speed (synchronous
        peripheral signals)
      - Both EMG and IMU then use the DTW-derived adaptive knot grid
      - EEG bypasses DTW (independent linear resample) — adding an EEG knot grid
        from a non-synchronous DTW would inject phase artefacts.
    """
    cfg = config or WBYDTWConfig()
    eeg_raw = _validate_signal("eeg", eeg, min_channels=1)
    emg_raw = _validate_signal("emg", emg, min_channels=1)
    imu_raw = _validate_signal("imu", imu, min_channels=1)

    out_len = int(cfg.output_length)
    dtw_len = int(np.clip(int(getattr(cfg, "dtw_length", 32)), 8, 96))

    metadata: dict[str, float | str] = {
        "alignment_score": float("nan"),
        "feature_correlation": float("nan"),
        "strategy_detail": "Tri-ADK-Knot (EMG↔IMU only)",
        "dtw_len": float(dtw_len),
        "band_radius": float(cfg.band_radius),
        "knot_strength": float(knot_strength),
        "max_shift_frac": float(max_shift_frac),
    }

    # ------ EMG / IMU adaptive grid via DTW ------
    base_emg_idx = np.linspace(0.0, float(max(emg_raw.shape[0] - 1, 0)), out_len, dtype=np.float32)
    base_imu_idx = np.linspace(0.0, float(max(imu_raw.shape[0] - 1, 0)), out_len, dtype=np.float32)
    emg_idx = base_emg_idx
    imu_idx = base_imu_idx

    if knot_strength > 0.0:
        try:
            emg_coarse = _resample_time_major(emg_raw, dtw_len)
            imu_coarse = _resample_time_major(imu_raw, dtw_len)
            ef, kf = _peripheral_features(emg_coarse, imu_coarse, cfg)
            path, avg_cost = _dtw_path(ef, kf, cfg)
            pi = path[:, 0]
            pj = path[:, 1]

            ac = ef[pi.astype(int)] - ef[pi.astype(int)].mean()
            bc = kf[pj.astype(int)] - kf[pj.astype(int)].mean()
            denom = float(np.linalg.norm(ac) * np.linalg.norm(bc))
            corr = float(np.dot(ac, bc) / denom) if denom > cfg.eps else 0.0

            expansion = float(len(path)) / float(max(dtw_len, 1))
            metadata.update(
                {
                    "feature_correlation": float(corr),
                    "path_expansion": float(expansion),
                    "path_length": float(len(path)),
                    "dtw_avg_cost": float(avg_cost),
                }
            )

            if corr >= min_corr and expansion <= max_expansion:
                canonical = (pi + pj) / float(max(2 * (dtw_len - 1), 1))
                canonical[0] = 0.0
                canonical[-1] = 1.0
                canonical = np.maximum.accumulate(canonical)
                keep = np.ones_like(canonical, dtype=bool)
                keep[1:] = np.diff(canonical) > 1e-6
                canonical = canonical[keep]
                pi = pi[keep]
                pj = pj[keep]

                if canonical.size >= 2:
                    target = np.linspace(0.0, 1.0, out_len, dtype=np.float32)

                    def _adaptive(raw_len: int, path_axis: np.ndarray) -> np.ndarray:
                        base = target * float(max(raw_len - 1, 0))
                        coarse = np.interp(target, canonical, path_axis).astype(np.float32)
                        dtw_idx_local = coarse * float(max(raw_len - 1, 0)) / float(max(dtw_len - 1, 1))
                        max_shift = max_shift_frac * float(max(raw_len - 1, 0))
                        delta = np.clip(dtw_idx_local - base, -max_shift, max_shift)
                        idx = base + knot_strength * delta
                        idx = np.maximum.accumulate(idx)
                        idx = np.clip(idx, 0.0, float(max(raw_len - 1, 0)))
                        return idx.astype(np.float32)

                    emg_idx = _adaptive(emg_raw.shape[0], pi)
                    imu_idx = _adaptive(imu_raw.shape[0], pj)
                    metadata["alignment_score"] = float(corr - 0.1 * avg_cost - 0.05 * max(0.0, expansion - 1.0))
                else:
                    metadata["fallback"] = "too_few_knots"
            else:
                metadata["fallback"] = "quality_filter"
        except Exception as exc:  # noqa: BLE001
            metadata["warning"] = str(exc)[:160]

    emg_resampled = _interp_rows(emg_raw, emg_idx)
    imu_resampled = _interp_rows(imu_raw, imu_idx)
    eeg_resampled = _resample_time_major(eeg_raw, out_len)

    eeg_aligned = _normalize_channels(eeg_resampled, cfg.eps, cfg.normalization, cfg.robust_clip).T
    emg_aligned = _normalize_channels(emg_resampled, cfg.eps, cfg.normalization, cfg.robust_clip).T
    imu_aligned = _normalize_channels(imu_resampled, cfg.eps, cfg.normalization, cfg.robust_clip).T

    return TriAlignedSignals(
        eeg_aligned=eeg_aligned.astype(np.float32),
        emg_aligned=emg_aligned.astype(np.float32),
        imu_aligned=imu_aligned.astype(np.float32),
        strategy="tri_adk_knot",
        metadata=metadata,
    )


def align_by_strategy_tri(
    eeg: np.ndarray,
    emg: np.ndarray,
    imu: np.ndarray,
    strategy: str,
    config: Optional[WBYDTWConfig] = None,
) -> TriAlignedSignals:
    normalized = (strategy or "").lower()
    if normalized in {"none", "tri_none"}:
        cfg = config or WBYDTWConfig()
        # No alignment, just length-fit + normalize.
        return align_tri_resample(eeg, emg, imu, cfg)
    if normalized in {"tri_resample", "resample", "linear", "linear_resample"}:
        return align_tri_resample(eeg, emg, imu, config)
    if normalized in {"tri_adk", "tri_adk_knot", "adk", "adk_resample", "adaptive_dtw_knot", "dtw_knot"}:
        return align_tri_adk_knot(eeg, emg, imu, config)
    if normalized in {"adk_no_dtw", "tri_adk_no_dtw", "no_dtw", "no_wby_dtw"}:
        result = align_tri_adk_knot(eeg, emg, imu, config, knot_strength=0.0)
        result.strategy = "tri_adk_no_dtw"
        return result
    raise ValueError(f"Unknown tri-modal alignment strategy: {strategy}")
