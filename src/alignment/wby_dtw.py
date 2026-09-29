from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class WBYDTWConfig:
    output_length: int = 200
    dtw_length: int = 128
    band_radius: float = 0.15
    alpha: float = 0.7
    beta: float = 0.3
    # Keep WBY-DTW stable for small subject-level FMA datasets.
    # The DTW-warped signal is blended with the plain resample signal.
    resample_guided: bool = True
    warp_blend: float = 0.0
    max_path_expansion: float = 1.10
    quality_fallback: float = 0.0
    min_feature_correlation: float = 0.95
    emg_feature_mode: str = "rms"
    feature_combo: str = "rms_env_vel_acc"
    emg_rms_window: int = 9
    kin_smooth_window: int = 5
    slope_weight: float = 0.25
    amplitude_gamma: float = 1.0
    phase_weight: float = 0.35
    event_weight: float = 0.5
    consistency_weight: float = 0.25
    acceleration_weight: float = 0.5
    normalization: str = "zscore"
    robust_clip: float = 5.0
    eps: float = 1e-6


@dataclass
class WBYDTWResult:
    emg_aligned: np.ndarray
    kin_aligned: np.ndarray
    path: np.ndarray
    emg_feature: np.ndarray
    kin_feature: np.ndarray
    forward_cost: float
    reverse_cost: float
    bidirectional_score: float
    feature_correlation: float
    alignment_quality: float
    config: WBYDTWConfig


def _validate_signal(name: str, x: np.ndarray, min_channels: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array [time, channels], got shape={arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(f"{name} must have at least 2 time steps, got shape={arr.shape}")
    if arr.shape[1] < min_channels:
        raise ValueError(f"{name} must have at least {min_channels} channels, got shape={arr.shape}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _zscore_1d(x: np.ndarray, eps: float) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    std = float(arr.std())
    if std < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - float(arr.mean())) / std).astype(np.float32)


def _zscore_channels(x: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return ((x - mean) / std).astype(np.float32)


def _robust_normalize_channels(x: np.ndarray, eps: float, clip_value: float = 5.0) -> np.ndarray:
    median = np.median(x, axis=0, keepdims=True)
    q25 = np.percentile(x, 25, axis=0, keepdims=True)
    q75 = np.percentile(x, 75, axis=0, keepdims=True)
    scale = (q75 - q25) / 1.349
    scale = np.where(scale < eps, 1.0, scale)
    normalized = (x - median) / scale
    if clip_value > 0:
        normalized = np.clip(normalized, -float(clip_value), float(clip_value))
    return normalized.astype(np.float32)


def _normalize_channels(x: np.ndarray, eps: float, mode: str = "zscore", robust_clip: float = 5.0) -> np.ndarray:
    normalized_mode = mode.lower()
    if normalized_mode == "zscore":
        return _zscore_channels(x, eps)
    if normalized_mode == "robust":
        return _robust_normalize_channels(x, eps, robust_clip)
    raise ValueError(f"Unknown channel normalization mode: {mode}")


def _moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(x, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(x, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _resample_time_major(x: np.ndarray, length: int) -> np.ndarray:
    if length <= 1:
        raise ValueError(f"length must be > 1, got {length}")
    if x.shape[0] == length:
        return x.astype(np.float32, copy=True)
    old_grid = np.linspace(0.0, 1.0, num=x.shape[0], dtype=np.float32)
    new_grid = np.linspace(0.0, 1.0, num=length, dtype=np.float32)
    channels = [np.interp(new_grid, old_grid, x[:, channel]) for channel in range(x.shape[1])]
    return np.stack(channels, axis=1).astype(np.float32)


def _resample_path_major(x: np.ndarray, length: int) -> np.ndarray:
    return _resample_time_major(x, length)


def _safe_corr(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    if a.size != b.size or a.size < 2:
        return 0.0
    ac = a - a.mean()
    bc = b - b.mean()
    denom = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    if denom < eps:
        return 0.0
    return float(np.dot(ac, bc) / denom)


def _pick_tokens(feature_combo: str) -> set[str]:
    if not feature_combo:
        return {"rms", "env", "vel", "acc"}
    return {token.strip().lower() for token in str(feature_combo).split("_") if token.strip()}


def _feature_events(feature: np.ndarray, smooth_window: int, eps: float) -> np.ndarray:
    slope = np.abs(np.diff(feature, prepend=feature[:1]))
    event = _moving_average_1d(slope, max(3, smooth_window))
    return _zscore_1d(event, eps)


def compute_emg_envelope(emg: np.ndarray, config: WBYDTWConfig) -> np.ndarray:
    emg_std = _normalize_channels(emg, config.eps, config.normalization, config.robust_clip)
    mode = config.emg_feature_mode.lower()
    if mode == "rms":
        base = np.sqrt(np.mean(emg_std**2, axis=1) + config.eps)
    elif mode == "mean_abs":
        base = np.mean(np.abs(emg_std), axis=1)
    else:
        raise ValueError(f"Unknown EMG feature mode for WBY-DTW: {config.emg_feature_mode}")
    return _zscore_1d(_moving_average_1d(base, config.emg_rms_window), config.eps)


def compute_emg_feature_bundle(emg: np.ndarray, config: WBYDTWConfig) -> dict[str, np.ndarray]:
    primary = compute_emg_envelope(emg, config)
    tokens = _pick_tokens(config.feature_combo)
    bundle = {"primary": primary}
    if "rms" in tokens:
        bundle["rms"] = primary
    if "env" in tokens:
        env = np.mean(np.abs(_normalize_channels(emg, config.eps, config.normalization, config.robust_clip)), axis=1)
        bundle["env"] = _zscore_1d(_moving_average_1d(env, config.emg_rms_window), config.eps)
    if "slope" in tokens or "phase" in tokens:
        bundle["slope"] = _zscore_1d(np.diff(primary, prepend=primary[:1]), config.eps)
    if "event" in tokens or "peak" in tokens:
        bundle["event"] = _feature_events(primary, config.emg_rms_window, config.eps)
    return bundle


def compute_kin_motion_feature(kin: np.ndarray, config: WBYDTWConfig) -> np.ndarray:
    kin_std = _normalize_channels(kin[:, :63], config.eps, config.normalization, config.robust_clip)
    velocity = np.diff(kin_std, axis=0, prepend=kin_std[:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])
    speed = np.linalg.norm(velocity, axis=1)
    accel_mag = np.linalg.norm(acceleration, axis=1)
    displacement = np.linalg.norm(kin_std - np.median(kin_std, axis=0, keepdims=True), axis=1)
    tokens = _pick_tokens(config.feature_combo)
    features = []
    weights = []
    if "vel" in tokens or not tokens:
        features.append(_zscore_1d(_moving_average_1d(speed, config.kin_smooth_window), config.eps))
        weights.append(0.5)
    if "disp" in tokens or "pos" in tokens or not tokens:
        features.append(_zscore_1d(_moving_average_1d(displacement, config.kin_smooth_window), config.eps))
        weights.append(0.3)
    if "acc" in tokens:
        features.append(_zscore_1d(_moving_average_1d(accel_mag, config.kin_smooth_window), config.eps))
        weights.append(float(config.acceleration_weight))
    if not features:
        features.append(_zscore_1d(_moving_average_1d(speed, config.kin_smooth_window), config.eps))
        weights.append(1.0)
    stacked = np.stack(features, axis=0)
    weights_arr = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
    combined = np.sum(stacked * weights_arr, axis=0) / max(float(weights_arr.sum()), config.eps)
    return _zscore_1d(combined, config.eps)


def compute_kin_feature_bundle(kin: np.ndarray, config: WBYDTWConfig) -> dict[str, np.ndarray]:
    primary = compute_kin_motion_feature(kin, config)
    kin_std = _normalize_channels(kin[:, :63], config.eps, config.normalization, config.robust_clip)
    velocity = np.diff(kin_std, axis=0, prepend=kin_std[:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])
    speed = np.linalg.norm(velocity, axis=1)
    accel_mag = np.linalg.norm(acceleration, axis=1)
    bundle = {"primary": primary}
    tokens = _pick_tokens(config.feature_combo)
    if "vel" in tokens or not tokens:
        bundle["vel"] = _zscore_1d(_moving_average_1d(speed, config.kin_smooth_window), config.eps)
    if "acc" in tokens:
        bundle["acc"] = _zscore_1d(_moving_average_1d(accel_mag, config.kin_smooth_window), config.eps)
    if "slope" in tokens or "phase" in tokens:
        bundle["slope"] = _zscore_1d(np.diff(primary, prepend=primary[:1]), config.eps)
    if "event" in tokens or "peak" in tokens:
        bundle["event"] = _feature_events(primary, config.kin_smooth_window, config.eps)
    return bundle


def _band_width(length: int, band_radius: float) -> int:
    if band_radius < 0:
        raise ValueError(f"band_radius must be non-negative, got {band_radius}")
    if band_radius < 1:
        return max(1, int(round(band_radius * length)))
    return int(round(band_radius))


def _local_distance(
    emg_bundle: dict[str, np.ndarray],
    kin_bundle: dict[str, np.ndarray],
    i: int,
    j: int,
    config: WBYDTWConfig,
) -> float:
    emg_weight = max(float(config.alpha), config.eps)
    kin_weight = max(float(config.beta), config.eps)
    norm = emg_weight + kin_weight

    emg_value = float(emg_bundle["primary"][i])
    kin_value = float(kin_bundle["primary"][j])
    shape_distance = (emg_value - kin_value) ** 2

    emg_slope = float(emg_bundle.get("slope", emg_bundle["primary"])[i])
    kin_slope = float(kin_bundle.get("slope", kin_bundle["primary"])[j])
    slope_distance = (emg_slope - kin_slope) ** 2

    emg_amp = abs(emg_value)
    kin_amp = abs(kin_value)
    amplitude_distance = (emg_amp - kin_amp) ** 2

    phase_distance = 0.0
    if "event" in emg_bundle and "event" in kin_bundle:
        phase_distance += (float(emg_bundle["event"][i]) - float(kin_bundle["event"][j])) ** 2
    if "acc" in kin_bundle:
        phase_distance += (emg_amp - abs(float(kin_bundle["acc"][j]))) ** 2

    event_distance = 0.0
    if "env" in emg_bundle and "vel" in kin_bundle:
        event_distance = (float(emg_bundle["env"][i]) - float(kin_bundle["vel"][j])) ** 2

    peak_salience = (emg_weight * emg_amp + kin_weight * kin_amp) / norm
    amplitude_salience = (emg_weight * emg_amp**2 + kin_weight * kin_amp**2) / norm
    peak_weight = 1.0 + float(config.amplitude_gamma) * peak_salience
    amplitude_term = float(config.amplitude_gamma) * (1.0 + amplitude_salience) * amplitude_distance
    consistency_term = float(config.consistency_weight) * ((emg_weight * emg_amp - kin_weight * kin_amp) / norm) ** 2
    return (
        peak_weight * (shape_distance + float(config.slope_weight) * slope_distance)
        + amplitude_term
        + float(config.phase_weight) * phase_distance
        + float(config.event_weight) * event_distance
        + consistency_term
    )


def _dtw_forward(
    emg_bundle: dict[str, np.ndarray],
    kin_bundle: dict[str, np.ndarray],
    config: WBYDTWConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    n = len(emg_bundle["primary"])
    m = len(kin_bundle["primary"])
    if n != m:
        raise ValueError(f"WBY-DTW expects equal feature lengths after resampling, got {n} and {m}")
    band = _band_width(max(n, m), config.band_radius)
    cost = np.full((n, m), np.inf, dtype=np.float32)
    parent = np.full((n, m, 2), -1, dtype=np.int32)

    for i in range(n):
        j_start = max(0, i - band)
        j_end = min(m, i + band + 1)
        for j in range(j_start, j_end):
            local = _local_distance(emg_bundle, kin_bundle, i, j, config)
            if i == 0 and j == 0:
                cost[i, j] = local
                continue
            candidates = []
            if i > 0 and np.isfinite(cost[i - 1, j]):
                candidates.append((cost[i - 1, j], i - 1, j))
            if j > 0 and np.isfinite(cost[i, j - 1]):
                candidates.append((cost[i, j - 1], i, j - 1))
            if i > 0 and j > 0 and np.isfinite(cost[i - 1, j - 1]):
                candidates.append((cost[i - 1, j - 1], i - 1, j - 1))
            if not candidates:
                continue
            previous_cost, previous_i, previous_j = min(candidates, key=lambda item: item[0])
            cost[i, j] = local + previous_cost
            parent[i, j] = (previous_i, previous_j)

    if not np.isfinite(cost[n - 1, m - 1]):
        raise RuntimeError(
            f"No valid WBY-DTW path found. Increase band_radius; current band_radius={config.band_radius}"
        )
    return cost, parent, float(cost[n - 1, m - 1])


def _backtrack(parent: np.ndarray) -> np.ndarray:
    i, j = parent.shape[0] - 1, parent.shape[1] - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        previous_i, previous_j = parent[i, j]
        if previous_i < 0 or previous_j < 0:
            raise RuntimeError(f"Broken DTW path at {(i, j)}")
        i, j = int(previous_i), int(previous_j)
        path.append((i, j))
    path.reverse()
    return np.asarray(path, dtype=np.int32)


def wby_dtw_path(
    emg_bundle: dict[str, np.ndarray],
    kin_bundle: dict[str, np.ndarray],
    config: WBYDTWConfig,
) -> Tuple[np.ndarray, float, float, float]:
    cost, parent, forward_cost = _dtw_forward(emg_bundle, kin_bundle, config)
    path = _backtrack(parent)
    emg_rev = {name: value[::-1] for name, value in emg_bundle.items()}
    kin_rev = {name: value[::-1] for name, value in kin_bundle.items()}
    reverse_cost = _dtw_forward(emg_rev, kin_rev, config)[2]
    path_length = max(len(path), 1)
    forward_score = forward_cost / path_length
    reverse_score = reverse_cost / path_length
    return path, forward_score, reverse_score, 0.5 * (forward_score + reverse_score)


def align_emg_kin_wby_dtw(
    emg: np.ndarray,
    kin: np.ndarray,
    config: Optional[WBYDTWConfig] = None,
) -> WBYDTWResult:
    cfg = config or WBYDTWConfig()
    emg_raw = _validate_signal("emg", emg, min_channels=12)[:, :12]
    kin_raw = _validate_signal("kin", kin, min_channels=63)[:, :63]

    emg_dtw = _resample_time_major(emg_raw, cfg.dtw_length)
    kin_dtw = _resample_time_major(kin_raw, cfg.dtw_length)
    emg_bundle = compute_emg_feature_bundle(emg_dtw, cfg)
    kin_bundle = compute_kin_feature_bundle(kin_dtw, cfg)
    path, forward_score, reverse_score, bidirectional_score = wby_dtw_path(emg_bundle, kin_bundle, cfg)

    feature_correlation = _safe_corr(emg_bundle["primary"][path[:, 0]], kin_bundle["primary"][path[:, 1]], cfg.eps)
    alignment_quality = float(feature_correlation - bidirectional_score)

    # Stable fixed-length baseline, identical in spirit to alignment_mode=resample.
    # This prevents WBY-DTW from over-warping small clinical datasets.
    emg_resampled = _resample_time_major(emg_raw, cfg.output_length)
    kin_resampled = _resample_time_major(kin_raw, cfg.output_length)

    # WBY-DTW path-guided candidate. It is useful only as a correction signal,
    # not as a hard replacement for the stable resample baseline.
    emg_path = emg_dtw[path[:, 0]]
    kin_path = kin_dtw[path[:, 1]]
    emg_warped = _resample_path_major(emg_path, cfg.output_length)
    kin_warped = _resample_path_major(kin_path, cfg.output_length)

    base_blend = float(np.clip(getattr(cfg, "warp_blend", 0.0), 0.0, 0.08))
    path_expansion = float(len(path)) / max(float(cfg.dtw_length), 1.0)
    if bool(getattr(cfg, "resample_guided", True)):
        max_expansion = float(getattr(cfg, "max_path_expansion", 1.10))
        min_quality = float(getattr(cfg, "quality_fallback", 0.0))
        min_corr = float(getattr(cfg, "min_feature_correlation", 0.95))

        if path_expansion > max_expansion or alignment_quality < min_quality or feature_correlation < min_corr:
            blend = 0.0
        else:
            corr_gate = np.clip((feature_correlation - min_corr) / max(1.0 - min_corr, cfg.eps), 0.0, 1.0)
            path_gate = np.clip((max_expansion - path_expansion) / max(max_expansion - 1.0, cfg.eps), 0.0, 1.0)
            blend = float(base_blend * corr_gate * path_gate)

        emg_pre_norm = (1.0 - blend) * emg_resampled + blend * emg_warped
        kin_pre_norm = (1.0 - blend) * kin_resampled + blend * kin_warped
    else:
        blend = base_blend
        emg_pre_norm = (1.0 - blend) * emg_resampled + blend * emg_warped
        kin_pre_norm = (1.0 - blend) * kin_resampled + blend * kin_warped

    emg_aligned = _normalize_channels(
        emg_pre_norm,
        cfg.eps,
        cfg.normalization,
        cfg.robust_clip,
    ).T
    kin_aligned = _normalize_channels(
        kin_pre_norm,
        cfg.eps,
        cfg.normalization,
        cfg.robust_clip,
    ).T

    return WBYDTWResult(
        emg_aligned=emg_aligned.astype(np.float32),
        kin_aligned=kin_aligned.astype(np.float32),
        path=path,
        emg_feature=emg_bundle["primary"],
        kin_feature=kin_bundle["primary"],
        forward_cost=forward_score,
        reverse_cost=reverse_score,
        bidirectional_score=bidirectional_score,
        feature_correlation=feature_correlation,
        alignment_quality=alignment_quality,
        config=cfg,
    )
