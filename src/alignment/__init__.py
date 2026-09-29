from .wby_dtw import WBYDTWConfig, WBYDTWResult, align_emg_kin_wby_dtw
# `strategies` is only required by the legacy 2-modal scripts; the tri-modal
# pipeline does not need it. Tolerate its absence so the package still imports.
try:
    from .strategies import (  # noqa: F401
        AlignedSignals,
        align_by_strategy,
        align_no_alignment,
        align_old_alignment,
        align_simple_resampling,
    )
except ModuleNotFoundError:
    AlignedSignals = None  # type: ignore[assignment]
    align_by_strategy = None  # type: ignore[assignment]
    align_no_alignment = None  # type: ignore[assignment]
    align_old_alignment = None  # type: ignore[assignment]
    align_simple_resampling = None  # type: ignore[assignment]
from .tri_strategies import (
    TriAlignedSignals,
    align_by_strategy_tri,
    align_tri_adk_knot,
    align_tri_resample,
)

__all__ = [
    "AlignedSignals",
    "TriAlignedSignals",
    "WBYDTWConfig",
    "WBYDTWResult",
    "align_by_strategy",
    "align_by_strategy_tri",
    "align_emg_kin_wby_dtw",
    "align_no_alignment",
    "align_old_alignment",
    "align_simple_resampling",
    "align_tri_adk_knot",
    "align_tri_resample",
]
