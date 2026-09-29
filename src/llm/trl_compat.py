"""Small compatibility helpers for the two TRL generations used on the server."""
from __future__ import annotations


def select_trl_mode(*, has_sft_config: bool, has_completion_collator: bool) -> str:
    """Choose the available completion-only SFT implementation.

    TRL >= 1 uses ``SFTConfig(completion_only_loss=True)``.  Older TRL
    releases expose ``DataCollatorForCompletionOnlyLM`` instead.  Keeping the
    decision independent of TRL imports makes both environments testable.
    """
    if has_sft_config:
        return "modern_completion_only"
    if has_completion_collator:
        return "legacy_collator"
    raise RuntimeError(
        "Installed TRL exposes neither SFTConfig nor "
        "DataCollatorForCompletionOnlyLM; install a supported TRL version."
    )


def use_gradient_checkpointing(model_cfg: dict) -> bool:
    """Return the model-specific memory setting, enabled by default."""
    return not bool(model_cfg.get("disable_gradient_checkpointing", False))
