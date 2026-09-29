"""Name -> builder registry for baselines.

Each baseline module imports ``register`` and decorates its top-level builder
factory. The factory signature is::

    @register("xgboost")
    def build_xgboost(spec: TaskSpec, args) -> MLBaseline | nn.Module: ...

The CLI in :mod:`baselines.train_baseline` dispatches via ``REGISTRY[args.model]``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

Builder = Callable[..., Any]
REGISTRY: Dict[str, Builder] = {}


def register(name: str) -> Callable[[Builder], Builder]:
    def deco(fn: Builder) -> Builder:
        if name in REGISTRY:
            raise KeyError(f"Baseline {name!r} already registered.")
        REGISTRY[name] = fn
        return fn
    return deco


def list_models() -> list[str]:
    return sorted(REGISTRY)
