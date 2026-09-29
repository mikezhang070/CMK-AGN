"""Delete a single model's HuggingFace hub cache after its folds finish.

Invoked from ``run_all_models.sh`` between models so a 24 GB disk can host the
4 × 4-bit QLoRA sweep without piling up ~62 GB of base weights.

Usage:
    python -m src.llm.clean_hf_cache --model-id qwen25_3b

Exits 0 even when nothing was cached, so a failure here cannot abort the sweep.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.llm.model_registry import MODEL_REGISTRY


def _fmt_gb(n_bytes: int) -> str:
    return f"{n_bytes / (1024 ** 3):.2f} GB"


def clean(hf_id: str) -> int:
    """Remove all cached revisions of ``hf_id``. Returns bytes freed."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        scan_cache_dir = None

    freed = 0
    if scan_cache_dir is not None:
        info = scan_cache_dir()
        hashes = [
            rev.commit_hash
            for repo in info.repos
            if repo.repo_id == hf_id
            for rev in repo.revisions
        ]
        if hashes:
            strategy = info.delete_revisions(*hashes)
            freed = strategy.expected_freed_size
            strategy.execute()
            return freed

    # Fallback: nuke the hub directory directly.
    org, _, name = hf_id.partition("/")
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{org}--{name}"
    if hub_dir.exists():
        freed = sum(p.stat().st_size for p in hub_dir.rglob("*") if p.is_file())
        shutil.rmtree(hub_dir, ignore_errors=True)
    return freed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", required=True, help="Short id from MODEL_REGISTRY")
    args = ap.parse_args()

    if args.model_id not in MODEL_REGISTRY:
        print(
            f"[clean_hf_cache] WARN: unknown model-id {args.model_id!r}, "
            f"known: {list(MODEL_REGISTRY)}",
            file=sys.stderr,
        )
        return 0

    hf_id = MODEL_REGISTRY[args.model_id]["hf_id"]
    freed = clean(hf_id)
    if freed > 0:
        print(f"[clean_hf_cache] freed {_fmt_gb(freed)} for {hf_id}")
    else:
        print(f"[clean_hf_cache] no cache found for {hf_id} (already clean)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
