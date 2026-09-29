"""Shared utilities for the manuscript-ordered figure scripts."""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PAPER_FIGURES = ROOT / "figures"
SOURCE_DATA = ROOT / "visualization" / "source_data"


def parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--root", type=Path, default=ROOT, help="Release root.")
    ap.add_argument(
        "--output",
        type=Path,
        default=PAPER_FIGURES / "rebuilt",
        help="Destination directory; defaults to figures/rebuilt.",
    )
    ap.add_argument(
        "--source-data-dir",
        type=Path,
        default=None,
        help="Optional alternate source-data directory (used by regenerated ablation figures).",
    )
    return ap


def export(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        (".png", {"dpi": 600}),
        (".pdf", {}),
        (".svg", {}),
    ):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def copy_frozen_asset(root: Path, output: Path, stem: str) -> list[Path]:
    """Copy a frozen, manuscript-embedded asset without claiming regeneration."""
    source_dir = root / "figures"
    source = source_dir / f"{stem}.png"
    if not source.exists():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    copied = []
    for suffix in (".png", ".pdf", ".svg"):
        member = source_dir / f"{stem}{suffix}"
        if not member.exists():
            continue
        target = output / member.name
        if member.resolve() != target.resolve():
            shutil.copy2(member, target)
        copied.append(target)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    print(f"verified frozen asset: {source} sha256={digest}")
    return copied


def copy_family(source_dir: Path, source_stem: str, output: Path, target_stem: str) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for suffix in (".png", ".pdf", ".svg"):
        source = source_dir / f"{source_stem}{suffix}"
        if not source.exists():
            raise FileNotFoundError(source)
        target = output / f"{target_stem}{suffix}"
        shutil.copy2(source, target)
        copied.append(target)
    return copied
