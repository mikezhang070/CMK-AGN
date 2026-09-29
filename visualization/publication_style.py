"""Shared figures4papers-inspired style for frozen CMK-AGN figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


PALETTE = {
    "proposed": "#0F4D92",
    "secondary": "#3775BA",
    "improvement": "#8BCF8B",
    "comparator": "#B64342",
    "pale_red": "#F6CFCB",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "gold": "#D89B22",
    "neutral": "#CFCECE",
    "grey": "#767676",
    "dark": "#272727",
}


def apply_style(font_size: float = 12, axes_linewidth: float = 1.6) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 1,
        "axes.titleweight": "bold",
        "axes.linewidth": axes_linewidth,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def finalise_axis(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color="#D9D9D9", linewidth=.6, alpha=.85)
    ax.tick_params(width=1.1, length=3.5)


def export(fig, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(f".{suffix}") for suffix in ("png", "pdf", "svg")]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight", pad_inches=.06, facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=.06, facecolor="white")
    fig.savefig(paths[2], bbox_inches="tight", pad_inches=.06, facecolor="white")
    plt.close(fig)
    return paths
