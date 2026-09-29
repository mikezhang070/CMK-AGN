"""Paper Figure 9: exploratory row-normalized ordinal confusion matrices."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from ._paper_common import export, parser
    from .publication_style import PALETTE, apply_style
except ImportError:
    from _paper_common import export, parser
    from publication_style import PALETTE, apply_style


def _matrix(frame: pd.DataFrame, task: str, labels: list[str]) -> np.ndarray:
    part = frame.loc[frame["task"].eq(task), labels].apply(pd.to_numeric, errors="coerce").fillna(0)
    matrix = part.to_numpy(dtype=float)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError(f"{task}: expected {len(labels)}x{len(labels)}, got {matrix.shape}")
    return matrix


def build(root, output) -> None:
    source = root / "visualization" / "source_data" / "Fig09_Ordinal_confusion_matrices.csv"
    data = pd.read_csv(source, dtype=str)
    apply_style(10.5)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), constrained_layout=True)
    specs = [
        ("hand_tone", ["0", "1", "1+", "2", "3", "4"], "Hand muscle tone"),
        ("hand_function", ["1", "2", "3", "4", "5", "6"], "Brunnstrom hand stage"),
    ]
    for panel, (ax, (task, labels, title)) in enumerate(zip(axes, specs)):
        counts = _matrix(data, task, labels)
        totals = counts.sum(axis=1, keepdims=True)
        normalized = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
        im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        for r in range(len(labels)):
            for c in range(len(labels)):
                color = "white" if normalized[r, c] > 0.55 else PALETTE["dark"]
                ax.text(c, r, f"{normalized[r,c]:.2f}\n({int(counts[r,c])})", ha="center", va="center", fontsize=7.5, color=color)
        ax.set(xticks=range(len(labels)), xticklabels=labels, yticks=range(len(labels)), yticklabels=labels,
               xlabel="Predicted grade/stage", ylabel="Observed grade/stage", title=title)
        ax.text(-0.16, 1.08, "AB"[panel], transform=ax.transAxes, fontweight="bold", fontsize=14)
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03, label="Row-normalized proportion")
    fig.suptitle("Exploratory ordinal confusion matrices", x=0.02, ha="left", fontweight="bold", fontsize=14.5)
    export(fig, output / "Fig09_Ordinal_confusion_matrices")


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    build(args.root.resolve(), args.output.resolve())
