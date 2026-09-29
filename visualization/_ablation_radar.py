"""Source-data-driven radar builder shared by manuscript Figures 10 and 11."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from ._paper_common import export
    from .publication_style import PALETTE, apply_style
except ImportError:
    from _paper_common import export
    from publication_style import PALETTE, apply_style


COLORS = [PALETTE["proposed"], "#F28E2B", PALETTE["teal"], "#D4504C", "#7B6FC2", "#D6A800"]
STYLES = ["-", "--", "-.", ":", "--", ":"]


def _direction_unified(group: pd.DataFrame, metrics: list[str], lower_is_better: set[str]) -> pd.DataFrame:
    means = group.groupby(["variant", "variant_label", "metric"], as_index=False)["value"].mean()
    table = means.pivot(index=["variant", "variant_label"], columns="metric", values="value").reindex(columns=metrics)
    score = table.copy()
    for metric in metrics:
        col = table[metric].astype(float)
        lo, hi = float(col.min()), float(col.max())
        if np.isclose(lo, hi):
            score[metric] = 1.0
        elif metric in lower_is_better:
            score[metric] = (hi - col) / (hi - lo)
        else:
            score[metric] = (col - lo) / (hi - lo)
    return 0.82 + 0.18 * score


def _panel(ax, data: pd.DataFrame, kind: str, task: str, metrics: list[str], metric_labels: list[str], lower: set[str], title: str) -> None:
    group = data.loc[data["kind"].eq(kind) & data["task"].eq(task)].copy()
    radar = _direction_unified(group, metrics, lower)
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    closed = np.r_[angles, angles[0]]
    for i, ((variant, label), row) in enumerate(radar.iterrows()):
        values = np.r_[row.to_numpy(dtype=float), float(row.iloc[0])]
        ax.plot(closed, values, color=COLORS[i % len(COLORS)], linestyle=STYLES[i % len(STYLES)], linewidth=1.8,
                marker="o", markersize=2.8, label=label)
        ax.fill(closed, values, color=COLORS[i % len(COLORS)], alpha=0.07)
    ax.set_xticks(angles, metric_labels)
    ax.set_ylim(0.80, 1.0)
    ax.set_yticks([0.82, 0.88, 0.94, 1.0])
    ax.set_yticklabels([".82", ".88", ".94", "1.00"], fontsize=7)
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=7.5, frameon=False)


def build(root, output, source_name: str, target_name: str, tasks: list[tuple[str, str]], metrics: list[str], metric_labels: list[str], lower: set[str], title: str, source_data_dir=None) -> None:
    source_dir = source_data_dir if source_data_dir is not None else root / "visualization" / "source_data"
    source = source_dir / source_name
    data = pd.read_csv(source)
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    apply_style(9.8)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.7), subplot_kw={"projection": "polar"}, constrained_layout=True)
    panels = [
        ("modality", tasks[0]), ("modality", tasks[1]),
        ("module", tasks[0]), ("module", tasks[1]),
    ]
    for i, (ax, (kind, (task, task_label))) in enumerate(zip(axes.ravel(), panels)):
        heading = f"({'abcd'[i]})  {task_label}"
        _panel(ax, data, kind, task, metrics, metric_labels, lower, heading)
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold", fontsize=15)
    export(fig, output / target_name)
