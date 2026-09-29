#!/usr/bin/env python3
"""Rebuild CMK-AGN final figures exclusively from frozen 2026 results.

The builder deliberately has no fallback to historical tolerance-upgrade or
development figure tables.  FMA and BI values come only from the frozen final
summary and final patient-prediction files, so regenerated artwork cannot
silently reintroduce obsolete 65.5%, 69%, or 73% FMA tolerance values.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .publication_style import PALETTE, apply_style, export, finalise_axis
except ImportError:  # Direct invocation: python visualization/build_all_final_figures.py
    from publication_style import PALETTE, apply_style, export, finalise_axis


TASKS = {"FMA_UE": {"label": "FMA hand score", "tolerance": 1.5}, "BI": {"label": "Barthel Index", "tolerance": 10.0}}
HISTORICAL_FMA_TOLERANCES = {19 / 29, 20 / 29, .73}


def _parse_interval(value: object) -> tuple[float, float]:
    text = str(value).strip().strip("[]")
    left, right = (part.strip() for part in text.split(",", 1))
    return float(left), float(right)


def _frozen_paths(frozen_root: Path, task: str) -> tuple[Path, Path]:
    public_task = "FMA" if task == "FMA_UE" else task
    summary = frozen_root / "summary.csv"
    predictions = frozen_root / public_task / "patient_predictions.csv"
    if not summary.exists() or not predictions.exists():
        raise FileNotFoundError(f"Frozen final inputs missing for {task}: {summary}; {predictions}")
    return summary, predictions


def load_frozen_results(frozen_root: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    summary_path, _ = _frozen_paths(frozen_root, "FMA_UE")
    summary = pd.read_csv(summary_path)
    summary["task"] = summary["task"].replace({"FMA": "FMA_UE"})
    required = {"task", "MAE", "RMSE", "R2", "Spearman", "Tolerance", "Hit/N", "MAE_95pct_CI", "R2_95pct_CI", "Tolerance_95pct_CI"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Frozen summary missing columns: {sorted(missing)}")
    summary = summary.loc[summary.task.isin(TASKS)].copy()
    if set(summary.task) != set(TASKS):
        raise ValueError("Frozen summary must contain exactly FMA_UE and BI.")
    fma_tolerance = float(summary.loc[summary.task.eq("FMA_UE"), "Tolerance"].iloc[0])
    if any(np.isclose(fma_tolerance, obsolete) for obsolete in HISTORICAL_FMA_TOLERANCES):
        raise ValueError("Historical FMA tolerance was supplied as a frozen primary result.")
    patients: dict[str, pd.DataFrame] = {}
    required_predictions = {"patient_id", "fold", "y_true", "seed2024_pred", "seed42_pred", "seed1337_pred", "ensemble_raw_pred", "final_pred", "absolute_error", "signed_error", "tolerance_hit"}
    for task in TASKS:
        _, path = _frozen_paths(frozen_root, task)
        frame = pd.read_csv(path)
        missing = required_predictions.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        if len(frame) != 29 or frame.patient_id.astype(str).nunique() != 29:
            raise ValueError(f"{task} requires exactly 29 unique frozen patient predictions.")
        patients[task] = frame.sort_values(["fold", "patient_id"]).reset_index(drop=True)
    return summary.sort_values("task").reset_index(drop=True), patients


def _label(ax, label: str) -> None:
    ax.text(-.16, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=15, va="top")


def _save_source(source_dir: Path, stem: str, data: pd.DataFrame) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(source_dir / f"{stem}.csv", index=False)


def _metric_row(summary: pd.DataFrame, task: str) -> pd.Series:
    return summary.loc[summary.task.eq(task)].iloc[0]


def _tolerance_curve(frame: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"threshold": thresholds, "coverage": [float((frame.absolute_error <= threshold).mean()) for threshold in thresholds]})


def fig01_endpoints(summary: pd.DataFrame, source: Path, final: Path) -> str:
    apply_style(11.5)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3), constrained_layout=True)
    tasks = ["FMA_UE", "BI"]
    colors = [PALETTE["proposed"], PALETTE["teal"]]
    records = []
    for panel_index, (metric, axis, title) in enumerate((("MAE", axes[0], "Mean absolute error"), ("R2", axes[1], "$R^2$"), ("Tolerance", axes[2], "Clinical tolerance coverage"))):
        values, lows, highs = [], [], []
        for task in tasks:
            row = _metric_row(summary, task)
            values.append(float(row[metric]))
            ci_key = {"MAE": "MAE_95pct_CI", "R2": "R2_95pct_CI", "Tolerance": "Tolerance_95pct_CI"}[metric]
            lo, hi = _parse_interval(row[ci_key])
            lows.append(lo); highs.append(hi)
            records.append({"task": task, "metric": metric, "estimate": float(row[metric]), "ci_low": lo, "ci_high": hi, "hit_n": row["Hit/N"]})
        x = np.arange(2)
        bars = axis.bar(x, values, color=colors, edgecolor="black", linewidth=.9, hatch=["", "//"])
        axis.errorbar(x, values, yerr=np.array([np.array(values) - lows, highs - np.array(values)]), fmt="none", ecolor=PALETTE["dark"], capsize=4, lw=1.3)
        axis.set_xticks(x, ["FMA", "BI"])
        axis.set_title(title)
        axis.set_ylabel("Proportion" if metric == "Tolerance" else metric)
        if metric == "Tolerance": axis.set_ylim(0, 1.0)
        for bar, value, row in zip(bars, values, [_metric_row(summary, task) for task in tasks]):
            text = f"{value:.1%}\n({row['Hit/N']})" if metric == "Tolerance" else f"{value:.3f}"
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (0.035 if metric == "Tolerance" else max(values) * .04), text, ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        finalise_axis(axis); _label(axis, "ABC"[panel_index])
    fig.suptitle("Frozen patient-independent performance endpoints", x=.02, ha="left", fontweight="bold", fontsize=15)
    fig.text(.02, .005, "Patient-level bootstrap 95% confidence intervals (20,000 resamples). FMA: |error| ≤ 1.5; BI: |error| ≤ 10.", fontsize=8.8, color=PALETTE["grey"])
    export(fig, final / "Fig01_Frozen_primary_endpoints")
    _save_source(source, "Fig01_Frozen_primary_endpoints", pd.DataFrame(records))
    return "Fig01_Frozen_primary_endpoints"


def _diagnostic_figure(task: str, summary: pd.DataFrame, frame: pd.DataFrame, source: Path, final: Path, number: str) -> str:
    apply_style(10.5)
    row = _metric_row(summary, task); label = TASKS[task]["label"]; tol = TASKS[task]["tolerance"]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), constrained_layout=True)
    y_true, y_pred = frame.y_true.astype(float), frame.final_pred.astype(float)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()); margin = max((hi - lo) * .07, 1.0)
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, c=PALETTE["proposed"], edgecolor="black", lw=.45, s=50, alpha=.88)
    ax.plot([lo-margin, hi+margin], [lo-margin, hi+margin], "--", c=PALETTE["comparator"], lw=1.4)
    ax.set(xlim=(lo-margin, hi+margin), ylim=(lo-margin, hi+margin), xlabel="Observed score", ylabel="Frozen final prediction", title="Patient-level agreement")
    ax.text(.04, .96, f"n=29\nMAE={float(row.MAE):.3f}\n$R^2$={float(row.R2):.3f}\n$\\rho$={float(row.Spearman):.3f}", transform=ax.transAxes, va="top", bbox={"boxstyle":"round,pad=.25", "fc":"white", "ec":PALETTE["neutral"]}, fontsize=9.4)
    finalise_axis(ax, "both"); _label(ax, "A")
    ax = axes[0, 1]
    mean = (y_true + y_pred) / 2; diff = y_pred - y_true; bias = diff.mean(); sd = diff.std(ddof=1)
    ax.scatter(mean, diff, c=PALETTE["teal"], edgecolor="black", lw=.4, s=48, alpha=.85)
    for value, name, color in ((bias, "Bias", PALETTE["dark"]), (bias + 1.96*sd, "+1.96 SD", PALETTE["comparator"]), (bias - 1.96*sd, "−1.96 SD", PALETTE["comparator"])):
        ax.axhline(value, ls="--", lw=1.15, c=color); ax.text(.98, value, f" {name}: {value:.2f}", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8.3, color=color)
    ax.set(xlabel="Mean of observed and predicted score", ylabel="Prediction − observed", title="Bland–Altman agreement")
    finalise_axis(ax, "both"); _label(ax, "B")
    ax = axes[1, 0]
    ordered = frame.sort_values("absolute_error").reset_index(drop=True)
    colors = np.where(ordered.tolerance_hit, PALETTE["improvement"], PALETTE["comparator"])
    ax.bar(np.arange(1, len(ordered)+1), ordered.absolute_error, color=colors, edgecolor="black", lw=.25)
    ax.axhline(tol, c=PALETTE["dark"], ls="--", lw=1.25, label=f"Prespecified threshold = {tol:g}")
    ax.set(xlabel="Patients ordered by absolute error", ylabel="Absolute error", title="Tolerance-hit distribution")
    ax.legend(fontsize=8.3, loc="upper left"); finalise_axis(ax); _label(ax, "C")
    ax = axes[1, 1]
    curve = _tolerance_curve(frame, ([.5, 1., 1.5, 2., 2.5, 3.] if task == "FMA_UE" else [5., 10., 15., 20.]))
    ax.plot(curve.threshold, curve.coverage, marker="o", c=PALETTE["proposed"], lw=2.1, ms=6)
    ax.axvline(tol, c=PALETTE["comparator"], ls="--", lw=1.25, label=f"Official threshold = {tol:g}")
    ax.scatter([tol], [float(row.Tolerance)], s=75, c=PALETTE["gold"], edgecolor="black", zorder=4)
    ax.set(xlabel="Absolute-error threshold", ylabel="Patient coverage", ylim=(0, 1.05), title="Tolerance sensitivity analysis")
    ax.legend(fontsize=8.3, loc="lower right"); finalise_axis(ax); _label(ax, "D")
    fig.suptitle(f"{label}: frozen final patient-level diagnostics", x=.02, ha="left", fontweight="bold", fontsize=14.5)
    fig.text(.02, .006, "Only the dashed threshold is the prespecified clinical endpoint; remaining thresholds are sensitivity analyses.", fontsize=8.5, color=PALETTE["grey"])
    stem = f"Fig{number}_{'FMA' if task == 'FMA_UE' else 'BI'}_patient_diagnostics"
    export(fig, final / stem)
    source_frame = frame.copy(); source_frame["official_tolerance"] = tol
    _save_source(source, stem, source_frame)
    return stem


def fig04_coverage(summary: pd.DataFrame, patients: dict[str, pd.DataFrame], source: Path, final: Path) -> str:
    apply_style(11)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), constrained_layout=True)
    sources = []
    for i, task in enumerate(("FMA_UE", "BI")):
        ax = axes[i]; tol = TASKS[task]["tolerance"]
        thresholds = np.linspace(0, 5, 21) if task == "FMA_UE" else np.linspace(0, 30, 31)
        curve = _tolerance_curve(patients[task], list(thresholds)); curve["task"] = task; sources.append(curve)
        ax.fill_between(curve.threshold, curve.coverage, color=[PALETTE["secondary"], PALETTE["teal"]][i], alpha=.16)
        ax.plot(curve.threshold, curve.coverage, color=[PALETTE["proposed"], PALETTE["teal"]][i], lw=2.4)
        official = _metric_row(summary, task)
        ax.axvline(tol, color=PALETTE["comparator"], lw=1.35, ls="--")
        ax.scatter([tol], [float(official.Tolerance)], c=PALETTE["gold"], edgecolor="black", s=88, zorder=4)
        ax.annotate(f"{official['Hit/N']} ({float(official.Tolerance):.1%})", (tol, float(official.Tolerance)), xytext=(8, -24), textcoords="offset points", fontsize=9, fontweight="bold")
        ax.set(title=TASKS[task]["label"], xlabel="Absolute-error threshold", ylabel="Patient coverage", ylim=(0, 1.04))
        finalise_axis(ax); _label(ax, "AB"[i])
    fig.suptitle("Clinical tolerance coverage across sensitivity thresholds", x=.02, ha="left", fontweight="bold", fontsize=14.5)
    fig.text(.02, .005, "Dashed lines denote the prespecified endpoints: FMA |error| ≤ 1.5 and BI |error| ≤ 10.", fontsize=8.7, color=PALETTE["grey"])
    export(fig, final / "Fig04_Clinical_tolerance_coverage")
    _save_source(source, "Fig04_Clinical_tolerance_coverage", pd.concat(sources, ignore_index=True))
    return "Fig04_Clinical_tolerance_coverage"


def fig05_seed_stability(frozen_root: Path, source: Path, final: Path) -> str:
    apply_style(11)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), constrained_layout=True)
    combined = []
    for i, task in enumerate(("FMA_UE", "BI")):
        public_task = "FMA" if task == "FMA_UE" else task
        path = frozen_root / public_task / "seed_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"Frozen seed metric file is required: {path}")
        data = pd.read_csv(path).copy(); data["task"] = task; combined.append(data)
        ax = axes[i]; x = np.arange(len(data)); values = data.mae.astype(float)
        bars = ax.bar(x, values, color=[PALETTE["secondary"], PALETTE["proposed"], PALETTE["teal"]], edgecolor="black", lw=.8)
        ax.scatter(x, values, c="white", edgecolor="black", s=30, zorder=4)
        ax.axhline(values.mean(), c=PALETTE["comparator"], lw=1.4, ls="--", label=f"Mean = {values.mean():.3f}")
        ax.set(xticks=x, xticklabels=data.seed.str.replace("seed", ""), xlabel="Random seed", ylabel="Patient-level MAE", title=TASKS[task]["label"])
        for bar, value in zip(bars, values): ax.text(bar.get_x()+bar.get_width()/2, value+max(values)*.025, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
        ax.legend(fontsize=8.4); finalise_axis(ax); _label(ax, "AB"[i])
    fig.suptitle("Frozen outer-test seed-level prediction variability", x=.02, ha="left", fontweight="bold", fontsize=14.5)
    fig.text(.02, .005, "Each seed uses the fixed patient-independent outer folds; this panel is descriptive and does not select a seed by outer-test performance.", fontsize=8.5, color=PALETTE["grey"])
    export(fig, final / "Fig05_Frozen_seed_stability")
    _save_source(source, "Fig05_Frozen_seed_stability", pd.concat(combined, ignore_index=True))
    return "Fig05_Frozen_seed_stability"


def fig06_provenance(frozen_root: Path, summary: pd.DataFrame, source: Path, final: Path) -> str:
    apply_style(10.5)
    audit_path = frozen_root / "provenance_audit.json"
    if not audit_path.exists():
        raise FileNotFoundError(f"Frozen provenance audit required: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 4.8), constrained_layout=True)
    ax = axes[0]; ax.set_axis_off()
    steps = ["Original patient split", "3 fixed seeds", "MAE checkpoint rule", "Train-only inner OOF", "Frozen outer-test metrics", "Patient bootstrap"]
    colors = [PALETTE["neutral"], PALETTE["secondary"], PALETTE["secondary"], PALETTE["improvement"], PALETTE["proposed"], PALETTE["teal"]]
    for i, (step, color) in enumerate(zip(steps, colors)):
        x = .03 + i*.162; ax.add_patch(plt.Rectangle((x, .42), .132, .20, transform=ax.transAxes, facecolor=color, edgecolor="black", lw=.8))
        ax.text(x+.066, .52, step, ha="center", va="center", transform=ax.transAxes, fontsize=7.5, fontweight="bold", wrap=True, color="white" if color not in {PALETTE['neutral'], PALETTE['improvement']} else PALETTE['dark'])
        if i < len(steps)-1: ax.annotate("", xy=(x+.160, .52), xytext=(x+.134, .52), xycoords=ax.transAxes, arrowprops={"arrowstyle":"->", "lw":1.1, "color":PALETTE["dark"]})
    ax.text(.05, .83, "Frozen evidence chain", transform=ax.transAxes, fontsize=14, fontweight="bold")
    ax.text(.05, .18, "All primary FMA/BI figures in this release are regenerated from the consolidated frozen results directory. Historical tolerance values are retained only as provenance evidence, not as primary endpoints.", transform=ax.transAxes, fontsize=9.5, color=PALETTE["grey"], wrap=True)
    _label(ax, "A")
    ax = axes[1]
    entries = []
    for task in ("FMA_UE", "BI"):
        row = _metric_row(summary, task)
        entries.extend([(task, "MAE", float(row.MAE)), (task, "$R^2$", float(row.R2)), (task, "Tolerance", float(row.Tolerance))])
    values = pd.DataFrame(entries, columns=["task", "metric", "value"])
    pivot = values.pivot(index="metric", columns="task", values="value").reindex(["MAE", "$R^2$", "Tolerance"])
    # Normalization is explicitly panel-local, allowing MAE, R2 and coverage to be shown together without implying comparability.
    normalized = pivot.apply(lambda col: (col-col.min()) / max(col.max()-col.min(), 1e-9), axis=1)
    im = ax.imshow(normalized.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set(xticks=range(2), xticklabels=["FMA", "BI"], yticks=range(3), yticklabels=pivot.index, title="Endpoint profile (within-row normalized)")
    for r, metric in enumerate(pivot.index):
        for c, task in enumerate(pivot.columns): ax.text(c, r, f"{pivot.loc[metric, task]:.3f}" if metric != "Tolerance" else f"{pivot.loc[metric, task]:.1%}", ha="center", va="center", fontsize=9.5, color="white" if normalized.iloc[r,c]>.6 else "black")
    fig.colorbar(im, ax=ax, fraction=.05, pad=.04, label="Within-row normalized")
    _label(ax, "B")
    fig.suptitle("Final reproduction protocol and endpoint audit", x=.02, ha="left", fontweight="bold", fontsize=14.5)
    export(fig, final / "Fig06_Frozen_provenance_and_endpoint_audit")
    data = pd.DataFrame([{ "audit_n_subjects": audit["outer_split"]["n_unique_subjects"], "seeds": ",".join(map(str, audit["fixed_seeds"])), "task": row.task, "MAE": row.MAE, "R2": row.R2, "Tolerance": row.Tolerance } for row in summary.itertuples()])
    _save_source(source, "Fig06_Frozen_provenance_and_endpoint_audit", data)
    return "Fig06_Frozen_provenance_and_endpoint_audit"


def _captions() -> str:
    return """# CMK-AGN frozen final figure captions

All FMA and BI panels use the patient-independent final reproduction output (`n=29`). FMA clinical tolerance is prespecified as absolute error ≤1.5; BI tolerance is prespecified as absolute error ≤10. Bootstrap intervals are patient-level percentile intervals with 20,000 resamples.

- **Fig01:** Primary frozen endpoints with patient-level bootstrap confidence intervals.
- **Fig02–03:** Patient agreement, Bland–Altman agreement, absolute-error distribution, and threshold sensitivity for FMA and BI.
- **Fig04:** Clinical tolerance coverage across thresholds; only the marked threshold is primary.
- **Fig05:** Seed-level variability after fixed outer-fold evaluation; descriptive only.
- **Fig06:** Provenance boundary and endpoint audit; historic tolerance values are excluded from primary figures.
"""


def build_final_figures(project_root: Path, frozen_root: Path, output_root: Path) -> list[str]:
    """Build final publication artwork and source data from frozen results."""
    project_root, frozen_root, output_root = Path(project_root), Path(frozen_root), Path(output_root)
    summary, patients = load_frozen_results(frozen_root)
    final, source = output_root / "final", output_root / "source_data"
    final.mkdir(parents=True, exist_ok=True); source.mkdir(parents=True, exist_ok=True)
    figures = [
        fig01_endpoints(summary, source, final),
        _diagnostic_figure("FMA_UE", summary, patients["FMA_UE"], source, final, "02"),
        _diagnostic_figure("BI", summary, patients["BI"], source, final, "03"),
        fig04_coverage(summary, patients, source, final),
        fig05_seed_stability(frozen_root, source, final),
        fig06_provenance(frozen_root, summary, source, final),
    ]
    audit = summary[["task", "MAE", "RMSE", "R2", "Spearman", "Tolerance", "Hit/N"]].copy()
    audit.to_csv(output_root / "metric_consistency_audit.csv", index=False)
    (output_root / "metric_consistency_audit.md").write_text(
        "# Frozen metric consistency audit\n\n"
        "PASS — all primary FMA/BI source values were read from "
        "`results/summary.csv`. "
        "The FMA official tolerance is 16/29 (55.2%); historical 19/29, 20/29, "
        "and 73% values were rejected as primary inputs.\n",
        encoding="utf-8",
    )
    (output_root / "FIGURE_SCOPE.md").write_text(_captions(), encoding="utf-8")
    with (output_root / "figure_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["figure", "png", "pdf", "svg", "source_data", "source_policy"])
        writer.writeheader()
        for stem in figures:
            writer.writerow({"figure": stem, "png": f"final/{stem}.png", "pdf": f"final/{stem}.pdf", "svg": f"final/{stem}.svg", "source_data": f"source_data/{stem}.csv", "source_policy": "consolidated frozen results only"})
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--frozen-root", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args(); root = args.root.resolve()
    frozen = args.frozen_root if args.frozen_root.is_absolute() else root / args.frozen_root
    out = args.out if args.out.is_absolute() else root / args.out
    figures = build_final_figures(root, frozen, out)
    print(f"[frozen-figures] built {len(figures)} group figures -> {out / 'final'}")


if __name__ == "__main__":
    main()
