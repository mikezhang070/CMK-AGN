"""Aggregate per-fold BLEU/ROUGE reports across all 4 fine-tuned LLMs and
pick the winner by mean ``<group>.<metric>`` (default: ``all.char_bleu4``).

Reads:
    results/llm/evaluation/<model_id>/fold{1,2,3}.json   (from evaluate.py)

Writes:
    results/llm/evaluation/summary.csv      one row per (model_id, fold)
    results/llm/evaluation/mean.csv         one row per model with mean & std
    results/llm/evaluation/best_model.json  {best_model_id, best_hf_id, ...}

Example:
    python -m src.llm.select_best_model \
        --reports-glob "results/llm/evaluation/*/fold*.json" \
        --metric all.char_bleu4 \
        --out outputs/llm/comparison/best_model.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from .model_registry import MODEL_REGISTRY


_FOLD_RE = re.compile(r"fold(\d+)\.json$")


def _parse_path(p: Path) -> Tuple[str, int]:
    """Return (model_id, fold_idx) from .../<model_id>/eval_report_fold<k>.json."""
    m = _FOLD_RE.search(p.name)
    if not m:
        raise ValueError(f"Cannot parse fold from {p}")
    return p.parent.name, int(m.group(1))


def _pick_metric(report: dict, dotted: str) -> float:
    """``'all.char_bleu4'`` -> report['all']['char_bleu4']."""
    group, key = dotted.split(".", 1)
    try:
        return float(report[group][key])
    except (KeyError, TypeError) as e:
        raise KeyError(
            f"Metric {dotted!r} not found in report (groups={list(report)})"
        ) from e


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    mu = sum(xs) / len(xs)
    if len(xs) == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return mu, math.sqrt(var)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pick the best LLM across folds.")
    ap.add_argument("--reports-glob", required=True,
                    help='e.g. "results/llm/evaluation/*/fold*.json"')
    ap.add_argument("--metric", default="all.char_bleu4",
                    help="Dotted path 'group.metric' (default: all.char_bleu4).")
    ap.add_argument("--extra", nargs="*",
                    default=["all.rougeL_f"],
                    help="Additional metrics to surface in the report.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Path for best_model.json. CSVs are written next to it.")
    args = ap.parse_args()

    paths = sorted(Path(p) for p in glob.glob(args.reports_glob))
    if not paths:
        raise SystemExit(f"No reports matched {args.reports_glob!r}")

    rows: List[Dict[str, object]] = []
    grouped: Dict[str, List[float]] = defaultdict(list)
    extras: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for path in paths:
        model_id, fold = _parse_path(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        primary = _pick_metric(report, args.metric)
        grouped[model_id].append(primary)
        row: Dict[str, object] = {
            "model_id": model_id,
            "fold": fold,
            args.metric: round(primary, 4),
        }
        for ex in args.extra:
            try:
                v = _pick_metric(report, ex)
            except KeyError:
                v = float("nan")
            row[ex] = round(v, 4)
            extras[model_id][ex].append(v)
        rows.append(row)

    rows.sort(key=lambda r: (r["model_id"], r["fold"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)

    summary_csv = args.out.parent / "summary.csv"
    fieldnames = ["model_id", "fold", args.metric] + list(args.extra)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    leaderboard: List[Tuple[str, float, float, Dict[str, Tuple[float, float]]]] = []
    for model_id, vals in grouped.items():
        mu, sd = _mean_std(vals)
        ex_stats = {ex: _mean_std([v for v in extras[model_id][ex] if not math.isnan(v)])
                    for ex in args.extra}
        leaderboard.append((model_id, mu, sd, ex_stats))
    leaderboard.sort(key=lambda x: x[1], reverse=True)

    mean_csv = args.out.parent / "mean.csv"
    with mean_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        head = ["model_id", f"{args.metric}_mean", f"{args.metric}_std"]
        for ex in args.extra:
            head += [f"{ex}_mean", f"{ex}_std"]
        w.writerow(head)
        for model_id, mu, sd, ex_stats in leaderboard:
            row_out = [model_id, f"{mu:.4f}", f"{sd:.4f}"]
            for ex in args.extra:
                emu, esd = ex_stats[ex]
                row_out += [f"{emu:.4f}", f"{esd:.4f}"]
            w.writerow(row_out)

    best_id, best_mu, best_sd, _ = leaderboard[0]
    runner_id, runner_mu = (leaderboard[1][0], leaderboard[1][1]) if len(leaderboard) > 1 else ("", 0.0)
    best_hf_id = MODEL_REGISTRY[best_id]["hf_id"] if best_id in MODEL_REGISTRY else ""
    payload = {
        "metric": args.metric,
        "best_model_id": best_id,
        "best_hf_id": best_hf_id,
        f"best_{args.metric}_mean": round(best_mu, 4),
        f"best_{args.metric}_std": round(best_sd, 4),
        "runner_up_id": runner_id,
        "margin_over_runner_up": round(best_mu - runner_mu, 4),
        "n_folds": len(grouped[best_id]),
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"\n# LLM leaderboard (metric={args.metric})\n")
    head = f"| model_id | {args.metric} mean ± std |"
    sep  = "|---|---|"
    for ex in args.extra:
        head += f" {ex} mean |"
        sep += "---|"
    print(head)
    print(sep)
    for model_id, mu, sd, ex_stats in leaderboard:
        line = f"| `{model_id}` | **{mu:.2f}** ± {sd:.2f} |"
        for ex in args.extra:
            emu, _ = ex_stats[ex]
            line += f" {emu:.2f} |"
        print(line)
    print(f"\n→ winner: **{best_id}** ({best_hf_id})")
    print(f"→ wrote {summary_csv}")
    print(f"→ wrote {mean_csv}")
    print(f"→ wrote {args.out}")


if __name__ == "__main__":
    main()
