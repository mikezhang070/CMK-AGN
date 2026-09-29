"""BLEU / ROUGE evaluation for generated Chinese rehab text.

Reads the JSON written by generate.py and reports:
  - char-level BLEU (sacrebleu, tokenize="zh"; both corpus & per-n-gram averages)
  - word-level BLEU on jieba tokens (nltk corpus_bleu with smoothing)
  - ROUGE-1 / ROUGE-2 / ROUGE-L (rouge-chinese)

All reported rows must have a traceable human-authored reference report.

Example:
    python -m src.llm.evaluate \
        --pred outputs/llm/fold1_test.json \
        --out  outputs/llm/eval_report_fold1.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


import jieba
import sacrebleu
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from rouge_chinese import Rouge


def _avg(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _char_bleu_per_n(hyps: List[str], refs: List[str]) -> Dict[str, float]:
    """Per-n-gram BLEU averaged over sentences, char-tokenized."""
    smooth = SmoothingFunction().method1
    char_hyps = [list(h) for h in hyps]
    char_refs = [[list(r)] for r in refs]
    weights = {
        "bleu1": (1.0, 0.0, 0.0, 0.0),
        "bleu2": (0.5, 0.5, 0.0, 0.0),
        "bleu3": (1 / 3, 1 / 3, 1 / 3, 0.0),
        "bleu4": (0.25, 0.25, 0.25, 0.25),
    }
    out: Dict[str, float] = {}
    for name, w in weights.items():
        sents = [
            sentence_bleu(ref, hyp, weights=w, smoothing_function=smooth)
            for ref, hyp in zip(char_refs, char_hyps)
        ]
        out[name] = _avg(sents) * 100.0
    return out


def _word_bleu(hyps: List[str], refs: List[str]) -> Dict[str, float]:
    smooth = SmoothingFunction().method1
    word_hyps = [list(jieba.cut(h)) for h in hyps]
    word_refs = [[list(jieba.cut(r))] for r in refs]
    weights = {
        "bleu1": (1.0, 0.0, 0.0, 0.0),
        "bleu2": (0.5, 0.5, 0.0, 0.0),
        "bleu3": (1 / 3, 1 / 3, 1 / 3, 0.0),
        "bleu4": (0.25, 0.25, 0.25, 0.25),
    }
    out: Dict[str, float] = {}
    for name, w in weights.items():
        score = corpus_bleu(
            word_refs, word_hyps, weights=w, smoothing_function=smooth,
        )
        out[name] = float(score) * 100.0
    return out


def _sacrebleu_char(hyps: List[str], refs: List[str]) -> float:
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh")
    return float(bleu.score)


def _rouge_chinese(hyps: List[str], refs: List[str]) -> Dict[str, float]:
    """rouge-chinese expects pre-tokenized whitespace-separated strings."""
    rouge = Rouge()
    h_tok = [" ".join(jieba.cut(h)) if h.strip() else " " for h in hyps]
    r_tok = [" ".join(jieba.cut(r)) if r.strip() else " " for r in refs]
    scores = rouge.get_scores(h_tok, r_tok, avg=True)
    return {
        "rouge1_f": float(scores["rouge-1"]["f"]) * 100.0,
        "rouge2_f": float(scores["rouge-2"]["f"]) * 100.0,
        "rougeL_f": float(scores["rouge-l"]["f"]) * 100.0,
    }


def _eval_group(hyps: List[str], refs: List[str]) -> Dict[str, float]:
    if not hyps:
        return {}
    metrics: Dict[str, float] = {}
    char_bleu = _char_bleu_per_n(hyps, refs)
    metrics.update({f"char_{k}": v for k, v in char_bleu.items()})
    metrics["sacrebleu_zh"] = _sacrebleu_char(hyps, refs)
    word_bleu = _word_bleu(hyps, refs)
    metrics.update({f"word_{k}": v for k, v in word_bleu.items()})
    metrics.update(_rouge_chinese(hyps, refs))
    metrics["n"] = float(len(hyps))
    return metrics


def _split_by_source(rows: List[Dict[str, object]]) -> Dict[str, Tuple[List[str], List[str]]]:
    groups: Dict[str, Tuple[List[str], List[str]]] = {"all": ([], [])}
    for r in rows:
        if not r.get("has_human_reference", False) or not str(r.get("ref", "")).strip():
            continue
        hyp = str(r["hyp"])
        ref = str(r["ref"])
        groups["all"][0].append(hyp)
        groups["all"][1].append(ref)
    return groups


def _write_summary_csv(report: Dict[str, Dict[str, float]], path: Path) -> None:
    metric_keys = [
        "n",
        "char_bleu1", "char_bleu2", "char_bleu3", "char_bleu4", "sacrebleu_zh",
        "word_bleu1", "word_bleu2", "word_bleu3", "word_bleu4",
        "rouge1_f", "rouge2_f", "rougeL_f",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group"] + metric_keys)
        for group, metrics in report.items():
            row = [group] + [f"{metrics.get(k, 0.0):.4f}" for k in metric_keys]
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="BLEU/ROUGE eval for rehab text.")
    ap.add_argument("--pred", type=Path, required=True,
                    help="JSON from generate.py with [{subject_id, hyp, ref, source}]")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSON path. CSV summary is written next to it.")
    args = ap.parse_args()

    rows: List[Dict[str, object]] = json.loads(args.pred.read_text(encoding="utf-8"))
    if not rows:
        raise SystemExit("Prediction file is empty.")

    groups = _split_by_source(rows)
    report: Dict[str, Dict[str, float]] = {}
    for group, (hyps, refs) in groups.items():
        if not hyps:
            continue
        print(f"[evaluate] group={group}  n={len(hyps)}")
        report[group] = _eval_group(hyps, refs)

    if not report.get("all", {}).get("n"):
        raise SystemExit("No prediction with a traceable human reference text was supplied.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {**report, "evaluation_scope": {
                "reference_type": "real_human_rehab_text",
                "note": "Small-sample exploratory text-overlap metrics; no universal pass threshold is applied.",
            }},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    csv_path = args.out.with_suffix(".csv")
    _write_summary_csv(report, csv_path)

    print(f"[evaluate] report → {args.out}")
    print(f"[evaluate] summary → {csv_path}")
    for group, metrics in report.items():
        bleu4 = metrics.get("char_bleu4", 0.0)
        rougeL = metrics.get("rougeL_f", 0.0)
        sb = metrics.get("sacrebleu_zh", 0.0)
        print(
            f"  {group:>20s}  n={int(metrics['n']):>3d}  "
            f"char-BLEU4={bleu4:6.2f}  sacreBLEU={sb:6.2f}  ROUGE-L={rougeL:6.2f}"
        )


if __name__ == "__main__":
    main()
