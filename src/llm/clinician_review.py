"""Blinded clinician-review packets and patient-clustered score summaries.

The module deliberately separates the reviewer-facing packet from the private
linking key.  Reviewer packets contain the structured inputs necessary to judge
a report but never a subject identifier or model/system identifier.
"""
from __future__ import annotations

import itertools
import random
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import cohen_kappa_score


RATING_FIELDS: Tuple[str, ...] = (
    "fact_consistency",
    "clinical_appropriateness",
    "actionability",
    "individualization",
    "readability",
)


def _sort_subjects(subjects: Iterable[str]) -> List[str]:
    def key(value: str) -> Tuple[int, object]:
        value = str(value)
        return (0, int(value)) if value.isdigit() else (1, value)

    return sorted({str(subject) for subject in subjects}, key=key)


def _structured_input(labels: Mapping[str, object]) -> str:
    return (
        "病例结构化输入（仅用于评价报告与已知输入的一致性）："
        f"FMA手部分数 {int(labels['FMA_UE'])}/20；"
        f"BI {int(labels['BI'])}/100；"
        f"手部肌张力 {str(labels['hand_tone']).strip()}级；"
        f"Brunnstrom手分期 {int(labels['hand_function'])}期。"
    )


def _deidentify_report(text: str) -> str:
    """Remove explicit patient-number tokens before the reviewer packet is shared."""
    return re.sub(r"患者\s*(?:编号\s*[:：]?\s*)?(?:S\s*)?\d+", "该患者", text, flags=re.IGNORECASE)


def build_blinded_packet(
    systems: Mapping[str, Sequence[Mapping[str, object]]], seed: int = 2026,
    require_human_reference: bool = True,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Create blinded reviewer rows and a separate unblinded linking key.

    By default, only rows with traceable human references are eligible. This
    keeps the historical review set at the 13 human-text cases. Setting
    ``require_human_reference=False`` explicitly enables a separate qualitative
    review of structured-only cases; it does not create BLEU/ROUGE ground truth.
    """
    candidate_rows: List[Dict[str, object]] = []
    seen: set[Tuple[str, str]] = set()
    for system_id, rows in systems.items():
        for row in rows:
            if require_human_reference and not bool(row.get("has_human_reference", False)):
                continue
            subject_id = str(row.get("subject_id", "")).strip()
            if not subject_id or not str(row.get("hyp", "")).strip():
                continue
            key = (str(system_id), subject_id)
            if key in seen:
                raise ValueError(f"Duplicate evaluation report for system/patient {key}.")
            seen.add(key)
            candidate_rows.append({
                "system_id": str(system_id),
                "subject_id": subject_id,
                "source": str(row.get("source", "")),
                "labels": dict(row["labels"]),
                "report_text": _deidentify_report(str(row["hyp"]).strip()),
            })
    if not candidate_rows:
        raise ValueError("No non-empty reports with traceable human references were supplied.")

    subjects = _sort_subjects(row["subject_id"] for row in candidate_rows)
    case_codes = {subject_id: f"C{index:03d}" for index, subject_id in enumerate(subjects, start=1)}
    rng = random.Random(seed)
    rng.shuffle(candidate_rows)

    packet: List[Dict[str, object]] = []
    linking_key: List[Dict[str, object]] = []
    for index, row in enumerate(candidate_rows, start=1):
        review_id = f"R{index:03d}"
        case_code = case_codes[str(row["subject_id"])]
        packet.append({
            "review_id": review_id,
            "case_code": case_code,
            "structured_input": _structured_input(row["labels"]),
            "report_text": row["report_text"],
            **{field: "" for field in RATING_FIELDS},
            "comments": "",
        })
        linking_key.append({
            "review_id": review_id,
            "case_code": case_code,
            "system_id": row["system_id"],
            "subject_id": row["subject_id"],
            "source": row["source"],
            "has_human_reference": bool(row.get("has_human_reference", False)),
        })
    return packet, linking_key


def _parse_rating(value: object, field: str, review_id: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{review_id}: {field} must be an integer score in 1..5, got {value!r}") from exc
    if score not in {1, 2, 3, 4, 5}:
        raise ValueError(f"{review_id}: {field} must be in 1..5, got {score}")
    return score


def _cluster_ci(case_scores: Mapping[str, Sequence[float]], n_boot: int, seed: int) -> Tuple[float, float, float]:
    cases = sorted(case_scores)
    observed = float(np.mean([score for values in case_scores.values() for score in values]))
    if len(cases) < 2 or n_boot <= 0:
        return observed, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled_means = []
    for _ in range(n_boot):
        selected = rng.choice(cases, size=len(cases), replace=True)
        sampled_means.append(float(np.mean([
            score for case in selected for score in case_scores[str(case)]
        ])))
    low, high = np.quantile(sampled_means, [0.025, 0.975])
    return observed, float(low), float(high)


def summarize_ratings(
    reviewers: Mapping[str, Sequence[Mapping[str, object]]],
    linking_key: Sequence[Mapping[str, object]],
    n_boot: int = 2000,
    seed: int = 2026,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Aggregate blinded ratings with case-clustered CIs and weighted κ.

    The returned system summary averages each report over available reviewers;
    the source reviewer files remain the record of individual raw scores.
    """
    if len(reviewers) < 2:
        raise ValueError("At least two independent reviewers are required for agreement analysis.")
    key_by_review = {str(row["review_id"]): dict(row) for row in linking_key}
    if not key_by_review:
        raise ValueError("The linking key is empty.")

    parsed: Dict[str, Dict[str, Dict[str, float]]] = {}
    expected_ids = set(key_by_review)
    for reviewer_id, rows in reviewers.items():
        by_review: Dict[str, Dict[str, float]] = {}
        for row in rows:
            review_id = str(row.get("review_id", "")).strip()
            if review_id not in key_by_review:
                raise ValueError(f"{reviewer_id}: unknown review_id {review_id!r}.")
            by_review[review_id] = {
                field: _parse_rating(row.get(field), field, review_id)
                for field in RATING_FIELDS
            }
        if set(by_review) != expected_ids:
            missing = sorted(expected_ids - set(by_review))
            extra = sorted(set(by_review) - expected_ids)
            raise ValueError(f"{reviewer_id}: review IDs do not match key; missing={missing}, extra={extra}")
        parsed[str(reviewer_id)] = by_review

    summary: List[Dict[str, object]] = []
    reviewer_ids = sorted(parsed)
    systems = sorted({str(row["system_id"]) for row in linking_key})
    for system_id in systems:
        system_ids = [rid for rid, key in key_by_review.items() if str(key["system_id"]) == system_id]
        case_count = len({str(key_by_review[rid]["case_code"]) for rid in system_ids})
        for field_index, field in enumerate(RATING_FIELDS):
            case_scores: Dict[str, List[float]] = defaultdict(list)
            per_reviewer_means: Dict[str, float] = {}
            for reviewer_id in reviewer_ids:
                values = [parsed[reviewer_id][rid][field] for rid in system_ids]
                per_reviewer_means[reviewer_id] = float(np.mean(values))
            for rid in system_ids:
                case = str(key_by_review[rid]["case_code"])
                case_scores[case].append(float(np.mean([
                    parsed[reviewer_id][rid][field] for reviewer_id in reviewer_ids
                ])))
            mean, low, high = _cluster_ci(case_scores, n_boot=n_boot, seed=seed + field_index)
            summary.append({
                "system_id": system_id,
                "dimension": field,
                "n_reports": len(system_ids),
                "n_cases": case_count,
                "mean_score": mean,
                "ci95_low": low,
                "ci95_high": high,
                **{f"{reviewer_id}_mean": value for reviewer_id, value in per_reviewer_means.items()},
            })

    agreement: List[Dict[str, object]] = []
    for first, second in itertools.combinations(reviewer_ids, 2):
        for field in RATING_FIELDS:
            values_a = [parsed[first][rid][field] for rid in sorted(expected_ids)]
            values_b = [parsed[second][rid][field] for rid in sorted(expected_ids)]
            if len(set(values_a) | set(values_b)) < 2:
                # Agreement is undefined when the pair never uses more than one score.
                kappa = float("nan")
            else:
                try:
                    kappa = float(cohen_kappa_score(values_a, values_b, weights="linear"))
                except Exception:
                    kappa = float("nan")
            agreement.append({
                "reviewer_a": first,
                "reviewer_b": second,
                "dimension": field,
                "n_reports": len(expected_ids),
                "weighted_kappa": kappa,
            })
    return summary, agreement


__all__ = ["RATING_FIELDS", "build_blinded_packet", "summarize_ratings"]
