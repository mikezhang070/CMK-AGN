"""Build patient-disjoint ChatML files with a human-reference evaluation set.

The 29-case label file is also used for inference.  Only records with
``has_human_rehab_text=true`` is the only source admitted to validation, test,
or automatic text-overlap evaluation. Rule-generated text from the remaining
real patients can be added to training only as labelled weak supervision.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

from .prompts import build_chat_messages
from .rehabfact import render_policy_recommendation_target, render_recommendation_target


_COURSE_RE = re.compile(r"病程[^，。]*?(?=[，。])")


def has_human_reference(item: Dict[str, object]) -> bool:
    """Return whether a record has a traceable human-authored target text."""
    return bool(item.get("has_human_rehab_text")) and bool(
        str(item.get("rehab_text", "")).strip()
    )


def has_auxiliary_generated_target(item: Dict[str, object]) -> bool:
    """Return whether a rule-based weak-supervision target is available."""
    return bool(item.get("has_auxiliary_generated_rehab_text")) and bool(
        str(item.get("rehab_text", "")).strip()
    )


def has_supervision_target(item: Dict[str, object]) -> bool:
    return has_human_reference(item) or has_auxiliary_generated_target(item)


def filter_subject_ids_by_source(
    subjects: Dict[str, Dict[str, object]],
    subject_ids: List[str],
    case_source: str = "all",
) -> List[str]:
    """Filter a selected patient list by traceable supervision source.

    ``human`` and ``auxiliary`` are deliberately explicit because auxiliary
    rule-generated reports must never be silently mixed into the human-text
    evaluation set.  The function preserves the caller's order.
    """
    if case_source not in {"all", "human", "auxiliary"}:
        raise ValueError(f"Unknown case source: {case_source}")
    if case_source == "all":
        return [str(sid) for sid in subject_ids if str(sid) in subjects]
    predicate = has_human_reference if case_source == "human" else has_auxiliary_generated_target
    return [
        str(sid) for sid in subject_ids
        if str(sid) in subjects and predicate(subjects[str(sid)])
    ]


def normalize_rehab_text(item: Dict[str, object]) -> str:
    """Normalize the first course-duration phrase without fabricating text."""
    if not has_supervision_target(item):
        return ""
    text = str(item["rehab_text"]).strip()
    days_post = int(item["demographics"]["days_post"])
    return _COURSE_RE.sub(f"病程{days_post}天", text, count=1)


def _load_suggestions(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Real-case JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    subjects = payload.get("subjects")
    if not isinstance(subjects, dict):
        raise ValueError(f"{path} must contain a top-level 'subjects' object")
    return {str(sid): item for sid, item in subjects.items()}


def _sort_ids(ids: List[str]) -> List[str]:
    return sorted(ids, key=lambda value: int(value))


def make_split(
    subjects: Dict[str, Dict[str, object]], n_folds: int = 3, seed: int = 1024,
) -> Dict[str, object]:
    """Create a deterministic patient-disjoint split for real text records.

    Every available human-authored report appears in exactly one outer test
    partition.  A validation subset is then selected solely from the remaining
    training patients.  This is an exploratory, small-sample protocol, not a
    substitute for a large independent clinical text corpus.
    """
    ids = _sort_ids([sid for sid, item in subjects.items() if has_human_reference(item)])
    if len(ids) < n_folds * 2:
        raise ValueError(
            f"Need at least {n_folds * 2} human-authored reports for {n_folds}-fold "
            f"training; found {len(ids)}."
        )

    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    outer_test = [shuffled[index::n_folds] for index in range(n_folds)]
    all_ids = set(ids)
    folds: List[Dict[str, object]] = []
    for fold_number, test_ids in enumerate(outer_test, start=1):
        remaining = _sort_ids(list(all_ids - set(test_ids)))
        rng = random.Random(seed + fold_number)
        rng.shuffle(remaining)
        n_val = max(1, round(len(remaining) * 0.2))
        val_ids = _sort_ids(remaining[:n_val])
        train_ids = _sort_ids(remaining[n_val:])
        folds.append({
            "fold": fold_number,
            "train_subjects": train_ids,
            "val_subjects": val_ids,
            "test_subjects": _sort_ids(test_ids),
        })
    return {
        "schema_version": 2,
        "split_unit": "subject_id",
        "n_splits": n_folds,
        "seed": seed,
        "strategy": "real_human_text_patient_disjoint_cv",
        "eligible_subject_ids": ids,
        "n_human_reference_texts": len(ids),
        "note": "Only real cases with an Excel-traceable human rehab report are eligible.",
        "folds": folds,
    }


def _get_fold(split_payload: Dict[str, object], fold: int) -> Dict[str, object]:
    for candidate in split_payload["folds"]:
        if int(candidate["fold"]) == int(fold):
            return candidate
    raise KeyError(f"Fold {fold} not found in split file")


def partition_subjects(split_payload: Dict[str, object], fold: int, partition: str) -> List[str]:
    """Return one explicit partition from a v2 real-text split."""
    selected = _get_fold(split_payload, fold)
    key = f"{partition}_subjects"
    if key not in selected:
        raise ValueError(
            "Expected a real-text split with train_subjects, val_subjects, and "
            "test_subjects. Rebuild it with src.llm.data_builder."
        )
    return list(map(str, selected[key]))


def _write_jsonl(records: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def build_records(
    subjects: Dict[str, Dict[str, object]], subject_ids: List[str], allow_auxiliary: bool = False,
    rehabfact: bool = False, rehabfact_policy: bool = False,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for sid in subject_ids:
        item = subjects.get(sid)
        if item is None or not has_supervision_target(item):
            continue
        if not has_human_reference(item) and not allow_auxiliary:
            continue
        target = (
            (
                render_policy_recommendation_target(item["labels"], item["demographics"])
                if rehabfact_policy else render_recommendation_target(item["labels"])
            )
            if rehabfact else normalize_rehab_text(item)
        )
        messages = build_chat_messages(
            subject_id=sid,
            demographics=item["demographics"],
            labels=item["labels"],
            rehab_text=target,
            rehabfact=rehabfact,
            rehabfact_policy=rehabfact_policy,
        )
        rows.append({
            "subject_id": sid,
            "messages": messages,
            "source": (
                "real_human_rehab_text" if has_human_reference(item)
                else "rule_generated_auxiliary_from_real_variables"
            ),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChatML data from real reports and optional auxiliary templates.")
    parser.add_argument("--suggestions", type=Path, default=Path("data/patient_rehab_29subjects.json"))
    parser.add_argument("--split", type=Path, default=Path("splits/llm_real_text13_patient_3fold.json"))
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rebuild-split", action="store_true")
    parser.add_argument("--no-aux-generated", action="store_true",
                        help="Exclude rule-generated real-case reports from training.")
    parser.add_argument("--rehabfact", action="store_true",
                        help="Use deterministic RehabFact-LoRA fact-card prompts and canonical targets.")
    parser.add_argument("--rehabfact-policy", action="store_true",
                        help="Use RehabFact v3 clinical strategy cards; requires --rehabfact.")
    args = parser.parse_args()

    subjects = _load_suggestions(args.suggestions)
    if args.rebuild_split or not args.split.exists():
        split_payload = make_split(subjects, n_folds=args.n_folds, seed=args.seed)
        args.split.parent.mkdir(parents=True, exist_ok=True)
        args.split.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        split_payload = json.loads(args.split.read_text(encoding="utf-8"))

    train_human_ids = partition_subjects(split_payload, args.fold, "train")
    records = {
        "train": build_records(subjects, train_human_ids, rehabfact=args.rehabfact,
                               rehabfact_policy=args.rehabfact_policy),
        "val": build_records(subjects, partition_subjects(split_payload, args.fold, "val"),
                             rehabfact=args.rehabfact, rehabfact_policy=args.rehabfact_policy),
        "test": build_records(subjects, partition_subjects(split_payload, args.fold, "test"),
                              rehabfact=args.rehabfact, rehabfact_policy=args.rehabfact_policy),
    }
    auxiliary_rows: List[Dict[str, object]] = []
    if not args.no_aux_generated:
        auxiliary_ids = _sort_ids([
            sid for sid, item in subjects.items() if has_auxiliary_generated_target(item)
        ])
        auxiliary_rows = build_records(
            subjects, auxiliary_ids, allow_auxiliary=True, rehabfact=args.rehabfact,
            rehabfact_policy=args.rehabfact_policy,
        )
        records["train"].extend(auxiliary_rows)
    for partition, rows in records.items():
        if not rows:
            raise ValueError(f"Fold {args.fold} has no {partition} records; cannot train/evaluate safely.")
        _write_jsonl(rows, args.out / f"{partition}.jsonl")
    print(
        f"[data_builder] fold={args.fold} rehabfact={args.rehabfact} policy={args.rehabfact_policy} | "
        f"train={len(records['train'])} (human={len(train_human_ids)}, auxiliary={len(auxiliary_rows)})  "
        f"val={len(records['val'])} (human only)  test={len(records['test'])} (human only)"
    )


if __name__ == "__main__":
    main()
