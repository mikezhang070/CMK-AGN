from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SPLIT_PATH = Path("splits") / "split_patient_3fold.json"

VALID_SPLIT_MODES = ("auto", "subject", "trial")

# Optional label columns we balance across folds when present in the manifest.
# Order matters for tie-breaking: earlier columns dominate cost weighting.
REGRESSION_LABEL_COLS: Tuple[str, ...] = ("fma_ue", "bi")
CATEGORICAL_LABEL_COLS: Tuple[str, ...] = ("hand_tone", "hand_function")
SOURCE_COL = "source"

# Default cost weights for the multi-label balanced greedy assignment. These
# are intentionally small relative to capacity (which is enforced as a hard
# cap); they only need to break ties in a clinically meaningful way.
DEFAULT_BALANCE_WEIGHTS: Dict[str, float] = {
    "sample_count": 1.0,
    "fma_ue": 0.6,
    "bi": 0.05,           # BI lives on a 0..100 scale, so down-weight per-unit
    "hand_tone": 1.5,
    "hand_function": 1.5,
    "source": 4.0,        # 5 real patients in 150 → keep them spread across folds
}


def _subject_sort_key(subject_id: str) -> int:
    return int(subject_id)


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _normalize_label_value(col: str, raw: Any) -> Any:
    """Coerce a manifest cell into a hashable label value.

    Tone labels contain the literal string "1+" which must NOT be coerced to a
    number. `hand_function` is stored as integer 1..6. Empty cells map to None.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.lower() == "nan":
        return None
    if col == "hand_function":
        try:
            return int(float(s))
        except ValueError:
            return s
    return s


def _load_subject_stats(manifest_path: Path) -> Dict[str, Dict[str, object]]:
    stats, _ = _load_manifest_rows(manifest_path)
    return stats


def _load_manifest_rows(
    manifest_path: Path,
) -> tuple[Dict[str, Dict[str, object]], List[Dict[str, object]]]:
    """Read the manifest once; return both subject-level stats and per-trial rows.

    Per-trial rows always carry: subject_id, trial_id, fma_score, key.
    Optional label columns (fma_ue / bi / hand_tone / hand_function / source)
    are attached to each row and aggregated into subject_stats when present in
    the manifest header.
    """
    rows: List[Dict[str, object]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"subject_id", "trial_id", "fma_score"}
        missing = required - fieldnames
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

        optional_cols = [c for c in (*REGRESSION_LABEL_COLS, *CATEGORICAL_LABEL_COLS, SOURCE_COL)
                         if c in fieldnames]

        fma_values: Dict[str, set[float]] = defaultdict(set)
        sample_counts: Counter[str] = Counter()
        seen_keys: set[str] = set()
        # subject_id → {col → set of distinct values}
        per_subject_labels: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))

        for row in reader:
            subject_id = str(row["subject_id"])
            trial_id = str(row["trial_id"])
            key = f"{subject_id}:{trial_id}"
            if key in seen_keys:
                raise ValueError(f"Duplicate subject/trial sample found: {key}")
            seen_keys.add(key)
            sample_counts[subject_id] += 1
            fma_values[subject_id].add(float(row["fma_score"]))

            row_record: Dict[str, object] = {
                "subject_id": subject_id,
                "trial_id": trial_id,
                "fma_score": float(row["fma_score"]),
                "key": key,
            }
            for col in optional_cols:
                value = _normalize_label_value(col, row.get(col))
                row_record[col] = value
                if value is not None:
                    per_subject_labels[subject_id][col].add(value)
            rows.append(row_record)

    stats: Dict[str, Dict[str, object]] = {}
    for subject_id in sorted(sample_counts, key=_subject_sort_key):
        if len(fma_values[subject_id]) != 1:
            raise ValueError(f"Subject {subject_id} has inconsistent FMA labels: {sorted(fma_values[subject_id])}")
        record: Dict[str, object] = {
            "sample_count": int(sample_counts[subject_id]),
            "fma_score": next(iter(fma_values[subject_id])),
        }
        for col, values in per_subject_labels.get(subject_id, {}).items():
            if not values:
                continue
            if len(values) > 1:
                # Trials of the same subject should share the same patient-level
                # labels in this dataset. Flag the inconsistency early.
                raise ValueError(
                    f"Subject {subject_id} has inconsistent {col!r} labels across trials: {sorted(map(str, values))}"
                )
            record[col] = next(iter(values))
        stats[subject_id] = record
    return stats, rows


def _target_fold_sizes(n_subjects: int, n_splits: int) -> List[int]:
    base = n_subjects // n_splits
    remainder = n_subjects % n_splits
    return [base + (1 if fold_index < remainder else 0) for fold_index in range(n_splits)]


# --------------------------------------------------------------------------- #
# Multi-label balanced greedy assignment                                      #
# --------------------------------------------------------------------------- #


def _collect_category_universe(
    subject_stats: Dict[str, Dict[str, object]],
    cols: Sequence[str],
) -> Dict[str, List[Any]]:
    """For each categorical column, return the sorted list of observed classes."""
    universe: Dict[str, List[Any]] = {}
    for col in cols:
        classes = sorted({s[col] for s in subject_stats.values() if col in s and s[col] is not None},
                         key=lambda v: (isinstance(v, str), v))
        if classes:
            universe[col] = classes
    return universe


def _subject_rarity_score(
    subject: Dict[str, object],
    class_counts: Dict[str, Counter],
) -> float:
    """Lower count classes contribute more to rarity. Drives processing order
    so that hard-to-place subjects (rare class combinations) land first.
    """
    score = 0.0
    for col, counter in class_counts.items():
        value = subject.get(col)
        if value is None:
            continue
        c = counter.get(value, 0)
        if c > 0:
            score += 1.0 / c
    return score


def _fold_cost_increment(
    fold_idx: int,
    subject: Dict[str, object],
    fold_sample_counts: List[int],
    fold_reg_sums: Dict[str, List[float]],
    fold_class_counts: Dict[str, List[Counter]],
    fold_source_counts: Counter,
    target_sample: float,
    target_reg: Dict[str, float],
    target_class: Dict[str, Dict[Any, float]],
    target_source: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """Marginal cost of adding `subject` to fold `fold_idx`.

    For every balanced quantity we compute the *change* in squared deviation
    from the per-fold target, i.e. (after - target)^2 - (before - target)^2.
    Using the marginal form is critical: with absolute (after-target)^2 the
    cost would favor adding to whichever fold is already heaviest while
    everything is still ramping up toward the target.
    """
    sample = int(subject["sample_count"])
    cost = 0.0

    # Sample-count balance — marginal cost.
    before_n = fold_sample_counts[fold_idx]
    after_n = before_n + sample
    cost += weights["sample_count"] * (
        (after_n - target_sample) ** 2 - (before_n - target_sample) ** 2
    )

    # Regression label balance — marginal cost on the sum per fold.
    for col, sums in fold_reg_sums.items():
        value = subject.get(col)
        if value is None:
            continue
        before = sums[fold_idx]
        after = before + float(value)
        target = target_reg[col]
        cost += weights.get(col, 1.0) * ((after - target) ** 2 - (before - target) ** 2)

    # Categorical class balance: each class should appear ~target_class[col][cls]
    # times per fold. Increment count for the subject's class only.
    for col, fold_counts in fold_class_counts.items():
        value = subject.get(col)
        if value is None:
            continue
        target = target_class[col].get(value, 0.0)
        before = fold_counts[fold_idx].get(value, 0)
        after = before + 1
        cost += weights.get(col, 1.0) * ((after - target) ** 2 - (before - target) ** 2)

    # Source (real/synthetic) balance — important when real cases are few.
    src = subject.get(SOURCE_COL)
    if src is not None and SOURCE_COL in target_source:
        target = target_source[src]
        before = fold_source_counts[(fold_idx, src)]
        after = before + 1
        cost += weights.get(SOURCE_COL, 1.0) * ((after - target) ** 2 - (before - target) ** 2)

    return cost


def _assign_balanced_folds(
    subject_stats: Dict[str, Dict[str, object]],
    n_splits: int,
    weights: Optional[Dict[str, float]] = None,
) -> List[List[str]]:
    """Multi-label balanced greedy assignment.

    Balances simultaneously: subject count, sample (trial) count, regression
    label sums (FMA / BI) and per-class counts for categorical labels
    (hand_tone, hand_function) and source (real/synthetic).

    The algorithm is deterministic (no randomness): subjects are processed in
    rarity-descending order, and on each step we assign to the fold yielding
    the smallest *post-assignment* multi-label imbalance cost. Folds at their
    size cap are excluded as candidates, so subject-count balance is hard.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    n_subjects = len(subject_stats)
    if n_subjects < n_splits:
        raise ValueError(f"Need at least {n_splits} subjects, got {n_subjects}")

    weights = {**DEFAULT_BALANCE_WEIGHTS, **(weights or {})}

    target_sizes = _target_fold_sizes(n_subjects, n_splits)

    # Determine which optional columns are actually present in the dataset.
    reg_cols = [c for c in REGRESSION_LABEL_COLS
                if any(c in s and s[c] is not None for s in subject_stats.values())]
    cat_cols = [c for c in CATEGORICAL_LABEL_COLS
                if any(c in s and s[c] is not None for s in subject_stats.values())]
    has_source = any(SOURCE_COL in s and s[SOURCE_COL] is not None for s in subject_stats.values())

    # --- Per-fold targets (totals × 1/n_splits) -------------------------------
    total_samples = sum(int(s["sample_count"]) for s in subject_stats.values())
    target_sample = total_samples / n_splits

    target_reg: Dict[str, float] = {}
    for col in reg_cols:
        total = sum(float(s[col]) for s in subject_stats.values() if s.get(col) is not None)
        target_reg[col] = total / n_splits

    target_class: Dict[str, Dict[Any, float]] = {}
    universe = _collect_category_universe(subject_stats, cat_cols)
    global_class_counts: Dict[str, Counter] = {}
    for col in cat_cols:
        counter: Counter = Counter()
        for s in subject_stats.values():
            value = s.get(col)
            if value is not None:
                counter[value] += 1
        global_class_counts[col] = counter
        target_class[col] = {cls: counter[cls] / n_splits for cls in universe[col]}

    target_source: Dict[str, float] = {}
    source_counter: Counter = Counter()
    if has_source:
        for s in subject_stats.values():
            src = s.get(SOURCE_COL)
            if src is not None:
                source_counter[src] += 1
        target_source = {src: count / n_splits for src, count in source_counter.items()}

    # --- Per-fold accumulators -----------------------------------------------
    folds: List[List[str]] = [[] for _ in range(n_splits)]
    fold_sample_counts: List[int] = [0 for _ in range(n_splits)]
    fold_reg_sums: Dict[str, List[float]] = {col: [0.0] * n_splits for col in reg_cols}
    fold_class_counts: Dict[str, List[Counter]] = {
        col: [Counter() for _ in range(n_splits)] for col in cat_cols
    }
    fold_source_counts: Counter = Counter()

    # --- Subject processing order: rarest combinations first, then large
    # sample counts. Subject-id is the final tiebreaker for determinism.
    rarity_counts = {col: global_class_counts[col] for col in cat_cols}
    if has_source:
        rarity_counts[SOURCE_COL] = source_counter

    ordered_subjects = sorted(
        subject_stats,
        key=lambda sid: (
            -_subject_rarity_score(subject_stats[sid], rarity_counts),
            -int(subject_stats[sid]["sample_count"]),
            -float(subject_stats[sid].get("fma_score", 0.0)),
            _subject_sort_key(sid),
        ),
    )

    for subject_id in ordered_subjects:
        subject = subject_stats[subject_id]
        candidates = [idx for idx in range(n_splits) if len(folds[idx]) < target_sizes[idx]]
        if not candidates:
            raise RuntimeError("No fold has remaining capacity while assigning subjects.")

        best_idx = min(
            candidates,
            key=lambda idx: (
                _fold_cost_increment(
                    idx, subject,
                    fold_sample_counts, fold_reg_sums, fold_class_counts, fold_source_counts,
                    target_sample, target_reg, target_class, target_source,
                    weights,
                ),
                fold_sample_counts[idx],   # tiebreak: lighter fold first
                len(folds[idx]),
                idx,
            ),
        )

        folds[best_idx].append(subject_id)
        fold_sample_counts[best_idx] += int(subject["sample_count"])
        for col in reg_cols:
            value = subject.get(col)
            if value is not None:
                fold_reg_sums[col][best_idx] += float(value)
        for col in cat_cols:
            value = subject.get(col)
            if value is not None:
                fold_class_counts[col][best_idx][value] += 1
        if has_source:
            src = subject.get(SOURCE_COL)
            if src is not None:
                fold_source_counts[(best_idx, src)] += 1

    return [sorted(fold, key=_subject_sort_key) for fold in folds]


# --------------------------------------------------------------------------- #
# Distribution summaries (for diagnostics in the JSON)                        #
# --------------------------------------------------------------------------- #


def _fma_distribution(subject_ids: Sequence[str], subject_stats: Dict[str, Dict[str, object]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for subject_id in subject_ids:
        score = float(subject_stats[subject_id]["fma_score"])
        key = str(int(score)) if score.is_integer() else str(score)
        counts[key] += 1
    return dict(sorted(counts.items(), key=lambda item: float(item[0])))


def _sample_count(subject_ids: Iterable[str], subject_stats: Dict[str, Dict[str, object]]) -> int:
    return sum(int(subject_stats[subject_id]["sample_count"]) for subject_id in subject_ids)


def _category_distribution(
    subject_ids: Sequence[str],
    subject_stats: Dict[str, Dict[str, object]],
    col: str,
) -> Dict[str, int]:
    counts: Counter = Counter()
    for sid in subject_ids:
        value = subject_stats[sid].get(col)
        if value is None:
            continue
        counts[str(value)] += 1
    # Sort numeric-like keys numerically; fall back to string sort otherwise.
    def _key(item):
        k = item[0]
        try:
            return (0, float(k))
        except ValueError:
            return (1, k)
    return dict(sorted(counts.items(), key=_key))


def _label_distribution(
    subject_ids: Sequence[str],
    subject_stats: Dict[str, Dict[str, object]],
    cat_cols: Sequence[str],
    reg_cols: Sequence[str],
    has_source: bool,
) -> Dict[str, object]:
    out: Dict[str, object] = {"fma": _fma_distribution(subject_ids, subject_stats)}
    for col in cat_cols:
        if any(col in subject_stats[sid] for sid in subject_ids):
            out[col] = _category_distribution(subject_ids, subject_stats, col)
    for col in reg_cols:
        values = [float(subject_stats[sid][col]) for sid in subject_ids
                  if subject_stats[sid].get(col) is not None]
        if values:
            out[col + "_stats"] = {
                "n": len(values),
                "sum": round(sum(values), 4),
                "mean": round(sum(values) / len(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }
    if has_source:
        out["source"] = _category_distribution(subject_ids, subject_stats, SOURCE_COL)
    return out


def _trial_fma_distribution(rows: Sequence[Dict[str, object]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        score = float(row["fma_score"])
        key = str(int(score)) if score.is_integer() else str(score)
        counts[key] += 1
    return dict(sorted(counts.items(), key=lambda item: float(item[0])))


def _assign_trial_folds(
    rows: Sequence[Dict[str, object]],
    n_splits: Optional[int],
) -> List[List[Dict[str, object]]]:
    """Assign trials to folds using a balanced round-robin.

    The ordering is deterministic: rows are sorted by (subject_id, task_id,
    trial_number) and rotated across folds — so for a single subject with k
    trials and n_splits=k this is exactly leave-one-trial-out (LOO). For
    multiple subjects we still hold out one trial per subject per fold (when
    sample counts are balanced), avoiding the worst case where a fold contains
    only one subject's trials.
    """
    if not rows:
        raise ValueError("No trials to split.")
    n = len(rows)
    if n_splits is None or n_splits <= 0:
        n_splits = n
    n_splits = int(min(max(n_splits, 2), n))

    ordered = sorted(rows, key=lambda r: (int(r["subject_id"]), str(r["trial_id"])))
    folds: List[List[Dict[str, object]]] = [[] for _ in range(n_splits)]
    for idx, row in enumerate(ordered):
        folds[idx % n_splits].append(row)
    return folds


def _build_subject_splits(
    manifest_relative: str,
    subject_stats: Dict[str, Dict[str, object]],
    n_splits: int,
    seed: int,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    heldout_folds = _assign_balanced_folds(subject_stats, n_splits, weights=weights)
    all_subjects = sorted(subject_stats, key=_subject_sort_key)
    # Detect which optional columns are present (purely for distribution reporting).
    reg_cols = [c for c in REGRESSION_LABEL_COLS
                if any(c in s and s[c] is not None for s in subject_stats.values())]
    cat_cols = [c for c in CATEGORICAL_LABEL_COLS
                if any(c in s and s[c] is not None for s in subject_stats.values())]
    has_source = any(SOURCE_COL in s and s[SOURCE_COL] is not None for s in subject_stats.values())

    folds: List[Dict[str, object]] = []
    for fold_index, heldout_subjects in enumerate(heldout_folds, start=1):
        heldout_set = set(heldout_subjects)
        train_subjects = [s for s in all_subjects if s not in heldout_set]
        overlap = set(train_subjects) & heldout_set
        assert not overlap, f"Subject leakage in fold {fold_index}: {sorted(overlap)}"
        folds.append(
            {
                "fold": fold_index,
                "train_subjects": train_subjects,
                "val_test_subjects": list(heldout_subjects),
                "train_samples": _sample_count(train_subjects, subject_stats),
                "val_test_samples": _sample_count(heldout_subjects, subject_stats),
                "fma_distribution": {
                    "train": _fma_distribution(train_subjects, subject_stats),
                    "val_test": _fma_distribution(heldout_subjects, subject_stats),
                },
                "label_distribution": {
                    "train": _label_distribution(train_subjects, subject_stats,
                                                 cat_cols, reg_cols, has_source),
                    "val_test": _label_distribution(heldout_subjects, subject_stats,
                                                    cat_cols, reg_cols, has_source),
                },
            }
        )
    return {
        "schema_version": 3,
        "split_unit": "subject_id",
        "split_mode": "subject",
        "n_splits": n_splits,
        "seed": seed,
        "strategy": "multi_label_balanced_greedy_subject",
        "balanced_labels": reg_cols + cat_cols + ([SOURCE_COL] if has_source else []),
        "manifest_path": manifest_relative,
        "subject_stats": subject_stats,
        "folds": folds,
    }


def _build_trial_splits(
    manifest_relative: str,
    subject_stats: Dict[str, Dict[str, object]],
    rows: Sequence[Dict[str, object]],
    n_splits: Optional[int],
    seed: int,
) -> Dict[str, object]:
    heldout_groups = _assign_trial_folds(rows, n_splits)
    actual_splits = len(heldout_groups)
    all_keys = [str(r["key"]) for r in rows]
    folds: List[Dict[str, object]] = []
    for fold_index, heldout in enumerate(heldout_groups, start=1):
        heldout_keys = [str(r["key"]) for r in heldout]
        train_keys = [k for k in all_keys if k not in set(heldout_keys)]
        train_rows = [r for r in rows if str(r["key"]) in set(train_keys)]
        train_subjects = sorted({str(r["subject_id"]) for r in train_rows}, key=_subject_sort_key)
        val_subjects = sorted({str(r["subject_id"]) for r in heldout}, key=_subject_sort_key)
        folds.append(
            {
                "fold": fold_index,
                "train_keys": train_keys,
                "val_test_keys": heldout_keys,
                "train_subjects": train_subjects,
                "val_test_subjects": val_subjects,
                "train_samples": len(train_keys),
                "val_test_samples": len(heldout_keys),
                "fma_distribution": {
                    "train": _trial_fma_distribution(train_rows),
                    "val_test": _trial_fma_distribution(heldout),
                },
            }
        )
    return {
        "schema_version": 3,
        "split_unit": "trial_id",
        "split_mode": "trial",
        "n_splits": actual_splits,
        "seed": seed,
        "strategy": "leave_one_trial_out_round_robin",
        "manifest_path": manifest_relative,
        "subject_stats": subject_stats,
        "folds": folds,
    }


def _resolve_split_mode(
    requested: str,
    n_subjects: int,
    n_trials: int,
    n_splits: int,
) -> str:
    requested = (requested or "auto").lower()
    if requested not in VALID_SPLIT_MODES:
        raise ValueError(f"split_mode must be one of {VALID_SPLIT_MODES}, got {requested!r}")
    if requested == "auto":
        if n_subjects >= max(n_splits, 2):
            return "subject"
        if n_trials >= 2:
            return "trial"
        raise ValueError(
            f"Cannot auto-select a split mode: n_subjects={n_subjects}, n_trials={n_trials}, "
            f"n_splits={n_splits}. Need either >= n_splits subjects or >= 2 trials."
        )
    if requested == "subject" and n_subjects < max(n_splits, 2):
        raise ValueError(
            f"split_mode='subject' requires at least {max(n_splits, 2)} subjects; "
            f"got {n_subjects}. Use 'trial' or 'auto' for the single-patient case."
        )
    if requested == "trial" and n_trials < 2:
        raise ValueError(f"split_mode='trial' requires at least 2 trials; got {n_trials}.")
    return requested


def build_patient_splits(
    manifest_path: Path | str = "samples_manifest.csv",
    output_path: Path | str = DEFAULT_SPLIT_PATH,
    n_splits: int = 3,
    seed: int = 1024,
    split_mode: str = "auto",
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Build deterministic cross-validation folds.

    split_mode="auto" (default): if at least `max(n_splits, 2)` subjects are
    present, build subject-level CV (no subject leakage). Otherwise fall back
    to leave-one-trial-out, which is the only sensible scheme when the dataset
    has a single patient.

    split_mode="subject": force subject-level CV (raises if not enough subjects).
    split_mode="trial":   force trial-level LOO/round-robin folds.

    For subject-level CV, fold assignment jointly balances:
      - subject count (hard cap)
      - sample (trial) count
      - regression labels (FMA_UE, BI)
      - categorical labels (hand_tone, hand_function)
      - source (real vs synthetic)
    when those columns exist in the manifest. `weights` lets callers override
    the default cost weights for the multi-label cost function.
    """
    manifest = Path(manifest_path).resolve()
    output = Path(output_path)
    if not output.is_absolute():
        output = manifest.parent / output

    subject_stats, rows = _load_manifest_rows(manifest)
    chosen = _resolve_split_mode(
        split_mode,
        n_subjects=len(subject_stats),
        n_trials=len(rows),
        n_splits=n_splits,
    )
    manifest_rel = _relative_path(manifest, manifest.parent)

    if chosen == "subject":
        split = _build_subject_splits(manifest_rel, subject_stats, n_splits, seed, weights=weights)
    else:
        split = _build_trial_splits(manifest_rel, subject_stats, rows, n_splits, seed)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(split, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return split


def load_patient_splits(path: Path | str = DEFAULT_SPLIT_PATH) -> Dict[str, object]:
    split_path = Path(path)
    with split_path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    validate_patient_splits(split)
    return split


def validate_patient_splits(split: Dict[str, object]) -> None:
    unit = split.get("split_unit")
    if unit not in {"subject_id", "trial_id"}:
        raise ValueError(f"Split unit must be subject_id or trial_id, got {unit!r}")
    subject_stats = split.get("subject_stats", {})
    if not subject_stats:
        raise ValueError("Split file has no subject_stats.")

    folds = split.get("folds", [])
    if unit == "subject_id":
        all_subjects = set(subject_stats)
        for fold in folds:
            fold_id = fold["fold"]
            train_subjects = {str(s) for s in fold["train_subjects"]}
            heldout_subjects = {str(s) for s in fold["val_test_subjects"]}
            overlap = train_subjects & heldout_subjects
            assert not overlap, f"Subject leakage in fold {fold_id}: {sorted(overlap)}"
            if train_subjects | heldout_subjects != all_subjects:
                raise ValueError(f"Fold {fold_id} does not cover all subjects exactly once.")
        return

    # trial_id mode: every key (subject_id:trial_id) must appear in exactly one
    # held-out fold across the cross-validation; subject overlap is allowed.
    seen_heldout: Counter[str] = Counter()
    all_keys: set[str] = set()
    for fold in folds:
        fold_id = fold["fold"]
        if "train_keys" not in fold or "val_test_keys" not in fold:
            raise ValueError(f"Fold {fold_id} missing train_keys/val_test_keys for trial_id mode.")
        train_keys = {str(k) for k in fold["train_keys"]}
        heldout_keys = {str(k) for k in fold["val_test_keys"]}
        overlap = train_keys & heldout_keys
        assert not overlap, f"Trial leakage in fold {fold_id}: {sorted(overlap)}"
        all_keys |= train_keys | heldout_keys
        for k in heldout_keys:
            seen_heldout[k] += 1
    multi = [k for k, c in seen_heldout.items() if c > 1]
    if multi:
        raise ValueError(f"Trial keys held out in more than one fold: {sorted(multi)[:8]}")
    not_held = [k for k in all_keys if seen_heldout[k] == 0]
    if not_held:
        raise ValueError(f"Trial keys never held out: {sorted(not_held)[:8]}")


def ensure_patient_split_file(
    manifest_path: Path | str = "samples_manifest.csv",
    output_path: Path | str = DEFAULT_SPLIT_PATH,
    n_splits: int = 3,
    seed: int = 1024,
    force: bool = False,
    verbose: bool = False,
    split_mode: str = "auto",
    weights: Optional[Dict[str, float]] = None,
) -> Path:
    output = Path(output_path)
    if not output.is_absolute():
        output = Path(manifest_path).resolve().parent / output
    if force or not output.exists():
        split = build_patient_splits(
            manifest_path, output, n_splits=n_splits, seed=seed,
            split_mode=split_mode, weights=weights,
        )
    else:
        split = load_patient_splits(output)
    validate_patient_splits(split)
    if verbose:
        print_split_stats(split, output)
    return output


def print_split_stats(split: Dict[str, object], split_path: Optional[Path] = None) -> None:
    if split_path is not None:
        print(f"\nPatient split file: {split_path}")
    mode = split.get("split_mode", "subject" if split.get("split_unit") == "subject_id" else "trial")
    balanced = split.get("balanced_labels", [])
    print(f"Split unit: {split['split_unit']} | mode: {mode} | folds: {split['n_splits']}"
          + (f" | balanced: {balanced}" if balanced else ""))
    for fold in split["folds"]:
        print(f"\nFold {fold['fold']}")
        if "train_keys" in fold:
            train_preview = list(fold["train_keys"])[:6]
            heldout_preview = list(fold["val_test_keys"])[:6]
            extra_train = "" if len(fold["train_keys"]) <= 6 else f" (+{len(fold['train_keys']) - 6})"
            extra_heldout = "" if len(fold["val_test_keys"]) <= 6 else f" (+{len(fold['val_test_keys']) - 6})"
            print(f"  train trials: {train_preview}{extra_train}")
            print(f"  val/test trials: {heldout_preview}{extra_heldout}")
        print(f"  train subjects ({len(fold['train_subjects'])}): {fold['train_subjects']}")
        print(f"  val/test subjects ({len(fold['val_test_subjects'])}): {fold['val_test_subjects']}")
        print(f"  train samples: {fold['train_samples']}")
        print(f"  val/test samples: {fold['val_test_samples']}")
        if "label_distribution" in fold:
            for split_name in ("train", "val_test"):
                dist = fold["label_distribution"][split_name]
                print(f"  {split_name} FMA: {dist.get('fma', {})}")
                for col in CATEGORICAL_LABEL_COLS:
                    if col in dist:
                        print(f"  {split_name} {col}: {dist[col]}")
                for col in REGRESSION_LABEL_COLS:
                    key = col + "_stats"
                    if key in dist:
                        s = dist[key]
                        print(f"  {split_name} {col}: n={s['n']} mean={s['mean']} "
                              f"min={s['min']} max={s['max']}")
                if "source" in dist:
                    print(f"  {split_name} source: {dist['source']}")
        else:
            print(f"  train FMA distribution: {fold['fma_distribution']['train']}")
            print(f"  val/test FMA distribution: {fold['fma_distribution']['val_test']}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict patient-level 3-fold CV splits.")
    parser.add_argument("--manifest", default="data/samples_manifest_real_archive.csv", help="Path to the 29-case real-data manifest.")
    parser.add_argument("--output", default=str(DEFAULT_SPLIT_PATH), help="Output split JSON path.")
    parser.add_argument("--n-splits", type=int, default=3, help="Number of patient-level folds.")
    parser.add_argument("--seed", type=int, default=1024, help="Recorded reproducibility seed.")
    parser.add_argument("--force", action="store_true", help="Regenerate the split file if it already exists.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    split_path = ensure_patient_split_file(
        manifest_path=args.manifest,
        output_path=args.output,
        n_splits=args.n_splits,
        seed=args.seed,
        force=args.force,
        verbose=True,
    )
    split = load_patient_splits(split_path)
    validate_patient_splits(split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
