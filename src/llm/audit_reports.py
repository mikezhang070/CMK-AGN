"""Audit RehabFact-LoRA predictions for factual coverage and unsafe artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .rehabfact import audit_policy_coverage, audit_recommendation, audit_report


def _rate(rows: List[Dict[str, object]], key: str) -> float:
    return sum(bool(row["audit"].get(key, False)) for row in rows) / len(rows) if rows else 0.0


def audit_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    """Attach audits to prediction rows and produce aggregate rates."""
    audited: List[Dict[str, object]] = []
    for row in rows:
        labels = dict(row["labels"])
        final_audit = audit_report(str(row.get("hyp", "")), labels)
        has_split_sections = "fact_section" in row and "model_narrative" in row
        if has_split_sections:
            fact_audit = audit_report(str(row["fact_section"]), labels)
            recommendation_audit = audit_recommendation(str(row["model_narrative"]), labels)
            raw_recommendation_audit = None
            if row.get("raw_model_narrative") is not None:
                raw_recommendation_audit = audit_recommendation(
                    str(row["raw_model_narrative"]), labels
                )
            recommendation_safe = bool(recommendation_audit["safe_recommendation"])
            policy_audit = (
                audit_policy_coverage(str(row["model_narrative"]), labels)
                if "policy_card" in row else None
            )
        else:
            # Legacy predictions contain a single model-generated full report.
            # Their safety is preserved, but recommendation-only safety is N/A.
            fact_audit = final_audit
            recommendation_audit = None
            raw_recommendation_audit = None
            recommendation_safe = bool(final_audit["safe_format"])
            policy_audit = None
        audit = {
            **final_audit,
            "fact_values_complete": fact_audit["all_input_values_correct"],
            "recommendation_safe": recommendation_safe,
            "recommendation_audit": recommendation_audit,
            "raw_recommendation_audit": raw_recommendation_audit,
            "raw_recommendation_safe": (
                bool(raw_recommendation_audit["safe_recommendation"])
                if raw_recommendation_audit else None
            ),
            "safety_fallback_applied": bool(row.get("safety_fallback_applied", False)),
            "policy_audit": policy_audit,
            "policy_covered": bool(policy_audit["policy_covered"]) if policy_audit else None,
            "final_report_safe": bool(fact_audit["all_input_values_correct"] and recommendation_safe),
        }
        audited.append({
            "subject_id": str(row.get("subject_id", "")),
            "audit": audit,
            "generated_tokens": row.get("generated_tokens"),
            "generation_seconds": row.get("generation_seconds"),
        })
    n = len(audited)
    summary = {
        "n": n,
        "nonempty_report_rate": sum(bool(str(row.get("hyp", "")).strip()) for row in rows) / n if n else 0.0,
        "all_input_values_correct_rate": _rate(audited, "all_input_values_correct"),
        "fact_value_complete_rate": _rate(audited, "fact_values_complete"),
        "recommendation_safe_rate": _rate(audited, "recommendation_safe"),
        "final_report_safe_rate": _rate(audited, "final_report_safe"),
        "safe_format_rate": _rate(audited, "safe_format"),
        "placeholder_rate": _rate(audited, "has_placeholder"),
        "thinking_leak_rate": _rate(audited, "has_thinking_leak"),
        "prompt_echo_rate": _rate(audited, "has_prompt_echo"),
        "cross_patient_continuation_rate": _rate(audited, "has_cross_patient_continuation"),
        "unsupported_wrist_score_rate": _rate(audited, "has_unsupported_wrist_score"),
        "unsupported_claim_rate": sum(bool(row["audit"].get("unsupported_claims")) for row in audited) / n if n else 0.0,
        "constraint_echo_rate": _rate(audited, "has_constraint_echo"),
        "raw_recommendation_evaluable_n": sum(
            row["audit"].get("raw_recommendation_safe") is not None for row in audited
        ),
        "raw_recommendation_safe_rate": (
            sum(bool(row["audit"].get("raw_recommendation_safe")) for row in audited
                if row["audit"].get("raw_recommendation_safe") is not None)
            / sum(row["audit"].get("raw_recommendation_safe") is not None for row in audited)
            if any(row["audit"].get("raw_recommendation_safe") is not None for row in audited)
            else None
        ),
        "safety_fallback_rate": _rate(audited, "safety_fallback_applied"),
        "policy_evaluable_n": sum(row["audit"].get("policy_covered") is not None for row in audited),
        "policy_coverage_rate": (
            sum(bool(row["audit"].get("policy_covered")) for row in audited
                if row["audit"].get("policy_covered") is not None)
            / sum(row["audit"].get("policy_covered") is not None for row in audited)
            if any(row["audit"].get("policy_covered") is not None for row in audited) else None
        ),
    }
    return {"summary": summary, "rows": audited}


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit RehabFact-LoRA prediction JSON.")
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = json.loads(args.pred.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit("Prediction file must contain a JSON list.")
    report = audit_rows(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    s = report["summary"]
    policy_coverage = s["policy_coverage_rate"]
    policy_coverage_text = "NA" if policy_coverage is None else f"{policy_coverage:.3f}"
    print(
        "[rehabfact-audit] "
        f"n={s['n']} nonempty={s['nonempty_report_rate']:.3f} "
        f"value_correct={s['all_input_values_correct_rate']:.3f} "
        f"safe_format={s['safe_format_rate']:.3f} "
        f"policy_coverage={policy_coverage_text}"
    )
    print(f"[rehabfact-audit] wrote {args.out}")


if __name__ == "__main__":
    main()
