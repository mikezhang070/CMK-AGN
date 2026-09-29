"""Clinical fact cards and lightweight safety auditing for RehabFact-LoRA.

The module deliberately uses only the four fields available at inference.
It does not infer wrist scores, item-level ADL dependence, or other facts
that are absent from the prompt.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List


def _as_int(labels: Dict[str, object], name: str) -> int:
    return int(labels[name])


def _fma_interpretation(score: int) -> str:
    if score <= 4:
        return "手部运动功能严重受限"
    if score <= 10:
        return "手部运动功能中度受限"
    if score <= 15:
        return "手部运动功能轻度受限"
    if score <= 19:
        return "手部运动功能轻微受限"
    return "手部运动功能接近正常"


def _bi_interpretation(score: int) -> str:
    if score <= 20:
        return "日常生活高度依赖"
    if score <= 60:
        return "日常生活中度依赖"
    if score <= 90:
        return "日常生活基本自理，但部分活动仍可能需要协助"
    return "日常生活基本独立"


def _tone_interpretation(tone: str) -> str:
    mapping = {
        "0": "手部肌张力正常",
        "1": "手部肌张力轻度增高",
        "1+": "手部肌张力轻度增高",
        "2": "手部肌张力明显增高",
        "3": "手部肌张力显著增高",
        "4": "手部肌张力僵硬",
    }
    return mapping.get(str(tone).strip(), f"手部肌张力为{tone}级")


_BRUNNSTROM = {
    1: ("尚无有效随意运动", "尚不能完成有效抓握"),
    2: ("开始出现少量共同运动或联合反应", "主动抓握能力仍有限"),
    3: ("可在共同运动模式下完成部分主动运动", "抓握和手指分离能力仍受限"),
    4: ("开始出现部分脱离共同运动的主动运动", "复杂抓握和手指分离仍需训练"),
    5: ("可完成较复杂的主动手部运动", "精细操作和协调性仍可能受限"),
    6: ("可完成多种抓握动作", "速度、协调性和耐力仍可能受限"),
}


def _rehab_suggestion(stage: int) -> str:
    if stage <= 3:
        return "以关节活动度维持、诱发主动运动和任务导向抓握训练为主"
    if stage <= 5:
        return "以脱离共同运动、抓握和手指分离训练为主"
    return "以精细操作、协调性和耐力训练为主"


def build_fact_card(labels: Dict[str, object]) -> Dict[str, object]:
    """Return deterministic, prompt-supported facts for one patient."""
    fma = _as_int(labels, "FMA_UE")
    bi = _as_int(labels, "BI")
    tone = str(labels["hand_tone"]).strip()
    stage = _as_int(labels, "hand_function")
    if not 0 <= fma <= 20:
        raise ValueError(f"FMA_UE must be in [0, 20], got {fma}")
    if not 0 <= bi <= 100:
        raise ValueError(f"BI must be in [0, 100], got {bi}")
    if stage not in _BRUNNSTROM:
        raise ValueError(f"Brunnstrom hand stage must be in [1, 6], got {stage}")

    brunn_interp, brunn_limit = _BRUNNSTROM[stage]
    # These states/symptoms are not provided by the four structured inputs.
    # They are therefore forbidden in both generated facts and recommendations.
    forbidden = [
        "完全恢复",
        "输入中未提供的腕部评分",
        "输入中未提供的具体ADL条目",
        "感觉",
        "疼痛",
        "认知",
        "情绪",
        "心理",
    ]
    if fma <= 4:
        forbidden.append("轻度运动功能受限")
    if stage < 6:
        forbidden.append("所有抓握均能完成")
    return {
        "FMA_UE": fma,
        "BI": bi,
        "hand_tone": tone,
        "hand_function": stage,
        "fma_interpretation": _fma_interpretation(fma),
        "bi_interpretation": _bi_interpretation(bi),
        "tone_interpretation": _tone_interpretation(tone),
        "brunnstrom_interpretation": brunn_interp,
        "brunnstrom_limitation": brunn_limit,
        "recommendation_direction": _rehab_suggestion(stage),
        "forbidden_claims": forbidden,
    }


def render_fact_card(labels: Dict[str, object]) -> str:
    """Render a compact Chinese fact card for the model input."""
    card = build_fact_card(labels)
    return "临床事实卡（不得改写数值或与其矛盾）：\n" + json.dumps(
        card, ensure_ascii=False, separators=(",", "：")
    )


def render_canonical_report(
    subject_id: str | int,
    demographics: Dict[str, object],
    labels: Dict[str, object],
) -> str:
    """Create a complete deterministic reference report from supported facts."""
    return (
        render_fact_section(subject_id, demographics, labels)
        + render_recommendation_target(labels)
    )


def render_fact_section(
    subject_id: str | int,
    demographics: Dict[str, object],
    labels: Dict[str, object],
) -> str:
    """Render the non-negotiable, programmatic clinical fact section.

    This section is assembled directly from structured input rather than by the
    language model. It is therefore authoritative for identifiers and scores.
    """
    card = build_fact_card(labels)
    side = {"L": "左", "R": "右", "左": "左", "右": "右"}.get(
        str(demographics.get("affected_side", "")), str(demographics.get("affected_side", ""))
    )
    return (
        f"患者S{subject_id}，{demographics.get('gender', '')}性，"
        f"{int(demographics.get('age', 0))}岁，{demographics.get('disease', '')}，"
        f"病程{int(demographics.get('days_post', 0))}天，{side}侧偏瘫。"
        f"当前FMA手{card['FMA_UE']}分，提示{card['fma_interpretation']}；"
        f"BI评分{card['BI']}分，{card['bi_interpretation']}。"
        f"手部肌张力{card['hand_tone']}级，Brunnstrom手分期为{card['hand_function']}期，"
        f"提示{card['brunnstrom_interpretation']}，{card['brunnstrom_limitation']}。"
    )


def render_recommendation_target(labels: Dict[str, object]) -> str:
    """Return the recommendation-only supervision target for RehabFact v2."""
    card = build_fact_card(labels)
    return f"建议{card['recommendation_direction']}，并结合患者耐受情况循序渐进调整训练。"


_STAGE_POLICY = {
    1: {
        "text": "以关节活动度维持、诱发主动运动和抓握准备为重点",
        "keywords": ("关节活动度", "主动运动", "抓握准备"),
    },
    2: {
        "text": "以诱发主动运动、抑制不必要代偿和基础抓握准备为重点",
        "keywords": ("主动运动", "基础抓握", "抓握准备"),
    },
    3: {
        "text": "以脱离共同运动后的任务导向抓握和手指分离准备为重点",
        "keywords": ("脱离共同运动", "抓握", "手指分离"),
    },
    4: {
        "text": "以抓握、手指分离和精细控制训练为重点",
        "keywords": ("抓握", "手指分离", "精细控制"),
    },
    5: {
        "text": "以复杂抓握、手指分离和精细操作训练为重点",
        "keywords": ("复杂抓握", "手指分离", "精细操作"),
    },
    6: {
        "text": "以精细操作、协调性、耐力和实际生活任务中的持续使用为重点",
        "keywords": ("精细操作", "协调", "耐力", "生活任务"),
    },
}


def _fma_policy(score: int) -> Dict[str, object]:
    if score <= 4:
        return {
            "text": "促进患侧手主动参与，并建立基础抓握准备",
            "keywords": ("患侧手", "主动", "抓握"),
        }
    if score <= 10:
        return {
            "text": "扩大主动运动范围并改善抓握控制",
            "keywords": ("主动运动", "抓握"),
        }
    if score <= 15:
        return {
            "text": "改善抓握质量、手指控制和任务完成效率",
            "keywords": ("抓握", "手指", "任务"),
        }
    return {
        "text": "提升精细操作、协调性和患侧手持续使用",
        "keywords": ("精细操作", "协调", "持续使用"),
    }


def build_policy_card(
    labels: Dict[str, object], demographics: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """Build a deterministic clinical strategy card for RehabFact v3.

    The policy is intentionally narrow: every field comes from one of the four
    structured assessments, while ``days_post`` is background only.  It does
    not infer a disease phase, symptoms, or item-level ADL dependence.
    """
    card = build_fact_card(labels)
    fma = int(card["FMA_UE"])
    bi = int(card["BI"])
    tone = str(card["hand_tone"])
    stage = int(card["hand_function"])
    del demographics  # Explicitly avoid any unvalidated time-window rule.

    adl_required = bi <= 60
    tone_required = tone != "0"
    return {
        "hand_function_goal": _fma_policy(fma),
        "adl_requirement": {
            "required": adl_required,
            "text": (
                "将患侧手纳入日常生活活动参与和任务训练"
                if adl_required
                else "在安全前提下将患侧手逐步融入日常生活任务"
            ),
            "keywords": ("日常生活", "任务训练", "功能活动"),
        },
        "tone_precaution": {
            "required": tone_required,
            "text": (
                "训练中注意肌张力管理与动作质量，避免以速度代偿完成动作"
                if tone_required
                else "训练中保持动作质量并观察肌张力变化"
            ),
            "keywords": ("肌张力管理", "肌张力", "动作质量"),
        },
        "brunnstrom_focus": _STAGE_POLICY[stage],
        "pace": {
            "text": "结合患者耐受情况分级递进，并根据训练反应调整训练负荷",
            "keywords": ("耐受", "分级递进", "训练负荷"),
        },
        "required_coverage": [
            "hand_goal", "stage", "pace",
            *( ["adl"] if adl_required else [] ),
            *( ["tone"] if tone_required else [] ),
        ],
        "forbidden_claims": list(card["forbidden_claims"]),
    }


def render_policy_card(
    labels: Dict[str, object], demographics: Dict[str, object] | None = None,
) -> str:
    """Render a machine-readable v3 strategy card for the model prompt."""
    return "临床策略卡（仅覆盖下列已知要求，不得补充未提供的症状或诊断）：\n" + json.dumps(
        build_policy_card(labels, demographics), ensure_ascii=False, separators=(",", "：")
    )


def render_policy_recommendation_target(
    labels: Dict[str, object], demographics: Dict[str, object] | None = None,
) -> str:
    """Deterministic recommendation supervision target for RehabFact v3."""
    policy = build_policy_card(labels, demographics)
    clauses = [
        str(policy["hand_function_goal"]["text"]),
        str(policy["brunnstrom_focus"]["text"]),
    ]
    if bool(policy["adl_requirement"]["required"]):
        clauses.append(str(policy["adl_requirement"]["text"]))
    if bool(policy["tone_precaution"]["required"]):
        clauses.append(str(policy["tone_precaution"]["text"]))
    clauses.append(str(policy["pace"]["text"]))
    return "建议" + "；".join(clauses) + "。"


def enforce_policy_safety(
    recommendation: str,
    labels: Dict[str, object],
    demographics: Dict[str, object] | None = None,
) -> tuple[str, bool]:
    """Fail closed when a v3 model recommendation violates the safety gate.

    The raw model text is retained by the caller for audit. The user-visible
    recommendation falls back to the deterministic policy target whenever the
    model adds unsupported claims, constraint text, or other unsafe artifacts.
    """
    text = str(recommendation)
    audit = audit_recommendation(text, labels)
    coverage = audit_policy_coverage(text, labels, demographics)
    if audit["safe_recommendation"] and coverage["policy_covered"]:
        return str(recommendation).strip(), False
    return render_policy_recommendation_target(labels, demographics), True


def audit_policy_coverage(
    text: str,
    labels: Dict[str, object],
    demographics: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """Check whether a v3 recommendation covers every case-required policy."""
    policy = build_policy_card(labels, demographics)
    requirements = {
        "hand_goal": tuple(policy["hand_function_goal"]["keywords"]),
        "stage": tuple(policy["brunnstrom_focus"]["keywords"]),
        "pace": tuple(policy["pace"]["keywords"]),
    }
    if bool(policy["adl_requirement"]["required"]):
        requirements["adl"] = tuple(policy["adl_requirement"]["keywords"])
    if bool(policy["tone_precaution"]["required"]):
        requirements["tone"] = tuple(policy["tone_precaution"]["keywords"])

    matched = {
        name: any(keyword in text for keyword in keywords)
        for name, keywords in requirements.items()
    }
    missing = [name for name, present in matched.items() if not present]
    return {
        "required_coverage": list(requirements),
        "coverage": matched,
        "missing_coverage": missing,
        "policy_covered": not missing,
    }


def compose_final_report(
    subject_id: str | int,
    demographics: Dict[str, object],
    labels: Dict[str, object],
    recommendation: str,
    policy: bool = False,
) -> tuple[str, str]:
    """Combine programmatic facts with the model-owned recommendation section."""
    fact_section = render_fact_section(subject_id, demographics, labels)
    narrative = re.sub(r"^(?:建议\s*[:：]?\s*)", "", str(recommendation).strip())
    if not narrative:
        fallback = (
            render_policy_recommendation_target(labels, demographics)
            if policy else render_recommendation_target(labels)
        )
        narrative = fallback.removeprefix("建议")
    if narrative[-1] not in "。！？!?":
        narrative += "。"
    return fact_section, f"{fact_section}建议{narrative}"


def _field_value_issue(text: str, field: str, expected: str, pattern: str) -> str | None:
    values = re.findall(pattern, text, flags=re.IGNORECASE)
    if not values:
        return f"missing_{field}"
    found = {str(value) for value in values}
    if expected not in found or found != {expected}:
        return field
    return None


def _has_constraint_echo(text: str) -> bool:
    """Detect a model copying the safety instructions into the report.

    A recommendation may legitimately contain a word such as ``avoid``.  The
    failure pattern observed in v3 is narrower: an instruction marker followed
    by phrases from the internal forbidden-claim list or by ``input not
    provided`` wording.  This is a report-format failure, not a clinical fact
    assertion, but it must still fail the safety gate.
    """
    instruction_markers = (
        "\u907f\u514d", "\u4e0d\u8981", "\u7981\u6b62", "\u4e0d\u5f97", "\u4e0d\u5e94",
    )
    echoed_instruction = (
        "\u8f93\u5165\u4e2d\u672a\u63d0\u4f9b" in text
        or "\u7b49\u8868\u8ff0" in text
        or "\u4e0d\u652f\u6301\u7684\u8868\u8ff0" in text
    )
    return any(marker in text for marker in instruction_markers) and echoed_instruction


def audit_report(text: str, labels: Dict[str, object]) -> Dict[str, object]:
    """Check numeric fact coverage and common unsafe generation artifacts."""
    card = build_fact_card(labels)
    issues: List[str] = []
    checks = {
        "FMA_UE": (
            str(card["FMA_UE"]),
            r"FMA(?:手(?:部分数|部评分|评分)?|评分|手部分数)?\s*(?:为|是)?\s*(\d+)(?:\s*/\s*20)?",
        ),
        "BI": (
            str(card["BI"]),
            r"(?:BI\s*评分|Barthel(?:指数)?(?:\s*\(BI\))?)\s*(?:为|是)?\s*(\d+)(?:\s*/\s*100)?",
        ),
        "hand_tone": (
            str(card["hand_tone"]),
            r"(?:手部)?肌张力(?:分级)?\s*(?:为|是)?\s*(\d\+?)\s*级",
        ),
        "hand_function": (
            str(card["hand_function"]),
            r"Brunnstrom(?:手)?(?:分期)?\s*(?:为|是)?\s*(\d+)\s*期",
        ),
    }
    for field, (expected, pattern) in checks.items():
        issue = _field_value_issue(text, field, expected, pattern)
        if issue:
            issues.append(issue)

    unsupported = [claim for claim in card["forbidden_claims"] if claim in text]
    has_constraint_echo = _has_constraint_echo(text)
    if has_constraint_echo:
        unsupported.append("constraint_echo")
    has_placeholder = bool(re.search(r"\{[A-Za-z_][A-Za-z0-9_]*\}", text))
    has_thinking_leak = bool(re.search(r"</?think>", text, flags=re.IGNORECASE))
    has_prompt_echo = "患者编号" in text
    has_cross_patient_continuation = len(re.findall(r"患者S\d+", text)) > 1
    has_unsupported_wrist_score = bool(re.search(r"腕部\s*\d+\s*分", text))
    return {
        "all_input_values_correct": not issues,
        "value_issues": issues,
        "has_placeholder": has_placeholder,
        "has_thinking_leak": has_thinking_leak,
        "has_prompt_echo": has_prompt_echo,
        "has_cross_patient_continuation": has_cross_patient_continuation,
        "has_unsupported_wrist_score": has_unsupported_wrist_score,
        "has_constraint_echo": has_constraint_echo,
        "unsupported_claims": unsupported,
        "safe_format": not unsupported
        and not has_placeholder
        and not has_thinking_leak
        and not has_prompt_echo
        and not has_cross_patient_continuation
        and not has_unsupported_wrist_score,
    }


def audit_recommendation(text: str, labels: Dict[str, object]) -> Dict[str, object]:
    """Audit the model-owned recommendation section independently of facts.

    RehabFact v2 does not allow the model to repeat numerical assessments. That
    prevents a generated recommendation from contradicting the programmatic
    fact section in the complete report.
    """
    card = build_fact_card(labels)
    unsupported = [claim for claim in card["forbidden_claims"] if claim in text]
    has_constraint_echo = _has_constraint_echo(text)
    if has_constraint_echo:
        unsupported.append("constraint_echo")
    has_placeholder = bool(re.search(r"\{[A-Za-z_][A-Za-z0-9_]*\}", text))
    has_thinking_leak = bool(re.search(r"</?think>", text, flags=re.IGNORECASE))
    has_prompt_echo = "患者编号" in text
    has_cross_patient_continuation = len(re.findall(r"患者S\d+", text)) > 0
    has_unsupported_wrist_score = bool(re.search(r"腕部\s*\d+\s*分", text))
    has_metric_restatement = bool(re.search(
        r"FMA|BI\s*评分|Barthel|肌张力(?:分级)?\s*(?:为|是)?\s*\d\+?\s*级|Brunnstrom",
        text,
        flags=re.IGNORECASE,
    ))
    safe_recommendation = not (
        unsupported
        or has_placeholder
        or has_thinking_leak
        or has_prompt_echo
        or has_cross_patient_continuation
        or has_unsupported_wrist_score
        or has_constraint_echo
        or has_metric_restatement
    )
    return {
        "unsupported_claims": unsupported,
        "has_placeholder": has_placeholder,
        "has_thinking_leak": has_thinking_leak,
        "has_prompt_echo": has_prompt_echo,
        "has_cross_patient_continuation": has_cross_patient_continuation,
        "has_unsupported_wrist_score": has_unsupported_wrist_score,
        "has_constraint_echo": has_constraint_echo,
        "has_metric_restatement": has_metric_restatement,
        "safe_recommendation": safe_recommendation,
    }


__all__ = [
    "audit_policy_coverage",
    "audit_recommendation",
    "audit_report",
    "build_fact_card",
    "build_policy_card",
    "compose_final_report",
    "enforce_policy_safety",
    "render_canonical_report",
    "render_fact_card",
    "render_fact_section",
    "render_policy_card",
    "render_policy_recommendation_target",
    "render_recommendation_target",
]
