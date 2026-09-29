"""Prompt templates for the rehab-text LLM.

Both training (data_builder.py) and inference (generate.py) build chat
messages here so that the prompt format stays in one place.
"""
from __future__ import annotations

from typing import Dict, List

from .rehabfact import render_fact_card, render_policy_card


REPORT_TEMPLATE = (
    "患者S{sid}，{gender}性，{age}岁，{disease}，病程{days_post}天，{side_zh}侧偏瘫。"
    "当前FMA手{FMA_UE}分，提示{fma_interp}；"
    "BI评分{BI}分，{bi_interp}。"
    "手部肌张力{hand_tone}级，手分期为Brunnstrom {hand_function}期，"
    "表明{brunn_interp}，可完成{brunn_action}，但{brunn_limit}。"
    "建议{rehab_action}，结合{rehab_modality}，提升手部{rehab_goal}。"
)


SYSTEM_PROMPT = (
    "你是康复医学辅助助手。根据患者的人口学信息和四项临床评估指标"
    "（FMA手部分数、Barthel指数、手部肌张力、Brunnstrom手分期），"
    "输出一段中文康复评估与建议。\n"
    "必须严格使用以下句法骨架，仅替换 {} 内的槽位，不增删句子、不改变标点：\n"
    f"{REPORT_TEMPLATE}\n"
    "硬性要求：必须保留输入数值原样（FMA、BI、肌张力、Brunnstrom 期数）；"
    "病程统一以「天」为单位；用语正式、贴合临床康复师风格；"
    "不要输出任何思考过程或推理文字，直接给出评估文本。"
)


_SIDE_ZH = {"L": "左", "R": "右", "左": "左", "右": "右"}
REHABFACT_SYSTEM_PROMPT = (
    "You are a rehabilitation medicine assistant. The application, not you, will "
    "render the patient identity and four clinical scores deterministically. From "
    "the clinical fact card in the user turn, output only one or two concise Chinese "
    "recommendation sentences. Do not restate patient identifiers, FMA, BI, muscle "
    "tone, Brunnstrom stage, or any score. Do not invent wrist scores, item-level "
    "ADL details, or a full-recovery claim. Output only the recommendation text, "
    "never a fact card, reasoning trace, chat marker, or placeholder."
)
REHABFACT_POLICY_SYSTEM_PROMPT = (
    "You are a rehabilitation medicine assistant. The application, not you, will "
    "render patient identity and the four clinical assessments deterministically. "
    "Use the clinical fact card and clinical policy card to write only one or two "
    "concise Chinese recommendation sentences. Cover every required policy item, "
    "especially ADL participation when marked required and muscle-tone management "
    "when marked required. Do not restate identity, scores, muscle tone, Brunnstrom "
    "stage, or the card itself. Do not invent symptoms or states such as sensation, "
    "pain, cognition, emotion, psychology, wrist scores, item-level ADL details, "
    "or a full-recovery claim. Output only the recommendation text, never reasoning, "
    "a fact card, chat markers, or placeholders."
)


def build_user_message(
    subject_id: str | int,
    demographics: Dict[str, object],
    labels: Dict[str, object],
    rehabfact: bool = False,
    rehabfact_policy: bool = False,
) -> str:
    """Render the user turn as a fixed-field clinical brief."""
    side_raw = str(demographics.get("affected_side", "R"))
    side_zh = _SIDE_ZH.get(side_raw, side_raw)
    message = (
        f"患者编号: S{subject_id}\n"
        f"性别: {demographics.get('gender', '')}    "
        f"年龄: {int(demographics.get('age', 0))}岁\n"
        f"诊断: {demographics.get('disease', '')}    "
        f"病程: {int(demographics.get('days_post', 0))}天    "
        f"偏瘫侧: {side_zh}\n"
        f"FMA手部分数: {int(labels['FMA_UE'])}/20\n"
        f"Barthel指数(BI): {int(labels['BI'])}/100\n"
        f"手部肌张力分级: {labels['hand_tone']}\n"
        f"Brunnstrom手分期: {int(labels['hand_function'])}\n"
        f"请生成康复评估与建议。"
    )
    if rehabfact:
        message += "\n" + render_fact_card(labels) + "\n禁止写入：仅限事实卡支持的内容。\n"
    if rehabfact_policy:
        if not rehabfact:
            raise ValueError("rehabfact_policy requires rehabfact=True")
        message += render_policy_card(labels, demographics) + "\n"
    return message


def build_chat_messages(
    subject_id: str | int,
    demographics: Dict[str, object],
    labels: Dict[str, object],
    rehab_text: str | None = None,
    rehabfact: bool = False,
    rehabfact_policy: bool = False,
) -> List[Dict[str, str]]:
    """Build the chat-format message list.

    If `rehab_text` is given, the assistant turn is appended for SFT.
    Leave `rehab_text=None` at inference time.
    """
    if rehabfact_policy and not rehabfact:
        raise ValueError("rehabfact_policy requires rehabfact=True")
    system_prompt = (
        REHABFACT_POLICY_SYSTEM_PROMPT if rehabfact_policy
        else REHABFACT_SYSTEM_PROMPT if rehabfact
        else SYSTEM_PROMPT
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_message(
            subject_id, demographics, labels,
            rehabfact=rehabfact, rehabfact_policy=rehabfact_policy,
        )},
    ]
    if rehab_text is not None:
        messages.append({"role": "assistant", "content": rehab_text})
    return messages


__all__ = [
    "REPORT_TEMPLATE",
    "REHABFACT_SYSTEM_PROMPT",
    "REHABFACT_POLICY_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "build_user_message",
    "build_chat_messages",
]
