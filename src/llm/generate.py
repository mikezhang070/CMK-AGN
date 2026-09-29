"""Generate rehab assessment text from labels using a fine-tuned LoRA adapter.

Loads the registered base model + LoRA adapter, iterates over a subject
subset (filtered by split + fold + partition, or by ``--subjects``), and
writes a hyp/ref JSON file consumable by evaluate.py.

Example:
    python -m src.llm.generate \
        --model-id qwen25_3b \
        --adapter checkpoints/llm/qwen25_3b/fold1 \
        --suggestions data/patient_rehab_29subjects.json \
        --split splits/llm_real_text13_patient_3fold.json \
        --fold 1 --partition test \
        --out outputs/llm/qwen25_3b/fold1_test.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .data_builder import (
    _load_suggestions,
    filter_subject_ids_by_source,
    has_human_reference,
    normalize_rehab_text,
    partition_subjects,
)
from .model_registry import resolve
from .prompts import build_chat_messages
from .rehabfact import (
    build_policy_card,
    compose_final_report,
    enforce_policy_safety,
)


def _resolve_subjects(
    suggestions: Dict[str, Dict[str, object]],
    args: argparse.Namespace,
) -> List[str]:
    if args.subjects:
        wanted = [s.strip() for s in args.subjects.split(",") if s.strip()]
        return filter_subject_ids_by_source(suggestions, wanted, args.case_source)

    if args.split is None:
        raise ValueError("Either --subjects or --split must be provided.")
    payload = json.loads(args.split.read_text(encoding="utf-8"))
    train = partition_subjects(payload, args.fold, "train")
    val = partition_subjects(payload, args.fold, "val")
    test = partition_subjects(payload, args.fold, "test")
    if args.partition == "train":
        return train
    if args.partition == "val":
        return val
    if args.partition == "test":
        return test
    if args.partition == "all":
        selected = (
            list(suggestions)
            if args.case_source == "auxiliary"
            else sorted(set(train) | set(val) | set(test), key=lambda s: int(s))
        )
        return filter_subject_ids_by_source(suggestions, selected, args.case_source)
    raise ValueError(f"Unknown partition: {args.partition}")


def _load_model(base: str, adapter: Path, load_4bit: bool, bf16: bool,
                trust_remote_code: bool):
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {"trust_remote_code": trust_remote_code}
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if bf16 else torch.float16
        kwargs["device_map"] = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    model = PeftModel.from_pretrained(base_model, str(adapter))
    model.eval()
    return model, tok


def _resolve_base(args: argparse.Namespace) -> tuple[str, dict]:
    """Pick the HF base id + per-model config from --model-id, --base, or
    the ``model_id.txt`` left in the adapter dir by train_lora."""
    if args.model_id:
        _, cfg = resolve(args.model_id)
        return cfg["hf_id"], cfg
    if args.base:
        _, cfg = resolve(args.base)
        return cfg["hf_id"], cfg
    pin = args.adapter / "model_id.txt"
    if pin.exists():
        _, cfg = resolve(pin.read_text(encoding="utf-8").strip())
        return cfg["hf_id"], cfg
    raise SystemExit(
        "Cannot resolve base model. Provide --model-id, --base, or train "
        "the adapter with the updated train_lora.py (writes model_id.txt)."
    )


_CHAT_TAG_SUFFIXES = (
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "<|user|>",
    "<|assistant|>",
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "</s>",
    "[INST]",
    "[/INST]",
)


def _resolve_eos_ids(tok, cfg: dict) -> List[int]:
    """Build the eos_token_id list passed to model.generate().

    Starts from the tokenizer's eos_token_id and appends any tokens declared
    in ``cfg["extra_eos_tokens"]`` whose ids resolve to something other than
    the unk token. transformers.generate accepts a list and stops on any
    member firing — this is what lets Yi-1.5 actually stop at <|im_end|>.
    """
    ids: List[int] = []
    if tok.eos_token_id is not None:
        ids.append(int(tok.eos_token_id))
    unk_id = tok.unk_token_id
    for name in cfg.get("extra_eos_tokens", []) or []:
        tid = tok.convert_tokens_to_ids(name)
        if tid is None or tid == unk_id:
            continue
        if tid not in ids:
            ids.append(int(tid))
    return ids


def _strip_think_block(text: str) -> str:
    """Remove a leading chain-of-thought block (<think>...</think>).

    Thinking-capable models (DeepSeek-R1, Qwen3 with thinking on) emit their
    reasoning before the actual reply. The reasoning is an internal artifact
    — not part of the fixed-skeleton output — so it must not count toward
    BLEU/ROUGE. Only strips the FIRST block; if nothing follows it (the model
    ran out of tokens mid-thought), the empty remainder is returned so the
    failure stays visible in the metrics.
    """
    m = re.search(r"<think>.*?</think>", text, flags=re.DOTALL)
    if m is not None:
        return text[m.end():].strip()
    return text


def _strip_trailing_chat_tags(text: str) -> str:
    """Defensively cut anything from the first chat-control marker onward.

    skip_special_tokens=True drops registered special tokens but Yi-1.5 et al.
    emit some chat markers as regular text fragments (e.g. when the model
    invents a phantom next turn after running past EOS), and those slip
    through. Trim them so BLEU/ROUGE see only the intended assistant turn.
    """
    cut = len(text)
    for tag in _CHAT_TAG_SUFFIXES:
        idx = text.find(tag)
        if idx != -1 and idx < cut:
            cut = idx
    return text[:cut].rstrip()


def _generate_one(
    model,
    tok,
    messages: List[Dict[str, str]],
    args: argparse.Namespace,
    device: torch.device,
    eos_ids: List[int],
    cfg: dict,
) -> tuple[str, str, int]:
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        **cfg.get("chat_template_kwargs", {}),
    )
    inputs = tok(prompt, return_tensors="pt").to(device)
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tok.pad_token_id,
        eos_token_id=eos_ids if len(eos_ids) > 1 else (eos_ids[0] if eos_ids else None),
    )
    if args.sample:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
    else:
        gen_kwargs.update(do_sample=False, num_beams=args.num_beams)
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0, inputs["input_ids"].shape[-1]:]
    raw_text = tok.decode(new_tokens, skip_special_tokens=True).strip()
    text = raw_text
    if cfg.get("strip_think_block", False):
        text = _strip_think_block(text)
    return _strip_trailing_chat_tags(text), raw_text, int(new_tokens.shape[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate rehab text with LoRA adapter.")
    ap.add_argument("--model-id", default=None,
                    help="Short id from model_registry. If omitted, falls back "
                         "to --base, then to adapter/model_id.txt.")
    ap.add_argument("--base", default=None,
                    help="Raw HF id or local path. Optional if --model-id is set "
                         "or the adapter dir contains model_id.txt.")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--suggestions", type=Path,
                    default=Path("data/patient_rehab_29subjects.json"))
    ap.add_argument("--split", type=Path, default=None,
                    help="Split JSON; required unless --subjects is given.")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--partition", choices=["train", "val", "test", "all"], default="test")
    ap.add_argument("--subjects", default="",
                     help="Optional comma list of subject_ids; overrides --split/--partition.")
    ap.add_argument(
        "--case-source", choices=["all", "human", "auxiliary"], default="all",
        help="Restrict generation to human-reference or rule-generated auxiliary cases. "
             "With --partition all and auxiliary, all auxiliary cases are selected.",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="Override the registry default (e.g. 1024 for "
                         "thinking models like deepseek_r1_llama8b).")
    ap.add_argument("--rehabfact", action="store_true",
                    help="Add the deterministic RehabFact clinical fact card to each prompt.")
    ap.add_argument("--rehabfact-policy", action="store_true",
                    help="Add the RehabFact v3 clinical strategy card; requires --rehabfact.")
    ap.add_argument("--keep-raw", action="store_true",
                    help="Store pre-cleaning text, generated token count, and elapsed time for audit probes.")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--sample", action="store_true",
                    help="Use sampling (temperature/top_p) instead of beam search.")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no-4bit", action="store_true",
                    help="Load base model in bf16/fp16 instead of 4-bit.")
    args = ap.parse_args()
    if args.rehabfact_policy and not args.rehabfact:
        ap.error("--rehabfact-policy requires --rehabfact")

    bf16 = not args.fp16
    hf_id, cfg = _resolve_base(args)
    if args.max_new_tokens is None:
        args.max_new_tokens = int(cfg.get("max_new_tokens", 320))
    suggestions = _load_suggestions(args.suggestions)
    subjects = _resolve_subjects(suggestions, args)
    if not subjects:
        raise SystemExit("No subjects selected for generation.")
    print(f"[generate] hf_id={hf_id}  n_subjects={len(subjects)}")

    model, tok = _load_model(
        hf_id, args.adapter, load_4bit=not args.no_4bit, bf16=bf16,
        trust_remote_code=cfg["trust_remote_code"],
    )
    device = next(model.parameters()).device
    eos_ids = _resolve_eos_ids(tok, cfg)
    print(f"[generate] eos_token_ids={eos_ids}")

    rows: List[Dict[str, object]] = []
    for sid in subjects:
        item = suggestions[sid]
        messages = build_chat_messages(
            subject_id=sid,
            demographics=item["demographics"],
            labels=item["labels"],
            rehab_text=None,
            rehabfact=args.rehabfact,
            rehabfact_policy=args.rehabfact_policy,
        )
        started = time.perf_counter()
        model_narrative, raw_hyp, generated_tokens = _generate_one(
            model, tok, messages, args, device, eos_ids, cfg,
        )
        raw_model_narrative = model_narrative
        safety_fallback_applied = False
        if args.rehabfact_policy:
            model_narrative, safety_fallback_applied = enforce_policy_safety(
                model_narrative, item["labels"], item["demographics"]
            )
        fact_section = None
        hyp = model_narrative
        if args.rehabfact:
            fact_section, hyp = compose_final_report(
                subject_id=sid,
                demographics=item["demographics"],
                labels=item["labels"],
                recommendation=model_narrative,
                policy=args.rehabfact_policy,
            )
        row = {
            "subject_id": sid,
            "source": item.get("source", ""),
            "prompt_user": messages[1]["content"],
            "hyp": hyp,
            "ref": normalize_rehab_text(item) if has_human_reference(item) else None,
            "has_human_reference": has_human_reference(item),
            "labels": item["labels"],
        }
        if args.rehabfact:
            row.update({
                "fact_section": fact_section,
                "model_narrative": model_narrative,
            })
        if args.rehabfact_policy:
            row.update({
                "policy_card": build_policy_card(item["labels"], item["demographics"]),
                "raw_model_narrative": raw_model_narrative,
                "safety_fallback_applied": safety_fallback_applied,
            })
        if args.keep_raw:
            row.update({
                "raw_hyp": raw_hyp,
                "generated_tokens": generated_tokens,
                "generation_seconds": round(time.perf_counter() - started, 4),
            })
        rows.append(row)
        print(f"  S{sid} ({item.get('source','')}): {hyp[:60]}…")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[generate] wrote {len(rows)} predictions → {args.out}")


if __name__ == "__main__":
    main()
