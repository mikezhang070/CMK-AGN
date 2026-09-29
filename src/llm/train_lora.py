"""QLoRA SFT for the rehab-text generator across multiple LLM families.

Designed for a single cloud GPU (RTX 4090D 24 GB / A100 / L4). Uses
bitsandbytes 4-bit NF4 + PEFT LoRA + TRL SFTTrainer with completion-only
loss. Per-model specifics (chat-template response marker, LoRA target
modules, max_seq_length) live in ``model_registry.py``.

Example:
    python -m src.llm.train_lora \
        --model-id qwen25_3b \
        --train data/llm/fold1/train.jsonl \
        --val   data/llm/fold1/val.jsonl \
        --out   checkpoints/llm/qwen25_3b/fold1 \
        --epochs 3 --rank 16
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import List, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

try:
    from trl import SFTConfig
except ImportError:
    SFTConfig = None

try:
    from trl import DataCollatorForCompletionOnlyLM  # removed from trl top-level in TRL >= 1.x
except ImportError:
    DataCollatorForCompletionOnlyLM = None

from .model_registry import MODEL_REGISTRY, resolve
from .trl_compat import select_trl_mode, use_gradient_checkpointing


class _TokenTypeCompatibleSFTTrainer(SFTTrainer):
    """Supply text-only token-type ids for models that require them in training.

    Gemma3ForConditionalGeneration in transformers 5.4 requires
    ``token_type_ids`` while training. The generic TRL text pipeline omits the
    field, so use zero ids (one ordinary text segment) only for registries
    that explicitly opt in.
    """

    def compute_loss(self, model, inputs, *args, **kwargs):
        if "token_type_ids" not in inputs and "input_ids" in inputs:
            inputs = dict(inputs)
            inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
        return super().compute_loss(model, inputs, *args, **kwargs)


def _load_model_and_tokenizer(
    base: str,
    bf16: bool,
    trust_remote_code: bool,
    gradient_checkpointing: bool,
):
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=gradient_checkpointing,
    )
    return model, tok


def _find_subseq(haystack: List[int], needle: List[int]) -> int:
    """Return the start index of the first occurrence of needle in haystack,
    or -1 if absent. needle must be non-empty."""
    if not needle:
        return -1
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return -1


def _resolve_response_template_ids(tok, response_template: str) -> List[int]:
    """Return the token-id sequence to hand to DataCollatorForCompletionOnlyLM.

    SentencePiece tokenizers (Llama-style — Yi-1.5, Mistral, etc.) tokenize a
    marker differently depending on what precedes it: e.g. ``[/INST]`` at the
    start of a string may absorb a leading space byte that it would NOT
    absorb when it appears mid-prompt. If we pass the *string* marker, the
    collator re-tokenizes it standalone and the resulting id list may not
    appear contiguously in the real training sequence → masking silently
    fails, the model learns to predict the user turn too, and inference
    produces prompt-echoing garbage (this is exactly how yi15_6b broke).

    Workaround (TRL's documented recommendation): tokenize a rendered chat
    sample once, locate the marker by string position, then slice the
    matching id range and pass *that id list* to the collator.
    """
    rendered = tok.apply_chat_template(
        [
            {"role": "system", "content": "x"},
            {"role": "user", "content": "y"},
            {"role": "assistant", "content": "z"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    if response_template not in rendered:
        raise RuntimeError(
            "response_template not found in chat-template output.\n"
            f"  template marker : {response_template!r}\n"
            f"  rendered prompt : {rendered!r}\n"
            "Update MODEL_REGISTRY[<model_id>]['response_template'] to a "
            "substring that actually appears in this tokenizer's output."
        )

    prefix = rendered.split(response_template, 1)[0]
    prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    prefix_plus_marker_ids = tok(prefix + response_template, add_special_tokens=False)["input_ids"]
    full_ids = tok(rendered, add_special_tokens=False)["input_ids"]

    if (
        len(prefix_plus_marker_ids) <= len(prefix_ids)
        or prefix_plus_marker_ids[: len(prefix_ids)] != prefix_ids
    ):
        raise RuntimeError(
            "Could not derive a stable id sequence for response_template.\n"
            f"  template marker : {response_template!r}\n"
            f"  prefix tokens   : {prefix_ids}\n"
            f"  prefix+marker   : {prefix_plus_marker_ids}\n"
            "The tokenizer reshuffles tokens across the marker boundary; "
            "pick a different marker substring in MODEL_REGISTRY."
        )

    candidate = prefix_plus_marker_ids[len(prefix_ids) :]
    if _find_subseq(full_ids, candidate) == -1:
        raise RuntimeError(
            "response_template id sequence not found in the tokenized chat.\n"
            f"  template marker : {response_template!r}\n"
            f"  candidate ids   : {candidate}\n"
            f"  full ids        : {full_ids}\n"
            "This is the silent-masking bug. Fix the marker in MODEL_REGISTRY."
        )
    return candidate


def main() -> None:
    ap = argparse.ArgumentParser(description="QLoRA SFT for rehab-text generation.")
    ap.add_argument("--model-id", default=None,
                    help="Short id from model_registry (qwen25_3b, "
                         "mistral7b_v03, glm4_9b, yi15_6b). "
                         "Mutually exclusive with --base.")
    ap.add_argument("--base", default=None,
                    help="Raw HF id or local path. If set, must match a "
                         "registered model so we can resolve LoRA targets "
                         "and response_template.")
    ap.add_argument("--train", type=Path, required=True, help="train.jsonl")
    ap.add_argument("--val", type=Path, required=True, help="val.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="Output adapter dir")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-length", type=int, default=None,
                    help="Override the registry default (e.g. 768 for GLM-4-9B).")
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--fp16", action="store_true",
                    help="Use fp16 instead of bf16 (for T4 where bf16 may be slower).")
    args = ap.parse_args()

    if args.model_id is None and args.base is None:
        ap.error("Provide --model-id (preferred) or --base.")
    selector = args.model_id if args.model_id is not None else args.base
    model_id, cfg = resolve(selector)
    hf_id = cfg["hf_id"]
    max_seq_length = args.max_seq_length or cfg["max_seq_length"]

    bf16 = not args.fp16
    print(f"[train_lora] model_id={model_id}  hf_id={hf_id}  "
          f"max_seq_length={max_seq_length}")
    gradient_checkpointing = use_gradient_checkpointing(cfg)
    model, tok = _load_model_and_tokenizer(
        hf_id,
        bf16=bf16,
        trust_remote_code=cfg["trust_remote_code"],
        gradient_checkpointing=gradient_checkpointing,
    )

    response_template_ids = _resolve_response_template_ids(tok, cfg["response_template"])
    print(
        f"[train_lora] response_template={cfg['response_template']!r} "
        f"→ ids={response_template_ids}"
    )

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        target_modules=cfg["target_modules"],
        task_type="CAUSAL_LM",
    )

    print(f"[train_lora] loading datasets: train={args.train}  val={args.val}")
    ds = load_dataset(
        "json",
        data_files={"train": str(args.train), "val": str(args.val)},
    )

    args.out.mkdir(parents=True, exist_ok=True)
    trainer_class = (
        _TokenTypeCompatibleSFTTrainer
        if cfg.get("requires_token_type_ids", False)
        else SFTTrainer
    )

    trl_mode = select_trl_mode(
        has_sft_config=SFTConfig is not None,
        has_completion_collator=DataCollatorForCompletionOnlyLM is not None,
    )

    # Shared SFTConfig kwargs across both TRL generations.
    _common = dict(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        optim="paged_adamw_8bit",
        bf16=bf16,
        fp16=not bf16,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=gradient_checkpointing,
        report_to="none",
        seed=args.seed,
    )
    if gradient_checkpointing:
        _common["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    if trl_mode == "modern_completion_only":
        # TRL >= 1.x removed the collator; completion_only_loss masks natively.
        # Note: max_seq_length was renamed to max_length in newer SFTConfig.
        print("[train_lora] TRL >=1.x (no DataCollatorForCompletionOnlyLM) — "
              "using SFTConfig.completion_only_loss=True")
        seq_kw = ("max_seq_length" if "max_seq_length"
                  in inspect.signature(SFTConfig).parameters else "max_length")
        sft_kwargs = dict(_common)
        sft_kwargs[seq_kw] = max_seq_length
        sft_kwargs["completion_only_loss"] = True
        sft_args = SFTConfig(**sft_kwargs)
        trainer = trainer_class(
            model=model,
            args=sft_args,
            train_dataset=ds["train"],
            eval_dataset=ds["val"],
            peft_config=lora,
            processing_class=tok,
        )
    else:
        # Baichuan's isolated environment intentionally retains an older TRL.
        # It has no SFTConfig, so render the conversational records explicitly
        # and use its completion-only collator with plain TrainingArguments.
        print("[train_lora] legacy TRL detected — using DataCollatorForCompletionOnlyLM")

        def _render_messages(example):
            return {
                "text": tok.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            }

        ds = ds.map(_render_messages, remove_columns=ds["train"].column_names)
        legacy_kwargs = dict(_common)
        legacy_kwargs.pop("eval_strategy")
        legacy_kwargs.pop("gradient_checkpointing_kwargs", None)
        legacy_kwargs.pop("optim")
        legacy_kwargs["optim"] = "paged_adamw_8bit"
        eval_name = (
            "eval_strategy"
            if "eval_strategy" in inspect.signature(TrainingArguments).parameters
            else "evaluation_strategy"
        )
        legacy_kwargs[eval_name] = "epoch"
        sft_args = TrainingArguments(**legacy_kwargs)
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids,
            tokenizer=tok,
        )
        trainer = trainer_class(
            model=model,
            args=sft_args,
            train_dataset=ds["train"],
            eval_dataset=ds["val"],
            peft_config=lora,
            tokenizer=tok,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            packing=False,
            data_collator=collator,
        )

    print("[train_lora] starting training …")
    trainer.train()

    print(f"[train_lora] saving best adapter → {args.out}")
    trainer.save_model(str(args.out))
    tok.save_pretrained(str(args.out))
    # Persist the model_id so generate.py can resolve hf_id without --model-id.
    (args.out / "model_id.txt").write_text(model_id, encoding="utf-8")

    print("[train_lora] done.")


if __name__ == "__main__":
    main()
