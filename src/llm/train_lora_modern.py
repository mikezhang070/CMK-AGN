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
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from .model_registry import MODEL_REGISTRY, resolve


def _load_model_and_tokenizer(base: str, bf16: bool, trust_remote_code: bool):
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
    # Baichuan2 的远程模型代码使用旧版梯度检查点接口，
    # 不能接受 Transformers 4.35.2 传入的 enable= 参数。
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=False,
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
    model, tok = _load_model_and_tokenizer(
        hf_id, bf16=bf16, trust_remote_code=cfg["trust_remote_code"],
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

    # TRL 0.8.6 不会自动处理 messages 对话列，
    # 使用当前模型 tokenizer 的 chat template 渲染为完整 text。
    if "text" not in ds["train"].column_names:
        if "messages" not in ds["train"].column_names:
            raise KeyError(
                "训练数据既没有 text 列，也没有 messages 列。"
                f" 当前列：{ds['train'].column_names}"
            )

        def _messages_to_text(example):
            messages = example.get("messages")

            if not messages:
                raise ValueError(
                    "发现空 messages："
                    f"subject_id={example.get('subject_id', 'unknown')}"
                )

            rendered = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if not isinstance(rendered, str) or not rendered.strip():
                raise ValueError(
                    "chat template 渲染结果为空："
                    f"subject_id={example.get('subject_id', 'unknown')}"
                )

            return {"text": rendered}

        ds = ds.map(
            _messages_to_text,
            desc="Rendering messages to text",
        )

        print(
            "[train_lora] messages 已转换为 text；"
            f"train={len(ds['train'])}, val={len(ds['val'])}"
        )
        print(
            "[train_lora] text 示例：",
            repr(ds["train"][0]["text"][:300]),
        )

    args.out.mkdir(parents=True, exist_ok=True)
    # TRL 0.8.6 需要已经渲染好的文本列。
    if "text" not in ds["train"].column_names:
        raise KeyError(
            "训练数据中没有 text 列。"
            f" 当前列为：{ds['train'].column_names}。"
            " 请先把对话渲染为完整文本并保存到 text 字段。"
        )

    sft_args = TrainingArguments(
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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=False,
        report_to="none",
        seed=args.seed,
    )

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tok,
    )

    trainer = SFTTrainer(
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
