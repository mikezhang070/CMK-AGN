"""Per-model fine-tuning config for the 4-LLM comparison pipeline.

Each entry maps a short ``model_id`` (used in checkpoint / output paths)
to everything that differs across model families:

  - ``hf_id``               HuggingFace repo id (or local path) for the base model
  - ``response_template``   String marker for ``DataCollatorForCompletionOnlyLM``
                            so loss is computed only on the assistant turn.
                            MUST appear verbatim once in the tokenized prompt
                            produced by ``tokenizer.apply_chat_template(...,
                            add_generation_prompt=False)``.
  - ``target_modules``      LoRA injection points. Llama-style models share
                            q/k/v/o_proj + gate/up/down_proj (Qwen2.5,
                            Mistral-7B, Yi-1.5, and GLM-4-0414 which uses
                            ``Glm4ForCausalLM`` since transformers 4.52);
                            the older ChatGLM-based GLM-4-Chat used a fused
                            ``query_key_value`` linear.
  - ``max_seq_length``      Per-model default. GLM-4-9B is capped at 768 to
                            fit on a 24 GB RTX 4090D under QLoRA 4-bit.
  - ``trust_remote_code``   Required by GLM-4 (custom modeling).
  - ``extra_eos_tokens``    Optional list of additional token strings that
                            should terminate generation. Needed when the
                            tokenizer's ``eos_token`` does not match the
                            chat-template's turn boundary (e.g. Yi-1.5's
                            ``<|im_end|>``), otherwise generate.py runs to
                            ``max_new_tokens`` and fabricates a next turn.

All four candidates are usable with bitsandbytes 4-bit NF4 + PEFT LoRA on
a single RTX 4090D (24 GB).
"""
from __future__ import annotations

from typing import Tuple

# Llama-style projection names shared by Qwen2.5, Mistral, Yi-1.5, GLM-4-0414.
_LLAMA_STYLE_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


MODEL_REGISTRY: dict[str, dict] = {
    "qwen25_3b": {
        # Pre-quantized bnb-NF4 variant (~1.8 GB) — keeps tokenizer & module
        # names identical to Qwen/Qwen2.5-3B-Instruct so response_template
        # and target_modules below remain valid, but cuts HF cache footprint
        # to fit the 25 GB cloud disk budget.
        "hf_id": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "response_template": "<|im_start|>assistant\n",
        "target_modules": list(_LLAMA_STYLE_TARGETS),
        "max_seq_length": 1024,
        "trust_remote_code": True,
        # Baichuan's remote model code implements the pre-Transformers-5
        # gradient-checkpointing hook.  Its 4-bit QLoRA run fits on the A6000
        # without checkpointing, so disable it rather than patching model code.
        "disable_gradient_checkpointing": True,
        "extra_eos_tokens": [],
    },
    "mistral7b_v03": {
        # Replaces deepseek_r1_distill_qwen_7b: R1-Distill is a think-then-answer
        # model that burns its entire max_new_tokens budget on chain-of-thought
        # before emitting the actual reply, which is incompatible with our
        # fixed-句法骨架 SFT setup. Mistral-7B-Instruct-v0.3 is a clean
        # LlamaForCausalLM, Apache 2.0 (no HF gating), and ~5 GB at bnb-NF4.
        # Chat template renders the assistant boundary as `[/INST]` (then a
        # space, then the response, terminated with </s>). The SentencePiece
        # boundary at `[/INST]` is handled by train_lora.py's id-sequence
        # search the same way Yi's <|im_start|>assistant\n marker is.
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "response_template": "[/INST]",
        "target_modules": list(_LLAMA_STYLE_TARGETS),
        "max_seq_length": 1024,
        "trust_remote_code": False,
        "extra_eos_tokens": [],
    },
    "glm4_9b": {
        # Pre-quantized bnb-NF4 variant (~5 GB) of GLM-4-9B-0414, which uses
        # the Llama-style ``Glm4ForCausalLM`` architecture introduced in
        # transformers 4.52 — NOT the older fused-QKV ChatGLM modeling used
        # by GLM-4-9B-Chat. target_modules / response_template below reflect
        # the 0414 arch; if you swap back to a Chat-based GLM-4 you must
        # restore ``query_key_value`` etc.
        "hf_id": "unsloth/GLM-4-9B-0414-bnb-4bit",
        "response_template": "<|assistant|>\n",
        "target_modules": list(_LLAMA_STYLE_TARGETS),
        "max_seq_length": 768,
        "trust_remote_code": True,
        "extra_eos_tokens": [],
    },
    "yi15_6b": {
        # Standard LlamaForCausalLM under the hood and a ChatML-style
        # chat_template (same "<|im_start|>assistant\n" marker as Qwen2.5),
        # so it slots into the existing LoRA / response-template plumbing
        # with no special casing. NOTE: this entry originally replaced
        # Baichuan2-7B-Chat, whose HF-repo tokenizer shipped without a
        # chat_template; the SERVER-LOCAL Baichuan2-7B-Chat build was
        # empirically verified to train fine on 2026-08-06 (chat_template
        # present, response marker <reserved_107> resolves) — see
        # the baichuan2_7b entry below.
        # NOTE: Unsloth maintains a bnb-4bit pre-quant of the *base*
        # Yi-1.5-6B but NOT of the Chat variant, so we use the official
        # 01-ai fp16 repo (~12 GB cache) and let bitsandbytes 4-bit-quantize
        # at load time. Pair with DEEP_CLEAN=1 on tight 25 GB cloud disks.
        # Yi-1.5-Chat's tokenizer eos_token is NOT <|im_end|>, so add it as
        # an extra stop token for generate.py — otherwise the model runs to
        # max_new_tokens and fabricates a phantom next user/assistant turn.
        "hf_id": "01-ai/Yi-1.5-6B-Chat",
        "response_template": "<|im_start|>assistant\n",
        "target_modules": list(_LLAMA_STYLE_TARGETS),
        "max_seq_length": 1024,
        "trust_remote_code": False,
        "extra_eos_tokens": ["<|im_end|>"],
    },
    # ------------------------------------------------------------------ #
    # Server-local models (deployed under /root/offline/models/).         #
    # QLoRA never writes to these paths: the base is loaded read-only and  #
    # only the LoRA adapters are saved under results/llm/checkpoints/.     #
    # R1-Distill and Baichuan2 were historically excluded (CoT token burn, #
    # missing chat_template) but are re-registered below for EMPIRICAL     #
    # TEST — see the section comment above deepseek_r1_llama8b.            #
    # ------------------------------------------------------------------ #
    "qwen3_8b": {
        # Qwen3 keeps the ChatML format of Qwen2.5 ("<|im_start|>assistant\n"),
        # so this slots into the existing plumbing unchanged. Local path —
        # no HuggingFace download needed. Qwen3 is a thinking model: its chat
        # template renders thinking-mode instructions by default, so the model
        # wastes max_new_tokens on a <think> block and the real reply gets
        # truncated (observed 2026-08-06 in the formal fold1 run). Generation
        # must disable it via chat_template_kwargs={"enable_thinking": False}.
        "hf_id": "/root/offline/models/Qwen3-8B",
        "response_template": "<|im_start|>assistant\n",
        "target_modules": list(_LLAMA_STYLE_TARGETS),
        "max_seq_length": 1024,
        "trust_remote_code": False,
        "extra_eos_tokens": [],
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "llama31_8b": {
        # Llama-3.1 chat template: system/user/assistant header tokens.
        "hf_id": "/root/offline/models/Meta-Llama-3.1-8B-Instruct",
        "response_template": "<|start_header_id|>assistant<|end_header_id|>",
        "target_modules": list(_LLAMA_STYLE_TARGETS),
        "max_seq_length": 1024,
        "trust_remote_code": False,
        "extra_eos_tokens": [],
    },
    "gemma3_4b": {
        # Gemma-3 turn markers: <start_of_turn>model\n ends at <end_of_turn>.
        "hf_id": "/root/offline/models/Gemma-3-4B-IT",
        "response_template": "<start_of_turn>model\n",
        "target_modules": list(_LLAMA_STYLE_TARGETS),
        "max_seq_length": 1024,
        "trust_remote_code": False,
        "extra_eos_tokens": ["<end_of_turn>"],
        # The deployed Gemma3ForConditionalGeneration requires this input
        # during training even for text-only batches (transformers 5.4.0).
        "requires_token_type_ids": True,
    },
    "glm4_9b_chat": {
        # Server-local GLM-4-9B-Chat uses Llama/Qwen-style attention names,
        # but a fused MLP gate/up projection. Verified against the deployed
        # checkpoint on 2026-08-06: q/k/v/o_proj, gate_up_proj, down_proj.
        "hf_id": "/root/offline/models/glm-4-9b-chat-hf",
        "response_template": "<|assistant|>",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_up_proj", "down_proj",
        ],
        "max_seq_length": 768,
        "trust_remote_code": True,
        "extra_eos_tokens": [],
    },
    # ------------------------------------------------------------------ #
    # Deliberately-unregistered models, re-registered for EMPIRICAL TEST.  #
    # Historical notes in this file said R1-Distill burns its token budget #
    # on chain-of-thought and Baichuan2's tokenizer lacks a chat_template. #
    # The user chose to verify both claims by running them. If the smoke   #
    # test passes, these stay; if it reproduces the issue, drop them again.#
    # ------------------------------------------------------------------ #
    "deepseek_r1_llama8b": {
        # EMPIRICAL (2026-08-06): this model does NOT use Llama-3.1 header
        # tokens. Its chat template renders DeepSeek-style fullwidth markers
        # (<｜User｜> / <｜Assistant｜> / <｜end▁of▁sentence｜>) — verified via
        # _resolve_response_template_ids, which rejected the old
        # "<|start_header_id|>..." marker at runtime. The template EOS also
        # differs from the tokenizer eos_token, so it is registered as an
        # extra stop. Thinking cannot be switched off (unlike Qwen3), so
        # generation budget is raised and the <think> block is stripped
        # post-hoc by generate.py (strip_think_block) before evaluation.
        "hf_id": "/root/offline/models/DeepSeek-R1-Distill-Llama-8B",
        "response_template": "<｜Assistant｜>",
        "target_modules": list(_LLAMA_STYLE_TARGETS),
        "max_seq_length": 1024,
        "trust_remote_code": False,
        "extra_eos_tokens": ["<｜end▁of▁sentence｜>"],
        "max_new_tokens": 1024,
        "strip_think_block": True,
    },
    "baichuan2_7b": {
        # Llama-style projections (q/k/v/o + gate/up/down).
        # Empirical (2026-08-06): the SERVER-LOCAL Baichuan2-7B-Chat build is
        # confirmed trainable — chat_template present, response marker
        # <reserved_107> resolves to id 196, and a 1-epoch smoke SFT finished
        # cleanly (train_loss 2.50, eval_loss 2.21). This refutes the old
        # "no chat_template" note. GENERATION is still untested: whether this
        # model stays in the comparison is decided by the formal 3-fold run's
        # BLEU/ROUGE + eyeball check of the generated skeleton text.
        "hf_id": "/root/offline/models/Baichuan2-7B-Chat",
        "response_template": "<reserved_107>",
        "target_modules": list(_LLAMA_STYLE_TARGETS),
        "max_seq_length": 1024,
        "trust_remote_code": True,
        "extra_eos_tokens": [],
        # Baichuan's custom model implementation is incompatible with the
        # ``enable=`` keyword used by transformers 5.x when turning gradient
        # checkpointing on. A 7B model in 4-bit fits on the A6000 without it.
        "disable_gradient_checkpointing": True,
    },
}


def list_model_ids() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def resolve(model_id_or_hf: str) -> Tuple[str, dict]:
    """Resolve either a short model_id or a raw HF id to (model_id, config).

    If ``model_id_or_hf`` is a known short id, return its config directly.
    If it matches a registered ``hf_id``, return the corresponding short id
    and config. Otherwise raise KeyError listing the valid ids.
    """
    if model_id_or_hf in MODEL_REGISTRY:
        return model_id_or_hf, MODEL_REGISTRY[model_id_or_hf]
    for mid, cfg in MODEL_REGISTRY.items():
        if cfg["hf_id"] == model_id_or_hf:
            return mid, cfg
    raise KeyError(
        f"Unknown model: {model_id_or_hf!r}. "
        f"Known short ids: {list(MODEL_REGISTRY)}"
    )
