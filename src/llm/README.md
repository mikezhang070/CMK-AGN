# RehabFact six-model experiment

This module implements the language-model comparison and the constrained
RehabFact reporting workflow. It is downstream from CMK-AGN: the language
model renders structured rehabilitation facts and does not replace the
EEG/sEMG/IMU assessment network.

## Six-model selection experiment

The completed selection experiment used three patient-disjoint folds and six
server-local instruction models:

| Model ID | Base model |
|---|---|
| `baichuan2_7b` | Baichuan2-7B-Chat |
| `qwen3_8b` | Qwen3-8B |
| `gemma3_4b` | Gemma-3-4B-IT |
| `glm4_9b_chat` | GLM-4-9B-Chat |
| `llama31_8b` | Meta-Llama-3.1-8B-Instruct |
| `deepseek_r1_llama8b` | DeepSeek-R1-Distill-Llama-8B |

The selected model was `baichuan2_7b`. Later RehabFact experiments therefore
continued only with Baichuan2-7B-Chat, including the human-only versus
human-plus-auxiliary ablation packaged under `results/RehabFact/`.

Run the fixed comparison with:

```bash
CORPUS=/secure/authorized/patient_rehab_29subjects.json \
  bash scripts/run_rehabfact_six_llm_experiment.sh run
```

Outputs are written to `runs/rehabfact_six_model_comparison/`; packaged frozen
results are never overwritten. The default selection-stage LoRA settings are
rank 16, alpha 32, dropout 0.05, three epochs, and seed 2024. These are
experiment parameters, not a substitute for reading `adapter_config.json`
when auditing a saved adapter.

To aggregate an existing run:

```bash
RUN_ROOT=runs/rehabfact_six_model_comparison \
  bash scripts/run_rehabfact_six_llm_experiment.sh aggregate
```

To export the historical server-side evaluation reports without retraining:

```bash
SOURCE_RESULTS_ROOT=results/llm \
  bash scripts/run_rehabfact_six_llm_experiment.sh export-existing
```

The export contains evaluation JSON/CSV files and a regenerated leaderboard.
Predictions are excluded by default because they contain clinician-authored
text. Set `INCLUDE_PREDICTIONS=1` only for an authorized internal archive.

## Data and leakage controls

- The fixed split is `splits/llm_real_text13_patient_3fold.json`.
- Human-reference patients are separated at patient level across folds.
- Rule-generated auxiliary cases may enter training only.
- Validation and test partitions contain human-authored references only.
- The protected corpus and clinician-authored reports are not distributed.

## Result provenance

The complete six-row metric table was restored from the original server-side
`results/llm/evaluation/` tree through the documented exporter and is packaged
under `results/RehabFact/six_model_comparison/evaluation/` (summary.csv,
mean.csv, best_model.json, and per-model fold reports). The selected model is
`baichuan2_7b` (char-BLEU4 47.29 ± 5.33) with runner-up `qwen3_8b`
(42.49 ± 5.61). Per-case predictions and clinician-authored text were excluded
by the exporter and are not distributed. Missing values must never be
reconstructed from memory or manuscript prose.
