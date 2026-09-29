#!/usr/bin/env bash
# Six-base-model RehabFact selection experiment and result exporter.
# New runs are written under runs/ and never overwrite packaged frozen results.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

ACTION="${1:-run}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CORPUS="${CORPUS:-data/patient_rehab_29subjects.json}"
SPLIT="${SPLIT:-splits/llm_real_text13_patient_3fold.json}"
RUN_ROOT="${RUN_ROOT:-runs/rehabfact_six_model_comparison}"
SOURCE_RESULTS_ROOT="${SOURCE_RESULTS_ROOT:-results/llm}"
EXPORT_ROOT="${EXPORT_ROOT:-exports}"
EPOCHS="${EPOCHS:-3}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
SEED="${SEED:-2024}"
INCLUDE_AUX_GENERATED="${INCLUDE_AUX_GENERATED:-1}"
INCLUDE_PREDICTIONS="${INCLUDE_PREDICTIONS:-0}"
DRY_RUN="${DRY_RUN:-0}"

MODELS=(
  baichuan2_7b
  qwen3_8b
  gemma3_4b
  glm4_9b_chat
  llama31_8b
  deepseek_r1_llama8b
)
FOLDS=(1 2 3)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_rehabfact_six_llm_experiment.sh run
  bash scripts/run_rehabfact_six_llm_experiment.sh aggregate
  bash scripts/run_rehabfact_six_llm_experiment.sh export-existing

Actions:
  run              Build the fixed three-fold data, train all six LoRA models,
                   generate held-out reports, evaluate, and aggregate.
  aggregate        Rebuild summary.csv, mean.csv, and best_model.json from an
                   existing RUN_ROOT/evaluation tree.
  export-existing  Copy the historical SOURCE_RESULTS_ROOT evaluation evidence
                   into a timestamped archive without modifying the source.

Important environment variables:
  CORPUS, SPLIT, RUN_ROOT, SOURCE_RESULTS_ROOT, EXPORT_ROOT, PYTHON_BIN
  EPOCHS, LORA_RANK, LORA_ALPHA, LORA_DROPOUT, SEED
  INCLUDE_AUX_GENERATED=0|1, INCLUDE_PREDICTIONS=0|1, DRY_RUN=0|1
EOF
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: required file not found: $1" >&2
    exit 1
  fi
}

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

verify_registry() {
  "${PYTHON_BIN}" - "${MODELS[@]}" <<'PY'
import sys
from src.llm.model_registry import MODEL_REGISTRY

missing = [model_id for model_id in sys.argv[1:] if model_id not in MODEL_REGISTRY]
if missing:
    raise SystemExit(f"Missing MODEL_REGISTRY entries: {missing}")
print("[six-llm] registry OK:", ", ".join(sys.argv[1:]))
PY
}

verify_reports() {
  local evaluation_root="$1"
  local missing=0
  for model in "${MODELS[@]}"; do
    for fold in "${FOLDS[@]}"; do
      if [[ ! -f "${evaluation_root}/${model}/fold${fold}.json" ]]; then
        echo "ERROR: missing evaluation report: ${evaluation_root}/${model}/fold${fold}.json" >&2
        missing=1
      fi
    done
  done
  if [[ "${missing}" != "0" ]]; then
    exit 1
  fi
}

aggregate_results() {
  local evaluation_root="$1"
  verify_reports "${evaluation_root}"
  run_cmd "${PYTHON_BIN}" -m src.llm.select_best_model \
    --reports-glob "${evaluation_root}/*/fold*.json" \
    --metric all.char_bleu4 \
    --extra all.sacrebleu_zh all.word_bleu4 all.rouge1_f all.rouge2_f all.rougeL_f \
    --out "${evaluation_root}/best_model.json"
}

run_experiment() {
  require_file "${CORPUS}"
  require_file "${SPLIT}"
  verify_registry

  for fold in "${FOLDS[@]}"; do
    data_dir="${RUN_ROOT}/data/fold${fold}"
    builder_args=(
      "${PYTHON_BIN}" -m src.llm.data_builder
      --suggestions "${CORPUS}"
      --split "${SPLIT}"
      --fold "${fold}"
      --out "${data_dir}"
    )
    if [[ "${INCLUDE_AUX_GENERATED}" != "1" ]]; then
      builder_args+=(--no-aux-generated)
    fi
    run_cmd "${builder_args[@]}"
  done

  for model in "${MODELS[@]}"; do
    for fold in "${FOLDS[@]}"; do
      data_dir="${RUN_ROOT}/data/fold${fold}"
      checkpoint="${RUN_ROOT}/checkpoints/${model}/fold${fold}"
      prediction="${RUN_ROOT}/predictions/${model}/fold${fold}_test.json"
      report="${RUN_ROOT}/evaluation/${model}/fold${fold}.json"

      run_cmd "${PYTHON_BIN}" -m src.llm.train_lora \
        --model-id "${model}" \
        --train "${data_dir}/train.jsonl" \
        --val "${data_dir}/val.jsonl" \
        --out "${checkpoint}" \
        --epochs "${EPOCHS}" \
        --rank "${LORA_RANK}" \
        --alpha "${LORA_ALPHA}" \
        --dropout "${LORA_DROPOUT}" \
        --seed "${SEED}"

      run_cmd "${PYTHON_BIN}" -m src.llm.generate \
        --model-id "${model}" \
        --adapter "${checkpoint}" \
        --suggestions "${CORPUS}" \
        --split "${SPLIT}" \
        --fold "${fold}" \
        --partition test \
        --out "${prediction}"

      run_cmd "${PYTHON_BIN}" -m src.llm.evaluate \
        --pred "${prediction}" \
        --out "${report}"
    done
  done

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[six-llm] dry run complete; aggregation skipped because no reports were written."
  else
    aggregate_results "${RUN_ROOT}/evaluation"
  fi
}

export_existing() {
  local source_evaluation="${SOURCE_RESULTS_ROOT}/evaluation"
  verify_registry
  verify_reports "${source_evaluation}"

  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local package_name="rehabfact_six_model_results_${stamp}"
  local stage="${EXPORT_ROOT}/${package_name}"
  local archive="${EXPORT_ROOT}/${package_name}.tar.gz"

  mkdir -p "${stage}/results/llm" "${stage}/src/llm" "${stage}/splits"
  cp -a "${source_evaluation}" "${stage}/results/llm/evaluation"
  if [[ "${INCLUDE_PREDICTIONS}" == "1" ]]; then
    if [[ ! -d "${SOURCE_RESULTS_ROOT}/predictions" ]]; then
      echo "ERROR: INCLUDE_PREDICTIONS=1 but predictions are missing." >&2
      exit 1
    fi
    cp -a "${SOURCE_RESULTS_ROOT}/predictions" "${stage}/results/llm/predictions"
  fi
  cp "src/llm/model_registry.py" "src/llm/select_best_model.py" \
    "${stage}/src/llm/"
  cp "${SPLIT}" "${stage}/splits/"

  "${PYTHON_BIN}" -m src.llm.select_best_model \
    --reports-glob "${stage}/results/llm/evaluation/*/fold*.json" \
    --metric all.char_bleu4 \
    --extra all.sacrebleu_zh all.word_bleu4 all.rouge1_f all.rouge2_f all.rougeL_f \
    --out "${stage}/results/llm/evaluation/best_model.json"

  tar -C "${EXPORT_ROOT}" -czf "${archive}" "${package_name}"
  sha256sum "${archive}" > "${archive}.sha256"

  echo "[six-llm] exported: ${archive}"
  echo "[six-llm] checksum: ${archive}.sha256"
  if [[ "${INCLUDE_PREDICTIONS}" != "1" ]]; then
    echo "[six-llm] predictions excluded by default because they contain clinician-authored text."
  fi
}

case "${ACTION}" in
  run)
    run_experiment
    ;;
  aggregate)
    verify_registry
    aggregate_results "${RUN_ROOT}/evaluation"
    ;;
  export-existing)
    require_file "${SPLIT}"
    export_existing
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "ERROR: unknown action: ${ACTION}" >&2
    usage >&2
    exit 2
    ;;
esac
