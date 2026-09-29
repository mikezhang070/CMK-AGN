#!/usr/bin/env bash
set -euo pipefail

# Full CMK-AGN ablation protocol:
#   1) full model: 4 tasks x 3 seeds x 3 patient-level folds
#   2) modality ablation: 4 tasks x 6 variants x 3 folds
#   3) module ablation: 4 tasks x 3 variants x 3 folds
#   4) aggregate tables and rebuild manuscript Figures 10 and 11

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${MANIFEST:-${ROOT}/data/samples_manifest_real_archive.csv}"
SPLIT_JSON="${SPLIT_JSON:-${ROOT}/splits/split_patient_3fold.json}"
ABLATION_ROOT="${ABLATION_ROOT:-${ROOT}/runs/ablation_multiseed}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${ABLATION_ROOT}/_summary}"
FIGURE_SOURCE_DIR="${FIGURE_SOURCE_DIR:-${ABLATION_ROOT}/figure_source_data}"
FIGURE_OUTPUT_DIR="${FIGURE_OUTPUT_DIR:-${ABLATION_ROOT}/figures}"
CACHE_DIR="${CACHE_DIR:-${ABLATION_ROOT}/alignment_cache}"
SEEDS_TEXT="${SEEDS:-2024 42 1337}"
ABLATION_SEED="${ABLATION_SEED:-2024}"
TASKS_TEXT="${TASKS:-FMA_UE BI hand_tone hand_function}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-120}"
PATIENCE="${PATIENCE:-25}"
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"

[[ -f "${MANIFEST}" ]] || { echo "ERROR: authorized manifest not found: ${MANIFEST}" >&2; exit 2; }
[[ -f "${SPLIT_JSON}" ]] || { echo "ERROR: split file not found: ${SPLIT_JSON}" >&2; exit 2; }
for required in src/train.py src/aggregate_ablation.py visualization/fig10_regression_task_ablation.py visualization/fig11_ordinal_task_ablation.py; do
  [[ -f "${ROOT}/${required}" ]] || { echo "ERROR: missing ${ROOT}/${required}" >&2; exit 2; }
done
case "${STAGE}" in train|aggregate|figures|all) ;; *) echo "ERROR: STAGE must be train, aggregate, figures, or all" >&2; exit 2 ;; esac

read -r -a SEEDS_ARRAY <<< "${SEEDS_TEXT}"
read -r -a TASKS_ARRAY <<< "${TASKS_TEXT}"

run_cmd() {
  printf 'RUN:'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

checkpoint_metric() {
  case "$1" in
    FMA_UE|BI) printf '%s' mae ;;
    hand_tone|hand_function) printf '%s' weighted_kappa ;;
    *) echo "ERROR: unknown task $1" >&2; exit 2 ;;
  esac
}

train_variant() {
  local kind="$1" tag="$2" task="$3" seed="$4"
  shift 4
  local out_dir="${ABLATION_ROOT}/${kind}/${tag}/${task}"
  local summary="${out_dir}/${task}_3fold_summary.csv"
  local checkpoint="${out_dir}/checkpoints/fold{fold}_best.pt"

  if [[ -f "${summary}" ]]; then
    echo "SKIP: completed run ${summary}"
    return
  fi
  if [[ -d "${out_dir}" && "${ALLOW_EXISTING}" != "1" ]]; then
    echo "ERROR: partial output exists: ${out_dir}" >&2
    echo "Move it aside, or set ALLOW_EXISTING=1 to continue into that directory." >&2
    exit 3
  fi

  run_cmd "${PYTHON_BIN}" "${ROOT}/src/train.py" \
    --root "${ROOT}" \
    --task "${task}" \
    --manifest "${MANIFEST}" \
    --split-json "${SPLIT_JSON}" \
    --fold 0 \
    --seed "${seed}" \
    --device "${DEVICE}" \
    --num-workers "${NUM_WORKERS}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --cache-dir "${CACHE_DIR}" \
    --checkpoint-metric "$(checkpoint_metric "${task}")" \
    --out-dir "${out_dir}" \
    --checkpoint "${checkpoint}" \
    "$@"
}

if [[ "${STAGE}" == "train" || "${STAGE}" == "all" ]]; then
  for seed in "${SEEDS_ARRAY[@]}"; do
    tag="tri_seed${seed}"
    for task in "${TASKS_ARRAY[@]}"; do
      train_variant module "${tag}" "${task}" "${seed}" \
        --alignment-mode adk --modalities eeg+emg+imu
    done
  done

  modality_specs=(
    "eeg_only:eeg"
    "emg_only:emg"
    "imu_only:imu"
    "eeg_emg:eeg+emg"
    "eeg_imu:eeg+imu"
    "emg_imu:emg+imu"
  )
  for spec in "${modality_specs[@]}"; do
    tag="${spec%%:*}"
    modalities="${spec#*:}"
    for task in "${TASKS_ARRAY[@]}"; do
      train_variant modality "${tag}" "${task}" "${ABLATION_SEED}" \
        --alignment-mode adk --modalities "${modalities}"
    done
  done

  for task in "${TASKS_ARRAY[@]}"; do
    train_variant module wo_mdfan "${task}" "${ABLATION_SEED}" \
      --alignment-mode adk --modalities eeg+emg+imu --no-mdfan
    train_variant module wo_wbydtw "${task}" "${ABLATION_SEED}" \
      --alignment-mode adk_no_dtw --modalities eeg+emg+imu
    train_variant module wo_trialign "${task}" "${ABLATION_SEED}" \
      --alignment-mode resample --modalities eeg+emg+imu
  done
fi

if [[ "${STAGE}" == "aggregate" || "${STAGE}" == "all" ]]; then
  run_cmd "${PYTHON_BIN}" "${ROOT}/src/aggregate_ablation.py" \
    --root "${ABLATION_ROOT}" \
    --out "${SUMMARY_ROOT}" \
    --figure-source-dir "${FIGURE_SOURCE_DIR}"
fi

if [[ "${STAGE}" == "figures" || "${STAGE}" == "all" ]]; then
  run_cmd "${PYTHON_BIN}" "${ROOT}/visualization/fig10_regression_task_ablation.py" \
    --root "${ROOT}" --source-data-dir "${FIGURE_SOURCE_DIR}" --output "${FIGURE_OUTPUT_DIR}"
  run_cmd "${PYTHON_BIN}" "${ROOT}/visualization/fig11_ordinal_task_ablation.py" \
    --root "${ROOT}" --source-data-dir "${FIGURE_SOURCE_DIR}" --output "${FIGURE_OUTPUT_DIR}"
fi

echo "Ablation workflow complete: ${ABLATION_ROOT}"
