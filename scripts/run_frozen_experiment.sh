#!/usr/bin/env bash
set -euo pipefail

# Frozen CMK-AGN mainline: FMA and BI, 3 seeds x 3 patient-level folds.
# Raw study data are intentionally external. Example:
#   MANIFEST=/secure/path/samples_manifest_real_archive.csv bash scripts/run_frozen_experiment.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${MANIFEST:-${ROOT}/data/samples_manifest_real_archive.csv}"
SPLIT_JSON="${SPLIT_JSON:-${ROOT}/splits/split_patient_3fold.json}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/frozen_experiment}"
CACHE_DIR="${CACHE_DIR:-${RUN_ROOT}/alignment_cache}"
SEEDS_TEXT="${SEEDS:-2024 42 1337}"
TASKS_TEXT="${TASKS:-FMA_UE BI}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-120}"
PATIENCE="${PATIENCE:-25}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"

[[ -f "${MANIFEST}" ]] || { echo "ERROR: authorized manifest not found: ${MANIFEST}" >&2; exit 2; }
[[ -f "${SPLIT_JSON}" ]] || { echo "ERROR: split file not found: ${SPLIT_JSON}" >&2; exit 2; }
[[ -f "${ROOT}/src/train.py" ]] || { echo "ERROR: src/train.py not found below ${ROOT}" >&2; exit 2; }

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

for seed in "${SEEDS_ARRAY[@]}"; do
  for task in "${TASKS_ARRAY[@]}"; do
    case "${task}" in
      FMA_UE) public_task="FMA" ;;
      BI) public_task="BI" ;;
      *) echo "ERROR: frozen mainline task must be FMA_UE or BI, got ${task}" >&2; exit 2 ;;
    esac

    out_dir="${RUN_ROOT}/seed${seed}/${public_task}"
    checkpoint="${RUN_ROOT}/checkpoints/${public_task}/seed${seed}/fold{fold}_best.pt"
    summary="${out_dir}/${task}_3fold_summary.csv"

    if [[ -f "${summary}" ]]; then
      echo "SKIP: completed run ${summary}"
      continue
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
      --alignment-mode adk \
      --modalities eeg+emg+imu \
      --checkpoint-metric mae \
      --out-dir "${out_dir}" \
      --checkpoint "${checkpoint}"
  done
done

echo "Frozen mainline complete: ${RUN_ROOT}"
