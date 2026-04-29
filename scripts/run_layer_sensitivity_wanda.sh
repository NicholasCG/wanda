#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-meta-llama/Llama-2-7b-hf}"
CACHE_DIR="${CACHE_DIR:-llm_weights}"
SAVE_ROOT="${SAVE_ROOT:-layer_sensitivity_results}"
NSAMPLES="${NSAMPLES:-128}"
SEED="${SEED:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYER_SCRIPT="${SCRIPT_DIR}/layer_sensitivity.py"

if [[ ! -f "${LAYER_SCRIPT}" ]]; then
  echo "Error: ${LAYER_SCRIPT} not found. Put this script next to layer_sensitivity.py in the wanda repo root." >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${SAVE_ROOT}/wanda_${STAMP}"
mkdir -p "${RUN_ROOT}"

run_case() {
  local sparsity_type="$1"
  local out_dir="${RUN_ROOT}/${sparsity_type//:/-}"
  mkdir -p "${out_dir}"

  echo "============================================================"
  echo "Running Wanda layer sensitivity: ${sparsity_type}"
  echo "Saving to: ${out_dir}"

  ${PYTHON_BIN} "${LAYER_SCRIPT}" \
    --model "${MODEL}" \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type "${sparsity_type}" \
    --nsamples "${NSAMPLES}" \
    --seed "${SEED}" \
    --cache_dir "${CACHE_DIR}" \
    --save "${out_dir}" \
    ${EXTRA_ARGS}
}

run_case unstructured
run_case 2:4
run_case 4:8

echo "============================================================"
echo "All Wanda layer sensitivity runs finished."
echo "Results root: ${RUN_ROOT}"
echo "Expected CSV files:"
echo "  ${RUN_ROOT}/unstructured/layer_sensitivity_wanda_unstructured.csv"
echo "  ${RUN_ROOT}/2-4/layer_sensitivity_wanda_2-4.csv"
echo "  ${RUN_ROOT}/4-8/layer_sensitivity_wanda_4-8.csv"
