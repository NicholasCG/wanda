#!/usr/bin/env bash
# run_speedup.sh – Measure 2:4 sparsity GEMM speedup for LLaMA-7b and LLaMA-2-7b
# using Wanda pruning.  Results are written to out/speedup/ as JSON.
#
# Usage:
#   bash scripts/run_speedup.sh              # CUTLASS sparse, cuBLAS dense
#   SEQ_LEN=2048 bash scripts/run_speedup.sh
#   USE_CUSPARSELT=1 bash scripts/run_speedup.sh
#   SPARSE_ONLY=1 bash scripts/run_speedup.sh  # skip dense benchmarks (lower memory)

set -euo pipefail

CACHE_DIR="${CACHE_DIR:-llm_weights}"
SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NSAMPLES="${NSAMPLES:-128}"
SAVE_DIR="${SAVE_DIR:-out/speedup}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

EXTRA_FLAGS=""
if [[ "${USE_CUSPARSELT:-0}" == "1" ]]; then
    EXTRA_FLAGS="--use_cusparselt"
fi
if [[ "${SPARSE_ONLY:-0}" == "1" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --sparse_only"
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

echo "========================================================"
echo "  WANDA 2:4 Sparsity Speedup Benchmark"
echo "  seq_len=${SEQ_LEN}  batch_size=${BATCH_SIZE}  gpu=${CUDA_DEVICE}"
echo "========================================================"

# ── LLaMA-7b (huggyllama) ────────────────────────────────────────────────────
echo ""
echo "[1/2] huggyllama/llama-7b"
python measure_speedup.py \
    --model       huggyllama/llama-7b \
    --cache_dir   "$CACHE_DIR" \
    --nsamples    "$NSAMPLES" \
    --batch_size  "$BATCH_SIZE" \
    --save        "$SAVE_DIR" \
    $EXTRA_FLAGS

# ── LLaMA-2-7b (meta-llama) ──────────────────────────────────────────────────
echo ""
echo "[2/2] meta-llama/Llama-2-7b-hf"
python measure_speedup.py \
    --model       meta-llama/Llama-2-7b-hf \
    --cache_dir   "$CACHE_DIR" \
    --nsamples    "$NSAMPLES" \
    --batch_size  "$BATCH_SIZE" \
    --save        "$SAVE_DIR" \
    $EXTRA_FLAGS

echo ""
echo "Done. JSON results written to: $SAVE_DIR"
