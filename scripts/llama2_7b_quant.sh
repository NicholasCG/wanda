#!/bin/bash
# Runs the quantization experiment: compares Wanda pruning on FP16 vs INT8
# (bitsandbytes LLM.int8()) models for LLaMA-2-7B.
#
# Evaluates and reports:
#   - FP16 dense perplexity
#   - FP16 sparse perplexity
#   - INT8 dense perplexity
#   - INT8 sparse perplexity
#   - Mask difference (%) between FP16 sparse and INT8 sparse models

# Set common variables
model="meta-llama/Llama-2-7b-hf"
sparsity_ratio=0.5
cuda_device=0
seed=0

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define a helper that runs one sparsity-type experiment
run_quant_experiment () {
    local sparsity_type=$1
    local save_dir=$2

    echo ""
    echo "============================================================"
    echo "Running quantization experiment"
    echo "  model         : $model"
    echo "  sparsity_type : $sparsity_type"
    echo "  sparsity_ratio: $sparsity_ratio"
    echo "  save_dir      : $save_dir"
    echo "============================================================"

    python main_quant.py \
        --model "$model" \
        --sparsity_ratio "$sparsity_ratio" \
        --sparsity_type "$sparsity_type" \
        --save "$save_dir" \
        --seed "$seed"
}

# Unstructured 50% sparsity
run_quant_experiment "unstructured" "out/llama2_7b_quant/seed_${seed}/unstructured/"

# Structured 2:4 sparsity
run_quant_experiment "2:4" "out/llama2_7b_quant/seed_${seed}/2-4/"

# Structured 4:8 sparsity
run_quant_experiment "4:8" "out/llama2_7b_quant/seed_${seed}/4-8/"

echo ""
echo "All quantization experiments complete."
echo "Results saved under out/llama2_7b_quant/seed_${seed}/"
