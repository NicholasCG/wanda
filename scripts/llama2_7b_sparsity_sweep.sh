#!/bin/bash

# Sweeps unstructured sparsity from 10% to 90% (in 10% steps) for Wanda,
# SparseGPT, and Magnitude pruning on LLaMA-7B.
# Save paths encode the method and sparsity level, e.g.:
#   out/llama2_7b/sparsity_sweep/wanda/sp50pct/

model="meta-llama/Llama-2-7b-hf"
cuda_device=0
seed=0
sparsity_type="unstructured"

export CUDA_VISIBLE_DEVICES=$cuda_device

methods=("wanda" "sparsegpt" "magnitude")

for method in "${methods[@]}"; do
    echo "===== Method: $method ====="
    for pct in 10 20 30 40 50 60 70 80 90; do
        sparsity_ratio=$(printf "0.%02d" $pct)
        save_dir="out/llama2_7b/sparsity_sweep/${method}/sp${pct}pct/"
        echo "  Sparsity ${pct}%  ->  $save_dir"
        python main.py \
            --model $model \
            --prune_method $method \
            --sparsity_ratio $sparsity_ratio \
            --sparsity_type $sparsity_type \
            --save "$save_dir" \
            --seed $seed
    done
    echo "===== Finished $method ====="
done
