#!/bin/bash

# Sweeps calibration sample counts from 1 to 512 (powers of two) for Wanda
# and SparseGPT on LLaMA-2-7B at 50% unstructured sparsity.
# Save paths encode the method and sample count, e.g.:
#   out/llama2_7b/nsamples_sweep/wanda/nsamples128/

model="meta-llama/Llama-2-7b-hf"
cuda_device=0
seed=0
sparsity_ratio=0.5
sparsity_type="unstructured"

export CUDA_VISIBLE_DEVICES=$cuda_device

methods=("wanda" "sparsegpt")

for method in "${methods[@]}"; do
    echo "===== Method: $method ====="
    nsamples=1
    while [ $nsamples -le 512 ]; do
        save_dir="out/llama2_7b/nsamples_sweep/${method}/nsamples${nsamples}/"
        echo "  nsamples=${nsamples}  ->  $save_dir"
        python main.py \
            --model $model \
            --prune_method $method \
            --sparsity_ratio $sparsity_ratio \
            --sparsity_type $sparsity_type \
            --nsamples $nsamples \
            --save "$save_dir" \
            --seed $seed
        nsamples=$(( nsamples * 2 ))
    done
    echo "===== Finished $method ====="
done
