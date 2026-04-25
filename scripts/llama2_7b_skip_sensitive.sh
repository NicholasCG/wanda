#!/bin/bash

# Runs Wanda pruning on LLaMA-7B while preserving the top-25% most sensitive
# layers (by per-layer delta-PPL from layer_sensitivity_results) for each
# sparsity type. All other settings match llama_7b.sh.
#
# Sensitive layers (top 8 of 32 by delta_ppl, hardcoded from
# layer_sensitivity_results/wanda_20260405_130855):
#   unstructured: 2 9 13 16 17 29 30 31
#   2:4:          2 4  9 16 25 29 30 31
#   4:8:          2 4  9 16 25 29 30 31

model="meta-llama/Llama-2-7b-hf"
sparsity_ratio=0.67
cuda_device=0
seed=0

export CUDA_VISIBLE_DEVICES=$cuda_device

# echo "Running with no pruning (dense baseline)"
# python main.py \
#     --model $model \
#     --prune_method "wanda" \
#     --sparsity_ratio 0 \
#     --sparsity_type "unstructured" \
#     --save "out/llama2_7b_skip_25_percent/seed_$seed/dense/" \
#     --seed $seed
# echo "Finished dense baseline"

echo "Running wanda unstructured (skip sensitive layers: 2 9 13 16 17 29 30 31)"
python main.py \
    --model $model \
    --prune_method wanda \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type "unstructured" \
    --save "out/llama2_7b_skip_25_percent_67_sparsity/seed_$seed/unstructured/wanda/" \
    --seed $seed \
    --skip_layers 2 9 13 16 17 29 30 31 

echo "Running wanda 2:4 (skip sensitive layers: 2 4 9 16 25 29 30 31)"
python main.py \
    --model $model \
    --prune_method wanda \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type "2:4" \
    --save "out/llama2_7b_skip_25_percent_67_sparsity/seed_$seed/2-4/wanda/" \
    --seed $seed \
    --skip_layers 2 4 9 16 25 29 30 31 

echo "Running wanda 4:8 (skip sensitive layers: 2 4 9 16 25 29 30 31)"
python main.py \
    --model $model \
    --prune_method wanda \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type "4:8" \
    --save "out/llama2_7b_skip_25_percent_67_sparsity/seed_$seed/4-8/wanda/" \
    --seed $seed \
    --skip_layers 2 4 9 16 25 29 30 31 

echo "Finished wanda pruning"
