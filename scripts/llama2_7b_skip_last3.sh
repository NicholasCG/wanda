#!/bin/bash

# Runs Wanda pruning on LLaMA-7B while preserving the final 3 transformer
# layers (leaving them fully dense). All other settings match llama_7b.sh.

model="meta-llama/Llama-2-7b-hf"
sparsity_ratio=0.5
cuda_device=0
seed=0

export CUDA_VISIBLE_DEVICES=$cuda_device

run_wanda_skip_last3 () {
    python main.py \
    --model $model \
    --prune_method wanda \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $1 \
    --save $2 \
    --seed $seed \
    --skip_last_n_layers 3 --eval_zero_shot
}

echo "Running with no pruning (dense baseline)"
python main.py \
    --model $model \
    --prune_method "wanda" \
    --sparsity_ratio 0 \
    --sparsity_type "unstructured" \
    --save "out/llama_7b_skip_last3/seed_$seed/dense/" \
    --seed $seed 
echo "Finished dense baseline"

echo "Running wanda with last-3-layers preserved"
run_wanda_skip_last3 "unstructured" "out/llama_7b_skip_last3/seed_$seed/unstructured/wanda/"
run_wanda_skip_last3 "2:4"          "out/llama_7b_skip_last3/seed_$seed/2-4/wanda/"
run_wanda_skip_last3 "4:8"          "out/llama_7b_skip_last3/seed_$seed/4-8/wanda/"
echo "Finished wanda pruning"
