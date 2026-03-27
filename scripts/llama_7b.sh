#!/bin/bash

# Set common variables
model="meta-llama/Llama-2-7b-hf"
sparsity_ratio=0.5
cuda_device=0
seed=0

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command () {
    python main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $2 \
    --save $3 \
    --seed $seed
}

# echo "Running with no pruning"
# python main.py \
#     --model $model \
#     --prune_method "wanda" \
#     --sparsity_ratio 0 \
#     --sparsity_type "unstructured" \
#     --save "out/llama2_7b/seed_$seed/dense/" \
#     --seed $seed
# echo "Finished no pruning"

# llama-7b with wanda pruning method
echo "Running with wanda pruning method"
# run_python_command "wanda" "unstructured" "out/llama2_7b/seed_$seed/unstructured/wanda/"
run_python_command "wanda" "2:4" "out/llama2_7b/seed_$seed/2-4/wanda/"
run_python_command "wanda" "4:8" "out/llama2_7b/seed_$seed/4-8/wanda/"
# echo "Finished wanda pruning method"

# # llama-7b with sparsegpt pruning method
# echo "Running with sparsegpt pruning method"
# run_python_command "sparsegpt" "unstructured" "out/llama2_7b/seed_$seed/unstructured/sparsegpt/"
# run_python_command "sparsegpt" "2:4" "out/llama2_7b/seed_$seed/2-4/sparsegpt/"
# run_python_command "sparsegpt" "4:8" "out/llama2_7b/seed_$seed/4-8/sparsegpt/"
# echo "Finished sparsegpt pruning method"

# # llama-7b with magnitude pruning method
# echo "Running with magnitude pruning method"
# run_python_command "magnitude" "unstructured" "out/llama2_7b/seed_$seed/unstructured/magnitude/"
# run_python_command "magnitude" "2:4" "out/llama2_7b/seed_$seed/2-4/magnitude/"
# run_python_command "magnitude" "4:8" "out/llama2_7b/seed_$seed/4-8/magnitude/"
# echo "Finished magnitude pruning method"