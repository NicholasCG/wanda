#!/bin/bash

# Set common variables
model="meta-llama/Llama-2-7b-hf"
sparsity_ratio=0.5
cuda_device=0

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
    --compare_original_modified
}

# llama-7b with wanda pruning method
echo "Running with wanda pruning method"
run_python_command "wanda" "unstructured" "out/llama2_7b/compare/unstructured/"
run_python_command "wanda" "2:4" "out/llama2_7b/compare/2-4/"
run_python_command "wanda" "4:8" "out/llama2_7b/compare/4-8/"
echo "Finished wanda pruning method"

# llama-7b with sparsegpt pruning method
echo "Running with sparsegpt pruning method"
run_python_command "sparsegpt" "unstructured" "out/llama2_7b/compare/unstructured/"
run_python_command "sparsegpt" "2:4" "out/llama2_7b/compare/2-4/"
run_python_command "sparsegpt" "4:8" "out/llama2_7b/compare/4-8/"
echo "Finished sparsegpt pruning method"

# llama-7b with magnitude pruning method
echo "Running with magnitude pruning method"
run_python_command "magnitude" "unstructured" "out/llama2_7b/compare/unstructured/"
run_python_command "magnitude" "2:4" "out/llama2_7b/compare/2-4/"
run_python_command "magnitude" "4:8" "out/llama2_7b/compare/4-8/"
echo "Finished magnitude pruning method"