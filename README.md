# Pruning LLMs by Weights and Activations

PyTorch implementation of **Wanda** (Pruning by **W**eights **and a**ctivations), based on the paper:

**A Simple and Effective Pruning Approach for Large Language Models** </br>
*Mingjie Sun\*, Zhuang Liu\*, Anna Bair, J. Zico Kolter* (* indicates equal contribution) <br>
Carnegie Mellon University, Meta AI Research and Bosch Center for AI  <br>
[Paper](https://arxiv.org/abs/2306.11695) - [Project page](https://eric-mingjie.github.io/wanda/home.html)

```bibtex
@article{sun2023wanda,
  title={A Simple and Effective Pruning Approach for Large Language Models}, 
  author={Sun, Mingjie and Liu, Zhuang and Bair, Anna and Kolter, J. Zico},
  year={2023},
  journal={arXiv preprint arXiv:2306.11695}
}
```

Compared to magnitude pruning which removes weights solely based on their magnitudes, **Wanda** removes weights on a *per-output* basis, by the product of weight magnitudes and input activation norms.

---

## Table of Contents

- [Pruning LLMs by Weights and Activations](#pruning-llms-by-weights-and-activations)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
    - [Recommended: Docker](#recommended-docker)
    - [Backup: requirements.txt](#backup-requirementstxt)
    - [Secondary Backup: Manual Installation](#secondary-backup-manual-installation)
  - [Model Weights](#model-weights)
  - [Running the Code](#running-the-code)
    - [Command-Line Arguments](#command-line-arguments)
    - [Pruning LLaMA-7B](#pruning-llama-7b)
    - [Pruning LLaMA-2-7B](#pruning-llama-2-7b)
  - [Reproducing Experiments](#reproducing-experiments)
    - [1. Validating Correctness Against Original Codebase](#1-validating-correctness-against-original-codebase)
    - [2. Perplexity and Zero-Shot Accuracy](#2-perplexity-and-zero-shot-accuracy)
    - [3. Sparsity Range Experiment](#3-sparsity-range-experiment)
    - [4. Number of Calibration Samples Experiment](#4-number-of-calibration-samples-experiment)
    - [5. Inference Speed Comparison](#5-inference-speed-comparison)
    - [6. Pruning Speed Comparison](#6-pruning-speed-comparison)
    - [7. Layer Sensitivity Search](#7-layer-sensitivity-search)
    - [8. Skip Last 3 Layers](#8-skip-last-3-layers)
    - [9. Skip Top 25% Most Sensitive Layers](#9-skip-top-25-most-sensitive-layers)
    - [10. Quantization and Pruning Test](#10-quantization-and-pruning-test)
  - [Acknowledgement](#acknowledgement)
  - [License](#license)

---

## Setup

### Recommended: Docker

The easiest way to reproduce the environment is via Docker. The provided `Dockerfile` installs all dependencies (Miniconda, PyTorch 1.10.1 with CUDA 11.3, Hugging Face Transformers, `lm_eval`, and `bitsandbytes`) into a conda environment named `prune_llm`.

**Build the image:**
```sh
docker build -t wanda .
```

**Run an interactive container** (mount model weights so they are not re-downloaded on each run):
```sh
docker run --gpus all -it --rm \
    -v /path/to/llm_weights:/workspace/llm_weights \
    wanda bash
```

Inside the container all commands below can be run directly from `/workspace`.

---

### Backup: requirements.txt

If Docker is not available, you can reproduce the environment using the provided `requirements.txt`. You will still need conda for the PyTorch + CUDA step.

**1. Create and activate a conda environment:**
```sh
conda create -n prune_llm python=3.9 -y
conda activate prune_llm
```

**2. Install PyTorch with CUDA 11.3 via conda:**
```sh
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 \
    -c pytorch -c conda-forge -y
```

**3. Install remaining dependencies from `requirements.txt`:**
```sh
pip install -r requirements.txt
```

**4. Set the project root on `PYTHONPATH`:**
```sh
export PYTHONPATH=/path/to/wanda
```

---

### Secondary Backup: Manual Installation

If you prefer to install without Docker, the following steps replicate the environment on a machine with CUDA 11.3:

**1. Create and activate a conda environment:**
```sh
conda create -n prune_llm python=3.9 -y
conda activate prune_llm
```

**2. Install PyTorch with CUDA 11.3:**
```sh
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 \
    -c pytorch -c conda-forge -y
```

**3. Install Python dependencies:**
```sh
pip install "numpy<2" "pyarrow<13" "setuptools<81" \
    transformers==4.28.0 "datasets<3.0" wandb sentencepiece accelerate==0.18.0
pip install "lm_eval==0.4.2" "peft==0.7.1"
pip install "bitsandbytes==0.48.2"
```

**4. Set the project root on `PYTHONPATH`:**
```sh
export PYTHONPATH=/path/to/wanda
```

---

## Model Weights

Model weights are loaded from Hugging Face Hub and cached locally in the `llm_weights/` directory by default. This path can be changed with `--cache_dir`.

**LLaMA-7B** (`huggyllama/llama-7b`) — publicly available on HF Hub, downloaded automatically on first run.

**LLaMA-2-7B** (`meta-llama/Llama-2-7b-hf`) — requires accepting Meta's license agreement on the [Hugging Face model page](https://huggingface.co/meta-llama/Llama-2-7b-hf) and authenticating with `huggingface-cli login` before downloading.

---

## Running the Code

### Command-Line Arguments

`main.py` is the primary entry point for pruning and evaluation.

| Argument | Description |
|---|---|
| `--model` | Hugging Face model identifier (e.g. `huggyllama/llama-7b`) |
| `--cache_dir` | Directory for caching model weights. Default: `llm_weights` |
| `--prune_method` | Pruning method: `magnitude`, `wanda`, `sparsegpt`, or ablation variants (`ablate_mag_seq`, `ablate_wanda_seq`, `ablate_mag_iter`, `ablate_wanda_iter`) |
| `--sparsity_ratio` | Fraction of weights to prune (e.g. `0.5` for 50%). Must be `0.5` for N:M sparsity types. |
| `--sparsity_type` | Sparsity pattern: `unstructured`, `2:4`, or `4:8` |
| `--nsamples` | Number of calibration samples. Default: `128` |
| `--seed` | Random seed for calibration data sampling. Default: `0` |
| `--use_variant` | Use the Wanda appendix variant for unstructured pruning |
| `--save` | Directory to write result JSON files |
| `--save_model` | Directory to save the pruned model weights |
| `--eval_zero_shot` | Also run zero-shot evaluation on BoolQ, RTE, HellaSwag, WinoGrande, ARC-easy, ARC-challenge, and OpenbookQA |
| `--zero_shot_batch_size` | Batch size for zero-shot evaluation (auto-detected by default) |
| `--skip_last_n_layers` | Leave the final N transformer layers unpruned (Wanda only) |
| `--skip_layers` | Space-separated list of layer indices to leave unpruned (Wanda only) |
| `--compare_original_modified` | Run both the original and modified implementations back-to-back and diff their pruned weights and perplexity |
| `--compare_sample_limit` | Max per-module sample entries in the comparison report. Default: `5` |

### Pruning LLaMA-7B

The [scripts/llama_7b.sh](scripts/llama_7b.sh) script runs Wanda, SparseGPT, and magnitude pruning across all three sparsity types, plus a dense (unpruned) baseline.

```sh
bash scripts/llama_7b.sh
```

To run a single configuration manually:
```sh
python main.py \
    --model huggyllama/llama-7b \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_7b/unstructured/wanda/
```

For structured N:M sparsity:
```sh
python main.py \
    --model huggyllama/llama-7b \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save out/llama_7b/2-4/wanda/
```

### Pruning LLaMA-2-7B

The [scripts/llama2_7b.sh](scripts/llama2_7b.sh) script runs the same suite for LLaMA-2-7B, including zero-shot evaluation.

```sh
bash scripts/llama2_7b.sh
```

To run a single configuration manually:
```sh
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama2_7b/unstructured/wanda/ \
    --eval_zero_shot
```

---

## Reproducing Experiments

### 1. Validating Correctness Against Original Codebase

The `--compare_original_modified` flag runs both the upstream original implementation and our modified implementation on the same model, then produces a JSON diff of pruned weight masks and perplexity values. This verifies that our changes do not alter the pruning outcome.

```sh
bash scripts/llama2_7b_compare.sh
```

Or for LLaMA-7B:
```sh
bash scripts/llama_7b_compare.sh
```

The scripts cover all three sparsity types (`unstructured`, `2:4`, `4:8`) and all three pruning methods (`wanda`, `sparsegpt`, `magnitude`). A JSON comparison report is written to the `--save` directory for each run.

### 2. Perplexity and Zero-Shot Accuracy

Full perplexity and zero-shot accuracy results for LLaMA-7B and LLaMA-2-7B are produced by the main experiment scripts. The LLaMA-2-7B script includes `--eval_zero_shot` automatically.

```sh
# LLaMA-7B (perplexity only)
bash scripts/llama_7b.sh

# LLaMA-2-7B + LLaMA-7B (perplexity + zero-shot)
bash scripts/llama2_7b.sh
```

For larger LLaMA variants:
```sh
bash scripts/llama_13b.sh
bash scripts/llama_30b.sh
bash scripts/llama_65b.sh
```

Zero-shot evaluation covers: BoolQ, RTE, HellaSwag, WinoGrande, ARC-easy, ARC-challenge, and OpenbookQA. Pass `--eval_zero_shot` to any `main.py` invocation to include it.

### 3. Sparsity Range Experiment

Sweeps unstructured sparsity from 10% to 90% in 10% increments for Wanda, SparseGPT, and magnitude pruning on LLaMA-2-7B.

```sh
bash scripts/llama2_7b_sparsity_sweep.sh
```

### 4. Number of Calibration Samples Experiment

Sweeps calibration sample counts from 1 to 512 (powers of two) for Wanda and SparseGPT on LLaMA-2-7B at 50% unstructured sparsity.

```sh
bash scripts/llama2_7b_nsamples_sweep.sh
```

### 5. Inference Speed Comparison

Measures GEMM speedup from 2:4 structured sparsity using PyTorch's `SparseSemiStructuredTensor`. Requires a GPU with compute capability ≥ 8.0 (Ampere or later) and PyTorch ≥ 2.1.

```sh
bash scripts/run_speedup.sh
```

Environment variables can override defaults:
```sh
SEQ_LEN=2048 BATCH_SIZE=1 USE_CUSPARSELT=1 bash scripts/run_speedup.sh
```

| Variable | Default | Description |
|---|---|---|
| `SEQ_LEN` | `2048` | Sequence length for benchmarks |
| `BATCH_SIZE` | `1` | Batch size |
| `NSAMPLES` | `128` | Calibration samples for pruning |
| `CUDA_DEVICE` | `0` | GPU index |
| `USE_CUSPARSELT` | `0` | Use cuSPARSELt kernel instead of CUTLASS |
| `SPARSE_ONLY` | `0` | Skip dense benchmarks (lower memory) |

### 6. Pruning Speed Comparison

Pruning speed (time spent computing masks, excluding forward passes) is reported automatically at the end of every `main.py` run. To compare methods side-by-side, run the main experiment scripts and compare the `pruning_time` field in the JSON output files. 

The `--compare_original_modified` flag (see [section 1](#1-validating-correctness-against-original-codebase)) also reports pruning time for both implementations in its JSON diff output.

### 7. Layer Sensitivity Search

Runs Wanda pruning one layer at a time while keeping all other layers dense, recording the per-layer perplexity impact. Results are saved as CSV files.

```sh
bash scripts/run_layer_sensitivity_wanda.sh
```

By default this evaluates LLaMA-2-7B. To use a different model or override settings:
```sh
MODEL=meta-llama/Llama-2-7b-hf \
NSAMPLES=128 \
SEED=0 \
bash scripts/run_layer_sensitivity_wanda.sh
```

The `layer_sensitivity.py` script can also be invoked directly for more control:

| Argument | Description |
|---|---|
| `--model` | HF model name/path (required) |
| `--prune_method` | `magnitude`, `wanda`, or `sparsegpt` (required) |
| `--sparsity_ratio` | Default: `0.5` |
| `--sparsity_type` | `unstructured`, `2:4`, `4:8`. Default: `unstructured` |
| `--nsamples` | Calibration samples. Default: `128` |
| `--seed` | Default: `0` |
| `--cache_dir` | Model weight cache. Default: `llm_weights` |
| `--save` | Output directory for CSV results |
| `--layers` | Comma/range list of layers to test, e.g. `0,1,5-8`. Default: all |
| `--skip_baseline` | Skip the dense baseline evaluation |

### 8. Skip Last 3 Layers

Runs Wanda pruning while leaving the final 3 transformer layers fully dense, across all sparsity types.

```sh
bash scripts/llama2_7b_skip_last3.sh
```

The `--skip_last_n_layers` argument to `main.py` controls how many final layers to preserve. Example:
```sh
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --skip_last_n_layers 3 \
    --save out/llama2_7b_skip_last3/unstructured/wanda/
```

### 9. Skip Top 25% Most Sensitive Layers

Runs Wanda pruning at a higher sparsity ratio (67%) while preserving the top 25% most sensitive layers (identified from the layer sensitivity search results). The sensitive layer indices are derived from the CSV outputs of the layer sensitivity search and are passed via `--skip_layers`.

The layers skipped for each sparsity type (top 8 of 32 by delta-PPL, from `layer_sensitivity_results/`):

| Sparsity type | Skipped layer indices |
|---|---|
| `unstructured` | 2, 9, 13, 16, 17, 29, 30, 31 |
| `2:4` | 2, 4, 9, 16, 25, 29, 30, 31 |
| `4:8` | 2, 4, 9, 16, 25, 29, 30, 31 |

```sh
bash scripts/llama2_7b_skip_sensitive.sh
```

Example for unstructured sparsity:
```sh
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.67 \
    --sparsity_type unstructured \
    --skip_layers 2 9 13 16 17 29 30 31 \
    --save out/llama2_7b_skip_sensitive/unstructured/wanda/
```

### 10. Quantization and Pruning Test

Compares Wanda pruning on FP16 vs INT8 (bitsandbytes `LLM.int8()`) models. Reports perplexity for dense and sparse variants of both precision types, and the mask difference (%) between FP16 and INT8 sparse models.

```sh
bash scripts/llama2_7b_quant.sh
```

Or directly:
```sh
python main_quant.py \
    --model meta-llama/Llama-2-7b-hf \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --seed 0 \
    --save out/llama2_7b_quant/unstructured/
```

---

## Acknowledgement

This repository is built upon the [SparseGPT](https://github.com/IST-DASLab/sparsegpt) repository.

## License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information. 