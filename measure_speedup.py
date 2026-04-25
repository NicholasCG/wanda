#!/usr/bin/env python3
"""
measure_speedup.py – Measure GEMM speedup from WANDA 2:4 sparsity.

Compares dense (zeroed 2:4-pruned) vs sparse (SparseSemiStructuredTensor) for:
  - Per-layer-type micro-benchmarks: q/k/v/o_proj, up/gate_proj, down_proj
  - End-to-end model forward-pass latency

Reuses get_llm / get_processing_device from main.py and
prune_wanda / find_layers from lib/prune.py without modification.

Requirements:
  PyTorch >= 2.1 with a CUDA GPU of compute capability >= 8.0 (Ampere or later).

Sparse backend: CUTLASS by default (dense always uses cuBLAS).
Pass --use_cusparselt to switch the sparse path to cuSPARSELt.

Usage:
  python measure_speedup.py \\
      --model huggyllama/llama-7b \\
      --cache_dir llm_weights \\
      --seq_len 512 --batch_size 1 \\
      --save out/speedup/
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from transformers import AutoTokenizer

# ── Re-use existing codebase without modification ────────────────────────────
from main import get_llm, get_processing_device
from lib.prune import prune_wanda, find_layers

# ── Layer groupings matching the Wanda paper's Table 5 ──────────────────────
LAYER_GROUPS = {
    "q/k/v/o_proj": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "up/gate_proj":  ["up_proj", "gate_proj"],
    "down_proj":     ["down_proj"],
}

# Projection layer names eligible for sparse conversion
_PROJ_SUFFIXES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "gate_proj", "down_proj",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_prune_args(cli_args):
    """
    Build a minimal args namespace that satisfies prune_wanda's expectations.
    Deliberately pins sparsity_type to "2:4" and sparsity_ratio to 0.5.
    """
    class _Args:
        pass

    a = _Args()
    a.model              = cli_args.model
    a.cache_dir          = cli_args.cache_dir
    a.nsamples           = cli_args.nsamples
    a.seed               = cli_args.seed
    a.sparsity_ratio     = 0.5      # required by n:m pruning path
    a.sparsity_type      = "2:4"
    a.use_variant        = False
    a.skip_last_n_layers = 0
    a.skip_layers        = []
    return a


def _find_proj_layer(transformer_block: nn.Module, short_name: str):
    """
    Return the nn.Linear whose last name component equals *short_name* within
    a single transformer block (e.g. 'q_proj', 'down_proj').
    Returns None if not found.
    """
    for mod_name, module in transformer_block.named_modules():
        if isinstance(module, nn.Linear) and mod_name.split(".")[-1] == short_name:
            return module
    return None


def _time_linear_ms(layer: nn.Linear, x: torch.Tensor, min_run_time: float = 1.0) -> float:
    """
    Return median forward-pass latency (ms) of layer(x) using
    torch.utils.benchmark.Timer.blocked_autorange().

    Runs inside torch.inference_mode() to match real inference conditions and
    to invoke the aten.linear dispatch path used by SparseSemiStructuredTensor.
    """
    with torch.inference_mode():
        t = benchmark.Timer(
            stmt="layer(x)",
            globals={"layer": layer, "x": x},
        ).blocked_autorange(min_run_time=min_run_time)
    return t.median * 1e3


def _time_model_ms(model, input_ids: torch.Tensor, min_run_time: float = 2.0) -> float:
    """Return median full-model forward-pass latency (ms)."""
    with torch.inference_mode():
        t = benchmark.Timer(
            stmt="model(input_ids)",
            globals={"model": model, "input_ids": input_ids},
        ).blocked_autorange(min_run_time=min_run_time)
    return t.median * 1e3


def _convert_to_sparse_inplace(model) -> tuple:
    """
    Replace projection layer weight parameters in-place with
    SparseSemiStructuredTensor. Operates on the already-pruned (zeroed) model
    produced by prune_wanda so weights already satisfy the 2:4 pattern.

    Shape constraint for fp16: both dims must be >= 32/64 and multiples of 32/64.
    Returns (n_converted, n_skipped).

    Memory strategy: get_llm uses device_map="auto" (accelerate), which registers
    AlignDevicesHook on every module. That hook's weights_map holds a reference to
    the original weight tensor, so simply reassigning module.weight does not drop
    the old GPU tensor's refcount to zero — memory grows instead of shrinking.
    Fix: copy to CPU, null _parameters["weight"] (drops all Python refs), call
    empty_cache() to actually free the GPU pages, then move back to GPU and convert.
    Peak transient overhead is ~one layer at a time rather than the full model.
    """
    converted = skipped = 0
    for fqn, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if fqn.split(".")[-1] not in _PROJ_SUFFIXES:
            continue
        r, c = module.weight.shape
        if r < 32 or c < 64 or r % 32 or c % 64:
            skipped += 1
            continue

        device = module.weight.device
        # Step 1: back up weight on CPU (no GPU alloc).
        w_cpu = module.weight.data.cpu()
        # Step 2: drop the GPU tensor — null both the parameter slot and any
        # accelerate hook weight_map entry that might hold a stale reference.
        module._parameters["weight"] = None
        _clear_hook_weight_map(module, "weight")
        torch.cuda.empty_cache()
        # Step 3: move back to GPU and compress.
        w_gpu = w_cpu.to(device)
        del w_cpu
        module.weight = nn.Parameter(
            to_sparse_semi_structured(w_gpu.contiguous()),
            requires_grad=False,
        )
        del w_gpu
        torch.cuda.empty_cache()
        converted += 1
    return converted, skipped


def _clear_hook_weight_map(module: nn.Module, param_name: str) -> None:
    """Remove a parameter entry from accelerate AlignDevicesHook.weights_map."""
    for hook in module._forward_pre_hooks.values():
        wmap = getattr(hook, "weights_map", None)
        if wmap is not None and param_name in wmap:
            wmap[param_name] = None
    for hook in module._forward_hooks.values():
        wmap = getattr(hook, "weights_map", None)
        if wmap is not None and param_name in wmap:
            wmap[param_name] = None


# ── Benchmark routines ───────────────────────────────────────────────────────

def _run_micro_benchmarks_dense(model, seq_len: int, batch_size: int, device) -> dict:
    """
    Time each projection layer type (layer 0 as representative) in dense mode.
    Returns a dict: short_name -> (ms, in_features, weight_shape, input_shape).
    """
    results = {}
    for _, short_names in LAYER_GROUPS.items():
        for sname in short_names:
            layer = _find_proj_layer(model.model.layers[0], sname)
            if layer is None:
                print(f"  [warn] '{sname}' not found in layer 0 – skipping")
                continue
            x = torch.randn(
                batch_size * seq_len, layer.in_features,
                dtype=torch.float16, device=device,
            )
            ms = _time_linear_ms(layer, x)
            results[sname] = (ms, layer.in_features, list(layer.weight.shape), list(x.shape))
            print(f"  {sname:<14}  {ms:>8.3f} ms")
    return results


def _run_micro_benchmarks_sparse(model, dense_results, seq_len: int,
                                  batch_size: int, device) -> dict:
    """
    Time each projection layer type (layer 0 as representative) in sparse mode.
    dense_results provides the cached dense latencies for speedup calculation;
    pass None to benchmark sparse latency only (no speedup ratio computed).
    Returns structured results per LAYER_GROUPS.
    """
    layer_results = {}
    for group_name, short_names in LAYER_GROUPS.items():
        per_layer = {}
        for sname in short_names:
            layer = _find_proj_layer(model.model.layers[0], sname)
            if layer is None:
                continue
            # Derive in_features from the sparse weight's original shape.
            in_feat = layer.weight.shape[1]
            if dense_results is not None and sname not in dense_results:
                continue
            dense_ms = dense_results[sname][0] if dense_results is not None else None
            w_shape  = list(layer.weight.shape)
            x_shape  = [batch_size * seq_len, in_feat]
            x = torch.randn(batch_size * seq_len, in_feat,
                            dtype=torch.float16, device=device)
            sparse_ms = _time_linear_ms(layer, x)
            entry = {
                "weight_shape": w_shape,
                "input_shape":  x_shape,
                "sparse_ms":    round(sparse_ms, 4),
            }
            if dense_ms is not None:
                speedup = dense_ms / sparse_ms
                entry["dense_ms"] = round(dense_ms, 4)
                entry["speedup"]  = round(speedup,  4)
                print(f"  {sname:<14}"
                      f"  dense={dense_ms:>7.3f} ms"
                      f"  sparse={sparse_ms:>7.3f} ms"
                      f"  speedup={speedup:>6.3f}x")
            else:
                print(f"  {sname:<14}  sparse={sparse_ms:>7.3f} ms")
            per_layer[sname] = entry

        if per_layer:
            sparse_vals = [v["sparse_ms"] for v in per_layer.values()]
            avg_s = sum(sparse_vals) / len(sparse_vals)
            grp_entry = {
                "per_layer":     per_layer,
                "avg_sparse_ms": round(avg_s, 4),
            }
            if dense_results is not None:
                dense_vals = [v["dense_ms"] for v in per_layer.values()]
                avg_d = sum(dense_vals) / len(dense_vals)
                grp_entry["avg_dense_ms"] = round(avg_d, 4)
                grp_entry["avg_speedup"]  = round(avg_d / avg_s, 4)
            layer_results[group_name] = grp_entry

    return layer_results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Measure 2:4 sparsity GEMM speedup after Wanda pruning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model id or local path (e.g. huggyllama/llama-7b)",
    )
    parser.add_argument("--cache_dir",  default="llm_weights",
                        help="Directory for cached model weights")
    parser.add_argument("--nsamples",   type=int, default=128,
                        help="Calibration samples for Wanda")
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--seq_len",    type=int, default=None,
                        help="Token sequence length for micro- and e2e benchmarks "
                             "(default: model.config.max_position_embeddings)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for the end-to-end benchmark")
    parser.add_argument("--save",       default=None,
                        help="Directory to write a JSON results file")
    parser.add_argument(
        "--use_cusparselt", action="store_true",
        help="Use cuSPARSELt backend instead of the default CUTLASS",
    )
    parser.add_argument(
        "--sparse_only", action="store_true",
        help="Skip all dense benchmarks; convert to sparse immediately after "
             "pruning to minimise peak GPU memory usage.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # ── GPU check ────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA is required for SparseSemiStructuredTensor.")

    # ── Sparse backend selection ─────────────────────────────────────────────
    # Dense nn.Linear always dispatches through cuBLAS (PyTorch default).
    # Sparse path defaults to CUTLASS; pass --use_cusparselt to override.
    if args.use_cusparselt:
        print("Sparse backend: cuSPARSELt")
    else:
        SparseSemiStructuredTensor._FORCE_CUTLASS = True
        print("Sparse backend: CUTLASS (use --use_cusparselt to override)")

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = get_llm(args.model, args.cache_dir)
    model.eval()

    device = get_processing_device(model, args.model)
    print(f"Device: {device}")

    if args.seq_len is None:
        args.seq_len = model.seqlen
        print(f"seq_len not specified – using model.seqlen = {args.seq_len}")

    # ── Wanda 2:4 pruning ────────────────────────────────────────────────────
    print("\nRunning Wanda 2:4 pruning …")
    prune_wanda(_make_prune_args(args), model, tokenizer, device, prune_n=2, prune_m=4)
    print("Pruning complete.")

    if args.sparse_only:
        # ── Sparse-only path: convert immediately after pruning ───────────────
        # No dense benchmarks are run, so the allocator cache is clean and
        # the conversion can use all available GPU memory.
        torch.cuda.empty_cache()
        print("\nConverting projection layer weights to SparseSemiStructuredTensor …")
        n_conv, n_skip = _convert_to_sparse_inplace(model)
        print(f"  {n_conv} layers converted, {n_skip} skipped (shape constraint)")

        print(f"\n[micro / sparse]  layer 0 representative  "
              f"(seq_len={args.seq_len}, batch_size={args.batch_size})")
        layer_results = _run_micro_benchmarks_sparse(
            model, None, args.seq_len, args.batch_size, device
        )

        print("\n[e2e / sparse]  full model forward pass …")
        input_ids = torch.randint(
            0, model.config.vocab_size,
            (args.batch_size, args.seq_len),
            device=device,
        )
        sparse_e2e_ms = _time_model_ms(model, input_ids)
        print(f"  sparse e2e: {sparse_e2e_ms:.1f} ms")

        sep  = "=" * 64
        dash = "-" * 64
        print(f"\n{sep}")
        print(f"  Sparse-only: {args.model}")
        print(f"  Pruning: wanda 2:4  |  seq_len={args.seq_len}  batch_size={args.batch_size}")
        print(sep)
        print(f"  {'Layer type':<20}  {'Sparse':>9}")
        print(f"  {dash}")
        for grp, data in layer_results.items():
            print(f"  {grp:<20}  {data['avg_sparse_ms']:>8.3f}ms")
        print(f"  {dash}")
        print(f"  {'End-to-end':<20}  {sparse_e2e_ms:>8.1f}ms")
        print(sep)

        report = {
            "model":         args.model,
            "prune_method":  "wanda",
            "sparsity_type": "2:4",
            "sparse_only":   True,
            "seq_len":       args.seq_len,
            "batch_size":    args.batch_size,
            "layer_micro_benchmarks": layer_results,
            "end_to_end": {"sparse_ms": round(sparse_e2e_ms, 2)},
        }
        if args.save:
            os.makedirs(args.save, exist_ok=True)
            tag      = args.model.replace("/", "--")
            out_path = os.path.join(args.save, f"speedup_{tag}_seq{args.seq_len}_sparse_only.json")
            with open(out_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nResults saved → {out_path}")
        return

    # ── Full path (dense + sparse) ────────────────────────────────────────────
    # These run BEFORE in-place sparse conversion so the underlying weights are
    # still dense fp16 tensors (just with 2:4 zeros applied by wanda).
    print(f"\n[micro / dense]  layer 0 representative  "
          f"(seq_len={args.seq_len}, batch_size={args.batch_size})")
    dense_micro = _run_micro_benchmarks_dense(
        model, args.seq_len, args.batch_size, device
    )

    # ── End-to-end: dense ─────────────────────────────────────────────────────
    print(f"\n[e2e / dense]  full model forward pass …")
    input_ids = torch.randint(
        0, model.config.vocab_size,
        (args.batch_size, args.seq_len),
        device=device,
    )
    dense_e2e_ms = _time_model_ms(model, input_ids)
    print(f"  dense e2e: {dense_e2e_ms:.1f} ms")

    # ── Convert weights to SparseSemiStructuredTensor (in-place) ─────────────
    # blocked_autorange above ran the full forward pass many times, leaving
    # gigabytes of freed activation tensors in PyTorch's CUDA allocator cache.
    # Releasing them now gives cudaMalloc headroom for the compressed tensors.
    torch.cuda.empty_cache()
    # In-place conversion avoids holding a second 7 B-parameter copy in memory.
    print("\nConverting projection layer weights to SparseSemiStructuredTensor …")
    n_conv, n_skip = _convert_to_sparse_inplace(model)
    print(f"  {n_conv} layers converted, {n_skip} skipped (shape constraint)")

    # ── Micro-benchmarks: sparse ──────────────────────────────────────────────
    print(f"\n[micro / sparse]  layer 0 representative  "
          f"(seq_len={args.seq_len}, batch_size={args.batch_size})")
    layer_results = _run_micro_benchmarks_sparse(
        model, dense_micro, args.seq_len, args.batch_size, device
    )

    # ── End-to-end: sparse ────────────────────────────────────────────────────
    print("\n[e2e / sparse]  full model forward pass …")
    sparse_e2e_ms = _time_model_ms(model, input_ids)
    e2e_speedup   = dense_e2e_ms / sparse_e2e_ms
    print(f"  sparse e2e: {sparse_e2e_ms:.1f} ms  speedup={e2e_speedup:.3f}x")

    # ── Summary table ─────────────────────────────────────────────────────────
    sep = "=" * 64
    dash = "-" * 64
    print(f"\n{sep}")
    print(f"  Speedup: {args.model}")
    print(f"  Pruning: wanda 2:4  |  seq_len={args.seq_len}  batch_size={args.batch_size}")
    print(sep)
    print(f"  {'Layer type':<20}  {'Dense':>9}  {'Sparse':>9}  {'Speedup':>9}")
    print(f"  {dash}")
    for grp, data in layer_results.items():
        print(f"  {grp:<20}  {data['avg_dense_ms']:>8.3f}ms"
              f"  {data['avg_sparse_ms']:>8.3f}ms"
              f"  {data['avg_speedup']:>8.2f}x")
    print(f"  {dash}")
    print(f"  {'End-to-end':<20}  {dense_e2e_ms:>8.1f}ms"
          f"  {sparse_e2e_ms:>8.1f}ms"
          f"  {e2e_speedup:>8.2f}x")
    print(sep)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    report = {
        "model":         args.model,
        "prune_method":  "wanda",
        "sparsity_type": "2:4",
        "seq_len":       args.seq_len,
        "batch_size":    args.batch_size,
        "layer_micro_benchmarks": layer_results,
        "end_to_end": {
            "dense_ms":  round(dense_e2e_ms,  2),
            "sparse_ms": round(sparse_e2e_ms, 2),
            "speedup":   round(e2e_speedup,   4),
        },
    }

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        tag      = args.model.replace("/", "--")
        out_path = os.path.join(args.save, f"speedup_{tag}_seq{args.seq_len}.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
