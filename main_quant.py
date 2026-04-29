"""
Quantization experiment: compare Wanda pruning on FP16 vs INT8 (LLM.int8()) models.

Evaluates four configurations:
  1. FP16 dense  - wikitext2 perplexity of the unmodified FP16 model
  2. FP16 sparse - wikitext2 perplexity after Wanda pruning on FP16 weights
  3. INT8 dense  - wikitext2 perplexity of the unmodified INT8 model
  4. INT8 sparse - wikitext2 perplexity after Wanda pruning on INT8 weights

Additionally reports the mask difference (%) between the FP16 sparse and INT8
sparse models: this quantifies how much quantization changes the set of weights
that Wanda chooses to prune.
"""

import argparse
import gc
import json
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# Re-use all existing library code.
from lib.prune import find_layers, check_sparsity, prepare_calibration_input
from lib.data import get_loaders
from lib.eval import eval_ppl
from lib.layerwrapper import WrappedGPT


def _linear_layer_types():
    """
    Return the list of layer types that find_layers should match.

    find_layers uses exact ``type(module) in layers`` matching (not isinstance),
    so bitsandbytes Linear8bitLt — which subclasses nn.Linear but has a
    different concrete type — must be added explicitly, otherwise the subset
    dict comes back empty on INT8 models.
    """
    types = [nn.Linear]
    try:
        import bitsandbytes as bnb
        types.append(bnb.nn.Linear8bitLt)
    except ImportError:
        pass
    return types


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def get_fp16_model(model_name, cache_dir="llm_weights"):
    """Load model in FP16 (standard Wanda setup)."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.seqlen = model.config.max_position_embeddings
    return model


def get_int8_model(model_name, cache_dir="llm_weights"):
    """Load model with LLM.int8() 8-bit quantization via bitsandbytes."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.seqlen = model.config.max_position_embeddings
    return model


def get_processing_device(model, model_name):
    device = torch.device("cuda:0")
    if "30b" in model_name or "65b" in model_name:
        device = model.hf_device_map["lm_head"]
    return device


# ---------------------------------------------------------------------------
# INT8-aware weight helpers
# ---------------------------------------------------------------------------

def _is_int8_linear(layer):
    """Return True if layer is a bitsandbytes Linear8bitLt (already quantized)."""
    try:
        import bitsandbytes as bnb
        return isinstance(layer, bnb.nn.Linear8bitLt)
    except ImportError:
        return False


def _get_weight_float(layer):
    """
    Return the effective weight matrix as a float32 CPU tensor.

    For a standard nn.Linear this is simply layer.weight.data.
    For a quantized Linear8bitLt the int8 weights (CB) are stored in
    weight.data after the first forward pass has been run (which happens
    during calibration-input collection).  We dequantize via:
        W_float = CB.float() * (SCB / 127.0).unsqueeze(1)
    """
    if _is_int8_linear(layer):
        weight_param = layer.weight
        # CB is stored in weight.data for quantized layers; SCB holds row scales.
        CB = getattr(weight_param, "CB", None)
        SCB = getattr(weight_param, "SCB", None)
        if CB is not None and SCB is not None:
            return (CB.float() * (SCB.float() / 127.0).unsqueeze(1)).cpu()
        # Fallback: weight may still be fp16 if not yet quantized (e.g. on CPU).
        return layer.weight.data.float().cpu()
    return layer.weight.data.float().cpu()


def _apply_mask_to_layer(layer, W_mask):
    """
    Zero out pruned weights in-place.

    For FP16 layers: zero weight.data directly (same as standard Wanda).
    For INT8 layers: zero the int8 CB tensor in weight.data so that the
    pruned coefficients stay at zero after dequantization.
    """
    if _is_int8_linear(layer):
        weight_param = layer.weight
        CB = getattr(weight_param, "CB", None)
        if CB is not None:
            CB[W_mask.to(CB.device)] = 0
            return
        # Fallback if CB is not yet populated (layer not yet quantized)
        layer.weight.data[W_mask.to(layer.weight.device)] = 0
    else:
        layer.weight.data[W_mask.to(layer.weight.device)] = 0


# ---------------------------------------------------------------------------
# Quantization-aware Wanda pruning
# ---------------------------------------------------------------------------

def prune_wanda_quant(args, model, tokenizer, device, prune_n=0, prune_m=0):
    """
    Wanda pruning that works on both FP16 and INT8 (bitsandbytes) models.

    The importance metric and mask computation use the dequantized float
    representation of the weights, ensuring that the Wanda criterion is
    applied correctly regardless of storage format.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibration data")
    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, args.nsamples
        )

    layers = model.model.layers
    skip_last_n = getattr(args, "skip_last_n_layers", 0)
    skip_set = set(getattr(args, "skip_layers", None) or [])
    if skip_last_n > 0:
        skip_set |= set(range(len(layers) - skip_last_n, len(layers)))

    layer_types = _linear_layer_types()

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer, layers=layer_types)

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
        else:
            dev = next(iter(subset.values())).weight.device
        attention_mask_dev = attention_mask.to(dev) if attention_mask is not None else None
        position_ids_dev = position_ids.to(dev) if position_ids is not None else None

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                inp_j = inps[j].unsqueeze(0).to(dev)
                out_j = layer(inp_j, attention_mask=attention_mask_dev, position_ids=position_ids_dev)[0]
                outs[j] = out_j.squeeze(0).to("cpu")

        for h in handles:
            h.remove()

        if i in skip_set:
            print(f"layer {i} skipped (preserved: dense)")
            inps, outs = outs, inps
            continue

        for name in subset:
            print(f"pruning layer {i} name {name}")

            # Obtain float weights (handles both FP16 and INT8).
            W_float = _get_weight_float(subset[name]).to(dev)
            scaler_row = wrapped_layers[name].scaler_row.to(dev)

            W_metric = torch.abs(W_float) * torch.sqrt(scaler_row.reshape(1, -1))

            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)

            if prune_n != 0:
                # Structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : ii + prune_m].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                if args.use_variant:
                    from lib.prune import return_given_alpha
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)
                    alpha = 0.4
                    alpha_hist = [0.0, 0.8]
                    W_mask, cur_sparsity = return_given_alpha(
                        alpha, sort_res, W_metric, tmp_metric, sum_before
                    )
                    while (
                        torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001
                        and (alpha_hist[1] - alpha_hist[0] >= 0.001)
                    ):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha
                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before
                        )
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    indices = torch.topk(
                        W_metric,
                        int(W_metric.shape[1] * args.sparsity_ratio),
                        dim=1,
                        largest=False,
                    )[1]
                    W_mask.scatter_(1, indices, True)

            _apply_mask_to_layer(subset[name], W_mask)
            del W_float, W_metric, W_mask, scaler_row

        print(f"layer {i} done")

        for j in range(args.nsamples):
            with torch.no_grad():
                inp_j = inps[j].unsqueeze(0).to(dev)
                out_j = layer(inp_j, attention_mask=attention_mask_dev, position_ids=position_ids_dev)[0]
                outs[j] = out_j.squeeze(0).to("cpu")
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Sparsity check for INT8 models
# ---------------------------------------------------------------------------

def check_sparsity_quant(model):
    """
    Like lib.prune.check_sparsity but dequantizes INT8 layers before checking.
    For FP16 layers falls back to the standard zero-count approach.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0

    layer_types = _linear_layer_types()

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer, layers=layer_types)
        sub_count = 0
        sub_params = 0
        for name in subset:
            W = _get_weight_float(subset[name])
            zeros = (W == 0).sum().item()
            numel = W.numel()
            count += zeros
            total_params += numel
            sub_count += zeros
            sub_params += numel
        print(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params


# ---------------------------------------------------------------------------
# Mask snapshot & comparison (works for both FP16 and INT8)
# ---------------------------------------------------------------------------

def snapshot_masks_cpu(model):
    """
    Record the zero-mask for every Linear layer (supports FP16 + INT8).
    Returns dict: (layer_idx, layer_name) -> BoolTensor on CPU.
    """
    masks = {}
    layer_types = _linear_layer_types()
    layers = model.model.layers
    for i in range(len(layers)):
        subset = find_layers(layers[i], layers=layer_types)
        for name in subset:
            W = _get_weight_float(subset[name])
            masks[(i, name)] = (W == 0)
    return masks


def compare_masks(masks_fp16, masks_int8):
    """
    Compute element-wise mask difference between two sparse models.

    Returns a dict with:
      total_weights         - total number of weight elements compared
      fp16_zero_count       - number of zeros in FP16 sparse model
      int8_zero_count       - number of zeros in INT8 sparse model
      mask_diff_count       - positions where the two masks disagree
      mask_diff_pct         - mask_diff_count / total_weights * 100
    """
    total_weights = 0
    fp16_zeros = 0
    int8_zeros = 0
    diff_count = 0

    for key in masks_fp16:
        m_fp16 = masks_fp16[key]
        m_int8 = masks_int8[key]
        total_weights += m_fp16.numel()
        fp16_zeros += int(m_fp16.sum().item())
        int8_zeros += int(m_int8.sum().item())
        diff_count += int((m_fp16 != m_int8).sum().item())

    return {
        "total_weights": total_weights,
        "fp16_zero_count": fp16_zeros,
        "int8_zero_count": int8_zeros,
        "mask_diff_count": diff_count,
        "mask_diff_pct": 100.0 * diff_count / total_weights if total_weights else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Wanda pruning on FP16 vs INT8 (bitsandbytes) models."
    )
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--sparsity_ratio", type=float, default=0.5)
    parser.add_argument(
        "--sparsity_type",
        type=str,
        default="unstructured",
        choices=["unstructured", "4:8", "2:4"],
    )
    parser.add_argument("--cache_dir", type=str, default="llm_weights")
    parser.add_argument(
        "--use_variant",
        action="store_true",
        help="Use the Wanda variant described in the appendix.",
    )
    parser.add_argument(
        "--skip_last_n_layers",
        type=int,
        default=0,
        help="Number of final transformer layers to leave unpruned.",
    )
    parser.add_argument("--skip_layers", type=int, nargs="*", default=[])
    parser.add_argument(
        "--save",
        type=str,
        default="out/quant_experiment",
        help="Directory to save the JSON result report.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, (
            "sparsity ratio must be 0.5 for structured N:M sparsity"
        )
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model_name = args.model.split("/")[-1]

    results = {
        "model": args.model,
        "sparsity_type": args.sparsity_type,
        "sparsity_ratio": args.sparsity_ratio,
        "nsamples": args.nsamples,
        "seed": args.seed,
    }

    # ------------------------------------------------------------------
    # 1. FP16 dense perplexity
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 1/4: FP16 dense perplexity")
    print("=" * 60)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    model_fp16 = get_fp16_model(args.model, args.cache_dir)
    model_fp16.eval()
    device_fp16 = get_processing_device(model_fp16, args.model)

    ppl_fp16_dense = eval_ppl(args, model_fp16, tokenizer, device_fp16)
    print(f"FP16 dense wikitext2 PPL: {ppl_fp16_dense:.4f}")
    results["fp16_dense_ppl"] = ppl_fp16_dense

    # ------------------------------------------------------------------
    # 2. FP16 sparse perplexity
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2/4: FP16 sparse perplexity (Wanda pruning)")
    print("=" * 60)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    prune_wanda_quant(args, model_fp16, tokenizer, device_fp16, prune_n, prune_m)

    sparsity_fp16 = check_sparsity_quant(model_fp16)
    print(f"FP16 actual sparsity: {sparsity_fp16:.4f}")
    ppl_fp16_sparse = eval_ppl(args, model_fp16, tokenizer, device_fp16)
    print(f"FP16 sparse wikitext2 PPL: {ppl_fp16_sparse:.4f}")
    results["fp16_sparse_actual_sparsity"] = sparsity_fp16
    results["fp16_sparse_ppl"] = ppl_fp16_sparse

    # Snapshot the FP16 masks before freeing the model.
    masks_fp16 = snapshot_masks_cpu(model_fp16)

    del model_fp16
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3. INT8 dense perplexity
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3/4: INT8 dense perplexity (LLM.int8() via bitsandbytes)")
    print("=" * 60)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    model_int8 = get_int8_model(args.model, args.cache_dir)
    model_int8.eval()
    device_int8 = get_processing_device(model_int8, args.model)

    ppl_int8_dense = eval_ppl(args, model_int8, tokenizer, device_int8)
    print(f"INT8 dense wikitext2 PPL: {ppl_int8_dense:.4f}")
    results["int8_dense_ppl"] = ppl_int8_dense

    # ------------------------------------------------------------------
    # 4. INT8 sparse perplexity
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4/4: INT8 sparse perplexity (Wanda pruning on INT8 model)")
    print("=" * 60)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    prune_wanda_quant(args, model_int8, tokenizer, device_int8, prune_n, prune_m)

    sparsity_int8 = check_sparsity_quant(model_int8)
    print(f"INT8 actual sparsity: {sparsity_int8:.4f}")
    ppl_int8_sparse = eval_ppl(args, model_int8, tokenizer, device_int8)
    print(f"INT8 sparse wikitext2 PPL: {ppl_int8_sparse:.4f}")
    results["int8_sparse_actual_sparsity"] = sparsity_int8
    results["int8_sparse_ppl"] = ppl_int8_sparse

    # ------------------------------------------------------------------
    # 5. Mask difference between FP16 sparse and INT8 sparse
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5/5: Mask difference (FP16 sparse vs INT8 sparse)")
    print("=" * 60)
    masks_int8 = snapshot_masks_cpu(model_int8)
    mask_comparison = compare_masks(masks_fp16, masks_int8)

    print(f"Total weights compared : {mask_comparison['total_weights']:,}")
    print(f"FP16 pruned (zeros)    : {mask_comparison['fp16_zero_count']:,}")
    print(f"INT8 pruned (zeros)    : {mask_comparison['int8_zero_count']:,}")
    print(f"Mask differences       : {mask_comparison['mask_diff_count']:,}")
    print(f"Mask difference (%%)   : {mask_comparison['mask_diff_pct']:.4f}%%")
    results["mask_comparison"] = mask_comparison

    del model_int8
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model            : {args.model}")
    print(f"Sparsity type    : {args.sparsity_type}")
    print(f"Sparsity ratio   : {args.sparsity_ratio}")
    print(f"FP16 dense PPL   : {ppl_fp16_dense:.4f}")
    print(f"FP16 sparse PPL  : {ppl_fp16_sparse:.4f}")
    print(f"INT8 dense PPL   : {ppl_int8_dense:.4f}")
    print(f"INT8 sparse PPL  : {ppl_int8_sparse:.4f}")
    print(f"Mask diff (%%)    : {mask_comparison['mask_diff_pct']:.4f}%%")

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    os.makedirs(args.save, exist_ok=True)
    save_path = os.path.join(
        args.save,
        f"quant_experiment_{model_name}_{args.sparsity_type.replace(':', '-')}.json",
    )
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
