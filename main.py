import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
import json
import gc

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
from lib.eval import eval_ppl, eval_zero_shot
from lib.prune_original import (
    prune_wanda as prune_wanda_original,
    prune_magnitude as prune_magnitude_original,
    prune_sparsegpt as prune_sparsegpt_original,
    prune_ablate as prune_ablate_original,
)
from lib.eval_original import eval_ppl as eval_ppl_original

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings
    # model.seqlen = 512
    return model


def get_processing_device(model, model_name):
    device = torch.device("cuda:0")
    if "30b" in model_name or "65b" in model_name:
        device = model.hf_device_map["lm_head"]
    return device


def run_pruning(args, model, tokenizer, device, prune_n, prune_m, use_original=False):
    if args.sparsity_ratio == 0:
        return

    if use_original:
        if args.prune_method == "wanda":
            prune_wanda_original(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude_original(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt_original(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate_original(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        return

    if args.prune_method == "wanda":
        prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "magnitude":
        prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "sparsegpt":
        prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif "ablate" in args.prune_method:
        prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)


def snapshot_pruned_weights_cpu(model):
    snapshot = {}
    layers = model.model.layers
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        for name in subset:
            snapshot[(i, name)] = subset[name].weight.data.detach().to("cpu").clone()
    return snapshot


def compare_pruned_models(original_snapshot_cpu, model_modified, sample_limit=5):
    diff_entries = []
    total_mask_diffs = 0
    total_weights = 0

    layers_modified = model_modified.model.layers

    for i in range(len(layers_modified)):
        subset_modified = find_layers(layers_modified[i])

        for name in subset_modified:
            W_orig = original_snapshot_cpu[(i, name)]
            W_mod = subset_modified[name].weight.data.detach().to("cpu")

            zero_orig = W_orig == 0
            zero_mod = W_mod == 0
            mask_diff = zero_orig != zero_mod

            mask_diff_count = int(mask_diff.sum().item())
            weight_count = int(W_orig.numel())
            total_mask_diffs += mask_diff_count
            total_weights += weight_count

            abs_diff = (W_orig - W_mod).abs()
            max_abs_diff = float(abs_diff.max().item())
            mean_abs_diff = float(abs_diff.mean().item())

            if mask_diff_count > 0 or max_abs_diff > 0:
                sample_diffs = []
                if mask_diff_count > 0:
                    sample_idx = torch.nonzero(mask_diff, as_tuple=False)[:sample_limit]
                    for idx in sample_idx:
                        r = int(idx[0].item())
                        c = int(idx[1].item())
                        sample_diffs.append(
                            {
                                "index": [r, c],
                                "original_value": float(W_orig[r, c].item()),
                                "modified_value": float(W_mod[r, c].item()),
                                "original_zero": bool(zero_orig[r, c].item()),
                                "modified_zero": bool(zero_mod[r, c].item()),
                            }
                        )

                diff_entries.append(
                    {
                        "layer": i,
                        "module": name,
                        "num_weights": weight_count,
                        "mask_diff_count": mask_diff_count,
                        "mask_diff_ratio": mask_diff_count / weight_count,
                        "max_abs_weight_diff": max_abs_diff,
                        "mean_abs_weight_diff": mean_abs_diff,
                        "sample_mask_differences": sample_diffs,
                    }
                )

    diff_entries.sort(
        key=lambda x: (x["mask_diff_count"], x["max_abs_weight_diff"]),
        reverse=True,
    )

    return {
        "total_mask_diff_count": total_mask_diffs,
        "total_weights": total_weights,
        "total_mask_diff_ratio": (total_mask_diffs / total_weights) if total_weights else 0.0,
        "different_modules": diff_entries,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument("--eval_zero_shot", action="store_true")
    parser.add_argument(
        "--compare_original_modified",
        action="store_true",
        help="Run original and modified implementations and report result differences.",
    )
    parser.add_argument(
        "--compare_sample_limit",
        type=int,
        default=5,
        help="Max sample entries per module in comparison report.",
    )
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    if args.compare_original_modified:
        print("comparison mode: running original implementation")
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        model_original = get_llm(args.model, args.cache_dir)
        model_original.eval()
        device_original = get_processing_device(model_original, args.model)
        print("original use device", device_original)

        run_pruning(args, model_original, tokenizer, device_original, prune_n, prune_m, use_original=True)
        sparsity_original = check_sparsity(model_original)
        ppl_original = eval_ppl_original(args, model_original, tokenizer, device_original)

        # Move original run artifacts off GPU before running modified path.
        original_snapshot_cpu = snapshot_pruned_weights_cpu(model_original)
        del model_original
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

        print("comparison mode: running modified implementation")
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        model_modified = get_llm(args.model, args.cache_dir)
        model_modified.eval()
        device_modified = get_processing_device(model_modified, args.model)
        print("modified use device", device_modified)

        run_pruning(args, model_modified, tokenizer, device_modified, prune_n, prune_m, use_original=False)
        sparsity_modified = check_sparsity(model_modified)
        ppl_modified = eval_ppl(args, model_modified, tokenizer, device_modified)

        model_diff = compare_pruned_models(
            original_snapshot_cpu, model_modified, sample_limit=args.compare_sample_limit
        )

        comparison_report = {
            "method": args.prune_method,
            "model": args.model,
            "sparsity_type": args.sparsity_type,
            "sparsity_ratio": args.sparsity_ratio,
            "nsamples": args.nsamples,
            "original": {
                "actual_sparsity": sparsity_original,
                "wikitext_ppl": ppl_original,
            },
            "modified": {
                "actual_sparsity": sparsity_modified,
                "wikitext_ppl": ppl_modified,
            },
            "delta": {
                "sparsity": float(sparsity_modified - sparsity_original),
                "ppl": float(ppl_modified - ppl_original),
            },
            "model_weight_comparison": model_diff,
        }

        has_diff = (
            abs(comparison_report["delta"]["sparsity"]) > 0
            or abs(comparison_report["delta"]["ppl"]) > 0
            or model_diff["total_mask_diff_count"] > 0
        )

        print("=" * 60)
        print("comparison summary")
        print("differences_found:", has_diff)
        print("original sparsity:", f"{sparsity_original:.6f}")
        print("modified sparsity:", f"{sparsity_modified:.6f}")
        print("delta sparsity:", f"{comparison_report['delta']['sparsity']:.6f}")
        print("original ppl:", f"{ppl_original:.6f}")
        print("modified ppl:", f"{ppl_modified:.6f}")
        print("delta ppl:", f"{comparison_report['delta']['ppl']:.6f}")
        print("mask diff ratio:", f"{model_diff['total_mask_diff_ratio']:.8f}")

        if model_diff["different_modules"]:
            print("sample module differences:")
            for entry in model_diff["different_modules"][:3]:
                print(
                    f"layer={entry['layer']} module={entry['module']} "
                    f"mask_diff_count={entry['mask_diff_count']} "
                    f"mask_diff_ratio={entry['mask_diff_ratio']:.8f} "
                    f"max_abs_weight_diff={entry['max_abs_weight_diff']:.8f}"
                )
                if entry["sample_mask_differences"]:
                    print("sample_mask_differences:", entry["sample_mask_differences"])
        else:
            print("no module-level differences detected")

        print("=" * 60)

        save_dir = args.save if args.save else os.path.join("out", "compare")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        report_path = os.path.join(save_dir, f"compare_{args.prune_method}.json")
        with open(report_path, "w") as f:
            json.dump(comparison_report, f, indent=2)
        print(f"comparison report saved to {report_path}")
        return

    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()

    device = get_processing_device(model, args.model)
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    with open(save_filepath, "w") as f:
        print("method\tactual_sparsity\tppl_test", file=f, flush=True)
        print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    if args.eval_zero_shot:
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()