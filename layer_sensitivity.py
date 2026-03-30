import argparse
import csv
import os
import time
import inspect
from importlib.metadata import version

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.eval import eval_ppl
from lib.prune import find_layers, prepare_calibration_input
from lib.data import get_loaders
from lib.layerwrapper import WrappedGPT
from lib.sparsegpt import SparseGPT


print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def get_llm(model_name, cache_dir='llm_weights'):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map='auto',
    )
    model.seqlen = model.config.max_position_embeddings
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Layer sensitivity analysis for locuslab/wanda')
    parser.add_argument('--model', type=str, required=True, help='HF model name/path')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--sparsity_ratio', type=float, default=0.5)
    parser.add_argument('--sparsity_type', type=str, default='unstructured', choices=['unstructured', '4:8', '2:4'])
    parser.add_argument('--prune_method', type=str, required=True, choices=['magnitude', 'wanda', 'sparsegpt'])
    parser.add_argument('--cache_dir', type=str, default='llm_weights')
    parser.add_argument('--use_variant', action='store_true', help='Use Wanda appendix variant for unstructured pruning')
    parser.add_argument('--save', type=str, default='layer_sensitivity_results', help='Output directory')
    parser.add_argument('--layers', type=str, default=None,
                        help='Comma list and/or ranges, e.g. 0,1,5-8 . Default: all layers')
    parser.add_argument('--skip_baseline', action='store_true', help='Skip dense baseline evaluation')
    return parser.parse_args()


def resolve_nm(args):
    prune_n, prune_m = 0, 0
    if args.sparsity_type != 'unstructured':
        if abs(args.sparsity_ratio - 0.5) > 1e-8:
            raise ValueError('sparsity_ratio must be 0.5 for structured N:M sparsity to match wanda repo behavior')
        prune_n, prune_m = map(int, args.sparsity_type.split(':'))
    return prune_n, prune_m


def parse_layers_arg(layers_arg, num_layers):
    if layers_arg is None:
        return list(range(num_layers))
    result = []
    for part in layers_arg.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-', 1)
            start, end = int(start), int(end)
            if end < start:
                raise ValueError(f'Invalid layer range: {part}')
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    result = sorted(set(result))
    for idx in result:
        if idx < 0 or idx >= num_layers:
            raise ValueError(f'Layer index {idx} out of range [0, {num_layers - 1}]')
    return result


def get_main_device(model, model_name):
    device = torch.device('cuda:0')
    if '30b' in model_name or '65b' in model_name or '70b' in model_name:
        device = model.hf_device_map['lm_head']
    return device


def maybe_move_tensor(x, dev):
    if x is None:
        return None
    return x.to(dev)


def move_calib_to_layer_device(model, layer_idx, inps, outs, attention_mask, position_ids):
    dev = torch.device('cuda:0')
    if hasattr(model, 'hf_device_map') and f'model.layers.{layer_idx}' in model.hf_device_map:
        dev = model.hf_device_map[f'model.layers.{layer_idx}']
    inps = inps.to(dev)
    outs = outs.to(dev)
    attention_mask = maybe_move_tensor(attention_mask, dev)
    position_ids = maybe_move_tensor(position_ids, dev)
    return dev, inps, outs, attention_mask, position_ids


def get_layer_inputs(model, tokenizer, args, device, target_layer_idx):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        dataloader, _ = get_loaders(
            'c4',
            nsamples=args.nsamples,
            seed=args.seed,
            seqlen=model.seqlen,
            tokenizer=tokenizer,
        )

        with torch.no_grad():
            sig = inspect.signature(prepare_calibration_input)
            if len(sig.parameters) >= 4:
                inps, outs, attention_mask, position_ids = prepare_calibration_input(
                    model, dataloader, device, args.nsamples
                )
            else:
                inps, outs, attention_mask, position_ids = prepare_calibration_input(
                    model, dataloader, device
                )

        layers = model.model.layers
        for i in range(target_layer_idx):
            dev_i, inps, outs, attention_mask, position_ids = move_calib_to_layer_device(
                model, i, inps, outs, attention_mask, position_ids
            )
            layer = layers[i]
            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
            inps, outs = outs, inps
            if isinstance(dev_i, str) or (hasattr(dev_i, 'type') and str(dev_i).startswith('cuda')):
                torch.cuda.empty_cache()

        dev_t, inps, outs, attention_mask, position_ids = move_calib_to_layer_device(
            model, target_layer_idx, inps, outs, attention_mask, position_ids
        )
        return dev_t, inps, outs, attention_mask, position_ids
    finally:
        model.config.use_cache = use_cache


def move_wrapped_layer_state_to_device(wrapped_layers, subset):
    for name in wrapped_layers:
        wdev = subset[name].weight.device
        if hasattr(wrapped_layers[name], 'scaler_row'):
            sr = wrapped_layers[name].scaler_row
            target_shape = (subset[name].weight.data.shape[1],)
            if sr.device.type == 'meta':
                wrapped_layers[name].scaler_row = torch.zeros(
                    target_shape,
                    dtype=torch.float32,
                    device=wdev,
                )
            else:
                wrapped_layers[name].scaler_row = sr.to(wdev)


@torch.no_grad()
def prune_single_layer_magnitude(model, target_layer_idx, sparsity_ratio, prune_n=0, prune_m=0):
    layer = model.model.layers[target_layer_idx]
    subset = find_layers(layer)
    for name in subset:
        print(f'pruning layer {target_layer_idx} name {name}')
        W = subset[name].weight.data
        W_metric = torch.abs(W)
        if prune_n != 0:
            W_mask = torch.zeros_like(W, dtype=torch.bool)
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:, ii:(ii + prune_m)].float()
                    W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
        else:
            thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel() * sparsity_ratio)].cpu()
            W_mask = W_metric <= thresh
        W[W_mask] = 0


@torch.no_grad()
def prune_single_layer_wanda(model, tokenizer, args, device, target_layer_idx, prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    handles = []
    try:
        print('loading calibration data')
        dev, inps, outs, attention_mask, position_ids = get_layer_inputs(
            model, tokenizer, args, device, target_layer_idx
        )
        print('dataset loading complete')

        layer = model.model.layers[target_layer_idx]
        subset = find_layers(layer)
        wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}
        move_wrapped_layer_state_to_device(wrapped_layers, subset)

        def add_batch(name):
            def tmp(_, inp, out):
                if hasattr(wrapped_layers[name], 'scaler_row'):
                    if wrapped_layers[name].scaler_row.device != inp[0].device:
                        if wrapped_layers[name].scaler_row.device.type == 'meta':
                            wrapped_layers[name].scaler_row = torch.zeros(
                                (subset[name].weight.data.shape[1],),
                                dtype=torch.float32,
                                device=inp[0].device,
                            )
                        else:
                            wrapped_layers[name].scaler_row = wrapped_layers[name].scaler_row.to(inp[0].device)
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]

        for h in handles:
            h.remove()
        handles = []

        for name in subset:
            print(f'pruning layer {target_layer_idx} name {name}')
            if wrapped_layers[name].scaler_row.device != subset[name].weight.device:
                wrapped_layers[name].scaler_row = wrapped_layers[name].scaler_row.to(subset[name].weight.device)

            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)

            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                if args.use_variant:
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)
                    alpha = 0.4
                    alpha_hist = [0.0, 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha
                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f'alpha found {alpha} sparsity {cur_sparsity:.6f}')
                else:
                    indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
            subset[name].weight.data[W_mask] = 0
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        model.config.use_cache = use_cache
        torch.cuda.empty_cache()


@torch.no_grad()
def prune_single_layer_sparsegpt(model, tokenizer, args, device, target_layer_idx, prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    handles = []
    try:
        print('Starting ...')
        dev, inps, outs, attention_mask, position_ids = get_layer_inputs(
            model, tokenizer, args, device, target_layer_idx
        )
        print('Ready.')

        layer = model.model.layers[target_layer_idx]
        subset = find_layers(layer)
        gpts = {name: SparseGPT(subset[name]) for name in subset}

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]

        for h in handles:
            h.remove()
        handles = []

        for name in gpts:
            print(target_layer_idx, name)
            print('Pruning ...')
            gpts[name].fasterprune(
                args.sparsity_ratio,
                prune_n=prune_n,
                prune_m=prune_m,
                percdamp=0.01,
                blocksize=128,
            )
            gpts[name].free()
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        model.config.use_cache = use_cache
        torch.cuda.empty_cache()


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1)
    W_mask = W_metric <= thres
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def target_layer_sparsity(model, target_layer_idx):
    layer = model.model.layers[target_layer_idx]
    subset = find_layers(layer)
    count = 0
    total = 0
    for name in subset:
        W = subset[name].weight.data
        count += (W == 0).sum().item()
        total += W.numel()
    return count / total


def run_one_layer(args, layer_idx, prune_n, prune_m):
    print('=' * 80)
    print(f'Running layer sensitivity for layer {layer_idx}')
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    device = get_main_device(model, args.model)
    print('use device ', device)

    start = time.time()
    if args.prune_method == 'magnitude':
        prune_single_layer_magnitude(model, layer_idx, args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == 'wanda':
        prune_single_layer_wanda(model, tokenizer, args, device, layer_idx, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == 'sparsegpt':
        prune_single_layer_sparsegpt(model, tokenizer, args, device, layer_idx, prune_n=prune_n, prune_m=prune_m)
    else:
        raise ValueError(f'Unsupported prune_method: {args.prune_method}')
    prune_time_s = time.time() - start

    sparsity = target_layer_sparsity(model, layer_idx)
    ppl = eval_ppl(args, model, tokenizer, device)
    del tokenizer
    del model
    torch.cuda.empty_cache()
    return {
        'layer': layer_idx,
        'target_layer_sparsity': sparsity,
        'ppl': float(ppl),
        'prune_time_s': prune_time_s,
    }


def run_baseline(args):
    print('=' * 80)
    print('Running dense baseline')
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    device = get_main_device(model, args.model)
    print('use device ', device)
    ppl = eval_ppl(args, model, tokenizer, device)
    del tokenizer
    del model
    torch.cuda.empty_cache()
    return float(ppl)


def write_results_csv(csv_path, results):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'layer', 'prune_method', 'sparsity_type', 'sparsity_ratio',
                'target_layer_sparsity', 'baseline_ppl', 'ppl', 'delta_ppl', 'prune_time_s'
            ],
        )
        writer.writeheader()
        writer.writerows(results)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    prune_n, prune_m = resolve_nm(args)

    probe_model = get_llm(args.model, args.cache_dir)
    num_layers = len(probe_model.model.layers)
    del probe_model
    torch.cuda.empty_cache()

    target_layers = parse_layers_arg(args.layers, num_layers)
    os.makedirs(args.save, exist_ok=True)

    csv_name = f'layer_sensitivity_{args.prune_method}_{args.sparsity_type.replace(":", "-")}.csv'
    csv_path = os.path.join(args.save, csv_name)

    baseline_ppl = None
    if not args.skip_baseline:
        baseline_ppl = run_baseline(args)
        print(f'dense baseline ppl {baseline_ppl}')

    results = []
    for layer_idx in target_layers:
        row = run_one_layer(args, layer_idx, prune_n, prune_m)
        row['baseline_ppl'] = baseline_ppl if baseline_ppl is not None else ''
        row['delta_ppl'] = '' if baseline_ppl is None else row['ppl'] - baseline_ppl
        row['prune_method'] = args.prune_method
        row['sparsity_type'] = args.sparsity_type
        row['sparsity_ratio'] = args.sparsity_ratio
        results.append(row)
        print(row)

        # checkpoint partial progress after every completed layer
        write_results_csv(csv_path, results)

    print('=' * 80)
    print(f'Saved results to {csv_path}')


if __name__ == '__main__':
    main()