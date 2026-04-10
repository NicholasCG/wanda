# Import necessary modules
import time
import math
import torch
import torch.nn as nn

# Import get_loaders function from data module within the same directory
from .data import get_loaders 

from collections import defaultdict
import fnmatch


# Nicholas Gray: resolves the embedding device for multi-GPU models to avoid cross-device tensor errors.
def _resolve_eval_device(model, device):
    if hasattr(model, "hf_device_map") and "model.embed_tokens" in model.hf_device_map:
        return model.hf_device_map["model.embed_tokens"]
    return device


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = "wikitext2"

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
    )

    device = _resolve_eval_device(model, device)

    # Nicholas Gray: disables KV cache and uses inference_mode to reduce activation memory during evaluation.
    use_cache = model.config.use_cache
    model.config.use_cache = False
    with torch.inference_mode():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    model.config.use_cache = use_cache
    return ppl_test 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
    # Get input IDs
    # testenc = testenc.input_ids

    # Calculate number of samples
    # nsamples = testenc.numel() // model.seqlen
    nsamples = len(trainloader)

    # Nicholas Gray: accumulates NLL as a running scalar and uses the model's built-in loss to avoid materializing logit tensors.
    total_nll = 0.0
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Compute token prediction loss without materializing long-lived logits tensors.
        loss = model(inputs, labels=inputs, use_cache=False).loss

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float().item() * model.seqlen * (j-i)

        total_nll += neg_log_likelihood

        del inputs, loss
        if torch.cuda.is_available() and (i // bs) % 50 == 0:
            torch.cuda.empty_cache()

    # Compute perplexity
    ppl = math.exp(total_nll / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # Nicholas Gray: accumulates NLL as a running scalar and uses the model's built-in loss to avoid materializing logit tensors.
    total_nll = 0.0
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Compute token prediction loss without materializing long-lived logits tensors.
        loss = model(inputs, labels=inputs, use_cache=False).loss

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float().item() * model.seqlen * (j-i)

        total_nll += neg_log_likelihood

        del inputs, loss
        if torch.cuda.is_available() and (i // bs) % 50 == 0:
            torch.cuda.empty_cache()

    # Compute perplexity
    ppl = math.exp(total_nll / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl


def eval_zero_shot(model_name, model, tokenizer, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], 
        num_fewshot=0, use_accelerate=False, add_special_tokens=False):
    from lm_eval import tasks, evaluator 
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    task_names = pattern_match(task_list, tasks.ALL_TASKS)
    model_args = f"pretrained={model_name},cache_dir=./llm_weights"
    limit = None 
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained={model_name},cache_dir=./llm_weights,use_accelerate=True"
    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=None,
        no_cache=True,
        limit=limit,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        pretrained_model=model,
        tokenizer=tokenizer, 
        add_special_tokens=add_special_tokens
    )

    return results 