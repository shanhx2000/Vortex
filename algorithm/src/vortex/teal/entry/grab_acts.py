import argparse
import torch
import transformers
from vortex.teal.utils.data import get_dataset
from tqdm import tqdm
import gc
import os
import logging

def grab_activations(model_mode, args):
    dist_args = getattr(args, 'dist_args', {})
    sparse_fn_args = getattr(args, 'sparse_fn_args', {})
    model_args = {
        "device": "auto",
        "histogram_path": os.path.join(args.teal_path, "histograms"),
        "grab_acts": True,
        "dist_args": dist_args,
        "sparse_fn_args": sparse_fn_args,
    }
    if "llama2-7b" in args.model_name.lower():
        model_args["device"] = "cuda"

    if model_mode == 'vq':
        from vortex.teal.utils.utils_model import get_tokenizer, get_sparse_model
        from transformers import AutoConfig, AutoModelForCausalLM
        assert hasattr(args, 'model_args'), "model_args must be provided for vq mode."
        assert "vq_ctx" in args.model_args, "vq_ctx must be provided in model_args for vq mode."
        from vortex.teal.teal_rvq.model_vq import RvqLlamaSparseForCausalLM, RvqLlamaSparseConfig
        AutoConfig.register("rvq_llama_sparse", RvqLlamaSparseConfig)
        AutoModelForCausalLM.register(RvqLlamaSparseConfig, RvqLlamaSparseForCausalLM)
        # extend model_args with args.model_args
        model_args.update(args.model_args)
    elif model_mode == 'regular':
        from vortex.teal.utils.utils_model import get_tokenizer, get_sparse_model
        from vortex.teal.teal.model import LlamaSparseForCausalLM, LlamaSparseConfig
        from vortex.teal.teal.model import MistralSparseForCausalLM, MistralSparseConfig
        from transformers import AutoConfig, AutoModelForCausalLM
        AutoConfig.register("llama_sparse", LlamaSparseConfig)
        AutoConfig.register("mistral_sparse", MistralSparseConfig)
        AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
        AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)
    else:
        raise NotImplementedError(f"Mode {model_mode} is not supported.")

    logging.info(f"Loading tokenizer and model of {args.model_name} for {model_mode} mode.")

    tokenizer = get_tokenizer(args.model_name)
    print("=" * 5)
    print("Creating sparse model with ", model_args)
    print("=" * 5)
    model = get_sparse_model(args.model_name,
                            **model_args,
                            )
    logging.info("Tokenizer and model loaded successfully.")

    bsz = getattr(args, 'batch_size', 10)
    seq_len = getattr(args, 'seq_len', 2048)
    logging.info(f"Using batch size: {bsz}, sequence length: {seq_len}")

    assert args.dataset == "wikitext", "Only wikitext dataset is supported currently."
    dataset_size = getattr(args, 'dataset_size', 500)
    dataset = get_dataset(
        "Salesforce/wikitext",
        subset="wikitext-2-raw-v1",
        split="train",
        size=dataset_size
    )
    logging.warning(f"using wikitext dataset of size {dataset_size}")

    text = ""
    for sample in tqdm(dataset):
        text += sample["text"] + "\n\n"

    print(len(text))
    encodings = tokenizer(text, truncation=True, return_tensors="pt", max_length=seq_len, return_overflowing_tokens=True, padding="max_length")
    input_ids = encodings.input_ids[:bsz,:].to(device="cuda")
    print(input_ids.shape)
    assert input_ids.shape[0] <= bsz
    if input_ids.shape[0] < bsz:
        logging.warning(f"Reducing batch size from {bsz} to {input_ids.shape[0]} due to insufficient data.")
        bsz = input_ids.shape[0] # reduce bsz if no enough data

    hidden_states = model.model.embed_tokens(input_ids)
    attention_mask = None
    position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)
    past_key_value=None
    output_attentions = False
    use_cache = False
    cache_position=None
    act_path = os.path.join(args.teal_path, "activations")
    os.makedirs(act_path, exist_ok=True)

    logging.info(f"Saving activations to {act_path}")
    for i in tqdm(range(len(model.model.layers))):
        # for greedyopt
        torch.save(hidden_states, os.path.join(act_path, f"act_{i}.pt"))
        layer = model.model.layers[i]
        obj = layer.self_attn.q_proj
        if obj.__class__.__name__ == "QuantizedLinear" and str(obj.__class__.__module__) == "aqlm.inference":
            hidden_states_device = layer.self_attn.q_proj.codes.device
        else:
            hidden_states_device = layer.self_attn.q_proj.weight.data.device

        hidden_states = hidden_states.to(hidden_states_device)
        hidden_states = layer(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position)[0]

        layer.mlp.activation_module.find_histogram()
        layer.self_attn.activation_module.find_histogram()
        layer.mlp.activation_module.save_histogram()
        layer.self_attn.activation_module.save_histogram()

        del layer.mlp.activation_module.activations
        del layer.self_attn.activation_module.activations

        model.model.layers[i] = None

        gc.collect()
        torch.cuda.empty_cache()
    logging.info("Finished grabbing activations.")
