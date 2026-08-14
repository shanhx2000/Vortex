import os
import torch
import logging
def get_model_class_name(model_name):
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
        from vortex.core.model.custom_rvq import \
            CustomRvqLlamaConfig, CustomRvqLlamaForCausalLM, \
            CustomRvqMistralConfig, CustomRvqMistralForCausalLM
        AutoConfig.register("custom_rvq_llama", CustomRvqLlamaConfig)
        AutoModelForCausalLM.register(CustomRvqLlamaConfig, CustomRvqLlamaForCausalLM)
        AutoConfig.register("custom_rvq_mistral", CustomRvqMistralConfig)
        AutoModelForCausalLM.register(CustomRvqMistralConfig, CustomRvqMistralForCausalLM)

        # Fetch the model config
        config = AutoConfig.from_pretrained(model_name)
        # Get the model class name from the config
        model_class_name = config.architectures[0] if config.architectures else None
        return model_class_name
    except Exception as e:
        print(f"Error fetching model class name: {e}")
        assert False, f"Error fetching model class name: {e}"
        return None


def get_sparse_model(model_name, device, histogram_path, **kwargs):
    print(f"Getting sparse model for {model_name} on device {device} with histogram path {histogram_path} and additional args {kwargs}")
    logging.info(f"Getting sparse model for {model_name} on device {device} with histogram path {histogram_path} and additional args {kwargs}")
    from transformers import AutoConfig, AutoModelForCausalLM
    from vortex.teal.teal.model import LlamaSparseForCausalLM, MistralSparseForCausalLM, LlamaSparseConfig, MistralSparseConfig
    AutoConfig.register("llama_sparse", LlamaSparseConfig)
    AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
    AutoConfig.register("mistral_sparse", MistralSparseConfig)
    AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)

    if "vq_ctx" in kwargs:
        from vortex.teal.teal_rvq.model_vq import \
            RvqLlamaSparseForCausalLM, RvqLlamaSparseConfig, \
            RvqMistralSparseForCausalLM, RvqMistralSparseConfig
        AutoConfig.register("rvq_llama_sparse", RvqLlamaSparseConfig)
        AutoConfig.register("rvq_mistral_sparse", RvqMistralSparseConfig)
        AutoModelForCausalLM.register(RvqLlamaSparseConfig, RvqLlamaSparseForCausalLM)
        AutoModelForCausalLM.register(RvqMistralSparseConfig, RvqMistralSparseForCausalLM)

    class_name = get_model_class_name(model_name)

    # print("class_name=", class_name)
    assert class_name in ["LlamaForCausalLM", "MistralForCausalLM",
                          "LlamaSparseForCausalLM", "MistralSparseForCausalLM",
                          "CustomRvqLlamaForCausalLM",
                          "RvqLlamaSparseForCausalLM",

                          "CustomRvqMistralForCausalLM",
                          "RvqMistralSparseForCausalLM",
                        ], f"Model class name {class_name} not supported"

    if "vq_ctx" in kwargs:
        if "llama" in model_name.lower():
            SparseModel = RvqLlamaSparseForCausalLM
        elif "mistral" in model_name.lower():
            SparseModel = RvqMistralSparseForCausalLM
        else:
            raise ValueError("RVQ Sparse models are only supported for Llama and Mistral models.")
    else:
        if "llama" in model_name.lower():
            SparseModel = LlamaSparseForCausalLM
        elif "mistral" in model_name.lower():
            SparseModel = MistralSparseForCausalLM
        else:
            raise ValueError("Sparse models are only supported for Llama and Mistral models.")

    print("=" * 10)
    print("Using SparseModel=", SparseModel.__name__)
    print("=" * 10)


    return SparseModel.from_pretrained(model_name,
                                       torch_dtype=torch.float16,
                                       device_map=device,
                                       attn_implementation="flash_attention_2",
                                       histogram_path=histogram_path,
                                       **kwargs)

def get_tokenizer(tokenizer_name):

    print (f"Tokenizer name:", tokenizer_name)
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    from vortex.core.model.custom_rvq import \
        CustomRvqLlamaConfig, CustomRvqLlamaForCausalLM, \
        CustomRvqMistralConfig, CustomRvqMistralForCausalLM
    AutoConfig.register("custom_rvq_llama", CustomRvqLlamaConfig)
    AutoTokenizer.register(CustomRvqLlamaConfig, CustomRvqLlamaForCausalLM)
    AutoConfig.register("custom_rvq_mistral", CustomRvqMistralConfig)
    AutoTokenizer.register(CustomRvqMistralConfig, CustomRvqMistralForCausalLM)

    if "AQLM" in tokenizer_name and "Llama-2-7b" in tokenizer_name and "rvq" in tokenizer_name:
        quant_path = "meta-llama/Llama-2-7b-hf"
        tokenizer = AutoTokenizer.from_pretrained(
            quant_path, use_fast=True, trust_remote_code=True
        )
    elif "AQLM" in tokenizer_name and "Mistral-7B" in tokenizer_name and "rvq" in tokenizer_name:
        quant_path = "mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(
            quant_path, use_fast=True, trust_remote_code=True
        )
    elif "AQLM" in tokenizer_name and "Mistral-7B-Instruct-v0.2-AQLM-2Bit-2x8" in tokenizer_name:
        quant_path = "mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(
            quant_path, use_fast=True, trust_remote_code=True
        )
    elif "AQLM" in tokenizer_name and "Llama-2-13b" in tokenizer_name and "rvq" in tokenizer_name:
        quant_path = "meta-llama/Llama-2-13b-hf"
        tokenizer = AutoTokenizer.from_pretrained(
            quant_path, use_fast=True, trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, use_fast=True, trust_remote_code=True
        )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    return tokenizer


def get_module_device(module: torch.nn.Module) -> torch.device:
    """
    Returns the device of a given module.
    If the module has no parameters, tries to use a buffer.
    Falls back to CPU if both are missing.
    """
    try:
        # Try first parameter
        return next(module.parameters()).device
    except StopIteration:
        # No parameters found — try buffers
        try:
            return next(module.buffers()).device
        except StopIteration:
            # No buffers either — fallback
            return torch.device("cpu")



def get_layer_greedy_sparsities(layer_sparsities, results_dir, vq_ctx = None):
    import pandas as pd
    num_layers = len(layer_sparsities)
    if vq_ctx is not None and not vq_ctx["uniform_sparse_fn"]:
        assert vq_ctx is not None and 'config_num_codebooks_glb' in vq_ctx
        num_codebooks = vq_ctx['config_num_codebooks_glb']
        projs = []
        for i in range(num_codebooks):
            projs += [f'q{i}', f'k{i}', f'v{i}', f'o{i}', f'gate{i}', f'up{i}', f'down{i}']
    else:
        projs = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']

    sparsities = {proj: [0.0] * num_layers for proj in projs}

    for layer, target_sparsity in enumerate(layer_sparsities):
        file_path = os.path.join(results_dir, f'layer-{layer}', 'results.csv')
        df = pd.read_csv(file_path)

        # Find the row with the closest effective sparsity
        closest_row = df.iloc[(df['Effective Sparsity'] - target_sparsity).abs().argsort()[:1]]

        for proj in projs:
            sparsities[proj][layer] = closest_row[proj].values[0]

    return sparsities


def utils_module_param_buffer_devices(module):
    """Return set of devices for params and buffers of a module."""
    devs = set()
    for p in module.parameters(recurse=True):
        if p is None:
            continue
        devs.add(str(p.device))
    for b in module.buffers(recurse=True):
        if b is None:
            continue
        devs.add(str(b.device))
    return devs
