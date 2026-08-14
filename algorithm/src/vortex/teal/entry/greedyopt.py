
import os
import argparse
import torch
import csv
from copy import deepcopy

# from teal.model import LlamaSparseForCausalLM, LlamaSparseConfig, MistralSparseForCausalLM, MistralSparseConfig
from vortex.teal.teal.model import LlamaSparseForCausalLM, LlamaSparseConfig, MistralSparseForCausalLM, MistralSparseConfig
from vortex.teal.teal_rvq.model_vq import RvqLlamaSparseForCausalLM, RvqLlamaSparseConfig
from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register("rvq_llama_sparse", RvqLlamaSparseConfig)
AutoModelForCausalLM.register(RvqLlamaSparseConfig, RvqLlamaSparseForCausalLM)
AutoConfig.register("llama_sparse", LlamaSparseConfig)
AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
AutoConfig.register("mistral_sparse", MistralSparseConfig)
AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)
# from utils.utils import get_sparse_model, get_tokenizer
from vortex.teal.utils.utils_model import get_sparse_model, get_tokenizer

weight_dict = {
    "Llama-3-8B": {
        'q': 1, 'k': 1/4, 'v': 1/4, 'o': 1,
        'gate': 3.5, 'up': 3.5, 'down': 3.5
    },
    "Llama-3-70B": {
        'q': 1, 'k': 1/8, 'v': 1/8, 'o': 1,
        'gate': 3.5, 'up': 3.5, 'down': 3.5
    },
    # "Llama-2-7B": {
    #     'q': 1, 'k': 1/8, 'v': 1/8, 'o': 1,
    #     'gate': 2.6875, 'up': 2.6875, 'down': 2.6875
    # },
    "Llama-2-7B": {
        'q': 1, 'k': 1, 'v': 1, 'o': 1,
        'gate': 2.6875, 'up': 2.6875, 'down': 2.6875
    }, # 2026/01/13

    # "Llama-2-13B": {
    #     'q': 1, 'k': 1/8, 'v': 1/8, 'o': 1,
    #     'gate': 2.7, 'up': 2.7, 'down': 2.7
    # },
    "Llama-2-13B": {
        'q': 1, 'k': 1, 'v': 1, 'o': 1,
        'gate': 2.7, 'up': 2.7, 'down': 2.7
    }, # 2026/02/11

    "Llama-2-70B": {
        'q': 1, 'k': 1/8, 'v': 1/8, 'o': 1,
        'gate': 3.5, 'up': 3.5, 'down': 3.5
    },
    # "Mistral-7B": {
    #     'q': 1, 'k': 1/8, 'v': 1/8, 'o': 1,
    #     'gate': 3.5, 'up': 3.5, 'down': 3.5
    # },
    "Mistral-7B": {
        'q': 1, 'k': 1/4, 'v': 1/4, 'o': 1,
        'gate': 3.5, 'up': 3.5, 'down': 3.5
    }, # 2026/01/13
}

def set_layer_sparsities(layer, sparsities):
    if hasattr(layer.mlp, "vq_ctx") and not layer.mlp.vq_ctx["uniform_sparse_fn"]:
        vq_ctx = layer.mlp.vq_ctx
        num_codebooks = vq_ctx['config_num_codebooks_glb']
        for i in range(num_codebooks):
            layer.mlp.sparse_fns[f'gate{i}'].set_threshold(sparsities[f'gate{i}'])
            layer.mlp.sparse_fns[f'up{i}'].set_threshold(sparsities[f'up{i}'])
            layer.mlp.sparse_fns[f'down{i}'].set_threshold(sparsities[f'down{i}'])

            layer.self_attn.sparse_fns[f'q{i}'].set_threshold(sparsities[f'q{i}'])
            layer.self_attn.sparse_fns[f'k{i}'].set_threshold(sparsities[f'k{i}'])
            layer.self_attn.sparse_fns[f'v{i}'].set_threshold(sparsities[f'v{i}'])
            layer.self_attn.sparse_fns[f'o{i}'].set_threshold(sparsities[f'o{i}'])
    
    else:
        layer.mlp.sparse_fns['gate'].set_threshold(sparsities['gate'])
        layer.mlp.sparse_fns['up'].set_threshold(sparsities['up'])
        layer.mlp.sparse_fns['down'].set_threshold(sparsities['down'])

        layer.self_attn.sparse_fns['q'].set_threshold(sparsities['q'])
        layer.self_attn.sparse_fns['k'].set_threshold(sparsities['k'])
        layer.self_attn.sparse_fns['v'].set_threshold(sparsities['v'])
        layer.self_attn.sparse_fns['o'].set_threshold(sparsities['o'])

def f(sparsities, weights):
    total_weight = sum(weights.values())
    weighted_sparsity_sum = 0
    for projection_type, value in sparsities.items():
        if projection_type in weights:
            weighted_sparsity_sum += value * weights[projection_type]

    # print(weighted_sparsity_sum / total_weight)
    return weighted_sparsity_sum / total_weight

def layer_forward(layer, hidden_states):
    bsz, seq_len, _ = hidden_states.shape
    # layer = model.model.layers[layer_idx]

    attention_mask = None
    position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)
    past_key_value=None
    output_attentions = False
    use_cache = False
    cache_position=None

    return layer(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position)[0]


def calculate_activation_error(target_acts, new_activations, last_fraction=0.25):
    start_idx = int(new_activations.shape[1] * (1 - last_fraction))
    res = torch.norm(target_acts[:, start_idx:] - new_activations[:, start_idx:], dim=1).mean()
    # print(res)
    return res

def calculate_baseline_error(layer, input_acts, target_acts, baseline_sparsities, last_fraction):
    set_layer_sparsities(layer, baseline_sparsities)
    new_activations = layer_forward(layer, input_acts)
    return calculate_activation_error(target_acts, new_activations, last_fraction)




def process_layer(layer, model_type, layer_idx, target_sparsity, base_step_size, last_fraction, teal_path, lookup_suffix=""):
    weights = weight_dict[model_type]

    histogram_path = os.path.join(teal_path, 'histograms')
    activations_path = os.path.join(teal_path, 'activations', f'act_{layer_idx}.pt')
    output_path = os.path.join(teal_path, 'lookup'+lookup_suffix, f'layer-{layer_idx}', f'results.csv')

    device = "cuda"
    input_acts = torch.load(activations_path, map_location='cpu').to(device)
    print("input_acts", input_acts.shape)
    layer = layer.to(device)
    # check all modules of this layer is sent to device
    from vortex.utils.cuda_device import check_module_device
    check_module_device(layer, device)
    

    if hasattr(layer.mlp, "vq_ctx") and not layer.mlp.vq_ctx["uniform_sparse_fn"]:
        vq_ctx = layer.mlp.vq_ctx
        num_codebooks = vq_ctx['config_num_codebooks_glb']
        wd = {}
        for w in weights.keys():
            for i in range(num_codebooks):
                wd[f"{w}{i}"] = weights[w]
        weights = wd
        print(wd)
        # projs = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']
        projs = []
        for i in range(num_codebooks):
            projs += [f'q{i}', f'k{i}', f'v{i}', f'o{i}', f'gate{i}', f'up{i}', f'down{i}']        
    else:
        num_codebooks = 1
        projs = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']
    sparsities = {proj: 0.0 for proj in projs}
    set_layer_sparsities(layer, sparsities)
    target_acts = layer_forward(layer, input_acts)


    step_sizes = {proj: base_step_size * (1 / weights[proj]) for proj in projs}
    
    sparsities = {proj: 0.0 for proj in projs}
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Effective Sparsity'] + ['Activation Error', 'Baseline Error'] + [f"{proj}" for proj in projs])
        
        while f(sparsities, weights) < target_sparsity:
            best_error = float('inf')
            best_proj = None
            
            for proj in projs:
                temp_sparsities = deepcopy(sparsities)

                if temp_sparsities[proj] >= 1:
                    continue

                temp_sparsities[proj] += step_sizes[proj]
                # if temp_sparsities[proj] > 1:
                #     temp_sparsities[proj] = 1.0
                
                set_layer_sparsities(layer, temp_sparsities)
                new_activations = layer_forward(layer, input_acts)
                error = calculate_activation_error(target_acts, new_activations, last_fraction)
                
                if error < best_error:
                    best_error = error
                    best_proj = proj
            
            assert best_proj is not None, f"No projection found to increase sparsity. This should not happen. sparsities={sparsities}. projs={projs}. weights={weights}."
            assert best_proj in projs, f"Best projection {best_proj} not in projection list {projs} or {sparsities.keys()}."
            sparsities[best_proj] += step_sizes[best_proj]
            set_layer_sparsities(layer, sparsities)
            
            effective_sparsity = f(sparsities, weights)
            baseline_sparsities = {proj: effective_sparsity for proj in projs}
            baseline_error = calculate_baseline_error(layer, input_acts, target_acts, baseline_sparsities, last_fraction)
            
            row = [effective_sparsity] + [best_error.item(), baseline_error.item()] + [sparsities[proj] for proj in projs]
            csvwriter.writerow(row)
            csvfile.flush()
            
            # print(f"Updated: Effective Sparsity: {effective_sparsity:.4f}, Activation Error: {best_error:.4f}, Baseline Error: {baseline_error:.4f}")
    
    layer.to('cpu')
    return sparsities


def greedyopt(args):
    # import argparse
    # if __name__ == "__main__":
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument("--model_name", type=str, required=True)
    #     parser.add_argument("--model_type", type=str, required=True)
    #     parser.add_argument("--teal_path", type=str, required=True)
    #     parser.add_argument("--target_sparsity", type=float, default=0.9, help="Target effective sparsity")
    #     parser.add_argument("--base_step_size", type=float, default=0.05, help="Base step size for sparsity updates")
    #     parser.add_argument("--last_fraction", type=float, default=0.25, help="Fraction of sequence to use for error calculation")

    #     args = parser.parse_args()
    
    dist_args = getattr(args, "dist_args", None)
    sparse_fn_args = getattr(args, "sparse_fn_args", None)
    # model_args = getattr(args, "model_args", None)
    model_args = getattr(args, "model_args", {}) or {}

    histogram_path = os.path.join(args.teal_path, 'histograms')
    lookup_suffix = ""
    if dist_args is not None:
        # lookup_suffix = f"_v{dist_args.get('vec_length', 4)}{dist_args.get('use_abs', True)}"
        vec_length = dist_args.get("vec_length", 4)
        use_abs = dist_args.get("use_abs", True)
        phi_func = dist_args.get("phi_func", "l1-norm")
        lookup_suffix = f"_v{vec_length}{use_abs}"
        if phi_func != "l1-norm":
            phi_map = {
                "l0-norm": "l0",
                "l2-norm": "l2",
                "l_inf-norm": "linf",
            }
            lookup_suffix += f"_phi{phi_map[phi_func]}"

    # from utils.utils import get_model_class_name
    from vortex.teal.utils.utils_model import get_model_class_name


    class_name = get_model_class_name(args.model_name)
    assert class_name in [
        'LlamaSparseForCausalLM', 
        'MistralSparseForCausalLM', 
        'LlamaForCausalLM', 
        'MistralForCausalLM',
        'RvqLlamaSparseForCausalLM',
        'CustomRvqLlamaForCausalLM',
        'CustomRvqMistralForCausalLM',
        ], f"Model {args.model_name} not supported"

    model = get_sparse_model(args.model_name, device='cpu', histogram_path=histogram_path,
                             dist_args=dist_args, 
                             sparse_fn_args=sparse_fn_args,
                             **model_args
                             )

    os.makedirs(os.path.join(args.teal_path, 'lookup'+lookup_suffix), exist_ok=True)

    num_layers = len(model.model.layers)
    
    # layer_idx = 0

    # layer = model.model.layers[0]
    # process_layer(layer, args.model_type, layer_idx, args.target_sparsity, args.base_step_size, args.last_fraction, args.teal_path)

    for layer_idx in range(num_layers):
        print(f"Processing layer {layer_idx}")
        layer = model.model.layers[layer_idx]
        process_layer(layer, 
                      args.model_type, 
                      layer_idx, 
                      args.target_sparsity, 
                      args.base_step_size, 
                      args.last_fraction, 
                      args.teal_path,
                      lookup_suffix=lookup_suffix,
                    )

