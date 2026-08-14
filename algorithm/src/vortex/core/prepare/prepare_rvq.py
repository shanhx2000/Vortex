from vortex.core.model.custom_rvq import CustomRvqLinear, \
        CustomRvqLlamaForCausalLM, CustomRvqLlamaConfig, \
        CustomRvqMistralForCausalLM, CustomRvqMistralConfig
from transformers import AutoModelForCausalLM, LlamaForCausalLM, MistralForCausalLM
import torch
from aqlm.utils import _dequantize_weight, unpack_int_data
from transformers.modeling_utils import no_init_weights
import logging

def _get_dequantized_weights_by_codebooks(layer):
    """
    Dequant weight of each codebook separately. scales not multiplied.
    """

    num_codebooks = layer.num_codebooks
    out_features = layer.out_features
    in_features = layer.in_features
    in_group_size = layer.in_group_size
    out_group_size = layer.out_group_size
    num_in_groups = in_features // in_group_size
    num_out_groups = out_features // out_group_size

    dequantized_weights = []
    for i in range(num_codebooks):
        codes_i = layer.codes[..., i:i+1]  # [num_out_groups, num_in_groups, 1]
        assert codes_i.shape == (
            num_out_groups,
            num_in_groups,
            1,
        ), f"codes_i.shape wrong. {codes_i.shape} != ({num_out_groups}, {num_in_groups}, 1)"
        unpacked_code = unpack_int_data(codes_i, layer.codebooks.shape[1].bit_length() - 1)
        codebook_i = layer.codebooks[i]
        assert isinstance(unpacked_code, torch.Tensor), "unpacked_code is not a tensor"
        assert isinstance(codebook_i, torch.Tensor), (
            "codebook_i is not a tensor"
            + f"{codebook_i.dtype}, {type(codebook_i)}"
        )
        assert codebook_i.shape == (
            layer.codebook_size,
            out_group_size,
            in_group_size,
        ), f"codebook_i.shape wrong. {codebook_i.shape} != ({layer.codebook_size}, {out_group_size}, {in_group_size})"

        codebook_i = codebook_i.unsqueeze(0)  # [1, codebook_size, out_g, in_g]
        # assert isinstance(layer.scales, torch.Tensor), "layer.scales is not a tensor"

        weight_i = _dequantize_weight(
            codes=unpacked_code,
            codebooks=codebook_i,
        )
        assert weight_i.shape == (
            out_features,
            in_features,
        ), f"weight_i.shape wrong. Got {weight_i.shape} but expect ({out_features}, {in_features})"

        dequantized_weights.append(weight_i)

    dequantized_weights = torch.stack(dequantized_weights, dim=0)
    
    assert dequantized_weights.shape == (num_codebooks, out_features, in_features), f"dequantized_weights.shape wrong. {dequantized_weights.shape} != ({num_codebooks}, {out_features}, {in_features})"
    return dequantized_weights


def _manually_copy_state(model_src, model_tgt, skip_keywords=None):
    if skip_keywords is None:
        skip_keywords = []

    # get named parameters (and buffers for completeness)
    src_state = dict(model_src.named_parameters())
    tgt_state = dict(model_tgt.named_parameters())

    # also copy buffers (like running_mean, running_var in BatchNorm)
    src_buffers = dict(model_src.named_buffers())
    tgt_buffers = dict(model_tgt.named_buffers())

    # merge parameters + buffers into one dict
    src_state.update(src_buffers)
    tgt_state.update(tgt_buffers)

    for name, src_param in src_state.items():
        if any(k in name for k in skip_keywords):
            print(f"Skipping {name}")
            continue

        if name not in tgt_state:
            print(f"Not found in target: {name}")
            continue

        tgt_param = tgt_state[name]
        if tgt_param.shape != src_param.shape:
            print(f"Shape mismatch for {name}: {src_param.shape} vs {tgt_param.shape}")
            continue

        with torch.no_grad():
            tgt_param.copy_(src_param)
        print(f"Copied {name}")

    print("✅ Manual parameter copy complete.")

def _update_rvq_model_from_aqlm(rvq_model, aqlm_model):
    """
    Updates rvq_model weights using w4_model's parameters.

    Specifically maps QuantizedLinear → CustomRvqLinear by unpacking quantized weights
    and broadcasting scales.
    """
    
    state_dict_1 = aqlm_model.state_dict()
    state_dict_2 = rvq_model.state_dict()
    for name, param in state_dict_1.items():
        #     continue
        if name in state_dict_2 and state_dict_2[name].shape == param.shape:
            state_dict_2[name].copy_(param)
            print(f"Loaded {name}")
        else:
            print(f"Shape mismatch or not found: {name}")
    # rvq_model.load_state_dict(state_dict_2)
    _manually_copy_state(aqlm_model, rvq_model)

    from tqdm import tqdm
    for name, aqlm_module in tqdm(aqlm_model.named_modules(), desc="update_rvq_model_from_aqlm"):
        # make sure same module exists in rvq_model
        rvq_module = dict(rvq_model.named_modules()).get(name, None)
        assert rvq_module is not None, f"Module {name} missing in rvq_model"

        if aqlm_module.__class__.__name__ == "QuantizedLinear":
            assert isinstance(rvq_module, CustomRvqLinear), f"Expected CustomRvqLinear for {name}, got {rvq_module.__class__.__name__}"
            # assert rvq_module.__class__.__name__ == "CustomRvqLinear", \
            #     f"Expected CustomRvqLinear for {name}, got {rvq_module.__class__.__name__}"
            print(f"replacing {name} with aqlm layer")

            # unpack quantized weights into multiple codebooks
            w_q = _get_dequantized_weights_by_codebooks(aqlm_module)
            new_weight = w_q
            
            # # expand weight_scale to match
            s = aqlm_module.scales[:,:,0,0]
            # (num_out_groups, num_in_groups, 1, 1) => (num_out_groups, num_in_groups) => (1, num_out_groups, num_in_groups)
            new_scales = s.unsqueeze(0)

            with torch.no_grad():
                if rvq_module.wq.shape != new_weight.shape:
                    # Re-register the buffer with the new shape
                    del rvq_module._buffers["wq"]
                    rvq_module.register_buffer("wq", new_weight.clone())
                else:
                    rvq_module.wq.copy_(new_weight)
                # rvq_module.wq.to(torch.int8)

                if rvq_module.scales.shape != new_scales.shape:
                    del rvq_module._buffers["scales"]
                    rvq_module.register_buffer("scales", new_scales.clone())
                else:
                    rvq_module.scales.copy_(new_scales)
            
            if aqlm_module.bias is not None:
                rvq_module.bias.data.copy_(aqlm_module.bias.data)

        elif isinstance(aqlm_module, torch.nn.Linear) and isinstance(rvq_module, torch.nn.Linear):
            # direct copy if both are standard linears
            rvq_module.weight.data.copy_(aqlm_module.weight.data)
            if aqlm_module.bias is not None:
                rvq_module.bias.data.copy_(aqlm_module.bias.data)

    print("✅ rvq_model successfully updated from aqlm_model.")
    return rvq_model

def generate_rvq_model_from_aqlm(configs: dict):
    # Example configs values:
    #   base_model_path  = "meta-llama/Llama-2-7b-hf"
    #   quant_model_path = "ISTA-DASLab/Llama-2-7b-AQLM-PV-2Bit-2x8-hf"
    #   save_path        = <repo>/ckpts/rvq_models   (see vortex.paths)
    base_model_path = configs["base_model_path"]
    quant_model_path = configs["quant_model_path"]
    save_path = configs["save_path"]
    # check save_path exists
    import os
    assert os.path.exists(save_path), f"save_path {save_path} does not exist."
    
    base_model_name = quant_model_path.split("/")[-1]
    rvq_model_save_path = f"{save_path}/{base_model_name}-rvq"
    os.makedirs(rvq_model_save_path, exist_ok=True)
    
    logging.info(f"Preparing RVQ model and saving to {rvq_model_save_path}")
    
    if "llama" in base_model_name.lower():
        aqlm_model = LlamaForCausalLM.from_pretrained(quant_model_path)
        rvq_model_1_config = CustomRvqLlamaConfig.from_pretrained(base_model_path)
        with no_init_weights():
            rvq_model_1 = CustomRvqLlamaForCausalLM(config=rvq_model_1_config)
    elif "mistral" in base_model_name.lower():
        aqlm_model = MistralForCausalLM.from_pretrained(quant_model_path)
        rvq_model_1_config = CustomRvqMistralConfig.from_pretrained(base_model_path)
        with no_init_weights():
            rvq_model_1 = CustomRvqMistralForCausalLM(config=rvq_model_1_config)

    else:
        raise ValueError("Only Llama and Mistral models are supported for RVQ conversion.")
    logging.info("AQLM model loaded.")
    
    # # create and update
    rvq_model_1 = _update_rvq_model_from_aqlm(rvq_model_1, aqlm_model)
    logging.info("RVQ model created and updated from AQLM model.")
    
    
    # # print all state dict keys and shapes

    # # load and eval
    from transformers import AutoModelForCausalLM, AutoConfig
    # AutoConfig.register("custom_rvq_llama", CustomRvqConfig)
    # AutoModelForCausalLM.register(CustomRvqConfig, CustomRvqLlamaForCausalLM)
    AutoConfig.register("custom_rvq_llama", CustomRvqLlamaConfig)
    AutoModelForCausalLM.register(CustomRvqLlamaConfig, CustomRvqLlamaForCausalLM)
    AutoConfig.register("custom_rvq_mistral", CustomRvqMistralConfig)
    AutoModelForCausalLM.register(CustomRvqMistralConfig, CustomRvqMistralForCausalLM)
    
    rvq_model_1.save_pretrained(rvq_model_save_path)
    rvq_model_1.config.save_pretrained(rvq_model_save_path)
    logging.info(f"RVQ model saved to {rvq_model_save_path}")

    import time
    start_time = time.time()
    if "llama" in base_model_name.lower():
        rvq_model_reload = CustomRvqLlamaForCausalLM.from_pretrained(rvq_model_save_path, ignore_mismatched_sizes=True)
    elif "mistral" in base_model_name.lower():
        rvq_model_reload = CustomRvqMistralForCausalLM.from_pretrained(rvq_model_save_path, ignore_mismatched_sizes=True)
    else:
        raise ValueError("Only Llama and Mistral models are supported for RVQ conversion.")
    time_elapsed = time.time() - start_time
    logging.info(f"RVQ model reloaded from saved checkpoint in {time_elapsed:.2f} seconds.")
