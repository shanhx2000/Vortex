import os
import argparse
from transformers import AutoConfig, AutoModelForCausalLM
import json
from datetime import datetime
import logging

def ppl_test(args):
    logging.info(f"Running ppl_test for args={args}")
    dist_args = getattr(args, 'dist_args', {})
    sparse_fn_args = getattr(args, 'sparse_fn_args', {})
    model_args = {
        # "device": "cuda", #  device="auto", 
        "device": "auto",
        "histogram_path": os.path.join(args.teal_path, "histograms"), 
        # "grab_acts": False,
        "dist_args": dist_args,
        "sparse_fn_args": sparse_fn_args,
    }
    if "llama2-7b" in args.model_name.lower():
        model_args["device"] = "cuda"

    sparsity = getattr(args, "sparsity", 0)
    vq_uniform_sparse_fn_flag = getattr(args, "vq_uniform_sparse_fn", False)
    debug_flag = getattr(args, "debug_flag", False)
    use_vq = getattr(args, "use_vq", True)
    tasks = getattr(args, "eval_tasks", None)
    assert tasks is not None, f"eval_tasks must be provided. but only got {args}"


    from vortex.core.eval.lm_eval_m import lm_eval_m
    if use_vq:
        # vortex.core.model.custom_rvq
        from vortex.teal.utils.utils_model import get_tokenizer, get_sparse_model
        # from vortex.teal.teal_rvq.model_vq import LlamaSparseForCausalLM, LlamaSparseConfig
        # from vortex.teal.teal_rvq.model_vq import RvqLlamaSparseForCausalLM, RvqLlamaSparseConfig
        # AutoConfig.register("rvq_llama_sparse", RvqLlamaSparseConfig)
        # AutoModelForCausalLM.register(RvqLlamaSparseConfig, RvqLlamaSparseForCausalLM)
        # from vortex.teal.teal_rvq.model_vq import MistralSparseForCausalLM, MistralSparseConfig
        assert hasattr(args, 'model_args'), "model_args must be provided for vq mode."
        assert "vq_ctx" in args.model_args, "vq_ctx must be provided in model_args for vq mode."
        # extend model_args with args.model_args
        model_args.update(args.model_args)
        # model_args["vq_ctx"] = {
        #     "config_num_codebooks_glb": 2,
        #     "uniform_sparse_fn": vq_uniform_sparse_fn_flag,
        # }
    else:
        from vortex.teal.utils.utils_model import get_tokenizer, get_sparse_model
        # from vortex.teal.teal.model import LlamaSparseForCausalLM, LlamaSparseConfig
        # from vortex.teal.teal.model import MistralSparseForCausalLM, MistralSparseConfig

    # AutoConfig.register("llama_sparse", LlamaSparseConfig)
    # AutoConfig.register("mistral_sparse", MistralSparseConfig)
    # AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
    # AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)

    tokenizer = get_tokenizer(args.model_name)
    model = get_sparse_model(args.model_name, 
                            **model_args,
                            )
    
    
    # print("=" * 30)
    # from vortex.teal.utils.utils_model import utils_module_param_buffer_devices
    # for i, layer in enumerate(model.model.layers):
    #     print(f"Decoder Layer {i}:")
    #     for sub_name, sub_module in layer.named_children():
    #         devs = utils_module_param_buffer_devices(sub_module)
    #         print(f"  {sub_name} ({sub_module.__class__.__name__}): devices={devs}")
    #     for sub_name, sub_module in layer.self_attn.named_children():
    #         devs = utils_module_param_buffer_devices(sub_module)
    #         print(f"  self_attn.{sub_name} ({sub_module.__class__.__name__}): devices={devs}")
        
    # print("=" * 30)
    

    print("=" * 40)

    if sparsity > 0:
        print("Evaluating sparse PPL at sparsity level:", sparsity)

        if args.greedy_flag:
            print("Evaluating greedy PPL")
            lookup_suffix = ""
            if dist_args is not None:
                lookup_suffix = f"_v{dist_args.get('vec_length', 4)}{dist_args.get('use_abs', True)}"
            greedy_path = os.path.join(args.teal_path, "lookup"+lookup_suffix)
            
            if use_vq:
                model.load_greedy_sparsities(
                    greedy_path,
                    sparsity,
                    vq_ctx=model_args["vq_ctx"]
                )
            else:
                 model.load_greedy_sparsities(
                    greedy_path,
                    sparsity,
                )
        else:
            print("Evaluating uniform PPL")
            model.set_uniform_sparsity(sparsity)
    else:
        print("Evaluating dense PPL")

    
    results = lm_eval_m(
        model_name=args.model_name,
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        configs=None,
        debug=debug_flag,   # fix: use correct flag name
    )

    # print(f"PPL: {ppl}")
    print(results)
    print("=" * 40)
    
    return results

def log_result(args_dict, result, log_file="example_eval_results.jsonl"):
    """Append args + result + timestamp to a JSONL log file."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "args": args_dict,
        "result": result,
    }

    # append to file
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")