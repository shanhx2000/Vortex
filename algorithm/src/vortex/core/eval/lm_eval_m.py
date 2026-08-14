def lm_eval_m(model_name="", model=None, tokenizer=None, tokenizer_name=None, tasks=None, configs=None, debug=None):
    from .wikitext_eval import manual_eval_wikitext_awq
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    """
    tasks: should be a list of strings
    configs: a dict of configuration arguments
        "model_seqlen": model.seqlen set for eval. Default is 2048. 
        "xxx": add specific configs for tasks.
            For example: "wikitext" can choose "subset" and "split". 
        device_map_fn: allows a customized function for device mapping. 
    In this eval, we use lm_eval for most evaluation. But use AWQ evaluation for wikitext.
    """
    if configs is None:
        configs = {}

    if tasks is None:
        tasks = []
    else:
        import copy
        tasks = copy.deepcopy(tasks)
    
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

        device = configs.get("device", "cuda")
        if "device_map_fn" in configs:
            model = configs["device_map_fn"](model, device)
        else:
            model = model.to(device)
    
    if tokenizer_name is None:
        tokenizer_name = model_name
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    all_results = {}
    if "wikitext" in tasks:
        tasks.remove("wikitext")
        wikitext_config = configs.get("wikitext", {})
        subset = wikitext_config.get("subset", "wikitext-2-raw-v1")
        split = wikitext_config.get("split", "test")
        dataset = load_dataset("wikitext", subset, split=split)
        model_seqlen = wikitext_config.get("model_seqlen", 2048)
        ppl = manual_eval_wikitext_awq(model=model, tokenizer=tokenizer, dataset=dataset, debug=debug, model_seqlen=model_seqlen)
        all_results["wikitext"] = {"ppl": ppl}
        if debug:
            print(f"wikitext ppl={ppl}")
    if tasks: # more tasks
        from lm_eval.models.huggingface import HFLM
        from lm_eval import evaluator
        model_hflm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=1,
            device="cuda"
        )
        results = evaluator.simple_evaluate(
            model=model_hflm,
            tasks=tasks,
            verbosity="ERROR",
            log_samples=False,
        )
        all_results["lm_eval"] = results["results"]
    
    return all_results


def log_result(args_dict, result, log_file="example_log_file.jsonl"):
    from datetime import datetime
    import json
    """Append args + result + timestamp to a JSONL log file."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "args": args_dict,
        "result": result,
    }

    # append to file
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")