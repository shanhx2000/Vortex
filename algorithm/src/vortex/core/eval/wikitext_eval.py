import torch
from tqdm import tqdm

def manual_eval_wikitext_awq(model, tokenizer, dataset, debug=True, model_seqlen=2048):
    # AWQ: https://github.com/mit-han-lab/llm-awq/blob/main/awq/entry.py
    import torch.nn as nn
    testenc = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    model.seqlen = model_seqlen
    testenc = testenc.input_ids.to(model.device)
    nsamples = testenc.numel() // model.seqlen
    print("nsamples=", nsamples)
    model = model.eval()
    nlls = []
    
    pbar = tqdm(range(nsamples), desc="manual_eval_wikitext_awq", position=2, leave=False)
    
    for i in pbar:
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[
            :, (i * model.seqlen) : ((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
        if debug:
            pbar.set_description(
                f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood / model.seqlen).item():.2f}"
            )

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    return ppl.item()