"""The TEAL pipeline as callable functions, keyed by `models.MODEL_REGISTRY`.

Adapted from the original TEAL entry points (`{example_func,example_func_rvq,
extensive_eval_main,gather_teal_sparsities_thresholds}.py`. Those scripts
each inlined the same three steps -- build an `argparse.Namespace`, call one
of `teal.entry.{grab_acts,greedyopt,ppl_test}`, log the result -- as
hand-edited `__main__` blocks with most of the interesting lines commented
out. This module is that logic as functions, so a script (or a notebook, or
the tmux driver in `scripts/run_teal_group.sh`) can call
`run_grab_acts("llama2_7b")` directly instead of un-commenting the right
seven lines.

Pipeline stages, in order:
    1. run_prepare_rvq   (vq entries only) AQLM -> CustomRvqModel checkpoint
    2. run_grab_acts     forward the calibration set once, save activation
                          histograms per layer (`ActivationModule`)
    3. run_greedyopt     greedy per-layer/per-projection threshold search
                          against those histograms -> lookup/layer-N/results.csv
    4. run_ppl_test      evaluate at a given sparsity (uniform or greedy-looked-up)
    5. gather_sparsity_thresholds
                          re-derive thresholds at a fixed sparsity grid from
                          (3)'s lookup tables -> the JSONL simulator/sparsity.py reads

`run_tag`: every stage that writes into `ckpts/teal_output/<teal_dirname>/`
takes an optional `run_tag`. When set, the pipeline writes to
`<teal_dirname>-<run_tag>/` instead of the canonical `teal_dirname` --
required for any run that must not touch the committed checkpoints (see
`models.py`'s docstring). `run_tag=None` uses
the canonical directory verbatim, which is only safe when you specifically
intend to reproduce/extend the committed data.
"""
import argparse
import copy
import logging
import os
from datetime import datetime
from typing import Optional

from . import paths
from .hf_auth import ensure_hf_login
from .models import (
    DEFAULT_DIST_ARGS,
    DEFAULT_SPARSE_FN_ARGS,
    MODEL_TYPE_BY_KEY,
    ModelSpec,
    get_spec,
)


def _dated_run_tag() -> str:
    return f"aerun{datetime.now().strftime('%Y%m%d')}"


def teal_dir_for(spec: ModelSpec, run_tag: Optional[str]) -> str:
    return spec.teal_dirname if run_tag is None else f"{spec.teal_dirname}-{run_tag}"


def _model_name_for(spec: ModelSpec) -> str:
    """The value TEAL's `model_name`/`teal_path` machinery expects: an HF hub
    id for "regular" entries, or the local converted checkpoint path for
    "vq" entries."""
    if spec.model_mode == "regular":
        return spec.model_name
    return str(paths.rvq_model_path(spec.rvq_dirname))


def _model_args_extra(spec: ModelSpec) -> dict:
    return {"vq_ctx": spec.vq_ctx} if spec.model_mode == "vq" else {}


# ---------------------------------------------------------------------------
# 1. AQLM -> CustomRvqModel conversion (core/prepare/prepare_rvq.py)
# ---------------------------------------------------------------------------

def run_prepare_rvq(model_key: str) -> str:
    """Convert `spec.quant_model_path` (AQLM) into a CustomRvqModel checkpoint
    at ckpts/rvq_models/<rvq_dirname>/, unpacking each AQLM codebook into its
    own FP16 weight tensor (see core/model/custom_rvq.py). Skips the
    conversion (just returns the path) if it already exists.
    """
    spec = get_spec(model_key)
    assert spec.model_mode == "vq", f"{model_key} is not a _rvq entry"
    ensure_hf_login()
    paths.ensure_dirs()

    target = paths.rvq_model_path(spec.rvq_dirname)
    if target.exists() and any(target.iterdir()):
        logging.info(f"{target} already exists and is non-empty; skipping conversion.")
        return str(target)

    from .core.prepare.prepare_rvq import generate_rvq_model_from_aqlm
    generate_rvq_model_from_aqlm({
        "base_model_path": spec.base_model_path,
        "quant_model_path": spec.quant_model_path,
        "save_path": str(paths.RVQ_MODELS_DIR),
    })
    assert target.exists(), (
        f"generate_rvq_model_from_aqlm finished but {target} does not exist -- "
        f"it derives the save directory from quant_model_path's basename + '-rvq'; "
        f"check that still matches rvq_dirname={spec.rvq_dirname!r}."
    )
    return str(target)


# ---------------------------------------------------------------------------
# 2. Grab activations (teal/entry/grab_acts.py)
# ---------------------------------------------------------------------------

def run_grab_acts(
    model_key: str,
    run_tag: Optional[str] = None,
    dataset_size: int = 500,
    batch_size: int = 10,
    seq_len: int = 2048,
    dist_args: Optional[dict] = None,
    sparse_fn_args: Optional[dict] = None,
):
    """Forward `dataset_size` wikitext-2 tokens through the model once,
    layer by layer, saving a per-layer/per-hidden-type activation histogram
    to ckpts/teal_output/<teal_dir>/histograms/ (consumed by run_greedyopt)
    and the layer-0 input activations to .../activations/act_0.pt (consumed
    by run_greedyopt's per-layer error search).
    """
    spec = get_spec(model_key)
    ensure_hf_login()
    paths.ensure_dirs()

    from .teal.entry.grab_acts import grab_activations

    arg_dict = {
        "model_name": _model_name_for(spec),
        "teal_path": str(paths.teal_path(teal_dir_for(spec, run_tag))),
        "dataset": "wikitext",
        "dataset_size": dataset_size,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "dist_args": dist_args or DEFAULT_DIST_ARGS,
        "sparse_fn_args": sparse_fn_args or DEFAULT_SPARSE_FN_ARGS,
    }
    if spec.model_mode == "vq":
        arg_dict["model_args"] = _model_args_extra(spec)

    logging.info(f"[grab_acts] {model_key}: {arg_dict}")
    grab_activations(spec.model_mode, argparse.Namespace(**arg_dict))


# ---------------------------------------------------------------------------
# 3. Greedy per-layer threshold search (teal/entry/greedyopt.py)
# ---------------------------------------------------------------------------

def run_greedyopt(
    model_key: str,
    run_tag: Optional[str] = None,
    target_sparsity: float = 0.6,
    base_step_size: float = 0.025,
    last_fraction: float = 0.5,
    dist_args: Optional[dict] = None,
    sparse_fn_args: Optional[dict] = None,
):
    """Greedily raise per-projection (or, for _rvq entries, per-codebook
    per-projection) sparsity in `base_step_size` increments, always taking
    the projection whose increase adds the least activation error (weighted
    by `teal.entry.greedyopt.weight_dict`'s per-projection hardware-cost
    table), until the weighted-average effective sparsity reaches
    `target_sparsity`. Writes one CSV per layer to
    ckpts/teal_output/<teal_dir>/lookup_v{vec_length}{use_abs}/layer-N/results.csv,
    which run_ppl_test's greedy_flag=True path and gather_sparsity_thresholds
    both read back.

    Requires run_grab_acts to have populated .../histograms/ and
    .../activations/ first.
    """
    spec = get_spec(model_key)
    ensure_hf_login()
    paths.ensure_dirs()

    from .teal.entry.greedyopt import greedyopt

    arg_dict = {
        "model_name": _model_name_for(spec),
        "model_type": MODEL_TYPE_BY_KEY[model_key],
        "teal_path": str(paths.teal_path(teal_dir_for(spec, run_tag))),
        "target_sparsity": target_sparsity,
        "base_step_size": base_step_size,
        "last_fraction": last_fraction,
        "dist_args": dist_args or DEFAULT_DIST_ARGS,
        "sparse_fn_args": sparse_fn_args or DEFAULT_SPARSE_FN_ARGS,
    }
    if spec.model_mode == "vq":
        arg_dict["model_args"] = _model_args_extra(spec)

    logging.info(f"[greedyopt] {model_key}: {arg_dict}")
    greedyopt(argparse.Namespace(**arg_dict))


# ---------------------------------------------------------------------------
# 4. PPL / lm-eval evaluation (teal/entry/ppl_test.py)
# ---------------------------------------------------------------------------

def run_ppl_test(
    model_key: str,
    run_tag: Optional[str] = None,
    sparsity: float = 0.1,
    greedy_flag: bool = True,
    eval_tasks: Optional[list] = None,
    dist_args: Optional[dict] = None,
    sparse_fn_args: Optional[dict] = None,
) -> dict:
    """Evaluate the model at a given TEAL sparsity level. `greedy_flag=True`
    (default) loads the per-layer/per-projection thresholds run_greedyopt
    already searched for `sparsity` (interpolating to the nearest row in
    each layer's results.csv); `greedy_flag=False` applies `sparsity`
    uniformly to every projection instead, as a naive baseline. `sparsity=0`
    evaluates the dense model. Returns the `lm_eval_m()` results dict
    (`{"wikitext": {"ppl": ...}, "lm_eval": {...}}`).
    """
    spec = get_spec(model_key)
    ensure_hf_login()
    paths.ensure_dirs()

    from .teal.entry.ppl_test import ppl_test

    arg_dict = {
        "model_name": _model_name_for(spec),
        "teal_path": str(paths.teal_path(teal_dir_for(spec, run_tag))),
        "eval_tasks": eval_tasks or ["wikitext"],
        "greedy_flag": greedy_flag,
        "sparsity": sparsity,
        "use_vq": spec.model_mode == "vq",
        "dist_args": dist_args or DEFAULT_DIST_ARGS,
        "sparse_fn_args": sparse_fn_args or DEFAULT_SPARSE_FN_ARGS,
    }
    if spec.model_mode == "vq":
        arg_dict["model_args"] = _model_args_extra(spec)

    logging.info(f"[ppl_test] {model_key} @ sparsity={sparsity}: {arg_dict}")
    return ppl_test(argparse.Namespace(**arg_dict))


def log_eval_result(args_dict: dict, result: dict, log_file: str):
    from .teal.entry.ppl_test import log_result
    log_result(args_dict, result, log_file=log_file)


# ---------------------------------------------------------------------------
# 5. Gather thresholds at a fixed sparsity grid (examples/gather_teal_sparsities_thresholds.py)
# ---------------------------------------------------------------------------

def gather_sparsity_thresholds(
    model_keys: list,
    out_path: str,
    sparsities: Optional[list] = None,
    run_tag: Optional[str] = None,
    dist_args: Optional[dict] = None,
    sparse_fn_args: Optional[dict] = None,
):
    """Re-derive, for each sparsity in `sparsities`, the per-layer/per-projection
    (or per-layer/per-projection/per-codebook, for _rvq entries) thresholds
    that run_greedyopt's lookup table maps to that sparsity level, and write
    them all to one JSONL -- the same shape/producer as the committed
    `stats/ref/sparsity_info/teal_sparsities_thresholds_*.jsonl` that
    `simulator/sparsity.py`'s `find_sparsity_info()` reads.

    Writes to exactly `out_path` -- it never touches the committed reference
    file itself. If you intend to update the reference thresholds, that is a
    deliberate decision (re-running the full simulation set and every
    figure).

    Requires run_greedyopt to have already produced lookup tables for every
    key in `model_keys`.
    """
    sparsities = sparsities or [round(0.05 * i, 2) for i in range(1, 17)]  # 0.05 .. 0.80
    dist_args = dist_args or DEFAULT_DIST_ARGS
    sparse_fn_args = sparse_fn_args or DEFAULT_SPARSE_FN_ARGS

    from .teal.utils.utils_model import get_sparse_model

    all_sparsity_dict = []
    for model_key in model_keys:
        spec = get_spec(model_key)
        teal_dir = teal_path_str = str(paths.teal_path(teal_dir_for(spec, run_tag)))

        model_args = {
            "device": "cpu",
            "histogram_path": os.path.join(teal_path_str, "histograms"),
            "dist_args": dist_args,
            "sparse_fn_args": sparse_fn_args,
        }
        if spec.model_mode == "vq":
            model_args["vq_ctx"] = spec.vq_ctx

        model = get_sparse_model(_model_name_for(spec), **model_args)

        lookup_suffix = f"_v{dist_args['vec_length']}{dist_args['use_abs']}"
        greedy_path = os.path.join(teal_path_str, "lookup" + lookup_suffix)

        model_sparsity_dict = {}
        for sparsity in sparsities:
            if sparsity > 0:
                if spec.model_mode == "vq":
                    model.load_greedy_sparsities(greedy_path, sparsity, vq_ctx=model_args["vq_ctx"])
                else:
                    model.load_greedy_sparsities(greedy_path, sparsity)

            sparsity_dict = {}
            for i in range(len(model.model.layers)):
                if spec.model_mode != "vq":
                    sparsity_dict[f"layer_{i}"] = {
                        proj_full: {
                            "sparsity": model.model.layers[i].self_attn.sparse_fns[proj].get_sparsity()
                            if proj in ("q", "k", "v", "o")
                            else model.model.layers[i].mlp.sparse_fns[proj].get_sparsity(),
                            "threshold": model.model.layers[i].self_attn.sparse_fns[proj].get_threshold()
                            if proj in ("q", "k", "v", "o")
                            else model.model.layers[i].mlp.sparse_fns[proj].get_threshold(),
                        }
                        for proj, proj_full in [
                            ("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "o_proj"),
                            ("up", "up_proj"), ("down", "down_proj"), ("gate", "gate_proj"),
                        ]
                    }
                else:
                    layer_dict = {p: {} for p in
                                  ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj")}
                    for j in range(spec.vq_ctx["config_num_codebooks_glb"]):
                        for proj, proj_full in [
                            ("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "o_proj"),
                            ("up", "up_proj"), ("down", "down_proj"), ("gate", "gate_proj"),
                        ]:
                            sfn = (model.model.layers[i].self_attn.sparse_fns if proj in ("q", "k", "v", "o")
                                   else model.model.layers[i].mlp.sparse_fns)[f"{proj}{j}"]
                            layer_dict[proj_full][f"cb_{j}"] = {
                                "sparsity": sfn.get_sparsity(),
                                "threshold": sfn.get_threshold(),
                            }
                    sparsity_dict[f"layer_{i}"] = layer_dict
            model_sparsity_dict[sparsity] = copy.deepcopy(sparsity_dict)

        all_sparsity_dict.append({
            "model_mode": spec.model_mode,
            "model_name": _model_name_for(spec),
            "teal_path": teal_dir,
            "sparsities": sparsities,
            "model_args": {
                "use_vq": spec.model_mode == "vq",
                "dist_args": dist_args,
                "sparse_fn_args": sparse_fn_args,
                **({"vq_ctx": spec.vq_ctx} if spec.model_mode == "vq" else {}),
            },
            "sparsity_dict": model_sparsity_dict,
        })
        del model

    import json
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(json.dumps(all_sparsity_dict, indent=4))
    logging.info(f"Wrote thresholds for {model_keys} to {out_path}")
    return out_path
