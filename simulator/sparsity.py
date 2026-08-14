"""Attaches measured contextual-sparsity ratios to ops.

Reads the per-layer, per-projection sparsity produced by the algorithm side and
writes it into `op.metadata["sparsity_info"]` for the cycle models to consume.

Which table is read, in order of precedence:

    set_sparsity_info_path(p)     an explicit call -- what run_experiments.py's
                                  --sparsity-info does
    $VORTEX_SPARSITY_INFO         environment override, used by run_simulation.sh
    DEFAULT_SPARSITY_INFO_PATH    stats/ref/sparsity_info/, the committed table
                                  every published number was simulated against

An entry is matched by **model identity** -- the quantized model the search ran
against, plus the vector length -- and deliberately *not* by the output
directory name, because `run_greedyopt.py --run-tag TAG` suffixes that directory
and a table you generated yourself would otherwise never match.

Sparsity is still joined to an individual op by parsing "layer<N>" and
"<x>_proj" out of the op name; renaming an op breaks that join silently, and the
op simply gets no sparsity info.
"""
import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The committed threshold table: C2's output, C1's input. Resolved relative to
# this file so runs do not depend on the CWD.
DEFAULT_SPARSITY_INFO_PATH = (
    REPO_ROOT / "stats" / "ref" / "sparsity_info"
    / "teal_sparsities_thresholds_20260402_231905.jsonl"
)

_SPARSITY_INFO_PATH = None


def set_sparsity_info_path(path):
    """Point every subsequent lookup at `path`. Highest precedence."""
    global _SPARSITY_INFO_PATH
    _SPARSITY_INFO_PATH = Path(path).expanduser().resolve() if path else None
    return _SPARSITY_INFO_PATH


def sparsity_info_path():
    if _SPARSITY_INFO_PATH is not None:
        return Path(_SPARSITY_INFO_PATH)
    env = os.environ.get("VORTEX_SPARSITY_INFO")
    return Path(env) if env else DEFAULT_SPARSITY_INFO_PATH


def _identity(info):
    """The (model, vec_length) a record was searched for, independent of where
    its output happened to be written."""
    return (
        info.get("model_mode"),
        # `model_name` is an HF hub id for regular entries and a local
        # ckpts/rvq_models/<dir> path for vq ones; the basename is the stable
        # part in both cases.
        Path(str(info.get("model_name", ""))).name,
        info.get("model_args", {}).get("dist_args", {}).get("vec_length"),
    )


def find_sparsity_info(search_dict, sparsity, path=None):
    file_path = Path(path) if path else sparsity_info_path()
    if not file_path.exists():
        raise FileNotFoundError(
            f"sparsity info file not found: {file_path}\n"
            f"  The committed table is {DEFAULT_SPARSITY_INFO_PATH.relative_to(REPO_ROOT)}.\n"
            f"  To use your own, pass --sparsity-info or set $VORTEX_SPARSITY_INFO.")

    with open(file_path, "r") as f:
        all_info = json.load(f)

    want = (search_dict["model_mode"], search_dict["model_name"], search_dict["vec_length"])
    matches = [info for info in all_info if _identity(info) == want]

    if not matches:
        available = sorted({f"{m}/{n} vec={v}" for m, n, v in map(_identity, all_info)})
        raise LookupError(
            f"no sparsity info for {want[1]} (mode={want[0]}, vec_length={want[2]}) "
            f"in {file_path}\n  available: " + "\n             ".join(available))
    if len(matches) > 1:
        paths = "\n             ".join(str(m.get("teal_path")) for m in matches)
        raise LookupError(
            f"{len(matches)} entries for {want[1]} in {file_path} -- ambiguous.\n"
            f"  they came from: {paths}\n"
            f"  Keep one search per model in a threshold file.")
    found = matches[0]

    # find the closest sparsity
    sparsity_info = found["sparsity_dict"]
    closest_sparsity = min(sparsity_info.keys(), key=lambda x: abs(float(x) - sparsity))
    closest_sparsity_info = sparsity_info[closest_sparsity]
    if abs(sparsity) < 1e-6:
        # set all sparsity values to 0.
        for layer in closest_sparsity_info.keys():
            for proj in closest_sparsity_info[layer].keys():
                if "cb_0" in closest_sparsity_info[layer][proj]:
                    for cb in closest_sparsity_info[layer][proj].keys():
                        closest_sparsity_info[layer][proj][cb]["sparsity"] = 0.0
                else:
                    closest_sparsity_info[layer][proj]["sparsity"] = 0.0
        closest_sparsity = 0.0

    return closest_sparsity_info

# Simulator model name -> the per-codebook (RVQ) model whose thresholds we want.
# These are `rvq_dirname`s from algorithm/src/vortex/models.py, i.e. the model
# the search ran against -- not the directory the search wrote to, which carries
# a --run-tag suffix in any run you do yourself.
_RVQ_MODEL_FOR = {
    "llama_2_7b":  "Llama-2-7b-AQLM-PV-2Bit-2x8-hf-rvq",
    "llama-2-7b":  "Llama-2-7b-AQLM-PV-2Bit-2x8-hf-rvq",
    "mistral_7b":  "Mistral-7B-Instruct-v0.2-AQLM-2Bit-2x8-rvq",
    "mistral-7b":  "Mistral-7B-Instruct-v0.2-AQLM-2Bit-2x8-rvq",
    "llama_2_13b": "Llama-2-13b-AQLM-2Bit-2x8-hf-rvq",
    "llama-2-13b": "Llama-2-13b-AQLM-2Bit-2x8-hf-rvq",
}


def find_rvq_sparsity_info_by_model(model_name, sparsity, vec_length=8, path=None):
    try:
        rvq_model = _RVQ_MODEL_FOR[model_name.lower()]
    except KeyError:
        raise ValueError(
            f"Unsupported model name: {model_name} "
            f"(known: {sorted(set(_RVQ_MODEL_FOR))})") from None
    return find_sparsity_info(
        {"model_mode": "vq", "model_name": rvq_model, "vec_length": vec_length},
        sparsity, path=path)

def apply_sparsity_info(ops, sparsity_config,
                        use_layer0=True,
                        use_layer_i=None
                        ):
    """
    To improve the efficiency of the simulation,
    we apply the sparsity to layer 0 and scales for the rest layers.
    Since our sparsity is obtained assuming identical sparsity across layers,
    this is a reasonable approximation.
    """

    assert sparsity_config["method"] == "RVQ", f"Unsupported sparsity method: {sparsity_config['method']}"
    contextual_sparsity = sparsity_config.get("contextual_sparsity", None)
    assert isinstance(contextual_sparsity, float), f"Contextual sparsity should be a float, got {type(contextual_sparsity)}"
    model_sparsity_info = find_rvq_sparsity_info_by_model(
        model_name=sparsity_config["model_name"],
        sparsity=sparsity_config["contextual_sparsity"],
        # Optional per-call override; otherwise sparsity_info_path() decides.
        path=sparsity_config.get("sparsity_info_path"),
    )

    #  "layer_31": {
    # "q_proj": {
    #   "cb_0": {
    #     "sparsity": 0.6500000000000002,
    #     "threshold": 2.784783363342285
    #   },
    #   "cb_1": {
    #     "sparsity": 0.9750000000000004,
    #     "threshold": 4.238309860229492
    #   }
    # },
    if use_layer_i is not None:
        assert use_layer0 is False
    else:
        assert use_layer0 is True
    for op in ops:
        from operations import OP
        assert isinstance(op, OP)
        name = op.name
        # Pull the layer index out of the op name.
        layer_match = re.search(r"layer(\d+)", name)
        if not layer_match:
            continue
        layer_id = layer_match.group(1)
        if use_layer_i is not None:
            layer_id = use_layer_i
        # ...and the projection name (q_proj, k_proj, ...).
        proj_match = re.search(r"([a-z]+_proj)", name)
        if not proj_match:
            continue
        proj = proj_match.group(1)
        sparsity_dict = model_sparsity_info.get(f"layer_{layer_id}", {}).get(proj, {})
        sparsity_list = [cb["sparsity"] for cb in sparsity_dict.values()]
        op.metadata["sparsity_info"] = {
            "contextual_sparsity": sparsity_list
        }
