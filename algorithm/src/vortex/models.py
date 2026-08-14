"""Model registry: the (model, quantization) combinations the pipeline knows.

One place for what used to be hand-typed in each entry point -- the HuggingFace
model id, the `ckpts/teal_output/` subdirectory, and the `vq_ctx` describing the
codebook layout.

Ten entries: {Llama-2-7B, Llama-2-13B, Mistral-7B} x {dense, AQLM 2-bit, AQLM
2-bit converted to CustomRvqModel with a per-codebook ("_rvq") TEAL search},
plus `llama2_7b_aqlm_rvq_uniform`, which applies one threshold across all
codebooks and exists for the ablation.

`teal_dirname` / `rvq_dirname` are the *canonical* names -- what is committed
under `ckpts/` (see `ckpts/README.md`). Pipeline runs must NOT write back into
those directories: pass `run_tag`, which appends `-<tag>`, or
`run_algorithm.sh --use-ref` to read them without writing.

`teal_dirname` equals `key` for every entry, so the directory a reviewer looks
in is spelled the same as the `-m` argument they typed. It stays a separate
field because `rvq_dirname` does not follow that rule -- several keys share one
converted checkpoint -- and because a future entry may need to point at a
directory it does not name.

The threshold table `simulator/sparsity.py` reads records the same names in its
`teal_path`, but matches on model identity rather than on that string, so a
`--run-tag`ged search of your own resolves the same way the committed one does.
"""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_mode: str                    # "regular" | "vq"
    teal_dirname: str                  # ckpts/teal_output/<teal_dirname>/

    # "regular" entries (dense or AQLM-quantized, loaded directly from HF hub)
    model_name: Optional[str] = None

    # "vq" entries: converted from an AQLM checkpoint via prepare_rvq
    base_model_path: Optional[str] = None      # dense HF id (config/tokenizer source)
    quant_model_path: Optional[str] = None     # AQLM-quantized HF id (conversion input)
    rvq_dirname: Optional[str] = None          # ckpts/rvq_models/<rvq_dirname>/
    vq_ctx: Optional[dict] = None

    def __post_init__(self):
        if self.model_mode == "regular":
            assert self.model_name, f"{self.key}: regular entries need model_name"
        elif self.model_mode == "vq":
            assert self.base_model_path and self.quant_model_path and self.rvq_dirname, \
                f"{self.key}: vq entries need base_model_path/quant_model_path/rvq_dirname"
            assert self.vq_ctx is not None, f"{self.key}: vq entries need vq_ctx"
        else:
            raise ValueError(f"{self.key}: unknown model_mode {self.model_mode!r}")


# Standard AQLM RVQ context used by every "_rvq" entry: 2 codebooks, one
# SparsifyFn threshold searched independently per codebook (not the uniform
# single-threshold-for-all-codebooks alternative `uniform_sparse_fn=True`
# would give).
_RVQ_CTX_2CB = {"config_num_codebooks_glb": 2, "uniform_sparse_fn": False}

# Ablation variant used only by the phi-function exploration: same 2-codebook
# AQLM->CustomRvqModel conversion as llama2_7b_aqlm_rvq, but with a single
# threshold shared across both codebooks instead of one per codebook -- the
# "codebook-uniform" (TEAL-style) comparison point for the phi sweep.
_RVQ_CTX_2CB_UNIFORM = {"config_num_codebooks_glb": 2, "uniform_sparse_fn": True}

MODEL_REGISTRY: Dict[str, ModelSpec] = {
    spec.key: spec for spec in [
        ModelSpec(
            key="llama2_7b",
            model_mode="regular",
            model_name="meta-llama/Llama-2-7b-hf",
            teal_dirname="llama2_7b",
        ),
        ModelSpec(
            key="llama2_7b_aqlm",
            model_mode="regular",
            model_name="ISTA-DASLab/Llama-2-7b-AQLM-PV-2Bit-2x8-hf",
            teal_dirname="llama2_7b_aqlm",
        ),
        ModelSpec(
            key="llama2_7b_aqlm_rvq",
            model_mode="vq",
            base_model_path="meta-llama/Llama-2-7b-hf",
            quant_model_path="ISTA-DASLab/Llama-2-7b-AQLM-PV-2Bit-2x8-hf",
            rvq_dirname="Llama-2-7b-AQLM-PV-2Bit-2x8-hf-rvq",
            teal_dirname="llama2_7b_aqlm_rvq",
            vq_ctx=_RVQ_CTX_2CB,
        ),
        ModelSpec(
            key="llama2_7b_aqlm_rvq_uniform",
            model_mode="vq",
            base_model_path="meta-llama/Llama-2-7b-hf",
            quant_model_path="ISTA-DASLab/Llama-2-7b-AQLM-PV-2Bit-2x8-hf",
            rvq_dirname="Llama-2-7b-AQLM-PV-2Bit-2x8-hf-rvq",
            teal_dirname="llama2_7b_aqlm_rvq_uniform",
            vq_ctx=_RVQ_CTX_2CB_UNIFORM,
        ),
        ModelSpec(
            key="llama2_13b",
            model_mode="regular",
            model_name="meta-llama/Llama-2-13b-hf",
            teal_dirname="llama2_13b",
        ),
        ModelSpec(
            key="llama2_13b_aqlm",
            model_mode="regular",
            # NOT "...AQLM-PV-2Bit-2x8-hf" (that repo does not exist on the
            # Hub for the 13B model).
            model_name="ISTA-DASLab/Llama-2-13b-AQLM-2Bit-2x8-hf",
            teal_dirname="llama2_13b_aqlm",
        ),
        ModelSpec(
            key="llama2_13b_aqlm_rvq",
            model_mode="vq",
            base_model_path="meta-llama/Llama-2-13b-hf",
            quant_model_path="ISTA-DASLab/Llama-2-13b-AQLM-2Bit-2x8-hf",
            rvq_dirname="Llama-2-13b-AQLM-2Bit-2x8-hf-rvq",
            teal_dirname="llama2_13b_aqlm_rvq",
            vq_ctx=_RVQ_CTX_2CB,
        ),
        ModelSpec(
            key="mistral_7b",
            model_mode="regular",
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            teal_dirname="mistral_7b",
        ),
        ModelSpec(
            key="mistral_7b_aqlm",
            model_mode="regular",
            model_name="ISTA-DASLab/Mistral-7B-Instruct-v0.2-AQLM-2Bit-2x8",
            teal_dirname="mistral_7b_aqlm",
        ),
        ModelSpec(
            key="mistral_7b_aqlm_rvq",
            model_mode="vq",
            base_model_path="mistralai/Mistral-7B-Instruct-v0.2",
            quant_model_path="ISTA-DASLab/Mistral-7B-Instruct-v0.2-AQLM-2Bit-2x8",
            rvq_dirname="Mistral-7B-Instruct-v0.2-AQLM-2Bit-2x8-rvq",
            teal_dirname="mistral_7b_aqlm_rvq",
            vq_ctx=_RVQ_CTX_2CB,
        ),
    ]
}

# The TEAL search configuration used throughout (per the
# task's instruction): l1-norm group metric, vector length 8. `use_abs=True`
# groups by sum of |x| within each length-8 chunk of the hidden dim before
# thresholding -- see `teal/utils/utils_activations.py`'s `SparsifyFn`/`Distribution`.
DEFAULT_DIST_ARGS = {"vec_length": 8, "use_abs": True, "phi_func": "l1-norm"}
DEFAULT_SPARSE_FN_ARGS = {"vec_length": 8, "phi_func": "l1-norm"}

# examples/entry/greedyopt.py's per-projection cost weights, keyed by the
# model *architecture* name greedyopt.py's weight_dict uses (not our
# model_key) -- greedy search spends its step budget proportional to these so
# that projections with a cheaper hardware cost get sparsified more eagerly.
MODEL_TYPE_BY_KEY = {
    "llama2_7b": "Llama-2-7B", "llama2_7b_aqlm": "Llama-2-7B", "llama2_7b_aqlm_rvq": "Llama-2-7B",
    "llama2_7b_aqlm_rvq_uniform": "Llama-2-7B",
    "llama2_13b": "Llama-2-13B", "llama2_13b_aqlm": "Llama-2-13B", "llama2_13b_aqlm_rvq": "Llama-2-13B",
    "mistral_7b": "Mistral-7B", "mistral_7b_aqlm": "Mistral-7B", "mistral_7b_aqlm_rvq": "Mistral-7B",
}


def get_spec(model_key: str) -> ModelSpec:
    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model key {model_key!r}. Known keys: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_key]
