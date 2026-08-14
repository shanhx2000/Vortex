"""Path resolution for the algorithm side, resolved from `__file__` rather
than the working directory (the convention `evaluation/visualization/_paths.py`
established on the simulator side, after CWD-relative paths there caused runs
to only work from inside one directory).

Layout (relative to the repository root):

    ckpts/teal_output/<model>/     TEAL histograms, activations, per-layer
                                    greedy-search lookup tables (`Distribution`,
                                    `ActivationModule`, `SparsifyFn` state)
    ckpts/rvq_models/<model>/      CustomRvqModel checkpoints converted from
                                    AQLM (see core/prepare/prepare_rvq.py)
    stats/ref/sparsity_info/       TEAL threshold JSONL the simulator reads
                                    (`find_sparsity_info()` in
                                    simulator/sparsity.py) -- the one artifact
                                    crossing the algorithm/simulator boundary
    logs/                          setup_logging() output for pipeline runs
"""
from pathlib import Path

VORTEX_PKG_DIR = Path(__file__).resolve().parent          # .../algorithm/src/vortex
ALGORITHM_DIR = VORTEX_PKG_DIR.parent.parent               # .../algorithm
REPO_ROOT = ALGORITHM_DIR.parent                           # the repository root

CKPTS_DIR = REPO_ROOT / "ckpts"
TEAL_OUTPUT_DIR = CKPTS_DIR / "teal_output"
RVQ_MODELS_DIR = CKPTS_DIR / "rvq_models"

STATS_DIR = REPO_ROOT / "stats"
SPARSITY_INFO_DIR = STATS_DIR / "ref" / "sparsity_info"
# The committed threshold table every published number was simulated against.
# A pipeline run must never overwrite it: doing so invalidates every figure
# without changing anything visible. `run_algorithm.sh gather` writes to
# logs/algorithm/ instead, and gather_thresholds.py refuses this filename.
REF_SPARSITY_INFO_GLOB = "teal_sparsities_thresholds_*.jsonl"

LOGS_DIR = REPO_ROOT / "logs"
ALGORITHM_LOGS_DIR = LOGS_DIR / "algorithm"

# HuggingFace token, read for the gated meta-llama/* and mistralai/* repos.
HF_TOKEN_PATH = Path.home() / "vortex" / "hf_token"


def teal_path(model_dirname: str) -> Path:
    """ckpts/teal_output/<model_dirname> -- histograms/activations/lookup for one model."""
    return TEAL_OUTPUT_DIR / model_dirname


def rvq_model_path(model_dirname: str) -> Path:
    """ckpts/rvq_models/<model_dirname> -- a converted CustomRvqModel checkpoint."""
    return RVQ_MODELS_DIR / model_dirname


def ensure_dirs():
    for d in (TEAL_OUTPUT_DIR, RVQ_MODELS_DIR, SPARSITY_INFO_DIR, ALGORITHM_LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
