#!/usr/bin/env python3
"""Re-derive per-layer/per-projection TEAL thresholds at a fixed sparsity
grid from run_greedyopt.py's lookup tables, and write them to one JSONL --
the same shape/producer as the committed
stats/ref/sparsity_info/teal_sparsities_thresholds_*.jsonl that
simulator/sparsity.py's find_sparsity_info() reads.

Usually reached as `./run_algorithm.sh ... gather` rather than directly.

    python gather_thresholds.py --models llama2_7b,llama2_7b_aqlm,llama2_7b_aqlm_rvq \\
        --run-tag aerun20260807 --out ../../logs/algorithm/thresholds_aerun20260807.jsonl

This NEVER writes to the committed reference filename -- pass --out explicitly.
Replacing that file means re-running the full simulation set and every figure,
so diff the result against it yourself first. To simulate against the table this
writes, no replacement is needed:

    ./run_simulation.sh --sparsity-info <the --out path>
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vortex import paths
from vortex.models import MODEL_REGISTRY
from vortex.pipelines import gather_sparsity_thresholds
from vortex.utils.logging import setup_logging


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", required=True,
                    help=f"comma-separated model keys. Known: {sorted(MODEL_REGISTRY)}")
    p.add_argument("--run-tag", default=None, help="must match the run_greedyopt.py --run-tag used")
    p.add_argument("--sparsities", default=None,
                    help="comma-separated sparsity floats (default: 0.05..0.80 step 0.05, matching "
                         "the committed reference file's grid)")
    p.add_argument("--vec-length", type=int, default=8)
    p.add_argument("--phi-func", default="l1-norm",
                    choices=["l0-norm", "l1-norm", "l2-norm", "l_inf-norm"])
    p.add_argument("--out", required=True, help="output JSONL path -- must not be the committed reference file")
    args = p.parse_args()

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    for k in keys:
        if k not in MODEL_REGISTRY:
            p.error(f"{k!r} not in models.MODEL_REGISTRY: {sorted(MODEL_REGISTRY)}")

    out_path = Path(args.out).resolve()
    if out_path.parent.samefile(paths.SPARSITY_INFO_DIR) and out_path.name.startswith("teal_sparsities_thresholds_"):
        existing = sorted(paths.SPARSITY_INFO_DIR.glob(paths.REF_SPARSITY_INFO_GLOB))
        if any(out_path.name == e.name for e in existing):
            p.error(f"refusing to overwrite the committed reference table {out_path} "
                    f"-- pick a different --out name.")

    sparsities = [float(s) for s in args.sparsities.split(",")] if args.sparsities else None
    dist_args = {"vec_length": args.vec_length, "use_abs": True, "phi_func": args.phi_func}
    sparse_fn_args = {"vec_length": args.vec_length, "phi_func": args.phi_func}

    paths.ensure_dirs()
    setup_logging(filename=str(paths.ALGORITHM_LOGS_DIR / "gather_thresholds"), level="INFO")

    written = gather_sparsity_thresholds(
        keys, out_path=str(out_path), sparsities=sparsities, run_tag=args.run_tag,
        dist_args=dist_args, sparse_fn_args=sparse_fn_args,
    )
    print(f"Wrote {written}")


if __name__ == "__main__":
    main()
