#!/usr/bin/env python3
"""Greedy per-layer/per-projection TEAL threshold search, against the
histograms run_grab_acts.py produced. Writes lookup_v{vec}{abs}/layer-N/results.csv
under the model's teal_output directory.

    python run_greedyopt.py --models llama2_7b --target-sparsity 0.6
    python run_greedyopt.py --models llama2_7b_aqlm_rvq --run-tag aerun20260807

Must be run with the *same* --run-tag (or omit it in both) used for the
matching run_grab_acts.py call -- it reads that run's histograms/activations.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vortex import paths
from vortex.models import MODEL_REGISTRY
from vortex.pipelines import run_greedyopt
from vortex.utils.logging import setup_logging


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", required=True,
                    help=f"comma-separated model keys. Known: {sorted(MODEL_REGISTRY)}")
    p.add_argument("--run-tag", default=None, help="must match the run_grab_acts.py --run-tag used")
    p.add_argument("--target-sparsity", type=float, default=0.6,
                    help="weighted-average effective sparsity to search up to")
    p.add_argument("--base-step-size", type=float, default=0.025)
    p.add_argument("--last-fraction", type=float, default=0.5,
                    help="fraction of the sequence (from the end) used for the activation-error comparison")
    p.add_argument("--vec-length", type=int, default=8)
    p.add_argument("--phi-func", default="l1-norm",
                    choices=["l0-norm", "l1-norm", "l2-norm", "l_inf-norm"])
    args = p.parse_args()

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    for k in keys:
        if k not in MODEL_REGISTRY:
            p.error(f"{k!r} not in models.MODEL_REGISTRY: {sorted(MODEL_REGISTRY)}")

    dist_args = {"vec_length": args.vec_length, "use_abs": True, "phi_func": args.phi_func}
    sparse_fn_args = {"vec_length": args.vec_length, "phi_func": args.phi_func}

    paths.ensure_dirs()
    setup_logging(filename=str(paths.ALGORITHM_LOGS_DIR / "run_greedyopt"), level="INFO")

    for k in keys:
        print(f"=== greedyopt: {k} (run_tag={args.run_tag}, target_sparsity={args.target_sparsity}) ===")
        run_greedyopt(
            k, run_tag=args.run_tag,
            target_sparsity=args.target_sparsity, base_step_size=args.base_step_size,
            last_fraction=args.last_fraction, dist_args=dist_args, sparse_fn_args=sparse_fn_args,
        )


if __name__ == "__main__":
    main()
