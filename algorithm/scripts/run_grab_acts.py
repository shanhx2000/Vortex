#!/usr/bin/env python3
"""Forward the wikitext-2 calibration set through a model once, saving
per-layer activation histograms (input to run_greedyopt.py).

    python run_grab_acts.py --models llama2_7b
    python run_grab_acts.py --models llama2_7b,llama2_7b_aqlm --run-tag aerun20260807

`--run-tag` writes to ckpts/teal_output/<teal_dirname>-<run-tag>/ instead of
the canonical <teal_dirname>/, so a fresh run never overwrites the committed
checkpoints. Omit it only if you deliberately intend to touch the canonical
directory, which is where the committed searches live.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vortex import paths
from vortex.models import MODEL_REGISTRY
from vortex.pipelines import run_grab_acts
from vortex.utils.logging import setup_logging


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", required=True,
                    help=f"comma-separated model keys. Known: {sorted(MODEL_REGISTRY)}")
    p.add_argument("--run-tag", default=None,
                    help="suffix for the output teal_output directory (recommended; see above)")
    p.add_argument("--dataset-size", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=2048)
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
    setup_logging(filename=str(paths.ALGORITHM_LOGS_DIR / "run_grab_acts"), level="INFO")

    for k in keys:
        print(f"=== grab_acts: {k} (run_tag={args.run_tag}) ===")
        run_grab_acts(
            k, run_tag=args.run_tag,
            dataset_size=args.dataset_size, batch_size=args.batch_size, seq_len=args.seq_len,
            dist_args=dist_args, sparse_fn_args=sparse_fn_args,
        )


if __name__ == "__main__":
    main()
