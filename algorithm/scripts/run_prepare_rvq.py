#!/usr/bin/env python3
"""Convert AQLM checkpoints into CustomRvqModel checkpoints (unpack each AQLM
codebook into its own FP16 weight tensor -- core/model/custom_rvq.py), the
input the "_rvq" TEAL entries in models.py need.

    python run_prepare_rvq.py --models llama2_7b_aqlm_rvq,mistral_7b_aqlm_rvq
    python run_prepare_rvq.py --models all_rvq        # every _rvq entry
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vortex import paths
from vortex.models import MODEL_REGISTRY
from vortex.pipelines import run_prepare_rvq
from vortex.utils.logging import setup_logging

RVQ_KEYS = [k for k, s in MODEL_REGISTRY.items() if s.model_mode == "vq"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="all_rvq",
                    help=f"comma-separated model keys, or 'all_rvq'. Known _rvq keys: {RVQ_KEYS}")
    args = p.parse_args()

    keys = RVQ_KEYS if args.models == "all_rvq" else [k.strip() for k in args.models.split(",") if k.strip()]
    for k in keys:
        if k not in RVQ_KEYS:
            p.error(f"{k!r} is not a _rvq entry in models.MODEL_REGISTRY. Known: {RVQ_KEYS}")

    paths.ensure_dirs()
    setup_logging(filename=str(paths.ALGORITHM_LOGS_DIR / "run_prepare_rvq"), level="INFO")

    for k in keys:
        print(f"=== prepare_rvq: {k} ===")
        out = run_prepare_rvq(k)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
