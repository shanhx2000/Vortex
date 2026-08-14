#!/usr/bin/env python3
"""Evaluate wikitext PPL (and optionally lm-eval tasks) at one or more TEAL
sparsity levels, logging each result as a JSONL row.

    python run_ppl_test.py --models llama2_7b --sparsities 0.0,0.3,0.6
    python run_ppl_test.py --models llama2_7b_aqlm_rvq --run-tag aerun20260807 \\
        --sparsities 0.0,0.3,0.6 --eval-tasks wikitext,arc_easy,arc_challenge

--sparsities 0.0 evaluates the dense model regardless of --greedy/--uniform.
Requires run_greedyopt.py to have already produced lookup tables (unless
every requested sparsity is 0.0).
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vortex import paths
from vortex.models import MODEL_REGISTRY
from vortex.pipelines import log_eval_result, run_ppl_test
from vortex.utils.logging import setup_logging


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", required=True,
                    help=f"comma-separated model keys. Known: {sorted(MODEL_REGISTRY)}")
    p.add_argument("--run-tag", default=None, help="must match the run_greedyopt.py --run-tag used")
    p.add_argument("--sparsities", default="0.0", help="comma-separated sparsity floats")
    p.add_argument("--eval-tasks", default="wikitext", help="comma-separated lm_eval task names")
    p.add_argument("--uniform", action="store_true",
                    help="apply --sparsities uniformly to every projection instead of the greedy lookup")
    p.add_argument("--vec-length", type=int, default=8)
    p.add_argument("--phi-func", default="l1-norm",
                    choices=["l0-norm", "l1-norm", "l2-norm", "l_inf-norm"])
    p.add_argument("--out", default=None,
                    help="results JSONL path (default: logs/algorithm/ppl_test_<timestamp>.jsonl)")
    args = p.parse_args()

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    for k in keys:
        if k not in MODEL_REGISTRY:
            p.error(f"{k!r} not in models.MODEL_REGISTRY: {sorted(MODEL_REGISTRY)}")
    sparsities = [float(s) for s in args.sparsities.split(",") if s.strip()]
    eval_tasks = [t.strip() for t in args.eval_tasks.split(",") if t.strip()]

    dist_args = {"vec_length": args.vec_length, "use_abs": True, "phi_func": args.phi_func}
    sparse_fn_args = {"vec_length": args.vec_length, "phi_func": args.phi_func}

    paths.ensure_dirs()
    setup_logging(filename=str(paths.ALGORITHM_LOGS_DIR / "run_ppl_test"), level="INFO")

    out_path = args.out or str(
        paths.ALGORITHM_LOGS_DIR / f"ppl_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    for k in keys:
        for s in sparsities:
            print(f"=== ppl_test: {k} @ sparsity={s} (run_tag={args.run_tag}, greedy={not args.uniform}) ===")
            arg_dict = {
                "model_key": k, "run_tag": args.run_tag, "sparsity": s,
                "greedy_flag": not args.uniform, "eval_tasks": eval_tasks,
                # Recorded so build_algorithm_csvs.py can tell figure AE's
                # three phi curves apart; they differ only by these two.
                "phi_func": args.phi_func, "vec_length": args.vec_length,
            }
            result = run_ppl_test(
                k, run_tag=args.run_tag, sparsity=s, greedy_flag=not args.uniform,
                eval_tasks=eval_tasks, dist_args=dist_args, sparse_fn_args=sparse_fn_args,
            )
            print(result)
            log_eval_result(arg_dict, result, log_file=out_path)

    print(f"\nResults logged to {out_path}")


if __name__ == "__main__":
    main()
