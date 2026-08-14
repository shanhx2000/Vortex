"""Per-kernel GEMM benchmark: Vortex vs baseline accelerators.

Produces stats/simulation/kernel_results.csv, the input to figure BB (kernel_benchmark).

    python kernel_evaluation.py                  # baselines + Vortex
    python kernel_evaluation.py --methods baseline
    python kernel_evaluation.py --methods vortex

The two halves can be run independently -- that is what run_simulation.sh's
`baseline_kernel` and `kernel_eval` groups do. Each run rewrites only its own
rows in kernel_results.csv and leaves the other half untouched, so the file can
be built up incrementally and re-running a half is idempotent.
"""
import argparse
import time
import pandas as pd

from typing import List, Dict, Any

import _paths  # noqa: F401  (also puts simulator/ on sys.path)

# ===== simulator modules =====
from main import run_method_workload
from model_runner import Workload
from operations import GEMM
from quantization import apply_quant_methods
from sparsity import apply_sparsity_info

def apply_uniform_sparsity(ops, sparsity_list):
    for op in ops:
        op.metadata["sparsity_info"] = {
            "contextual_sparsity": sparsity_list
        }

# =========================================================
# build one GEMM kernel
# =========================================================
def build_gemm_workload(M, K, N, batch_size=1, dtype="fp16"):
    ops = []

    gemm = GEMM(
        name=f"fc_test_gemm_{M}x{K}x{N}",
        M=M,
        K=K,
        N=N,
        A_info={"dtype": dtype},
        B_info={"dtype": dtype},
        out_info={"dtype": dtype},
        parent_name="fc_test"
    )

    ops.append(gemm)

    return Workload(
        model_name="fc_test",
        input_length=0,
        output_length=0,
        batch_size=batch_size,
        prefill_ops=ops,
        decode_ops=[],   # prefill only
    )


# =========================================================
# run one method
# =========================================================
def run_single(method, workload, *,
               quant_scheme=None,
               sparsity=None,
               force_dataflow=None):

    # ===== copy the ops so one method cannot contaminate the next =====
    import copy
    workload = copy.deepcopy(workload)

    # ===== quant =====
    if quant_scheme:
        schemes = quant_scheme.split("|")
        for scheme in schemes:
            apply_quant_methods(workload.prefill_ops, [scheme])

    # ===== sparsity =====
    if sparsity is not None:
        assert sparsity == 0.3, "Currently only support uniform sparsity of 0.3 for kernel evaluation"
        apply_uniform_sparsity(workload.prefill_ops, [0.1, 0.5])


    # ===== args =====
    runtime_args = {
        "force_dataflow": force_dataflow,
        "compute_mode": "lut_based",
        "phase": "prefill",
    }

    result = run_method_workload(workload, method, args=runtime_args)

    total = result["total"]

    return {
        "cycles": total.get("total_cycles", 0),
        "energy": total.get("total_energy", 0.0),
    }


# =========================================================
# main benchmark
# =========================================================
def main(method_set="all"):
    run_baseline = method_set in ("all", "baseline")
    run_vortex = method_set in ("all", "vortex")

    kernels = [
        (1, 4096, 4096),
        (4, 4096, 4096),
        (8, 4096, 4096),
        (16, 4096, 4096),
        (32, 4096, 4096),
        (64, 4096, 4096),
        (128, 4096, 4096),
        (256, 4096, 4096),

        (1, 4096, 1024),
        (4, 4096, 1024),
        (8, 4096, 1024),
        (16, 4096, 1024),
        (32, 4096, 1024),
        (64, 4096, 1024),
        (128, 4096, 1024),
        (256, 4096, 1024),
    ]

    baseline_methods = ["systolic_array", "ant", "figlut", "figna"]

    results = []

    for M, K, N in kernels:
        print(f"\n===== Kernel {M}x{K}x{N} =====")

        workload = build_gemm_workload(M, K, N)


        # =================================================
        # 1️⃣ Baselines
        # =================================================
        for method in (baseline_methods if run_baseline else []):
            if method == "ant":
                quant = "W8A8"
            elif method in ["figlut", "figna"]:
                quant = "W4A16"
            else:
                quant = None

            res = run_single(
                method,
                workload,
                quant_scheme=quant,
            )

            print(f"{method}: {res}")


            results.append({
                "kernel": f"{M}x{K}x{N}",
                "method": method,
                "config": "baseline",
                **res
            })

        if run_vortex:
            # =================================================
            # 2️⃣ Vortex variants
            # =================================================

            # ---- default (AQLM) ----
            res = run_single("vortex", workload, quant_scheme="AQLM")
            results.append({
                "kernel": f"{M}x{K}x{N}",
                "method": "vortex",
                "config": "AQLM",
                **res
            })

            # ---- LUF ----
            res = run_single(
                "vortex",
                workload,
                quant_scheme="AQLM",
                force_dataflow="LUF"
            )
            results.append({
                "kernel": f"{M}x{K}x{N}",
                "method": "vortex",
                "config": "AQLM+LUF",
                **res
            })

            # ---- MUF ----
            res = run_single(
                "vortex",
                workload,
                quant_scheme="AQLM",
                force_dataflow="MUF"
            )
            results.append({
                "kernel": f"{M}x{K}x{N}",
                "method": "vortex",
                "config": "AQLM+MUF",
                **res
            })

            # ---- LUF ----
            res = run_single(
                "vortex",
                workload,
                quant_scheme="AQLM",
                force_dataflow="LUF",
                sparsity=0.3
            )
            results.append({
                "kernel": f"{M}x{K}x{N}",
                "method": "vortex",
                "config": "AQLM+LUF+Sparsity0.3",
                **res
            })

            # ---- MUF ----
            res = run_single(
                "vortex",
                workload,
                quant_scheme="AQLM",
                force_dataflow="MUF",
                sparsity=0.3
            )
            results.append({
                "kernel": f"{M}x{K}x{N}",
                "method": "vortex",
                "config": "AQLM+MUF+Sparsity0.3",
                **res
            })

            # ---- Vortex ----
            res = run_single(
                "vortex",
                workload,
                quant_scheme="AQLM",
                force_dataflow=None,
                sparsity=0.3
            )
            results.append({
                "kernel": f"{M}x{K}x{N}",
                "method": "vortex",
                "config": "AQLM+Sparsity0.3",
                **res
            })

    # =========================================================
    # save the CSV
    # =========================================================
    df_new = pd.DataFrame(results)
    out_path = _paths.results_dir() / "kernel_results.csv"

    # Merge rather than overwrite: a partial run (--methods baseline or
    # --methods vortex) must not drop the rows the other half wrote. Rows are
    # keyed on (kernel, method, config); freshly computed rows win.
    if out_path.exists():
        df_old = pd.read_csv(out_path)
        key = ["kernel", "method", "config"]
        if set(key).issubset(df_old.columns):
            df = pd.concat([df_old, df_new], ignore_index=True)
            df = df.drop_duplicates(subset=key, keep="last")
        else:
            print(f"[warn] {out_path} has an unexpected schema; overwriting")
            df = df_new
    else:
        df = df_new

    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df_new)} rows ({method_set}) to {out_path} "
          f"[{len(df)} rows total]")


# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--methods",
        choices=["all", "baseline", "vortex"],
        default="all",
        help="Which half of the kernel benchmark to run (default: all)",
    )
    parser.add_argument(
        "--out-dir", default=None, metavar="DIR",
        help="write kernel_results.csv here instead of stats/simulation/ "
             "(run_simulation.sh -s NAME passes the session directory)",
    )
    cli_args = parser.parse_args()
    _paths.set_results_dir(cli_args.out_dir)
    main(cli_args.methods)