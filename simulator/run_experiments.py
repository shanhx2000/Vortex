"""Sweep driver: runs a grid of (model, length pair, batch, scheme) points to CSV.

SUPERSEDED as the AE entry point by `run_simulation.sh`, which
groups runs by the figure they feed and parallelises them. Kept for one-off
sweeps and because the AE script calls into it.

Baseline sweeps skip points already present in the target CSV, so re-running is
cheap and idempotent. Vortex sweeps always append.
"""
import argparse
from main import run_simulation
import pandas as pd
import time
import sparsity
from helper import log_results_to_csv

# =========================================================
# utils
# =========================================================
def parse_pairs(pairs_list):
    pairs = []
    for item in pairs_list:
        in_len, out_len = item.split(",")
        pairs.append((int(in_len), int(out_len)))
    return pairs


def run_baselines(
    model_names,
    pairs,
    baseline_methods,
    csv_path,
    batch_sizes,
    force_rerun=False,
):
    import os
    import pandas as pd

    # Load the existing CSV so completed rows can be skipped.
    if (not force_rerun) and os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)

        # Index the completed runs for O(1) lookup.
        existing_keys = set(
            zip(
                df_existing["model_name"],
                df_existing["method"],
                df_existing["input_length"],
                df_existing["output_length"],
                df_existing["batch_size"],
                # df_existing["quant_scheme"],
                # df_existing["processed_sparsity"],
            )
        )
    else:
        existing_keys = set()


    for model_name in model_names:
        for in_len, out_len in pairs:
            for bs in batch_sizes:
                for method in baseline_methods:

                    # ===== quant =====
                    if method == "ant":
                        quant_scheme = "W8A8"
                    elif method in ["figlut", "figna"]:
                        quant_scheme = "W4A16"
                    else:
                        quant_scheme = None

                    processed_sparsity = 0.0  # baselines are always dense

                    key = (
                        model_name,
                        method,
                        in_len,
                        out_len,
                        bs,
                        # "" if quant_scheme is None else quant_scheme,
                        # processed_sparsity,
                    )

                    # Already in the CSV -- skip.
                    if (not force_rerun) and (key in existing_keys):
                        print(
                            f"[SKIP] {model_name} | {method} | "
                            f"bs={bs} | in={in_len} out={out_len}"
                        )
                        continue

                    print(
                        f"\n[BASELINE] {model_name} | {method} | "
                        f"bs={bs} | in={in_len} out={out_len}"
                    )

                    results, args = run_simulation(
                        model_name=model_name,
                        methods=[method],
                        input_length=in_len,
                        output_length=out_len,
                        batch_size=bs,
                        quant_scheme=quant_scheme,
                    )

                    st_time = time.time()
                    log_results_to_csv(results, args, csv_path)
                    print(f"{time.time()-st_time} seconds to log results.")

                    # Record it immediately so a duplicate point inside this
                    # same invocation is not run twice.
                    if not force_rerun:
                        existing_keys.add(key)

# =========================================================
# Vortex
# =========================================================
def run_vortex_sweep(
    model_names,
    pairs,
    csv_path,
    batch_sizes,
    quant_schemes,
    sparsity_list,
):

    for model_name in model_names:
        for in_len, out_len in pairs:
            for bs in batch_sizes:
                for quant_scheme in quant_schemes:
                    for sparsity in sparsity_list:

                        print(
                            f"\n[VORTEX] {model_name} | {quant_scheme} | "
                            f"sparsity={sparsity} | bs={bs} | in={in_len} out={out_len}"
                        )

                        results, args = run_simulation(
                            model_name=model_name,
                            methods=["vortex"],
                            input_length=in_len,
                            output_length=out_len,
                            batch_size=bs,
                            quant_scheme=quant_scheme,
                            processed_sparsity=f"RVQ_{sparsity}",
                        )

                        log_results_to_csv(results, args, csv_path)

def run_vortex_flow_sweep(
    model_names,
    pairs,
    csv_path,
    batch_sizes,
    quant_schemes,
    sparsity_list,
    dataflows,
):
    for model_name in model_names:
        for in_len, out_len in pairs:
            for bs in batch_sizes:
                for quant_scheme in quant_schemes:
                    for sparsity in sparsity_list:
                        for flow in dataflows:
                            if flow == "None":
                                flow = None
                            assert flow in [None, "MUF", "LUF"], f"{flow}"

                            print(
                                f"\n[VORTEX-FLOW] {model_name} | {quant_scheme} | "
                                f"sparsity={sparsity} | flow={flow} | "
                                f"bs={bs} | in={in_len} out={out_len}"
                            )

                            results, args = run_simulation(
                                model_name=model_name,
                                methods=["vortex"],
                                input_length=in_len,
                                output_length=out_len,
                                batch_size=bs,
                                quant_scheme=quant_scheme,
                                processed_sparsity=f"RVQ_{sparsity}",
                                force_dataflow=flow,
                            )

                            log_results_to_csv(results, args, csv_path)

# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    # ===== models =====
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names (e.g. llama-2-7b)",
    )

    # ===== pairs =====
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="List of input,output pairs (e.g. 0,0 0,1 512,512)",
    )

    parser.add_argument(
        "--batch_sizes",
        nargs="+",
        type=int,
        default=[1],
        help="Batch sizes (e.g. 1 4 8)",
    )

    # ===== csv =====
    parser.add_argument(
        "--csv_path",
        type=str,
        default="evaluation_results.csv",
    )

    # ===== run switches =====
    parser.add_argument("--run_baseline", action="store_true")
    parser.add_argument("--run_vortex", action="store_true")

    # ===== baseline methods =====
    parser.add_argument(
        "--baseline_methods",
        nargs="+",
        default=["systolic_array", "ant", "figlut", "figna"],
        help="Subset of baseline methods",
    )

    # ===== vortex methods =====
    parser.add_argument(
        "--vortex_quant_schemes",
        nargs="+",
        default=["AQLM", "AQLM|CQ"],
        help="Quant schemes for Vortex",
    )

    parser.add_argument(
        "--vortex_sparsity_list",
        nargs="+",
        type=float,
        default=[0.0, 0.3],
        help="Sparsity list for Vortex (e.g. 0.0 0.3)",
    )

    # ===== new mode =====
    parser.add_argument("--run_vortex_flow", action="store_true")

    parser.add_argument(
        "--vortex_force_dataflow",
        nargs="+",
        default=["None"],
        choices=["None", "MUF", "LUF"],
        help="Force dataflow modes",
    )
    parser.add_argument(
        "--sparsity-info", "--sparsity_info", dest="sparsity_info", default=None,
        metavar="FILE",
        help="contextual-sparsity threshold table to simulate against. Default: "
             "the committed stats/ref/sparsity_info/ table, which every "
             "published number came from. Point this at a table you generated "
             "yourself with algorithm/scripts/gather_thresholds.py.",
    )

    args = parser.parse_args()

    pairs = parse_pairs(args.pairs)

    # Applies to the Vortex paths only -- baselines have no contextual sparsity.
    if args.sparsity_info:
        sparsity.set_sparsity_info_path(args.sparsity_info)

    print("\n===== CONFIG =====")
    print("Models:", args.models)
    print("Pairs:", pairs)
    print("Batch sizes:", args.batch_sizes)
    print("CSV:", args.csv_path)
    print("Run baseline:", args.run_baseline)
    print("Run vortex:", args.run_vortex)
    print("Baseline methods:", args.baseline_methods)
    print("Sparsity info:", sparsity.sparsity_info_path())
    print("==================\n")

    # ===== run =====
    if args.run_baseline:
        run_baselines(
            model_names=args.models,
            pairs=pairs,
            baseline_methods=args.baseline_methods,
            batch_sizes=args.batch_sizes,
            csv_path=args.csv_path,
        )

    if args.run_vortex:
        run_vortex_sweep(
            model_names=args.models,
            pairs=pairs,
            batch_sizes=args.batch_sizes,
            csv_path=args.csv_path,
            quant_schemes=args.vortex_quant_schemes,
            sparsity_list=args.vortex_sparsity_list,
        )

    if args.run_vortex_flow:
        run_vortex_flow_sweep(
            model_names=args.models,
            pairs=pairs,
            batch_sizes=args.batch_sizes,
            csv_path=args.csv_path,
            quant_schemes=args.vortex_quant_schemes,
            sparsity_list=args.vortex_sparsity_list,
            dataflows=args.vortex_force_dataflow,
        )

# =========================================================
if __name__ == "__main__":
    main()