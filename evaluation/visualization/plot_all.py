"""Regenerate every paper figure.

    python plot_all.py                    # from stats/ref/ -> figures/   [default]
    python plot_all.py --use-simulation   # hardware figures from your C1 run
    python plot_all.py --use-algorithm    # algorithm figures from your C2 run
    python plot_all.py --use-submitted    # hardware figures, submitted paper
    python plot_all.py --only BA BB       # a subset, by index

The default reads the committed reference data, so a fresh clone reproduces
figures/ref/ -- and therefore the paper -- without running anything. The two
component switches are independent; pass both to plot a full run of your own.

Figures whose input data is missing are reported and skipped rather than
aborting the run, so a partial simulation still produces everything it can.
"""
import argparse
import importlib
import sys
import traceback

import _paths

# index -> (module, function). BD and BE share one module and differ by argument.
FIGURES = [
    ("AA", "AA_motivation_vq", "plot_AA_motivation_vq", {}),
    ("AB", "AB_motivation_combined", "plot_AB_motivation_combined", {}),
    ("AC", "AC_sparsity_combined", "plot_AC_sparsity_combined", {}),
    ("AD", "AD_accuracy_vs_bitwidth", "plot_AD_accuracy_vs_bitwidth", {}),
    ("AE", "AE_phi_function", "plot_AE_phi_function", {}),
    ("BA", "BA_baseline_vs_vortex", "plot_BA_baseline_vs_vortex", {}),
    ("BB", "BB_kernel_benchmark", "plot_BB_kernel_benchmark", {}),
    ("BC", "BC_power_area_pie", "plot_BC_power_area_pie", {}),
    ("BD", "BD_BE_hardware_speedup", "plot_BD_BE_hardware_speedup",
     {"workload": "attention"}),
    ("BE", "BD_BE_hardware_speedup", "plot_BD_BE_hardware_speedup",
     {"workload": "projection"}),
    ("BF", "BF_forceflow_geomean", "plot_BF_forceflow_geomean", {}),
]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _paths.add_common_args(parser)
    parser.add_argument("--only", nargs="+", metavar="INDEX",
                        help="figure indices to render (e.g. AA BD)")
    parser.add_argument("--list", action="store_true",
                        help="list figure indices and exit")
    args = parser.parse_args()

    if args.list:
        for index, module, _, kwargs in FIGURES:
            extra = f"  {kwargs}" if kwargs else ""
            print(f"  {index}  {module}{extra}")
        return 0

    _paths.apply_common_args(args)
    wanted = [f for f in FIGURES if not args.only or f[0] in args.only]
    if args.only:
        unknown = set(args.only) - {f[0] for f in FIGURES}
        if unknown:
            print(f"Unknown figure index: {sorted(unknown)}", file=sys.stderr)
            return 2

    print(f"source: BA-BF {_paths.results_dir()}")
    print(f"        AC-AE {_paths.algorithm_dir()}")
    print(f"output: {_paths.out_dir()}\n")

    ok, skipped, failed = [], [], []
    for index, module_name, func_name, kwargs in wanted:
        print(f"[{index}] {module_name}.{func_name}")
        try:
            module = importlib.import_module(module_name)
            getattr(module, func_name)(**kwargs)
            ok.append(index)
        except FileNotFoundError as exc:
            print(f"  SKIP: {exc}\n")
            skipped.append(index)
        except SystemExit as exc:
            print(f"  SKIP: {exc}\n")
            skipped.append(index)
        except Exception:
            traceback.print_exc()
            failed.append(index)
        print()

    print(f"rendered {len(ok)}: {' '.join(ok)}")
    if skipped:
        print(f"skipped  {len(skipped)}: {' '.join(skipped)}  (missing input data)")
    if failed:
        print(f"FAILED   {len(failed)}: {' '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
