"""BB -- kernel_benchmark.pdf: per-GEMM speedup and energy reduction.

Six representative GEMM shapes (M in {1,4,16} x K=4096 x N in {4096,1024}),
each normalized to the systolic array. Isolates Vortex's kernel-level behaviour
from any end-to-end model effects: the M=1 shapes are decode, M=16 is small-batch
prefill.

Bars that exceed the axis are clipped and annotated with their true value rather
than rescaling the axis -- Vortex's decode speedup is large enough to flatten
everything else otherwise.

Data:   kernel_results.csv (groups: baseline_kernel + kernel_eval)

    python BB_kernel_benchmark.py [--use-simulation] [--out-dir DIR]
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import _paths
import _style
from _common import require_rows

FIGURE = "kernel_benchmark.pdf"
INPUT_CSV = "kernel_results.csv"

PICKED_KERNELS = ["1x4096x4096", "4x4096x4096", "16x4096x4096",
                  "1x4096x1024", "4x4096x1024", "16x4096x1024"]
METHODS = ["systolic_array", "ant", "figna", "figlut", "AQLM+Sparsity0.3"]

YLIM_SPEEDUP = 5
YLIM_ENERGY = 3
FONT = 16

_cmap = plt.get_cmap("tab20c")
STYLE = {
    "systolic_array": {"color": _cmap(17), "hatch": "|"},
    "ant": {"color": _cmap(1), "hatch": "-"},
    "figlut": {"color": _cmap(9), "hatch": "xx"},
    "figna": {"color": _cmap(13), "hatch": "o"},
    "AQLM+Sparsity0.3": {"color": _cmap(4), "hatch": "//"},
}
METHOD_NAME = {"systolic_array": "SA", "ant": "ANT", "figlut": "FIGLUT",
               "figna": "FIGNA", "AQLM+Sparsity0.3": "Vortex"}


def _bars_with_cap(ax, xs, values, bar_width, offset, ylim_top, style, label=None):
    """Draw bars, clipping at ylim_top and printing the real value above."""
    capped = [np.nan if np.isnan(v) else min(v, ylim_top) for v in values]
    ax.bar(xs + offset, capped, width=bar_width, label=label,
           color=style.get("color"), hatch=style.get("hatch"),
           edgecolor="black")
    for x, v in zip(xs, values):
        if not np.isnan(v) and v > ylim_top:
            ax.text(x + offset + bar_width / 2, ylim_top, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=FONT)


def plot_BB_kernel_benchmark(picked_kernels=None):
    picked_kernels = PICKED_KERNELS if picked_kernels is None else picked_kernels
    plt.rcParams.update({"font.size": FONT, "pdf.fonttype": 42, "ps.fonttype": 42})

    df = pd.read_csv(_paths.sim_data(INPUT_CSV))

    # Baselines are identified by their method name, Vortex variants by their
    # config string (AQLM, AQLM+LUF, AQLM+Sparsity0.3, ...).
    df["method_label"] = df.apply(
        lambda r: r["method"] if r["method"] != "vortex" else r["config"], axis=1)
    require_rows(df[df["method_label"] == "AQLM+Sparsity0.3"],
                 "Vortex rows in kernel_results.csv")

    kernels = sorted(df["kernel"].unique(), key=lambda k: int(k.split("x")[0]))
    kernels = [k for k in kernels if k in picked_kernels]
    kernel_labels = [f"{k.split('x')[0]}xKx{k.split('x')[2]}" for k in kernels]

    def value(kernel, method, metric):
        sub = df[(df["kernel"] == kernel) & (df["method_label"] == method)]
        return sub.iloc[0][metric] if len(sub) else np.nan

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    bar_width = 1 / (len(METHODS) + 2)
    x = np.arange(len(kernels))

    sa_cycles = {k: value(k, "systolic_array", "cycles") for k in kernels}
    sa_energy = {k: value(k, "systolic_array", "energy") for k in kernels}

    for mi, method in enumerate(METHODS):
        offset = (mi - len(METHODS) / 2) * bar_width
        style = STYLE.get(method, {})

        speedup, energy_red = [], []
        for k in kernels:
            c, e = value(k, method, "cycles"), value(k, method, "energy")
            speedup.append(sa_cycles[k] / c if c and sa_cycles[k] else np.nan)
            energy_red.append(sa_energy[k] / e if e and sa_energy[k] else np.nan)

        print(f"  {METHOD_NAME[method]:<7s} speedup {np.round(speedup, 2)}")
        print(f"  {'':<7s} energy  {np.round(energy_red, 2)}")

        _bars_with_cap(ax1, x, speedup, bar_width, offset, YLIM_SPEEDUP,
                       style, label=METHOD_NAME.get(method, method))
        _bars_with_cap(ax2, x, energy_red, bar_width, offset, YLIM_ENERGY, style)

    ax1.set_ylabel("Norm. Speedup")
    ax2.set_ylabel("Norm. Energy Reduction")
    ax2.set_xticks(x)
    ax2.set_xticklabels(kernel_labels)
    ax1.set_ylim(0.3, YLIM_SPEEDUP)
    ax2.set_ylim(0.3, YLIM_ENERGY)

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    ax1.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.02),
               frameon=False)

    plt.tight_layout()
    path = _style.save(plt, _paths.figure_path(FIGURE), bbox_inches=None)
    plt.close(fig)
    return path


if __name__ == "__main__":
    _paths.parse_args(__doc__)
    plot_BB_kernel_benchmark()
