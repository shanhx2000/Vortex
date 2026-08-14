"""BF -- forceflow_geomean.pdf: why the dataflow has to be chosen per workload.

Latency (geomean over the three models) against batch size, for each dataflow
forced on and for Vortex's automatic selection. Two panels: linear-projection
only (1,0) and attention-intensive (512,4096). The crossover between MUF and LUF
moves with batch size and workload, which is the argument for selecting at
runtime rather than fixing one.

Data:   vortex_forceflow_evaluation_results.csv (group: batch_size_sweep)

    python BF_forceflow_geomean.py [--use-simulation] [--out-dir DIR]
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

import _paths
import _style
from _common import load_latest, geo_mean, require_rows

FIGURE = "forceflow_geomean.pdf"
INPUT_CSV = "vortex_forceflow_evaluation_results.csv"

CLOCK_HZ = 5e8
PAIRS = [(1, 0), (512, 4096)]

# "None" is not a missing value here -- it is the run where no dataflow was
# forced, i.e. Vortex's own choice. That is the line the figure argues for.
FLOWS = ["MUF", "LUF", "None"]
FLOW_LABEL = {"None": "Selected", "MUF": "MUF", "LUF": "LUF"}

_cmap = plt.get_cmap("tab20c")
FLOW_COLOR = {"None": _cmap(4), "MUF": _cmap(0), "LUF": _cmap(8)}

TITLES = {"(1,0)": "(a) Linear Projection Only",
          "(512,4096)": "(b) Attention-Intensive Workload"}

# Minor-tick bases, chosen per panel so the log2 axis stays readable over each
# panel's very different dynamic range.
MINOR_TICK_BASE = {0: 8, 1: 4}


def _uniform_subs(base, n=4):
    """n linearly spaced minor ticks strictly inside (1, base)."""
    return np.linspace(1, base, n + 2)[1:-1]


def plot_BF_forceflow_geomean():
    _style.use(_style.FONT_HARDWARE)

    df = load_latest(_paths.sim_data(INPUT_CSV))
    df = df[(df["method"] == "vortex")
            & (df["quant_scheme"] == "AQLM")
            & (df["processed_sparsity"] == "RVQ_0.0")].copy()
    require_rows(df, "Vortex AQLM / RVQ_0.0 forced-dataflow runs")

    df["force_dataflow"] = df["force_dataflow"].fillna("None")
    batch_sizes = sorted(df["batch_size"].unique())

    records = []
    for in_len, out_len in PAIRS:
        for flow in FLOWS:
            for bs in batch_sizes:
                sub = df[(df["input_length"] == in_len)
                         & (df["output_length"] == out_len)
                         & (df["force_dataflow"] == flow)
                         & (df["batch_size"] == bs)]
                if len(sub) == 0:
                    continue
                records.append({
                    "pair": f"({in_len},{out_len})",
                    "flow": flow,
                    "batch_size": bs,
                    "latency": geo_mean(sub["total_cycles"].values) / CLOCK_HZ,
                })
    df_plot = require_rows(pd.DataFrame(records), "forceflow latency records")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=False)
    x = np.arange(len(batch_sizes))

    for ax_idx, (ax, pair) in enumerate(zip(axes, ["(1,0)", "(512,4096)"])):
        for flow in FLOWS:
            ys = []
            for bs in batch_sizes:
                sub = df_plot[(df_plot["pair"] == pair)
                              & (df_plot["flow"] == flow)
                              & (df_plot["batch_size"] == bs)]
                ys.append(sub.iloc[0]["latency"] if len(sub) else np.nan)
            ax.plot(x, ys, marker="o", markersize=6, linewidth=2,
                    color=FLOW_COLOR[flow], label=FLOW_LABEL[flow])
            print(f"  {pair:<12s} {FLOW_LABEL[flow]:<9s} "
                  + " ".join(f"{v:.3g}" for v in ys))

        ax.set_xticks(x)
        ax.set_xticklabels(batch_sizes)
        ax.set_title(TITLES[pair])
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_locator(LogLocator(base=2))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda t, _: str(int(t)) if t >= 1 else f"{t:.2f}".rstrip("0").rstrip(".")))
        ax.tick_params(axis="y", which="major", length=6, width=1.5, pad=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    axes[0].set_ylabel("Latency (s, Geo. Mean)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.10), frameon=False)
    fig.supxlabel("Batch Size", y=0.1)

    plt.tight_layout()

    for ax_idx, ax in enumerate(axes):
        base = MINOR_TICK_BASE[ax_idx]
        ax.yaxis.set_minor_locator(LogLocator(base=base, subs=_uniform_subs(base)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="y", which="minor", length=3, width=1.5)

    path = _style.save(plt, _paths.figure_path(FIGURE))
    plt.close(fig)
    return path


if __name__ == "__main__":
    _paths.parse_args(__doc__)
    plot_BF_forceflow_geomean()
