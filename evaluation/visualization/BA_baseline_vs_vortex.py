"""BA -- baseline_vs_vortex_combinedprefill512.pdf: Vortex vs baseline accelerators.

Two stacked panels (speedup, energy reduction) over prefill-512 workloads with
growing decode length, for three models and four baselines, all normalized to
the systolic array. The right-hand group is the geometric mean across every
(model, decode length) point, drawn on a twin axis so it can be read separately.


Data:   baseline_evaluation_results.csv  (group: baseline_e2e_small)
        vortex_evaluation_results.csv    (group: end_to_end_eval)

    python BA_baseline_vs_vortex.py [--use-simulation] [--out-dir DIR]
                                    [--variant {published,sfu,sfu_in_core}]

--variant selects which baseline dataset to normalize against; each writes its
own PDF so they can be compared side by side. See
the baselines skip softmax cycles in attention entirely, which makes Vortex's
speedup a conservative estimate.

  published    baselines carry 0.05 W of softmax leakage in their own configs,
               on top of a core power that already includes the softmax
  sfu          that 0.05 W replaced by Vortex's 0.00202 W (historical: it
               predates the double-counting finding)
  sfu_in_core  softmax power removed from the baseline configs entirely,
               because modules.core.power.static already accounts for it
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

import _paths
import _style
from _common import load_latest, geo_mean, model_label, require_rows

FIGURE = "baseline_vs_vortex_combinedprefill512.pdf"
BASELINE_CSV = "baseline_evaluation_results.csv"
VORTEX_CSV = "vortex_evaluation_results.csv"

# Baseline variants: same plot, different softmax accounting on the baselines.
# Vortex data is identical across all three -- only the normalizer changes.
VARIANTS = {
    "published":   (BASELINE_CSV, FIGURE),
    "sfu":         ("baseline_sfu_evaluation_results.csv",
                    "baseline_vs_vortex_combinedprefill512_sfu.pdf"),
    "sfu_in_core": ("baseline_sfu_in_core_evaluation_results.csv",
                    "baseline_vs_vortex_combinedprefill512_sfu_in_core.pdf"),
}

PAIRS = [(512, 512), (512, 1024), (512, 2048), (512, 4096)]
SAVEFIG_SUFFIX = "prefill512"

# The Vortex configuration the paper headlines: attention quantization on top of
# weight quantization, with 30% contextual sparsity.
VORTEX_LABEL = "vortex(AQLM|CQ,30%)"
METHODS = ["systolic_array", "ant", "figna", "figlut", VORTEX_LABEL]
LEGEND = {"systolic_array": "SA", "ant": "ANT", "figna": "FIGNA",
          "figlut": "FIGLUT", VORTEX_LABEL: "Ours"}

_cmap = plt.get_cmap("tab20c")
STYLE = {
    "systolic_array": {"color": _cmap(17), "hatch": "|"},
    "ant": {"color": _cmap(1), "hatch": "-"},
    "figna": {"color": _cmap(9), "hatch": "o"},
    "figlut": {"color": _cmap(13), "hatch": "xx"},
    VORTEX_LABEL: {"color": _cmap(4), "hatch": "//"},
}

# Vertical offset of the geo-mean value label, tuned per method so the five
# labels do not collide. Kept as data rather than logic -- it is pure layout.
GM_LABEL_LIFT = {"systolic_array": 1.4, "ant": 1.35, "figna": 1.0,
                 VORTEX_LABEL: 1.0}

METRICS = [("speedup", "Speedup", 32), ("energy_reduction", "Energy Reduction", 16)]

BAR_WIDTH = 0.15
GM_TOTAL_WIDTH = 2


def plot_BA_baseline_vs_vortex(pairs=None, variant="published"):
    pairs = PAIRS if pairs is None else pairs
    _style.use(_style.FONT_HARDWARE, titles=False)

    baseline_csv, figure = VARIANTS[variant]

    df_base = load_latest(_paths.sim_data(baseline_csv))
    df_vtx = load_latest(_paths.sim_data(VORTEX_CSV))

    def at_pairs(df):
        return df[df.apply(lambda r: (r["input_length"], r["output_length"]) in pairs
                           and r["batch_size"] == 1, axis=1)]

    df_base = at_pairs(df_base)
    df_vtx = at_pairs(df_vtx)

    df_vtx = df_vtx[(df_vtx["method"] == "vortex")
                    & (df_vtx["quant_scheme"] == "AQLM|CQ")
                    & (df_vtx["processed_sparsity"].isin(["RVQ_0.3", "RVQ_0.30"]))]
    require_rows(df_vtx, f"Vortex AQLM|CQ + 30% sparsity at {pairs}")
    require_rows(df_base, f"baselines at {pairs}")
    df_vtx = df_vtx.copy()
    df_vtx["method"] = VORTEX_LABEL

    df_all = pd.concat([df_base, df_vtx], ignore_index=True)
    models = sorted(df_all["model_name"].unique())
    kernels = [p[1] for p in pairs]

    # ---- normalize everything to the systolic array at the same point ----
    records = []
    for model in models:
        for in_len, out_len in pairs:
            base = df_all[(df_all["model_name"] == model)
                          & (df_all["input_length"] == in_len)
                          & (df_all["output_length"] == out_len)
                          & (df_all["method"] == "systolic_array")]
            if len(base) == 0:
                continue
            base_cycles = base.iloc[0]["total_cycles"]
            base_energy = base.iloc[0]["total_energy"]

            for method in METHODS:
                sub = df_all[(df_all["model_name"] == model)
                             & (df_all["input_length"] == in_len)
                             & (df_all["output_length"] == out_len)
                             & (df_all["method"] == method)]
                if len(sub) == 0:
                    continue
                records.append({
                    "model": model, "out_len": out_len, "method": method,
                    "speedup": base_cycles / sub.iloc[0]["total_cycles"],
                    "energy_reduction": base_energy / sub.iloc[0]["total_energy"],
                })
    df_plot = require_rows(pd.DataFrame(records), "normalized comparisons")

    # ---- x layout: models x decode lengths, then a detached geo-mean group ----
    x_positions, x_labels = [], []
    pos = 0
    for model in models:
        for k in kernels:
            x_positions.append(pos)
            x_labels.append((model, k))
            pos += 1
    gm_pos = pos + GM_TOTAL_WIDTH / 2
    x_positions.append(gm_pos)
    x_labels.append(("GM", "GM"))
    x_positions = np.array(x_positions)

    fig, axes = plt.subplots(2, 1, figsize=(20, 8))
    axes_right = []

    for ax_idx, (metric, ylabel, _) in enumerate(METRICS):
        ax = axes[ax_idx]
        ax_right = ax.twinx()
        axes_right.append(ax_right)

        for mi, method in enumerate(METHODS):
            values = []
            for model in models:
                for k in kernels:
                    sub = df_plot[(df_plot["model"] == model)
                                  & (df_plot["out_len"] == k)
                                  & (df_plot["method"] == method)]
                    values.append(sub.iloc[0][metric] if len(sub) else np.nan)
            values.append(geo_mean([v for v in values if not np.isnan(v)]))

            style = STYLE.get(method, {})
            offset = (mi - len(METHODS) / 2) * BAR_WIDTH

            ax.bar(x_positions[:-1] + offset, values[:-1], width=BAR_WIDTH,
                   label=LEGEND[method] if ax_idx == 0 else None,
                   color=style.get("color"), hatch=style.get("hatch"),
                   edgecolor="black", linewidth=1.8)

            gm_offset = (mi - len(METHODS) / 2) * GM_TOTAL_WIDTH / len(METHODS)
            ax_right.bar(gm_pos + gm_offset, values[-1], width=BAR_WIDTH * 2,
                         color=style.get("color"), hatch=style.get("hatch"),
                         edgecolor="black", linewidth=1.8)

            gm_y = values[-1]
            if not np.isnan(gm_y):
                text = "1x" if method == "systolic_array" else f"{gm_y:.3g}x"
                ax_right.text(gm_pos + gm_offset,
                              gm_y * GM_LABEL_LIFT.get(method, 1.1), text,
                              ha="center", va="bottom", fontsize=16,
                              fontweight="bold")
            print(f"  {metric:18s} {LEGEND[method]:<6s} geomean {gm_y:.3f}x")

        ax.set_yscale("log", base=2)
        ax_right.set_yscale("log", base=2)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([""] * len(x_positions))

        for i, (_, k) in enumerate(x_labels):
            ax.text(x_positions[i], -0.05, "Geo. Mean" if k == "GM" else k,
                    ha="center", va="top", transform=ax.get_xaxis_transform())

        # Model names go under the lower panel only, to avoid repeating them.
        group_size = len(kernels)
        if metric == "energy_reduction":
            for i, model in enumerate(models):
                ax.text(i * group_size + group_size / 2 - 0.5, -0.15,
                        model_label(model), ha="center", va="top",
                        transform=ax.get_xaxis_transform())

        for i in range(len(models) - 1):
            ax.axvline((i + 1) * group_size - 0.5, linestyle="--",
                       linewidth=2, color="black")
        ax.axvline(gm_pos - GM_TOTAL_WIDTH / 2 - GM_TOTAL_WIDTH / (len(METHODS) - 1),
                   linestyle="-", linewidth=2, color="black")
        ax.set_xlim(-0.5, gm_pos + GM_TOTAL_WIDTH / 2)

        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax_right.spines["right"].set_linewidth(2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(METHODS),
               bbox_to_anchor=(0.5, 0.95), frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.subplots_adjust(hspace=0.20)

    # Both y-axes are locked to the same range and tick set so the twin axis
    # under the geo-mean group reads on the same scale as the main one.
    for i, ax in enumerate(axes):
        ax_right = axes_right[i]
        top = METRICS[i][2]
        ax.set_ylim(top=top, bottom=0.8)
        ax_right.set_ylim(top=top, bottom=0.8)
        ax.autoscale(False)
        ax_right.autoscale(False)

        ticks = ax.get_yticks()[1:-1]
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(int(t)) for t in ticks])
        ax_right.set_yticks(ticks)
        ax_right.set_yticklabels([str(int(t)) for t in ticks])

        ymin, ymax = ax.get_ylim()
        minor = np.arange(np.floor(ymin), np.ceil(ymax) + 1)
        for a in (ax, ax_right):
            a.yaxis.set_minor_locator(FixedLocator(minor))
            a.yaxis.set_minor_formatter(NullFormatter())
            a.tick_params(axis="y", which="minor", length=3, width=1.5)
            a.tick_params(axis="y", which="major", width=1.5, length=6)

    path = _style.save(plt, _paths.figure_path(figure))
    plt.close(fig)
    return path


def _extra_args(parser):
    parser.add_argument(
        "--variant", choices=sorted(VARIANTS), default="published",
        help="which baseline dataset to normalize against; each variant needs "
             "its own simulation group (baseline_e2e_small / baseline_sfu / "
             "baseline_sfu_in_core). Default: published",
    )


if __name__ == "__main__":
    args = _paths.parse_args(__doc__, extra=_extra_args)
    print(plot_BA_baseline_vs_vortex(variant=args.variant))
