"""BD / BE -- vortex_hardware_speedup_inlen{I}_outlen{O}.pdf: improvement breakdown.

One figure per operating point, both produced by the same function:

  BD  (512, 4096)  attention-intensive  -> vortex_hardware_speedup_inlen512_outlen4096.pdf
  BE  (1, 0)       projection-intensive -> vortex_hardware_speedup_inlen1_outlen0.pdf

Latency (left half) and energy (right half) on a shared x axis split by a
divider, showing what each Vortex feature contributes on top of the systolic
array: forced LUF dataflow, weight quantization, attention quantization, and
contextual sparsity.

Data:   vortex_projection_attention_intensive_results.csv (group: impr_ablation)
        baseline_evaluation_results.csv                   (group: baseline_e2e_small)

    python BD_BE_hardware_speedup.py [--use-simulation] [--out-dir DIR]
    python BD_BE_hardware_speedup.py --workload attention   # just BD
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import _paths
import _style
from _common import load_latest, geo_mean, model_label, require_rows

VORTEX_CSV = "vortex_projection_attention_intensive_results.csv"
BASELINE_CSV = "baseline_evaluation_results.csv"

CLOCK_HZ = 5e8      # 500 MHz; cycles -> seconds

# Bars, in draw order. These are method_labels built from the CSV columns, so
# they follow the MUF/LUF naming -- code still looking for "AQLM+LtC" silently
# matches nothing.
METHODS = ["systolic_array", "AQLM+LUF", "AQLM", "AQLM|CQ", "AQLM|CQ+Sparsity"]
METHOD_NAME = {"systolic_array": "SA", "AQLM+LUF": "LUF", "AQLM": "Wgt-Q",
               "AQLM|CQ": "Wgt-Q|Att-Q", "AQLM|CQ+Sparsity": "Vortex"}

_cmap = plt.get_cmap("tab20c")
STYLE = {
    "systolic_array": {"color": _cmap(17), "hatch": "|"},
    "AQLM+LUF": {"color": _cmap(5), "hatch": "\\"},
    "AQLM": {"color": _cmap(1), "hatch": "-"},
    "AQLM|CQ": {"color": _cmap(9), "hatch": "xx"},
    "AQLM|CQ+Sparsity": {"color": _cmap(4), "hatch": "//"},
}

WORKLOADS = {
    # name        -> (input_length, output_length, latency_ylim, energy_ylim, energy_scale, unit)
    "attention":  (512, 4096, 1500, 10.0, 1000.0, "kJ"),
    "projection": (1, 0, 0.07, 0.7, 1.0, "J"),
}


def _has_effective_sparsity(s):
    if pd.isna(s) or s is None:
        return False
    s = str(s).strip()
    if s == "" or s.lower() == "none":
        return False
    # Accepts "RVQ_0.3", "RVQ_30.0" and bare floats.
    return float(s.split("_")[-1] if "_" in s else s) > 0


def _build_label(row):
    """quant_scheme [+dataflow] [+Sparsity] -- the bar identity."""
    quant = row.get("quant_scheme", None)
    if pd.isna(quant) or quant is None or str(quant).strip() == "":
        return None
    label = str(quant).strip()
    flow = row.get("force_dataflow", None)
    if not pd.isna(flow) and str(flow).strip() != "":
        label += f"+{str(flow).strip()}"
    if _has_effective_sparsity(row.get("processed_sparsity", None)):
        label += "+Sparsity"
    return label


def _bars_with_cap(ax, xs, values, bar_width, offset, ylim_top, style,
                   label=None, text_x_offset=0.0, text_y_offset=0.0):
    capped = [np.nan if np.isnan(v) else min(v, ylim_top) for v in values]
    ax.bar(xs + offset, capped, width=bar_width, label=label,
           color=style.get("color"), hatch=style.get("hatch"),
           edgecolor="black")
    for x, v in zip(xs, values):
        if not np.isnan(v) and v > ylim_top:
            ax.text(x + offset + bar_width / 2 + text_x_offset,
                    ylim_top + text_y_offset, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=_style.FONT_HARDWARE)


def plot_BD_BE_hardware_speedup(workload="attention"):
    in_len, out_len, lat_ylim, ene_ylim, ene_scale, unit = WORKLOADS[workload]
    figure = f"vortex_hardware_speedup_inlen{in_len}_outlen{out_len}.pdf"
    _style.use(_style.FONT_HARDWARE)

    df = load_latest(_paths.sim_data(VORTEX_CSV))
    df = df.copy()
    df["method_label"] = df.apply(_build_label, axis=1)
    df = df[df["method_label"].notna()]

    df_base = load_latest(_paths.sim_data(BASELINE_CSV))
    df_base = df_base[df_base["method"] == "systolic_array"]

    sel = lambda d: d[(d["input_length"] == in_len)
                      & (d["output_length"] == out_len)
                      & (d["batch_size"] == 1)]
    df, df_base = sel(df), sel(df_base)
    require_rows(df, f"Vortex ablation at ({in_len},{out_len})")
    require_rows(df_base, f"systolic_array at ({in_len},{out_len})")

    missing = [m for m in METHODS[1:] if m not in set(df["method_label"])]
    if missing:
        raise SystemExit(
            f"Missing bars {missing} at ({in_len},{out_len}).\n"
            f"  Present: {sorted(set(df['method_label']))}\n"
            f"  Run:     ./run_simulation.sh impr_ablation")

    models = sorted(df["model_name"].unique())

    def value(model, method, metric):
        sub = (df_base[df_base["model_name"] == model] if method == "systolic_array"
               else df[(df["model_name"] == model) & (df["method_label"] == method)])
        if not len(sub):
            return np.nan
        val = sub.iloc[0][metric]
        if metric == "total_cycles":
            val = val / CLOCK_HZ
        if metric == "total_energy":
            val = val / ene_scale
        return val

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    bar_width = 0.8 / len(METHODS)
    x_left = np.arange(len(models))
    x_right = x_left + len(models) + 1     # energy group, separated by a gap

    stats = {m: {"cycles": [], "energy": []} for m in METHODS if m != "systolic_array"}

    for mi, method in enumerate(METHODS):
        offset = (mi - (len(METHODS) - 1) / 2) * bar_width
        style = STYLE.get(method, {})

        latency = [value(m, method, "total_cycles") for m in models]
        energy = [value(m, method, "total_energy") for m in models]

        if method != "systolic_array":
            for model, lat, ene in zip(models, latency, energy):
                base_lat = value(model, "systolic_array", "total_cycles")
                base_ene = value(model, "systolic_array", "total_energy")
                if not np.isnan(lat) and not np.isnan(base_lat) and lat > 0:
                    stats[method]["cycles"].append(base_lat / lat)
                if not np.isnan(ene) and not np.isnan(base_ene) and ene > 0:
                    stats[method]["energy"].append(base_ene / ene)

        # The first two bars' overflow labels would collide; nudge them apart.
        text_x_offset = -0.15 if mi == 0 else (0.15 if mi == 1 else 0.0)
        lat_text_y = lat_ylim * 0.09 if mi == 0 else 0.0
        ene_text_y = ene_ylim * 0.09 if mi == 0 else 0.0

        _bars_with_cap(ax1, x_left, latency, bar_width, offset, lat_ylim, style,
                       label=METHOD_NAME.get(method, method),
                       text_x_offset=text_x_offset, text_y_offset=lat_text_y)
        _bars_with_cap(ax2, x_right, energy, bar_width, offset, ene_ylim, style,
                       text_x_offset=text_x_offset, text_y_offset=ene_text_y)

    ax1.set_ylabel("Latency (s)")
    ax2.set_ylabel(f"Energy ({unit})")
    ax1.set_xticks(list(x_left) + list(x_right))
    ax1.set_xticklabels([model_label(m) for m in models] * 2, rotation=20)
    ax1.axvline(len(models), color="black", linewidth=2)

    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    ax2.spines["right"].set_linewidth(2)

    ax1.set_ylim(0, lat_ylim)
    ax2.set_ylim(0, ene_ylim)
    ax1.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.1), frameon=False)

    plt.tight_layout()
    path = _style.save(plt, _paths.figure_path(figure), bbox_inches=None)
    plt.close(fig)

    label = "Attention-Intensive" if workload == "attention" else "Projection-Intensive"
    print(f"  === geomean vs SA, {label} ({in_len},{out_len}) ===")
    for method, v in stats.items():
        print(f"    {METHOD_NAME[method]:<12s} speedup {geo_mean(v['cycles']):7.2f}x"
              f"   energy {geo_mean(v['energy']):7.2f}x")
    return path


if __name__ == "__main__":
    def _extra(p):
        p.add_argument("--workload", choices=["attention", "projection", "both"],
                       default="both",
                       help="attention = BD (512,4096); projection = BE (1,0)")
    args = _paths.parse_args(__doc__, _extra)
    for w in (["attention", "projection"] if args.workload == "both" else [args.workload]):
        plot_BD_BE_hardware_speedup(w)
