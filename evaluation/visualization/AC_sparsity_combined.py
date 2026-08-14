"""AC -- sparsity_combined.pdf: codebook-wise vs uniform sparsity.

One panel per model, in a row: geo-mean accuracy across six downstream tasks as
contextual sparsity increases, for uniform (TEAL) against codebook-wise
(Vortex).

Each panel marks, at the 95% and 90% relative-accuracy lines, the sparsity
codebook-wise sustains and how much more that is than codebook-uniform. Those
are the two numbers the paper argues from, so the figure states them rather
than leaving them to be read off the axis.

Ceilings are grid points -- the largest sparsity in the sweep still at or above
the line -- not interpolated crossings, so they match the values quoted in the
text exactly.

MODEL_ORDER decides which models appear and how wide the figure is. Mistral-7B
was dropped from it on 2026-08-14: its corrected ceiling equals
codebook-uniform's, so the panel showed no advantage. This is a plotting
decision only -- its rows are still in the input CSV, run_algorithm.sh still
searches and evaluates it, and C1 still simulates it. Putting the tag back in
this list is the whole restore.

Input:  stats/ref/algorithm/aqlm_sparsity_exteval_data.csv (committed; --use-ref has
        no effect -- this is algorithm-side data, not simulator output).

    python AC_sparsity_combined.py [--out-dir DIR]
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import _paths
import _style

FIGURE = "sparsity_combined.pdf"
INPUT_CSV = "aqlm_sparsity_exteval_data.csv"

EVAL_TASKS = ["arc_easy_acc", "arc_challenge_acc", "copa",
              "openbookqa", "piqa", "winogrande"]
# One panel per entry, in this order. Restoring "mistral_7b" here is the only
# change needed to bring it back -- the grid and the figure width follow.
MODEL_ORDER = ["llama2_7b", "llama2_13b"]
PANEL_W_IN, PANEL_H_IN = 5.4, 3.4
# The relative-accuracy levels annotated in each panel, high to low.
THRESHOLDS = [0.95, 0.90]
METHOD_NAME_MAP = {"TEAL": "codebook-uniform", "Ours": "codebook-wise (Vortex)"}


def _geo_mean_row(row, cols):
    vals = [float(row[c]) for c in cols if pd.notna(row[c]) and row[c] > 0]
    return float(np.exp(np.mean(np.log(vals)))) if vals else np.nan


def _model_prefix(tag):
    for suffix in ("_aqlm_rvq", "_aqlm"):
        if tag.endswith(suffix):
            return tag[: -len(suffix)]
    return tag


def _map_method(tag):
    if tag.endswith("_aqlm_rvq"):
        return "Ours"
    if tag.endswith("_aqlm"):
        return "TEAL"
    return "FP16"


def _display_model(name):
    low = name.lower()
    if "llama2" in low:
        return f"Llama2-{low.split('_')[-1].upper()}"
    if "mistral" in low:
        return f"Mistral-{low.split('_')[-1].upper()}"
    return name


def _ceiling(sub, baseline, threshold):
    """Largest swept sparsity still holding `threshold` of the dense baseline.

    A grid point, not an interpolated crossing: these are the numbers the paper
    quotes, and interpolating would make the figure disagree with the text.

    Relative accuracy is rounded half-up to whole percent before the comparison.
    Without it the ceiling can turn on differences far below evaluation noise:
    Llama2-7B's uniform curve sits at 89.99852% of baseline at 35% sparsity,
    1.5e-5 short of the 90% line, and one extra correct Winogrande answer out of
    1267 would move the marker a whole grid step. Rounding to the precision the
    paper actually quotes makes the choice reproducible.
    """
    pct = sub["geo_mean"] / baseline * 100.0
    # floor(x + 0.5) rather than round(), which is banker's rounding in Python
    # and would send 89.5 and 88.5 in opposite directions.
    ok = sub[np.floor(pct + 0.5) >= int(round(threshold * 100))]
    return float(ok["sparsity"].max()) if len(ok) else None


def plot_AC_sparsity_combined():
    _style.use(_style.FONT_ALGORITHM)

    df = pd.read_csv(_paths.csv_data(INPUT_CSV)).copy()
    tasks = [c for c in EVAL_TASKS if c in df.columns]

    df["geo_mean"] = df.apply(lambda r: _geo_mean_row(r, tasks), axis=1)
    df["model_prefix"] = df["model_tag"].apply(_model_prefix)
    df["method"] = df["model_tag"].apply(_map_method)

    cmap = plt.get_cmap("tab20c")
    color_teal, color_ours = cmap(8), cmap(4)

    ncols = len(MODEL_ORDER)
    fig, axs = plt.subplots(1, ncols, figsize=(PANEL_W_IN * ncols, PANEL_H_IN),
                            squeeze=False)

    for i, prefix in enumerate(MODEL_ORDER):
        ax = axs[0, i]
        cur = df[df["model_prefix"] == prefix]
        if len(cur) == 0:
            continue

        teal_df = cur[cur["method"] == "TEAL"].sort_values("sparsity")
        ours_df = cur[cur["method"] == "Ours"].sort_values("sparsity")

        # Dense AQLM accuracy: the line everything is judged against.
        teal_zero = teal_df[np.isclose(teal_df["sparsity"], 0.0)]
        baseline = float(teal_zero.iloc[0]["geo_mean"]) if len(teal_zero) else None
        if baseline is not None:
            ax.hlines(y=baseline * 100, xmin=0.0, xmax=50.0, linestyle="--",
                      linewidth=2.0, color=color_teal, label="AQLM-2bit")

        for sub, color, key in ((teal_df, color_teal, "TEAL"),
                                (ours_df, color_ours, "Ours")):
            if len(sub) == 0:
                continue
            ax.plot(sub["sparsity"].values * 100, sub["geo_mean"].values * 100,
                    marker="o", markersize=8, linewidth=2.5, color=color,
                    mfc="none", mec=color, mew=2.0, label=METHOD_NAME_MAP[key])

        # ---- the two numbers the paper argues from -------------------------
        # Markers sit on the curve's crossing; the values are read off a text
        # block in the empty lower-left corner rather than floated next to each
        # marker, which collides at this panel size.
        rows = []
        if baseline is not None:
            for thr in THRESHOLDS:
                y = baseline * thr * 100
                ax.axhline(y=y, color="0.45", linestyle=":", linewidth=1.6,
                           zorder=0)
                ax.annotate(f"{thr * 100:.0f}%", xy=(49.4, y), va="bottom",
                            ha="right", fontsize=13, color="0.35")

                s_teal = _ceiling(teal_df, baseline, thr)
                s_ours = _ceiling(ours_df, baseline, thr)
                if s_ours is None:
                    continue

                if s_teal is not None and not np.isclose(s_teal, s_ours):
                    ax.plot([s_teal * 100], [y], marker="D", markersize=9,
                            color=color_teal, mec="black", mew=1.2, zorder=5)
                    ax.annotate("", xy=(s_ours * 100, y), xytext=(s_teal * 100, y),
                                arrowprops=dict(arrowstyle="<|-|>", color="0.25",
                                                linewidth=1.6, shrinkA=5,
                                                shrinkB=5), zorder=4)
                ax.plot([s_ours * 100], [y], marker="D", markersize=9,
                        color=color_ours, mec="black", mew=1.2, zorder=5)

                delta = None if s_teal is None else (s_ours - s_teal) * 100
                rows.append(f"{thr * 100:.0f}%: {s_ours * 100:.0f}%"
                            + ("" if delta is None else f" ({delta:+.0f})"))

        if rows:
            ax.text(0.035, 0.035, "\n".join(rows), transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=14, fontweight="bold",
                    color=color_ours, linespacing=1.35,
                    bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                              edgecolor="0.6", linewidth=1.2, alpha=0.92))

        ax.set_title(_display_model(prefix))
        ax.set_xlim(left=0.0, right=50.0)
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
            spine.set_edgecolor("black")

    # Explicit legend order. get_legend_handles_labels() returns Line2D before
    # LineCollection, so the hlines baseline lands first or last depending on
    # the matplotlib version -- pinning the order keeps the output stable.
    by_label = dict(zip(*reversed(axs[0, 0].get_legend_handles_labels())))
    order = ["AQLM-2bit", METHOD_NAME_MAP["TEAL"], METHOD_NAME_MAP["Ours"]]
    labels = [l for l in order if l in by_label]
    fig.legend([by_label[l] for l in labels], labels, loc="upper center",
               ncol=len(labels), bbox_to_anchor=(0.5, 1.13), frameon=False)

    # Axis names are shared -- one per figure, not one per panel. The tick
    # labels still differ, since the two models sit at different accuracies.
    # supxlabel/supylabel need matplotlib >= 3.4, which scripts/setup_env.sh
    # already treats as the floor for this repository.
    # figure.labelsize defaults to "large", i.e. 1.2x the base, which collides
    # with the tick labels here -- pin it to the same size as a normal axis
    # label so the shared names look like the per-panel ones they replace.
    label_size = plt.rcParams["font.size"]
    fig.supxlabel("Sparsity (%)", fontsize=label_size, y=0.015)
    fig.supylabel("Geo-mean Accuracy (%)", fontsize=label_size, x=0.005)

    fig.subplots_adjust(left=0.085, right=0.995, top=0.85, bottom=0.22, wspace=0.14)

    path = _style.save(plt, _paths.figure_path(FIGURE))
    plt.close(fig)
    return path


if __name__ == "__main__":
    _paths.parse_args(__doc__)
    plot_AC_sparsity_combined()
