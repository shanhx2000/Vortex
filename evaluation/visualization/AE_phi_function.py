"""AE -- phi_function_geomean_vs_sparsity.pdf: choice of importance metric.

Geo-mean zero-shot accuracy of Llama2-7B against contextual sparsity, for three
candidate phi functions used to rank codebook entries: l1, l2 and l-infinity
norm. Motivates the norm Vortex actually uses.

Input:  stats/ref/algorithm/phi_function_llama_2_7b_data.csv (committed).

NOTE: this figure is not in the current paper version. Every
`\\input{figtex/accuracy_vs_phi_func}` in the body is commented out; it is slated
to return, so it is ported and indexed like the rest. Its paper figure number is
`xxx` until then.

    python AE_phi_function.py [--out-dir DIR]
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import _paths
import _style

FIGURE = "phi_function_geomean_vs_sparsity.pdf"
INPUT_CSV = "phi_function_llama_2_7b_data.csv"

MODEL_TAG = "llama2_7b_aqlm_rvq"
VEC_LENGTH = 8
EVAL_TASKS = ["arc_easy_acc", "arc_challenge_acc", "copa",
              "openbookqa", "piqa", "winogrande"]

# Legend order, deliberately not alphabetical.
PHI_ORDER = ["l_inf-norm", "l2-norm", "l1-norm"]
PHI_LABEL = {"l2-norm": r"$\ell_2$-norm",
             "l1-norm": r"$\ell_1$-norm",
             "l_inf-norm": r"$\ell_\infty$-norm"}
REQUIRED_COLS = {"model_tag", "vec_length", "phi_function",
                 "sparsity", "task", "value"}


def _geo_mean(vals):
    vals = [float(v) for v in vals if pd.notna(v) and float(v) > 0]
    return float(np.exp(np.mean(np.log(vals)))) if vals else np.nan


def plot_AE_phi_function(model_tag=MODEL_TAG, vec_length=VEC_LENGTH,
                         eval_tasks=None):
    _style.use(_style.FONT_ALGORITHM)

    df = pd.read_csv(_paths.csv_data(INPUT_CSV))
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    df = df[(df["model_tag"] == model_tag) & (df["vec_length"] == vec_length)]
    if len(df) == 0:
        raise ValueError(f"No rows for model_tag={model_tag}, "
                         f"vec_length={vec_length}")

    if eval_tasks is None:
        eval_tasks = EVAL_TASKS
    df = df[df["task"].isin(eval_tasks)]
    df = df[df["task"] != "wikitext_ppl"]     # perplexity is not an accuracy
    if len(df) == 0:
        raise ValueError("No valid task data after filtering.")

    df_geo = (df.groupby(["phi_function", "sparsity"], as_index=False)["value"]
                .agg(lambda x: _geo_mean(list(x)))
                .rename(columns={"value": "geo_mean"}))
    df_geo["geo_mean"] *= 100

    cmap = plt.get_cmap("tab20c")
    color_map = {"l2-norm": cmap(0), "l1-norm": cmap(4), "l_inf-norm": cmap(8)}

    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5), constrained_layout=True)

    for phi in PHI_ORDER:
        sub = df_geo[df_geo["phi_function"] == phi].sort_values("sparsity")
        if len(sub) == 0:
            continue
        x = sub["sparsity"].to_numpy(dtype=float) * 100
        y = sub["geo_mean"].to_numpy(dtype=float)
        mask = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[mask], y[mask]
        if len(x) == 0:
            continue
        ax.plot(x, y, marker="o", markersize=8, linewidth=2.5,
                color=color_map[phi], mfc="none", mec=color_map[phi],
                mew=2.0, label=PHI_LABEL.get(phi, phi))

    ax.set_xlabel("Sparsity (%)")
    ax.set_ylabel("Geo-mean Accuracy (%)")
    ax.set_xlim(left=0.0, right=50)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_edgecolor("black")

    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.22),
              frameon=False)

    path = _style.save(plt, _paths.figure_path(FIGURE))
    plt.close(fig)
    return path


if __name__ == "__main__":
    _paths.parse_args(__doc__)
    plot_AE_phi_function()
