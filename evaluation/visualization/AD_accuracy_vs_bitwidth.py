"""AD -- llama2_7b_accuracy_vs_bitwidth.pdf: accuracy vs bit-width, four tasks.

Llama2-7B zero-shot accuracy on ARC-Easy, ARC-Challenge, PIQA and Winogrande as
effective quantization bit-width drops, comparing GPTQ, AWQ, LLM265, AQLM and
Vortex ("Ours"). Vortex is the only method plotted below 2 bits.

Input:  stats/ref/algorithm/llama2_7b_bitwidth_vs_accuracy.csv (committed).

    python AD_accuracy_vs_bitwidth.py [--out-dir DIR]
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import _paths
import _style

FIGURE = "llama2_7b_accuracy_vs_bitwidth.pdf"
INPUT_CSV = "llama2_7b_bitwidth_vs_accuracy.csv"

TASKS = [("arc_easy_acc", "ARC Easy"),
         ("arc_challenge_acc", "ARC Challenge"),
         ("piqa", "PIQA"),
         ("winogrande", "Winogrande")]
METHODS_ORDER = ["GPTQ", "AWQ", "LLM265", "AQLM", "Ours"]

# All tasks the CSV may carry; the pivot must produce every column the plot
# touches, even when a task is absent, or the lookup below raises.
EXPECTED_TASKS = ["arc_easy_acc", "arc_challenge_acc", "copa",
                  "openbookqa", "piqa", "winogrande", "wikitext_ppl"]


def _map_method(name):
    name = name.lower()
    for needle, label in (("awq", "AWQ"), ("gptq", "GPTQ"), ("llm265", "LLM265"),
                          ("aqlm_rvq", "Ours"), ("aqlm", "AQLM")):
        if needle in name:
            return label
    return None


def load_bitwidth_df(csv_path):
    """Long -> wide. The CSV is stored one row per (model, bitwidth, task)."""
    df = pd.read_csv(csv_path)
    wide = df.pivot_table(index=["model", "bitwidth", "sparsity"],
                          columns="task", values="accuracy").reset_index()
    wide = wide.rename(columns={"model": "Model",
                                "bitwidth": "Effective Quant Bits",
                                "sparsity": "Sparsity"})
    for t in EXPECTED_TASKS:
        if t not in wide.columns:
            wide[t] = None
    return wide.sort_values(["Model", "Effective Quant Bits"])


def plot_AD_accuracy_vs_bitwidth():
    _style.use(_style.FONT_ALGORITHM)

    aqlm_df = load_bitwidth_df(_paths.csv_data(INPUT_CSV))

    df = aqlm_df[(aqlm_df["Model"].str.startswith("llama2_7b"))
                 & (aqlm_df["Model"] != "llama2_7b")].copy()
    df["Method"] = df["Model"].apply(_map_method)
    df = df[df["Method"].notna()]

    # AQLM's sparse variants belong to the "Ours" curve, not the AQLM baseline.
    if "Sparsity" in df.columns:
        df = df[(df["Method"] != "AQLM") | (df["Sparsity"] == 0)]

    colors = [plt.get_cmap("tab20")(i) for i in (0, 4, 8, 6, 2)]

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs = axs.flatten()

    for i, (task_col, task_name) in enumerate(TASKS):
        ax = axs[i]
        for j, method in enumerate(METHODS_ORDER):
            sub = df[df["Method"] == method]
            if task_col not in sub.columns:
                continue
            sub = sub.sort_values("Effective Quant Bits")

            x = sub["Effective Quant Bits"].values.astype(float)
            y = sub[task_col].values.astype(float)
            mask = ~np.isnan(x) & ~np.isnan(y)
            if method == "Ours":
                # Vortex is only claimed in the sub-2-bit regime.
                mask = mask & (x < 2)
            x, y = x[mask], y[mask] * 100
            if method == "AWQ":
                x[np.abs(x - 4) < 0.1] = 4     # snap 3.99-ish to exactly 4
            if len(x) == 0:
                continue

            ax.plot(x, y, marker="o", markersize=8, mfc="none",
                    mec=colors[j], mew=3.0, color=colors[j],
                    linewidth=2, label=method)

        ax.set_title(task_name)
        ax.set_xlim(0.5, 4.2)
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
            spine.set_edgecolor("black")

    fig.supxlabel("Bitwidth", y=0.08)
    fig.supylabel("Accuracy (%)")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(METHODS_ORDER),
               bbox_to_anchor=(0.5, 1.05), frameon=False)

    plt.tight_layout()
    path = _style.save(plt, _paths.figure_path(FIGURE))
    plt.close(fig)
    return path


if __name__ == "__main__":
    _paths.parse_args(__doc__)
    plot_AD_accuracy_vs_bitwidth()
