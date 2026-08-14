"""AB -- motivation_combined.pdf: the cost of VQ, and why sparsity shrinks with batch.

Two panels: (a) measured decode latency of FP16 vs AQLM on three models, showing
that vector quantization is ~2x slower in software; (b) contextual sparsity of a
layer as batch size grows, showing the opportunity collapsing from 30% to 5%.
Together they set up the problem Vortex solves.


The numbers are measurements transcribed into the notebook rather than read from
a file, so they are inlined here too -- see MEASUREMENTS below for provenance.

    python AB_motivation_combined.py [--use-simulation] [--out-dir DIR]
"""
import numpy as np
import matplotlib.pyplot as plt

import _paths
import _style

FIGURE = "motivation_combined.pdf"

# (a) End-to-end decode latency in ms, FP16 vs AQLM, measured on GPU.
# Transcribed from the source notebook; not reproducible from this artifact.
MEASUREMENTS = {
    "Llama2-7B":  {"FP": 123.6109, "AQLM": 233.1239},
    "Llama2-13B": {"FP": 137.1450, "AQLM": 260.5430},
    "Mistral-7B": {"FP": 112.1243, "AQLM": 219.9454},
}

# (b) Sparsity of one layer's activations at increasing batch size. Union of
# per-token sparsity patterns shrinks fast -- this is the point of the panel.
BLOCK_SIZES = [1, 2, 4, 8, 16]
LAYER_BLOCK_SPARSITIES = [0.3036, 0.1646, 0.0911, 0.0625, 0.0506]


def plot_AB_motivation_combined():
    _style.use(_style.FONT_MOTIVATION_COMBINED, titles=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ---------------- (a) VQ slowdown ----------------
    ax = axes[0]
    groups = list(MEASUREMENTS.keys())
    x = np.arange(len(groups))
    width = 0.35
    cmap = plt.get_cmap("tab10")

    for i, label in enumerate(["FP", "AQLM"]):
        ax.bar(x + (i - 0.5) * width,
               [MEASUREMENTS[g][label] for g in groups],
               width=width,
               label="FP16" if label == "FP" else label,
               color=cmap(i), edgecolor="black", linewidth=1.5,
               hatch="//" if label == "AQLM" else None, zorder=3)

    for i, g in enumerate(groups):
        fp, vq = MEASUREMENTS[g]["FP"], MEASUREMENTS[g]["AQLM"]
        ax.text(x[i] + width / 2, vq * 1.02, f"{vq / fp:.1f}×",
                ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("(a) VQ Slowdown")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.set_ylim(bottom=0, top=300)
    ax.legend(loc="lower right")

    # ---------------- (b) sparsity vs batch ----------------
    ax = axes[1]
    sparsities = np.array(LAYER_BLOCK_SPARSITIES)

    ax.plot(BLOCK_SIZES, sparsities * 100, marker="o", linewidth=2)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Sparsity (%)")
    ax.set_title("(b) Sparsity vs. Batch Size")
    ax.set_xscale("log", base=2)
    ax.set_xticks(BLOCK_SIZES)
    ax.set_xticklabels([str(bs) for bs in BLOCK_SIZES])
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i, bs in enumerate(BLOCK_SIZES):
        ax.text(bs, sparsities[i] * 100 + 1, f"{sparsities[i] * 100:.1f}",
                ha="center", fontsize=16)

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.set_ylim(0, max(sparsities * 100) * 1.3)

    plt.tight_layout()
    path = _style.save(plt, _paths.figure_path(FIGURE))
    plt.close(fig)
    return path


if __name__ == "__main__":
    _paths.parse_args(__doc__)
    plot_AB_motivation_combined()
