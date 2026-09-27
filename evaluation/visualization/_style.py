"""Shared matplotlib style.

The source notebooks each set the same rcParams block from a `glb_font_size`
defined in their first cell. Two values were in use -- 19 for the algorithm
figures, 18 for the hardware ones -- so both are preserved here rather than
unified, since changing a font size changes the rendered PDF and these are
verified against published references.

`pdf.fonttype = 42` embeds TrueType rather than Type 3 fonts. The notebooks did
not set it; several venues require it, and it does not change the layout.

Every figure is also written as SVG next to the PDF, because Markdown previews
cannot display PDFs and the dev notes embed the figures. The PDF remains the
artifact of record -- the SVG is a convenience copy. Disable with --no-svg.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # plot scripts run headless; no notebook display
import matplotlib.pyplot as plt   # noqa: E402

FONT_ALGORITHM = 19    # visual_all_algorithm.ipynb
FONT_HARDWARE = 18     # visual_hardware_v2.ipynb, plot_ablation.ipynb
FONT_MOTIVATION = 18   # plot_motivation.ipynb
FONT_MOTIVATION_COMBINED = 16   # get_sparsity_vs_batch.ipynb


def use(font_size, titles=True):
    """Apply the notebooks' rcParams block.

    `titles`: plot_motivation and visual_all_algorithm also scale axes.titlesize;
    get_sparsity_vs_batch and the hardware notebooks do not. Keeping the
    distinction matters -- it is visible in the rendered output.

    Starts from matplotlib's defaults, so a figure never inherits the previous
    script's rcParams when plot_all.py runs them in one process. Without this,
    AB drew its titles at AA's 18 pt instead of the default 1.2 x 16 pt.
    """
    plt.rcdefaults()   # the backend is exempt, so Agg survives
    params = {
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    if titles:
        params["axes.titlesize"] = font_size
    plt.rcParams.update(params)


_EMIT_SVG = True


def set_svg(enabled):
    """Toggle the companion SVG. Called by _paths.apply_common_args."""
    global _EMIT_SVG
    _EMIT_SVG = bool(enabled)


def save(fig_or_plt, path, **kwargs):
    """Save and report. Every notebook used bbox_inches="tight".

    Also writes an SVG beside the PDF unless disabled: Markdown cannot render a
    PDF, and the dev notes embed these figures.
    """
    kwargs.setdefault("bbox_inches", "tight")
    fig_or_plt.savefig(path, **kwargs)
    print(f"  wrote {path}")

    path = Path(path)
    if _EMIT_SVG and path.suffix == ".pdf":
        svg = path.with_suffix(".svg")
        fig_or_plt.savefig(svg, **kwargs)
        print(f"  wrote {svg}")

    return path
