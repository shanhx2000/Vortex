"""BC -- vortex_pie_chart.pdf: where Vortex's area and power go.

Two pies: silicon area by module, and power by module including DRAM. VPU and
"Others" are merged into one slice, and the LUT array is labelled LUU, matching
the paper's terminology.

Data:   vortex_power_area_breakdown.json (group: power_area)

    python BC_power_area_pie.py [--use-simulation] [--out-dir DIR]
"""
import json

import matplotlib.pyplot as plt

import _paths
import _style

FIGURE = "vortex_pie_chart.pdf"
INPUT_JSON = "vortex_power_area_breakdown.json"

ORDER_RAW = ["MXU", "LUT", "VPU", "Buffers", "Others", "DRAM"]
ORDER = ["MXU", "LUU", "VPU&Others", "Buffers", "DRAM"]
FONT = 16

# Percentage-label nudges, tuned per slice so the text clears the wedge edges.
# Pure layout; index order follows ORDER.
PCT_OFFSETS_AREA = [(0.05, 0.05), (0.18, -0.03), (0.1, 0.0), (0.1, 0.0)]
PCT_OFFSETS_POWER = [(0.0, 0.0), (0.0, 0.02), (0.0, -0.05), (0.0, -0.15), (0.0, 0.0)]


def _merge_vpu_others(d, is_area=False):
    """Collapse VPU+Others into one slice and rename LUT -> LUU."""
    merged = {}
    for k in ORDER:
        if is_area and k == "DRAM":
            continue        # DRAM is off-chip: it has power but no die area
        if k == "LUU":
            merged[k] = d.get("LUT", 0)
        elif k == "VPU&Others":
            merged[k] = d.get("VPU", 0) + d.get("Others", 0)
        else:
            merged[k] = d.get(k, 0)
    return merged


def _autopct(values):
    def fmt(pct):
        return f"{pct * sum(values) / 100.0:.1f} ({pct:.0f}%)"
    return fmt


def _nudge(autotexts, offsets):
    for text, (dx, dy) in zip(autotexts, offsets or []):
        x, y = text.get_position()
        text.set_position((x + dx, y + dy))


def plot_BC_power_area_pie(bottom_margin=0.01, title_pad=2):
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    with open(_paths.sim_data(INPUT_JSON)) as fh:
        result = json.load(fh)

    power = {k: result["power_W"][k] for k in ORDER_RAW if k in result["power_W"]}
    area = {k: v for k, v in result["area_mm2"].items()
            if k != "Total" and k in ORDER_RAW and k != "DRAM"}
    area = {k: area[k] for k in ORDER_RAW if k in area}

    area = _merge_vpu_others(area, is_area=True)
    power = _merge_vpu_others(power)

    cmap = plt.get_cmap("tab20c")
    colors = [cmap(i) for i in range(len(ORDER))]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    _, _, auto_area = axes[0].pie(
        list(area.values()), labels=list(area.keys()),
        autopct=_autopct(list(area.values())), startangle=90,
        colors=colors[:len(area)],
        wedgeprops=dict(edgecolor="black", linewidth=2),
        textprops=dict(fontsize=FONT), pctdistance=0.60, labeldistance=1.03)

    _, _, auto_power = axes[1].pie(
        list(power.values()), labels=list(power.keys()),
        autopct=_autopct(list(power.values())), startangle=90,
        colors=colors[:len(power)],
        wedgeprops=dict(edgecolor="black", linewidth=2),
        textprops=dict(fontsize=FONT), pctdistance=0.60)

    _nudge(auto_area, PCT_OFFSETS_AREA)
    _nudge(auto_power, PCT_OFFSETS_POWER)

    axes[0].set_title(f"Area Breakdown (mm²)\nTotal: {sum(area.values()):.1f} mm²",
                      fontsize=FONT, pad=title_pad)
    axes[1].set_title(f"Power Breakdown (W)\nTotal: {sum(power.values()):.1f} W",
                      fontsize=FONT, pad=title_pad)

    print(f"  area  {sum(area.values()):.2f} mm2  " +
          " ".join(f"{k}={v:.2f}" for k, v in area.items()))
    print(f"  power {sum(power.values()):.2f} W    " +
          " ".join(f"{k}={v:.2f}" for k, v in power.items()))

    plt.tight_layout()
    plt.subplots_adjust(bottom=bottom_margin)
    plt.tight_layout(rect=[0, bottom_margin, 1, 1])

    path = _style.save(plt, _paths.figure_path(FIGURE), pad_inches=0.1)
    plt.close(fig)
    return path


if __name__ == "__main__":
    _paths.parse_args(__doc__)
    plot_BC_power_area_pie()
