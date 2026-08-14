"""Per-op energy estimates for baseline kernels.

DEAD IN PRACTICE: every value written here is recomputed and overwritten before
it reaches any output. `baselines/common.py` calls `update_stats_energy` per op,
but the reported numbers come from a final `get_energy` call on the merged
stats, which recomputes `dram_energy`, `buffer_energy`, `static_energy` and
`core_energy` from scratch.

Kept because the baseline kernels call it, but do not read numbers out of it.
Note `estimate_static_energy` returns power x nanoseconds, not power x seconds,
which is inconsistent with every other energy path -- harmless only because the
result is discarded.
"""
from __future__ import annotations

from typing import Any, Dict

from stats import Stats


def estimate_dram_energy(stats: Stats, config: Dict[str, Any]) -> float:
    dram_cfg = config.get("general", {}).get("dram", {})
    read_e = float(dram_cfg.get("power", {}).get("energy_per_bits", 0.0))
    write_e = float(dram_cfg.get("power", {}).get("energy_per_bits", 0.0))
    return (stats.reads.get("dram", 0) * read_e) + (stats.writes.get("dram", 0) * write_e)


def estimate_buffer_energy(stats: Stats, config: Dict[str, Any]) -> float:
    total = 0.0
    buffers = config.get("buffers", {})
    default_e = 0.0
    for _, buf_cfg in buffers.items():
        default_e += float(buf_cfg.get("power", {}).get("energy_per_bits", 0.0))
    default_e = default_e / max(len(buffers), 1)

    buffer_alias = {
        "buffer": [],
        "s_act": ["s_act"],
        "s_wgt": ["s_wgt", "s_weight"],
        "s_codebook": ["s_codebook", "s_parameter"],
        "s_cap": ["s_cap", "s_output"],
        "s_parameter": ["s_parameter"],
        "s_scale": ["s_scale", "s_parameter"],
        "s_output": ["s_output"],
        "s_double_cap": ["s_double_cap", "s_output"],
    }

    for space, cfg_names in buffer_alias.items():
        if not cfg_names:
            energy_per_bits = default_e
        else:
            energy_per_bits = default_e
            for cfg_name in cfg_names:
                if cfg_name in buffers:
                    energy_per_bits = float(buffers.get(cfg_name, {}).get("power", {}).get("energy_per_bits", default_e))
                    break
        total += stats.reads.get(space, 0) * energy_per_bits
        total += stats.writes.get(space, 0) * energy_per_bits
    return total


def estimate_static_energy(stats: Stats, config: Dict[str, Any]) -> float:
    freq_mhz = float(config.get("general", {}).get("frequency", 0.0))
    if freq_mhz <= 0:
        return 0.0
    cycle_time_ns = 1000.0 / freq_mhz
    active_time_ns = stats.total_cycles * cycle_time_ns

    total_static_power = 0.0
    for _, module_cfg in config.get("modules", {}).items():
        power = module_cfg.get("power", {})
        total_static_power += float(power.get("static", 0.0))
    for _, buffer_cfg in config.get("buffers", {}).items():
        power = buffer_cfg.get("power", {})
        total_static_power += float(power.get("static", 0.0))
    total_static_power += float(config.get("general", {}).get("dram", {}).get("power", {}).get("static", 0.0))
    return total_static_power * active_time_ns


def update_stats_energy(stats: Stats, config: Dict[str, Any], core_dynamic_energy: float = 0.0) -> Stats:
    stats.finalize()
    stats.dram_energy = estimate_dram_energy(stats, config)
    stats.buffer_energy = estimate_buffer_energy(stats, config)
    stats.static_energy = estimate_static_energy(stats, config)
    stats.core_energy += float(core_dynamic_energy)
    return stats
