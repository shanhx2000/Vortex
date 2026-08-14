"""Per-module power/area breakdown for Vortex.

Two stages:
  1. profile()  - simulate each model and dump raw per-op Stats to JSON
  2. eval()     - aggregate those into a geomean power breakdown

Combined with the static area breakdown, this writes
stats/simulation/vortex_power_area_breakdown.json, the input to figure BC (pie chart).

    python get_raw_stats.py            # profile if needed, then breakdown
    python get_raw_stats.py --profile  # force re-profiling
"""
from __future__ import annotations
import json
import copy

import argparse
from typing import Any, Dict, List
import time

import _paths  # noqa: F401  (sets up sys.path)
from _paths import HW_CONFIG_DIR

from stats import Stats, dict_to_stats
from model_runner import (
    CATEGORY_ORDER,
    DEFAULT_ATTENTION_CONFIG,
    available_methods,
    build_model_workload,
    run_method_workload,
    validate_workload,
    resolve_phase_method,
    get_hw_config,
    create_simulator,
)
from main import (
    resolve_methods,
    build_attention_config,
    build_runtime_args,
)
from evaluator import Evaluator

def _stats_to_dict(stats):
    """A Stats object as a plain dict, without the summary pass."""
    return {
        k: v
        for k, v in stats.__dict__.items()
        if not k.startswith("_")
    }

def _run_phase_raw(
    ops,
    method,
    phase,
    *,
    args=None,
):
    """_run_phase, but returning the raw stats instead of a summary."""
    phase_method = resolve_phase_method(method, phase)
    args["phase"] = phase

    hw_config = get_hw_config(phase_method)
    simulator = create_simulator(phase_method, hw_config)

    if not ops:
        empty_stats = Stats(name=f"{phase_method}.empty")
        return _stats_to_dict(empty_stats)

    phase_ops = copy.deepcopy(ops)

    result = simulator.run_model(
        phase_ops,
        method=phase_method,
        args=args,
        hw_config=hw_config,
    )

    return _stats_to_dict(result["stats"])

def collect_vortex_raw_stats(
    model_names,
    save_path,
    input_length = 512,
    output_length = 4096,
    batch_size = 1,
    quant_scheme = "AQLM|CQ",
    sparsity = 0.3,
):
    results_all = {}

    for model_name in model_names:
        print(f"\n[Collect RAW STATS] {model_name}")

        # ==== fixed parameters ====


        # ==== build the run args ====
        class Args:
            pass

        args = Args()
        args.model_name = model_name
        args.methods = ["vortex"]
        args.input_length = input_length
        args.output_length = output_length
        args.batch_size = batch_size
        args.default_dtype = "fp16"
        args.compute_mode = "lut_based"
        args.processed_sparsity = f"RVQ_{sparsity}"
        args.force_dataflow = None
        args.quant_scheme = quant_scheme

        attention_config = build_attention_config(args)
        runtime_args = build_runtime_args(args)

        sparsity_config = {
            "method": "RVQ",
            "model_name": model_name,
            "contextual_sparsity": sparsity,
        }

        # ==== workload ====
        workload = build_model_workload(
            model_name=model_name,
            input_length=input_length,
            output_length=output_length,
            batch_size=batch_size,
            default_dtype="fp16",
            attention_config=attention_config,
            sparsity_config=sparsity_config,
        )


        # ==== quant ====
        from quantization import apply_quant_methods

        schemes = quant_scheme.split("|")
        for scheme in schemes:
            if scheme not in ["CQ"]:
                apply_quant_methods(workload.prefill_ops, [scheme])
            apply_quant_methods(workload.decode_ops, [scheme])

        # ==== run the phase by hand so the raw stats survive ====
        assert len(args.methods) == 1, "For raw stats collection, only support one method at a time"
        method = args.methods[0]
        assert method == "vortex", "For raw stats collection, only support vortex method"

        prefill = _run_phase_raw(
            workload.prefill_ops,
            method,
            "prefill",
            args=runtime_args,
        )

        decode = _run_phase_raw(
            workload.decode_ops,
            method,
            "decode",
            args=runtime_args,
        )

        results_all[model_name] = {
            "prefill": prefill,
            "decode": decode,
        }

    # ==== save ====
    with open(save_path, "w") as f:
        json.dump(results_all, f, indent=2)

    print(f"\nSaved to {save_path}")


import json
import math


def geometric_mean(values):
    values = [v for v in values if v > 0]
    if not values:
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


def print_buffer_dyn_by_instance(self, hw_config, stats):
    cfg = self.area_power_energy_config
    buffers = hw_config["buffers"]
    accessed = {
        k for k in (set(stats.reads) | set(stats.writes))
        if k != "dram"
    }
    total_dyn = 0
    stats.mem_namespace = sorted(accessed)
    for buffer_name in stats.mem_namespace:
        if buffer_name == "buffer":
            print(f"[Skip] aggregated buffer namespace: {buffer_name}")
            continue

        if stats.reads.get(buffer_name, 0) == 0 and stats.writes.get(buffer_name, 0) == 0:
            print(f"[Skip] {buffer_name} has no accesses")
            continue

        # ===== Step 3: locate the matching hw buffer =====
        matched = []

        # ===== Step 4: s_cap is special -- it is costed against the LUT SRAM =====
        if "s_cap" in buffer_name:
            read_bits = stats.reads.get(buffer_name, 0)
            write_bits = stats.writes.get(buffer_name, 0)

            if read_bits == 0 and write_bits == 0:
                print(f"[Skip] {buffer_name} has no accesses")
                continue

            # use the LUT's own SRAM config
            cfg_buf = cfg["sram_256x16bits_1rw"]

            dyn = (
                read_bits * cfg_buf["read_energy"] +
                write_bits * cfg_buf["write_energy"]
            ) * 1e-9

            total_dyn += dyn

            print(
                f"{buffer_name:15s} | matched=LUT_SRAM "
                f"| reads={read_bits:<10} writes={write_bits:<10} "
                f"| dyn={dyn:.4e} J"
            )

            continue

        # case 1: exact name match (e.g. s_act_lt)
        assert buffer_name in buffers, f"Buffer '{buffer_name}' not found in hw_config buffers"
        matched = [buffer_name]


        # ===== Step 5: sanity check =====
        if not matched:
            print(f"[WARN] No HW buffer found for '{buffer_name}'")
            continue

        # ===== Step 6: read the access counts =====
        read_bits = stats.reads.get(buffer_name, 0)
        write_bits = stats.writes.get(buffer_name, 0)

        if read_bits == 0 and write_bits == 0:
            continue

        # ===== Step 7: compute energy =====
        dyn_energy_total = 0.0

        for name in matched:
            spec = buffers[name]

            if self._is_zero_buffer(spec):
                continue

            block_bytes = spec["block_size"]
            size_per_bank = spec["size"] // spec["num"]
            num_entries = size_per_bank // block_bytes

            key = f"sram_{num_entries}x{block_bytes*8}bits_1rw"
            if key not in cfg:
                raise KeyError(f"Missing buffer config: {key}")

            cfg_buf = cfg[key]

            dyn = (
                read_bits * cfg_buf["read_energy"] +
                write_bits * cfg_buf["write_energy"]
            ) * 1e-9

            dyn_energy_total += dyn * spec["num"]

        total_dyn += dyn_energy_total

        print(
            f"{buffer_name:15s} | matched={matched} "
            f"| reads={read_bits:<10} writes={write_bits:<10} "
            f"| dyn={dyn_energy_total:.4e} J"
        )

    print("\nTotal Buffer Dynamic Energy:", f"{total_dyn:.4e} J")

def compute_power_breakdown_for_models(model_name_path_pairs):
    results = {}

    hw_config = get_hw_config("vortex")
    simulator = create_simulator("vortex", hw_config)
    for model_name, path in model_name_path_pairs:
        print(f"[Power] {model_name}")

        # ===== load stats =====
        with open(path) as f:
            raw = json.load(f)

        # ===== prefill + decode combined =====
        prefill = dict_to_stats(raw[model_name]["prefill"])
        decode = dict_to_stats(raw[model_name]["decode"])
        stats = prefill + decode



        evaluator = Evaluator(buffer_specs=hw_config["buffers"])

        print_buffer_dyn_by_instance(evaluator, hw_config=hw_config, stats=stats)

        # Vortex's own energy model, not HWSimBase's.
        energy = evaluator.get_energy( hw_config=hw_config, stats=stats, frequency=hw_config["general"]["frequency"] )

        total_energy = energy["total_energy (J)"]
        total_power = energy["total_power (W)"]

        if total_energy == 0:
            results[model_name] = {
                "MXU": 0,
                "LUT": 0,
                "VPU": 0,
                "Buffers": 0,
                "DRAM": 0,
                "Others": 0,
            }
            continue

        # ===== breakdown =====
        core_bd = energy["energy_breakdown"]["core"]["core_breakdown"]
        dram_energy = energy["energy_breakdown"]["dram"]["dram_energy (J)"]

        # Keyed 'buffer_energy (J)' -- note the space before the unit, as with
        # every other key in this dict.
        buffer_energy = energy["energy_breakdown"]["buffer"]["buffer_energy (J)"]

        core_energy = energy["energy_breakdown"]["core"]["core_energy (J)"]

        # These three keys must match hw_vortex.py's core_breakdown exactly; a
        # typo silently reads 0.0 and lands the difference in 'Others'.
        mxu_energy = core_bd.get("mxu", 0.0)
        lut_energy = core_bd.get("lut", 0.0)
        vpu_energy = core_bd.get("vpu", 0.0)

        others_energy = core_energy - mxu_energy - lut_energy - vpu_energy

        def to_power(e):
            return (e / total_energy) * total_power

        results[model_name] = {
            "MXU": to_power(mxu_energy),
            "LUT": to_power(lut_energy),
            "VPU": to_power(vpu_energy),
            "Buffers": to_power(buffer_energy),
            "DRAM": to_power(dram_energy),
            "Others": to_power(others_energy),
        }

    # ===== geometric mean =====
    keys = ["MXU", "LUT", "VPU", "Buffers", "DRAM", "Others"]

    geo = {}
    model_names = list(results.keys())
    for k in keys:
        geo[k] = geometric_mean([results[m][k] for m in model_names])

    return {
        "per_model": results,
        "geomean": geo,
    }


def profile():
    for model_name in ["mistral_7b", "llama_2_13b", "llama-2-7b"]:
        collect_vortex_raw_stats(
            model_names=[model_name],
            save_path=str(_paths.results_dir() / f"vortex_raw_stats_{model_name}_i512o4096.json"),
            input_length=512,
            output_length=4096,
            batch_size=1,
            quant_scheme="AQLM|CQ",
            sparsity=0.3,
        )

def test_profile():
    for model_name in ["llama-2-7b"]:
        collect_vortex_raw_stats(
            model_names=[model_name],
            save_path=str(_paths.results_dir() / f"vortex_raw_stats_{model_name}_i1o1.json"),
            input_length=1,
            output_length=1,
            batch_size=1,
            quant_scheme="AQLM|CQ",
            sparsity=0.3,
        )

def test_eval():
    model_name = "llama-2-7b"
    print(
        compute_power_breakdown_for_models(
            [
                (model_name, str(_paths.results_dir() / f"vortex_raw_stats_{model_name}_i1o1.json"))
            ]))


def eval():
    # ["mistral_7b", "llama_2_13b", "llama-2-7b"]
    model_names = ["mistral_7b", "llama_2_13b", "llama-2-7b"]
    model_name_path_pairs = [
        (model_name, str(_paths.results_dir() / f"vortex_raw_stats_{model_name}_i512o4096.json"))
        for model_name in model_names
    ]

    res = compute_power_breakdown_for_models(model_name_path_pairs)
    print(res)
    return res
    pass

def run_vortex_gemm_experiment5(config_json_path, area_power_energy_csv_path):
    import json
    from evaluator import Evaluator

    with open(config_json_path, "r") as f:
        hw_config = json.load(f)

    evaluator = Evaluator(hw_config["buffers"], area_power_energy_csv_path)

    total_area = evaluator.get_area(hw_config)
    area_breakdown = evaluator.get_area_breakdown(hw_config)

    # ===== core parts =====
    MXU = area_breakdown["core"]["mxu"]
    LUT = area_breakdown["core"]["lut"]
    VPU = area_breakdown["core"]["vpu"]
    Buffers = area_breakdown["buffer"]["selected_total"]

    Others = total_area - MXU - LUT - VPU - Buffers

    # ===== absolute =====
    abs_breakdown = {
        "MXU": MXU,
        "LUT": LUT,
        "VPU": VPU,
        "Buffers": Buffers,
        "Others": Others,
        "Total": total_area,
    }

    # ===== percentage =====
    pct_breakdown = {
        k: (v / total_area * 100 if total_area > 0 else 0.0)
        for k, v in abs_breakdown.items()
        if k != "Total"
    }

    return {
        "absolute": abs_breakdown,
        "percentage": pct_breakdown,
    }


PROFILE_MODELS = ["mistral_7b", "llama_2_13b", "llama-2-7b"]


def _raw_stats_path(model_name):
    return _paths.results_dir() / f"vortex_raw_stats_{model_name}_i512o4096.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Re-run profiling even if raw stats already exist.",
    )
    parser.add_argument(
        "--out-dir", default=None, metavar="DIR",
        help="write the raw stats and the power/area breakdown here instead of "
             "stats/simulation/ (run_simulation.sh -s NAME passes the session "
             "directory)",
    )
    cli_args = parser.parse_args()
    _paths.set_results_dir(cli_args.out_dir)

    # Stage 1: raw per-op stats. eval() needs these, so generate them when
    # missing rather than failing with a bare FileNotFoundError.
    missing = [m for m in PROFILE_MODELS if not _raw_stats_path(m).exists()]
    if cli_args.profile or missing:
        if missing and not cli_args.profile:
            print(f"[profile] missing raw stats for: {', '.join(missing)}")
        profile()

    # Stage 2: power geomean + static area breakdown.
    power_res = eval()
    power_geo = power_res["geomean"]
    area_res = run_vortex_gemm_experiment5(
        config_json_path=str(HW_CONFIG_DIR / "vortex.json"),
        area_power_energy_csv_path=str(HW_CONFIG_DIR / "power_energy_config.json"),
    )

    final = {
        "power_W": power_geo,                     # W
        "area_mm2": area_res["absolute"],         # mm^2
        "area_pct": area_res["percentage"],       # %
    }
    output_path = _paths.results_dir() / "vortex_power_area_breakdown.json"
    # ===== dump =====
    with open(output_path, "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nSaved to {output_path}")
    print(json.dumps(final, indent=2))

