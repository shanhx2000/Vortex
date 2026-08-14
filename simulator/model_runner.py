"""Builds a workload from a model config and runs it through one simulator.

`build_model_workload` expands a model into an op list: one pass for prefill,
then one op list *per generated token* for decode -- which is why simulation
cost grows with decode length rather than being amortised.

`run_method_workload` runs prefill and decode through a simulator, summarises
per category (linear / attention_qk / attention_sv / sfu / other), and applies
the final `get_energy` pass whose output is what lands in the result CSVs.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time
from hw_base import HWSimBase
from hw_vortex import HWVortex
from hw_configs import get_hw_config, list_hw_presets
from networks import create_network
from quantization import apply_quant_methods
from stats import Stats


DEFAULT_ATTENTION_CONFIG: Dict[str, Any] = {
    "quantize_kv": False,
    "kv_cache_loc": "dram",
}


CATEGORY_ORDER = ["linear", "attention_qk", "attention_sv", "sfu", "other"]


@dataclass
class Workload:
    model_name: str
    input_length: int
    output_length: int
    batch_size: int
    prefill_ops: List[object]
    decode_ops: List[object]


def resolve_phase_method(method: str, phase: str) -> str:
    # Kept as the single place a method name is normalized. No method currently
    # resolves to a different preset per phase -- `vqarray` did, and was removed
    return str(method).strip().lower()


def available_methods() -> List[str]:
    return sorted(list_hw_presets())


def build_model_workload(
    model_name: str,
    input_length: int,
    output_length: int,
    *,
    batch_size: int = 1,
    default_dtype: str = "fp16",
    sparsity_config: Optional[Dict[str, Any]] = None,
    attention_config: Optional[Dict[str, Any]] = None,
) -> Workload:
    attn_cfg = dict(DEFAULT_ATTENTION_CONFIG if attention_config is None else attention_config)
    prefill_ops = create_network(
        model_name=model_name,
        inference_mode="prefill",
        context_len=input_length,
        batch_size=batch_size,
        default_dtype=default_dtype,
        attention_config=attn_cfg,
    )

    decode_ops: List[object] = []
    for step in range(output_length):
        kv_tokens = input_length + step + 1
        step_ops = create_network(
            model_name=model_name,
            inference_mode="decode",
            context_len=1,
            batch_size=batch_size,
            default_dtype=default_dtype,
            attention_config=attn_cfg,
            kv_cache_tokens=kv_tokens,
        )
        for op in step_ops:
            metadata = getattr(op, "metadata", {})
            metadata["decode_step"] = step
            metadata["decode_kv_tokens"] = kv_tokens
        decode_ops.extend(step_ops)

    if sparsity_config:
        from sparsity import apply_sparsity_info
        apply_sparsity_info(prefill_ops, sparsity_config)
        apply_sparsity_info(decode_ops, sparsity_config)

    return Workload(
        model_name=model_name,
        input_length=input_length,
        output_length=output_length,
        batch_size=batch_size,
        prefill_ops=prefill_ops,
        decode_ops=decode_ops,
    )


def validate_workload(workload: Workload) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    checks.append(
        {
            "name": "prefill_ops_match_input_length",
            "passed": (workload.input_length > 0 and len(workload.prefill_ops) > 0) or (workload.input_length <= 0 and len(workload.prefill_ops) == 0),
            "detail": f"input_length={workload.input_length} prefill_ops={len(workload.prefill_ops)}",
        }
    )

    decode_expected = workload.output_length == 0 or len(workload.decode_ops) > 0
    checks.append(
        {
            "name": "decode_ops_present_when_needed",
            "passed": decode_expected,
            "detail": f"decode_ops={len(workload.decode_ops)} output_length={workload.output_length}",
        }
    )

    decode_kv_tokens = [
        int(getattr(op, "metadata", {}).get("kv_tokens"))
        for op in workload.decode_ops
        if getattr(op, "metadata", {}).get("attn_phase") == "qkt"
    ]
    expected_kv_tokens = list(range(workload.input_length + 1, workload.input_length + workload.output_length + 1))
    checks.append(
        {
            "name": "decode_kv_tokens_increment",
            "passed": decode_kv_tokens == expected_kv_tokens,
            "detail": f"observed={decode_kv_tokens[:8]} expected={expected_kv_tokens[:8]}",
        }
    )

    layer_counts = {
        int(getattr(op, "metadata", {}).get("layer_count", 1))
        for op in workload.prefill_ops + workload.decode_ops
    }
    checks.append(
        {
            "name": "single_layer_count_metadata",
            "passed": len(layer_counts) <= 1,
            "detail": f"layer_counts={sorted(layer_counts)}",
        }
    )

    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "prefill_ops": len(workload.prefill_ops),
        "decode_ops": len(workload.decode_ops),
    }


def create_simulator(method: str, hw_config: Dict[str, Any]) -> HWSimBase:
    if method == "vortex":
        return HWVortex(name=method, config=hw_config)
    return HWSimBase(name=method, config=hw_config)


def run_method_workload(
    workload: Workload,
    method: str,
    *,
    args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    st_time = time.time()
    phase_results = {
        "prefill": _run_phase(workload.prefill_ops, method, "prefill", args=args),
        "decode": _run_phase(workload.decode_ops, method, "decode", args=args),
    }
    print(f"{time.time() - st_time} seconds for _run_phase prefill&decode")
    st_time = time.time()
    total_summary = merge_phase_summaries(phase_results["prefill"], phase_results["decode"])
    sanity = validate_method_summary(total_summary)
    print(f"{time.time() - st_time} seconds for the rest run_method_workload")
    return {
        "method": method,
        "prefill": phase_results["prefill"],
        "decode": phase_results["decode"],
        "total": total_summary,
        "sanity": sanity,
    }


def merge_phase_summaries(prefill: Dict[str, Any], decode: Dict[str, Any]) -> Dict[str, Any]:
    merged_categories = {
        category: merge_summary(prefill["categories"][category], decode["categories"][category])
        for category in CATEGORY_ORDER
    }
    total_time_s = float(prefill["time_s"]) + float(decode["time_s"])
    total_energy = float(prefill["total_energy"]) + float(decode["total_energy"])
    total_cycles = int(prefill["total_cycles"]) + int(decode["total_cycles"])
    area_breakdown = dict(prefill["area_breakdown"])
    for key, value in decode["area_breakdown"].items():
        area_breakdown[key] = max(float(area_breakdown.get(key, 0.0)), float(value))

    cycle_breakdown = dict(prefill.get("cycle_breakdown", {}))
    for key, value in decode.get("cycle_breakdown", {}).items():
        cycle_breakdown[key] = cycle_breakdown.get(key, 0) + value

    return {
        "phase_method": f"{prefill['phase_method']} + {decode['phase_method']}",
        "total_cycles": total_cycles,
        "compute_cycles": int(prefill["compute_cycles"]) + int(decode["compute_cycles"]),
        "mem_stall_cycles": int(prefill["mem_stall_cycles"]) + int(decode["mem_stall_cycles"]),
        "preprocess_stall_cycles": int(prefill["preprocess_stall_cycles"]) + int(decode["preprocess_stall_cycles"]),
        "num_ops": int(prefill["num_ops"]) + int(decode["num_ops"]),
        "dram_energy": float(prefill["dram_energy"]) + float(decode["dram_energy"]),
        "buffer_energy": float(prefill["buffer_energy"]) + float(decode["buffer_energy"]),
        "core_energy": float(prefill["core_energy"]) + float(decode["core_energy"]),
        "static_energy": float(prefill["static_energy"]) + float(decode["static_energy"]),
        "total_energy": total_energy,
        "time_s": total_time_s,
        "core_power": 0.0 if total_time_s <= 0 else (float(prefill["core_energy"]) + float(decode["core_energy"])) / total_time_s,
        "sram_power": 0.0 if total_time_s <= 0 else (float(prefill["buffer_energy"]) + float(decode["buffer_energy"])) / total_time_s,
        "dram_power": 0.0 if total_time_s <= 0 else (float(prefill["dram_energy"]) + float(decode["dram_energy"])) / total_time_s,
        "total_power": 0.0 if total_time_s <= 0 else total_energy / total_time_s,
        "area_mm2": max(float(prefill["area_mm2"]), float(decode["area_mm2"])),
        "area_breakdown": area_breakdown,
        "cycle_breakdown": cycle_breakdown,
        "categories": merged_categories,
    }


def validate_method_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    category_cycle_sum = sum(int(summary["categories"][category]["total_cycles"]) for category in CATEGORY_ORDER)
    category_energy_sum = sum(float(summary["categories"][category]["total_energy"]) for category in CATEGORY_ORDER)
    checks = [
        {
            "name": "positive_cycles",
            "passed": int(summary["total_cycles"]) >= 0,
            "detail": f"total_cycles={summary['total_cycles']}",
        },
        {
            "name": "non_negative_energy",
            "passed": all(float(summary[key]) >= 0.0 for key in ["dram_energy", "buffer_energy", "core_energy", "total_energy"]),
            "detail": f"total_energy={summary['total_energy']}",
        },
        {
            "name": "power_finite",
            "passed": float(summary["total_power"]) >= 0.0,
            "detail": f"total_power={summary['total_power']}",
        },
        {
            "name": "category_cycles_match_total",
            "passed": category_cycle_sum == int(summary["total_cycles"]),
            "detail": f"category_sum={category_cycle_sum} total={summary['total_cycles']}",
        },
        {
            "name": "category_energy_matches_total",
            "passed": abs(category_energy_sum - float(summary["total_energy"])) <= max(1e-12, abs(float(summary["total_energy"])) * 1e-9),
            "detail": f"category_sum={category_energy_sum} total={summary['total_energy']}",
        },
    ]
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def _run_phase(
    ops: List[object],
    method: str,
    phase: str,
    *,
    args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    phase_method = resolve_phase_method(method, phase)
    args["phase"] = phase
    hw_config = get_hw_config(phase_method)
    simulator = create_simulator(phase_method, hw_config)
    if not ops:
        empty_stats = Stats(name=f"{phase_method}.empty")
        energy = simulator.get_energy(empty_stats, hw_config)
        area = simulator.get_area(hw_config)
        return _summarize_phase_result(
            phase_method,
            empty_stats,
            energy,
            area,
            categories={category: _summarize_stats_only(Stats(name=f"{phase_method}.{category}.empty"), simulator, hw_config) for category in CATEGORY_ORDER},
        )

    phase_ops = copy.deepcopy(ops)

    result = simulator.run_model(phase_ops, method=phase_method, args=args, hw_config=hw_config)
    categories={category: _summarize_stats_only(Stats(name=f"{phase_method}.{category}.empty"), simulator, hw_config) for category in CATEGORY_ORDER}
    return _summarize_phase_result(phase_method, result["stats"], result["energy"], result["area"], categories=categories)


def _summarize_phase_result(
    phase_method: str,
    stats: Stats,
    energy: Dict[str, Any],
    area: Dict[str, Any],
    *,
    categories: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "phase_method": phase_method,
        "total_cycles": int(stats.total_cycles),
        "compute_cycles": int(stats.compute_cycles),
        "mem_stall_cycles": int(stats.mem_stall_cycles),
        "preprocess_stall_cycles": int(stats.preprocess_stall_cycles),
        "num_ops": int(stats.num_ops),
        "static_energy": float(energy["static_energy"]),
        "dram_energy": float(energy["dram_energy"]),
        "buffer_energy": float(energy["buffer_energy"]),
        "core_energy": float(energy["core_energy"]),
        "total_energy": float(energy["total_energy"]),
        "core_power": float(energy["core_power"]),
        "sram_power": float(energy["sram_power"]),
        "dram_power": float(energy["dram_power"]),
        "total_power": float(energy["total_power"]),
        "time_s": 0.0 if float(energy["total_power"]) <= 0.0 else float(energy["total_energy"]) / float(energy["total_power"]),
        "area_mm2": float(area["total_area (mm^2)"]),
        "area_breakdown": dict(area["breakdown"]),
        "cycle_breakdown": dict(stats.cycle_breakdown) if isinstance(stats.cycle_breakdown, dict) else {},
        "categories": categories,
    }


def _run_phase_categories(
    ops: List[object],
    *,
    simulator: HWSimBase,
    phase_method: str,
    args: Optional[Dict[str, Any]],
    hw_config: Dict[str, Any],
    layer_count: int,
) -> Dict[str, Dict[str, Any]]:
    category_stats = {category: Stats(name=f"{phase_method}.{category}") for category in CATEGORY_ORDER}
    for stats in category_stats.values():
        stats.num_ops = 0

    for op in ops:
        category = classify_op(op)
        metadata = getattr(op, "metadata", {})
        if metadata.get("attn_phase") is not None:
            op_stats = simulator.run_attention_ops([op], method=phase_method, args=args, hw_config=hw_config)
        else:
            op_stats = simulator.run_op(op, method=phase_method, args=args, hw_config=hw_config)
        category_stats[category].add_(op_stats)


    summaries: Dict[str, Dict[str, Any]] = {}
    for category, stats in category_stats.items():
        if layer_count > 1:
            stats.scale_(layer_count)
        stats.finalize()
        summaries[category] = _summarize_stats_only(stats, simulator, hw_config)
    return summaries


def _summarize_stats_only(stats: Stats, simulator: HWSimBase, hw_config: Dict[str, Any]) -> Dict[str, Any]:
    energy = simulator.get_energy(stats, hw_config)
    return {
        "total_cycles": int(stats.total_cycles),
        "compute_cycles": int(stats.compute_cycles),
        "mem_stall_cycles": int(stats.mem_stall_cycles),
        "preprocess_stall_cycles": int(stats.preprocess_stall_cycles),
        "num_ops": int(stats.num_ops),
        "static_energy": float(energy["static_energy"]),
        "dram_energy": float(energy["dram_energy"]),
        "buffer_energy": float(energy["buffer_energy"]),
        "core_energy": float(energy["core_energy"]),
        "total_energy": float(energy["total_energy"]),
        "core_power": float(energy["core_power"]),
        "sram_power": float(energy["sram_power"]),
        "dram_power": float(energy["dram_power"]),
        "total_power": float(energy["total_power"]),
        "time_s": 0.0 if float(energy["total_power"]) <= 0.0 else float(energy["total_energy"]) / float(energy["total_power"]),
    }


def merge_summary(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    total_time_s = float(left["time_s"]) + float(right["time_s"])
    total_energy = float(left["total_energy"]) + float(right["total_energy"])
    return {
        "total_cycles": int(left["total_cycles"]) + int(right["total_cycles"]),
        "compute_cycles": int(left["compute_cycles"]) + int(right["compute_cycles"]),
        "mem_stall_cycles": int(left["mem_stall_cycles"]) + int(right["mem_stall_cycles"]),
        "preprocess_stall_cycles": int(left["preprocess_stall_cycles"]) + int(right["preprocess_stall_cycles"]),
        "num_ops": int(left["num_ops"]) + int(right["num_ops"]),
        "static_energy": float(left["static_energy"]) + float(right["static_energy"]),
        "dram_energy": float(left["dram_energy"]) + float(right["dram_energy"]),
        "buffer_energy": float(left["buffer_energy"]) + float(right["buffer_energy"]),
        "core_energy": float(left["core_energy"]) + float(right["core_energy"]),
        "total_energy": total_energy,
        "core_power": 0.0 if total_time_s <= 0 else (float(left["core_energy"]) + float(right["core_energy"])) / total_time_s,
        "sram_power": 0.0 if total_time_s <= 0 else (float(left["buffer_energy"]) + float(right["buffer_energy"])) / total_time_s,
        "dram_power": 0.0 if total_time_s <= 0 else (float(left["dram_energy"]) + float(right["dram_energy"])) / total_time_s,
        "total_power": 0.0 if total_time_s <= 0 else total_energy / total_time_s,
        "time_s": total_time_s,
    }


def classify_op(op: object) -> str:
    metadata = getattr(op, "metadata", {})
    attn_phase = metadata.get("attn_phase")
    op_type = str(getattr(op, "op_type", "")).lower()
    if attn_phase in {"qk", "qkt"}:
        return "attention_qk"
    if attn_phase == "sv":
        return "attention_sv"
    if op_type in {"softmax", "vmul", "vadd", "sum"}:
        return "sfu"
    if op_type in {"gemm", "gemv"}:
        return "linear"
    return "other"


def _phase_quant_methods(phase_method: str) -> List[str]:
    if phase_method == "vortex":
        return ["AQLM", "CQ"]
    return []
