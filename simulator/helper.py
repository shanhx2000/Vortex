"""Small shared utilities: ceil division, dtype widths, config merging, CSV logging.

`log_results_to_csv` rewrites the whole CSV on every call, so concurrent runs
must not share a `--csv_path`; `run_simulation.sh` gives each
parallel job its own shard and merges afterwards.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DTYPE_BITS = {
    "fp32": 32,
    "fp16": 16,
    "bf16": 16,
    "int8": 8,
    "int4": 4,
    "int3": 3,
    "int2": 2,
    "quant": 4,
}


def deep_update(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively update a dictionary without modifying the input."""
    result = copy.deepcopy(base)
    if override is None:
        return result
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("Divider must be positive")
    return (a + b - 1) // b


def dtype_bits(dtype: str) -> int:
    return DTYPE_BITS.get(dtype, 16)


def tensor_numel(shape: Iterable[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def tensor_nbits(shape: Iterable[int], dtype: str) -> int:
    return tensor_numel(shape) * dtype_bits(dtype)


def tensor_info_bits(info: Optional[Dict[str, Any]], default_dtype: str = "fp16") -> int:
    if not info:
        return dtype_bits(default_dtype)

    qcfg = info.get("quantization_config") or {}
    if qcfg:
        if "bits" in qcfg:
            return int(qcfg["bits"])
        if "num_bits" in qcfg:
            return int(qcfg["num_bits"])
        if "storage_dtype" in qcfg:
            return dtype_bits(str(qcfg["storage_dtype"]))
        if info.get("dtype") == "quant" and "dtype" in qcfg:
            return dtype_bits(str(qcfg["dtype"]))

    return dtype_bits(str(info.get("dtype", default_dtype)))

def tensor_info_bits_v2(
    info: Optional[Dict[str, Any]],
    default_dtype: str = "fp16",
) -> int:
    DTYPE_BITS = {
        "fp32": 32,
        "fp16": 16,
        "bf16": 16,
        "int8": 8,
        "int4": 4,
        "int3": 3,
        "int2": 2,
    }
    if info is None:
        return DTYPE_BITS[default_dtype]
    dtype = info.get("dtype", default_dtype)
    if dtype in DTYPE_BITS:
        return DTYPE_BITS[dtype]
    if dtype == "quant":
        qcfg = info.get("quantization_config", None)
        assert qcfg is not None, "quant dtype requires quantization_config"
        assert "nbit" in qcfg, f"quantization_config must contain 'nbit', {qcfg}"
        return int(qcfg["nbit"])
    raise ValueError(f"Unsupported dtype: {dtype}")

def tensor_info_nbits(shape: Iterable[int], info: Optional[Dict[str, Any]], default_dtype: str = "fp16") -> int:
    return tensor_numel(shape) * tensor_info_bits(info, default_dtype=default_dtype)


def read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def normalize_sparsity_value(value: Any) -> float | List[float]:
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return float(value)


def mean_or_zero(values: Iterable[float]) -> float:
    values = [float(v) for v in values]
    if not values:
        return 0.0
    return sum(values) / len(values)


def get_input_sparsity(info: Optional[Dict[str, Any]]) -> float:
    value = get_input_sparsity_value(info)
    if isinstance(value, list):
        return mean_or_zero(value)
    return float(value)


def get_input_sparsity_value(info: Optional[Dict[str, Any]], batch_size: Optional[int] = None) -> float | List[float]:
    if not info:
        return 0.0
    sparsity_info = info.get("sparsity_info") or {}
    entry = sparsity_info.get("input_sparsity") or sparsity_info.get("input sparsity") or {}
    fn = entry.get("batched_sparsity_fn")
    if fn is not None and batch_size is not None:
        return normalize_sparsity_value(fn(batch_size))
    return normalize_sparsity_value(entry.get("s", 0.0))


def get_input_sparsity_profile(info: Optional[Dict[str, Any]], batch_size: Optional[int] = None) -> List[float]:
    value = get_input_sparsity_value(info, batch_size=batch_size)
    if isinstance(value, list):
        return value
    return [float(value)]


def get_batched_sparsity(info: Optional[Dict[str, Any]], batch_size: int) -> float:
    value = get_input_sparsity_value(info, batch_size=batch_size)
    if isinstance(value, list):
        return mean_or_zero(value)
    return float(value)


def is_proj_name(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ["fc", "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]) #  "linear",


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]

def resolve_json_values(obj):
    if isinstance(obj, dict):
        return {k: resolve_json_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_json_values(v) for v in obj]
    elif isinstance(obj, str):
        try:
            return eval(obj)
        except Exception:
            return obj  # not an expression -- keep the string as-is
    else:
        return obj

def load_json_with_eval(path):
    with open(path, "r") as f:
        data = json.load(f)
    return resolve_json_values(data)

import os
import pandas as pd
from datetime import datetime

FIELDS = [
    "timestamp",
    "model_name", "method",
    "input_length", "output_length", "batch_size",
    "compute_mode", "force_dataflow",
    "quant_scheme", "processed_sparsity",
    "total_cycles", "compute_cycles", "mem_stall_cycles",
    "total_energy", "core_energy", "buffer_energy", "dram_energy", "static_energy",
    "total_power",
    "area_mm2",
    "prefill_cycles", "decode_cycles",
    "prefill_energy", "decode_energy",
    "tokens_per_s", "energy_per_token",
]


def log_results_to_csv(results: dict, args, csv_path: str):
    rows = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ===== config fields =====
    quant_scheme = getattr(args, "quant_scheme", None)
    if isinstance(quant_scheme, list):
        quant_scheme = "|".join(quant_scheme)
    if quant_scheme is None:
        quant_scheme = "None"

    processed_sparsity = getattr(args, "processed_sparsity", None)
    if processed_sparsity is None:
        processed_sparsity = 0.0

    for method, res in results.items():
        total = res["total"]
        prefill = res["prefill"]
        decode = res["decode"]

        total_time = float(total.get("time_s", 0.0))
        total_energy = float(total.get("total_energy", 0.0))

        # ===== efficiency =====
        total_tokens = args.input_length + args.output_length
        tokens_per_s = total_tokens / total_time if total_time > 0 else 0.0
        energy_per_token = total_energy / total_tokens if total_tokens > 0 else 0.0

        row = {
            "timestamp": timestamp,

            # config
            "model_name": args.model_name,
            "method": method,
            "input_length": args.input_length,
            "output_length": args.output_length,
            "batch_size": args.batch_size,
            "compute_mode": args.compute_mode,
            "force_dataflow": args.force_dataflow,
            "quant_scheme": quant_scheme,
            "processed_sparsity": processed_sparsity,

            # performance
            "total_cycles": total.get("total_cycles", 0),
            "compute_cycles": total.get("compute_cycles", 0),
            "mem_stall_cycles": total.get("mem_stall_cycles", 0),

            # energy total
            "total_energy": total_energy,
            "core_energy": total.get("core_energy", 0.0),
            "buffer_energy": total.get("buffer_energy", 0.0),
            "dram_energy": total.get("dram_energy", 0.0),
            "static_energy": total.get("static_energy", 0.0),

            # power
            "total_power": total.get("total_power", 0.0),

            # area
            "area_mm2": total.get("area_mm2", 0.0),

            # phase
            "prefill_cycles": prefill.get("total_cycles", 0),
            "decode_cycles": decode.get("total_cycles", 0),
            "prefill_energy": prefill.get("total_energy", 0.0),
            "decode_energy": decode.get("total_energy", 0.0),

            # efficiency
            "tokens_per_s": tokens_per_s,
            "energy_per_token": energy_per_token,
        }

        rows.append(row)

    df_new = pd.DataFrame(rows, columns=FIELDS)

    # Append to the CSV, rewriting it whole. Concurrent writers therefore need
    # external locking or separate paths -- see run_simulation.sh.
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(csv_path, index=False)