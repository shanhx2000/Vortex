"""Rewrites op tensor descriptors to reflect a quantisation scheme.

`apply_quant_methods` is the entry point; schemes compose left to right, so
"AQLM|CQ" applies AQLM to the weights and then CQ to the attention path. These
functions change only metadata -- dtype, bit width, codebook counts -- which is
what the cycle models read to decide how much data moves and how many LUT
lookups are needed.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

from helper import is_proj_name


def apply_quant_methods(network: List[object], quant_methods: Iterable[str]) -> List[object]:
    """Apply quantization transforms sequentially in the given order."""
    methods = [m.upper() for m in quant_methods]
    for method in methods:
        parsed_gptq_bits = _parse_weight_only_bits(method, prefix="GPTQ")
        if method == "AQLM":
            apply_aqlm(network)
        elif method == "CQ":
            apply_cq(network)
        elif method == "W8A8":
            apply_proj_wxa8(network, weight_dtype="int8", act_dtype="int8")
        elif method == "W4A8":
            apply_proj_wxa8(network, weight_dtype="int4", act_dtype="int8")
        elif method in {"W4A16", "AWQ", "AWQ4"}:
            apply_awq(network)
        elif method in {"GPTQ", "GPTQ4", "W4A16_GPTQ"}:
            apply_gptq(network, bits=4)
        elif method in {"GPTQ3", "W3A16_GPTQ"}:
            apply_gptq(network, bits=3)
        elif method in {"GPTQ2", "W2A16_GPTQ"}:
            apply_gptq(network, bits=2)
        elif parsed_gptq_bits is not None:
            apply_gptq(network, bits=parsed_gptq_bits)
        elif method == "KV4":
            apply_kv4(network)
        elif method == "KV_CACHE":
            apply_kv_cache_quant(network)
        else:
            raise ValueError(f"Unsupported quantization method: {method}")
    return network


def apply_aqlm(network: List[object], config: Dict[str, Any] | None = None) -> None:
    """Assign AQLM metadata to projected weight tensors."""
    cfg = {
        "method": "AQLM",
        "num_codebooks": 2,
        "vector_size": 8,
        "num_entries": 256,
        "codebook_loc": "buffer",
        "wi_loc": "buffer",
        "nbit": 2,
    }
    if config:
        cfg.update(config)

    for op in network:
        if hasattr(op, "B_info") and is_proj_name(op.name):
            op.B_info["dtype"] = "quant"
            op.B_info["quantization_config"] = dict(cfg)


def apply_cq(network: List[object], config: Dict[str, Any] | None = None) -> None:
    """Codebook quantization for attention QK and SV phases.

    Default scheme models a 4c8b-style vector codebook where every 4 elements
    are grouped into a vector and quantized using 2^8 entries.
    """
    cfg = {
        "method": "CQ",
        "scheme": "4c8b",
        "vector_size": 4,
        "num_entries": 256,
        "codebook_loc": "buffer",
        "nbit": 2,
    }
    if config:
        cfg.update(config)

    for op in network:
        lowered = op.name.lower()
        if hasattr(op, "B_info") and ("qk" in lowered or "sv" in lowered):
            op.B_info["dtype"] = "quant"
            op.B_info["quantization_config"] = dict(cfg)
            if "qk" in lowered:
                op.B_info["quantization_config"]["attn_type"] = "qk"
            if "sv" in lowered:
                assert "attn_type" not in op.B_info["quantization_config"]
                op.B_info["quantization_config"]["attn_type"] = "sv"


def apply_proj_wxa8(network: List[object], weight_dtype: str, act_dtype: str) -> None:
    """Apply uniform activation and weight quantization for projection ops."""
    for op in network:
        if hasattr(op, "B_info") and is_proj_name(op.name):
            op.B_info["dtype"] = weight_dtype
            op.B_info["quantization_config"] = {
                "method": f"{weight_dtype.upper()}_{act_dtype.upper()}",
            }
        if hasattr(op, "A_info"):
            op.A_info["dtype"] = act_dtype


def apply_awq(network: List[object], config: Dict[str, Any] | None = None) -> None:
    cfg = {
        "method": "AWQ",
        "bits": 4,
        "group_size": 128,
        "storage_dtype": "int4",
        "scale_dtype": "fp16",
        "zero_dtype": "int4",
    }
    if config:
        cfg.update(config)
    apply_weight_only_quant(network, cfg)


def apply_gptq(network: List[object], bits: int = 4, config: Dict[str, Any] | None = None) -> None:
    cfg = {
        "method": "GPTQ",
        "bits": bits,
        "group_size": 128,
        "storage_dtype": f"int{bits}",
        "scale_dtype": "fp16",
        "zero_dtype": f"int{bits}",
    }
    if config:
        cfg.update(config)
    apply_weight_only_quant(network, cfg)


def apply_weight_only_quant(network: List[object], config: Dict[str, Any]) -> None:
    cfg = dict(config)
    bits = int(cfg.get("bits", 4))
    cfg.setdefault("storage_dtype", f"int{bits}")
    cfg.setdefault("zero_dtype", f"int{bits}")
    cfg.setdefault("scale_loc", "s_parameter")
    cfg.setdefault("weight_loc", "dram")
    for op in network:
        if hasattr(op, "B_info") and is_proj_name(op.name):
            op.B_info["dtype"] = "quant"
            op.B_info["quantization_config"] = dict(cfg)


def _parse_weight_only_bits(method: str, prefix: str) -> int | None:
    if not method.startswith(prefix):
        return None
    suffix = method[len(prefix):].replace("_", "")
    if suffix.endswith("B"):
        suffix = suffix[:-1]
    if suffix.isdigit():
        return int(suffix)
    return None


def apply_kv4(network: List[object], config: Dict[str, Any] | None = None) -> None:
    """Apply int4 quantization to attention K/V related tensors."""
    cfg = {
        "method": "KV4",
        "dtype": "int4",
        "vector_size": 8,
        "num_entries": 16,
        "codebook_loc": "buffer",
        "wi_loc": "buffer",
    }
    if config:
        cfg.update(config)
    apply_kv_cache_quant(network, cfg)


def apply_kv_cache_quant(network: List[object], config: Dict[str, Any] | None = None) -> None:
    """Attach KV-cache compression metadata to attention-related ops."""
    cfg = {
        "method": "KV_CACHE",
        "dtype": "int4",
        "vector_size": 8,
        "num_entries": 16,
        "codebook_loc": "buffer",
        "wi_loc": "buffer",
    }
    if config:
        cfg.update(config)

    for op in network:
        attn_phase = getattr(op, "metadata", {}).get("attn_phase")
        if attn_phase == "kv_quant":
            if hasattr(op, "out_info"):
                op.out_info["dtype"] = "quant"
                op.out_info["quantization_config"] = dict(cfg)
            op.metadata["quantization_config"] = dict(cfg)
            continue

        if hasattr(op, "B_info") and attn_phase in {"qkt", "sv"}:
            op.B_info["dtype"] = "quant"
            op.B_info["quantization_config"] = dict(cfg)
