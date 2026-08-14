"""Expands a model layer description into a flat list of operations.

`create_network` dispatches per layer type -- linear, attention, sfu -- and
produces GEMM/GEMV, Softmax, VMul, VAdd and Sum ops with tensor dtype and
location metadata attached. Attention expansion emits one block's worth of ops;
the head count is applied later as a scale factor on the resulting Stats.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from helper import deep_update, ensure_list
from llm_configs import get_model_definition
from operations import GEMM, GEMV, Quantize, Softmax, VMul, VAdd, Sum, default_tensor_info
from quantization import apply_quant_methods
from sparsity import apply_sparsity_info

@dataclass
class NetworkContext:
    inference_mode: str = "prefill"
    context_len: int = 1
    kv_cache_tokens: Optional[int] = None
    batch_size: int = 1
    default_dtype: str = "fp16"
    sparsity_config: Optional[Dict[str, Any]] = None
    attention_config: Optional[Dict[str, Any]] = None


def create_network(
    structure: Optional[List[Dict[str, Any]]] = None,
    quant_methods: Optional[List[str]] = None,
    inference_mode: str = "prefill",
    context_len: int = 1,
    batch_size: int = 1,
    default_dtype: str = "fp16",
    sparsity_config: Optional[Dict[str, Any]] = None,
    attention_config: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    kv_cache_tokens: Optional[int] = None,
) -> List[object]:
    """Build a flat operation list from a structured network description.

    Expected usage:
    - Explicit structure:
      create_network(structure=[...], ...)
    - Model name from llm_configs:
      create_network(model_name="llama-2-7b", ...)

    Expected structure examples:
    - Linear block:
      {"type": "linear", "name": "fc1", "in_features": 4096, "out_features": 11008}
    - Attention block:
      {
          "type": "attn",
          "name": "attn0",
          "hidden_size": 4096,
          "num_heads": 32,
          "head_dim": 128,
      }
    """
    ctx = NetworkContext(
        inference_mode=inference_mode,
        context_len=context_len,
        kv_cache_tokens=kv_cache_tokens,
        batch_size=batch_size,
        default_dtype=default_dtype,
        sparsity_config=sparsity_config,
        attention_config=attention_config,
    )

    if structure is None:
        if model_name is None:
            raise ValueError("Either structure or model_name must be provided")
        structure = _build_structure_from_model(model_name, ctx)

    ops: List[object] = []
    for layer in structure:
        layer_type = layer["type"].lower()
        if layer_type == "linear":
            ops.extend(_expand_linear(layer, ctx))
        elif layer_type in {"attn", "attention"}:
            ops.extend(_expand_attention(layer, ctx))
        elif layer_type in {"sfu", "vector"}:
            ops.extend(_expand_sfu(layer, ctx))
        else:
            raise ValueError(f"Unsupported layer type: {layer['type']}")

    if quant_methods:
        apply_quant_methods(ops, quant_methods)

    assert ctx.sparsity_config is None, f"Moved to outside create_network: {ctx.sparsity_config}"

    return ops


def _expand_linear(layer: Dict[str, Any], ctx: NetworkContext) -> List[object]:
    name = layer["name"]
    in_features = int(layer["in_features"])
    out_features = int(layer["out_features"])
    tokens = _effective_tokens(ctx)
    rows = tokens * ctx.batch_size

    act_info = _make_activation_info(layer, ctx)
    weight_info = _make_weight_info(layer, ctx)
    out_info = _make_output_info(layer, ctx)

    gemm_cls = GEMV if rows == 1 else GEMM
    if gemm_cls is GEMV:
        op = gemm_cls(name=name, K=in_features, N=out_features, A_info=act_info, B_info=weight_info, out_info=out_info, parent_name=name)
    else:
        op = gemm_cls(name=name, M=rows, K=in_features, N=out_features, A_info=act_info, B_info=weight_info, out_info=out_info, parent_name=name)
    op.metadata.update(layer.get("metadata", {}))
    return [op]

# Reachable from create_network's dispatch, but none of the three evaluated
# models (llama-2-7b, llama_2_13b, mistral_7b) emit sfu_* ops -- only the
# Mixtral and Qwen configs do. Untested against the paper's numbers.
def _expand_sfu(layer: Dict[str, Any], ctx: NetworkContext) -> List[object]:
    name = layer["name"]
    shape = ensure_list(layer.get("shape") or [])
    if not shape:
        raise ValueError(f"SFU layer {name} must provide a shape")
    length = ctx.batch_size
    for dim in shape:
        length *= int(dim)

    act_info = _make_activation_info(layer, ctx, name_suffix="in")
    out_info = _make_output_info(layer, ctx)
    op = VMul(
        name=name,
        N=length,
        A_info=act_info,
        B_info=default_tensor_info(dtype=ctx.default_dtype, loc="buffer"),
        out_info=out_info,
        parent_name=name,
    )
    op.metadata.update(
        {
            "sfu_type": layer.get("sfu_type", name),
            "shape": [int(dim) for dim in shape],
        }
    )
    op.metadata.update(layer.get("metadata", {}))
    return [op]


def _expand_attention(layer: Dict[str, Any], ctx: NetworkContext) -> List[object]:
    name = layer["name"]
    hidden_size = int(layer["hidden_size"])
    num_heads = int(layer.get("num_heads", 1))
    num_kv_heads = int(layer.get("num_kv_heads", num_heads))
    head_dim = int(layer.get("head_dim", hidden_size // max(num_heads, 1)))

    if hidden_size != num_heads * head_dim:
        raise ValueError(f"For layer {name}, hidden_size must equal num_heads * head_dim")
    if num_kv_heads <= 0:
        raise ValueError(f"For layer {name}, num_kv_heads must be positive")
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"For layer {name}, num_heads must be divisible by num_kv_heads")

    group_size = int(layer.get("group_size", num_heads // num_kv_heads))
    if group_size <= 0:
        raise ValueError(f"For layer {name}, group_size must be positive")
    if num_heads % group_size != 0:
        raise ValueError(f"For layer {name}, num_heads must be divisible by group_size")

    batch = ctx.batch_size
    q_tokens = _effective_tokens(ctx)
    if ctx.kv_cache_tokens is not None:
        kv_tokens = int(ctx.kv_cache_tokens)
    else:
        kv_tokens = ctx.context_len if ctx.inference_mode == "decode" else q_tokens
    q_rows = batch * q_tokens * group_size
    kv_rows = batch * kv_tokens * num_kv_heads
    attn_cfg = deep_update(
        {
            "quantize_kv": False,
            "kv_cache_loc": "dram",
            # "cq_config": {
            #     "method": "CQ",
            #     "scheme": "4c8b",
            #     "num_codebooks": 1,
            #     "vector_size": 4,
            #     "num_entries": 256,
            #     "bit_slice": 1,
            #     "codebook_loc": "buffer",
            # },
        },
        ctx.attention_config,
    )
    attn_cfg = deep_update(attn_cfg, layer.get("attention_config"))
    cq_cfg = dict(attn_cfg.get("cq_config") or {})
    cq_cfg.setdefault("codebook_loc", attn_cfg["kv_cache_loc"])
    kv_quant_cfg = dict(attn_cfg.get("kv_quant_config") or cq_cfg)

    qk_name = f"{name}.qk"
    sv_name = f"{name}.sv"
    softmax_name = f"{name}.softmax"
    kv_quant_name = f"{name}.kv_quant"

    kv_cache_info = _make_weight_like_info(ctx, dtype=ctx.default_dtype, loc=attn_cfg["kv_cache_loc"])
    sv_cache_info = _make_weight_like_info(ctx, dtype=ctx.default_dtype, loc=attn_cfg["kv_cache_loc"])
    if attn_cfg.get("quantize_kv"):
        kv_cache_info["dtype"] = "quant"
        kv_cache_info["quantization_config"] = dict(cq_cfg)
        sv_cache_info["dtype"] = "quant"
        sv_cache_info["quantization_config"] = dict(cq_cfg)

    qk = GEMM(
        name=qk_name,
        M=q_rows,
        K=head_dim,
        N=kv_tokens,
        A_info=_make_activation_info(layer, ctx, name_suffix="q"),
        B_info=kv_cache_info,
        out_info=_make_output_info(layer, ctx),
        parent_name=name,
    )
    softmax = Softmax(
        name=softmax_name,
        N=q_rows * kv_tokens,
        I_info=_make_output_info(layer, ctx),
        out_info=_make_output_info(layer, ctx),
        parent_name=name,
    )
    sv = GEMM(
        name=sv_name,
        M=q_rows,
        K=kv_tokens,
        N=head_dim,
        A_info=_make_output_info(layer, ctx),
        B_info=sv_cache_info,
        out_info=_make_output_info(layer, ctx),
        parent_name=name,
    )
    qk.metadata.update(
        {
            "attn_phase": "qkt",
            "runner_hint": "lut" if attn_cfg.get("quantize_kv") else "dense",
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "group_size": group_size,
            "head_dim": head_dim,
            "q_tokens": q_tokens,
            "kv_tokens": kv_tokens,
        }
    )
    qk.metadata.update(layer.get("metadata", {}))
    softmax.metadata.update(
        {
            "attn_phase": "softmax",
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "group_size": group_size,
            "q_tokens": q_tokens,
            "kv_tokens": kv_tokens,
        }
    )
    softmax.metadata.update(layer.get("metadata", {}))
    sv.metadata.update(
        {
            "attn_phase": "sv",
            "runner_hint": "dequant_gemv" if attn_cfg.get("quantize_kv") else "dense",
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "group_size": group_size,
            "head_dim": head_dim,
            "q_tokens": q_tokens,
            "kv_tokens": kv_tokens,
        }
    )
    sv.metadata.update(layer.get("metadata", {}))

    ops: List[object] = []
    if attn_cfg.get("quantize_kv"):
        kv_quant = Quantize(
            name=kv_quant_name,
            N=2 * kv_rows * head_dim,
            I_info=_make_weight_like_info(ctx, dtype=ctx.default_dtype, loc=attn_cfg["kv_cache_loc"]),
            out_info=_make_weight_like_info(ctx, dtype="quant", loc=attn_cfg["kv_cache_loc"]),
            parent_name=name,
        )
        kv_quant.out_info["quantization_config"] = dict(kv_quant_cfg)
        kv_quant.metadata.update(
            {
                "attn_phase": "kv_quant",
                "runner_hint": "kv_quant",
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "group_size": group_size,
                "head_dim": head_dim,
                "kv_tokens": kv_tokens,
            }
        )
        kv_quant.metadata.update(layer.get("metadata", {}))
        ops.append(kv_quant)

    ops.extend([qk, softmax, sv])
    return ops


def _effective_tokens(ctx: NetworkContext) -> int:
    if ctx.inference_mode == "decode":
        return 1
    if ctx.inference_mode == "prefill":
        return ctx.context_len
    raise ValueError(f"Unsupported inference mode: {ctx.inference_mode}")


def _build_structure_from_model(model_name: str, ctx: NetworkContext) -> List[Dict[str, Any]]:
    sequence_length = _effective_tokens(ctx)
    model_def = get_model_definition(model_name, sequence_length=sequence_length)
    structure: List[Dict[str, Any]] = []
    num_layers = int(model_def["num_layers"])
    num_heads = int(model_def["num_attention_heads"])
    head_dim = int(model_def["head_dim"])
    num_kv_heads = int(model_def.get("num_kv_heads", num_heads))

    for op_name, dims in model_def["layer"].items():
        full_name = f"layer0.{op_name}"
        common_metadata = {
            "layer_index": 0,
            "layer_count": num_layers,
            "model_name": model_name,
        }
        if op_name.startswith("fc_"):
            structure.append(
                {
                    "type": "linear",
                    "name": full_name,
                    "in_features": int(dims[1]),
                    "out_features": int(dims[0]),
                    "metadata": dict(common_metadata),
                }
            )
        elif op_name == "attention":
            hidden_size = int(dims[0])
            structure.append(
                {
                    "type": "attn",
                    "name": full_name,
                    "hidden_size": hidden_size,
                    "num_heads": num_heads,
                    "num_kv_heads": num_kv_heads,
                    "group_size": num_heads // num_kv_heads,
                    "head_dim": head_dim,
                    "metadata": {
                        **common_metadata,
                        "num_kv_heads": num_kv_heads,
                    },
                }
            )
        elif op_name.startswith("sfu_"):
            structure.append(
                {
                    "type": "sfu",
                    "name": full_name,
                    "shape": [int(dim) for dim in dims],
                    "sfu_type": op_name,
                    "metadata": dict(common_metadata),
                }
            )
        else:
            raise ValueError(f"Unsupported model layer op: {op_name}")
    return structure


def _make_activation_info(layer: Dict[str, Any], ctx: NetworkContext, name_suffix: str = "act") -> Dict[str, Any]:
    info = default_tensor_info(dtype=ctx.default_dtype, loc="buffer")
    info["name"] = f"{layer['name']}.{name_suffix}"
    info = deep_update(info, layer.get("A_info"))
    if layer.get("metadata"):
        info.setdefault("metadata", {}).update(layer["metadata"])
    return info


def _make_weight_info(layer: Dict[str, Any], ctx: NetworkContext) -> Dict[str, Any]:
    info = default_tensor_info(dtype=ctx.default_dtype, loc="dram")
    info["name"] = f"{layer['name']}.weight"
    info = deep_update(info, layer.get("B_info"))
    if layer.get("metadata"):
        info.setdefault("metadata", {}).update(layer["metadata"])
    return info


def _make_weight_like_info(ctx: NetworkContext, dtype: str, loc: str) -> Dict[str, Any]:
    return default_tensor_info(dtype=dtype, loc=loc)


def _make_output_info(layer: Dict[str, Any], ctx: NetworkContext) -> Dict[str, Any]:
    info = default_tensor_info(dtype=ctx.default_dtype, loc="buffer")
    info["name"] = f"{layer['name']}.out"
    info = deep_update(info, layer.get("out_info"))
    if layer.get("metadata"):
        info.setdefault("metadata", {}).update(layer["metadata"])
    return info


def format_op_compute_info(op: object) -> str:
    if isinstance(op, GEMM):
        base = f"{op.name:24s} {op.op_type:8s} M={op.M:<6d} K={op.K:<6d} N={op.N:<6d}"
    elif isinstance(op, Softmax):
        base = f"{op.name:24s} {op.op_type:8s} N={op.N:<6d}"
    elif isinstance(op, Quantize):
        base = f"{op.name:24s} {op.op_type:8s} N={op.N:<6d}"
    elif isinstance(op, (VMul, VAdd, Sum)):
        base = f"{op.name:24s} {op.op_type:8s} N={op.N:<6d}"
    else:
        base = f"{op.name:24s} {getattr(op, 'op_type', 'op'):8s}"

    metadata = getattr(op, "metadata", {})
    extras: List[str] = []
    if metadata.get("attn_phase"):
        extras.append(f"phase={metadata['attn_phase']}")
    if metadata.get("num_heads") is not None:
        extras.append(f"heads={metadata['num_heads']}")
    if metadata.get("num_kv_heads") is not None:
        extras.append(f"kv_heads={metadata['num_kv_heads']}")
    if metadata.get("group_size") is not None:
        extras.append(f"group_size={metadata['group_size']}")
    if metadata.get("q_tokens") is not None:
        extras.append(f"q_tokens={metadata['q_tokens']}")
    if metadata.get("kv_tokens") is not None:
        extras.append(f"kv_tokens={metadata['kv_tokens']}")
    if metadata.get("layer_count") is not None:
        extras.append(f"layer_count={metadata['layer_count']}")
    return base if not extras else f"{base}  {' '.join(extras)}"


def print_network_compute_info(ops: List[object]) -> None:
    for op in ops:
        print(format_op_compute_info(op))


def main() -> None:
    ops = create_network(
        model_name="llama-2-7b",
        inference_mode="prefill",
        context_len=16,
    )
    print_network_compute_info(ops)


if __name__ == "__main__":
    main()
