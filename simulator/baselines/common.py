from __future__ import annotations

import math
from typing import Any, Dict, Optional

from energy import update_stats_energy
from helper import ceil_div, deep_update, dtype_bits, tensor_info_bits
from operations import GEMM, GEMV, OP, Quantize, Softmax
from stats import Stats


def run_quantize_placeholder(
    op: Quantize,
    stats: Optional[Stats],
    args: Optional[Dict[str, Any]],
    hw_config: Dict[str, Any],
    *,
    name: str,
) -> Stats:
    """
    We assume KV quantization time can be fully overlapped. 
    """
    
    runtime = deep_update(hw_config.get("runtime", {}), resolve_runtime_args(args))
    mem_width = int(runtime.get("mem_width", 1024))
    compression_ratio = max(float(runtime.get("kv_compression_ratio", 1.0)), 1e-6)
    cycles = ceil_div(int(op.N / compression_ratio), mem_width)
    in_bits = op.N * tensor_info_bits(op.I_info)
    out_bits = op.N * tensor_info_bits(op.out_info, default_dtype="quant")

    result = Stats(name=name)
    result.num_ops = 1
    result.update_cycles(total=cycles, preprocess_stall=cycles)
    result.update_mem("dram", reads=in_bits, writes=0)
    result.update_mem("buffer", reads=in_bits, writes=out_bits)
    update_stats_energy(result, hw_config)
    return copy_stats(result, target=stats, name=name)


def run_softmax(op: Softmax, stats: Optional[Stats], hw_config: Dict[str, Any], *, name: str) -> Stats:
    width = int(hw_config.get("modules", {}).get("sfu", {}).get("softmax", {}).get("width", 32))
    units = max(1, int(hw_config.get("modules", {}).get("sfu", {}).get("softmax", {}).get("num", 1)))
    cycles = ceil_div(op.N, width * units)
    bits = op.N * tensor_info_bits(op.I_info)

    result = Stats(name=name)
    result.num_ops = 1
    result.update_cycles(total=cycles, compute=cycles)
    result.update_mem(op.I_info.get("loc", "buffer"), reads=bits)
    result.update_mem(op.out_info.get("loc", "buffer"), writes=bits)
    update_stats_energy(result, hw_config)
    return copy_stats(result, target=stats, name=name)


def copy_stats(result: Stats, *, target: Optional[Stats], name: str) -> Stats:
    result.name = name
    if target is None:
        return result
    target.__dict__.update(result.clone().__dict__)
    target.name = name
    return target


def resolve_runtime_args(args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        return dict(args)
    if hasattr(args, "__dict__"):
        return dict(vars(args))
    return {}


def estimate_mem_stall(total_read_bits: int, total_write_bits: int, init_bits: int, mem_width: int, compute_cycles: int) -> int:
    init_latency = ceil_div(init_bits, mem_width)
    total_bits = total_read_bits + total_write_bits
    remaining_bits = max(0, total_bits - init_bits)
    middle_latency = ceil_div(remaining_bits, mem_width)
    return init_latency + max(0, middle_latency - compute_cycles)


def dense_systolic_cycles(M: int, N: int, K: int, array_height: int, array_width: int) -> int:
    return ceil_div(N, array_height) * ceil_div(K, array_width) * M + array_height + array_width


def arg_get(args: Optional[Dict[str, Any]], name: str, default: Any) -> Any:
    if args is None:
        return default
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def ceil_a_by_b(a: int, b: int) -> int:
    if b == 0:
        return 0
    return (a + b - 1) // b


def init_stats(stats: Optional[Stats], name: str) -> Stats:
    target = stats or Stats()
    target.name = name
    return target


def add_buffer_reads(stats: Stats, bits: int) -> None:
    stats.reads["buffer"] += bits


def add_buffer_writes(stats: Stats, bits: int) -> None:
    stats.writes["buffer"] += bits


def add_act_reads(stats: Stats, bits: int) -> None:
    stats.reads["s_act"] += bits


def add_act_writes(stats: Stats, bits: int) -> None:
    stats.writes["s_act"] += bits


def add_weight_reads(stats: Stats, bits: int) -> None:
    stats.reads["s_wgt"] += bits


def add_weight_writes(stats: Stats, bits: int) -> None:
    stats.writes["s_wgt"] += bits


def add_codebook_reads(stats: Stats, bits: int) -> None:
    stats.reads["s_codebook"] += bits


def add_codebook_writes(stats: Stats, bits: int) -> None:
    stats.writes["s_codebook"] += bits


def add_cap_reads(stats: Stats, bits: int) -> None:
    stats.reads["s_cap"] += bits


def add_cap_writes(stats: Stats, bits: int) -> None:
    stats.writes["s_cap"] += bits


def add_parameter_reads(stats: Stats, bits: int) -> None:
    stats.reads["s_parameter"] += bits


def add_parameter_writes(stats: Stats, bits: int) -> None:
    stats.writes["s_parameter"] += bits


def add_scale_reads(stats: Stats, bits: int) -> None:
    stats.reads["s_scale"] += bits


def add_scale_writes(stats: Stats, bits: int) -> None:
    stats.writes["s_scale"] += bits


def add_output_reads(stats: Stats, bits: int) -> None:
    stats.reads["s_output"] += bits


def add_output_writes(stats: Stats, bits: int) -> None:
    stats.writes["s_output"] += bits


def weight_stationary_systolic_cycles(M: int, N: int, K: int, array_height: int, array_width: int, *, pipeline_penalty: int) -> int:
    return ceil_a_by_b(N, array_height) * ceil_a_by_b(K, array_width) * M + array_height + array_width + pipeline_penalty


def estimate_tiled_gemm_cycles(M_size: int, N_size: int, K_size: int, array_height: int, array_width: int, *, m_tile_size: int, pipeline_penalty: int) -> int:
    m_num_tile = ceil_a_by_b(M_size, m_tile_size)
    m_input_size = ceil_a_by_b(M_size, m_num_tile)
    compute_cycles = weight_stationary_systolic_cycles(
        m_input_size,
        array_height,
        array_width,
        array_height,
        array_width,
        pipeline_penalty=pipeline_penalty,
    ) * m_num_tile
    n_repeat = ceil_a_by_b(N_size, array_height)
    k_repeat = ceil_a_by_b(K_size, array_width)
    return compute_cycles * n_repeat * k_repeat


def run_tiled_systolic_gemm(
    op: GEMM | GEMV,
    stats: Stats,
    *,
    array_height: int,
    array_width: int,
    tile_size_m: int,
    tile_size_k: int,
    tile_size_n: int,
    buffer_size_m: int,
    buffer_size_k: int,
    buffer_size_n: int,
    mem_width: int,
    pipeline_penalty: int,
    act_nbits: Optional[int] = None,
    weight_nbits: Optional[int] = None,
    out_nbits: Optional[int] = None,
) -> Stats:
    M = int(op.M)
    K = int(op.K)
    N = int(op.N)
    act_bits = int(act_nbits if act_nbits is not None else tensor_info_bits(op.A_info))
    weight_bits = int(weight_nbits if weight_nbits is not None else tensor_info_bits(op.B_info))
    out_bits = int(out_nbits if out_nbits is not None else tensor_info_bits(op.out_info))

    tile_size_m = min(tile_size_m, M, buffer_size_m)
    tile_size_k = min(tile_size_k, K, buffer_size_k)
    tile_size_n = min(tile_size_n, N, buffer_size_n)

    tile_num_M = ceil_a_by_b(M, tile_size_m)
    tile_num_K = ceil_a_by_b(K, tile_size_k)
    tile_num_N = ceil_a_by_b(N, tile_size_n)

    for m in range(tile_num_M):
        for n in range(tile_num_N):
            for k_tile in range(tile_num_K):
                current_tile_size_K = min(tile_size_k, K - k_tile * tile_size_k)
                current_tile_size_M = min(tile_size_m, M - m * tile_size_m)
                current_tile_size_N = min(tile_size_n, N - n * tile_size_n)
                add_buffer_reads(stats, current_tile_size_K * current_tile_size_M * act_bits)
                add_buffer_reads(stats, current_tile_size_N * current_tile_size_K * weight_bits)
                add_output_reads(stats, current_tile_size_M * current_tile_size_N * out_bits)
                add_output_writes(stats, current_tile_size_M * current_tile_size_N * out_bits)

    stats.compute_cycles += estimate_tiled_gemm_cycles(
        M,
        N,
        K,
        array_height,
        array_width,
        m_tile_size=tile_size_m,
        pipeline_penalty=pipeline_penalty,
    )

    buffer_tile_num_M = ceil_a_by_b(M, buffer_size_m)
    buffer_tile_num_K = ceil_a_by_b(K, buffer_size_k)
    buffer_tile_num_N = ceil_a_by_b(N, buffer_size_n)

    for m in range(buffer_tile_num_M):
        for n in range(buffer_tile_num_N):
            current_buffer_tile_size_N = min(buffer_size_n, N - n * buffer_size_n)
            for k_tile in range(buffer_tile_num_K):
                current_buffer_tile_size_K = min(buffer_size_k, K - k_tile * buffer_size_k)
                current_buffer_tile_size_M = min(buffer_size_m, M - m * buffer_size_m)
                stats.reads["dram"] += current_buffer_tile_size_K * current_buffer_tile_size_M * act_bits
                add_buffer_writes(stats, current_buffer_tile_size_K * current_buffer_tile_size_M * act_bits)
                stats.reads["dram"] += current_buffer_tile_size_N * current_buffer_tile_size_K * weight_bits
                add_buffer_writes(stats, current_buffer_tile_size_N * current_buffer_tile_size_K * weight_bits)
                if k_tile > 0:
                    stats.reads["dram"] += current_buffer_tile_size_M * current_buffer_tile_size_N * out_bits
                    add_output_writes(stats, current_buffer_tile_size_M * current_buffer_tile_size_N * out_bits)

    add_output_reads(stats, M * N * out_bits)
    stats.writes["dram"] += M * N * out_bits

    first_buffer_tile_size_K = min(tile_size_k, K)
    first_buffer_tile_size_M = min(tile_size_m, M)
    first_buffer_tile_size_N = min(tile_size_n, N)
    init_mem_access = first_buffer_tile_size_K * first_buffer_tile_size_M * act_bits
    init_mem_access += first_buffer_tile_size_N * first_buffer_tile_size_K * weight_bits
    init_latency = ceil_a_by_b(init_mem_access, mem_width)
    stats.mem_stall_cycles += init_latency

    total_mem_access = stats.reads["dram"] + stats.writes["dram"]
    middle_mem_access = total_mem_access - init_mem_access
    middle_latency = ceil_a_by_b(middle_mem_access, mem_width)
    stats.mem_stall_cycles += max(0, middle_latency - stats.compute_cycles)
    stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles
    return stats


def input_stationary_systolic_cycles(M: int, N: int, K: int, array_height: int, array_width: int) -> int:
    return ceil_a_by_b(M, array_height) * ceil_a_by_b(K, array_width) * N + array_height + array_width


def prefill_systolic_cycles(M: int, N: int, K: int, array_height: int, array_width: int) -> int:
    return ceil_a_by_b(N, array_height) * ceil_a_by_b(K, array_width) * M + array_height + array_width
