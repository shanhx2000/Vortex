from __future__ import annotations

import math
from typing import Any, Dict, Optional

from helper import tensor_info_bits
from operations import GEMM, GEMV
from stats import Stats, init_cycle_breakdown

from .common import (
    add_buffer_reads,
    add_buffer_writes,
    add_output_reads,
    add_output_writes,
    add_parameter_reads,
    add_parameter_writes,
    arg_get,
    init_stats,
    ceil_a_by_b,
    weight_stationary_systolic_cycles,
)


def run_figlut_gemm(
    op: GEMM | GEMV,
    stats: Stats,
    *,
    array_height: int,
    array_width: int,
    bit_planes: int,
    mu: int,
    k: int,
    tile_size_m: int,
    tile_size_k: int,
    tile_size_n: int,
    tile_size_bit_planes: int,
    buffer_size_m: int,
    buffer_size_k: int,
    buffer_size_n: int,
    mem_width: int,
    act_nbits: Optional[int] = None,
    out_nbits: Optional[int] = None,
) -> Stats:
    M = int(op.M)
    K = int(op.K)
    N = int(op.N)
    act_bits = int(act_nbits if act_nbits is not None else tensor_info_bits(op.A_info))
    out_bits = int(out_nbits if out_nbits is not None else tensor_info_bits(op.out_info))

    if not hasattr(stats, "cycle_breakdown") or stats.cycle_breakdown is None:
        init_cycle_breakdown(stats)

    tile_num_M = ceil_a_by_b(M, tile_size_m)
    tile_num_K = ceil_a_by_b(K, tile_size_k)
    tile_num_N = ceil_a_by_b(N, tile_size_n)
    tile_num_bit_planes = ceil_a_by_b(bit_planes, tile_size_bit_planes)

    for m in range(tile_num_M):
        for n in range(tile_num_N):
            for k_tile in range(tile_num_K):
                for bit_plane_tile in range(tile_num_bit_planes):
                    current_tile_size_K = min(tile_size_k, K - k_tile * tile_size_k)
                    current_tile_size_M = min(tile_size_m, M - m * tile_size_m)
                    current_tile_size_N = min(tile_size_n, N - n * tile_size_n)
                    current_bit_planes = min(tile_size_bit_planes, bit_planes - bit_plane_tile * tile_size_bit_planes)
                    binary_weight_bits = current_tile_size_N * current_tile_size_K * current_bit_planes
                    current_lut_groups = ceil_a_by_b(current_tile_size_K, mu)

                    add_buffer_reads(stats, current_tile_size_K * current_tile_size_M * act_bits)
                    add_buffer_reads(stats, binary_weight_bits)
                    add_output_reads(stats, current_tile_size_M * current_tile_size_N * out_bits)
                    add_output_writes(stats, current_tile_size_M * current_tile_size_N * out_bits)
                    add_parameter_reads(stats, current_tile_size_N * act_bits * current_bit_planes)

                    lut_gen_cycles = 2
                    cur_output_size = ceil_a_by_b(current_tile_size_N, k)
                    array_compute_cycles = weight_stationary_systolic_cycles(
                        current_tile_size_M,
                        cur_output_size,
                        current_lut_groups,
                        array_height,
                        array_width,
                        pipeline_penalty=0,
                    )
                    array_ideal_cycles = ceil_a_by_b(
                        current_tile_size_M * cur_output_size * current_lut_groups,
                        array_height * array_width,
                    )
                    stats.cycle_breakdown["systolic array"] += array_compute_cycles
                    stats.cycle_breakdown["systolic array ideal"] = (
                        stats.cycle_breakdown.get("systolic array ideal", 0) + array_ideal_cycles
                    )
                    reduction_cycles = math.sqrt(bit_planes)
                    stats.compute_cycles += lut_gen_cycles + array_compute_cycles + reduction_cycles

    buffer_tile_num_M = ceil_a_by_b(M, buffer_size_m)
    buffer_tile_num_K = ceil_a_by_b(K, buffer_size_k)
    buffer_tile_num_N = ceil_a_by_b(N, buffer_size_n)

    scaling_factor_bits = N * act_bits * bit_planes
    stats.reads["dram"] += scaling_factor_bits
    add_parameter_writes(stats, scaling_factor_bits)

    offset_bits = N * out_bits
    stats.reads["dram"] += offset_bits
    add_parameter_writes(stats, offset_bits)

    for m in range(buffer_tile_num_M):
        for n in range(buffer_tile_num_N):
            current_buffer_tile_size_N = min(buffer_size_n, N - n * buffer_size_n)
            for k_tile in range(buffer_tile_num_K):
                for bit_plane_tile in range(tile_num_bit_planes):
                    current_buffer_tile_size_K = min(buffer_size_k, K - k_tile * buffer_size_k)
                    current_buffer_tile_size_M = min(buffer_size_m, M - m * buffer_size_m)
                    current_bit_planes = min(tile_size_bit_planes, bit_planes - bit_plane_tile * tile_size_bit_planes)
                    binary_weight_bits = current_buffer_tile_size_N * current_buffer_tile_size_K * current_bit_planes

                    stats.reads["dram"] += current_buffer_tile_size_K * current_buffer_tile_size_M * act_bits
                    add_buffer_writes(stats, current_buffer_tile_size_K * current_buffer_tile_size_M * act_bits)
                    stats.reads["dram"] += binary_weight_bits
                    add_buffer_writes(stats, binary_weight_bits)

    stats.reads["dram"] += M * N * out_bits
    add_output_writes(stats, M * N * out_bits)
    add_output_reads(stats, M * N * out_bits)
    stats.writes["dram"] += M * N * out_bits

    first_buffer_tile_size_K = min(tile_size_k, K)
    first_buffer_tile_size_M = min(tile_size_m, M)
    first_buffer_tile_size_N = min(tile_size_n, N)
    first_binary_weight_bits = first_buffer_tile_size_N * first_buffer_tile_size_K * bit_planes
    init_mem_access = first_buffer_tile_size_K * first_buffer_tile_size_M * act_bits
    init_mem_access += first_binary_weight_bits
    init_mem_access += scaling_factor_bits + offset_bits
    init_latency = ceil_a_by_b(init_mem_access, mem_width)
    stats.mem_stall_cycles += init_latency

    total_mem_access = stats.reads["dram"] + stats.writes["dram"]
    middle_mem_access = total_mem_access - init_mem_access
    middle_latency = ceil_a_by_b(middle_mem_access, mem_width)
    stats.mem_stall_cycles += max(0, middle_latency - stats.compute_cycles)
    stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles
    return stats


def run_figlut(
    op: GEMM | GEMV,
    stats: Optional[Stats] = None,
    args: Optional[Dict[str, Any]] = None,
    hw_config: Optional[Dict[str, Any]] = None,
) -> Stats:
    target = init_stats(stats, "figlut")
    runtime = (hw_config or {}).get("runtime", {})
    return run_figlut_gemm(
        op,
        target,
        array_height=arg_get(args, "array_height", int(runtime.get("array_height", 2))),
        array_width=arg_get(args, "array_width", int(runtime.get("array_width", 16))),
        bit_planes=arg_get(args, "bit_planes", int(runtime.get("bit_planes", 4))),
        mu=arg_get(args, "mu", int(runtime.get("mu", 4))),
        k=arg_get(args, "k", int(runtime.get("k_group", 8))),
        tile_size_m=arg_get(args, "tile_size_m", int(runtime.get("tile_size_m", 1024))),
        tile_size_k=arg_get(args, "tile_size_k", int(runtime.get("tile_size_k", 64))),
        tile_size_n=arg_get(args, "tile_size_n", int(runtime.get("tile_size_n", 16))),
        tile_size_bit_planes=arg_get(args, "tile_size_bit_planes", int(runtime.get("tile_size_bit_planes", 4))),
        buffer_size_m=arg_get(args, "buffer_size_m", int(runtime.get("buffer_size_m", 1024))),
        buffer_size_k=arg_get(args, "buffer_size_k", int(runtime.get("buffer_size_k", 64))),
        buffer_size_n=arg_get(args, "buffer_size_n", int(runtime.get("buffer_size_n", 64))),
        mem_width=arg_get(args, "mem_width", int(runtime.get("mem_width", 1024))),
        act_nbits=arg_get(args, "act_nbits", int(runtime.get("act_nbits", 16))),
        out_nbits=arg_get(args, "out_nbits", int(runtime.get("out_nbits", 16))),
    )
