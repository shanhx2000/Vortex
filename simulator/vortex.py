"""Cycle and memory-traffic model for the Vortex accelerator.

Each `vortex_gemm_*` function walks the tiling loops of one dataflow and
accumulates, into a Stats object, compute cycles, memory-stall cycles, and
per-memory-space read/write bit counts. Nothing is simulated at data level --
no tensors are allocated and no arithmetic is performed on values.

Two dataflows:

  MUF  matrix-unit-first. Small M. `vortex_gemm_MUF_M1` is a separate,
       more aggressive path for M == 1, where weights are used exactly once
       so partial-sum reuse need not be bounded by tile_n.
  LUF  LUT-first. Large M, and forced for `sv` (attention score x value) ops.
       Splits into `_mkn` and `_knm` loop orders at M == LUF_knm_2_mkn_M.

`_resolve_flow_and_runtime_args` picks the dataflow and its tiling parameters;
`run_vortex_gemm` is the entry point the hardware simulator calls.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
from operations import GEMM, GEMV
from stats import Stats, init_cycle_breakdown
from baselines.common import arg_get, init_stats, run_tiled_systolic_gemm
from baselines.common import (
    add_act_reads,
    add_act_writes,
    add_cap_reads,
    add_cap_writes,
    add_codebook_reads,
    add_codebook_writes,
    add_output_reads,
    add_output_writes,
    add_scale_writes,
    add_weight_reads,
    add_weight_writes,
    arg_get,
    init_stats,
    ceil_a_by_b,
    input_stationary_systolic_cycles,
    prefill_systolic_cycles,
)
from helper import ceil_div, deep_update, dtype_bits, tensor_info_bits_v2
from quantization import apply_aqlm, apply_cq

LUF_knm_2_mkn_M = 32


def vortex_gemm_LUF_mkn(op: GEMM | GEMV, stats: Optional[Stats] = None, args: Optional[Dict[str, Any]] = None, hw_config: Optional[Dict[str, Any]] = None):

    assert hw_config and args
    if stats is None: stats = Stats()
    if not hasattr(stats, "cycle_breakdown") or stats.cycle_breakdown is None: init_cycle_breakdown(stats)

    buf = args["buf_assignment"]

    M, K, N = int(op.M), int(op.K), int(op.N)
    act_bits, weight_bits, out_bits = tensor_info_bits_v2(op.A_info), tensor_info_bits_v2(op.B_info), tensor_info_bits_v2(op.out_info)

    # hw_config
    systolic, lut = hw_config["modules"]["systolic array"], hw_config["modules"]["lut array"]
    array_width, array_height, num_arrays = systolic["cols"], systolic["rows"], systolic["num"]
    assert num_arrays == 1, f"Only support single array for now. {systolic['num']}"
    lut_num, lut_block_size = lut["num"], lut["block_size"]
    lut_ports_rw, lut_ports_r = lut["num_ports"]["rw"], lut["num_ports"]["r"]
    lut_activated_ratio = args.get("lut_activated_ratio", 1.0)
    num_activated_lut = int(lut_num * lut_activated_ratio)
    mem_width = args.get("mem_width", 1024)

    # quantization
    qcfg = op.B_info["quantization_config"]; method = qcfg["method"]
    assert method in ["AQLM", "CQ"], f"Incorrect {method}."
    num_codebooks = qcfg["num_codebooks"] if method == "AQLM" else 1 if method == "CQ" else (_ for _ in ()).throw(ValueError(f"Unsupported quant method: {method}"))
    vector_size, num_entries = qcfg["vector_size"], qcfg["num_entries"]
    import math; from math import ceil
    index_n_bit, weight_fp_bit = ceil(math.log2(num_entries)), 16
    codebook_nbits = 16
    if method == "AQLM":
        assert num_codebooks*index_n_bit/vector_size == weight_bits
    if method == "CQ":
        assert index_n_bit/vector_size == weight_bits

    # tiling
    buffer_size_m, buffer_size_k, buffer_size_n = args["buffer_size_m"], args["buffer_size_k"], args["buffer_size_n"]
    tile_size_m, tile_size_k, tile_size_n = min(args["tile_size_m"], M), min(K, array_width), min(N, array_height)
    assert tile_size_m >= 32
    tile_size_m = min(tile_size_m, buffer_size_m)
    tile_size_k = min(tile_size_k, buffer_size_k)
    tile_size_n = min(tile_size_n, buffer_size_n)

    buffer_tile_num_M, buffer_tile_num_K, buffer_tile_num_N = ceil_a_by_b(M, buffer_size_m), ceil_a_by_b(K, buffer_size_k), ceil_a_by_b(N, buffer_size_n)
    tile_num_M, tile_num_K, tile_num_N = ceil_a_by_b(M, tile_size_m), ceil_a_by_b(K, tile_size_k), ceil_a_by_b(N, tile_size_n)


    if method == "AQLM":
        codebook_bits = num_codebooks * vector_size * num_entries * lut_block_size
        stats.reads["dram"] += codebook_bits
        stats.writes[f"s_cap{buf}"] = stats.writes.get(f"s_cap{buf}", 0) + codebook_bits * num_activated_lut

    for m in range(tile_num_M):
        current_tile_size_M = min(tile_size_m, M - m * tile_size_m)

        for k_tile in range(tile_num_K):
            current_tile_size_K = min(tile_size_k, K - k_tile * tile_size_k)

            if method == "CQ":
                num_codebooks_tile = ceil_a_by_b(current_tile_size_K, vector_size)
                codebook_bits = num_codebooks_tile * vector_size * num_entries * weight_fp_bit
                stats.reads[f"s_codebook{buf}"] = stats.reads.get(f"s_codebook{buf}", 0) + codebook_bits
                stats.writes[f"s_cap{buf}"] = stats.writes.get(f"s_cap{buf}", 0) + codebook_bits * num_activated_lut

            for n in range(tile_num_N):
                current_tile_size_N = min(tile_size_n, N - n * tile_size_n)

                # dequant
                stats.reads[f"s_wgt{buf}"] = stats.reads.get(f"s_wgt{buf}", 0) + current_tile_size_K * current_tile_size_N * weight_bits
                vector_num = ceil_a_by_b(current_tile_size_K, vector_size)
                dequant_ratio = ceil_a_by_b((vector_size * weight_fp_bit), lut_block_size)
                num_dequant_lookups = vector_num * current_tile_size_N * (num_codebooks if method == "AQLM" else 1)
                stats.reads[f"s_cap{buf}"] = stats.reads.get(f"s_cap{buf}", 0) + num_dequant_lookups * dequant_ratio * lut_block_size
                dequant_cycles = ceil_a_by_b(
                    (num_dequant_lookups * dequant_ratio),
                    (num_activated_lut * (lut_ports_rw + lut_ports_r))
                )
                stats.writes[f"s_wtmp{buf}"] = stats.writes.get(f"s_wtmp{buf}", 0) + current_tile_size_K * current_tile_size_N * weight_fp_bit

                if hasattr(stats, "cycle_breakdown"):
                    stats.cycle_breakdown["lut_array.rd"] += dequant_cycles
                    stats.cycle_breakdown["lut array"] += dequant_cycles

                stats.reads[f"s_act{buf}"] = stats.reads.get(f"s_act{buf}", 0) + current_tile_size_K * current_tile_size_M * act_bits
                stats.reads[f"s_wtmp{buf}"] = stats.reads.get(f"s_wtmp{buf}", 0) + current_tile_size_K * current_tile_size_N * weight_fp_bit

                gemm_cycles = prefill_systolic_cycles(
                    current_tile_size_M,
                    current_tile_size_N,
                    current_tile_size_K,
                    array_height,
                    array_width
                ) / num_arrays
                gemm_ideal_cycles = ceil_a_by_b(
                    current_tile_size_M * current_tile_size_N * current_tile_size_K,
                    array_height * array_width,
                ) / num_arrays

                if hasattr(stats, "cycle_breakdown"):
                    stats.cycle_breakdown["systolic array"] += gemm_cycles
                    stats.cycle_breakdown["systolic array ideal"] = (
                        stats.cycle_breakdown.get("systolic array ideal", 0) + gemm_ideal_cycles
                    )
                if k_tile > 0:
                    stats.reads[f"s_output{buf}"] = stats.reads.get(f"s_output{buf}", 0) + current_tile_size_M * current_tile_size_N * out_bits
                stats.writes[f"s_output{buf}"] = stats.writes.get(f"s_output{buf}", 0) + current_tile_size_M * current_tile_size_N * out_bits

                stats.compute_cycles += max(dequant_cycles, gemm_cycles)

    # buffer stage
    for m in range(buffer_tile_num_M):
        current_buffer_tile_size_M = min(buffer_size_m, M - m * buffer_size_m)

        for k_tile in range(buffer_tile_num_K):
            current_buffer_tile_size_K = min(buffer_size_k, K - k_tile * buffer_size_k)

            if method == "CQ":
                num_codebooks_tile = ceil_a_by_b(current_buffer_tile_size_K, vector_size)
                codebook_bits = num_codebooks_tile * vector_size * num_entries * weight_fp_bit
                stats.reads["dram"] += codebook_bits
                stats.writes[f"s_codebook{buf}"] = stats.writes.get(f"s_codebook{buf}", 0) + codebook_bits

            for n in range(buffer_tile_num_N):
                current_buffer_tile_size_N = min(buffer_size_n, N - n * buffer_size_n)

                stats.reads["dram"] += current_buffer_tile_size_N * ceil_a_by_b(current_buffer_tile_size_K, vector_size) * vector_size * weight_bits
                stats.writes[f"s_wgt{buf}"] = stats.writes.get(f"s_wgt{buf}", 0) + current_buffer_tile_size_N * ceil_a_by_b(current_buffer_tile_size_K, vector_size) * vector_size * weight_bits

                stats.reads["dram"] += current_buffer_tile_size_K * current_buffer_tile_size_M * act_bits
                stats.writes[f"s_act{buf}"] = stats.writes.get(f"s_act{buf}", 0) + current_buffer_tile_size_K * current_buffer_tile_size_M * act_bits

                if k_tile > 0:
                    stats.reads["dram"] += current_buffer_tile_size_M * current_buffer_tile_size_N * out_bits
                    stats.writes[f"s_output{buf}"] = stats.writes.get(f"s_output{buf}", 0) + current_buffer_tile_size_M * current_buffer_tile_size_N * out_bits

    stats.reads[f"s_output{buf}"] = stats.reads.get(f"s_output{buf}", 0) + M * N * out_bits
    stats.writes["dram"] += M * N * out_bits

    # memory stall (unchanged from the mkn variant)
    first_buffer_tile_size_K = min(tile_size_k, buffer_size_k)
    first_buffer_tile_size_M = min(tile_size_m, buffer_size_m)
    first_buffer_tile_codebook_size = num_codebooks

    init_mem_access = vector_size * num_entries * codebook_nbits * first_buffer_tile_codebook_size
    init_mem_access += first_buffer_tile_size_K * first_buffer_tile_size_M * act_bits

    init_latency = ceil_a_by_b(init_mem_access, mem_width)
    stats.mem_stall_cycles += init_latency

    total_mem_access = stats.reads["dram"] + stats.writes["dram"]
    middle_mem_access = total_mem_access - init_mem_access
    middle_latency = ceil_a_by_b(middle_mem_access, mem_width)

    stats.mem_stall_cycles += max(0, middle_latency - stats.compute_cycles)
    stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles
    return stats

def vortex_gemm_LUF_knm(op: GEMM | GEMV, stats: Optional[Stats] = None, args: Optional[Dict[str, Any]] = None, hw_config: Optional[Dict[str, Any]] = None):
    assert hw_config and args
    if stats is None: stats = Stats()
    if not hasattr(stats, "cycle_breakdown") or stats.cycle_breakdown is None: init_cycle_breakdown(stats)

    buf = args["buf_assignment"]

    M, K, N = int(op.M), int(op.K), int(op.N)
    act_bits, weight_bits, out_bits = tensor_info_bits_v2(op.A_info), tensor_info_bits_v2(op.B_info), tensor_info_bits_v2(op.out_info)

    # hw_config
    systolic, lut = hw_config["modules"]["systolic array"], hw_config["modules"]["lut array"]
    array_width, array_height, num_arrays = systolic["cols"], systolic["rows"], systolic["num"]
    assert num_arrays == 1, f"Only support single array for now. {systolic['num']}"
    lut_num, lut_block_size = lut["num"], lut["block_size"]
    lut_ports_rw, lut_ports_r = lut["num_ports"]["rw"], lut["num_ports"]["r"]
    lut_activated_ratio = args.get("lut_activated_ratio", 1.0)
    num_activated_lut = int(lut_num * lut_activated_ratio)
    mem_width = args.get("mem_width", 1024)

    # quantization
    qcfg = op.B_info["quantization_config"]; method = qcfg["method"]
    assert method in ["AQLM", "CQ"], f"Incorrect {method}."
    num_codebooks = qcfg["num_codebooks"] if method == "AQLM" else 1 if method == "CQ" else (_ for _ in ()).throw(ValueError(f"Unsupported quant method: {method}"))
    vector_size, num_entries = qcfg["vector_size"], qcfg["num_entries"]
    import math; from math import ceil
    index_n_bit, weight_fp_bit = ceil(math.log2(num_entries)), 16
    codebook_nbits = 16
    if method == "AQLM":
        assert num_codebooks*index_n_bit/vector_size == weight_bits
    if method == "CQ":
        assert index_n_bit/vector_size == weight_bits

    # tiling
    assert M <= array_height
    buffer_size_m, buffer_size_k, buffer_size_n = args["buffer_size_m"], args["buffer_size_k"], args["buffer_size_n"]
    tile_size_m, tile_size_k, tile_size_n = min(args["tile_size_m"], M), min(K, array_width), N

    tile_size_m = min(tile_size_m, buffer_size_m)
    tile_size_k = min(tile_size_k, buffer_size_k)
    tile_size_n = min(tile_size_n, buffer_size_n)

    buffer_tile_num_M, buffer_tile_num_K, buffer_tile_num_N = ceil_a_by_b(M, buffer_size_m), ceil_a_by_b(K, buffer_size_k), ceil_a_by_b(N, buffer_size_n)
    tile_num_M, tile_num_K, tile_num_N = ceil_a_by_b(M, tile_size_m), ceil_a_by_b(K, tile_size_k), ceil_a_by_b(N, tile_size_n)

    if method == "AQLM":
        codebook_bits = num_codebooks * vector_size * num_entries * lut_block_size
        stats.reads["dram"] += codebook_bits
        stats.writes[f"s_cap{buf}"] = stats.writes.get(f"s_cap{buf}", 0) + codebook_bits * num_activated_lut

    # =========================
    # compute
    # =========================
    for k_tile in range(tile_num_K):
        current_tile_size_K = min(tile_size_k, K - k_tile * tile_size_k)

        for n in range(tile_num_N):
            current_tile_size_N = min(tile_size_n, N - n * tile_size_n)

            if method == "CQ":
                num_codebooks_tile = ceil_a_by_b(current_tile_size_K, vector_size)
                codebook_bits = num_codebooks_tile * vector_size * num_entries * weight_fp_bit
                stats.reads[f"s_codebook{buf}"] = stats.reads.get(f"s_codebook{buf}", 0) + codebook_bits
                stats.writes[f"s_cap{buf}"] = stats.writes.get(f"s_cap{buf}", 0) + codebook_bits * num_activated_lut

            # dequant
            stats.reads[f"s_wgt{buf}"] = stats.reads.get(f"s_wgt{buf}", 0) + current_tile_size_K * current_tile_size_N * weight_bits
            vector_num = ceil_a_by_b(current_tile_size_K, vector_size)
            dequant_ratio = ceil_a_by_b((vector_size * weight_fp_bit), lut_block_size)
            num_dequant_lookups = vector_num * current_tile_size_N * (num_codebooks if method == "AQLM" else 1)
            stats.reads[f"s_cap{buf}"] = stats.reads.get(f"s_cap{buf}", 0) + num_dequant_lookups * dequant_ratio * lut_block_size

            dequant_cycles = ceil_a_by_b(
                (num_dequant_lookups * dequant_ratio),
                (num_activated_lut * (lut_ports_rw + lut_ports_r))
            )

            stats.writes[f"s_wtmp{buf}"] = stats.writes.get(f"s_wtmp{buf}", 0) + current_tile_size_K * current_tile_size_N * weight_fp_bit

            if hasattr(stats, "cycle_breakdown"):
                stats.cycle_breakdown["lut_array.rd"] += dequant_cycles
                stats.cycle_breakdown["lut array"] += dequant_cycles

            for m in range(tile_num_M):
                current_tile_size_M = min(tile_size_m, M - m * tile_size_m)

                # read A, once per (k, m) tile
                stats.reads[f"s_act{buf}"] = stats.reads.get(f"s_act{buf}", 0) + current_tile_size_K * current_tile_size_M * act_bits
                stats.reads[f"s_wtmp{buf}"] = stats.reads.get(f"s_wtmp{buf}", 0) + current_tile_size_K * current_tile_size_N * weight_fp_bit

                # GEMM
                gemm_cycles = input_stationary_systolic_cycles(
                    current_tile_size_M,
                    current_tile_size_N,
                    current_tile_size_K,
                    array_height,
                    array_width
                ) / num_arrays
                gemm_ideal_cycles = ceil_a_by_b(
                    current_tile_size_M * current_tile_size_N * current_tile_size_K,
                    array_height * array_width,
                ) / num_arrays

                if hasattr(stats, "cycle_breakdown"):
                    stats.cycle_breakdown["systolic array"] += gemm_cycles
                    stats.cycle_breakdown["systolic array ideal"] = (
                        stats.cycle_breakdown.get("systolic array ideal", 0) + gemm_ideal_cycles
                    )

                if k_tile > 0:
                    stats.reads[f"s_output{buf}"] = stats.reads.get(f"s_output{buf}", 0) + current_tile_size_M * current_tile_size_N * out_bits
                stats.writes[f"s_output{buf}"] = stats.writes.get(f"s_output{buf}", 0) + current_tile_size_M * current_tile_size_N * out_bits

                stats.compute_cycles += max(dequant_cycles, gemm_cycles)

    # =========================
    # buffer stage
    # =========================
    for k_tile in range(buffer_tile_num_K):
        current_buffer_tile_size_K = min(buffer_size_k, K - k_tile * buffer_size_k)

        if method == "CQ":
            num_codebooks_tile = ceil_a_by_b(current_buffer_tile_size_K, vector_size)
            codebook_bits = num_codebooks_tile * vector_size * num_entries * weight_fp_bit
            stats.reads["dram"] += codebook_bits
            stats.writes[f"s_codebook{buf}"] = stats.writes.get(f"s_codebook{buf}", 0) + codebook_bits

        for n in range(buffer_tile_num_N):
            current_buffer_tile_size_N = min(buffer_size_n, N - n * buffer_size_n)

            stats.reads["dram"] += current_buffer_tile_size_N * ceil_a_by_b(current_buffer_tile_size_K, vector_size) * vector_size * weight_bits
            stats.writes[f"s_wgt{buf}"] = stats.writes.get(f"s_wgt{buf}", 0) + current_buffer_tile_size_N * ceil_a_by_b(current_buffer_tile_size_K, vector_size) * vector_size * weight_bits

            for m in range(buffer_tile_num_M):
                current_buffer_tile_size_M = min(buffer_size_m, M - m * buffer_size_m)

                stats.reads["dram"] += current_buffer_tile_size_K * current_buffer_tile_size_M * act_bits
                stats.writes[f"s_act{buf}"] = stats.writes.get(f"s_act{buf}", 0) + current_buffer_tile_size_K * current_buffer_tile_size_M * act_bits

                if k_tile > 0:
                    stats.reads["dram"] += current_buffer_tile_size_M * current_buffer_tile_size_N * out_bits
                    stats.writes[f"s_output{buf}"] = stats.writes.get(f"s_output{buf}", 0) + current_buffer_tile_size_M * current_buffer_tile_size_N * out_bits

    stats.reads[f"s_output{buf}"] = stats.reads.get(f"s_output{buf}", 0) + M * N * out_bits
    stats.writes["dram"] += M * N * out_bits

    # memory stall (unchanged from the mkn variant)
    first_buffer_tile_size_K = min(tile_size_k, buffer_size_k)
    first_buffer_tile_size_M = min(tile_size_m, buffer_size_m)
    first_buffer_tile_codebook_size = num_codebooks

    init_mem_access = vector_size * num_entries * codebook_nbits * first_buffer_tile_codebook_size
    init_mem_access += first_buffer_tile_size_K * first_buffer_tile_size_M * act_bits

    init_latency = ceil_a_by_b(init_mem_access, mem_width)
    stats.mem_stall_cycles += init_latency

    total_mem_access = stats.reads["dram"] + stats.writes["dram"]
    middle_mem_access = total_mem_access - init_mem_access
    middle_latency = ceil_a_by_b(middle_mem_access, mem_width)

    stats.mem_stall_cycles += max(0, middle_latency - stats.compute_cycles)
    stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles
    return stats

def vortex_gemm_LUF(op: GEMM | GEMV, stats: Optional[Stats] = None, args: Optional[Dict[str, Any]] = None, hw_config: Optional[Dict[str, Any]] = None):
    if op.M > LUF_knm_2_mkn_M:
        return vortex_gemm_LUF_mkn(op, stats, args, hw_config)
    else:
        return vortex_gemm_LUF_knm(op, stats, args, hw_config)


def vortex_gemm_MUF_M1(
    op: GEMM | GEMV,
    stats: Optional[Stats] = None,
    args: Optional[Dict[str, Any]] = None,
    hw_config: Optional[Dict[str, Any]] = None,
):
    assert hw_config and args

    if stats is None:
        stats = Stats()
    if not hasattr(stats, "cycle_breakdown") or stats.cycle_breakdown is None:
        init_cycle_breakdown(stats)

    M, K, N = int(op.M), int(op.K), int(op.N)
    assert M == 1

    # ------------------------
    # bitwidth
    # ------------------------
    act_bits = tensor_info_bits_v2(op.A_info)
    out_bits = tensor_info_bits_v2(op.out_info)
    # ------------------------
    # quant config
    # ------------------------
    assert "quantization_config" in op.B_info, f"Expected quantization config for op {op.name}"
    assert op.B_info["quantization_config"]["method"] in ["AQLM", "CQ"], f"Incorrect quantization method for op {op.name}"
    assert "sv" not in op.name.lower(), f"Expected no CQ quantization for sv op {op.name} processed in MUF."
    qcfg = op.B_info["quantization_config"]
    method = qcfg["method"]
    num_codebooks = (
        qcfg["num_codebooks"] if method == "AQLM"
        else 1 if method == "CQ"
        else (_ for _ in ()).throw(ValueError(f"Unsupported quant method: {method}"))
    )
    vector_size = qcfg["vector_size"]
    num_entries = qcfg["num_entries"]
    import math
    index_nbits = math.ceil(math.log2(num_entries))
    codebook_nbits = 16 # per element

    # ------------------------
    # HW config
    # ------------------------
    systolic = hw_config["modules"]["systolic array"]
    total_cols = systolic["cols"] * systolic["num"]
    array_width = systolic["cols"]
    array_height = systolic["rows"]
    mem_width = args.get("mem_width", 1024)
    lut = hw_config["modules"]["lut array"]
    activated_lut_ratio = 1.0
    adder_tree = hw_config["modules"]["adder tree"]

    # ------------------------
    # tiling
    # ------------------------
    buf = args["buf_assignment"]
    assert buf == "_lt"
    tile_size_m = M
    tile_size_k = K
    tile_size_n = N

    buffer_size_m = M
    buffer_size_k = K
    buffer_size_n = N
    tile_size_codebook = args.get("tile_size_codebook", 1)

    tile_num_M = ceil_a_by_b(M, tile_size_m)
    tile_num_K = ceil_a_by_b(K, tile_size_k)
    tile_num_N = ceil_a_by_b(N, tile_size_n)
    tile_num_codebook = ceil_a_by_b(num_codebooks, tile_size_codebook)

    # ============================================================
    # sparsity
    # ============================================================
    sparsity_info = op.metadata.get("sparsity_info", None)
    if sparsity_info is None:
        sparsity_list = [0.0 for _ in range(num_codebooks)]
    else:
        sparsity_list = sparsity_info.get("contextual_sparsity", [0.0 for _ in range(num_codebooks)])
        assert len(sparsity_list) == num_codebooks or len(sparsity_list) == 1, \
            f"Expected sparsity list of length {num_codebooks} or 1, but got {len(sparsity_list)}. op={op}"
        if len(sparsity_list) == 1 and num_codebooks > 1:
            sparsity_list = [sparsity_list[0] for _ in range(num_codebooks)]

    # ============================================================
    # pipeline fill
    # ============================================================
    first_vector_num = array_height
    first_codebook_size = 1
    first_tile_size_M = 1
    init_cycles = input_stationary_systolic_cycles(
        array_height,
        num_entries * first_codebook_size,
        vector_size,
        array_height,
        array_width,
    ) * ceil_a_by_b(first_vector_num * first_tile_size_M, array_height)
    stats.compute_cycles += init_cycles

    # ============================================================
    # main loop
    # ============================================================

    for m in range(tile_num_M):
        current_tile_size_M = min(tile_size_m, M - m * tile_size_m)
        for k in range(tile_num_K):
            current_tile_size_K = min(tile_size_k, K - k * tile_size_k)
            vector_num = ceil_a_by_b(current_tile_size_K, vector_size)
            for n in range(tile_num_N):
                current_tile_size_N = min(tile_size_n, N - n * tile_size_n)
                for codebook in range(tile_num_codebook):
                    current_tile_size_codebook = min(
                        tile_size_codebook,
                        num_codebooks - codebook * tile_size_codebook,
                    )
                    # ====================================================
                    # LUT compute
                    # ====================================================
                    parallelism_in_N = args.get("parallelism_in_N", None)
                    activated_lut = lut["num"] * activated_lut_ratio
                    # load act + codebook
                    stats.reads[f"s_act{buf}"] = stats.reads.get(f"s_act{buf}",0) + current_tile_size_K * current_tile_size_M * act_bits
                    codebook_bits = (
                        current_tile_size_codebook * vector_size * num_entries
                        * ceil_a_by_b(current_tile_size_M * vector_num, array_height)
                        * codebook_nbits
                    )
                    stats.reads[f's_codebook{buf}'] = stats.reads.get(f's_codebook{buf}',0)+current_tile_size_codebook * vector_size * num_entries * ceil_a_by_b(current_tile_size_M * vector_num, array_height) * codebook_nbits

                    # Contextual sparsity: fewer active vectors per codebook,
                    # floored at array_height -- a partially filled array still
                    # costs a full pass, so sparsity below one array's worth
                    # buys nothing.
                    assert current_tile_size_codebook == 1, "Only support this for now."
                    sparse_current_tile_M_vector_num = current_tile_size_M * \
                        max(vector_num * (1.0 - sparsity_list[codebook]), array_height)
                    sparse_vector_num = max(vector_num * (1.0 - sparsity_list[codebook]), array_height)

                    # systolic cycles (lookup compute)
                    systolic_array_cycles = input_stationary_systolic_cycles(
                        array_height,
                        num_entries,
                        vector_size,
                        array_height,
                        array_width,
                    ) * ceil_a_by_b(
                        # ceil_a_by_b(sparse_current_tile_M_vector_num, array_height) * current_tile_size_codebook,
                        ceil_a_by_b(sparse_vector_num, array_height) * current_tile_size_M * current_tile_size_codebook,
                        # ceil_a_by_b(vector_num, array_height) * current_tile_size_M * current_tile_size_codebook,
                        systolic["num"]
                    )
                    systolic_array_ideal_cycles = ceil_a_by_b(
                        array_height * num_entries * vector_size,
                        array_height * array_width,
                    ) * ceil_a_by_b(
                        ceil_a_by_b(sparse_vector_num, array_height) * current_tile_size_M * current_tile_size_codebook,
                        systolic["num"]
                    )

                    # LUT write (parallel across N)
                    if parallelism_in_N is None:
                        parallelism_in_N = max(
                            1,
                            lut["num"] / (sparse_current_tile_M_vector_num*current_tile_size_codebook)
                        ) * (lut["num_ports"]["rw"] + lut["num_ports"]["r"])

                    penalty_factor_in_N = 1
                    lut_tile_size = (sparse_current_tile_M_vector_num * current_tile_size_codebook)
                    lut_read_bwd = (lut["num_ports"]["rw"] + lut["num_ports"]["r"])
                    if lut["num"]/lut_tile_size < parallelism_in_N / lut_read_bwd:
                        penalty_factor_in_N = ceil_a_by_b(lut_tile_size, lut["num"])
                    lut_writes = (
                        sparse_current_tile_M_vector_num * current_tile_size_codebook
                        * num_entries * parallelism_in_N
                    )
                    stats.writes[f's_cap{buf}'] = stats.writes.get(f's_cap{buf}',0)+lut_writes * act_bits

                    lut_write_cycles = 0
                    # assume double-buffer

                    # LUT read (parallel)
                    num_lookups = (
                        sparse_current_tile_M_vector_num * current_tile_size_codebook
                        * current_tile_size_N
                    )
                    stats.reads[f's_cap{buf}'] = stats.reads.get(f's_cap{buf}',0)+num_lookups * act_bits

                    stats.reads[f's_wgt{buf}'] = stats.reads.get(f's_wgt{buf}',0)+N * vector_num * index_nbits * current_tile_size_codebook

                    lut_read_cycles = ceil_a_by_b(current_tile_size_N , parallelism_in_N) * penalty_factor_in_N

                    assert lut_read_cycles > lut_write_cycles # double buffer
                    lut_cycles = lut_write_cycles + lut_read_cycles

                    # accumulate (adder tree)
                    num_accumulates = (
                        sparse_current_tile_M_vector_num
                        * current_tile_size_codebook * current_tile_size_N
                    )
                    accumulate_cycles = ceil_a_by_b(num_accumulates , (adder_tree["num"] * adder_tree["width"]))

                    # output
                    if codebook > 0 or k > 0:
                        stats.reads[f's_output{buf}'] = stats.reads.get(f's_output{buf}',0)+current_tile_size_M * current_tile_size_N * out_bits
                    stats.writes[f's_output{buf}'] = stats.writes.get(f's_output{buf}',0)+current_tile_size_M * current_tile_size_N * out_bits

                    # critical path
                    tile_cycles = max(systolic_array_cycles, lut_cycles, accumulate_cycles)
                    stats.compute_cycles += tile_cycles

                    # breakdown
                    stats.cycle_breakdown["systolic array"] += systolic_array_cycles
                    stats.cycle_breakdown["systolic array ideal"] = (
                        stats.cycle_breakdown.get("systolic array ideal", 0) + systolic_array_ideal_cycles
                    )
                    stats.cycle_breakdown["lut array"] += lut_cycles
                    stats.cycle_breakdown["lut_array.rd"] += lut_read_cycles
                    stats.cycle_breakdown["lut_array.wr"] += lut_write_cycles
                    stats.cycle_breakdown["adder tree"] += accumulate_cycles

    # ============================================================
    # DRAM
    # ============================================================
    if method == "AQLM":
        stats.reads["dram"] += num_codebooks * vector_size * num_entries * codebook_nbits
        stats.writes[f's_codebook{buf}'] = stats.writes.get(f's_codebook{buf}',0) + num_codebooks * vector_size * num_entries * codebook_nbits

    stats.reads["dram"] += N * act_bits

    # ============================================================
    # buffer loop
    # ============================================================
    buffer_tile_num_M = ceil_a_by_b(M, buffer_size_m)
    buffer_tile_num_K = ceil_a_by_b(K, buffer_size_k)
    buffer_tile_num_N = ceil_a_by_b(N, buffer_size_n)

    for m in range(buffer_tile_num_M):
        for k in range(buffer_tile_num_K):
            current_buffer_tile_size_K = min(buffer_size_k, K - k * buffer_size_k)
            current_buffer_tile_size_M = min(buffer_size_m, M - m * buffer_size_m)
            buffer_vector_num = ceil_a_by_b(current_buffer_tile_size_K, vector_size)

            if method == "CQ":
                # every vector along K carries its own codebook
                codebook_loads = buffer_vector_num
                codebook_bits_per_tile = (
                    codebook_loads * vector_size * num_entries * codebook_nbits
                )
                stats.reads["dram"] += codebook_bits_per_tile
                stats.writes[f's_codebook{buf}'] = stats.writes.get(f's_codebook{buf}',0) + codebook_bits_per_tile

            stats.reads["dram"] += current_buffer_tile_size_K * current_buffer_tile_size_M * act_bits
            stats.writes[f's_act{buf}'] = stats.writes.get(f's_act{buf}',0) + current_buffer_tile_size_K * current_buffer_tile_size_M * act_bits

            for n in range(buffer_tile_num_N):
                current_buffer_tile_size_N = min(buffer_size_n, N - n * buffer_size_n)

                for codebook in range(num_codebooks):
                    sparse_buffer_vector_num = buffer_vector_num * (1.0 - sparsity_list[codebook]**(current_buffer_tile_size_M))
                    # assume sparsity is uniformly distributed, weight is not read only when all tile_M of input are pruned. .
                    stats.reads["dram"] += current_buffer_tile_size_N * sparse_buffer_vector_num * index_nbits
                    stats.writes[f's_wgt{buf}'] = stats.writes.get(f's_wgt{buf}',0) + current_buffer_tile_size_N * sparse_buffer_vector_num * index_nbits

                if k > 0:
                    stats.reads["dram"] += current_buffer_tile_size_M * current_buffer_tile_size_N * out_bits
                    stats.writes[f's_output{buf}'] = stats.writes.get(f's_output{buf}',0) + current_buffer_tile_size_M * current_buffer_tile_size_N * out_bits

                stats.reads[f's_output{buf}'] = stats.reads.get(f's_output{buf}',0) + current_buffer_tile_size_M * current_buffer_tile_size_N * out_bits
                stats.writes["dram"] += current_buffer_tile_size_M * current_buffer_tile_size_N * out_bits

    # ============================================================
    # memory stall
    # ============================================================
    first_buffer_tile_size_K = min(tile_size_k, K)
    first_buffer_tile_size_M = min(tile_size_m, M)
    first_buffer_tile_codebook_size = min(tile_size_codebook, num_codebooks)
    init_mem_access = vector_size * num_entries * codebook_nbits * first_buffer_tile_codebook_size
    init_mem_access += first_buffer_tile_size_K * first_buffer_tile_size_M * act_bits
    init_latency = ceil_a_by_b(init_mem_access, mem_width)
    stats.mem_stall_cycles += init_latency
    total_mem_access = stats.reads["dram"] + stats.writes["dram"]
    middle_mem_access = total_mem_access - init_mem_access
    middle_latency = ceil_a_by_b(middle_mem_access, mem_width)
    stats.mem_stall_cycles += max(0, middle_latency - stats.compute_cycles)
    stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles
    return stats

def vortex_gemm_MUF(
    op: GEMM | GEMV,
    stats: Optional[Stats] = None,
    args: Optional[Dict[str, Any]] = None,
    hw_config: Optional[Dict[str, Any]] = None,
):
    assert hw_config and args

    if stats is None:
        stats = Stats()
    if not hasattr(stats, "cycle_breakdown") or stats.cycle_breakdown is None:
        init_cycle_breakdown(stats)

    M, K, N = int(op.M), int(op.K), int(op.N)

    if M == 1:
        # Why M=1 is a special case?
        # (1) All weights are only used once, thus the reuse of partial sums don't need to be limited by tile_n. It can just stream weights it needs. 
        # (2) The buffer size of when M=1 allows larger buffer_k, buffer_n.
        return vortex_gemm_MUF_M1(op=op, stats=stats, args=args, hw_config=hw_config)

    # ------------------------
    # bitwidth
    # ------------------------
    act_bits = tensor_info_bits_v2(op.A_info)
    out_bits = tensor_info_bits_v2(op.out_info)
    # ------------------------
    # quant config
    # ------------------------
    assert "quantization_config" in op.B_info, f"Expected quantization config for op {op.name}"
    assert op.B_info["quantization_config"]["method"] in ["AQLM", "CQ"], f"Incorrect quantization method for op {op.name}"
    assert "sv" not in op.name.lower(), f"Expected no CQ quantization for sv op {op.name} processed in MUF."

    qcfg = op.B_info["quantization_config"]
    method = qcfg["method"]
    num_codebooks = (
        qcfg["num_codebooks"] if method == "AQLM"
        else 1 if method == "CQ"
        else (_ for _ in ()).throw(ValueError(f"Unsupported quant method: {method}"))
    )
    vector_size = qcfg["vector_size"]
    num_entries = qcfg["num_entries"]
    import math
    index_nbits = math.ceil(math.log2(num_entries))
    codebook_nbits = 16 # per element

    # ------------------------
    # HW config
    # ------------------------
    systolic = hw_config["modules"]["systolic array"]
    total_cols = systolic["cols"] * systolic["num"]
    array_width = systolic["cols"]
    array_height = systolic["rows"]
    mem_width = args.get("mem_width", 1024)
    lut = hw_config["modules"]["lut array"]
    activated_lut_ratio = 1.0
    adder_tree = hw_config["modules"]["adder tree"]

    # ------------------------
    # tiling
    # ------------------------
    buf = args["buf_assignment"]

    buffer_size_m = args["buffer_size_m"]
    buffer_size_k = args["buffer_size_k"]
    buffer_size_n = args["buffer_size_n"]

    tile_size_m = args["tile_size_m"]
    tile_size_n = args.get("tile_size_n", N)
    tile_size_n = min(tile_size_n, buffer_size_n)
    tile_size_codebook = args.get("tile_size_codebook", 1)
    tile_size_k = buffer_size_k

    tile_num_M = ceil_a_by_b(M, tile_size_m)
    tile_num_K = ceil_a_by_b(K, tile_size_k)
    tile_num_N = ceil_a_by_b(N, tile_size_n)
    tile_num_codebook = ceil_a_by_b(num_codebooks, tile_size_codebook)

    # ============================================================
    # sparsity
    # ============================================================
    sparsity_info = op.metadata.get("sparsity_info", None)
    if sparsity_info is None:
        sparsity_list = [0.0 for _ in range(num_codebooks)]
    else:
        sparsity_list = sparsity_info.get("contextual_sparsity", [0.0 for _ in range(num_codebooks)])
        assert len(sparsity_list) == num_codebooks or len(sparsity_list) == 1, \
            f"Expected sparsity list of length {num_codebooks} or 1, but got {len(sparsity_list)}. op={op}"
        if len(sparsity_list) == 1 and num_codebooks > 1:
            sparsity_list = [sparsity_list[0] for _ in range(num_codebooks)]

    # ============================================================
    # pipeline fill
    # ============================================================
    first_vector_num = array_height
    first_codebook_size = 1
    first_tile_size_M = 1
    init_cycles = input_stationary_systolic_cycles(
        array_height,
        num_entries * first_codebook_size,
        vector_size,
        array_height,
        array_width,
    ) * ceil_a_by_b(first_vector_num * first_tile_size_M, array_height)
    stats.compute_cycles += init_cycles

    # ============================================================
    # main loop
    # ============================================================

    for m in range(tile_num_M):
        current_tile_size_M = min(tile_size_m, M - m * tile_size_m)
        for k in range(tile_num_K):
            current_tile_size_K = min(tile_size_k, K - k * tile_size_k)
            vector_num = ceil_a_by_b(current_tile_size_K, vector_size)

            for n in range(tile_num_N):
                current_tile_size_N = min(tile_size_n, N - n * tile_size_n)

                for codebook in range(tile_num_codebook):
                    current_tile_size_codebook = min(
                        tile_size_codebook,
                        num_codebooks - codebook * tile_size_codebook,
                    )

                    # ====================================================
                    # LUT compute
                    # ====================================================

                    parallelism_in_N = args.get("parallelism_in_N", None)
                    activated_lut = lut["num"] * activated_lut_ratio

                    # load act + codebook
                    stats.reads[f"s_act{buf}"] = stats.reads.get(f"s_act{buf}",0) + current_tile_size_K * current_tile_size_M * act_bits
                    codebook_bits = (
                        current_tile_size_codebook * vector_size * num_entries
                        * ceil_a_by_b(current_tile_size_M * vector_num, array_height)
                        * codebook_nbits
                    )
                    stats.reads[f's_codebook{buf}'] = stats.reads.get(f's_codebook{buf}',0)+current_tile_size_codebook * vector_size * num_entries * ceil_a_by_b(current_tile_size_M * vector_num, array_height) * codebook_nbits

                    # Contextual sparsity: fewer active vectors per codebook,
                    # floored at array_height -- a partially filled array still
                    # costs a full pass, so sparsity below one array's worth
                    # buys nothing.
                    assert current_tile_size_codebook == 1, "Only support this for now."
                    sparse_current_tile_M_vector_num = current_tile_size_M * \
                        max(vector_num * (1.0 - sparsity_list[codebook]), array_height)
                    sparse_vector_num = max(vector_num * (1.0 - sparsity_list[codebook]), array_height)

                    # systolic cycles (lookup compute)
                    systolic_array_cycles = input_stationary_systolic_cycles(
                        array_height,
                        num_entries,
                        vector_size,
                        array_height,
                        array_width,
                    ) * ceil_a_by_b(
                        # ceil_a_by_b(sparse_current_tile_M_vector_num, array_height) * current_tile_size_codebook,
                        ceil_a_by_b(sparse_vector_num, array_height) * current_tile_size_M * current_tile_size_codebook,
                        # ceil_a_by_b(vector_num, array_height) * current_tile_size_M * current_tile_size_codebook,
                        systolic["num"]
                    )
                    systolic_array_ideal_cycles = ceil_a_by_b(
                        array_height * num_entries * vector_size,
                        array_height * array_width,
                    ) * ceil_a_by_b(
                        ceil_a_by_b(sparse_vector_num, array_height) * current_tile_size_M * current_tile_size_codebook,
                        systolic["num"]
                    )

                    # LUT write (parallel across N)
                    if parallelism_in_N is None:
                        parallelism_in_N = max(
                            1,
                            lut["num"] / (sparse_current_tile_M_vector_num*current_tile_size_codebook)
                        ) * (lut["num_ports"]["rw"] + lut["num_ports"]["r"])


                    penalty_factor_in_N = 1
                    lut_tile_size = (sparse_current_tile_M_vector_num * current_tile_size_codebook)
                    lut_read_bwd = (lut["num_ports"]["rw"] + lut["num_ports"]["r"])
                    if lut["num"]/lut_tile_size < parallelism_in_N / lut_read_bwd:
                        penalty_factor_in_N = ceil_a_by_b(lut_tile_size, lut["num"])
                    lut_writes = (
                        sparse_current_tile_M_vector_num * current_tile_size_codebook
                        * num_entries * parallelism_in_N
                    )
                    stats.writes[f's_cap{buf}'] = stats.writes.get(f's_cap{buf}',0)+lut_writes * act_bits
                    lut_write_cycles = 0
                    # assume double-buffer

                    # LUT read (parallel)
                    num_lookups = (
                        sparse_current_tile_M_vector_num * current_tile_size_codebook
                        * current_tile_size_N
                    )
                    stats.reads[f's_cap{buf}'] = stats.reads.get(f's_cap{buf}',0)+num_lookups * act_bits

                    stats.reads[f's_wgt{buf}'] = stats.reads.get(f's_wgt{buf}',0)+N * vector_num * index_nbits * current_tile_size_codebook

                    lut_read_cycles = ceil_a_by_b(current_tile_size_N , parallelism_in_N) * penalty_factor_in_N

                    lut_cycles = lut_write_cycles + lut_read_cycles

                    # accumulate (adder tree)
                    num_accumulates = (
                        sparse_current_tile_M_vector_num
                        * current_tile_size_codebook * current_tile_size_N
                    )
                    accumulate_cycles = ceil_a_by_b(num_accumulates , (adder_tree["num"] * adder_tree["width"]))


                    # output
                    if codebook > 0 or k > 0:
                        stats.reads[f's_output{buf}'] = stats.reads.get(f's_output{buf}',0)+current_tile_size_M * current_tile_size_N * out_bits
                    stats.writes[f's_output{buf}'] = stats.writes.get(f's_output{buf}',0)+current_tile_size_M * current_tile_size_N * out_bits

                    # critical path
                    tile_cycles = max(systolic_array_cycles, lut_cycles, accumulate_cycles)
                    stats.compute_cycles += tile_cycles

                    # breakdown
                    stats.cycle_breakdown["systolic array"] += systolic_array_cycles
                    stats.cycle_breakdown["systolic array ideal"] = (
                        stats.cycle_breakdown.get("systolic array ideal", 0) + systolic_array_ideal_cycles
                    )
                    stats.cycle_breakdown["lut array"] += lut_cycles
                    stats.cycle_breakdown["lut_array.rd"] += lut_read_cycles
                    stats.cycle_breakdown["lut_array.wr"] += lut_write_cycles
                    stats.cycle_breakdown["adder tree"] += accumulate_cycles

    # ============================================================
    # DRAM
    # ============================================================
    if method == "AQLM":
        stats.reads["dram"] += num_codebooks * vector_size * num_entries * codebook_nbits
        stats.writes[f's_codebook{buf}'] = stats.writes.get(f's_codebook{buf}',0) + num_codebooks * vector_size * num_entries * codebook_nbits

    stats.reads["dram"] += N * act_bits

    # ============================================================
    # buffer loop
    # ============================================================
    buffer_tile_num_M = ceil_a_by_b(M, buffer_size_m)
    buffer_tile_num_K = ceil_a_by_b(K, buffer_size_k)
    buffer_tile_num_N = ceil_a_by_b(N, buffer_size_n)

    for m in range(buffer_tile_num_M):
        for k in range(buffer_tile_num_K):
            current_buffer_tile_size_K = min(buffer_size_k, K - k * buffer_size_k)
            current_buffer_tile_size_M = min(buffer_size_m, M - m * buffer_size_m)
            buffer_vector_num = ceil_a_by_b(current_buffer_tile_size_K, vector_size)

            if method == "CQ":
                # every vector along K carries its own codebook
                codebook_loads = buffer_vector_num
                codebook_bits_per_tile = (
                    codebook_loads * vector_size * num_entries * codebook_nbits
                )
                stats.reads["dram"] += codebook_bits_per_tile
                stats.writes[f's_codebook{buf}'] = stats.writes.get(f's_codebook{buf}',0) + codebook_bits_per_tile

            stats.reads["dram"] += current_buffer_tile_size_K * current_buffer_tile_size_M * act_bits
            stats.writes[f's_act{buf}'] = stats.writes.get(f's_act{buf}',0) + current_buffer_tile_size_K * current_buffer_tile_size_M * act_bits

            for n in range(buffer_tile_num_N):
                current_buffer_tile_size_N = min(buffer_size_n, N - n * buffer_size_n)

                for codebook in range(num_codebooks):
                    sparse_buffer_vector_num = buffer_vector_num * (1.0 - sparsity_list[codebook]**(current_buffer_tile_size_M))
                    # assume sparsity is uniformly distributed, weight is not read only when all tile_M of input are pruned. .
                    stats.reads["dram"] += current_buffer_tile_size_N * sparse_buffer_vector_num * index_nbits
                    stats.writes[f's_wgt{buf}'] = stats.writes.get(f's_wgt{buf}',0) + current_buffer_tile_size_N * sparse_buffer_vector_num * index_nbits

                if k > 0:
                    stats.reads["dram"] += current_buffer_tile_size_M * current_buffer_tile_size_N * out_bits
                    stats.writes[f's_output{buf}'] = stats.writes.get(f's_output{buf}',0) + current_buffer_tile_size_M * current_buffer_tile_size_N * out_bits

                stats.reads[f's_output{buf}'] = stats.reads.get(f's_output{buf}',0) + current_buffer_tile_size_M * current_buffer_tile_size_N * out_bits
                stats.writes["dram"] += current_buffer_tile_size_M * current_buffer_tile_size_N * out_bits

    # ============================================================
    # memory stall
    # ============================================================
    first_buffer_tile_size_K = min(tile_size_k, K)
    first_buffer_tile_size_M = min(tile_size_m, M)
    first_buffer_tile_codebook_size = min(tile_size_codebook, num_codebooks)
    init_mem_access = vector_size * num_entries * codebook_nbits * first_buffer_tile_codebook_size
    init_mem_access += first_buffer_tile_size_K * first_buffer_tile_size_M * act_bits
    init_latency = ceil_a_by_b(init_mem_access, mem_width)
    stats.mem_stall_cycles += init_latency
    total_mem_access = stats.reads["dram"] + stats.writes["dram"]
    middle_mem_access = total_mem_access - init_mem_access
    middle_latency = ceil_a_by_b(middle_mem_access, mem_width)
    stats.mem_stall_cycles += max(0, middle_latency - stats.compute_cycles)
    stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles
    return stats


def run_gemm_systolic_array(
    op: GEMM | GEMV,
    stats: Optional[Stats] = None,
    args: Optional[Dict[str, Any]] = None,
    hw_config: Optional[Dict[str, Any]] = None,
) -> Stats:
    assert hw_config and args
    stats = init_stats(stats, "systolic_array")

    if not hasattr(stats, "cycle_breakdown") or stats.cycle_breakdown is None:
        init_cycle_breakdown(stats)

    buf = args["buf_assignment"]

    M, K, N = int(op.M), int(op.K), int(op.N)

    act_bits = tensor_info_bits_v2(op.A_info)
    weight_bits = tensor_info_bits_v2(op.B_info)
    out_bits = tensor_info_bits_v2(op.out_info)

    systolic = hw_config["modules"]["systolic array"]
    H, W, num_arrays = systolic["rows"], systolic["cols"], systolic["num"]
    assert num_arrays == 1

    mem_width = args.get("mem_width", 1024)

    buffer_size_m = args.get("buffer_size_m", 512)
    buffer_size_k = args.get("buffer_size_k", 32)
    buffer_size_n = args.get("buffer_size_n", 32)

    tile_size_m = min(args.get("tile_size_m", H), M, buffer_size_m)
    tile_size_k = min(K, W, buffer_size_k)
    tile_size_n = min(N, H, buffer_size_n)

    tile_num_M = ceil_a_by_b(M, tile_size_m)
    tile_num_K = ceil_a_by_b(K, tile_size_k)
    tile_num_N = ceil_a_by_b(N, tile_size_n)

    buffer_tile_num_M = ceil_a_by_b(M, buffer_size_m)
    buffer_tile_num_K = ceil_a_by_b(K, buffer_size_k)
    buffer_tile_num_N = ceil_a_by_b(N, buffer_size_n)

    # ============================================================
    # compute
    # ============================================================

    if M > 32:
        for m in range(tile_num_M):
            cur_M = min(tile_size_m, M - m * tile_size_m)

            for k_tile in range(tile_num_K):
                cur_K = min(tile_size_k, K - k_tile * tile_size_k)

                # A read
                stats.reads[f"s_act{buf}"] = stats.reads.get(f"s_act{buf}", 0) + cur_M * cur_K * act_bits

                for n in range(tile_num_N):
                    cur_N = min(tile_size_n, N - n * tile_size_n)

                    # weight
                    stats.reads[f"s_wgt{buf}"] = stats.reads.get(f"s_wgt{buf}", 0) + cur_K * cur_N * weight_bits

                    # output
                    stats.reads[f"s_output{buf}"] = stats.reads.get(f"s_output{buf}", 0) + cur_M * cur_N * out_bits
                    stats.writes[f"s_output{buf}"] = stats.writes.get(f"s_output{buf}", 0) + cur_M * cur_N * out_bits

                    gemm_cycles = prefill_systolic_cycles(cur_M, cur_N, cur_K, H, W) / num_arrays

                    if hasattr(stats, "cycle_breakdown"):
                        stats.cycle_breakdown["systolic array"] += gemm_cycles

                    stats.compute_cycles += gemm_cycles

    else:
        for k_tile in range(tile_num_K):
            cur_K = min(tile_size_k, K - k_tile * tile_size_k)

            for n in range(tile_num_N):
                cur_N = min(tile_size_n, N - n * tile_size_n)

                # weight
                stats.reads[f"s_wgt{buf}"] = stats.reads.get(f"s_wgt{buf}", 0) + cur_K * cur_N * weight_bits

                for m in range(tile_num_M):
                    cur_M = min(tile_size_m, M - m * tile_size_m)

                    # A
                    stats.reads[f"s_act{buf}"] = stats.reads.get(f"s_act{buf}", 0) + cur_M * cur_K * act_bits

                    # output
                    stats.reads[f"s_output{buf}"] = stats.reads.get(f"s_output{buf}", 0) + cur_M * cur_N * out_bits
                    stats.writes[f"s_output{buf}"] = stats.writes.get(f"s_output{buf}", 0) + cur_M * cur_N * out_bits

                    gemm_cycles = input_stationary_systolic_cycles(cur_M, cur_N, cur_K, H, W) / num_arrays

                    if hasattr(stats, "cycle_breakdown"):
                        stats.cycle_breakdown["systolic array"] += gemm_cycles

                    stats.compute_cycles += gemm_cycles

    # ============================================================
    # memory traffic (buffer <-> dram)
    # ============================================================

    for m in range(buffer_tile_num_M):
        cur_M = min(buffer_size_m, M - m * buffer_size_m)

        for n in range(buffer_tile_num_N):
            cur_N = min(buffer_size_n, N - n * buffer_size_n)

            for k_tile in range(buffer_tile_num_K):
                cur_K = min(buffer_size_k, K - k_tile * buffer_size_k)

                stats.reads["dram"] += cur_K * cur_N * weight_bits
                stats.writes[f"s_wgt{buf}"] = stats.writes.get(f"s_wgt{buf}", 0) + cur_K * cur_N * weight_bits

                stats.reads["dram"] += cur_K * cur_M * act_bits
                stats.writes[f"s_act{buf}"] = stats.writes.get(f"s_act{buf}", 0) + cur_K * cur_M * act_bits

                if k_tile > 0:
                    stats.reads["dram"] += cur_M * cur_N * out_bits
                    stats.writes[f"s_output{buf}"] = stats.writes.get(f"s_output{buf}", 0) + cur_M * cur_N * out_bits

    stats.reads[f"s_output{buf}"] = stats.reads.get(f"s_output{buf}", 0) + M * N * out_bits
    stats.writes["dram"] += M * N * out_bits

    # ============================================================
    # memory stall
    # ============================================================

    first_K = min(tile_size_k, buffer_size_k)
    first_M = min(tile_size_m, buffer_size_m)

    init_mem = first_K * first_M * act_bits
    init_latency = ceil_a_by_b(init_mem, mem_width)

    stats.mem_stall_cycles += init_latency

    total_mem = stats.reads["dram"] + stats.writes["dram"]
    middle_mem = total_mem - init_mem
    middle_latency = ceil_a_by_b(middle_mem, mem_width)

    stats.mem_stall_cycles += max(0, middle_latency - stats.compute_cycles)

    stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles

    return stats


import json
import os
import re
from typing import Optional, Dict, Any

def _get_op_signature(op):
    M = getattr(op, "M", None)
    K = getattr(op, "K", None)
    N = getattr(op, "N", None)

    method = op.B_info["quantization_config"]["method"]

    return f"{M}x{K}x{N}-{method}", M, K, N, method


def _resolve_flow_and_runtime_args(
    op,
    args: Optional[Dict[str, Any]] = None,
    hw_config: Optional[Dict[str, Any]] = None,
):
    args = args or {}

    # ------------------------
    # manual override -- takes priority over the default flow
    # ------------------------
    force_flow = args.get("force_dataflow", None)
    if "sv" in op.name.lower():
        force_flow = "LUF"

    if op.M < 16 and "sv" not in op.name.lower(): # 040926
        default_flow = "MUF"
    else:
        default_flow = "LUF"

    flow_type = force_flow if force_flow is not None else default_flow
    assert flow_type in ["MUF", "LUF"], f"Unsupported flow type: {flow_type}, force_flow={force_flow}{type(force_flow)}"

    if flow_type == "MUF":
        m = min(op.M, 4)
        runtime_args = {
            "tile_size_m": m,
            "tile_size_codebook": 1,
            "buffer_size_m": m,
            "buffer_size_k": min(ceil_a_by_b(16384, m), 1024),
            "buffer_size_n": min(ceil_a_by_b(16384, m), 1024),
        }
        runtime_args["buf_assignment"] = "_lt"

    elif flow_type == "LUF":
        num_codebooks = op.B_info["quantization_config"]["num_codebooks"] if op.B_info["quantization_config"]["method"] == "AQLM" else 1
        if op.M > LUF_knm_2_mkn_M:
            runtime_args = {
                "tile_size_m": min(op.M, 512),
                "tile_size_codebook": num_codebooks,
                "buffer_size_m": min(op.M, 512),
                "buffer_size_k": min(op.K, 64),
                "buffer_size_n": min(op.N, 64),
            }
            runtime_args["buf_assignment"] = "_gt"
        else:
            m = op.M
            kn = min(ceil_a_by_b(16384, m), 1024)
            runtime_args = {
                "tile_size_m": m,
                "tile_size_codebook": num_codebooks,
                "buffer_size_m": m,
                "buffer_size_k": kn,
                "buffer_size_n": kn,
            }
            runtime_args["buf_assignment"] = "_lt"

    op_label, M, K, N, method = _get_op_signature(op)

    if flow_type == "MUF":
        if method == "CQ":
            assert "sv" not in op.name.lower(), "CQ direction of v is not along K."

    if flow_type is None:
        if "sv" in op.name.lower():
            flow_type = "LUF"

    return flow_type, runtime_args


def run_vortex_gemm(
    op: GEMM | GEMV,
    stats: Optional[Stats] = None,
    args: Optional[Dict[str, Any]] = None,
    hw_config: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
):
    if 0 in [op.M, op.K, op.N]:
        if stats is None:
            stats = Stats()
        else:
            stats.compute_cycles = 0
            stats.mem_stall_cycles = 0
            stats.reads = {}
            stats.writes = {}
        return stats

    if op.B_info["dtype"] != "quant":
        assert tensor_info_bits_v2(op.A_info) == 16
        assert tensor_info_bits_v2(op.B_info) == 16

        if op.M > LUF_knm_2_mkn_M:
            m = min(op.M, 512)
            runtime_args = {
                "tile_size_m": m,
                "buffer_size_m": m,
                "buffer_size_k": (int)(min(op.K, 64)),
                "buffer_size_n": (int)(min(op.N, 64)),
                "flow": "mkn",
            }
            runtime_args["buf_assignment"] = "_gt"
        else:
            m = op.M
            runtime_args = {
                "tile_size_m": m,
                "buffer_size_m": m,
                "buffer_size_k": (int)(min(op.K, 16384/m, 256)),
                "buffer_size_n": (int)(min(op.N, 16384/m, 512)),
                "flow": "knm",
            }
            runtime_args["buf_assignment"] = "_lt"

        if verbose:
            print(f"Running non-quantized GEMM with runtime args: {runtime_args}")

        return run_gemm_systolic_array(op, stats, runtime_args, hw_config)

    # =========================
    # quant path
    # =========================
    assert op.B_info["dtype"] == "quant"
    B_qcfg = op.B_info["quantization_config"]
    assert B_qcfg["method"] in ["AQLM", "CQ"]

    flow_type, runtime_args = _resolve_flow_and_runtime_args(op, args, hw_config)

    # Defensive: _resolve_flow_and_runtime_args should already have set this.
    if "buf_assignment" not in runtime_args:
        runtime_args["buf_assignment"] = "_gt" if op.M > LUF_knm_2_mkn_M else "_lt"

    assert flow_type is not None

    if flow_type == "LUF":
        stats = vortex_gemm_LUF(op, stats, runtime_args, hw_config)
    elif flow_type == "MUF":
        if B_qcfg["method"] == "CQ":
            assert "sv" not in op.name.lower()
        stats = vortex_gemm_MUF(op, stats, runtime_args, hw_config)

    if verbose:
        print(f"Ran with flow {flow_type} and runtime args: {runtime_args}")

    return stats

