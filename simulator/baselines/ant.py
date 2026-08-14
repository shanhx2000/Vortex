from __future__ import annotations

from typing import Any, Dict, Optional

from operations import GEMM, GEMV
from stats import Stats

from .common import arg_get, init_stats, run_tiled_systolic_gemm


def run_ant(
    op: GEMM | GEMV,
    stats: Optional[Stats] = None,
    args: Optional[Dict[str, Any]] = None,
    hw_config: Optional[Dict[str, Any]] = None,
) -> Stats:
    target = init_stats(stats, "ant")
    runtime = (hw_config or {}).get("runtime", {})
    return run_tiled_systolic_gemm(
        op,
        target,
        array_height=arg_get(args, "array_height", int(runtime.get("array_height", 32))),
        array_width=arg_get(args, "array_width", int(runtime.get("array_width", 32))),
        tile_size_m=arg_get(args, "tile_size_m", int(runtime.get("tile_size_m", 1024))),
        tile_size_k=arg_get(args, "tile_size_k", int(runtime.get("tile_size_k", 32))),
        tile_size_n=arg_get(args, "tile_size_n", int(runtime.get("tile_size_n", 32))),
        buffer_size_m=arg_get(args, "buffer_size_m", int(runtime.get("buffer_size_m", 1024))),
        buffer_size_k=arg_get(args, "buffer_size_k", int(runtime.get("buffer_size_k", 64))),
        buffer_size_n=arg_get(args, "buffer_size_n", int(runtime.get("buffer_size_n", 128))),
        mem_width=arg_get(args, "mem_width", int(runtime.get("mem_width", 1024))),
        pipeline_penalty=arg_get(args, "pipeline_penalty", int(runtime.get("pipeline_penalty", 2))),
        act_nbits=arg_get(args, "act_nbits", int(runtime.get("act_nbits", 8))),
        weight_nbits=arg_get(args, "weight_nbits", int(runtime.get("weight_nbits", 8))),
        out_nbits=arg_get(args, "out_nbits", int(runtime.get("out_nbits", 8))),
    )
