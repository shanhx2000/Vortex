from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

from operations import GEMM, GEMV
from stats import Stats

from .ant import run_ant
from .figna import run_figna
from .figlut import run_figlut
from .systolic_array import run_systolic_array


BASELINE_RUNNERS: Dict[str, Callable[..., Stats]] = {
    "ant": run_ant,
    "figna": run_figna,
    "figlut": run_figlut,
    "systolic_array": run_systolic_array,
}


def run_baseline_model(
    ops: Iterable[GEMM | GEMV],
    method: str,
    *,
    args: Optional[dict] = None,
    hw_config: Optional[dict] = None,
) -> Stats:
    if method not in BASELINE_RUNNERS:
        raise ValueError(f"Unknown baseline runner: {method}")

    total_stats = Stats(name=f"{method}.model")
    for op in ops:
        op_stats = BASELINE_RUNNERS[method](op, args=args, hw_config=hw_config)
        total_stats.add_(op_stats)
    total_stats.finalize()
    return total_stats


__all__ = [
    "BASELINE_RUNNERS",
    "run_baseline_model",
    "run_ant",
    "run_figna",
    "run_figlut",
    "run_systolic_array",
]
