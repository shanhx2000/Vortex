"""Vortex hardware simulator: HWSimBase with the Evaluator energy model.

Overrides `_apply_energy_breakdown` so all energy comes from `evaluator.py`
rather than the baselines' model, and zeroes `static_energy` because the
Evaluator already folds leakage into its core term.

Consequence worth knowing: `sfu.power.static` in `hw_configs/vortex.json` is
dead config for Vortex -- never read. The softmax number that is actually used
lives in `power_energy_config.json` under `softmax_unit`.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from operations import GEMM, GEMV, Idle, OP, Operations, Quantize, Softmax
from stats import Stats
from hw_base import HWSimBase
from vortex import run_vortex_gemm

class HWVortex(HWSimBase):
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        from evaluator import Evaluator
        self.evaluator = Evaluator(buffer_specs=self.config.get("buffers", None))

    def run_op(
        self,
        op: OP,
        *,
        method: str,
        args: Optional[Dict[str, Any]] = None,
        hw_config: Optional[Dict[str, Any]] = None,
    ) -> Stats:
        hw_cfg = hw_config or self.config
        resolved_method = self._resolve_method(method)

        from baselines.common import run_quantize_placeholder, run_softmax

        if isinstance(op, Softmax):
            return run_softmax(op, None, hw_cfg, name=resolved_method)
        if isinstance(op, Quantize):
            return run_quantize_placeholder(op, None, args, hw_cfg, name=resolved_method)
        assert isinstance(op, GEMM)
        stats = run_vortex_gemm(op, args=args, hw_config=hw_cfg)
        self._apply_energy_breakdown(stats, hw_cfg)
        return stats

    def _apply_energy_breakdown(self, stats: Stats, config: Dict[str, Any]) -> Stats:
        freq_mhz = float(config.get("general", {}).get("frequency", 0.0))
        total_time_s = 0.0 if freq_mhz <= 0 else stats.total_cycles / (freq_mhz * 1e6)

        energy = self.evaluator.get_energy(
            hw_config=self.config,
            stats=stats,
            frequency=freq_mhz,
        )

        stats.static_energy = 0
        stats.dram_energy = energy["energy_breakdown"]["dram"]["dram_energy (J)"]
        stats.buffer_energy = energy["energy_breakdown"]["buffer"]["buffer_energy (J)"]
        stats.core_energy = energy["energy_breakdown"]["core"]["core_energy (J)"]

        stats.core_power = 0.0 if total_time_s <= 0 else stats.core_energy / total_time_s
        stats.sram_power = 0.0 if total_time_s <= 0 else stats.buffer_energy / total_time_s
        stats.dram_power = 0.0 if total_time_s <= 0 else stats.dram_energy / total_time_s
        return stats