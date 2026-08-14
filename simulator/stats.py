"""The accumulator every hardware model writes into.

A Stats object carries cycle counts (total / compute / mem-stall / preprocess),
per-memory-space read and write bit counts, a cycle breakdown by hardware unit,
sparsity metadata, and energy fields filled in later by an energy model.

Stats are combined with `+` / `add_` as the op list is walked, and `scale_` is
used to expand one simulated attention block to the full head count. Memory
spaces carry an `_lt` / `_gt` suffix naming the buffer assignment in use.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional


class Stats:
    def __init__(self, name: str = "tetris"):
        self.name = name
        self.total_cycles = 0
        self.mem_stall_cycles = 0
        self.compute_cycles = 0
        self.num_ops = 0
        self.preprocess_stall_cycles = 0
        self.mem_namespace = [
            "dram",
            "buffer",
            "s_act",
            "s_wgt",
            "s_codebook",
            "s_cap",
            "s_parameter",
            "s_scale",
            "s_output",
            "s_double_cap",
        ]
        self.reads = {space: 0 for space in self.mem_namespace}
        self.writes = {space: 0 for space in self.mem_namespace}
        self.original_sparsity = 0
        self.processed_sparsity = 0
        self.ops_sparsity = 0
        self.rank_two_sparsity = 0
        self.avg_rank_one_prefix = 0
        self.avg_rank_two_prefix = 0
        self.cycle_breakdown = None
        self.static_energy = 0
        self.dram_energy = 0
        self.buffer_energy = 0
        self.core_energy = 0
        self.core_power = 0
        self.sram_power = 0
        self.dram_power = 0

    def __add__(self, other: "Stats") -> "Stats":
        if not isinstance(other, Stats):
            raise Exception("unsupported type")
        added_stats = Stats(name=f"{self.name}+{other.name}")
        added_stats.total_cycles = self.total_cycles + other.total_cycles
        added_stats.mem_stall_cycles = self.mem_stall_cycles + other.mem_stall_cycles
        added_stats.compute_cycles = self.compute_cycles + other.compute_cycles
        added_stats.preprocess_stall_cycles = self.preprocess_stall_cycles + other.preprocess_stall_cycles
        added_stats.num_ops = self.num_ops + other.num_ops


        # Merge mem_namespace. It must also pick up keys that only appear in
        # reads/writes: a space recorded by update_mem is not necessarily
        # declared in mem_namespace, and dropping it would lose its accesses.
        all_spaces = set(self.mem_namespace) | set(other.mem_namespace)
        all_spaces |= set(self.reads.keys())
        all_spaces |= set(self.writes.keys())
        all_spaces |= set(other.reads.keys())
        all_spaces |= set(other.writes.keys())

        # Sorted so the namespace order is reproducible across runs.
        merged_namespace = sorted(all_spaces)
        added_stats.mem_namespace = merged_namespace
        # ===== reads =====
        added_stats.reads = {}
        for space in merged_namespace:
            added_stats.reads[space] = (
                self.reads.get(space, 0) + other.reads.get(space, 0)
            )

        # ===== writes =====
        added_stats.writes = {}
        for space in merged_namespace:
            added_stats.writes[space] = (
                self.writes.get(space, 0) + other.writes.get(space, 0)
            )
        added_stats.original_sparsity = self.original_sparsity + other.original_sparsity
        added_stats.processed_sparsity = self.processed_sparsity + other.processed_sparsity
        added_stats.ops_sparsity = self.ops_sparsity + other.ops_sparsity
        added_stats.rank_two_sparsity = self.rank_two_sparsity + other.rank_two_sparsity
        added_stats.avg_rank_one_prefix = self.avg_rank_one_prefix + other.avg_rank_one_prefix
        added_stats.avg_rank_two_prefix = self.avg_rank_two_prefix + other.avg_rank_two_prefix
        if isinstance(self.cycle_breakdown, dict) or isinstance(other.cycle_breakdown, dict):
            left = self.cycle_breakdown if isinstance(self.cycle_breakdown, dict) else {}
            right = other.cycle_breakdown if isinstance(other.cycle_breakdown, dict) else {}
            keys = set(left) | set(right)
            added_stats.cycle_breakdown = {}
            for key in keys:
                left_value = left.get(key, 0)
                right_value = right.get(key, 0)
                if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
                    added_stats.cycle_breakdown[key] = left_value + right_value
                elif key in right:
                    added_stats.cycle_breakdown[key] = right_value
                else:
                    added_stats.cycle_breakdown[key] = left_value
        added_stats.static_energy = self.static_energy + other.static_energy
        added_stats.dram_energy = self.dram_energy + other.dram_energy
        added_stats.buffer_energy = self.buffer_energy + other.buffer_energy
        added_stats.core_energy = self.core_energy + other.core_energy
        added_stats.core_power = self.core_power + other.core_power
        added_stats.sram_power = self.sram_power + other.sram_power
        added_stats.dram_power = self.dram_power + other.dram_power
        return added_stats

    def add_(self, other: "Stats", total_cycles_mode: str = "sum") -> "Stats":
        if total_cycles_mode not in {"sum", "max"}:
            raise ValueError("total_cycles_mode must be either 'sum' or 'max'")
        merged = self + other
        merged.total_cycles = (
            self.total_cycles + other.total_cycles
            if total_cycles_mode == "sum"
            else max(self.total_cycles, other.total_cycles)
        )
        self.__dict__.update(merged.__dict__)
        return self

    def clone(self) -> "Stats":
        copied = Stats(name=self.name)
        copied.__dict__.update({k: v.copy() if isinstance(v, dict) else v for k, v in self.__dict__.items()})
        return copied

    def scale_(self, factor: int | float) -> "Stats":
        factor = float(factor)
        self.total_cycles = int(self.total_cycles * factor)
        self.mem_stall_cycles = int(self.mem_stall_cycles * factor)
        self.compute_cycles = int(self.compute_cycles * factor)
        self.preprocess_stall_cycles = int(self.preprocess_stall_cycles * factor)
        self.num_ops = int(self.num_ops * factor)
        self.reads = {space: int(value * factor) for space, value in self.reads.items()}
        self.writes = {space: int(value * factor) for space, value in self.writes.items()}
        self.original_sparsity *= factor
        self.processed_sparsity *= factor
        self.ops_sparsity *= factor
        self.rank_two_sparsity *= factor
        self.avg_rank_one_prefix *= factor
        self.avg_rank_two_prefix *= factor
        self.static_energy *= factor
        self.dram_energy *= factor
        self.buffer_energy *= factor
        self.core_energy *= factor
        if isinstance(self.cycle_breakdown, dict):
            scaled = {}
            for key, value in self.cycle_breakdown.items():
                if isinstance(value, (int, float)):
                    scaled[key] = value * factor
                else:
                    scaled[key] = value
            self.cycle_breakdown = scaled
        return self

    def update_cycles(
        self,
        *,
        total: int = 0,
        compute: int = 0,
        mem_stall: int = 0,
        preprocess_stall: int = 0,
    ) -> None:
        self.total_cycles += int(total)
        self.compute_cycles += int(compute)
        self.mem_stall_cycles += int(mem_stall)
        self.preprocess_stall_cycles += int(preprocess_stall)

    def update_mem(self, space: str, reads: int = 0, writes: int = 0) -> None:
        if space not in self.mem_namespace:
            raise KeyError(f"Unknown memory space: {space}")
        self.reads[space] += int(reads)
        self.writes[space] += int(writes)

    def update_energy(
        self,
        *,
        static: float = 0.0,
        dram: float = 0.0,
        buffer: float = 0.0,
        core: float = 0.0,
    ) -> None:
        self.static_energy += float(static)
        self.dram_energy += float(dram)
        self.buffer_energy += float(buffer)
        self.core_energy += float(core)

    def update_cycle_breakdown(self, breakdown: Optional[Dict[str, float]], factor: float = 1.0) -> None:
        if not isinstance(breakdown, dict):
            return
        if not isinstance(self.cycle_breakdown, dict):
            self.cycle_breakdown = {}
        for key, value in breakdown.items():
            if isinstance(value, (int, float)):
                self.cycle_breakdown[key] = self.cycle_breakdown.get(key, 0.0) + float(value) * factor
            else:
                self.cycle_breakdown[key] = value

    def finalize(self) -> None:
        self.total_cycles = max(
            self.total_cycles,
            self.compute_cycles + self.mem_stall_cycles + self.preprocess_stall_cycles,
        )

    @property
    def total_energy(self) -> float:
        return self.static_energy + self.dram_energy + self.buffer_energy + self.core_energy

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "total_cycles": self.total_cycles,
            "mem_stall_cycles": self.mem_stall_cycles,
            "compute_cycles": self.compute_cycles,
            "preprocess_stall_cycles": self.preprocess_stall_cycles,
            "num_ops": self.num_ops,
            "reads": dict(self.reads),
            "writes": dict(self.writes),
            "original_sparsity": self.original_sparsity,
            "processed_sparsity": self.processed_sparsity,
            "ops_sparsity": self.ops_sparsity,
            "rank_two_sparsity": self.rank_two_sparsity,
            "avg_rank_one_prefix": self.avg_rank_one_prefix,
            "avg_rank_two_prefix": self.avg_rank_two_prefix,
            "cycle_breakdown": dict(self.cycle_breakdown) if isinstance(self.cycle_breakdown, dict) else self.cycle_breakdown,
            "static_energy": self.static_energy,
            "dram_energy": self.dram_energy,
            "buffer_energy": self.buffer_energy,
            "core_energy": self.core_energy,
            "total_energy": self.total_energy,
            "core_power": self.core_power,
            "sram_power": self.sram_power,
            "dram_power": self.dram_power,
        }

    def __repr__(self) -> str:
        return f"Stats(name={self.name!r}, total_cycles={self.total_cycles}, total_energy={self.total_energy:.4f})"

def dict_to_stats(d):
    s = Stats(name=d.get("name", "restored"))

    # ===== cycles =====
    s.total_cycles = d.get("total_cycles", 0)
    s.mem_stall_cycles = d.get("mem_stall_cycles", 0)
    s.compute_cycles = d.get("compute_cycles", 0)
    s.preprocess_stall_cycles = d.get("preprocess_stall_cycles", 0)
    s.num_ops = d.get("num_ops", 0)

    # ===== memory =====
    s.reads = dict(d.get("reads", {}))
    s.writes = dict(d.get("writes", {}))

    # Rebuild mem_namespace from the access keys; it is not serialised.
    s.mem_namespace = sorted(
        set(s.reads.keys()) | set(s.writes.keys())
    )

    # ===== sparsity =====
    s.original_sparsity = d.get("original_sparsity", 0)
    s.processed_sparsity = d.get("processed_sparsity", 0)
    s.ops_sparsity = d.get("ops_sparsity", 0)
    s.rank_two_sparsity = d.get("rank_two_sparsity", 0)
    s.avg_rank_one_prefix = d.get("avg_rank_one_prefix", 0)
    s.avg_rank_two_prefix = d.get("avg_rank_two_prefix", 0)

    # ===== breakdown =====
    s.cycle_breakdown = d.get("cycle_breakdown", None)

    # ===== energy (absent unless the producer recorded it) =====
    s.static_energy = d.get("static_energy", 0)
    s.dram_energy = d.get("dram_energy", 0)
    s.buffer_energy = d.get("buffer_energy", 0)
    s.core_energy = d.get("core_energy", 0)

    # ===== power =====
    s.core_power = d.get("core_power", 0)
    s.sram_power = d.get("sram_power", 0)
    s.dram_power = d.get("dram_power", 0)

    return s

def merge_stats(stats_list: Iterable[Stats], name: str = "merged") -> Stats:
    merged = Stats(name=name)
    for stats in stats_list:
        merged.add_(stats)
    merged.finalize()
    return merged

def init_cycle_breakdown(stats: Stats):
    stats.cycle_breakdown = {
        "systolic array": 0,
        "lut array": 0,
        "lut_array.rd": 0,
        "lut_array.wr": 0,
        "adder tree": 0,
    }