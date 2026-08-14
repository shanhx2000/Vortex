"""Base hardware simulator: op dispatch, progress accounting, baseline energy.

`HWSimBase.run_op` routes each operation to a baseline kernel model, and
`run_attention_ops` groups the ops of one attention block so they can be scaled
together by head count.

The energy model here (`_apply_energy_breakdown`, `get_energy`, `get_area`) is
the *baselines'* model and is independent of `evaluator.py`. HWVortex overrides
`_apply_energy_breakdown` and sources everything from Evaluator instead, so the
two paths share no numbers -- including the softmax unit, which the baselines
read from their own hw config. They also skip softmax *cycles* in attention
altogether, so Vortex's speedup over them is a conservative estimate.

Note that for baselines, softmax ops inside attention are skipped outright,
so baselines pay neither cycles nor energy for them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from operations import GEMM, GEMV, Idle, OP, Operations, Quantize, Softmax
from stats import Stats
from tqdm import tqdm


class HWSimBase:
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.active_cycles = 0
        self._power_energy_config = None

    def idle_cycle(self, cycles: int = 1) -> Stats:
        """Consume idle cycles without issuing useful work."""
        stats = Stats(name=f"{self.name}.idle")
        stats.update_cycles(total=cycles)
        self.active_cycles += cycles
        return stats

    def handle_operation(self, op: Operations, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Default operation handler.

        Child classes should override this method and return a dictionary with at least
        the key "stats".
        """
        stats = Stats(name=f"{self.name}.{op.name}")
        stats.num_ops = 1
        if isinstance(op, Idle) or getattr(op, "op_type", "") == "idle":
            stats = self.idle_cycle(getattr(op, "cycles", 1))
        return {"stats": stats}

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

        if resolved_method == "vortex":
            if type(self).handle_operation is HWSimBase.handle_operation:
                raise ValueError("vortex method requires a concrete hardware simulator subclass")
            return self.handle_operation(op, args=args)["stats"]

        from baselines import BASELINE_RUNNERS
        from baselines.common import run_quantize_placeholder, run_softmax

        if isinstance(op, Softmax):
            return run_softmax(op, None, hw_cfg, name=resolved_method)
        if isinstance(op, Quantize):
            return run_quantize_placeholder(op, None, args, hw_cfg, name=resolved_method)
        if resolved_method not in BASELINE_RUNNERS:
            raise ValueError(f"Unsupported simulation method: {resolved_method}")
        assert isinstance(op, GEMM)
        if op.M <= 0 or op.K <= 0 or op.N <= 0:
            return Stats(name=f"{self.name}.{op.name}")
        stats = BASELINE_RUNNERS[resolved_method](op, args=args, hw_config=hw_cfg)
        self._apply_energy_breakdown(stats, hw_cfg)
        return stats

    def run_attention_ops(
        self,
        ops: Iterable[OP],
        *,
        method: str,
        args: Optional[Dict[str, Any]] = None,
        hw_config: Optional[Dict[str, Any]] = None,
    ) -> Stats:
        hw_cfg = hw_config or self.config
        resolved_method = self._resolve_method(method)
        attn_ops = list(ops)
        if not attn_ops:
            return Stats(name=f"{self.name}.attention")

        total_stats = Stats(name=f"{self.name}.{attn_ops[0].parent_name}")
        total_stats.num_ops = 0

        for op in attn_ops:
            phase = getattr(op, "metadata", {}).get("attn_phase")
            if resolved_method != "vortex" and phase == "softmax":
                continue
            op_stats = self.run_op(op, method=resolved_method, args=args, hw_config=hw_cfg)
            total_stats.add_(op_stats)

        scale = self._attention_group_scale(attn_ops[0])
        if scale > 1:
            total_stats.scale_(scale)
        total_stats.finalize()
        if resolved_method != "vortex":
            self._apply_energy_breakdown(total_stats, hw_cfg)
        return total_stats

    def run_model(
        self,
        ops_list: Iterable[OP],
        *,
        method: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
        hw_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        hw_cfg = hw_config or self.config
        resolved_method = self._resolve_method(method)
        ops = list(ops_list)
        total_stats = Stats(name=f"{self.name}.{resolved_method}.model")
        per_op = []
        layer_count = 1


        idx = 0
        pbar = tqdm(total=len(ops), desc="Running ops")

        while idx < len(ops):
            op = ops[idx]
            metadata = getattr(op, "metadata", {})
            layer_count = max(layer_count, int(metadata.get("layer_count", 1)))

            attn_phase = metadata.get("attn_phase")

            if attn_phase is not None:
                parent_name = op.parent_name
                grouped_ops = []
                start_idx = idx

                while idx < len(ops):
                    current = ops[idx]
                    current_meta = getattr(current, "metadata", {})

                    if current.parent_name != parent_name or current_meta.get("attn_phase") is None:
                        break

                    grouped_ops.append(current)
                    layer_count = max(layer_count, int(current_meta.get("layer_count", 1)))
                    idx += 1

                # run grouped
                attn_stats = self.run_attention_ops(
                    grouped_ops,
                    method=resolved_method,
                    args=args,
                    hw_config=hw_cfg
                )

                total_stats.add_(attn_stats)
                per_op.append({"name": parent_name, "stats": attn_stats})

                # An attention group consumed several ops at once.
                pbar.update(idx - start_idx)

                continue

            op_stats = self.run_op(op, method=resolved_method, args=args, hw_config=hw_cfg)
            total_stats.add_(op_stats)
            per_op.append({"name": op.name, "stats": op_stats})

            idx += 1
            pbar.update(1)

        pbar.close()

        if layer_count > 1:
            total_stats.scale_(layer_count)
        total_stats.finalize()
        self._apply_energy_breakdown(total_stats, hw_cfg)
        energy = self.get_energy(total_stats, hw_cfg)
        area = self.get_area(hw_cfg)
        return {
            "stats": total_stats,
            "per_op": per_op,
            "energy": energy,
            "area": area,
            "method": resolved_method,
            "layer_count": layer_count,
        }

    def get_energy(self, stats: Stats, config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        cfg = config or self.config
        estimated = stats.clone()
        self._apply_energy_breakdown(estimated, cfg)
        freq_mhz = float(cfg.get("general", {}).get("frequency", 0.0))
        total_time_s = 0.0 if freq_mhz <= 0 else estimated.total_cycles / (freq_mhz * 1e6)
        total_power = 0.0 if total_time_s <= 0 else estimated.total_energy / total_time_s
        dram_power = 0.0 if total_time_s <= 0 else estimated.dram_energy / total_time_s
        sram_power = 0.0 if total_time_s <= 0 else estimated.buffer_energy / total_time_s
        core_power = 0.0 if total_time_s <= 0 else estimated.core_energy / total_time_s
        estimated.dram_power = dram_power
        estimated.sram_power = sram_power
        estimated.core_power = core_power
        return {
            "static_energy": estimated.static_energy,
            "dram_energy": estimated.dram_energy,
            "buffer_energy": estimated.buffer_energy,
            "core_energy": estimated.core_energy,
            "total_energy": estimated.total_energy,
            "core_power": core_power,
            "sram_power": sram_power,
            "dram_power": dram_power,
            "total_power": total_power,
        }

    def get_area(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg = config or self.config
        pec = self._load_power_energy_config()
        modules = cfg.get("modules", {})
        breakdown: Dict[str, float] = {}

        core_cfg = modules.get("core")
        if core_cfg and "area" in core_cfg:
            breakdown["core"] = float(core_cfg.get("area", 0.0))

        sa_cfg = modules.get("systolic array")
        if sa_cfg:
            ratio = (int(sa_cfg.get("num", 1)) * int(sa_cfg.get("rows", 1)) * int(sa_cfg.get("cols", 1))) / (16 * 16)
            breakdown["systolic array"] = float(pec["systolic_array_16x16_fp16"]["area"]) * ratio

        lut_cfg = modules.get("lut array")
        if lut_cfg:
            breakdown["lut array"] = float(pec["sram_256x16bits_1rw"]["area"]) * int(lut_cfg.get("num", 1))

        adder_cfg = modules.get("adder tree")
        if adder_cfg:
            breakdown["adder tree"] = float(pec["fadd_fp16"]["area"]) * int(adder_cfg.get("num", 1)) * int(adder_cfg.get("width", 1))

        vpu_cfg = modules.get("vpu", {})
        vpu_area = 0.0
        if "vfadd" in vpu_cfg:
            cfg_item = vpu_cfg["vfadd"]
            vpu_area += float(pec["vfadd_WIDTH32_DATAWIDTH16"]["area"]) * int(cfg_item.get("num", 1)) * (int(cfg_item.get("width", 32)) / 32.0)
        if "vfmul" in vpu_cfg:
            cfg_item = vpu_cfg["vfmul"]
            vpu_area += float(pec["vfmul_WIDTH32_DATAWIDTH16"]["area"]) * int(cfg_item.get("num", 1)) * (int(cfg_item.get("width", 32)) / 32.0)
        if vpu_area > 0:
            breakdown["vpu"] = vpu_area

        total_area_um2 = sum(breakdown.values())
        return {
            "total_area (um^2)": total_area_um2,
            "total_area (mm^2)": total_area_um2 * 1e-6,
            "breakdown": breakdown,
        }

    def run_ops(self, ops: Iterable[Operations], args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        total_stats = Stats(name=f"{self.name}.network")
        total_stats.num_ops = 0
        per_op = []
        for op in ops:
            ret = self.handle_operation(op, args=args)
            op_stats = ret["stats"]
            total_stats.add_(op_stats)
            per_op.append({"name": op.name, "stats": op_stats})
        self._apply_energy_breakdown(total_stats, self.config)
        return {"stats": total_stats, "per_op": per_op}

    def run_fc_layer(
        self,
        op: OP,
        *,
        method: str,
        args: Optional[Dict[str, Any]] = None,
        hw_config: Optional[Dict[str, Any]] = None,
    ) -> Stats:
        return self.run_op(op, method=method, args=args, hw_config=hw_config)

    def run_attention_layer(
        self,
        ops: Iterable[OP],
        *,
        method: str,
        args: Optional[Dict[str, Any]] = None,
        hw_config: Optional[Dict[str, Any]] = None,
    ) -> Stats:
        return self.run_attention_ops(ops, method=method, args=args, hw_config=hw_config)

    def evaluate_networks(self, nw: Iterable[Operations], args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.run_ops(nw, args=args)

    def _resolve_method(self, method: Optional[str]) -> str:
        if method:
            return method
        general = self.config.get("general", {})
        configured = general.get("method") or general.get("baseline")
        if configured:
            return str(configured)
        return self.name

    def _attention_group_scale(self, op: OP) -> int:
        metadata = getattr(op, "metadata", {})
        if "num_kv_heads" in metadata:
            return max(1, int(metadata["num_kv_heads"]))
        num_heads = int(metadata.get("num_heads", 1))
        group_size = max(1, int(metadata.get("group_size", 1)))
        return max(1, num_heads // group_size)

    def _load_power_energy_config(self) -> Dict[str, Any]:
        if self._power_energy_config is None:
            config_path = Path(__file__).resolve().parent / "hw_configs" / "power_energy_config.json"
            with config_path.open("r", encoding="utf-8") as f:
                self._power_energy_config = json.load(f)
        return self._power_energy_config

    def _apply_energy_breakdown(self, stats: Stats, config: Dict[str, Any]) -> Stats:
        freq_mhz = float(config.get("general", {}).get("frequency", 0.0))
        total_time_s = 0.0 if freq_mhz <= 0 else stats.total_cycles / (freq_mhz * 1e6)
        active_cycles = max(0, stats.total_cycles - stats.mem_stall_cycles)
        active_time_s = 0.0 if freq_mhz <= 0 else active_cycles / (freq_mhz * 1e6)

        dram_cfg = config.get("general", {}).get("dram", {})
        dram_power_cfg = dram_cfg.get("power", {})
        dram_read_e = self._energy_per_bit(dram_power_cfg, "read")
        dram_write_e = self._energy_per_bit(dram_power_cfg, "write")
        dram_static_power = float(dram_power_cfg.get("static", 0.0))
        dram_energy = stats.reads.get("dram", 0) * dram_read_e + stats.writes.get("dram", 0) * dram_write_e + dram_static_power * total_time_s

        buffer_energy = 0.0
        buffer_static_power = 0.0
        buffers = config.get("buffers", {})
        buffer_alias = {
            "buffer": [],
            "s_act": ["s_act"],
            "s_wgt": ["s_wgt", "s_weight"],
            "s_codebook": ["s_codebook", "s_parameter"],
            "s_cap": ["s_cap", "s_output"],
            "s_parameter": ["s_parameter"],
            "s_scale": ["s_scale", "s_parameter"],
            "s_output": ["s_output"],
            "s_double_cap": ["s_double_cap", "s_output"],
        }
        for space, cfg_names in buffer_alias.items():
            power_cfg = self._resolve_buffer_power_cfg(buffers, cfg_names)
            buffer_energy += stats.reads.get(space, 0) * self._energy_per_bit(power_cfg, "read")
            buffer_energy += stats.writes.get(space, 0) * self._energy_per_bit(power_cfg, "write")
        for buffer_cfg in buffers.values():
            buffer_static_power += float(buffer_cfg.get("power", {}).get("static", 0.0))
        buffer_energy += buffer_static_power * total_time_s

        core_power_cfg = config.get("modules", {}).get("core", {}).get("power", {})
        core_static_power = float(core_power_cfg.get("static", 0.0))
        core_dynamic_power = float(core_power_cfg.get("dynamic", 0.0))
        core_energy = core_static_power * total_time_s + core_dynamic_power * active_time_s

        other_static_power = 0.0
        for module_name, module_cfg in config.get("modules", {}).items():
            if module_name == "core":
                continue
            power_cfg = module_cfg.get("power", {})
            if module_name == "sfu":
                for sub_cfg in module_cfg.values():
                    if isinstance(sub_cfg, dict):
                        power_cfg = sub_cfg.get("power", power_cfg)
                        other_static_power += float(power_cfg.get("static", 0.0))
                continue
            other_static_power += float(power_cfg.get("static", 0.0))
        static_energy = other_static_power * total_time_s

        stats.static_energy = static_energy
        stats.dram_energy = dram_energy
        stats.buffer_energy = buffer_energy
        stats.core_energy = core_energy
        stats.core_power = 0.0 if total_time_s <= 0 else core_energy / total_time_s
        stats.sram_power = 0.0 if total_time_s <= 0 else buffer_energy / total_time_s
        stats.dram_power = 0.0 if total_time_s <= 0 else dram_energy / total_time_s
        return stats

    def _energy_per_bit(self, power_cfg: Dict[str, Any], direction: str) -> float:
        if direction == "read":
            return float(power_cfg.get("read_energy_per_bits", power_cfg.get("energy_per_bits", 0.0)))
        return float(power_cfg.get("write_energy_per_bits", power_cfg.get("energy_per_bits", 0.0)))

    def _resolve_buffer_power_cfg(self, buffers: Dict[str, Any], cfg_names: list[str]) -> Dict[str, Any]:
        default_cfg: Dict[str, Any] = {}
        if buffers:
            default_cfg = next(iter(buffers.values())).get("power", {})
        if not cfg_names:
            return default_cfg
        for cfg_name in cfg_names:
            if cfg_name in buffers:
                return buffers[cfg_name].get("power", {})
        return default_cfg
