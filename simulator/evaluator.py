"""Area and energy model for Vortex, from CACTI/synthesis-derived unit costs.

Reads `hw_configs/power_energy_config.json` (areas in um^2, powers in mW,
energies in nJ) and converts a Stats object's cycle counts and memory access
counts into joules and mm^2.

Only some components are charged dynamic energy -- the MXU, the 32-wide adder
tree, and the three memory levels. The VPU, quant array and softmax unit are
charged leakage only, on the grounds that their dynamic contribution is small.

This is the Vortex-only model. Baselines never reach it: HWVortex is the sole
caller, and HWSimBase has its own, unrelated energy model.
"""
from pathlib import Path

from helper import load_json_with_eval
from stats import init_cycle_breakdown

# Resolved relative to this file so runs do not depend on the CWD.
_SIM_DIR = Path(__file__).resolve().parent
DEFAULT_POWER_ENERGY_CONFIG = _SIM_DIR / "hw_configs" / "power_energy_config.json"
LTC_TABLE_PATH = _SIM_DIR / "ltc_merged.csv"


class Evaluator:
    def __init__(self, buffer_specs=None, config_path=DEFAULT_POWER_ENERGY_CONFIG):
        self.config_path = config_path
        self.area_power_energy_config = self._load_config(buffer_specs)

    def _load_config(self, buffer_specs):
        import pandas as pd
        cfg = load_json_with_eval(self.config_path)
        df = pd.read_csv(LTC_TABLE_PATH)
        for _, row in df.iterrows():
            key = row["Config"]
            cfg[key] = {
                "area": row["Area(mm^2)"] * 1e6,  # mm^2 → um^2
                "static_power": row["Leakage Power (mW)"],
                "read_energy": row["Read_Energy (nJ/bit)"],
                "write_energy": row["Write_Energy (nJ/bit)"],
            }
        return cfg

    def _is_zero_buffer(self, spec):
        return spec["size"] == 0 or spec["num"] == 0

    def _group_buffers(self, buffers):
        grouped = {}

        for name in buffers:
            if not (name.endswith("_lt") or name.endswith("_gt")):
                raise AssertionError(f"[ERROR] Buffer {name} missing _lt/_gt suffix")

            base = name[:-3]  # remove _lt/_gt
            grouped.setdefault(base, {})[name[-2:]] = name

        # ensure both exist
        for base, d in grouped.items():
            if "lt" not in d or "gt" not in d:
                raise AssertionError(f"[ERROR] Buffer {base} missing lt/gt pair")

        return grouped
    # ------------------------
    # Area
    # ------------------------
    def get_core_area(self, hw_config):
        cfg = self.area_power_energy_config

        systolic = hw_config["modules"]["systolic array"]
        lut = hw_config["modules"]["lut array"]
        adder = hw_config["modules"]["adder tree"]
        vpu = hw_config["modules"]["vpu"]
        quantarray = hw_config["modules"]["quant array"]

        # ---------------- systolic ----------------
        systolic_ratio = (systolic["num"] * systolic["rows"] * systolic["cols"]) / (16 * 16)
        systolic_area = cfg["systolic_array_16x16_fp16"]["area"] * systolic_ratio

        # ---------------- lut ----------------
        lut_area = cfg["sram_256x16bits_1rw"]["area"] * lut["num"] * 2 # double buffer

        # ---------------- adder tree ----------------
        adder_ratio = adder["num"]
        assert adder["width"] == 32
        adder_area = cfg["fp16_add_tree_32"]["area"] * adder_ratio

        # ---------------- vpu ----------------
        vpu_area = sum([
            vpu["vfadd"]["num"] * cfg["vfadd_WIDTH32_DATAWIDTH16"]["area"],
            vpu["vfmul"]["num"] * cfg["vfmul_WIDTH32_DATAWIDTH16"]["area"],
            vpu["adder tree"]["num"] * cfg["fp16_add_tree_4"]["area"],
        ])

        # ---------------- quant array ----------------
        quant_area = (
            quantarray["num"]
            * quantarray["width"] / 8
            * cfg["qua_update_WIDTH8_DATAWIDTH16_IDXWIDTH8"]["area"]
        )

        softmax = hw_config["modules"]["sfu"]["softmax"]
        assert softmax["num"] == 1 and softmax["width"] == 32 and softmax["dtype"] == "fp16"
        sfu_area = cfg["softmax_unit"]["area"]

        # ---------------- total ----------------
        total_area = (
            systolic_area
            + lut_area
            + adder_area
            + vpu_area
            + quant_area
            + sfu_area
        )

        return total_area * 1e-6  # um^2 → mm^2


    def get_area_breakdown(self, hw_config):
        cfg = self.area_power_energy_config

        breakdown = {}

        # ================= core =================
        systolic = hw_config["modules"]["systolic array"]
        lut = hw_config["modules"]["lut array"]
        adder = hw_config["modules"]["adder tree"]
        vpu = hw_config["modules"]["vpu"]
        quantarray = hw_config["modules"]["quant array"]

        # ---- systolic ----
        systolic_ratio = (
            systolic["num"] * systolic["rows"] * systolic["cols"]
        ) / (16 * 16)
        systolic_area = cfg["systolic_array_16x16_fp16"]["area"] * systolic_ratio

        # ---- lut ----
        lut_area = cfg["sram_256x16bits_1rw"]["area"] * lut["num"] * 2 # double buffer

        # ---- adder tree ----
        assert adder["width"] == 32
        adder_area = cfg["fp16_add_tree_32"]["area"] * adder["num"]

        # ---- vpu ----
        vpu_area = sum([
            vpu["vfadd"]["num"] * cfg["vfadd_WIDTH32_DATAWIDTH16"]["area"],
            vpu["vfmul"]["num"] * cfg["vfmul_WIDTH32_DATAWIDTH16"]["area"],
            vpu["adder tree"]["num"] * cfg["fp16_add_tree_4"]["area"],
        ])

        # ---- quant ----
        quant_area = (
            quantarray["num"]
            * quantarray["width"] / 8
            * cfg["qua_update_WIDTH8_DATAWIDTH16_IDXWIDTH8"]["area"]
        )

        softmax = hw_config["modules"]["sfu"]["softmax"]
        assert softmax["num"] == 1 and softmax["width"] == 32 and softmax["dtype"] == "fp16"
        sfu_area = cfg["softmax_unit"]["area"]

        core_total = (
            systolic_area + lut_area + adder_area + vpu_area + quant_area + sfu_area
        )

        breakdown["core"] = {
            "mxu": systolic_area * 1e-6,
            "lut": lut_area * 1e-6,
            "vpu": (adder_area+vpu_area) * 1e-6,
            "quant": quant_area * 1e-6,
            "sfu": sfu_area * 1e-6,
            "total": core_total * 1e-6,
        }

        # ================= buffer =================
        grouped = self._group_buffers(hw_config["buffers"])

        lt_detail = {}
        gt_detail = {}

        total_lt = 0
        total_gt = 0

        for base, pair in grouped.items():
            for suffix, name in pair.items():
                spec = hw_config["buffers"][name]

                if self._is_zero_buffer(spec):
                    continue

                block_bytes = spec["block_size"]
                size_per_bank = spec["size"] // spec["num"]
                num_entries = size_per_bank // block_bytes

                key = f"sram_{num_entries}x{block_bytes*8}bits_1rw"
                if key not in cfg:
                    raise KeyError(f"Missing buffer config: {key}")

                area = cfg[key]["area"] * spec["num"]

                if suffix == "lt":
                    lt_detail[name] = area * 1e-6
                    total_lt += area
                else:
                    gt_detail[name] = area * 1e-6
                    total_gt += area

        buffer_total = max(total_lt, total_gt)

        breakdown["buffer"] = {
            "lt": lt_detail,
            "gt": gt_detail,
            "lt_total": total_lt * 1e-6,
            "gt_total": total_gt * 1e-6,
            "selected_total": buffer_total * 1e-6,
        }

        # ================= overall =================
        breakdown["total"] = (
            breakdown["core"]["total"] + breakdown["buffer"]["selected_total"]
        )

        return breakdown

    def get_area(self, hw_config):
        cfg = self.area_power_energy_config
        core_area = self.get_core_area(hw_config)

        grouped = self._group_buffers(hw_config["buffers"])

        total_lt = 0
        total_gt = 0

        for base, pair in grouped.items():
            for suffix, name in pair.items():
                spec = hw_config["buffers"][name]

                if self._is_zero_buffer(spec):
                    continue

                block_bytes = spec["block_size"]
                size_per_bank = spec["size"] // spec["num"]
                num_entries = size_per_bank // block_bytes

                key = f"sram_{num_entries}x{block_bytes*8}bits_1rw"
                if key not in cfg:
                    raise KeyError(f"Missing buffer config: {key}")

                area = cfg[key]["area"] * spec["num"]

                if suffix == "lt":
                    total_lt += area
                else:
                    total_gt += area

        buffer_area = max(total_lt, total_gt)

        return core_area + buffer_area * 1e-6

    # ------------------------
    # Power
    # ------------------------


    # ------------------------
    # ------------------------
    def get_core_energy(self, hw_config, stats, total_cycles, cycle_breakdown, frequency=500):
        cfg = self.area_power_energy_config
        factor = 1e-3 / (frequency * 1e6)

        systolic = hw_config["modules"]["systolic array"]
        lut = hw_config["modules"]["lut array"]
        adder = hw_config["modules"]["adder tree"]

        # systolic
        systolic_ratio = (systolic["num"] * systolic["rows"] * systolic["cols"]) / (16 * 16)
        systolic_energy = (
            cfg["systolic_array_16x16_fp16"]["static_power"] * total_cycles +
            cfg["systolic_array_16x16_fp16"]["dynamic_power"] * cycle_breakdown["systolic array"]
        ) * systolic_ratio * factor

        # ===== LUT (static + cap dynamic) =====
        lut_static = (
            cfg["sram_256x16bits_1rw"]["static_power"] * total_cycles * factor
        ) * lut["num"] * 2  # double buffer
        # Codebook-cap accesses are billed here, not in the buffer loop: s_cap
        # is not a declared buffer in the hw config, so get_energy skips it.
        cap_read = stats.reads.get("s_cap", 0)+stats.reads.get("s_cap_lt", 0)+stats.reads.get("s_cap_gt", 0)
        cap_write = stats.writes.get("s_cap", 0)+stats.writes.get("s_cap_lt", 0)+stats.writes.get("s_cap_gt", 0)
        cfg_lut = cfg["sram_256x16bits_1rw"]
        cap_dyn = (
            cap_read * cfg_lut["read_energy"] +
            cap_write * cfg_lut["write_energy"]
        ) * 1e-9
        lut_energy = lut_static + cap_dyn

        assert adder["width"] == 32
        adder_ratio = adder["num"]
        adder_energy = (
            cfg["fp16_add_tree_32"]["static_power"] * total_cycles +
            cfg["fp16_add_tree_32"]["dynamic_power"] * cycle_breakdown["adder tree"]
        ) * adder_ratio * factor


        vpu = hw_config["modules"]["vpu"]
        quantarray = hw_config["modules"]["quant array"]
        vpu_static = sum([
            vpu["vfadd"]["num"]*cfg["vfadd_WIDTH32_DATAWIDTH16"]["static_power"],
            vpu["vfmul"]["num"]*cfg["vfmul_WIDTH32_DATAWIDTH16"]["static_power"],
            vpu["adder tree"]["num"]*cfg["fp16_add_tree_4"]["static_power"],
        ])
        vpu_qua_energy = vpu_static * total_cycles * factor
        vpu_energy = vpu_static * total_cycles * factor

        qu_static = quantarray["num"]*quantarray["width"]/8*cfg["qua_update_WIDTH8_DATAWIDTH16_IDXWIDTH8"]["static_power"]
        qu_energy = qu_static * total_cycles * factor
        # dynamic power of these modules are ignored since they are too small.

        softmax = hw_config["modules"]["sfu"]["softmax"]
        assert softmax["num"] == 1 and softmax["width"] == 32 and softmax["dtype"] == "fp16"
        sfu_energy = cfg["softmax_unit"]["static_power"] * total_cycles * factor

        breakdown = {
            "mxu": systolic_energy,
            # "lut array (static)": lut_energy,
            "lut": lut_energy,
            "vpu": adder_energy+vpu_energy,
            "qu": qu_energy,
            "sfu": sfu_energy,
        }

        total_energy = sum(breakdown.values())
        if total_cycles == 0:
            total_power = 0.0
        else:
            total_power = total_energy / (total_cycles / (frequency * 1e6))

        return {
            "total_energy (J)": total_energy,
            "total_power (W)": total_power,
            "breakdown": breakdown,
        }


    def get_energy(self, hw_config, stats, frequency=500):
        cfg = self.area_power_energy_config

        total_cycles = stats.total_cycles
        if stats.cycle_breakdown is None:
            init_cycle_breakdown(stats)

        bd = stats.cycle_breakdown
        core = self.get_core_energy(hw_config, stats, total_cycles, bd, frequency)

        factor = 1e-3 / (frequency * 1e6)

        grouped = self._group_buffers(hw_config["buffers"])

        buffer_energy = 0.0
        static_lt = 0.0
        static_gt = 0.0

        # ===== consistency check =====
        valid_bases = set(grouped.keys())

        # Collect every memory space touched, excluding dram, and fail loudly on
        # one the hw config does not declare -- silently dropping its energy is
        # how the codebook cap went unbilled in an earlier build.
        def _to_base(name: str) -> str:
            if name.endswith("_lt") or name.endswith("_gt"):
                return name.rsplit("_", 1)[0]
            return name
        accessed = {
            _to_base(k)
            for k in set(stats.reads) | set(stats.writes)
            if k != "dram" and (
                stats.reads.get(k, 0) > 0 or stats.writes.get(k, 0) > 0
            )
        }
        valid_bases |= {"s_cap", "buffer"}
        invalid = accessed - valid_bases
        if invalid:
            raise ValueError(f"[Buffer Mismatch] Stats contains unknown buffers: {invalid}")

        for base, pair in grouped.items():
            # Skip s_cap: already charged in the core's LUT term.
            if base == "s_cap":
                continue
            for suffix, name in pair.items():
                spec = hw_config["buffers"][name]

                if self._is_zero_buffer(spec):
                    continue

                block_bytes = spec["block_size"]
                size_per_bank = spec["size"] // spec["num"]
                num_entries = size_per_bank // block_bytes

                key = f"sram_{num_entries}x{block_bytes*8}bits_1rw"
                if key not in cfg:
                    raise KeyError(f"Missing buffer config: {key}")

                cfg_buf = cfg[key]

                read_bits = stats.reads.get(base, 0)
                write_bits = stats.writes.get(base, 0)

                dyn_energy = (
                    read_bits * cfg_buf["read_energy"] +
                    write_bits * cfg_buf["write_energy"]
                ) * 1e-9

                static_energy = cfg_buf["static_power"] * total_cycles * factor

                if suffix == "lt":
                    static_lt += static_energy * spec["num"]
                else:
                    static_gt += static_energy * spec["num"]

                buffer_energy += (dyn_energy + static_energy) * spec["num"]

        # The _lt and _gt buffer sets are alternative assignments -- only one is
        # physically present -- so leakage is the max of the two, not the sum.
        buffer_static = max(static_lt, static_gt)

        # Dynamic energy needs no such adjustment: it is already driven by the
        # access counts, and only the assignment actually used records any.
        total_buffer_energy = buffer_energy - (static_lt + static_gt) + buffer_static

        # ===== DRAM =====
        dram_static_energy = cfg["default_dram"]["dram_leak_energy"] * total_cycles * factor
        dram_read_energy = stats.reads["dram"] * cfg["default_dram"]["dram_cost_read"] * 1e-9
        dram_write_energy = stats.writes["dram"] * cfg["default_dram"]["dram_cost_write"] * 1e-9
        dram_energy = dram_static_energy + dram_read_energy + dram_write_energy

        total_energy = core["total_energy (J)"] + total_buffer_energy + dram_energy

        if total_cycles == 0:
            total_power = 0.0
        else:
            total_power = total_energy / (total_cycles / (frequency * 1e6))

        ret = {
            "total_energy (J)": total_energy,
            "core_power (W)": core["total_power (W)"],
            "total_power (W)": total_power,
            "energy_breakdown": {
                "dram": {
                    "dram_energy (J)": dram_energy,
                    "dram_static_energy": dram_static_energy,
                    "dram_read_energy": dram_read_energy,
                    "dram_write_energy": dram_write_energy,
                },
                "core": {
                    "core_energy (J)": core["total_energy (J)"],
                    "core_breakdown": core["breakdown"],
                },
                "buffer": {
                    "buffer_energy (J)": buffer_energy,
                }
            },
        }
        return ret