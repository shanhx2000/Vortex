"""Loads hardware configuration JSON, one preset per simulated accelerator.

Presets in `hw_configs/` name the design: `vortex.json` plus one per baseline.
One preset per method, for both phases.

One environment override is supported, VORTEX_SFU_STATIC_W -- see
`apply_env_overrides`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from helper import deep_update, read_json


ROOT_DIR = Path(__file__).resolve().parent
HW_CONFIG_DIR = ROOT_DIR / "hw_configs"
DEFAULT_HW_CONFIG_PATH = HW_CONFIG_DIR / "hw_vortex_config.example.json"

# Overrides modules.sfu.softmax.power.static, in watts, for every preset.
#
# Exists for the `baseline_sfu` experiment: the baselines hardcode 0.05 W of
# softmax leakage in their own configs, while Vortex is charged 0.00201672 W
# from power_energy_config.json::softmax_unit -- a 25x gap between designs that
# should share one synthesised number. Setting this lets the baselines be re-run
# on Vortex's number without editing the checked-in configs.
#
# No effect on Vortex itself: HWVortex sources all energy from Evaluator and
# never reads sfu.power.static.
SFU_STATIC_ENV = "VORTEX_SFU_STATIC_W"


def load_base_hw_config(path: str | Path | None = None) -> Dict[str, Any]:
    return read_json(path or DEFAULT_HW_CONFIG_PATH)


def _preset_config_path(preset: str) -> Path:
    return HW_CONFIG_DIR / f"{preset}.json"


def load_preset_hw_config(preset: str) -> Dict[str, Any]:
    path = _preset_config_path(preset)
    if not path.exists():
        raise ValueError(f"Unknown HW preset: {preset}")
    return read_json(path)


def list_hw_presets() -> list[str]:
    ignored = {"hw_vortex_config.example", "power_energy_config"}
    return sorted(path.stem for path in HW_CONFIG_DIR.glob("*.json") if path.stem not in ignored)


def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply VORTEX_SFU_STATIC_W, if set, to a freshly loaded config.

    Mutates and returns `config`, which is safe because every caller has just
    built it from `read_json` / `deep_update` and owns it outright.
    """
    raw = os.environ.get(SFU_STATIC_ENV)
    if raw is None:
        return config
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{SFU_STATIC_ENV} must be a number in watts, got {raw!r}")

    softmax = config.get("modules", {}).get("sfu", {}).get("softmax")
    if softmax is None:
        return config
    softmax.setdefault("power", {})["static"] = value
    return config


def get_hw_config(
    preset: str = "vortex",
    overrides: Optional[Dict[str, Any]] = None,
    *,
    base_path: str | Path | None = None,
) -> Dict[str, Any]:
    config = load_preset_hw_config(preset)
    if base_path is not None:
        config = deep_update(load_base_hw_config(base_path), config)
    return apply_env_overrides(deep_update(config, overrides))
