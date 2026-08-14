"""Shared path setup for evaluation scripts.

These scripts live in evaluation/evaluation_scripts/ but import the simulator
modules, so the simulator directory has to be on sys.path. Importing this
module does that and exposes the repo's well-known directories.

    import _paths
    from _paths import SIM_RESULTS_DIR, FIGURES_DIR

Layout
------
    simulator/              simulator source + hw_configs
    stats/
      simulation/           generated: sweep CSVs, kernel results, raw stats
      algorithm/            generated: C2's accuracy CSVs (AC/AD/AE)
      ref/                  committed: what the AE submission expects
        simulation/           the same filenames a C1 run produces
        algorithm/            the same filenames a C2 run produces
        sparsity_info/        TEAL sparsity thresholds -- C1's input
    figures/                generated: figure PDFs

These scripts only ever *write*, so they resolve `stats/simulation/`. Reading a
different data set is a plotting concern, handled by
`evaluation/visualization/_paths.py`'s `--use-ref` / `--use-simulation` /
`--use-algorithm`.
"""
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent.parent

SIM_DIR = REPO_ROOT / "simulator"
HW_CONFIG_DIR = SIM_DIR / "hw_configs"

STATS_DIR = REPO_ROOT / "stats"
SIM_RESULTS_DIR = STATS_DIR / "simulation"
REF_ROOT = STATS_DIR / "ref"
SPARSITY_INFO_DIR = REF_ROOT / "sparsity_info"

FIGURES_DIR = REPO_ROOT / "figures"

_RESULTS_DIR = SIM_RESULTS_DIR


def set_results_dir(path):
    """Redirect this script's output. run_simulation.sh -s NAME points every
    group at stats/simulation/session_NAME/; without this the scripts that do
    not go through the sweep driver would keep writing to the shared
    directory, which is exactly what -s exists to prevent."""
    global _RESULTS_DIR
    if path:
        _RESULTS_DIR = Path(path).expanduser().resolve()
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return _RESULTS_DIR


def results_dir():
    return _RESULTS_DIR


# Make `from model_runner import ...` etc. resolve to the extracted simulator.
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

# Only generated directories are created on demand; committed inputs must exist.
for _d in (SIM_RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
