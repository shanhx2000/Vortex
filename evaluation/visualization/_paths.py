"""Path resolution and the data-source switches.

Every plot script asks this module where its data lives instead of hard-coding a
path, so one flag redirects all of them.

Two data sources, one output location
-------------------------------------
`stats/ref/` mirrors the layout a run writes, so the same relative path names
the same file in both:

    stats/{simulation,algorithm}/<name>       what you just produced
    stats/ref/{simulation,algorithm}/<name>   what the paper carries [DEFAULT]

`simulation/` holds C1's results and `algorithm/` holds C2's, under the same
filenames in either source.

    --use-ref         (default)  both halves from stats/ref/
    --use-simulation             B-series (C1) from stats/simulation/
    --use-algorithm              A-series (C2) from stats/algorithm/

**The default reads `ref`, so a fresh clone draws the paper's figures without
running anything.** The two component switches are independent, because the two
components are: someone who ran only C1 passes `--use-simulation` and still gets
the A-series from `ref`.

Every mode writes to `figures/`. There is no per-mode output directory: the
figures are the same eleven files whichever data made them, and a comparison
between runs is `figures/` against `figures/ref/`, not a hunt through parallel
directories.

One further reference input sits under `stats/ref/`, because it is equally
"what the AE submission expects" and is not produced by either component alone:

    stats/ref/sparsity_info/   the contextual-sparsity threshold table
                               simulator/sparsity.py reads (C2's output, C1's
                               input -- the one artifact crossing that boundary)

Where figures land
------------------
figures/                     every run writes here. Generated, git-ignored.
figures/ref/                 committed reference set. What the default
                             `--use-ref` render reproduces, byte for byte.

The switches exist to separate two failure modes that otherwise look identical.
If a regenerated figure looks wrong, is the plot code wrong or is the data
different? The default render answers that: match `figures/ref/` and the code is
faithful, so any difference under `--use-simulation` / `--use-algorithm` is real
change in the numbers.
"""
import argparse
import sys
from pathlib import Path

VIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = VIS_DIR.parent.parent

SIM_DIR = REPO_ROOT / "simulator"
HW_CONFIG_DIR = SIM_DIR / "hw_configs"

STATS_DIR = REPO_ROOT / "stats"
# The two sources mirror each other; `simulation/` is the subdirectory a run
# writes into, and sim_data() appends it to whichever root is selected.
SIM_RESULTS_ROOT = STATS_DIR
REF_ROOT = STATS_DIR / "ref"
RESULTS_SUBDIR = "simulation"     # C1 output
ALGORITHM_SUBDIR = "algorithm"    # C2 output: the accuracy CSVs behind AC/AD/AE

SPARSITY_INFO_DIR = REF_ROOT / "sparsity_info"

FIGURES_DIR = REPO_ROOT / "figures"
REF_FIGURES_DIR = FIGURES_DIR / "ref"

# Some plot scripts import simulator modules (BC reads the hardware config).
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

# One source per component, independently selectable. Both default to ref so a
# fresh clone draws the paper's figures with no arguments and nothing run.
_SIM_SOURCE = "ref"       # "ref" | "simulation"                  -- B-series
_ALGO_SOURCE = "ref"      # "ref" | "algorithm"                  -- A-series
_OUT_DIR = FIGURES_DIR

_SIM_ROOTS = {
    "ref": REF_ROOT,
    "simulation": SIM_RESULTS_ROOT,
}
_ALGO_ROOTS = {
    "ref": REF_ROOT,
    "algorithm": SIM_RESULTS_ROOT,
}


def add_common_args(parser):
    """Register the flags every plot script accepts."""
    parser.add_argument(
        "--use-ref", action="store_true",
        help="read stats/ref/ for both halves -- the data the AE submission "
             "expects. THE DEFAULT; reproduces figures/ref/, which is what the "
             "paper carries. Accepted explicitly so a command can be "
             "self-documenting.",
    )
    parser.add_argument(
        "--use-simulation", action="store_true",
        help="draw the hardware figures (BA-BF) from stats/simulation/ -- what "
             "run_simulation.sh just produced on this machine.",
    )
    parser.add_argument(
        "--use-algorithm", action="store_true",
        help="draw the algorithm figures (AC-AE) from stats/algorithm/ -- what "
             "`run_algorithm.sh ... build` just produced. Independent of "
             "--use-simulation; pass both to plot a full run of your own.",
    )
    parser.add_argument(
        "--no-svg", action="store_true",
        help="skip the companion .svg written next to each .pdf (the SVGs exist "
             "so Markdown previews can display the figures)",
    )
    parser.add_argument(
        "--out-dir", default=None, metavar="DIR",
        help="where to write the PDF (default: figures/, for every data source)",
    )
    return parser


def apply_common_args(args):
    global _SIM_SOURCE, _ALGO_SOURCE, _OUT_DIR
    import _style
    _style.set_svg(not getattr(args, "no_svg", False))

    sim = bool(getattr(args, "use_simulation", False))
    alg = bool(getattr(args, "use_algorithm", False))
    ref = bool(getattr(args, "use_ref", False))
    if ref and (sim or alg):
        raise SystemExit("--use-ref means both halves from stats/ref/, which is "
                         "already the default; drop it, or drop the other flag")
    _SIM_SOURCE = "simulation" if sim else "ref"
    _ALGO_SOURCE = "algorithm" if alg else "ref"
    # One output location for every source -- see the module docstring.
    _OUT_DIR = (Path(args.out_dir).expanduser().resolve()
                if getattr(args, "out_dir", None) else FIGURES_DIR)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUT_DIR


def parse_args(description, extra=None):
    """Standard entry point: build a parser, parse, apply. Returns the args."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    if extra is not None:
        extra(parser)
    args = parser.parse_args()
    apply_common_args(args)
    return args


def source():
    """The hardware-figure source: "ref" | "simulation"."""
    return _SIM_SOURCE


def algorithm_source():
    """The algorithm-figure source: "ref" | "algorithm"."""
    return _ALGO_SOURCE


def using_ref():
    return _SIM_SOURCE == "ref" and _ALGO_SOURCE == "ref"


def results_dir():
    """Where C1 results come from."""
    return _SIM_ROOTS[_SIM_SOURCE] / RESULTS_SUBDIR


def algorithm_dir():
    """Where C2 results come from."""
    return _ALGO_ROOTS[_ALGO_SOURCE] / ALGORITHM_SUBDIR


def out_dir():
    return _OUT_DIR


def figure_path(name):
    """Absolute path to write a figure to."""
    return out_dir() / name


def sim_data(name):
    """Resolve a simulation-produced input (CSV or JSON) under the selected
    source. Raises with an actionable message rather than letting pandas fail on
    a missing file, because "which group do I run?" is the only thing the caller
    wants to know.
    """
    path = results_dir() / name
    if path.exists():
        return path
    if _SIM_SOURCE == "simulation":
        raise FileNotFoundError(
            f"{path} does not exist.\n"
            f"  Generate it:      ./run_simulation.sh -l\n"
            f"  Or drop the flag to plot the committed reference data instead.")
    raise FileNotFoundError(
        f"{path} is missing from the committed {_SIM_SOURCE} data set.")


def csv_data(name):
    """Resolve a C2-produced input (the accuracy CSVs behind AC/AD/AE) under the
    selected source, the same way sim_data() resolves C1's."""
    path = algorithm_dir() / name
    if path.exists():
        return path
    if _ALGO_SOURCE == "algorithm":
        raise FileNotFoundError(
            f"{path} does not exist.\n"
            f"  Generate it:      ./run_algorithm.sh eval build\n"
            f"  Or drop --use-algorithm to plot the reference data instead.")
    raise FileNotFoundError(
        f"{path} is missing from the committed reference data.")


def reference_figure(name):
    """The committed reference PDF for a figure, for comparison against a fresh
    render. This is figures/ref/ -- the post-artifact-evaluation set, which is
    what a correct run should reproduce."""
    return REF_FIGURES_DIR / name

