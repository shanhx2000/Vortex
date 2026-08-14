#!/usr/bin/env bash
# =============================================================================
# Vortex AE -- draw the paper figures. Env `sim`; no GPU, no simulation.
#
#   ./visual_results.sh                    # all eleven, from stats/ref/
#   ./visual_results.sh AC AD              # only these
#   ./visual_results.sh -l                 # list figure indices
#   ./visual_results.sh --use-simulation   # BA-BF from your own C1 run
#   ./visual_results.sh --use-algorithm    # AC-AE from your own C2 run
#
# **Reads the committed reference data by default**, so this works on a fresh
# clone with no simulation and reproduces figures/ref/ byte for byte. The two
# component switches are independent: --use-simulation after ./run_simulation.sh
# (C1), --use-algorithm after `./run_algorithm.sh ... build` (C2), or both.
# Everything writes to figures/.
#
# A figure whose input data is missing is reported and skipped, not fatal, so
# running only one component is fine.
#
#   AA-AE  algorithm figures, always from stats/ref/algorithm/   (C2)
#   BA-BF  hardware figures, from the selected source        (C1)
#
# Bare indices become plot_all.py's --only; every other argument is forwarded
# as-is. See `./visual_results.sh -h` for the rest.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLOT_DIR="$HERE/evaluation/visualization"
[ -f "$PLOT_DIR/plot_all.py" ] || { echo "missing: $PLOT_DIR/plot_all.py" >&2; exit 1; }

# `conda env list` prints the path last on the line, with or without the `*`
# that marks the active environment. Falls back to whatever `python` is active.
env_python() {
    command -v conda >/dev/null 2>&1 || return 1
    local p
    p="$(conda env list 2>/dev/null | awk -v n="$1" '$1==n {print $NF}')"
    [ -n "$p" ] && [ -x "$p/bin/python" ] && echo "$p/bin/python"
}
PLOT_PYTHON="${PLOT_PYTHON:-$(env_python sim || true)}"
[ -n "$PLOT_PYTHON" ] || PLOT_PYTHON="${PYTHON:-python}"

# Positional indices are collected into one --only; -l is spelled --list there.
# If the caller wrote --only itself, forward everything verbatim instead --
# otherwise its indices would be pulled out and re-appended, leaving --only to
# swallow whichever flag came next.
ARGS=()
if [[ " $* " == *" --only "* ]]; then
    ARGS=("$@")
else
    ONLY=()
    for a in "$@"; do
        case "$a" in
            -l)          ARGS+=(--list) ;;
            [A-Z][A-Z])  ONLY+=("$a") ;;
            *)           ARGS+=("$a") ;;
        esac
    done
    [ ${#ONLY[@]} -gt 0 ] && ARGS+=(--only "${ONLY[@]}")
fi

echo "===== figures: plot_all.py ${ARGS[*]-} ====="
echo "      python: $PLOT_PYTHON"
cd "$PLOT_DIR" && exec "$PLOT_PYTHON" plot_all.py "${ARGS[@]+"${ARGS[@]}"}"
