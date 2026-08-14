#!/usr/bin/env bash
# =============================================================================
# Vortex AE -- create the two conda environments. Both by default.
#
#   ./setup_env_all.sh            # sim and alg
#   ./setup_env_all.sh sim        # just one
#   ./setup_env_all.sh --dry-run  # print the commands, create nothing
#
#   sim   C1 and the figures: pandas, tqdm, matplotlib, numpy, scikit-learn,
#         Pillow. Python 3.10. Used by run_simulation.sh and visual_results.sh,
#         which find it on their own -- no `conda activate` needed.
#         One environment covers simulating and plotting: the two requirement
#         sets share only pandas, at the same floor, and figure BC imports
#         simulator modules anyway, so splitting them would buy nothing.
#   alg   C2: torch, flash-attn, transformers, aqlm, lm-eval. Python 3.11,
#         needs a GPU, and takes far longer to build than sim -- mostly
#         flash-attn. Used by run_algorithm.sh; needs `conda activate alg`.
#
# CONDA REQUIRED. Naming two environments needs it: scripts/setup_env.sh's
# --venv backend writes to fixed paths. Without conda, call that script
# directly and manage the venvs yourself.
#
# If alg fails while building flash-attn, the host's default nvcc is too new
# for torch==2.9.1 (cu128). Re-run pointing at a 12.x toolkit:
#   CUDA_HOME=/usr/local/cuda-12.4 ./setup_env_all.sh alg
# See README.md.
#
# A wrapper over scripts/setup_env.sh, which stays the place to look for the
# per-component options (`scripts/setup_env.sh -h`).
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP="$HERE/scripts/setup_env.sh"

usage() { awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print } else { exit } }' "${BASH_SOURCE[0]}"; }

[ -f "$SETUP" ] || { echo "missing: $SETUP" >&2; exit 1; }

DRYRUN=()
WANTED=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)  DRYRUN=(--dry-run) ;;
        -h|--help)  usage; exit 0 ;;
        sim|alg) WANTED+=("$1") ;;
        *) echo "Unknown argument: $1 (expected sim | alg | --dry-run)" >&2
           exit 2 ;;
    esac
    shift
done
[ ${#WANTED[@]} -eq 0 ] && WANTED=(sim alg)

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found -- the two named environments require it." >&2
    echo "Run scripts/setup_env.sh directly instead:" >&2
    echo "  ./scripts/setup_env.sh --venv            # simulator + visualization" >&2
    echo "  ./scripts/setup_env.sh --venv algorithm  # the algorithm environment" >&2
    exit 1
fi

# env name -> the scripts/setup_env.sh component that fills it.
# `sim` gets `all` (simulator + visualization), not `simulator`: it runs the
# plot scripts too, and figure BC imports simulator modules either way.
component_for() {
    case "$1" in
        sim) echo all ;;
        alg) echo algorithm ;;
    esac
}

FAILED=()
for env in "${WANTED[@]}"; do
    comp="$(component_for "$env")"
    echo
    echo "===== $env  (component: $comp) ====="
    if ! "$SETUP" -c -n "$env" "${DRYRUN[@]+"${DRYRUN[@]}"}" "$comp"; then
        FAILED+=("$env")
    fi
done

echo
if [ ${#FAILED[@]} -ne 0 ]; then
    echo "failed: ${FAILED[*]}" >&2
    exit 1
fi

echo "===== done ====="
echo "  conda activate sim   # then ./run_simulation.sh or ./visual_results.sh"
echo "  conda activate alg   # then ./run_algorithm.sh"
echo
echo "run_simulation.sh and visual_results.sh pick sim up automatically when it"
echo "exists, so you can drive the whole of C1 from any shell."
