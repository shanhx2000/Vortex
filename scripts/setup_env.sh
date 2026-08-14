#!/usr/bin/env bash
# =============================================================================
# Vortex AE -- environment setup.
#
#   ./scripts/setup_env.sh                    # simulator + visualization
#   ./scripts/setup_env.sh --venv             # force a venv instead of conda
#   ./scripts/setup_env.sh -n myenv           # choose the environment name
#   ./scripts/setup_env.sh algorithm          # separate env: TEAL + AQLM/RVQ
#
# Options
#   -c, --conda   use conda (default when `conda` is on PATH)
#   -v, --venv    use python -m venv, created at ./.venv
#   -n NAME       conda environment name          (default: vortex-ae, or
#                 vortex-ae-algorithm for the `algorithm` component)
#   -p VERSION    python version for conda        (default: 3.10, or 3.11 for
#                 `algorithm` -- see algorithm/README.md)
#       --dry-run print the commands, run nothing
#   -h            this help
#
# ONE ENVIRONMENT SERVES BOTH the simulator and the plot scripts. That was
# checked, not assumed: the simulator produces bit-identical output on
# Python 3.8.8/pandas 1.2.4 and Python 3.10.20/pandas 2.3.3 -- every numeric
# column matches exactly, and the power/area JSON is byte-identical.
#
# The one real constraint is **matplotlib >= 3.4**, needed for Figure.supxlabel
# (figures AD and BF). Python 3.8 is fine; a matplotlib older than 3.4 is not.
#
# `algorithm` IS a separate environment, and always its own conda
# env/venv -- it pulls in torch, flash-attn and AQLM, none of which belong in
# the lightweight pandas/matplotlib environment above, and it is not
# combinable with `simulator`/`visualization`/`all` in one invocation (ask for
# it alone: `./scripts/setup_env.sh algorithm`). It installs
# algorithm/requirements.txt, then `pip install -e algorithm/`.
#
# Requirements are read from the component files rather than duplicated here:
#   simulator/requirements.txt
#   evaluation/visualization/requirements.txt
#   algorithm/requirements.txt
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"

BACKEND=""
ENV_NAME="vortex-ae"
PY_VERSION="3.10"
DRYRUN=0
COMPONENTS=()

usage() { awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print } else { exit } }' "${BASH_SOURCE[0]}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        -c|--conda) BACKEND=conda ;;
        -v|--venv)  BACKEND=venv ;;
        -n)         ENV_NAME="$2"; shift ;;
        -p)         PY_VERSION="$2"; shift ;;
        --dry-run)  DRYRUN=1 ;;
        -h|--help)  usage; exit 0 ;;
        -*)         echo "Unknown option: $1" >&2; exit 2 ;;
        *)          COMPONENTS+=("$1") ;;
    esac
    shift
done

[ ${#COMPONENTS[@]} -eq 0 ] && COMPONENTS=(simulator visualization)

run() {
    echo "  \$ $*"
    [ "$DRYRUN" -eq 1 ] || "$@"
}

# ---- algorithm: always its own environment, handled and exited here --------
# Not foldable into the REQS loop below: it needs its own conda env/venv (see
# header), so it cannot be combined with simulator/visualization/all in one
# invocation.
if [[ " ${COMPONENTS[*]} " == *" algorithm "* ]]; then
    if [ "${#COMPONENTS[@]}" -ne 1 ]; then
        echo "algorithm needs its own environment -- run it alone:" >&2
        echo "  ./scripts/setup_env.sh algorithm" >&2
        exit 2
    fi

    ALG_ENV_NAME="$ENV_NAME";      [ "$ENV_NAME" = "vortex-ae" ]   && ALG_ENV_NAME="vortex-ae-algorithm"
    ALG_PY_VERSION="$PY_VERSION";  [ "$PY_VERSION" = "3.10" ]      && ALG_PY_VERSION="3.11"
    ALG_REQ="$REPO/algorithm/requirements.txt"
    ALG_PKG="$REPO/algorithm"

    if [ -z "$BACKEND" ]; then
        if command -v conda >/dev/null 2>&1; then BACKEND=conda; else BACKEND=venv; fi
    fi

    echo "component  : algorithm"
    echo "backend    : $BACKEND"
    echo

    case "$BACKEND" in
    conda)
        if conda env list | awk '{print $1}' | grep -qx "$ALG_ENV_NAME"; then
            echo "conda env '$ALG_ENV_NAME' already exists; installing into it"
        else
            run conda create -y -n "$ALG_ENV_NAME" "python=$ALG_PY_VERSION"
        fi
        PIP=(conda run -n "$ALG_ENV_NAME" python -m pip)
        ACTIVATE="conda activate $ALG_ENV_NAME"
        ;;
    venv)
        VENV="$REPO/.venv-algorithm"
        [ -d "$VENV" ] || run python3 -m venv "$VENV"
        PIP=("$VENV/bin/python" -m pip)
        ACTIVATE="source ${VENV#$REPO/}/bin/activate"
        ;;
    esac

    run "${PIP[@]}" install --upgrade pip
    echo "installing algorithm/requirements.txt (torch, transformers, aqlm, lm_eval -- large; expect several minutes)"
    run "${PIP[@]}" install -r "$ALG_REQ"
    # flash-attn's setup.py imports torch at build time -- must be its own
    # command, after torch is already installed, with build isolation off so
    # its build sees the torch just installed above. See requirements.txt's
    # comment.
    echo "installing flash-attn (separate step, needs torch already present; can take 10-20 min to build)"
    run "${PIP[@]}" install flash-attn==2.8.3 --no-build-isolation
    echo "installing algorithm/ (editable, the vortex package)"
    run "${PIP[@]}" install -e "$ALG_PKG"

    echo
    echo "done. activate with:  $ACTIVATE"
    echo "then:  python -c 'import vortex; print(vortex.__version__)'"
    [ "$DRYRUN" -eq 1 ] && echo "(dry run -- nothing was executed)"
    exit 0
fi

# ---- resolve components to requirement files --------------------------------
REQS=()
for c in "${COMPONENTS[@]}"; do
    case "$c" in
        simulator)     REQS+=("$REPO/simulator/requirements.txt") ;;
        visualization) REQS+=("$REPO/evaluation/visualization/requirements.txt") ;;
        all)           REQS+=("$REPO/simulator/requirements.txt"
                              "$REPO/evaluation/visualization/requirements.txt") ;;
        *) echo "Unknown component: $c (simulator | visualization | algorithm | all)" >&2
           exit 2 ;;
    esac
done

# de-duplicate while preserving order
UNIQ=()
for r in "${REQS[@]}"; do
    case " ${UNIQ[*]-} " in *" $r "*) continue ;; esac
    UNIQ+=("$r")
done

# ---- pick a backend ---------------------------------------------------------
if [ -z "$BACKEND" ]; then
    if command -v conda >/dev/null 2>&1; then BACKEND=conda; else BACKEND=venv; fi
fi

echo "components : ${COMPONENTS[*]}"
echo "backend    : $BACKEND"
echo

case "$BACKEND" in
conda)
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "conda env '$ENV_NAME' already exists; installing into it"
    else
        run conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
    fi
    PIP=(conda run -n "$ENV_NAME" python -m pip)
    ACTIVATE="conda activate $ENV_NAME"
    ;;
venv)
    VENV="$REPO/.venv"
    [ -d "$VENV" ] || run python3 -m venv "$VENV"
    PIP=("$VENV/bin/python" -m pip)
    ACTIVATE="source ${VENV#$REPO/}/bin/activate"
    ;;
esac

run "${PIP[@]}" install --upgrade pip
for r in "${UNIQ[@]}"; do
    echo "installing $(realpath --relative-to="$REPO" "$r")"
    run "${PIP[@]}" install -r "$r"
done

# ---- optional system tool ---------------------------------------------------
if ! command -v pdftoppm >/dev/null 2>&1; then
    echo
    echo "note: pdftoppm not found (package: poppler-utils)."
    echo "      Only verify_ref.py needs it, to rasterize PDFs for the"
    echo "      side-by-side comparison sheets. Everything else works without it."
fi

echo
echo "done. activate with:  $ACTIVATE"
[ "$DRYRUN" -eq 1 ] && echo "(dry run -- nothing was executed)"
