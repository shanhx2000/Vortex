#!/usr/bin/env bash
# =============================================================================
# Vortex AE -- C2, codebook-wise contextual sparsity. Needs a GPU and env `alg`.
#
#   ./run_algorithm.sh                          # all models x methods, all phases
#   ./run_algorithm.sh search                   # one phase
#   ./run_algorithm.sh -m llama2_7b eval        # one model, one phase
#   ./run_algorithm.sh -w codebookwise          # one method
#   ./run_algorithm.sh --use-ref -m llama2_7b eval   # eval the SHIPPED search
#   ./run_algorithm.sh -n                       # dry run: print the commands
#   ./run_algorithm.sh -l                       # list models, methods, phases
#
# Phases, given as positional arguments and always run in this order:
#
#   prepare   build the per-codebook model from the AQLM checkpoint
#             (codebookwise only; nothing to do for uniform)
#   search    grab_acts -> greedyopt: find the sparsity thresholds
#   eval      ppl_test: WikiText-2 perplexity + six zero-shot tasks
#   gather    gather_thresholds: turn the search into the JSONL the simulator
#             reads, so C1 can be re-run against thresholds you searched
#             yourself. Not in the default set -- ask for it by name.
#   build     build_algorithm_csvs: collect the eval records into
#             stats/algorithm/, so `visual_results.sh --use-simulation AC AD AE`
#             draws your numbers instead of ours. Also opt-in.
#
# Options
#   -m LIST     models   (default: llama2_7b,mistral_7b,llama2_13b)
#   -w LIST     methods  (default: uniform,codebookwise)
#   -s LIST     sparsities for eval (default: 0.0,0.3)
#   -t TAG      write to ckpts/teal_output/<model>-TAG/ (default: aerun<date>)
#   --use-ref   read the COMMITTED search under ckpts/teal_output/<model>/
#               instead of a tagged run of your own. Equivalent to -t "".
#   -n / -l / -h
#
# --use-ref is how to spend ten minutes on C2 instead of a GPU-day and a half.
# Two Llama-2-7B searches ship (uniform and codebookwise), so
#
#   ./run_algorithm.sh --use-ref -m llama2_7b eval
#
# evaluates them directly and skips the ~1.5 GPU-days the search would cost. It
# reads those directories; nothing writes back into them unless you ask for a
# phase that produces output, which is why every other command gets a tag.
#
# All models are the 2-bit AQLM checkpoints. `uniform` is TEAL's codebook-
# uniform thresholding, `codebookwise` is ours; the paper compares the two.
#
# Runs in a tmux session, one model at a time -- there is one GPU, so asking
# for several models queues them rather than parallelizing. Forwards to
# algorithm/scripts/run_teal_group.sh; see that for the rest.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="$HERE/algorithm/scripts/run_teal_group.sh"
[ -f "$DRIVER" ] || { echo "missing: $DRIVER" >&2; exit 1; }

MODELS="llama2_7b,mistral_7b,llama2_13b"
METHODS="uniform,codebookwise"
SPARSITIES="0.0,0.3"
TAG=""
USE_REF=0
PASS=()          # options forwarded to the driver untouched

usage() { awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print } else { exit } }' "${BASH_SOURCE[0]}"; }

list_all() {
    cat <<'EOF'
models   llama2_7b  mistral_7b  llama2_13b        (2-bit AQLM checkpoints)
methods  uniform        TEAL, codebook-uniform thresholding
         codebookwise   ours
phases   prepare  build the per-codebook model     (codebookwise only)
         search   grab_acts -> greedyopt           ~1.5 GPU-days per model
         eval     ppl_test at the -s sparsities
         gather   thresholds -> the JSONL run_simulation.sh --sparsity-info
                  reads                            (not run by default)
         build    eval records -> stats/algorithm/, the CSVs AC/AD/AE read
                                                   (not run by default)

Default: every model x every method, prepare + search + eval, eval at 0.0,0.3.
Budget roughly 1.5 A100-days per model for `search`.

Cheap path -- evaluate the shipped Llama-2-7B searches, no search needed:
  ./run_algorithm.sh --use-ref -m llama2_7b eval
EOF
}

# --use-ref is a long option; getopts handles short ones only, so pull it out
# first. It means "target the canonical, committed teal_output directories",
# which the driver spells as an empty -t.
ARGV=()
while [ $# -gt 0 ]; do
    case "$1" in
        --use-ref) USE_REF=1; shift ;;
        *)         ARGV+=("$1"); shift ;;
    esac
done
set -- "${ARGV[@]+"${ARGV[@]}"}"

while getopts ":m:w:s:t:nlh" opt; do
    case "$opt" in
        m) MODELS="$OPTARG" ;;
        w) METHODS="$OPTARG" ;;
        s) SPARSITIES="$OPTARG" ;;
        t) TAG="$OPTARG"; PASS+=(-t "$OPTARG") ;;
        n) PASS+=(-n) ;;
        l) list_all; exit 0 ;;
        h) usage; exit 0 ;;
        \?) echo "Unknown option: -$OPTARG (try -h)" >&2; exit 2 ;;
        :)  echo "Option -$OPTARG needs an argument" >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

# --- phases: positional. `gather` is opt-in, so it is not in the default set.
STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(prepare search eval)
for st in "${STAGES[@]}"; do
    case "$st" in
        prepare|search|eval|gather|build) ;;
        *) echo "Unknown phase: $st (expected prepare | search | eval | gather | build)" >&2
           exit 2 ;;
    esac
done

# --use-ref and -t are two ways to say the same thing; refuse to guess.
if [ "$USE_REF" -eq 1 ]; then
    for p in "${PASS[@]+"${PASS[@]}"}"; do
        [ "$p" = "-t" ] && { echo "--use-ref and -t are mutually exclusive" >&2; exit 2; }
    done
    PASS+=(-t "")
    for st in "${STAGES[@]}"; do
        case "$st" in
            prepare|search)
                echo "--use-ref selects the committed directories, which a '$st' run would" >&2
                echo "write into. Drop --use-ref (a tagged run is written elsewhere) or" >&2
                echo "ask only for read-only phases: eval, gather, build." >&2
                exit 2 ;;
        esac
    done
fi

# --- models x methods -> the driver's model keys -----------------------------
# uniform      -> <model>_aqlm       (TEAL thresholding on the AQLM checkpoint)
# codebookwise -> <model>_aqlm_rvq   (ours)
KEYS=()
IFS=',' read -ra _models  <<< "$MODELS"
IFS=',' read -ra _methods <<< "$METHODS"
for m in "${_models[@]}"; do
    m="${m// /}"; [ -z "$m" ] && continue
    case "$m" in
        llama2_7b|mistral_7b|llama2_13b) ;;
        *) echo "Unknown model: $m (run -l to list)" >&2; exit 2 ;;
    esac
    for w in "${_methods[@]}"; do
        w="${w// /}"; [ -z "$w" ] && continue
        case "$w" in
            uniform)      KEYS+=("${m}_aqlm") ;;
            codebookwise) KEYS+=("${m}_aqlm_rvq") ;;
            *) echo "Unknown method: $w (expected uniform | codebookwise)" >&2; exit 2 ;;
        esac
    done
done
[ ${#KEYS[@]} -eq 0 ] && { echo "Nothing selected." >&2; exit 2; }

stages_csv="$(IFS=,; echo "${STAGES[*]}")"
exec "$DRIVER" "${PASS[@]+"${PASS[@]}"}" -S "$stages_csv" -s "$SPARSITIES" "${KEYS[@]}"
