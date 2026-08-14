#!/usr/bin/env bash
# =============================================================================
# Vortex algorithm side -- TEAL pipeline driver.
#
# Runs the full per-model pipeline (prepare_rvq [vq only] -> grab_acts ->
# greedyopt -> ppl_test smoke check) for one or more models from
# vortex.models.MODEL_REGISTRY, inside a tmux session so a multi-hour run
# survives an SSH disconnect.
#
#   ./run_teal_group.sh -l                       # list groups and what they cost
#   ./run_teal_group.sh -n llama2_7b              # dry run: print, don't run
#   ./run_teal_group.sh llama2_7b                 # one model, foreground-in-tmux
#   ./run_teal_group.sh dense                     # llama2_7b + llama2_13b + mistral_7b
#   ./run_teal_group.sh all                       # every model in the registry
#   ./run_teal_group.sh -S search llama2_7b_aqlm  # one stage only
#
# Options
#   -t TAG   ckpts/teal_output/<teal_dirname>-TAG/ instead of the canonical
#            <teal_dirname>/ (default: aerun<YYYYMMDD>, computed once so a
#            multi-group invocation shares one tag). Pass -t "" to target the
#            canonical directory -- only do this if you deliberately intend to
#            reproduce/extend the committed checkpoints (flagging.md #1).
#   -S LIST  comma-separated stages to run (default: prepare,search,eval)
#              prepare  run_prepare_rvq   (_rvq keys only; skipped otherwise)
#              search   run_grab_acts + run_greedyopt
#              eval     run_ppl_test
#              gather   gather_thresholds -> the JSONL simulator/sparsity.py
#                       reads. One file per invocation, covering every key.
#              build    build_algorithm_csvs -> stats/algorithm/, the CSVs
#                       figures AC/AD/AE read. Also once per invocation.
#            Stages always run in that order, whatever order they are listed in.
#   -s LIST  comma-separated sparsities for the eval stage (default: 0.0,0.3,0.6)
#   -n       dry run: print the commands, create no tmux session
#   -l       list groups and rough cost
#   -h       this help
#
# MIN_FREE_MB (env var, default 20000): before every model-loading step, wait
# (polling every 30s, no timeout) until at least this much GPU memory is
# free. Not optional on a shared host -- see the note below.
#
# -----------------------------------------------------------------------------
# Why this looks different from run_simulation.sh
#
# The simulator driver parallelizes across CPU cores with -j. There is
# exactly one GPU here, shared with whatever else is running on the host, so
# groups run ONE AT A TIME inside a single tmux session -- launching multiple
# concurrent model-loading jobs would just make all of them OOM against each
# other. Requesting several groups in one invocation queues them, it does not
# parallelize them.
#
# GPU headroom on a shared host is not stable for a session's duration --
# observed directly while building this: the GPU went from ~87/95 GB used, to
# fully free, back to ~87/95 GB used under a NEW pid for the same process
# name, all within one working session (flagging.md #8). So every
# model-loading step here waits for MIN_FREE_MB itself (`wait_for_gpu`,
# injected into the generated per-run pipeline.sh) rather than trusting a
# single check at the start of the script to still hold hours later.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALG="$(dirname "$HERE")"
REPO="$(dirname "$ALG")"
LOGROOT="$REPO/logs/algorithm"
PYTHON="${PYTHON:-python}"

TAG="aerun$(date +%Y%m%d)"
SPARSITIES="0.0,0.3,0.6"
STAGES="prepare,search,eval"
DRYRUN=0
TAG_SET=0

usage() { awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print } else { exit } }' "${BASH_SOURCE[0]}"; }

ALL_KEYS="llama2_7b llama2_7b_aqlm llama2_7b_aqlm_rvq llama2_13b llama2_13b_aqlm llama2_13b_aqlm_rvq mistral_7b mistral_7b_aqlm mistral_7b_aqlm_rvq"
DENSE_KEYS="llama2_7b llama2_13b mistral_7b"
AQLM_KEYS="llama2_7b_aqlm llama2_13b_aqlm mistral_7b_aqlm"
RVQ_KEYS="llama2_7b_aqlm_rvq llama2_13b_aqlm_rvq mistral_7b_aqlm_rvq"

list_groups() {
    cat <<EOF
model key (group)     mode     stages                              cost (A100)
--------------------  -------  ----------------------------------  ------------
llama2_7b             regular  grab_acts, greedyopt, ppl_test      not measured
llama2_7b_aqlm        regular  grab_acts, greedyopt, ppl_test      ~1.5 h *
llama2_7b_aqlm_rvq    vq       + prepare_rvq                       ~26.5 h *
llama2_13b            regular  grab_acts, greedyopt, ppl_test      not measured
llama2_13b_aqlm       regular  grab_acts, greedyopt, ppl_test      ~1.5 h
llama2_13b_aqlm_rvq   vq       + prepare_rvq                       ~1.5 days
mistral_7b            regular  grab_acts, greedyopt, ppl_test      not measured
mistral_7b_aqlm       regular  grab_acts, greedyopt, ppl_test      ~1.5 h
mistral_7b_aqlm_rvq   vq       + prepare_rvq                       ~1.5 days

  * measured here: greedyopt is 5354.8 s (uniform) and 95389.6 s (codebook-wise)
    for all 32 layers of Llama-2-7B. The rest are extrapolated from those.
    ppl_test is 90-330 s per sparsity point, dominated by model load.

Budget roughly 1.5 A100-DAYS PER MODEL. The codebook-wise search dominates, at
~18x the uniform cost per layer rather than the ~2x "twice as many thresholds"
suggests: CustomRvqLinear.forward() does not use AQLM's fused low-bit kernel --
it holds each codebook as a dense FP16 buffer and runs one F.linear per
codebook in a Python loop.

Reproducing C2 means paying that: grab_acts -> greedyopt -> ppl_test in full.
--use-ref evaluates a committed search instead, which exercises the plumbing but
does not reproduce its numbers (algorithm/README.md section 3):
  ../../run_algorithm.sh --use-ref -m llama2_7b eval

meta-groups: dense = llama2_7b llama2_13b mistral_7b
             aqlm  = llama2_7b_aqlm llama2_13b_aqlm mistral_7b_aqlm
             rvq   = llama2_7b_aqlm_rvq llama2_13b_aqlm_rvq mistral_7b_aqlm_rvq
             all   = every key above

Cost is not yet measured on this host -- run one group and record it in
greedyopt's cost scales with num_layers x num_projections x
(target_sparsity / base_step_size), each step a single-decoder-layer forward
pass (cheap); grab_acts is one full-model forward pass over --dataset-size
tokens; prepare_rvq downloads + dequantizes an AQLM checkpoint once per _rvq
key.

Groups run sequentially within one tmux session (see header) -- selecting
several groups queues them, it does not parallelize them.
EOF
}

while getopts ":t:s:S:nlh" opt; do
    case "$opt" in
        t) TAG="$OPTARG"; TAG_SET=1 ;;
        s) SPARSITIES="$OPTARG" ;;
        S) STAGES="$OPTARG" ;;
        n) DRYRUN=1 ;;
        l) list_groups; exit 0 ;;
        h) usage; exit 0 ;;
        \?) echo "Unknown option: -$OPTARG" >&2; exit 2 ;;
        :) echo "Option -$OPTARG needs an argument" >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

if [ $# -eq 0 ]; then
    echo "No group given. Run with -l to list groups." >&2
    exit 2
fi

# Stage selection. `want <name>` is how build_cmds asks whether to emit a stage.
DO_PREPARE=0; DO_SEARCH=0; DO_EVAL=0; DO_GATHER=0; DO_BUILD=0
IFS=',' read -ra _stages <<< "$STAGES"
for st in "${_stages[@]}"; do
    case "${st// /}" in
        prepare) DO_PREPARE=1 ;;
        search)  DO_SEARCH=1 ;;
        eval)    DO_EVAL=1 ;;
        gather)  DO_GATHER=1 ;;
        build)   DO_BUILD=1 ;;
        "")      ;;
        *) echo "Unknown stage: $st (expected prepare | search | eval | gather | build)" >&2
           exit 2 ;;
    esac
done
if [ $((DO_PREPARE + DO_SEARCH + DO_EVAL + DO_GATHER + DO_BUILD)) -eq 0 ]; then
    echo "No stage selected by -S '$STAGES'." >&2; exit 2
fi

TARGETS=("$@")
KEYS=()
for t in "${TARGETS[@]}"; do
    case "$t" in
        dense) KEYS+=($DENSE_KEYS) ;;
        aqlm)  KEYS+=($AQLM_KEYS) ;;
        rvq)   KEYS+=($RVQ_KEYS) ;;
        all)   KEYS+=($ALL_KEYS) ;;
        *)
            if [[ " $ALL_KEYS " != *" $t "* ]]; then
                echo "Unknown group/model key: $t (run -l to list)" >&2
                exit 2
            fi
            KEYS+=("$t")
            ;;
    esac
done

TAG_ARG="--run-tag $TAG"
[ -z "$TAG" ] && TAG_ARG=""
# Only warn when a stage would actually WRITE there. eval and gather only
# read, and reading the committed tables is a legitimate thing to do -- warning
# on it trains people to ignore the warning.
if [ -z "$TAG" ] && [ $((DO_PREPARE + DO_SEARCH)) -gt 0 ]; then
    echo "WARNING: -t '' targets the CANONICAL teal_output directory, and" >&2
    echo "         prepare/search WRITE there -- the same directories the" >&2
    echo "         committed data lives in. Pass -t TAG unless deliberate." >&2
fi

RUN_ID="$(date +%Y%m%d-%H%M%S)_$(echo "${KEYS[*]}" | tr ' ' '+')"
[ ${#RUN_ID} -gt 80 ] && RUN_ID="${RUN_ID:0:77}..."
RUN_DIR="$LOGROOT/runs/$RUN_ID"
THRESHOLDS_OUT="$LOGROOT/thresholds_${TAG:-ref}.jsonl"

# GPU headroom on this host is not stable for a session's duration (seen in
# practice: an unrelated job frees the GPU, then reclaims it under a new PID
# mid-session, under a new PID for the same process name). Every model-loading step
# waits here first rather than assuming the caller checked once at the start.
MIN_FREE_MB="${MIN_FREE_MB:-20000}"
wait_for_gpu() {
    local free
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | tail -1)"
    if [ "$free" -lt "$MIN_FREE_MB" ]; then
        echo "waiting for >= ${MIN_FREE_MB} MiB free GPU memory (currently ${free} MiB)..."
        while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | tail -1)" -lt "$MIN_FREE_MB" ]; do
            sleep 30
        done
        echo "GPU headroom available, continuing."
    fi
}

# Build the one sequential script that will run inside tmux.
build_cmds() {
    for k in "${KEYS[@]}"; do
        echo "echo; echo '===== $k ====='"
        # prepare is a no-op for non-_rvq keys: only they have a codebook form.
        if [ "$DO_PREPARE" -eq 1 ] && [[ "$k" == *_rvq ]]; then
            echo "wait_for_gpu; $PYTHON '$ALG/scripts/run_prepare_rvq.py' --models $k"
        fi
        if [ "$DO_SEARCH" -eq 1 ]; then
            echo "wait_for_gpu; $PYTHON '$ALG/scripts/run_grab_acts.py' --models $k $TAG_ARG"
            echo "wait_for_gpu; $PYTHON '$ALG/scripts/run_greedyopt.py' --models $k $TAG_ARG"
        fi
        if [ "$DO_EVAL" -eq 1 ]; then
            echo "wait_for_gpu; $PYTHON '$ALG/scripts/run_ppl_test.py' --models $k $TAG_ARG --sparsities '$SPARSITIES' --out '$RUN_DIR/$k.ppl_test.jsonl'"
        fi
    done
    # One gather for every key at once: the simulator reads a single file
    # holding one record per model, so emitting one per key would be wrong.
    if [ "$DO_GATHER" -eq 1 ]; then
        echo "echo; echo '===== gather thresholds ====='"
        echo "$PYTHON '$ALG/scripts/gather_thresholds.py' --models $(IFS=,; echo "${KEYS[*]}") $TAG_ARG --out '$THRESHOLDS_OUT'"
        echo "echo; echo 'thresholds written to $THRESHOLDS_OUT'"
        echo "echo 'simulate against them with:'"
        echo "echo '  ./run_simulation.sh -j 48 --sparsity-info $THRESHOLDS_OUT'"
    fi
    # Collect every eval record into the CSVs AC/AD/AE read. Reads logs only,
    # so it is not scoped to KEYS and needs no GPU.
    if [ "$DO_BUILD" -eq 1 ]; then
        echo "echo; echo '===== build algorithm CSVs ====='"
        echo "$PYTHON '$ALG/scripts/build_algorithm_csvs.py'${TAG:+ --run-tag $TAG}"
    fi
}

if [ "$DRYRUN" -eq 1 ]; then
    echo "Would run (tag=${TAG:-<canonical>}, stages=$STAGES):"
    build_cmds | sed 's/^/  /'
    exit 0
fi

mkdir -p "$RUN_DIR"
SCRIPT="$RUN_DIR/pipeline.sh"
{
    echo "#!/usr/bin/env bash"
    echo "set -uo pipefail"
    echo "cd '$ALG'"
    echo "MIN_FREE_MB=$MIN_FREE_MB"
    declare -f wait_for_gpu
    build_cmds
    echo "echo; echo '===== done ====='"
} > "$SCRIPT"
chmod +x "$SCRIPT"

SESSION="vortex-algo-$(date +%H%M%S)"
tmux new-session -d -s "$SESSION" \
    "bash '$SCRIPT' 2>&1 | tee '$RUN_DIR/run.log'; echo; echo '(session $SESSION stays open -- exit or Ctrl-B D)'; exec bash"

cat <<EOF
Launched tmux session '$SESSION' running:
  ${KEYS[*]}
tag: ${TAG:-<canonical>}
log: $RUN_DIR/run.log

  tmux attach -t $SESSION      # watch it
  tmux ls                      # list sessions
EOF
