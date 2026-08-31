#!/usr/bin/env bash
# =============================================================================
# Vortex AE -- C1, hardware simulation. No GPU needed; conda env `sim`.
#
#   ./run_simulation.sh              # every group the paper figures need
#   ./run_simulation.sh -j 48        # the same, 48-way parallel
#   ./run_simulation.sh -l           # list groups, outputs and measured cost
#   ./run_simulation.sh -n           # dry run: print the commands
#   ./run_simulation.sh impr_ablation power_area     # named groups only
#
# This runs simulations and writes CSVs to stats/simulation/. It draws nothing
# -- ./visual_results.sh does that.
#
# Options
#   -j N              run N simulations in parallel (default 1). Always use it.
#   -s NAME           write to stats/simulation/session_NAME/ instead of the
#                     shared directory -- for two shells at once; `merge` after.
#                     Covers every group, including the kernel and power/area
#                     ones, which take --out-dir rather than going through the
#                     sweep driver.
#   -m "..."          override the model list
#   --sparsity-info F contextual-sparsity thresholds to simulate against
#                     (default: the committed table -- see below)
#   -n / -l / -h
#
# Which groups each figure needs
# ------------------------------
#   BA   baseline_e2e_small + end_to_end_eval   speedup/energy vs baselines
#   BB   baseline_kernel + kernel_eval          standalone GEMM kernels
#   BC   power_area                             per-module power/area breakdown
#   BD   baseline_e2e_small + impr_ablation     improvement breakdown, attention
#   BE   baseline_e2e_small + impr_ablation     improvement breakdown, projection
#   BF   batch_size_sweep                       forced dataflow vs batch size
#   AA-AE  none -- those are C2, see run_algorithm.sh
#
# `figures` (the default) is the union of those. `all` adds baseline_e2e_all
# and rebuttal. Running one group reproduces its figures without paying for the
# rest; -l gives the measured cost of each.
#
# Sparsity thresholds
# -------------------
# Vortex's contextual sparsity comes from a threshold table the algorithm side
# produces. **By default this is stats/ref/sparsity_info/, the committed table
# every published number was produced from** -- nothing has to be generated to
# run C1. To simulate against thresholds you searched yourself:
#
#   ./run_simulation.sh --sparsity-info logs/algorithm/thresholds_TAG.jsonl ...
#
# which is what `run_algorithm.sh gather` writes. The path in use is printed at
# the start of every run, and recorded in the run manifest.
#
# Running two shells at once
# --------------------------
# The intended split is one slow baseline session and one Vortex session. Give
# each its own -s name so they cannot touch each other's files, then merge:
#
#   shell 1:  ./run_simulation.sh -s base -j 32 baseline_e2e_small
#   shell 2:  ./run_simulation.sh -s vtx  -j 16 end_to_end_eval impr_ablation
#   after:    ./run_simulation.sh merge
#
# `merge` folds every stats/simulation/session_*/ directory into
# stats/simulation/, de-duplicating on the configuration key and keeping the
# newest row. The plot scripts read the merged files.
#
# Without -s, groups already write disjoint files, so a split by group is safe
# on its own; -s is the belt-and-braces option and the one to use if two
# sessions might run the SAME group.
#
# How parallelism works
# ---------------------
# helper.log_results_to_csv rewrites the whole CSV on every run, so two
# processes writing one file will lose rows. Each job therefore writes its own
# shard under stats/simulation/parts/<bucket>/<tag>.csv, and the canonical CSV
# is rebuilt by concatenating the shards at the end of the group.
#
# Consequences worth knowing:
#   * Baseline shards persist and are resumable -- run_experiments.py skips
#     configurations already present, so an interrupted run picks up where it
#     stopped.
#   * Vortex shards are deleted at the start of their group, because the Vortex
#     paths append without de-duplicating. This makes re-running a group
#     idempotent instead of piling up duplicate rows.
#   * The canonical CSVs are REBUILT from parts/, not appended to. Rows written
#     by anything other than this script are not preserved.
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM="$REPO/simulator"
SCRIPTS="$REPO/evaluation/evaluation_scripts"
OUT="$REPO/stats/simulation"          # canonical, merged results
SESSION=""                            # set by -s; results land in a subdirectory
SESSION_OUT="$OUT"                    # where THIS invocation writes
PARTS="$OUT/parts"
LOGROOT="$REPO/logs"          # run records: manifests, driver logs, per-job logs
LOGDIR=""                     # set per run, once the run id is known

RUNEXP="$SIM/run_experiments.py"

# Interpreter: conda env `sim` when it exists, else whatever `python` is active.
# `conda env list` prints the path last on the line, with or without the `*`
# that marks the active environment.
env_python() {
    command -v conda >/dev/null 2>&1 || return 1
    local p
    p="$(conda env list 2>/dev/null | awk -v n="$1" '$1==n {print $NF}')"
    [ -n "$p" ] && [ -x "$p/bin/python" ] && echo "$p/bin/python"
}
PYTHON="${PYTHON:-$(env_python sim || true)}"
[ -n "$PYTHON" ] || PYTHON="python"

# Contextual-sparsity thresholds the Vortex model reads. Defaults to the
# committed table -- the one every published number was produced from. Point
# --sparsity-info at a different file to simulate against thresholds you
# generated yourself (see `run_algorithm.sh gather`).
SPARSITY_INFO="${VORTEX_SPARSITY_INFO:-$REPO/stats/ref/sparsity_info/teal_sparsities_thresholds_20260402_231905.jsonl}"

MODELS="mistral_7b llama_2_13b llama-2-7b"
BASELINE_METHODS="systolic_array ant figlut figna"

NJOBS=1
DRYRUN=0
CMDLINE="$0 $*"

# -----------------------------------------------------------------------------
# Operating points. Kept in one place so the figure -> point mapping stays
# auditable; -l prints which group each figure needs.
#
# Each list can be overridden from the environment, which is how to get a cheap
# smoke run out of an otherwise expensive group:
#
#   PAIRS_FIG="1,0 512,64" ./run_simulation.sh -j 8 baseline_e2e_small
#
# Overriding changes what the figures are plotted from, so only do it to test
# the plumbing -- or deliberately, to trade coverage for time.
# -----------------------------------------------------------------------------

# BA compares baselines against Vortex at prefill 512 with growing decode.
# BD uses (512,4096), BE uses (1,0). Nothing else is plotted at batch 1.
PAIRS_FIG="${PAIRS_FIG:-1,0 512,512 512,1024 512,2048 512,4096}"

# Full baseline sweep: decode-only points and the mixed points, for parity with
# the Vortex end-to-end sweep. No paper figure reads the extra points.
PAIRS_BASELINE_ALL="${PAIRS_BASELINE_ALL:-1,0 \
0,1 0,2 0,4 0,8 0,16 0,32 0,64 0,128 0,256 0,512 0,1024 \
512,256 512,512 512,1024 512,2048 512,4096}"

# Vortex end-to-end sweep (prefill-only, decode-only and mixed).
PAIRS_VORTEX_E2E="${PAIRS_VORTEX_E2E:-1,0 2,0 4,0 8,0 16,0 32,0 64,0 128,0 256,0 \
0,1 0,2 0,4 0,8 0,16 0,32 0,64 0,128 0,256 0,512 0,1024 \
512,1 512,64 512,128 512,256 512,512 512,1024 512,2048 512,4096}"

# Ablations. (1,0) is the projection-intensive point, (512,4096) the
# attention-intensive one; (512,256) is kept because the source notebook
# switches to it with a one-line edit.
PAIRS_ABLATION="${PAIRS_ABLATION:-1,0 512,256 512,4096}"

BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16 32 64 128 256}"

# -----------------------------------------------------------------------------
# Job queue
# -----------------------------------------------------------------------------
JOB_TAGS=()
JOB_CMDS=()

add_job() {   # add_job <tag> <command…>
    local tag="$1"; shift
    JOB_TAGS+=("$tag")
    JOB_CMDS+=("$*")
}

clear_jobs() { JOB_TAGS=(); JOB_CMDS=(); }

# Slugify a config into a filename-safe tag.
tag_of() {   # tag_of <field> …   -> filename-safe, collision-free job tag
    local s="$*"
    s="${s// /__}"; s="${s//,/_}"; s="${s//|/-}"; s="${s//./p}"
    echo "$s"
}

run_jobs() {   # run_jobs <group>
    local group="$1"
    local n=${#JOB_CMDS[@]}
    TOTAL_JOBS=$((TOTAL_JOBS + n))
    [ "$n" -eq 0 ] && { echo "  (no jobs)"; return 0; }

    if [ "$DRYRUN" -eq 1 ]; then
        printf '  %s\n' "${JOB_CMDS[@]}"
        echo "  -- $n job(s), not executed (-n)"
        return 0
    fi

    mkdir -p "$LOGDIR/$group"
    echo "  $n job(s), $NJOBS at a time"

    local wrapped=()
    local i
    for i in $(seq 0 $((n - 1))); do
        local tag="${JOB_TAGS[$i]}"
        local log="$LOGDIR/$group/$tag.log"
        wrapped+=("{ ${JOB_CMDS[$i]}; } >'$log' 2>&1 \
            && echo '  [ok]   $tag' \
            || { echo '  [FAIL] $tag  -> $log'; exit 1; }")
    done

    local rc=0
    if [ "$NJOBS" -le 1 ]; then
        for w in "${wrapped[@]}"; do
            eval "$w" || rc=1
        done
    else
        # -0 disables xargs quote processing; each item is one whole command.
        printf '%s\0' "${wrapped[@]}" \
            | xargs -0 -P "$NJOBS" -n1 bash -c 'eval "$1"' _ || rc=1
    fi
    [ "$rc" -ne 0 ] && echo "  !! at least one job failed; see $LOGDIR/$group/"
    return "$rc"
}

# -----------------------------------------------------------------------------
# Shards -> canonical CSV
# -----------------------------------------------------------------------------
merge_parts() {   # merge_parts <bucket> <output.csv>
    local bucket="$1" out="$2"
    [ "$DRYRUN" -eq 1 ] && { echo "  (dry run: would merge parts/$bucket -> $(basename "$out"))"; return 0; }
    "$PYTHON" - "$PARTS/$bucket" "$out" <<'PY'
import sys, glob, os
import pandas as pd

part_dir, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(part_dir, "*.csv")))
if not files:
    print(f"  (nothing to merge in {part_dir})")
    sys.exit(0)
frames = []
for f in files:
    try:
        df = pd.read_csv(f)
    except Exception as exc:                      # truncated shard from a kill
        print(f"  [warn] skipping unreadable shard {os.path.basename(f)}: {exc}")
        continue
    if len(df):
        frames.append(df)
if not frames:
    print(f"  (all shards in {part_dir} were empty)")
    sys.exit(0)
merged = pd.concat(frames, ignore_index=True)
merged.to_csv(out, index=False)
print(f"  merged {len(files)} shard(s), {len(merged)} rows -> {os.path.basename(out)}")
PY
}

reset_parts() {   # reset_parts <bucket>   (Vortex groups: no de-duplication on append)
    [ "$DRYRUN" -eq 1 ] && return 0
    rm -rf "${PARTS:?}/$1"
    mkdir -p "$PARTS/$1"
}

# -----------------------------------------------------------------------------
# Job builders
# -----------------------------------------------------------------------------
baseline_jobs() {   # baseline_jobs <pairs> <batch_sizes> [bucket] [env_prefix]
    local pairs="$1" batches="$2"
    local bucket="${3:-baseline}" env_prefix="${4:-}"
    mkdir -p "$PARTS/$bucket"
    local model method pair bs tag
    for model in $MODELS; do
        for method in $BASELINE_METHODS; do
            for pair in $pairs; do
                for bs in $batches; do
                    tag="$(tag_of "$model" "$method" "i${pair}" "bs${bs}")"
                    add_job "$tag" \
                        "${env_prefix}$PYTHON '$RUNEXP' --models '$model' --run_baseline" \
                        "--baseline_methods '$method'" \
                        "--pairs '$pair' --batch_sizes $bs" \
                        "--csv_path '$PARTS/$bucket/$tag.csv'"
                done
            done
        done
    done
}

# The softmax leakage Vortex is charged, in watts, read from the same file the
# Vortex energy model reads so the two cannot drift apart.
sfu_static_watts() {
    "$PYTHON" -c "import json,sys; \
print(json.load(open(sys.argv[1]))['softmax_unit']['static_power'] * 1e-3)" \
        "$SIM/hw_configs/power_energy_config.json"
}

# One job per configuration rather than per model. run_experiments.py loops
# internally over quant/sparsity/dataflow, but a job is the unit of parallelism,
# so leaving those loops inside would cap the speedup at (models x pairs). The
# per-process startup cost (~2 s) is noise next to a long-decode run.
vortex_jobs() {   # vortex_jobs <bucket> <mode> <pairs> <batches> <quants> <sparsities> [flows]
    local bucket="$1" mode="$2" pairs="$3" batches="$4" quants="$5" sparsities="$6"
    local flows="${7:-_}"   # "_" = let the simulator choose the dataflow

    local model pair bs quant sp flow tag flow_arg
    for model in $MODELS; do
        for pair in $pairs; do
            for bs in $batches; do
                for quant in $quants; do
                    for sp in $sparsities; do
                        for flow in $flows; do
                            flow_arg=""
                            [ "$flow" != "_" ] && flow_arg="--vortex_force_dataflow $flow"
                            tag="$(tag_of "$model" "i${pair}" "bs${bs}" "$quant" "s${sp}" "$flow")"
                            # --sparsity-info passed explicitly (not just via the
                            # exported env var) so each job's log records which
                            # threshold table it simulated against.
                            add_job "$tag" \
                                "$PYTHON '$RUNEXP' --models '$model' --$mode" \
                                "--pairs '$pair' --batch_sizes $bs" \
                                "--vortex_quant_schemes '$quant'" \
                                "--vortex_sparsity_list $sp" \
                                "$flow_arg" \
                                "--sparsity-info '$SPARSITY_INFO'" \
                                "--csv_path '$PARTS/$bucket/$tag.csv'"
                        done
                    done
                done
            done
        done
    done
}

# -----------------------------------------------------------------------------
# Groups
# -----------------------------------------------------------------------------
run_group() {
    clear_jobs
    case "$1" in

    # ---- BB: standalone GEMM kernels -------------------------------------
    # One process, 16 kernels; a few seconds. Both halves merge into
    # kernel_results.csv without disturbing each other's rows.
    # Both halves read-modify-write kernel_results.csv, so they are serialized
    # against each other -- they may legitimately be running in two shells at
    # once (one baseline session, one Vortex session). The lock is per -s
    # session, since two sessions write different files.
    baseline_kernel)
        add_job "kernel_baseline" \
            "flock '$SESSION_OUT/.kernel_results.lock' $PYTHON '$SCRIPTS/kernel_evaluation.py' --methods baseline --out-dir '$SESSION_OUT'"
        run_jobs baseline_kernel
        ;;

    kernel_eval)
        add_job "kernel_vortex" \
            "flock '$SESSION_OUT/.kernel_results.lock' $PYTHON '$SCRIPTS/kernel_evaluation.py' --methods vortex --out-dir '$SESSION_OUT'"
        run_jobs kernel_eval
        ;;

    # ---- BA, BD, BE: baseline accelerators end to end --------------------
    baseline_e2e_small)
        baseline_jobs "$PAIRS_FIG" "1"
        run_jobs baseline_e2e_small
        merge_parts baseline "$SESSION_OUT/baseline_evaluation_results.csv"
        ;;

    # ---- experiment: baselines charged Vortex's softmax leakage -----------
    # The baselines hardcode 0.05 W of softmax leakage in their own configs
    # while Vortex is charged 0.00201672 W from power_energy_config.json -- a
    # 25x gap between designs that should share one synthesised number
    # -- a 25x gap between designs that should share one synthesised number.
    #
    # This group re-runs the BA/BD/BE baseline points with that gap closed, into
    # a SEPARATE csv, so the impact can be measured without disturbing the
    # published results. Not part of `figures` or `all` -- ask for it by name.
    baseline_sfu)
        _sfu_w="$(sfu_static_watts)"
        echo "  charging baselines VORTEX_SFU_STATIC_W=$_sfu_w (was 0.05)"
        reset_parts baseline_sfu
        baseline_jobs "$PAIRS_FIG" "1" baseline_sfu "VORTEX_SFU_STATIC_W=$_sfu_w "
        run_jobs baseline_sfu
        merge_parts baseline_sfu "$SESSION_OUT/baseline_sfu_evaluation_results.csv"
        ;;

    # ---- experiment: sfu removed, already inside the baselines' core power --
    # The baseline configs carried modules.sfu.softmax.power.static = 0.05 W on
    # top of modules.core.power.static, which already includes the softmax --
    # double counting. That block was removed from hw_configs/*.json (tag
    # `baseline_sfu_in_core`), so this group re-runs the BA/BD/BE baseline
    # points against the corrected configs, into a separate CSV.
    #
    # No env override: the fix is in the checked-in configs. Which also means
    # baseline_evaluation_results.csv no longer matches them -- see -l.
    baseline_sfu_in_core)
        reset_parts baseline_sfu_in_core
        baseline_jobs "$PAIRS_FIG" "1" baseline_sfu_in_core
        run_jobs baseline_sfu_in_core
        merge_parts baseline_sfu_in_core \
            "$SESSION_OUT/baseline_sfu_in_core_evaluation_results.csv"
        ;;

    # Superset of _small: adds decode-only points and the batch sweep. Shares
    # the parts/baseline bucket, so anything _small already produced is skipped.
    baseline_e2e_all)
        baseline_jobs "$PAIRS_BASELINE_ALL" "1"
        baseline_jobs "1,0 512,4096" "2 4 8 16 32 64 128 256"
        run_jobs baseline_e2e_all
        merge_parts baseline "$SESSION_OUT/baseline_evaluation_results.csv"
        ;;

    # ---- BA: Vortex end to end, automatic dataflow -----------------------
    end_to_end_eval)
        reset_parts vortex_e2e
        vortex_jobs vortex_e2e run_vortex "$PAIRS_VORTEX_E2E" "1" \
            "AQLM AQLM|CQ" "0.0 0.3"
        run_jobs end_to_end_eval
        merge_parts vortex_e2e "$SESSION_OUT/vortex_evaluation_results.csv"
        ;;

    # ---- BD, BE: improvement breakdown -----------------------------------
    # Bars are Wgt-Q (AQLM), Wgt-Q|Att-Q (AQLM|CQ), Vortex (AQLM|CQ + 30%
    # sparsity) and LUF (AQLM forced to LUF), against systolic_array.
    impr_ablation)
        reset_parts vortex_ablation
        vortex_jobs vortex_ablation run_vortex_flow "$PAIRS_ABLATION" "1" \
            "AQLM AQLM|CQ" "0.0 0.3" "None LUF"
        run_jobs impr_ablation
        merge_parts vortex_ablation "$SESSION_OUT/vortex_projection_attention_intensive_results.csv"
        ;;

    # ---- BF: dataflow choice vs batch size -------------------------------
    batch_size_sweep)
        reset_parts vortex_forceflow
        vortex_jobs vortex_forceflow run_vortex_flow "$PAIRS_ABLATION" "$BATCH_SIZES" \
            "AQLM" "0.0" "None MUF LUF"
        run_jobs batch_size_sweep
        merge_parts vortex_forceflow "$SESSION_OUT/vortex_forceflow_evaluation_results.csv"
        ;;

    # ---- BC: power and area breakdown ------------------------------------
    # Two stages inside one script: profile the three models at (512,4096),
    # then aggregate into vortex_power_area_breakdown.json.
    power_area)
        add_job "power_area" "$PYTHON '$SCRIPTS/get_raw_stats.py' --profile --out-dir '$SESSION_OUT'"
        run_jobs power_area
        ;;

    # ---- Not a paper figure ----------------------------------------------
    rebuttal)
        reset_parts vortex_rebuttal
        local saved="$MODELS"
        MODELS="llama-2-7b"
        vortex_jobs vortex_rebuttal run_vortex_flow "512,4096" "1 2 4 8 16" \
            "AQLM AQLM|CQ" "0.0 0.3" "None LUF"
        MODELS="$saved"
        run_jobs rebuttal
        merge_parts vortex_rebuttal "$SESSION_OUT/rebuttal_batch_evaluation_results.csv"
        ;;

    # ---- fold session directories into the canonical results ------------
    merge)
        merge_sessions
        ;;

    *)
        echo "Unknown group: $1" >&2
        echo "Run with -l to list groups." >&2
        return 2
        ;;
    esac
}

merge_sessions() {
    [ "$DRYRUN" -eq 1 ] && { echo "  (dry run: would merge session_*/ into $OUT)"; return 0; }
    "$PYTHON" - "$OUT" <<'PY'
import glob
import os
import shutil
import sys

import pandas as pd

out = sys.argv[1]
session_dirs = sorted(glob.glob(os.path.join(out, "session_*")))
if not session_dirs:
    print(f"  no session_* directories in {out}; nothing to merge")
    sys.exit(0)
print(f"  sessions: {', '.join(os.path.basename(d) for d in session_dirs)}")

# The columns that identify one simulated configuration; must match
# helper.log_results_to_csv. Newest timestamp wins on collision.
KEY = ["model_name", "input_length", "output_length", "batch_size", "method",
       "quant_scheme", "processed_sparsity", "force_dataflow"]
# kernel_results.csv has its own schema and no timestamp -- without this it
# would concatenate rather than de-duplicate, and merging twice would double it.
KERNEL_KEY = ["kernel", "method", "config"]

by_name = {}
for d in session_dirs:
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        by_name.setdefault(os.path.basename(f), []).append(f)

# JSON results (the power/area breakdown, the raw per-op stats) are whole-file
# outputs with nothing to merge row-wise: the newest session simply wins.
for d in session_dirs:
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        target = os.path.join(out, os.path.basename(f))
        if not os.path.exists(target) or os.path.getmtime(f) > os.path.getmtime(target):
            shutil.copy2(f, target)
            print(f"  {os.path.basename(f):<52} copied from {os.path.basename(d)}")

for name, files in sorted(by_name.items()):
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as exc:
            print(f"  [warn] skipping {f}: {exc}")
            continue
        if len(df):
            frames.append(df)
    if not frames:
        continue
    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)

    target = os.path.join(out, name)
    # Fold in whatever is already in the canonical file so merging is additive
    # rather than replacing an earlier non-session run.
    if os.path.exists(target):
        try:
            existing = pd.read_csv(target)
            if len(existing):
                merged = pd.concat([existing, merged], ignore_index=True)
        except Exception as exc:
            print(f"  [warn] ignoring unreadable {target}: {exc}")

    if "timestamp" in merged.columns:
        key = [c for c in KEY if c in merged.columns]
        if key:
            merged["timestamp"] = pd.to_datetime(merged["timestamp"])
            merged = merged.sort_values("timestamp").drop_duplicates(subset=key,
                                                                     keep="last")
    elif all(c in merged.columns for c in KERNEL_KEY):
        merged = merged.drop_duplicates(subset=KERNEL_KEY, keep="last")
    merged.to_csv(target, index=False)
    print(f"  {name:<52} {len(files)} session(s), "
          f"{before} rows -> {len(merged)} after de-duplication")
PY
}

FIGURE_GROUPS="baseline_kernel kernel_eval baseline_e2e_small \
end_to_end_eval impr_ablation batch_size_sweep power_area"
ALL_GROUPS="$FIGURE_GROUPS baseline_e2e_all rebuttal"   # deliberately excludes `merge`

# Canonical artifacts each group is responsible for, recorded in the run
# manifest so a figure can be traced back to the run that produced its data.
group_outputs() {
    case "$1" in
        merge)                           echo "" ;;
        baseline_kernel|kernel_eval)     echo "kernel_results.csv" ;;
        baseline_e2e_small|baseline_e2e_all) echo "baseline_evaluation_results.csv" ;;
        baseline_sfu)                    echo "baseline_sfu_evaluation_results.csv" ;;
        baseline_sfu_in_core)            echo "baseline_sfu_in_core_evaluation_results.csv" ;;
        end_to_end_eval)                 echo "vortex_evaluation_results.csv" ;;
        impr_ablation)                   echo "vortex_projection_attention_intensive_results.csv" ;;
        batch_size_sweep)                echo "vortex_forceflow_evaluation_results.csv" ;;
        power_area)                      echo "vortex_power_area_breakdown.json" ;;
        rebuttal)                        echo "rebuttal_batch_evaluation_results.csv" ;;
    esac
}

list_groups() {
    cat <<'EOF'
group                figures    jobs  output                                             cost (serial / -j 48)
-------------------  ---------  ----  -------------------------------------------------  ---------------------
baseline_kernel      BB            1  kernel_results.csv (baseline rows)                  3 s *
kernel_eval          BB            1  kernel_results.csv (Vortex rows)                    3 s *
baseline_e2e_small   BA BD BE     60  baseline_evaluation_results.csv                    ~16 h  / 112 m *
baseline_e2e_all     -           396  baseline_evaluation_results.csv (superset)         ~5 d   / ~3 h
baseline_sfu         BA_sfu       60  baseline_sfu_evaluation_results.csv                ~16 h  / ~2 h
baseline_sfu_in_core BA_sfucore   60  baseline_sfu_in_core_evaluation_results.csv        ~16 h  / ~2 h
end_to_end_eval      BA          336  vortex_evaluation_results.csv                      ~1 h   / ~5 m
impr_ablation        BD BE        72  vortex_projection_attention_intensive_results.csv  ~55 m  / 5.2 m *
batch_size_sweep     BF          243  vortex_forceflow_evaluation_results.csv            ~28 h  / ~2.5 h *
power_area           BC            1  vortex_power_area_breakdown.json                    58 s *
rebuttal             -            20  rebuttal_batch_evaluation_results.csv              ~1 h   / ~10 m
merge                -             -  folds session_*/ into stats/simulation/             seconds

  * measured on this host. The rest are extrapolated -- treat as lower bounds.

meta-groups: figures = every group a paper figure needs
             all     = figures + baseline_e2e_all + rebuttal

Two baseline variants exist, neither in a meta-group -- ask for them by name.
Both write their own CSV and leave baseline_evaluation_results.csv alone.

  baseline_sfu           softmax leakage set to Vortex's 0.00202 W, via the
                         VORTEX_SFU_STATIC_W env override. Historical: it
                         predates the finding below, and now re-adds a separate
                         softmax charge on top of core.
  baseline_sfu_in_core   softmax power removed from the baseline configs
                         entirely, because modules.core.power.static already
                         includes it. This is the corrected model.

  python BA_baseline_vs_vortex.py --variant {sfu,sfu_in_core}

The committed stats/ref/simulation/ data was produced by the CURRENT configs,
so re-running baseline_e2e_small reproduces it rather than moving figure BA.

TWO THINGS DRIVE COST, NOT ONE.

1. Decode length. The simulator walks one op list per generated token, so
   (512,4096) steps through ~41k ops while (512,0) steps through ~10. Prefill is
   nearly free.

2. Batch size -- but only for Vortex. Measured wall time for one Vortex
   (512,4096) run:

       batch    1     8    32    64   128   256
       time   0.5m  1.9m  6.2m 10.8m 22.3m  129m

   Roughly linear above batch 8, ~250x from batch 1 to 256. The baselines are
   *flat*: systolic_array and ant sit at 55-57 min for every batch size.

   This inverts the usual picture. At batch 1 a baseline run costs ~25x a Vortex
   run, so the baselines dominate. In the batch sweep the opposite holds, and
   `batch_size_sweep` is the second most expensive group in the artifact despite
   touching only one accelerator.

At -j 48 wall time is bounded by the single longest job, which cannot be split:
~1 h for a figlut (512,4096) run, ~2.1 h for a Vortex (512,4096) run at batch
256. Start the long groups first and in the background.

Shrink a group for a smoke test by overriding its point list:
  PAIRS_FIG="1,0 512,32" ./run_simulation.sh -j 8 baseline_e2e_small

Two shells at once: give each -s NAME, then run `merge`:
  ./run_simulation.sh -s base -j 32 baseline_e2e_small   # shell 1
  ./run_simulation.sh -s vtx  -j 16 end_to_end_eval      # shell 2
  ./run_simulation.sh merge                              # after both finish
EOF
}

# Print the header comment block (everything from line 2 up to the first
# non-comment line), stripped of its leading "# ".
usage() {
    awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print } else { exit } }' "${BASH_SOURCE[0]}"
}

# getopts handles short options only, so lift the one long option out first.
ARGV=()
while [ $# -gt 0 ]; do
    case "$1" in
        --sparsity-info)
            [ $# -ge 2 ] || { echo "--sparsity-info needs a path" >&2; exit 2; }
            SPARSITY_INFO="$2"; shift 2 ;;
        --sparsity-info=*)
            SPARSITY_INFO="${1#*=}"; shift ;;
        *)  ARGV+=("$1"); shift ;;
    esac
done
set -- "${ARGV[@]+"${ARGV[@]}"}"

while getopts ":j:m:s:nlh" opt; do
    case "$opt" in
        j) NJOBS="$OPTARG" ;;
        m) MODELS="$OPTARG" ;;
        s) SESSION="$OPTARG" ;;
        n) DRYRUN=1 ;;
        l) list_groups; exit 0 ;;
        h) usage; exit 0 ;;
        \?) echo "Unknown option: -$OPTARG" >&2; exit 2 ;;
        :) echo "Option -$OPTARG needs an argument" >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

# The simulator resolves this through VORTEX_SPARSITY_INFO (simulator/sparsity.py).
if [ ! -f "$SPARSITY_INFO" ]; then
    echo "sparsity info file not found: $SPARSITY_INFO" >&2
    exit 2
fi
# Absolute, so jobs are independent of the working directory and so the
# committed-default check below compares paths rather than spellings.
SPARSITY_INFO="$(cd "$(dirname "$SPARSITY_INFO")" && pwd)/$(basename "$SPARSITY_INFO")"
export VORTEX_SPARSITY_INFO="$SPARSITY_INFO"

if [ -n "$SESSION" ]; then
    case "$SESSION" in
        */*|"") echo "Session name must not contain '/'" >&2; exit 2 ;;
    esac
    SESSION_OUT="$OUT/session_$SESSION"
    # Shards live under the session too, so two sessions running the same group
    # cannot collide -- at the cost of resuming per session rather than globally.
    PARTS="$SESSION_OUT/parts"
fi

# NB: do not name this GROUPS -- that is a read-only bash builtin array.
TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then
    echo "No group given. Running the paper-figure set; -l lists the options."
    TARGETS=($FIGURE_GROUPS)
fi

EXPANDED=()
for t in "${TARGETS[@]}"; do
    case "$t" in
        figures) EXPANDED+=($FIGURE_GROUPS) ;;
        all)     EXPANDED+=($ALL_GROUPS) ;;
        *)       EXPANDED+=("$t") ;;
    esac
done

mkdir -p "$OUT" "$SESSION_OUT" "$PARTS"

# -----------------------------------------------------------------------------
# Run record. Every invocation gets an id and a directory under logs/:
#
#   logs/index.md              one line per run, newest last
#   logs/runs/<id>/manifest.json   command, timing, git state, output checksums
#   logs/runs/<id>/run.log         driver output
#   logs/runs/<id>/jobs/*.log      per-job stdout (git-ignored, large)
#
# The point is version maintenance: given a figure, the manifest says which run
# produced its input, when, from which commit, and what the file hashed to.
# -----------------------------------------------------------------------------
SLUG="$(echo "${EXPANDED[*]}" | tr ' ' '+' )"
[ ${#SLUG} -gt 48 ] && SLUG="${SLUG:0:45}..."
RUN_ID="$(date +%Y%m%d-%H%M%S)_${SLUG}"
RUN_DIR="$LOGROOT/runs/$RUN_ID"
LOGDIR="$RUN_DIR/jobs"

if [ "$DRYRUN" -eq 0 ]; then
    mkdir -p "$LOGDIR"
    # Mirror everything the driver prints into run.log while keeping it on screen.
    exec > >(tee -a "$RUN_DIR/run.log") 2>&1
fi

echo "python:        $PYTHON"
if [ "$SPARSITY_INFO" = "$REPO/stats/ref/sparsity_info/teal_sparsities_thresholds_20260402_231905.jsonl" ]; then
    echo "sparsity info: ${SPARSITY_INFO#"$REPO/"}  (committed default)"
else
    echo "sparsity info: $SPARSITY_INFO  (OVERRIDE -- not the published table)"
fi

STARTED_AT="$(date -Is)"
START=$(date +%s)

write_manifest() {   # write_manifest <status>
    [ "$DRYRUN" -eq 1 ] && return 0
    RUN_ID="$RUN_ID" RUN_DIR="$RUN_DIR" REPO="$REPO" OUT="$OUT" \
    STATUS="$1" STARTED_AT="$STARTED_AT" DURATION="$(( $(date +%s) - START ))" \
    CMDLINE="$CMDLINE" GROUPS_RUN="${EXPANDED[*]}" FAILED_RUN="${FAILED[*]-}" \
    MODELS="$MODELS" NJOBS="$NJOBS" NRUNJOBS="$TOTAL_JOBS" \
    SPARSITY_INFO="$SPARSITY_INFO" \
    OUTPUTS="$(for g in "${EXPANDED[@]}"; do group_outputs "$g"; done | sort -u | tr '\n' ' ')" \
    "$PYTHON" - <<'PY'
import hashlib, json, os, platform, subprocess, sys
from datetime import datetime
from pathlib import Path

env = os.environ
repo, out = Path(env["REPO"]), Path(env["OUT"])

def _rel(p):
    """Repo-relative when possible: a manifest must not embed a home directory."""
    if not p:
        return p
    try:
        return str(Path(p).resolve().relative_to(repo))
    except ValueError:
        return str(p)


def git(*a):
    try:
        return subprocess.check_output(["git", "-C", str(repo), *a],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

outputs = []
for name in env["OUTPUTS"].split():
    f = out / name
    if not f.exists():
        outputs.append({"file": name, "present": False})
        continue
    h = hashlib.md5(f.read_bytes()).hexdigest()
    rows = None
    if f.suffix == ".csv":
        with f.open() as fh:
            rows = max(sum(1 for _ in fh) - 1, 0)
    outputs.append({
        "file": name,
        "present": True,
        "path": str(f.relative_to(repo)),
        "rows": rows,
        "bytes": f.stat().st_size,
        "md5": h,
        "mtime": datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds"),
    })

manifest = {
    "run_id": env["RUN_ID"],
    "status": env["STATUS"],
    "started": env["STARTED_AT"],
    "finished": datetime.now().isoformat(timespec="seconds"),
    "duration_s": int(env["DURATION"]),
    "command": env["CMDLINE"],
    "groups": env["GROUPS_RUN"].split(),
    "failed_groups": env["FAILED_RUN"].split(),
    "models": env["MODELS"].split(),
    "sparsity_info": _rel(env.get("SPARSITY_INFO")),
    "parallelism": int(env["NJOBS"]),
    "jobs": int(env["NRUNJOBS"]),
    "outputs": outputs,
    "environment": {
        # Hostname digest, not the hostname: runs from one machine group
        # together without the record naming it. See logs/README.md.
        "host": hashlib.blake2s(platform.node().encode(), digest_size=4).hexdigest(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
    },
}
Path(env["RUN_DIR"], "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

# Append one row to the shared index. Two sessions may finish at once, so the
# caller holds a lock around this script.
index = repo / "logs" / "index.md"
if not index.exists():
    index.write_text(
        "# Simulation run index\n\n"
        "Newest last. One row per invocation of `run_simulation.sh`.\n"
        "Full detail, including output checksums, is in `runs/<id>/manifest.json`.\n\n"
        "| Run id | Started | Duration | Jobs | -j | Status | Groups | Outputs |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n")
mins = manifest["duration_s"] / 60
dur = f"{manifest['duration_s']}s" if mins < 2 else f"{mins:.0f}m"
files = ", ".join(o["file"] for o in outputs if o["present"]) or "-"
status = manifest["status"]
if manifest["failed_groups"]:
    status += " (" + ",".join(manifest["failed_groups"]) + ")"
with index.open("a") as fh:
    fh.write("| `{}` | {} | {} | {} | {} | {} | {} | {} |\n".format(
        manifest["run_id"], manifest["started"].replace("T", " ")[:19], dur,
        manifest["jobs"], manifest["parallelism"], status,
        " ".join(manifest["groups"]), files))
print(f"  run record: logs/runs/{manifest['run_id']}/manifest.json")
PY
}

FAILED=()
TOTAL_JOBS=0
for g in "${EXPANDED[@]}"; do
    echo
    echo "===== $g ====="
    if ! run_group "$g"; then
        FAILED+=("$g")
    fi
done

echo
echo "===== done in $(( $(date +%s) - START ))s; results in ${SESSION_OUT#"$REPO/"} ====="

if [ "$DRYRUN" -eq 0 ]; then
    STATUS="ok"
    [ ${#FAILED[@]} -ne 0 ] && STATUS="failed"
    # Serialize the index append: a baseline session and a Vortex session can
    # finish at the same moment.
    mkdir -p "$LOGROOT"
    ( flock 9; write_manifest "$STATUS" ) 9>"$LOGROOT/.index.lock"
fi

if [ ${#FAILED[@]} -ne 0 ]; then
    echo "groups with failures: ${FAILED[*]}" >&2
    sleep 0.2   # let the tee subprocess drain before the shell exits
    exit 1
fi
sleep 0.2
