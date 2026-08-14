#!/usr/bin/env python3
"""Turn run_ppl_test.py's eval records into the three CSVs figures AC/AD/AE read.

This is C2's counterpart to run_simulation.sh: the pipeline produces per-point
JSONL records under logs/algorithm/, and this collects them into
stats/algorithm/, which `visual_results.sh --use-simulation` plots.

    ./run_algorithm.sh -m llama2_7b eval build     # usual way in
    python build_algorithm_csvs.py --list          # what is on disk
    python build_algorithm_csvs.py --run-tag aerun20260814

Each input line is one (model, sparsity) evaluation:

    {"timestamp": ..., "args": {"model_key", "sparsity", "run_tag", "phi_func",
     "vec_length", ...}, "result": {"wikitext": {"ppl": ...}, "lm_eval": {...}}}

Later records win on collision, so re-running one sparsity point updates it.

Coverage is reported, not enforced. A partial sweep produces partial CSVs and
the plot scripts draw what is there -- a sparsity curve with three points is a
useful check even though the paper's has eleven.

WHAT THIS CANNOT REBUILD FROM YOUR RUN ALONE
--------------------------------------------
AD (accuracy vs bit-width) also plots AWQ, GPTQ and LLM-QAT baselines, which
this artifact does not reproduce -- they are published numbers. Those rows are
carried over from the reference CSV unchanged; only the AQLM and AQLM+RVQ rows
come from your run. The bit-width for a given sparsity is likewise a fixed
function taken from the reference file.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vortex import paths

REF_ALGORITHM_DIR = paths.STATS_DIR / "ref" / "algorithm"
OUT_DIR = paths.STATS_DIR / "algorithm"

# Column names the plot scripts expect, per lm-eval task.
TASK_MAP = {
    "arc_easy": "arc_easy_acc",
    "arc_challenge": "arc_challenge_acc",
    "copa": "copa",
    "openbookqa": "openbookqa",
    "piqa": "piqa",
    "winogrande": "winogrande",
}
FIG_AC = "aqlm_sparsity_exteval_data.csv"
FIG_AD = "llama2_7b_bitwidth_vs_accuracy.csv"
FIG_AE = "phi_function_llama_2_7b_data.csv"


def _acc(metrics):
    """Plain `acc`, matching the reference CSVs.

    They use `acc` for arc_easy/arc_challenge/openbookqa/piqa, not `acc_norm`
    -- verified against the source records. Mixing the two silently compares
    different, harder metrics. `acc_norm` only as a fallback for a task that
    genuinely lacks plain acc.
    """
    return metrics.get("acc,none", metrics.get("acc_norm,none"))


def load_records(run_tag=None):
    """Every eval record under logs/algorithm/, newest-wins per key."""
    by_key = {}
    pattern = str(paths.ALGORITHM_LOGS_DIR / "**" / "*.jsonl")
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue          # a run killed mid-write leaves a partial line
                a = entry.get("args", {})
                if "model_key" not in a or "sparsity" not in a:
                    continue
                if run_tag is not None and a.get("run_tag") != run_tag:
                    continue
                key = (a["model_key"], float(a["sparsity"]),
                       a.get("phi_func", "l1-norm"), a.get("vec_length", 8))
                prev = by_key.get(key)
                if prev is None or entry.get("timestamp", "") >= prev.get("timestamp", ""):
                    by_key[key] = entry
    return by_key


def to_rows(by_key):
    rows = []
    for (model_key, sparsity, phi, vec), entry in sorted(by_key.items()):
        result = entry.get("result", {})
        row = {"model_tag": model_key, "vec_length": vec, "phi_function": phi,
               "sparsity": sparsity}
        wikitext = result.get("wikitext")
        if isinstance(wikitext, dict) and "ppl" in wikitext:
            row["wikitext_ppl"] = wikitext["ppl"]
        for task, col in TASK_MAP.items():
            if task in result.get("lm_eval", {}):
                row[col] = _acc(result["lm_eval"][task])
        rows.append(row)
    return pd.DataFrame(rows)


def build_ac(df, out_dir):
    """AC: geo-mean accuracy vs sparsity, all models, both methods."""
    cols = ["model_tag", "vec_length", "sparsity", "wikitext_ppl",
            "arc_easy_acc", "arc_challenge_acc", "copa", "openbookqa",
            "piqa", "winogrande"]
    sub = df[df.phi_function == "l1-norm"].copy()
    for c in cols:
        if c not in sub.columns:
            sub[c] = None
    sub = sub[cols].sort_values(["model_tag", "sparsity"])
    sub.to_csv(out_dir / FIG_AC, index=False)
    return sub


def build_ae(df, out_dir):
    """AE: the same Llama-2-7B model under three phi functions, long form."""
    sub = df[df.model_tag == "llama2_7b_aqlm_rvq"]
    rows = []
    for _, r in sub.iterrows():
        for task in TASK_MAP.values():
            if pd.notna(r.get(task)):
                rows.append({"model_tag": r.model_tag, "vec_length": r.vec_length,
                             "phi_function": r.phi_function, "sparsity": r.sparsity,
                             "task": task, "value": r[task]})
    out = pd.DataFrame(rows, columns=["model_tag", "vec_length", "phi_function",
                                      "sparsity", "task", "value"])
    out.to_csv(out_dir / FIG_AE, index=False)
    return out


def build_ad(df, out_dir):
    """AD: accuracy vs effective bit-width for Llama-2-7B.

    Only the AQLM / AQLM+RVQ rows come from this run. The published AWQ, GPTQ
    and LLM-QAT baselines, and the sparsity -> bit-width mapping, are carried
    over from the reference CSV -- see the module docstring.
    """
    ref_path = REF_ALGORITHM_DIR / FIG_AD
    if not ref_path.exists():
        print(f"  [skip] {FIG_AD}: needs {ref_path} for the baseline rows")
        return None
    ref = pd.read_csv(ref_path)
    ours = ["llama2_7b_aqlm", "llama2_7b_aqlm_rvq"]

    out_rows = [r.to_dict() for _, r in ref[~ref.model.isin(ours)].iterrows()]
    mine = df[(df.model_tag.isin(ours)) & (df.phi_function == "l1-norm")]
    lookup = {(r.model_tag, round(r.sparsity, 4)): r for _, r in mine.iterrows()}

    for _, r in ref[ref.model.isin(ours)].iterrows():
        row = r.to_dict()
        mine_row = lookup.get((r.model, round(float(r.sparsity), 4)))
        if mine_row is None:
            continue                      # this sparsity point was not evaluated
        for col in list(TASK_MAP.values()) + ["wikitext_ppl"]:
            if col in row and pd.notna(mine_row.get(col)):
                row[col] = mine_row[col]
        out_rows.append(row)

    out = pd.DataFrame(out_rows, columns=ref.columns)
    out.to_csv(out_dir / FIG_AD, index=False)
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-tag", default=None,
                   help="use only records from this --run-tag (default: all)")
    p.add_argument("--out-dir", default=None,
                   help=f"where to write (default: {OUT_DIR})")
    p.add_argument("--list", action="store_true",
                   help="report what is on disk and write nothing")
    args = p.parse_args()

    by_key = load_records(args.run_tag)
    if not by_key:
        where = paths.ALGORITHM_LOGS_DIR
        raise SystemExit(
            f"no eval records under {where}\n"
            f"  Produce some:  ./run_algorithm.sh -m llama2_7b eval\n"
            f"  Or evaluate the shipped searches:  "
            f"./run_algorithm.sh --use-ref -m llama2_7b eval")

    df = to_rows(by_key)
    print(f"{len(by_key)} eval point(s):")
    for (model, phi), g in df.groupby(["model_tag", "phi_function"]):
        pts = ", ".join(f"{s:g}" for s in sorted(g.sparsity.unique()))
        print(f"  {model:24s} phi={phi:11s} {len(g):2d} pts  [{pts}]")
    if args.list:
        return 0

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print()
    for name, built in (("AC", build_ac(df, out_dir)),
                        ("AE", build_ae(df, out_dir)),
                        ("AD", build_ad(df, out_dir))):
        if built is not None:
            print(f"  {name}: {len(built):4d} rows -> "
                  f"{os.path.relpath(out_dir / {'AC': FIG_AC, 'AE': FIG_AE, 'AD': FIG_AD}[name], paths.REPO_ROOT)}")
    print(f"\nPlot them:  ./visual_results.sh --use-simulation AC AD AE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
