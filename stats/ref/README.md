# stats/ref/ — the data the AE submission expects

Everything a figure needs, committed, so a fresh clone reproduces the paper
without running anything:

```bash
./visual_results.sh          # reads this directory by default
```

The eleven PDFs that come out are byte-identical to `figures/ref/`, which is
what the paper carries.

```
stats/ref/
  simulation/       C1 results, same filenames run_simulation.sh writes
  algorithm/        C2 results, same filenames `run_algorithm.sh build` writes
  sparsity_info/    contextual-sparsity thresholds: C2's output, C1's input
```

`simulation/` and `algorithm/` deliberately mirror `stats/simulation/` and
`stats/algorithm/`, so the same relative path names the same file in both, and
switching between them is one flag:

| Flag | Reads | Is |
| --- | --- | --- |
| *(none)*, or `--use-ref` | `stats/ref/` | what the AE submission expects |
| `--use-simulation` | `stats/` | what you just generated |
| `--use-submitted` | `stats/initial_submission/` | what the *submitted* paper used |

Output always goes to `figures/`, whichever source is read. `--use-submitted`
covers the B-series only: the submitted paper's algorithm data was never
separated out, so AA-AE fall back to `ref` under it.

## What is in `simulation/`

Exactly what the figure-feeding groups produce — no more, no less, so a diff
against your own run is meaningful rather than ragged.

| File | Rows | Produced by | Feeds |
| --- | ---: | --- | --- |
| `baseline_evaluation_results.csv` | 60 | `baseline_e2e_small` | BA BD BE |
| `vortex_evaluation_results.csv` | 336 | `end_to_end_eval` | BA |
| `vortex_projection_attention_intensive_results.csv` | 72 | `impr_ablation` | BD BE |
| `vortex_forceflow_evaluation_results.csv` | 243 | `batch_size_sweep` | BF |
| `kernel_results.csv` | 160 | `baseline_kernel` + `kernel_eval` | BB |
| `vortex_power_area_breakdown.json` | — | `power_area` | BC |

Row counts are the product of the operating points each group sweeps; `-l`
prints the job count that produces them.

## What is in `algorithm/`

The accuracy data figures AC, AD and AE plot, written by
`./run_algorithm.sh ... build` from the per-point records `eval` leaves in
`logs/algorithm/`.

| File | Feeds | Notes |
| --- | --- | --- |
| `aqlm_sparsity_exteval_data.csv` | AC | three models x two methods; AC plots two of them, see its docstring |
| `phi_function_llama_2_7b_data.csv` | AE | Llama-2-7B under l1 / l2 / l_inf |
| `llama2_7b_bitwidth_vs_accuracy.csv` | AD | also carries the published AWQ, GPTQ and LLM-QAT rows, which this artifact does not reproduce |

A run of your own produces the same three filenames under `stats/algorithm/`,
covering whatever sparsity points you evaluated. Partial coverage is fine -- the
plot scripts draw what is there.

## `sparsity_info/`

`teal_sparsities_thresholds_20260402_231905.jsonl` — the per-layer,
per-projection, per-codebook thresholds `simulator/sparsity.py` reads. It is the
one artifact that crosses the algorithm/simulator boundary, and it is a
reference *input* rather than a result, which is why it sits here rather than in
`simulation/`.

Every published Vortex number was simulated against this file. To simulate
against thresholds you searched yourself:

```bash
./run_algorithm.sh -m llama2_7b -t mytag search gather   # writes a new table
./run_simulation.sh -j 48 --sparsity-info logs/algorithm/thresholds_mytag.jsonl
```

Entries are matched by model identity and vector length, not by the directory
the search wrote into, so a `--run-tag`ged table of your own resolves the same
way this one does.

## Regenerating this directory

Only if a simulator change is meant to move the published numbers — that is a
deliberate re-baseline, not routine:

```bash
./run_simulation.sh -j 48                                    # ~4.5 h
cp stats/simulation/{baseline,vortex,kernel}*.{csv,json} stats/ref/simulation/
cd evaluation/visualization && python plot_all.py --out-dir ../../figures/ref
```

Then check `verify_ref.py` still reports every figure as matching, and say in
the commit message why the baseline moved.
