# Plot scripts

One script per paper figure, named by its two-letter index. `plot_all.py
--list` prints the index-to-output mapping.

```bash
python plot_all.py                    # all figures from stats/ref/ -> figures/
python plot_all.py --only BA BD       # a subset
python plot_all.py --use-simulation   # from stats/simulation/ instead
python plot_all.py --use-submitted    # from the submitted paper's data
python verify_ref.py                  # render + build comparison sheets
python BA_baseline_vs_vortex.py       # any script standalone
```

| Script | Fig | Output | Needs |
| --- | --- | --- | --- |
| `AA_motivation_vq.py` | AA | `motivation_vq.pdf` | nothing (synthetic) |
| `AB_motivation_combined.py` | AB | `motivation_combined.pdf` | nothing (inline measurements) |
| `AC_sparsity_combined.py` | AC | `sparsity_combined.pdf` | `stats/ref/algorithm/` |
| `AD_accuracy_vs_bitwidth.py` | AD | `llama2_7b_accuracy_vs_bitwidth.pdf` | `stats/ref/algorithm/` |
| `AE_phi_function.py` | AE | `phi_function_geomean_vs_sparsity.pdf` | `stats/ref/algorithm/` |
| `BA_baseline_vs_vortex.py` | BA | `baseline_vs_vortex_combinedprefill512.pdf` | `baseline_e2e_small`, `end_to_end_eval` |
| `BB_kernel_benchmark.py` | BB | `kernel_benchmark.pdf` | `baseline_kernel`, `kernel_eval` |
| `BC_power_area_pie.py` | BC | `vortex_pie_chart.pdf` | `power_area` |
| `BD_BE_hardware_speedup.py` | BD, BE | `vortex_hardware_speedup_inlen*.pdf` | `impr_ablation`, `baseline_e2e_small` |
| `BF_forceflow_geomean.py` | BF | `forceflow_geomean.pdf` | `batch_size_sweep` |

Simulation groups are run by [`../../run_simulation.sh`](../../run_simulation.sh).

## `--use-ref`

Every script takes it. It swaps the data source from `stats/simulation/` (fresh
simulation output) to `stats/ref/` (the data the published figures were made
from) and writes to `figures/ref/` so a verification run cannot overwrite the
real figures.

Its purpose is to separate two failure modes that look identical: **is the plot
code wrong, or did the data change?** Same code + same data as the paper should
give the same picture. If it does and a fresh run still differs, the difference
is in the numbers.

The A\* figures read committed algorithm-side CSVs that no simulation produces,
so `--use-ref` is a no-op for them.

## Shared modules

| Module | Role |
| --- | --- |
| `_paths.py` | resolves every input; implements `--use-ref` and `--out-dir` |
| `_style.py` | the notebooks' rcParams block, plus `pdf.fonttype = 42` |
| `_common.py` | `load_latest()`, `geo_mean()`, model labels, empty-selection guard |

No script hard-codes a path or a font size.

## Two things that will bite

**Filters must use the new vocabulary.** Freshly generated CSVs say
`method == "vortex"` and `force_dataflow in {MUF, LUF}`. Code still looking for
`"ccarray"` / `"CtL"` / `"LtC"` matches *nothing* and matplotlib renders a blank
chart without complaining. `_common.require_rows()` exists to turn that into an
error, and every B\* script calls it.

**This is a different environment from the simulator.** The simulator runs on
the older Python/pandas in `simulator/requirements.txt`; the figures need
`matplotlib>=3.4` for `Figure.supxlabel` (used by AD and BF). See
`requirements.txt` here.

## Provenance

Each script's docstring names the notebook and cell it came from. Cell indices
were verified by searching for the actual `savefig` target rather than trusting
position — four of ten were off by one in an earlier pass, because several
notebooks hold a draft cell immediately before the real one.
