# stats/initial_submission/ — the data behind the submitted paper

Kept so the camera-ready change can be seen, and for nothing else. **This is not
what a correct run reproduces** — [`../ref/`](../ref/README.md) is.

```bash
cd evaluation/visualization
python plot_all.py --use-submitted     # -> figures/
```

Pairs with `figures/initial_submission/`, the renders that went into the
submitted paper.

## What changed between here and `ref/`

Two energy corrections landed after submission:

1. **The softmax unit was charged static + dynamic power** where only the static
   component is real, and its area was 1000× too small (mm² read as µm²). Vortex
   only.
2. **The baselines were charged a separate 0.05 W of softmax leakage** on top of
   `modules.core.power.static`, which already includes it — double counting. The
   `power` block was removed from `modules.sfu.softmax` in every hw config.

They move five figures by about 1%, in energy only:

| Fig | here | `ref/` (and the paper) |
| --- | --- | --- |
| BA | Vortex vs SA energy 10.32× | **9.75×** |
| BB | max kernel energy 20.2× | **20.0×** |
| BC | 2.04 mm², 10.78 W | **2.10 mm²**, **10.60 W** |
| BD | Vortex energy 8.73× | **8.63×** |
| BE | Vortex energy 20.27× | **20.05×** |

**No speedup anywhere changed** — `total_cycles` is bit-identical across every
variant, verified programmatically. BF is identical in both directories: it
plots latency only, which the corrections do not touch.

The A-series is algorithm-side data no simulation touches, so none of it moved
with the softmax corrections. AC moved later and for an unrelated reason — see
`algorithm/` below.

## `algorithm/`

One file, holding one series: `mistral_7b_aqlm_rvq` as it stood in the
submitted paper. It is the only algorithm-side series whose committed values
differed materially from the corrected-`weight_dict` reproduction — measured
across all nine model tags, everything else is identical or agrees to ≤0.0011,
while this one differs by up to 84.4 on `wikitext_ppl`. Superseded 2026-08-14;
its sparsity ceiling at ≥95% relative accuracy was 30%, and is now 20%.

Archive only. No flag plots it: `--use-submitted` selects the hardware-figure
source, and the algorithm figures come from `ref` unless `--use-algorithm` says
otherwise.

## Provenance

Assembled when the plot scripts were first ported and verified against the
published figures. Two transformations, no numeric value touched:

**Renamed to the current vocabulary**, so the plot scripts need one set of
filters rather than branching on which data source they were handed:

| Column | Old | New |
| --- | --- | --- |
| `method` | `ccarray` | `vortex` |
| `force_dataflow` | `CtL` / `LtC` | `MUF` / `LUF` |
| `config` (kernel results only) | `...CtL...` / `...LtC...` | `...MUF...` / `...LUF...` |

**Shards concatenated and de-duplicated** on the configuration key, newest
timestamp winning — the Vortex sweeps appended without de-duplicating, so
several generations of the same run had accumulated.

## This data is frozen

Nothing regenerates it. It records a state of the paper that has already been
superseded; if it ever disagrees with `figures/initial_submission/`, the figures
are right and this is the copy to fix.
