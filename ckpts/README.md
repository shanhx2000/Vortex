# ckpts/

Algorithm-side checkpoints: TEAL search output (`teal_output/`) and converted
AQLM→CustomRvqModel checkpoints (`rvq_models/`, populated on demand). Consumed
by `algorithm/`; see [`../algorithm/README.md`](../algorithm/README.md).

**Nothing here is needed to run C1.** The simulator reads
`stats/ref/sparsity_info/`, which is committed and self-contained.

## `teal_output/`

One directory per model, holding the greedy search's final per-layer threshold
tables:

```
<model key>/lookup_v8True/layer-N/results.csv   sparsity -> threshold, per projection
```

All ten registry entries are committed, 29 MB:

| Directory | Method | Layers |
| --- | --- | ---: |
| `llama2_7b/` | dense baseline | 32 |
| `llama2_7b_aqlm/` | uniform (TEAL) | 32 |
| `llama2_7b_aqlm_rvq/` | codebook-wise (ours) | 32 |
| `llama2_7b_aqlm_rvq_uniform/` | one threshold across codebooks — the φ ablation's comparison point | 32 |
| `llama2_13b/` | dense baseline | 40 |
| `llama2_13b_aqlm/` | uniform (TEAL) | 40 |
| `llama2_13b_aqlm_rvq/` | codebook-wise (ours) | 40 |
| `mistral_7b/` | dense baseline | 32 |
| `mistral_7b_aqlm/` | uniform (TEAL) | 32 |
| `mistral_7b_aqlm_rvq/` | codebook-wise (ours) | 32 |

**The directory is named after the registry key**, so it is spelled exactly the
way you spell it on the command line: the table `-m llama2_13b -w codebookwise`
reads is `llama2_13b_aqlm_rvq/`. `algorithm/src/vortex/models.py` is the
mapping, and these are the *canonical* names — a pipeline call **without**
`--run-tag` writes here. That is why every documented command passes one; see
`algorithm/README.md` §2. `--use-ref` is the explicit read-only way in.

Renaming a directory is safe as far as the simulator is concerned:
`simulator/sparsity.py` joins on model identity (mode, model name, vector
length), not on this path. Rename in `models.py` and here together, and the
`teal_path` recorded in `stats/ref/sparsity_info/` becomes stale as
documentation but breaks nothing.

### What these are for

They are the **reference**: the thresholds every published Vortex number was
produced from, and what
`stats/ref/sparsity_info/teal_sparsities_thresholds_20260402_231905.jsonl` was
gathered from. Compare a search of your own against them.

They are **not a way to skip the search.** A `lookup/` table stores quantiles,
which `SparsifyFn.set_threshold` converts to magnitudes through whichever
histogram is attached, so a table is only numerically exact **paired with the
histograms it was searched against** — and those are several GB per model and
not committed (see below). Pairing a committed table with freshly regenerated
histograms builds a sparse model at approximately the intended sparsity; it does
not reproduce the perplexity the table's own search measured.
`algorithm/README.md` §3 has the full argument.

So reproducing C2 is the full pipeline — `grab_acts` → `greedyopt` → `ppl_test`,
roughly 1.5 A100-days per model:

```bash
./run_algorithm.sh -m llama2_13b
```

`--use-ref` runs `ppl_test` against the committed table instead. Useful to check
the plumbing end to end in minutes; not a substitute for the search.

### What is not committed

- **`histograms/` and `activations/`** — `grab_acts`' intermediate output,
  several GB per model. They must exist on disk before `greedyopt` or `ppl_test`
  can construct a sparse model, so a fresh run regenerates them; they are simply
  far too large for git.
- **The φ variants.** The φ-function ablation (figure AE) compares ℓ₁ / ℓ₂ / ℓ∞
  searches, which live in sibling `lookup_v8True_phil*/` directories. Only the
  ℓ₁ table each entry defaults to is committed. Note that `--phi-func` currently
  selects the metric for a *search* but does not change which lookup directory
  `eval` reads — `pipelines.py` derives that suffix from `vec_length`/`use_abs`
  alone. Regenerating AE means running the sweep, not passing a flag.

## `rvq_models/`

Where `run_prepare_rvq` writes the per-codebook form of an AQLM checkpoint.
Empty here and git-ignored: it is always regenerable from a public AQLM
checkpoint, and the unpacked FP16 copies are much larger than the 2-bit
originals.

```bash
./run_algorithm.sh -m llama2_7b -w codebookwise prepare
```

## Provenance

`llama2_7b_aqlm/` and `llama2_7b_aqlm_rvq/` are the searches the published
Llama-2-7B numbers came from. The other eight were added when this repository
was split out, from the same searches that produced
`stats/ref/sparsity_info/teal_sparsities_thresholds_20260402_231905.jsonl` —
that file's `teal_path` field names them, and every path in it resolves here.

One caveat worth stating: `mistral_7b_aqlm_rvq/` is the `20260402` search, whose
provenance we have not fully reconstructed. It is what the published Mistral-7B
thresholds were derived from, which is why it is the copy that ships, but a
re-run may not land on it exactly — see `algorithm/README.md` §3 on why searches
do not reproduce across environments.
