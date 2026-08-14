# ckpts/

Algorithm-side checkpoints: TEAL search output (`teal_output/`) and converted
AQLM→CustomRvqModel checkpoints (`rvq_models/`, populated on demand). Consumed
by `algorithm/`; see [`../algorithm/README.md`](../algorithm/README.md).

**Nothing here is needed to run C1.** The simulator reads
`stats/ref/sparsity_info/`, which is committed and self-contained.

## `teal_output/`

One subdirectory per model, holding the greedy search's final per-layer
threshold tables:

```
<model>/lookup_v8True/layer-N/results.csv     sparsity -> threshold, per projection
```

Two are committed, 5.2 MB:

| Directory | Registry key | Method |
| --- | --- | --- |
| `Llama-2-7b-AQLM-PV-2Bit-2x8-hf-ckpt20251220/` | `llama2_7b_aqlm` | uniform (TEAL) |
| `Llama-2-7b-AQLM-PV-2Bit-2x8-hf-rvq-cb-ckpt20251220/` | `llama2_7b_aqlm_rvq` | codebook-wise (ours) |

These are the canonical names `algorithm/src/vortex/models.py` resolves to, so a
pipeline call **without** `--run-tag` lands here. That is why every documented
command passes one — see `algorithm/README.md` §2.

They exist so `eval` can be run without paying for `search` first: roughly 1.5 h
for uniform and 26.5 h for codebook-wise on an A100, for this one model. Every
other model/method combination has to be searched.

**Do not rename these directories.** `simulator/sparsity.py` matches on the
directory name as a suffix; renaming breaks the join silently, and ops simply
get no sparsity info.

### What is not committed

- **`histograms/` and `activations/`** — `grab_acts`' intermediate output, several
  GB per model. They have to exist on disk before `greedyopt` or `ppl_test` can
  construct a sparse model, so a fresh run regenerates them; they are just far
  too large for git.
- **Every other model/method combination.** Regenerate with
  `./run_algorithm.sh -m <model> -w <method> search`.

A `lookup/` table is only numerically valid **paired with the histogram it was
searched against** — the stored values are quantiles, converted to magnitudes
through whichever histogram is attached. See `algorithm/README.md` §3.

## `rvq_models/`

Where `run_prepare_rvq` writes the per-codebook form of an AQLM checkpoint.
Empty here and git-ignored: it is always regenerable from a public AQLM
checkpoint, and the unpacked FP16 copies are much larger than the 2-bit
originals.

```bash
./run_algorithm.sh -m llama2_7b -w codebookwise prepare
```
