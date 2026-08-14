# algorithm/ — codebook-wise contextual sparsity (C2)

Picks an activation-sparsity threshold **per AQLM codebook** rather than per
projection, and evaluates what that costs in accuracy. This is the paper's
algorithmic contribution; its output is the threshold tables the simulator
reads.

Running this is **optional for artifact evaluation** — the thresholds and the
accuracy data behind figures AC/AD/AE are committed. Budget roughly **1.5
A100-days per model** if you do run it, dominated by the codebook-wise search.

## 1. Requirements

Its own environment: torch, flash-attn and AQLM do not belong next to the
simulator's pandas/matplotlib.

```bash
cd ..
./setup_env_all.sh alg
conda activate alg
```

- **Python 3.11.** `transformers==4.44.1` is a hard pin, not a lower bound —
  see [`requirements.txt`](./requirements.txt). Later versions change the
  attention-module API this port hooks into.
- **One CUDA GPU**, holding one model at a time: ~15 GB for a 7B in fp16, ~26 GB
  for 13B, plus the AQLM-dequantized copy for the `_aqlm`/`_rvq` entries. Check
  `nvidia-smi` first on a shared box.
- **A HuggingFace token** with read access to the gated `meta-llama/*` and
  `mistralai/*` repos, as a single line in `~/vortex/hf_token`.
  `src/vortex/hf_auth.py` reads it; every entry point calls `ensure_hf_login()`
  before touching the Hub.
- **If `flash-attn` fails to build**, the host's default `nvcc` is too new for
  `torch==2.9.1` (cu128):
  `CUDA_HOME=/usr/local/cuda-12.4 ./setup_env_all.sh alg`. This is the step
  most likely to cost you time.

## 2. One model, both methods

Worked example: 2-bit AQLM **Llama-2-7B**, under both thresholding methods.
`uniform` is TEAL's — one threshold per projection. `codebookwise` is ours —
one per AQLM codebook. The comparison between them is figure AC.

From the repo root, the short way:

```bash
./run_algorithm.sh -m llama2_7b -w uniform            # all three phases
./run_algorithm.sh -m llama2_7b -w codebookwise
```

The same thing stage by stage, which is what those commands generate. Model
keys here are the registry's: `uniform` → `llama2_7b_aqlm`, `codebookwise` →
`llama2_7b_aqlm_rvq`.

```bash
cd scripts

# --- codebookwise only: build the per-codebook model -------------------------
# Unpacks each AQLM codebook into its own FP16 weight tensor, so a threshold
# can be searched per codebook. `uniform` skips this.
python run_prepare_rvq.py --models llama2_7b_aqlm_rvq

# --- search: histograms, then the greedy threshold search --------------------
python run_grab_acts.py --models llama2_7b_aqlm     --run-tag TAG
python run_greedyopt.py --models llama2_7b_aqlm     --run-tag TAG
python run_grab_acts.py --models llama2_7b_aqlm_rvq --run-tag TAG
python run_greedyopt.py --models llama2_7b_aqlm_rvq --run-tag TAG

# --- eval: perplexity + accuracy at each sparsity level ----------------------
python run_ppl_test.py --models llama2_7b_aqlm     --run-tag TAG --sparsities 0.0,0.3
python run_ppl_test.py --models llama2_7b_aqlm_rvq --run-tag TAG --sparsities 0.0,0.3

# --- finally: thresholds in the shape the simulator reads --------------------
python gather_thresholds.py --models llama2_7b_aqlm,llama2_7b_aqlm_rvq \
    --run-tag TAG --out ../../logs/algorithm/thresholds_TAG.jsonl

# --- and: the eval records in the shape the figures read ---------------------
python build_algorithm_csvs.py --run-tag TAG     # -> ../../stats/algorithm/
```

Those last two are the `gather` and `build` phases. `gather` feeds C1
(`run_simulation.sh --sparsity-info`); `build` feeds the figures
(`visual_results.sh --use-simulation AC AD AE`). Neither runs by default.

**Always pass `--run-tag`.** Without it the run writes into the *canonical*
`ckpts/teal_output/<model>/`, which is where the committed search results live
and where the shipped threshold file was partly built from. A tag sends output
somewhere new and diffable. `run_algorithm.sh` generates one automatically
(`aerun<date>`), and `--use-ref` is the explicit, read-only way to target the
committed directories.

`gather_thresholds.py` never writes the committed reference filename. Replacing
that file means re-running every simulation and every figure.

Expect the codebook-wise search to dominate: measured **95 390 s (~26.5 h)** for
Llama-2-7B against **5 355 s (~1.5 h)** for uniform — about 18× per layer, not
the ~2× a "twice as many thresholds" estimate suggests. The cause is in
`CustomRvqLinear.forward()`: it does not use AQLM's fused low-bit kernel, but
holds each codebook as a dense FP16 buffer and runs one `F.linear` per codebook
in a Python loop.

## 3. What to expect from the search

**The search contains no randomness.** There is no seeding anywhere because
there is nothing to seed: the calibration set is the first `dataset_size`
documents of WikiText-2 train, taken deterministically
(`get_dataset(...)` → `.skip(0).take(size)`, no shuffle), and `greedyopt` is a
deterministic greedy argmin over a fixed step grid. On one machine, with one
checkpoint, a repeat run reproduces its own table.

**It is nonetheless not reproducible across environments**, for two reasons
worth knowing before you compare a fresh table against the committed one:

*The greedy search amplifies floating-point noise.* Each step picks the
projection whose activation error grows least, comparing GPU-computed floats
that are near-ties. Kernel selection and reduction order vary with GPU model,
CUDA version and batch shape, so a near-tie can flip — and because each step
changes the state the rest of the search proceeds from, one flip cascades
through every step after it.

*Thresholds are quantiles, not magnitudes.* `SparsifyFn.set_threshold(x)` does
not use `x` directly; it calls `self.distr.abs_icdf(x)`, converting a quantile
into a magnitude against whichever histogram is attached. So a `lookup/` table
is only numerically valid **paired with the exact histogram it was searched
against**. Pairing a committed table with freshly regenerated histograms builds
a plausible sparse model at approximately the intended sparsity, but does not
reproduce the perplexity the table's own search measured.

**Scale to expect.** In our own camera-ready pass, re-running the Llama-2-7B
search moved the maximum sparsity sustaining 95% relative accuracy by one grid
step — 35% to 30%. Treat a one-step difference as normal; treat the ordering as
the result. Codebook-wise beat codebook-uniform by 10 sparsity points on
Llama-2-7B both before and after that shift, and that ordering is what the
paper claims.

## 4. The accuracy evaluation pipeline

`run_ppl_test.py` evaluates one model at each sparsity in `--sparsities`. For
each level it constructs the sparse model from the searched thresholds, then
runs `--eval-tasks` through a single entry point, `lm_eval_m` in
`src/vortex/core/eval/`, which routes two kinds of metric differently:

| Task | Path |
| --- | --- |
| `arc_easy`, `arc_challenge`, `copa`, `openbookqa`, `piqa`, `winogrande` | lm-evaluation-harness (`HFLM` + `simple_evaluate`) |
| `wikitext` | `wikitext_eval.py`, our own loop — **not** lm-eval's `wikitext` task |

Accuracy is reported as the plain `acc` for every task, and the figures plot the
geometric mean across the six.

**WikiText-2 perplexity deliberately does not go through lm-eval.** It follows
AWQ's evaluation style, ported from
[`awq/entry.py`](https://github.com/mit-han-lab/llm-awq/blob/main/awq/entry.py):
the test split is concatenated with `\n\n`, tokenized once, cut into
non-overlapping windows of `model_seqlen` (2048), and the perplexity is the
exponential of the mean token-level negative log-likelihood over those windows.
lm-eval's own `wikitext` task instead reports word-level and byte-level
perplexity and bits-per-byte over individual documents — a different quantity,
not comparable with what the quantization literature publishes. Using AWQ's
loop keeps our PPL directly comparable with AWQ, AQLM and the other 2-bit
results the paper cites; the accuracy tasks, where lm-eval is the community
standard, stay on lm-eval.

Runs standardize on `phi_func="l1-norm"`, `vec_length=8` — group activations
into chunks of 8 along the hidden dimension, threshold on each chunk's sum of
absolute values. This matches what the committed lookup tables were searched
with (`lookup_v8True/`). Other φ functions and vector lengths work end to end
(every CLI takes `--phi-func`/`--vec-length`) and produce a differently
suffixed lookup directory; figure AE is the comparison between them.

## 5. Structure

```
scripts/                  thin CLIs over src/vortex/pipelines.py
  run_prepare_rvq.py        phase `prepare`
  run_grab_acts.py          phase `search`, stage 1: activation histograms
  run_greedyopt.py          phase `search`, stage 2: greedy threshold search
  run_ppl_test.py           phase `eval`
  gather_thresholds.py      phase `gather`: thresholds -> the JSONL the
                              simulator's --sparsity-info reads
  build_algorithm_csvs.py   phase `build`: eval records -> stats/algorithm/,
                              the CSVs figures AC/AD/AE read
  run_teal_group.sh         tmux driver; ../run_algorithm.sh wraps this

src/vortex/
  models.py               MODEL_REGISTRY: the 9 model x quantization combinations
  pipelines.py            run_prepare_rvq / run_grab_acts / run_greedyopt /
                            run_ppl_test / gather_sparsity_thresholds
  paths.py, hf_auth.py    path resolution; HuggingFace login
  core/
    model/custom_rvq.py     AQLM -> CustomRvqLinear/CustomRvqModel: unpacks each
                              codebook into its own FP16 tensor
    prepare/prepare_rvq.py  drives that conversion and saves the checkpoint
    eval/                   lm_eval_m(), wikitext_eval.py
  teal/
    teal/                   TEAL: one SparsifyFn per q/k/v/o/gate/up/down
    teal_rvq/               ours: one SparsifyFn per codebook per projection
    entry/                  grab_acts / greedyopt / ppl_test
    utils/utils_activations.py  SparsifyFn / Distribution -- the thresholding math

```

Output lands in `../ckpts/teal_output/<model>[-TAG]/`: `histograms/` and
`activations/` from `grab_acts` (multi-GB, never committed), `lookup_v8True/`
from `greedyopt` (per-layer `results.csv`, committed for two models — see
[`../ckpts/README.md`](../ckpts/README.md)).

## Attribution

The activation-sparsity implementation originates in
**[TEAL](https://github.com/FasterDecoding/TEAL)** (© 2024 FasterDecoding, MIT).
`src/vortex/teal/teal/` is a modified port of it — the per-projection
`SparsifyFn` formulation and the `grab_acts` / `greedyopt` / `ppl_test` entry
points all follow the original. **Our contribution is
`src/vortex/teal/teal_rvq/`**, the codebook-wise variant built on top.

```bibtex
@article{liu2024training_teal,
  title={Training-free activation sparsity in large language models},
  author={Liu, James and Ponnusamy, Pragaash and Cai, Tianle and Guo, Han and Kim, Yoon and Athiwaratkun, Ben},
  journal={arXiv preprint arXiv:2408.14690},
  year={2024}
}
```

The WikiText-2 perplexity loop in `src/vortex/core/eval/` comes from
**[llm-awq](https://github.com/mit-han-lab/llm-awq)** (© 2023 MIT HAN Lab, MIT).
Both notices are reproduced in
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
