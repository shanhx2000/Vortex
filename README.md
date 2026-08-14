# Vortex — Artifact Evaluation

Artifact for **Vortex: Bridging Extreme Compression and Efficient LLM
Inference**.

Vortex makes 2-bit residual vector-quantized (RVQ) LLMs fast. It has two parts,
and this artifact reproduces both:

- **C1, the accelerator.** A cycle-level simulator for Vortex and four baseline
  accelerators, producing the speedup, energy, area and power results. CPU
  only.
- **C2, codebook-wise contextual sparsity.** The algorithm that picks a
  sparsity threshold *per AQLM codebook* rather than per projection, and the
  accuracy evaluation of it. Needs a GPU.

They are coupled in one direction only: C2 produces the sparsity thresholds C1
reads, and those thresholds are committed — so **C1 runs without a GPU and
without running C2 at all.**

## 1. Set up

```bash
./setup_env_all.sh              # both conda environments
./setup_env_all.sh --dry-run    # print the commands, create nothing
./setup_env_all.sh sim          # just one
```

Two environments, one per component. Conda is required.

| Env | For | Contents |
| --- | --- | --- |
| `sim` | C1 and all figures | pandas, tqdm, matplotlib, numpy, scikit-learn, Pillow. Python 3.10 |
| `alg` | C2 | torch, flash-attn, transformers, aqlm, lm-eval. Python 3.11 |

`run_simulation.sh` and `visual_results.sh` **find `sim` themselves** — no
`conda activate` needed. C2 needs `conda activate alg`.

C2 additionally needs a HuggingFace token with access to the gated
`meta-llama/*` and `mistralai/*` repos, saved as one line in
`~/vortex/hf_token`. **If `flash-attn` fails to build**, the host's default
`nvcc` is too new for `torch==2.9.1` (cu128); point `CUDA_HOME` at a 12.x
toolkit for that step: `CUDA_HOME=/usr/local/cuda-12.4 ./setup_env_all.sh alg`.

## 2. Run C1 — hardware simulation

```bash
./visual_results.sh                   # draw all eleven figures, ~1 min, no simulation
./run_simulation.sh -l                # groups, what they feed, and measured cost
./run_simulation.sh -j 48             # everything the figures need, ~4.5 h
./visual_results.sh --use-simulation  # redraw from what you just produced
```

**Start with the first line.** `visual_results.sh` reads the committed
reference data by default, so it works on a fresh clone and takes about a
minute. The PDFs it writes to `figures/` are byte-identical to `figures/ref/`,
which is what the paper carries — that is the fastest possible check that the
artifact works before you commit a CPU-day to it.

**Always pass `-j`.** Cost scales with decode length and, for Vortex, with
batch size: a single FIGLUT run at (512, 4096) takes about an hour. Serially
the full set is days; at `-j 48` it is a few hours. `-l` prints measured
per-group costs, so one figure can be reproduced without paying for the rest.

`run_simulation.sh` writes CSVs to `stats/simulation/` and draws nothing;
`visual_results.sh` draws. Results, timings, git commit and output checksums
are recorded per run under `logs/`.

Once the run finishes, `./visual_results.sh --use-simulation` redraws the
hardware figures from your own numbers. Comparing those against `figures/ref/` is the actual reproduction
check.

## 3. Run C2 — codebook-wise sparsity *(optional)*

**Optional, and expensive: budget roughly 1.5 A100-days per model.** The
codebook-wise threshold search alone is ~26.5 h for Llama-2-7B. The
thresholds C1 needs are already committed under `stats/ref/sparsity_info/`, and the
accuracy data behind figures AC/AD/AE is already committed under `stats/ref/algorithm/`,
so **nothing in C1 or in any figure depends on running this.** Run it to
reproduce the algorithm side itself.

```bash
conda activate alg
./run_algorithm.sh -l                                  # models, methods, phases
./run_algorithm.sh -n                                  # dry run: print the commands
./run_algorithm.sh --use-ref -m llama2_7b eval build   # minutes, no search
./run_algorithm.sh                                     # everything: 3 models x 2 methods
./visual_results.sh --use-algorithm AC AD AE           # redraw from your own numbers
```

**Start with the `--use-ref` line.** Two Llama-2-7B searches ship, so it
evaluates them directly and skips the ~1.5 GPU-days a search costs.

Phases, given as positional arguments and always run in this order:

| Phase | Does |
| --- | --- |
| `prepare` | build the per-codebook model from the AQLM checkpoint (codebook-wise only) |
| `search` | `grab_acts` → `greedyopt`: find the sparsity thresholds |
| `eval` | `ppl_test`: WikiText-2 perplexity and six downstream tasks |
| `gather` | thresholds → the JSONL `run_simulation.sh --sparsity-info` reads *(opt-in)* |
| `build` | eval records → `stats/algorithm/`, the CSVs AC/AD/AE read *(opt-in)* |

`prepare`, `search` and `eval` are the default set. `gather` and `build` exist
so a run of your own can be fed back in, and are named explicitly.

Select a subset with `-m` (models: `llama2_7b`, `mistral_7b`, `llama2_13b` —
all 2-bit AQLM), `-w` (methods: `uniform` = TEAL's codebook-uniform
thresholding, `codebookwise` = ours), and `-s` (eval sparsities, default
`0.0,0.3`):

```bash
./run_algorithm.sh -m llama2_7b -w codebookwise search
./run_algorithm.sh -m llama2_7b -s 0.0,0.3 eval
```

Models run one at a time inside tmux — there is one GPU, so asking for several
queues them rather than parallelizing. See
[`algorithm/README.md`](./algorithm/README.md) for a worked single-model
example and what to expect from the search.

## 4. Using your own results instead of ours

Nothing has to be regenerated to reproduce the paper — but if you do generate
something, here is how to feed it back in: one knob for what C1 simulates
against, and one switch per component for what the figures are drawn from.

**Simulate against thresholds you searched yourself.** C1 reads a
contextual-sparsity threshold table, defaulting to
`stats/ref/sparsity_info/` — the table every published number came from. The
`gather` phase turns a search of your own into the same format:

```bash
./run_algorithm.sh -m llama2_7b -t mytag search gather
#   -> logs/algorithm/thresholds_mytag.jsonl
./run_simulation.sh -j 48 --sparsity-info logs/algorithm/thresholds_mytag.jsonl
```

Entries are matched by model identity and vector length, not by the directory
the search wrote into, so your `--run-tag`ged table resolves the same way ours
does. The path in use is printed at the start of every run and recorded in that
run's `logs/runs/<id>/manifest.json`, so a figure can always be traced back to
the thresholds behind it.

**Draw from your own results.** One switch per component; the output
directory does not change:

```bash
./visual_results.sh                    # everything from stats/ref/   [default]
./visual_results.sh --use-simulation   # BA-BF from your own C1 run
./visual_results.sh --use-algorithm    # AC-AE from your own C2 run
./visual_results.sh --use-simulation --use-algorithm    # both
./visual_results.sh --use-submitted    # BA-BF from the submitted paper's data
```

The two switches are independent, because the components are: having run only
C1, pass `--use-simulation` and the algorithm figures still come from `ref`.

## 5. Figures

Eleven figures, each with a stable two-letter index. `./visual_results.sh AC BD`
draws a subset; a figure whose input data is missing is reported and skipped
rather than fatal.

| Fig | Content | Needs |
| --- | --- | --- |
| **AA** | Scalar vs vector quantization of a 2-D Gaussian, at a fixed bit budget | nothing — synthetic |
| **AB** | The cost of VQ: decode latency FP16 vs AQLM, and sparsity vs batch size | nothing — measurements are in the script |
| **AC** | Codebook-wise vs codebook-uniform sparsity, Llama2-7B and Llama2-13B | C2 `eval` *(data committed)* |
| **AD** | Accuracy vs effective bit-width, Llama2-7B, below 2 bits | C2 `eval` *(data committed)* |
| **AE** | Choice of φ-function: ℓ₁ vs ℓ₂ vs ℓ∞ | C2 `eval` *(data committed)* |
| **BA** | Vortex vs the four baselines, end to end: speedup and energy | C1 `baseline_e2e_small` + `end_to_end_eval` |
| **BB** | Per-GEMM kernel speedup and energy, six shapes | C1 `baseline_kernel` + `kernel_eval` |
| **BC** | Where Vortex's area and power go, by module | C1 `power_area` |
| **BD** | Improvement breakdown, attention-intensive (512, 4096) | C1 `baseline_e2e_small` + `impr_ablation` |
| **BE** | Improvement breakdown, projection-intensive (1, 0) | C1 `baseline_e2e_small` + `impr_ablation` |
| **BF** | Forced dataflow vs batch size: why it must be chosen at runtime | C1 `batch_size_sweep` |

AA and AB need no data at all. Every other figure reads committed data by
default, so all eleven draw on a fresh clone — the "Needs" column says which
phase *regenerates* that data, which is what `--use-simulation` (BA–BF) and
`--use-algorithm` (AC–AE) then plot.

Each index maps to one script in `evaluation/visualization/`, named after it
(`BA_baseline_vs_vortex.py` and so on); `plot_all.py --list` prints the mapping
from index to output filename.

## 6. Repository structure

```
setup_env_all.sh      create conda envs `sim` and `alg`
run_simulation.sh     C1: simulate. Writes CSVs, draws nothing.
run_algorithm.sh      C2: prepare / search / eval, per model x method
visual_results.sh     draw figures, all or a named subset

simulator/            cycle-level simulator: Vortex and four baselines.
                      See simulator/README.md.
algorithm/            the vortex package: AQLM -> per-codebook conversion,
                      threshold search, accuracy evaluation.
                      See algorithm/README.md.
evaluation/           everything that produces a figure
  evaluation_scripts/   data generators that import the simulator
  visualization/        one plot script per figure, plus plot_all.py
stats/                all data. ref/ and initial_submission/ mirror the
                      layout a run writes, so one flag switches between them.
  simulation/           generated by C1. Git-ignored.
  algorithm/            generated by C2's `build` phase. Git-ignored.
  ref/                  what the AE submission expects. Read by default.
    simulation/           C1 results, same filenames as above
    algorithm/            C2 results, same filenames as above
    sparsity_info/        sparsity thresholds: C2's output, C1's input
  initial_submission/   the data behind the submitted paper, kept so the
                        camera-ready change can be seen
ckpts/                committed threshold tables for Llama-2-7B
figures/              where every render writes. Git-ignored; ref/ and
                      initial_submission/ are committed.
logs/                 per-run record: command, timing, commit, output md5s
scripts/              setup_env.sh, which setup_env_all.sh wraps
```

**The RTL is not released.** The MXU, VPU, QAU and SPU were written in
Verilog/SystemVerilog and synthesized with Synopsys Design Compiler against an
ARM 28 nm library at 500 MHz; the resulting area and power numbers are what
`simulator/hw_configs/` contains. Nothing becomes unreproducible: every figure
reads those config files, not the RTL, and redoing the synthesis would need
Design Compiler and a licensed library regardless.

## License

MIT — see [LICENSE](./LICENSE). Two parts of `algorithm/` derive from other
MIT-licensed projects, whose notices are preserved in
[THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md):

- **[TEAL](https://github.com/FasterDecoding/TEAL)** (© 2024 FasterDecoding) —
  the original activation-sparsity implementation. Our `algorithm/src/vortex/teal/`
  is a modified port; the codebook-wise variant is built on top of it.
- **[llm-awq](https://github.com/mit-han-lab/llm-awq)** (© 2023 MIT HAN Lab) —
  the WikiText-2 perplexity loop.

Model weights and datasets are not redistributed; they are downloaded from
HuggingFace at run time and keep their own terms.
