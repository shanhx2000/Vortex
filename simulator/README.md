# simulator/ — cycle-level hardware simulation (C1)

Models the Vortex accelerator and the baselines it is compared against,
producing the cycle counts, energy, area and power behind figures BA–BF.

**CPU only — no GPU, and no need to run the algorithm side.** The sparsity
thresholds the Vortex model reads are committed under `../stats/ref/sparsity_info/`.

## 1. Requirements

```bash
cd ..
./setup_env_all.sh sim
```

Conda env `sim`: pandas and tqdm, on Python 3.10
([`requirements.txt`](./requirements.txt)). That is the whole dependency list —
everything else the simulator imports is standard library. The env also carries
matplotlib and friends so the same environment can draw the figures.

`../run_simulation.sh` and `../visual_results.sh` find `sim` on their own, so no
`conda activate` is needed for C1.

## 2. What is being compared

Five accelerators, all simulated by the same op-level driver so the comparison
is like for like. Every figure normalizes to the systolic array.

| Method | Precision | What it is |
| --- | --- | --- |
| `systolic_array` | W8A8 | Conventional systolic array. The normalization reference: improves efficiency by reusing data on chip to cut memory traffic. |
| `ant` | W8A8 | **ANT** — adaptive numeric data type plus the architecture to support it, for low-bit quantization. |
| `figna` | W4A16 | **FIGNA** — converts FP activations to integers while preserving accuracy, so the GEMM runs on integer units. |
| `figlut` | W4A16 | **FIGLUT** — LUT-based FP×INT multiplication built on binary-coded quantization. The strongest baseline; the paper's headline speedups are quoted against it. |
| `vortex` | W2A16 | **Ours.** 2-bit RVQ weights, with codebook-wise contextual sparsity and a dataflow chosen per workload (MUF or LUF). |

Each has a hardware preset under [`hw_configs/`](./hw_configs/) — array
dimensions, SRAM banking, frequency, and the per-module area/power that
Synopsys Design Compiler produced for the Vortex RTL against an ARM 28 nm
library at 500 MHz. Those JSON files, not the RTL, are what every figure reads,
which is why not releasing the RTL costs no reproducibility.

## 3. Running

**Use the driver, not this directory directly.** `run_simulation.sh` groups
runs by the figure they feed, parallelizes them, and records provenance:

```bash
cd ..
./run_simulation.sh -l          # groups, outputs, measured cost
./run_simulation.sh -j 48       # everything the figures need
./run_simulation.sh power_area  # one group
./visual_results.sh             # draw
```

**Always pass `-j`.** Cost scales with decode length and, for Vortex, with
batch size — a single FIGLUT run at (512, 4096) is about an hour, and a Vortex
run at the same point is half a minute at batch 1 but over two hours at batch
256. Serially the full set is days; at `-j 48` it is a few hours, floored by the
longest single job.

For one configuration at a time — debugging, or a point no group covers:

```bash
python main.py --model-name llama-2-7b --methods vortex \
    --input-length 512 --output-length 4096 --batch-size 1
```

One method per invocation. `--methods` accepts any of the five above;
`--model-name` accepts `llama-2-7b`, `llama_2_13b` and `mistral_7b` (plus the
other entries in `llm_configs.py`'s registry). `--force-dataflow MUF|LUF|None`
pins Vortex's dataflow instead of letting it select, which is what figure BF
sweeps. `run_experiments.py` sweeps a grid of these to CSV; it is what the
driver calls into.

Results land in `../stats/simulation/`, and each run's command, timing, git
commit and output checksums under `../logs/`.

## 4. Which figures need which groups

| Fig | Content | Groups |
| --- | --- | --- |
| **BA** | Vortex vs all four baselines end to end: speedup and energy reduction, three models × growing decode length | `baseline_e2e_small` + `end_to_end_eval` |
| **BB** | Per-GEMM kernel speedup and energy, six shapes (M ∈ {1,4,16}) | `baseline_kernel` + `kernel_eval` |
| **BC** | Where Vortex's area and power go, by module | `power_area` |
| **BD** | Improvement breakdown at (512, 4096), attention-intensive | `baseline_e2e_small` + `impr_ablation` |
| **BE** | Improvement breakdown at (1, 0), projection-intensive | `baseline_e2e_small` + `impr_ablation` |
| **BF** | Forced dataflow vs batch size — why it must be chosen at runtime | `batch_size_sweep` |

`./run_simulation.sh -l` prints the inverse — group → figures, outputs and
measured cost. Figures AA–AE come from the algorithm side; see
[`../algorithm/README.md`](../algorithm/README.md).

Every figure can also be drawn without simulating anything:
`../visual_results.sh` reads the committed data the published
figures were made from.

## 5. Structure

```
main.py             single-configuration entry point (one method per run)
run_experiments.py  sweeps a grid of configurations to CSV; the driver calls this
model_runner.py     builds a workload from a model config, runs it on one simulator
networks.py         expands a model layer description into a flat list of operations
operations.py       operation types and tensor descriptors (GEMM, GEMV, ...)
llm_configs.py      per-model layer shapes, head counts, op sequences

hw_base.py          base simulator: op dispatch, progress accounting, baseline energy
hw_vortex.py        Vortex simulator = HWSimBase + the Evaluator energy model
vortex.py           Vortex's cycle and memory-traffic model
evaluator.py        area and energy model, from CACTI/synthesis-derived unit costs
energy.py           per-op energy for the baseline kernels
stats.py            the accumulator every hardware model writes into

baselines/          systolic_array.py, ant.py, figna.py, figlut.py, common.py
hw_configs/         one JSON preset per accelerator, plus power_energy_config.json
quantization.py     rewrites tensor descriptors for a quantization scheme
sparsity.py         attaches measured contextual-sparsity ratios to ops
hw_configs.py       loads the JSON presets
helper.py           ceil division, dtype widths, config merging, CSV logging
ltc_merged.csv      CACTI SRAM area/energy table the evaluator interpolates
```

Only the modules reachable from `run_experiments.py` are here.
