# Third-party notices

This repository is released under the MIT License (see [LICENSE](./LICENSE)).
Parts of it derive from the projects below, which are also MIT-licensed. Their
copyright notices are reproduced here and continue to apply to the files they
cover, as MIT requires.

Both upstream licenses were checked against MIT before this repository adopted
it; there is no incompatibility.

---

## TEAL — https://github.com/FasterDecoding/TEAL

**Covers:** `algorithm/src/vortex/teal/` — the activation-sparsity search. This
is a modified port, not a copy: `teal/` follows the original per-projection
`SparsifyFn` formulation, and `teal_rvq/` is our codebook-wise variant built on
top of it, which is the contribution the paper describes. The entry points
(`grab_acts`, `greedyopt`, `ppl_test`) keep the upstream structure.

```
MIT License

Copyright (c) 2024 FasterDecoding

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## llm-awq — https://github.com/mit-han-lab/llm-awq

**Covers:** the WikiText-2 perplexity loop in
`algorithm/src/vortex/core/eval/wikitext_eval.py`, ported from
[`awq/entry.py`](https://github.com/mit-han-lab/llm-awq/blob/main/awq/entry.py)
so our perplexity stays comparable with the quantization literature. See
[`algorithm/README.md`](./algorithm/README.md#evaluation).

```
MIT License

Copyright (c) 2023 MIT HAN Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Models and datasets — not redistributed

No model weights or datasets are shipped here; both are downloaded from
HuggingFace at run time and remain under their own terms.

| | Source | Terms |
| --- | --- | --- |
| AQLM 2-bit checkpoints | `ISTA-DASLab/*` on HuggingFace | the base models' licenses — Llama-2 Community License for the two Llama-2 checkpoints, Apache-2.0 for Mistral-7B-Instruct-v0.2 |
| ARC, COPA, OpenBookQA, PIQA, Winogrande, WikiText-2 | HuggingFace `datasets`, via lm-evaluation-harness | each dataset's own license |

The MIT license above covers this repository's code, not those artifacts.
