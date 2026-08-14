# src/vortex/__init__.py
"""
Vortex algorithm package.

TEAL-based activation-sparsity search (dense and AQLM/RVQ codebook-wise),
the AQLM -> CustomRvqModel conversion, and the model registry / pipeline
functions that produce the checkpoints and sparsity thresholds the Vortex
hardware simulator consumes (see ../../README.md, `paths.py`, `pipelines.py`).

Only the modules
reachable from the grab_acts/greedyopt/ppl_test entry points were kept; see
"""

__version__ = "0.1.0"

from . import core
from . import teal
from . import utils
