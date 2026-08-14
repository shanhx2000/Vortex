# src/vortex/teal/__init__.py
"""
Local modified port of TEAL (https://github.com/FasterDecoding/TEAL):
activation-sparsity search for dense models (`teal`) and its AQLM/RVQ
codebook-wise counterpart (`teal_rvq`).
"""
__version__ = "0.1.0-modified"
from . import teal
from . import teal_rvq
from . import utils
from . import entry
