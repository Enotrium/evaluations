"""elsuite — Evaluation suite for SNN and HDC systems.

This package contains all benchmark implementations organized
by category (snn, hdc, hardware, energy).
"""

from evals.elsuite.prompt.base import SolverEval
from evals.elsuite.prompt.arthedain_adapter import find_arthedain, get_arthedain_version

__all__ = [
    "SolverEval",
    "find_arthedain",
    "get_arthedain_version",
]
