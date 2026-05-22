"""evaluations — open-source registry of benchmarks for SNN and HDC systems.

Usage:
    oaieval list
    oaieval run hdc.classification
    oaieval run-all
    oaieval compare results/
    oaieval status
"""

from .api import EvalRegistry, register, list_evals
from .base import BaseEval, EvalConfig, EvalResult
from .formatting import format_result
from .metrics import MetricSuite, Compare
from .registry import Registry

__all__ = [
    "EvalRegistry",
    "register",
    "list_evals",
    "BaseEval",
    "EvalConfig",
    "EvalResult",
    "format_result",
    "MetricSuite",
    "Compare",
    "Registry",
]
