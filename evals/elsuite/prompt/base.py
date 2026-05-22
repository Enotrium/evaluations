"""SolverEval — Base eval class for solver-based evaluations.

Provides standalone solver implementations when Arthedain
is not available, so benchmarks always run.
"""

from __future__ import annotations

import time
import numpy as np

from evals.base import BaseEval, EvalConfig, EvalResult


class SolverEval(BaseEval):
    """Base class for solver-based evaluations.

    Subclasses load a model/solver and benchmark it using run().
    """

    def __init__(self, config: EvalConfig | None = None):
        super().__init__(config)
        self._solver = None

    @property
    def solver(self) -> any:
        """Lazy-loaded solver instance."""
        if self._solver is None:
            self._solver = self._load_solver()
        return self._solver

    def _load_solver(self) -> any:
        """Load the solver/model. Override in subclasses."""
        raise NotImplementedError

    def _make_arthedain_solver(self) -> any | None:
        """Try to return an Arthedain model instance, or None."""
        try:
            from .arthedain_adapter import import_arthedain_model

            mod = import_arthedain_model("elite_pipeline")
            if mod and hasattr(mod, "ElitePipeline"):
                return mod.ElitePipeline()
        except Exception:
            pass
        return None
