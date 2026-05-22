"""SolverCompletionFn — wraps a solver/pipeline for SNN/HDC model evaluation."""

from __future__ import annotations

from typing import Any, Optional

from .base import CompletionFn, CompletionResult


class SolverCompletionFn(CompletionFn):
    """Completion fn for solver/pipeline-based models (SNN, HDC)."""

    def __init__(self, solver: Any):
        self.solver = solver

    def complete(self, prompt: str, **kwargs) -> CompletionResult:
        """Run the solver on input data."""
        try:
            result = self.solver(prompt)
            return CompletionResult(
                text=str(result),
                raw=result,
            )
        except Exception as e:
            return CompletionResult(
                text="",
                error=str(e),
            )
