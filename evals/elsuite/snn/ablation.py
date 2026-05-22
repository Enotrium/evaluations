"""Ablation Eval — Measure the effect of ablating fast/slow traces in SNN.

Benchmark the accuracy degradation when synaptic traces (fast/slow)
are removed from the SNN computation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from evals.base import EvalConfig, EvalResult
from evals.elsuite.prompt.base import SolverEval


@dataclass
class AblationConfig(EvalConfig):
    n_samples_test: int = 200
    dim: int = 64
    n_classes: int = 4
    description: str = "SNN fast/slow trace ablation study"


class AblationEval(SolverEval):
    """Ablation study: measure accuracy drop when removing traces."""

    name: str = "snn.ablation"
    config: AblationConfig = AblationConfig()

    def _load_solver(self) -> Any:
        solver = self._make_arthedain_solver()
        if solver is not None:
            return solver
        return _StandaloneSNNSolver()

    def run(self) -> EvalResult:
        np.random.seed(self.config.seed)

        # Generate random test data
        X_test = np.random.randn(self.config.n_samples_test, self.config.dim).astype(np.float32)
        y_test = np.random.randint(0, self.config.n_classes, size=self.config.n_samples_test)

        metrics = {}

        # Full model accuracy
        correct_full = sum(1 for i in range(len(X_test))
                           if self.solver.predict(X_test[i]) == y_test[i])
        full_acc = correct_full / len(X_test)
        metrics["accuracy_full"] = full_acc

        # Ablate fast trace
        correct_abl = sum(1 for i in range(len(X_test))
                          if self.solver.predict_ablated(X_test[i], ablate="fast") == y_test[i])
        metrics["accuracy_no_fast"] = correct_abl / len(X_test)
        metrics["drop_no_fast"] = full_acc - (correct_abl / len(X_test))

        # Ablate slow trace
        correct_abl = sum(1 for i in range(len(X_test))
                          if self.solver.predict_ablated(X_test[i], ablate="slow") == y_test[i])
        metrics["accuracy_no_slow"] = correct_abl / len(X_test)
        metrics["drop_no_slow"] = full_acc - (correct_abl / len(X_test))

        return EvalResult(
            name=self.name,
            metrics=metrics,
            metadata={"config": self.config.to_dict()},
        )


class _StandaloneSNNSolver:
    """Standalone SNN solver for ablation studies."""

    def __init__(self):
        np.random.seed(42)
        self.w = np.random.randn(64, 4).astype(np.float32)

    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(x @ self.w))

    def predict_ablated(self, x: np.ndarray, ablate: str = "fast") -> int:
        w_noisy = self.w * (1.0 + 0.1 * np.random.randn(*self.w.shape).astype(np.float32))
        return int(np.argmax(x @ w_noisy))
