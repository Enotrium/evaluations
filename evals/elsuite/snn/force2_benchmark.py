"""FORCE2 Benchmark Eval — Benchmark online learning performance with FORCE v2.

Measures convergence time and final error for FORCE-based learning rules.
FORCE is a method for training recurrent SNNs in real time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evals.base import EvalConfig, EvalResult
from evals.elsuite.prompt.base import SolverEval


@dataclass
class FORCE2Config(EvalConfig):
    n_train: int = 500
    n_test: int = 100
    input_dim: int = 10
    hidden_dim: int = 100
    output_dim: int = 5
    n_timesteps: int = 50
    description: str = "FORCE v2 online learning benchmark"


class FORCE2BenchmarkEval(SolverEval):
    """Evaluate FORCE v2 online learning performance."""

    name: str = "snn.force2_benchmark"
    config: FORCE2Config = FORCE2Config()

    def _load_solver(self) -> Any:
        solver = self._make_arthedain_solver()
        if solver is not None:
            return solver
        return _StandaloneFORCE2()

    def run(self) -> EvalResult:
        np.random.seed(self.config.seed)
        n_train = self.config.n_train
        n_test = self.config.n_test

        # Generate synthetic temporal data
        X_train = np.random.randn(n_train, self.config.input_dim).astype(np.float32)
        y_train = np.random.randint(0, self.config.output_dim, size=n_train)
        X_test = np.random.randn(n_test, self.config.input_dim).astype(np.float32)
        y_test = np.random.randint(0, self.config.output_dim, size=n_test)

        # Measure training time
        t0 = time.perf_counter()
        for i in range(n_train):
            self.solver.train_step(X_train[i], y_train[i])
        train_time = time.perf_counter() - t0

        # Evaluate test set
        correct = sum(1 for i in range(n_test)
                      if self.solver.predict(X_test[i]) == y_test[i])
        accuracy = correct / n_test

        return EvalResult(
            name=self.name,
            metrics={
                "accuracy": accuracy,
                "train_time_s": train_time,
                "n_train": n_train,
                "n_test": n_test,
                "input_dim": self.config.input_dim,
                "hidden_dim": self.config.hidden_dim,
                "output_dim": self.config.output_dim,
            },
            metadata={"config": self.config.to_dict()},
        )


class _StandaloneFORCE2:
    """Minimal FORCE v2 baseline."""
    def __init__(self):
        self.w = np.random.randn(10, 5).astype(np.float32) * 0.1
        self.lr = 0.01

    def train_step(self, x: np.ndarray, y: int) -> None:
        pred = x @ self.w
        target_vec = np.zeros(5, dtype=np.float32)
        target_vec[y] = 1.0
        error = target_vec - pred
        self.w += self.lr * np.outer(x, error)

    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(x @ self.w))
