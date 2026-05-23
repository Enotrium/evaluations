"""HDC Robustness Eval — Measure HDC fault tolerance under hardware errors.

Evaluates how HDC accuracy degrades under bit-flip and stuck-at-fault
conditions in the associative memory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from evals.base import EvalConfig, EvalResult
from evals.elsuite.prompt.base import SolverEval
from evals.data import make_synthetic_dataset, train_test_split


@dataclass
class HDCRobustnessConfig(EvalConfig):
    n_samples: int = 500
    n_features: int = 64
    n_classes: int = 4
    dim: int = 4096
    description: str = (
        "HDC fault tolerance under bit-flip and stuck-at-fault conditions"
    )


class HDCRobustnessEval(SolverEval):
    """Evaluate HDC robustness under hardware faults."""

    name: str = "hdc.robustness"
    config: HDCRobustnessConfig = HDCRobustnessConfig()

    def _load_solver(self) -> Any:
        solver = self._make_arthedain_solver()
        if solver is not None:
            return solver
        return _StandaloneHDCClassifier(
            dim=self.config.dim, n_classes=self.config.n_classes
        )

    def run(self) -> EvalResult:
        np.random.seed(self.config.seed)
        n = self.config.n_samples
        feats = self.config.n_features
        n_classes = self.config.n_classes

        X, y = make_synthetic_dataset(n, feats, n_classes, seed=self.config.seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, seed=self.config.seed)

        # Train
        for xi, yi in zip(X_train, y_train):
            self.solver.train(xi, yi)

        # Clean accuracy
        correct_clean = sum(
            1 for xi, yi in zip(X_test, y_test) if self.solver.predict(xi) == yi
        )
        clean_acc = correct_clean / len(X_test)

        metrics = {"accuracy_clean": clean_acc}

        # Fault injection tests
        for fault_type, rates in [("flip", [0.1, 0.2]), ("stuck", [0.1, 0.2])]:
            for rate in rates:
                key = f"accuracy_{int(rate*100)}pct_{fault_type}"
                correct_faulty = 0
                for xi, yi in zip(X_test, y_test):
                    self.solver.inject_fault(fault_type, rate=rate)
                    pred = self.solver.predict(xi)
                    if pred == yi:
                        correct_faulty += 1
                    self.solver.clear_fault()
                metrics[key] = correct_faulty / len(X_test)
                metrics[f"accuracy_drop_{int(rate*100)}pct_{fault_type}"] = (
                    clean_acc - metrics[key]
                )

        # Overall fault tolerance score (average relative accuracy)
        fault_accs = [
            v
            for k, v in metrics.items()
            if k.startswith("accuracy_") and k != "accuracy_clean"
        ]
        metrics["fault_tolerance_score"] = float(
            np.mean(fault_accs) / clean_acc if clean_acc > 0 else 0.0
        )

        return EvalResult(
            name=self.name, metrics=metrics, metadata={"config": self.config.to_dict()}
        )


class _StandaloneHDCClassifier:
    """Standalone HDC classifier with fault injection support."""

    def __init__(self, dim: int = 4096, n_classes: int = 4):
        self.dim = dim
        self.n_classes = n_classes
        self.assoc_memory: np.ndarray = np.zeros((n_classes, dim), dtype=np.float32)
        self.rng = np.random.RandomState(42)
        self._proj = None
        self._faulty_memory = None

    def _encode(self, x: np.ndarray) -> np.ndarray:
        if self._proj is None:
            self._proj = self.rng.randn(len(x), self.dim).astype(np.float32)
        hv = x @ self._proj
        return np.where(hv > 0, 1.0, -1.0).astype(np.float32)

    def train(self, x: np.ndarray, y: int) -> None:
        hv = self._encode(x)
        self.assoc_memory[y] += hv

    def predict(self, x: np.ndarray) -> int:
        hv = self._encode(x)
        mem = (
            self._faulty_memory
            if self._faulty_memory is not None
            else self.assoc_memory
        )
        sims = mem @ hv
        return int(np.argmax(sims))

    def inject_fault(self, fault_type: str, rate: float = 0.1) -> None:
        self._faulty_memory = self.assoc_memory.copy()
        mask = self.rng.random(self._faulty_memory.shape) < rate
        if fault_type == "flip":
            self._faulty_memory[mask] *= -1
        elif fault_type == "stuck":
            self._faulty_memory[mask] = 0.0

    def clear_fault(self) -> None:
        self._faulty_memory = None
