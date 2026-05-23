"""HDC Classification Eval — Benchmark hyperdimensional classification.

Evaluates HDC encoding and associative memory classification accuracy
using synthetic data. Falls back to standalone implementation when
Arthedain HDC module is unavailable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from evals.base import EvalConfig, EvalResult
from evals.elsuite.prompt.base import SolverEval
from evals.data import make_synthetic_dataset, train_test_split


@dataclass
class HDCClassificationConfig(EvalConfig):
    n_samples: int = 1000
    n_features: int = 128
    n_classes: int = 8
    dim: int = 4096
    description: str = "HDC hypervector encoding + associative memory classification"


class HDCClassificationEval(SolverEval):
    """Evaluate HDC classification performance."""

    name: str = "hdc.classification"
    config: HDCClassificationConfig = HDCClassificationConfig()

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
        t0 = time.perf_counter()
        for xi, yi in zip(X_train, y_train):
            self.solver.train(xi, yi)
        train_time = time.perf_counter() - t0

        # Test
        correct = 0
        t0 = time.perf_counter()
        for xi, yi in zip(X_test, y_test):
            pred = self.solver.predict(xi)
            if pred == yi:
                correct += 1
        infer_time = time.perf_counter() - t0

        accuracy = correct / len(X_test)
        infer_us = (infer_time / len(X_test)) * 1e6

        return EvalResult(
            name=self.name,
            metrics={
                "accuracy": accuracy,
                "n_classes": n_classes,
                "dim": self.config.dim,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "infer_us_per_sample": infer_us,
                "train_time_s": train_time,
            },
            metadata={"config": self.config.to_dict()},
        )


class _StandaloneHDCClassifier:
    """Standalone HDC classifier — hypervector encoding + associative memory."""

    def __init__(self, dim: int = 4096, n_classes: int = 8):
        self.dim = dim
        self.n_classes = n_classes
        self.assoc_memory: Optional[np.ndarray] = None
        self.rng = np.random.RandomState(42)
        self._proj = None

    def _encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input into a hyperdimensional vector using random projection + binarization."""
        if self._proj is None:
            self._proj = self.rng.randn(len(x), self.dim).astype(np.float32)
        hv = x @ self._proj
        return np.where(hv > 0, 1.0, -1.0).astype(np.float32)

    def train(self, x: np.ndarray, y: int) -> None:
        hv = self._encode(x)
        if self.assoc_memory is None:
            self.assoc_memory = np.zeros((self.n_classes, self.dim), dtype=np.float32)
        self.assoc_memory[y] += hv

    def predict(self, x: np.ndarray) -> int:
        hv = self._encode(x)
        if self.assoc_memory is None:
            return 0
        sims = self.assoc_memory @ hv
        return int(np.argmax(sims))
