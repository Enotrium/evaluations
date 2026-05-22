"""BCI Decoding Eval — Benchmark SNN performance on motor-imagery decoding.

Measures decoding accuracy and inference latency for BCI applications.
Uses synthetic data when Arthedain is not available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evals.base import EvalConfig, EvalResult
from evals.elsuite.prompt.base import SolverEval


@dataclass
class BCIDecodingConfig(EvalConfig):
    n_samples: int = 1000
    n_channels: int = 128
    n_classes: int = 4
    n_timepoints: int = 200
    description: str = "BCI motor-imagery velocity decoding benchmark"


class BCIDecodingEval(SolverEval):
    """Evaluate SNN decoding of motor-imagery signals."""

    name: str = "snn.bci_decoding"
    config: BCIDecodingConfig = BCIDecodingConfig()

    def _load_solver(self) -> Any:
        """Try Arthedain, fall back to standalone."""
        solver = self._make_arthedain_solver()
        if solver is not None:
            return solver
        return _StandaloneBCIDecoder()

    def run(self) -> EvalResult:
        np.random.seed(self.config.seed)
        n = self.config.n_samples
        ch = self.config.n_channels
        tp = self.config.n_timepoints

        # Simulate BCI signals (band-limited noise)
        X = np.random.randn(n, ch, tp).astype(np.float32)
        y = np.random.randint(0, self.config.n_classes, size=n)

        # Measure inference accuracy and timing
        correct = 0
        t0 = time.perf_counter()
        for i in range(n):
            pred = self.solver.predict(X[i])
            if pred == y[i]:
                correct += 1
        elapsed = time.perf_counter() - t0

        accuracy = correct / n
        infer_us_per_sample = (elapsed / n) * 1e6

        return EvalResult(
            name=self.name,
            metrics={
                "accuracy": accuracy,
                "n_classes": self.config.n_classes,
                "n_channels": ch,
                "n_timepoints": tp,
                "train_samples": n,
                "infer_us_per_sample": infer_us_per_sample,
            },
            metadata={"config": self.config.to_dict()},
        )


class _StandaloneBCIDecoder:
    """Simple baseline decoder for BCI signals when Arthedain is unavailable."""

    def predict(self, signal: np.ndarray) -> int:
        """Decode by power band ratios."""
        power = np.mean(signal ** 2)
        return int(power * 100) % 4
