"""Fixed-Point Quantization Eval — Measure accuracy degradation under reduced precision.

Benchmarks SNN/HDC model accuracy when weights are quantized to
INT4, INT8, or INT16 from FP32 baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from evals.base import EvalConfig, EvalResult
from evals.elsuite.prompt.base import SolverEval
from evals.data import make_synthetic_dataset, train_test_split


@dataclass
class FixedPointConfig(EvalConfig):
    n_samples: int = 1000
    n_features: int = 64
    n_classes: int = 4
    description: str = "Fixed-point quantization degradation analysis"


class FixedPointQuantizationEval(SolverEval):
    """Evaluate accuracy degradation under reduced precision."""

    name: str = "hardware.fixed_point_quantization"
    config: FixedPointConfig = FixedPointConfig()

    def _load_solver(self) -> Any:
        solver = self._make_arthedain_solver()
        if solver is not None:
            return solver
        return _StandaloneSolver(n_features=self.config.n_features,
                                 n_classes=self.config.n_classes)

    def run(self) -> EvalResult:
        np.random.seed(self.config.seed)
        n = self.config.n_samples
        feats = self.config.n_features

        X, y = make_synthetic_dataset(n, feats, self.config.n_classes, seed=self.config.seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, seed=self.config.seed)

        # Train
        for xi, yi in zip(X_train, y_train):
            self.solver.train(xi, yi)

        # Baseline FP32 accuracy
        correct_fp32 = sum(1 for xi, yi in zip(X_test, y_test)
                           if self.solver.predict(xi) == yi)
        fp32_acc = correct_fp32 / len(X_test)

        metrics = {"accuracy_fp32": fp32_acc}

        # Quantize weights and evaluate
        for bits in [16, 8, 4]:
            self.solver.quantize_weights(bits=bits)
            correct_q = sum(1 for xi, yi in zip(X_test, y_test)
                            if self.solver.predict(xi) == yi)
            q_acc = correct_q / len(X_test)
            metrics[f"accuracy_int{bits}"] = q_acc
            metrics[f"degradation_int{bits}"] = fp32_acc - q_acc

        return EvalResult(name=self.name, metrics=metrics, metadata={"config": self.config.to_dict()})


class _StandaloneSolver:
    """Simple linear classifier with weight quantization."""

    def __init__(self, n_features: int = 64, n_classes: int = 4):
        self.w: np.ndarray = np.zeros((n_features, n_classes), dtype=np.float32)
        self._w_original: np.ndarray = np.zeros((n_features, n_classes), dtype=np.float32)
        self.lr = 0.01

    def train(self, x: np.ndarray, y: int) -> None:
        pred = x @ self.w
        target = np.zeros(self.w.shape[1], dtype=np.float32)
        target[y] = 1.0
        error = target - pred
        self.w += self.lr * np.outer(x, error)

    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(x @ self.w))

    def quantize_weights(self, bits: int = 8) -> None:
        self._w_original = self.w.copy()
        max_val = np.max(np.abs(self.w))
        if max_val == 0:
            return
        scale = (2 ** (bits - 1)) - 1
        self.w = np.round(self.w / max_val * scale) / scale * max_val

    def restore_weights(self) -> None:
        self.w = self._w_original.copy()
