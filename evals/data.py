"""Data utilities and dataset loading for evaluations."""

from __future__ import annotations

import numpy as np


def make_synthetic_dataset(
    n_samples: int,
    n_features: int,
    n_classes: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic classification dataset.

    Returns (X, y) where X is shape (n_samples, n_features) and
    y is shape (n_samples,) with integer labels.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    # Hidden linear decision boundary with noise
    w = rng.randn(n_features, n_classes)
    logits = X @ w
    y = np.argmax(logits, axis=1)
    return X.astype(np.float32), y


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple train/test split."""
    rng = np.random.RandomState(seed)
    n = len(X)
    perm = rng.permutation(n)
    split = int(n * train_ratio)
    idx_train = perm[:split]
    idx_test = perm[split:]
    return X[idx_train], X[idx_test], y[idx_train], y[idx_test]
