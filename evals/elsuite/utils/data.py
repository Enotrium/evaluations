"""Data utilities for elsuite evaluations."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def make_benchmark_data(
    n_samples: int = 1000,
    n_features: int = 64,
    n_classes: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic benchmark data with train/test split.

    Returns (X_train, X_test, y_train, y_test).
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=max(n_features // 2, n_classes),
        random_state=seed,
    )
    X = X.astype(np.float32)
    return train_test_split(X, y, test_size=0.2, random_state=seed)
