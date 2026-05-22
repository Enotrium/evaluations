"""Tests for data module."""

from evals.data import make_synthetic_dataset, train_test_split


def test_make_synthetic_dataset():
    X, y = make_synthetic_dataset(100, 20, 5, seed=42)
    assert X.shape == (100, 20)
    assert y.shape == (100,)
    assert len(set(y)) == 5


def test_train_test_split():
    X, y = make_synthetic_dataset(100, 10, 3)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_ratio=0.8)
    assert len(X_tr) == 80
    assert len(X_te) == 20
