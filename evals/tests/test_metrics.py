"""Tests for metric aggregation and comparison."""

import pytest
from evals.base import EvalResult
from evals.metrics import MetricSuite, Compare


def test_metric_suite():
    results = [
        EvalResult(name="test.eval", metrics={"acc": 0.9}),
        EvalResult(name="test.eval", metrics={"acc": 0.8}),
        EvalResult(name="test.eval", metrics={"acc": 0.7}),
    ]
    suite = MetricSuite(results)
    assert suite.name == "test.eval"
    assert "acc" in suite.metric_names
    assert suite.mean("acc") == pytest.approx(0.8)


def test_metric_suite_empty():
    results = [EvalResult(name="test.eval", metrics={"acc": 0.5})]
    suite = MetricSuite(results)
    assert suite.mean("acc") == 0.5


def test_compare(tmp_path):
    r1 = EvalResult(name="eval.a", metrics={"score": 1.0})
    r2 = EvalResult(name="eval.b", metrics={"score": 2.0})
    r1.save(tmp_path / "r1.json")
    r2.save(tmp_path / "r2.json")
    comp = Compare(tmp_path)
    table = comp.table()
    assert "eval.a" in table
    assert "eval.b" in table
