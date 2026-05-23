"""Tests for the EvalRegistry API."""

from evals.base import BaseEval, EvalConfig, EvalResult
from evals.api import EvalRegistry, register, list_evals


class DummyEvalA(BaseEval):
    name = "test.a"
    config = EvalConfig()

    def run(self) -> EvalResult:
        return EvalResult(name=self.name, metrics={"score": 1.0})


class DummyEvalB(BaseEval):
    name = "test.b"
    config = EvalConfig()

    def run(self) -> EvalResult:
        return EvalResult(name=self.name, metrics={"score": 2.0})


def test_register():
    EvalRegistry.register("test.a", DummyEvalA, force=True)
    EvalRegistry.register("test.b", DummyEvalB, force=True)
    assert "test.a" in EvalRegistry.list()
    assert "test.b" in EvalRegistry.list()


def test_get():
    cls = EvalRegistry.get("test.a")
    assert cls is DummyEvalA


def test_run():
    EvalRegistry.register("test.a", DummyEvalA, force=True)
    result = EvalRegistry.run("test.a")
    assert result.metrics["score"] == 1.0


def test_list_filter():
    EvalRegistry.register("test.a", DummyEvalA, force=True)
    EvalRegistry.register("test.b", DummyEvalB, force=True)
    names = EvalRegistry.list(prefix="test")
    assert "test.a" in names
    assert "test.b" in names


def test_register_convenience():
    register("test.new", DummyEvalA)
    assert "test.new" in list_evals()
