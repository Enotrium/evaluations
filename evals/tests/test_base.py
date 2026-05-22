"""Tests for base module."""

from evals.base import BaseEval, EvalConfig, EvalResult


def test_eval_config_defaults():
    cfg = EvalConfig()
    assert cfg.seed == 42
    assert cfg.device == "cpu"
    assert cfg.num_trials == 1


def test_eval_config_to_dict():
    cfg = EvalConfig(name="test.eval", seed=7)
    d = cfg.to_dict()
    assert d["name"] == "test.eval"
    assert d["seed"] == 7


def test_eval_result_to_dict():
    r = EvalResult(name="test.eval", metrics={"acc": 0.95})
    d = r.to_dict()
    assert d["name"] == "test.eval"
    assert d["metrics"]["acc"] == 0.95
    assert d["status"] == "completed"


def test_eval_result_save_load(tmp_path):
    r = EvalResult(name="test.eval", metrics={"acc": 0.95})
    path = r.save(tmp_path / "result.json")
    assert path.exists()
    loaded = EvalResult.load(path)
    assert loaded.name == "test.eval"
    assert loaded.metrics["acc"] == 0.95


class SimpleEval(BaseEval):
    name = "simple"
    config = EvalConfig()

    def run(self) -> EvalResult:
        return EvalResult(name=self.name, metrics={"answer": 42})


def test_base_eval():
    e = SimpleEval()
    result = e.run()
    assert result.name == "simple"
    assert result.metrics["answer"] == 42
