"""Integration tests for core evaluations."""

from evals.api import EvalRegistry
from evals.base import EvalConfig


def _register_evals():
    """Manually register evals for testing."""
    from evals.elsuite.energy.comparison import EnergyComparisonEval
    from evals.elsuite.hardware.fixed_point import FixedPointQuantizationEval
    from evals.elsuite.hdc.classification import HDCClassificationEval
    from evals.elsuite.hdc.robustness import HDCRobustnessEval

    EvalRegistry.register("energy.comparison", EnergyComparisonEval, force=True)
    EvalRegistry.register(
        "hardware.fixed_point_quantization", FixedPointQuantizationEval, force=True
    )
    EvalRegistry.register("hdc.classification", HDCClassificationEval, force=True)
    EvalRegistry.register("hdc.robustness", HDCRobustnessEval, force=True)


def test_energy_comparison():
    _register_evals()
    result = EvalRegistry.run("energy.comparison")
    assert result.status == "completed"
    assert "energy_nJ_hdc" in result.metrics
    assert result.metrics["energy_nJ_hdc"] > 0


def test_hdc_classification():
    _register_evals()
    result = EvalRegistry.run(
        "hdc.classification", {"n_samples": 100, "n_features": 16, "n_classes": 4}
    )
    assert result.status == "completed"
    assert "accuracy" in result.metrics
    assert 0 <= result.metrics["accuracy"] <= 1


def test_hdc_robustness():
    _register_evals()
    result = EvalRegistry.run(
        "hdc.robustness", {"n_samples": 100, "n_features": 16, "n_classes": 4}
    )
    assert result.status == "completed"
    assert "accuracy_clean" in result.metrics


def test_fixed_point_quantization():
    _register_evals()
    result = EvalRegistry.run(
        "hardware.fixed_point_quantization",
        {"n_samples": 100, "n_features": 16, "n_classes": 4},
    )
    assert result.status == "completed"
    assert "accuracy_fp32" in result.metrics
    assert "accuracy_int8" in result.metrics
