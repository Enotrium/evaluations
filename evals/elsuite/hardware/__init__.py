"""Hardware evaluation benchmarks."""


def register_all():
    """Register all hardware evals."""
    from evals.api import EvalRegistry
    from .fixed_point import FixedPointQuantizationEval

    EvalRegistry.register(
        "hardware.fixed_point_quantization", FixedPointQuantizationEval
    )


register_all()
