"""Energy efficiency benchmarks."""


def register_all():
    """Register all energy evals."""
    from evals.api import EvalRegistry
    from .comparison import EnergyComparisonEval

    EvalRegistry.register("energy.comparison", EnergyComparisonEval)


register_all()
