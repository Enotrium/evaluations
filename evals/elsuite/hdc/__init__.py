"""HDC (Hyperdimensional Computing) evaluations."""


def register_all():
    """Register all HDC evals."""
    from evals.api import EvalRegistry
    from .classification import HDCClassificationEval
    from .robustness import HDCRobustnessEval

    EvalRegistry.register("hdc.classification", HDCClassificationEval)
    EvalRegistry.register("hdc.robustness", HDCRobustnessEval)


register_all()
