"""SNN (Spiking Neural Network) evaluations."""


def register_all():
    """Register all SNN evals."""
    from evals.api import EvalRegistry
    from .bci_decoding import BCIDecodingEval
    from .ablation import AblationEval
    from .force2_benchmark import FORCE2BenchmarkEval

    EvalRegistry.register("snn.bci_decoding", BCIDecodingEval)
    EvalRegistry.register("snn.ablation", AblationEval)
    EvalRegistry.register("snn.force2_benchmark", FORCE2BenchmarkEval)


register_all()
