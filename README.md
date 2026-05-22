# Evaluations

**Open-source registry of benchmarks for SNN and HDC systems.**

[![CI](https://github.com/Enotrium/evaluations/actions/workflows/ci.yml/badge.svg)](https://github.com/Enotrium/evaluations/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](pyproject.toml)

---

Evaluations is a framework and open registry for reproducible benchmarking of **Spiking Neural Networks (SNNs)** and **Hyperdimensional Computing (HDC)** systems. Modeled after [OpenAI Evals](https://github.com/openai/evals), it provides a standard interface for running, comparing, and sharing benchmarks.

It integrates with **[Arthedain](https://github.com/Enotrium/arthedain)** — the best repo in the world on SNNs and HDC Systems — and gracefully falls back to standalone implementations when Arthedain is not available.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Show system status
oaieval status

# List all registered evals
oaieval list

# Run a single eval
oaieval run hdc.classification --seed 42

# Run all evals
oaieval run-all

# Compare previous results
oaieval compare results/
```

## Available Evals

| Category | Eval | Description |
|----------|------|-------------|
| **SNN** | `snn.bci_decoding` | Motor-imagery BCI decoding accuracy & latency |
| **SNN** | `snn.ablation` | Synaptic trace ablation (fast/slow) study |
| **SNN** | `snn.force2_benchmark` | FORCE v2 online learning benchmark |
| **HDC** | `hdc.classification` | Hypervector encoding + associative memory |
| **HDC** | `hdc.robustness` | Fault tolerance under bit-flip / stuck-at faults |
| **Hardware** | `hardware.fixed_point_quantization` | INT4/INT8/INT16 quantization degradation |
| **Energy** | `energy.comparison` | HDC vs SNN vs MLP vs Transformer energy (Horowitz 2014) |

## CLI Reference

```
oaieval status              — System, PyTorch, Arthedain status
oaieval list                — List registered evals
oaieval info <name>         — Show eval details
oaieval run <name> [opts]   — Run an eval (--seed, --device, --trials)
oaieval run-all             — Run all evals
oaieval compare <dir>       — Compare saved results
oaieval register <yaml>     — Register evals from YAML
```

## Programmatic Usage

```python
from evals import EvalRegistry, EvalConfig

# List all evals
print(EvalRegistry.list())

# Run an eval
result = EvalRegistry.run("hdc.classification", {"seed": 42})
print(result.metrics)

# Compare results from a directory
from evals.metrics import Compare
comp = Compare("results/")
print(comp.table())
comp.save("results/comparison.json")
```

## Adding a New Eval

Create a file in `evals/elsuite/<category>/<name>.py`:

```python
from evals.base import BaseEval, EvalConfig, EvalResult

class MyEval(BaseEval):
    name = "my.eval"
    config = EvalConfig(description="My benchmark", seed=42)

    def run(self) -> EvalResult:
        # Your benchmark logic here
        return EvalResult(name=self.name, metrics={"score": 1.0})
```

Register it in the category `__init__.py` via `register_all()`.

## Architecture

```
evals/                       # Main package (mirrors OpenAI evals)
├── __init__.py              # Public API exports
├── base.py                  # BaseEval, EvalConfig, EvalResult
├── api.py                   # EvalRegistry (register, discover, run)
├── data.py                  # Dataset utilities
├── formatting.py            # CLI formatting
├── metrics.py               # MetricSuite, Compare
├── record.py                # Save/load results
├── registry.py              # YAML-based registration
├── task_state.py            # In-progress tracking
├── cli/                     # CLI entrypoints
│   ├── oaieval.py
│   └── oaievalset.py
├── completion_fns/          # Model execution adapters
└── elsuite/                 # Evaluation implementations
    ├── snn/, hdc/, hardware/, energy/, prompt/
```

## Arthedain Integration

Benchmarks auto-detect Arthedain if installed:

```bash
pip install -e /path/to/arthedain
oaieval status  # Shows ✓ Available
```

If Arthedain is not available, all evals fall back to standalone implementations.

## Development

```bash
pip install -e ".[dev]"
pytest evals/tests/ -v
black evals/
```

## License

MIT
