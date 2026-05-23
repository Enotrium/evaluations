# Evaluations — Benchmark Registry for SNN & HDC

[![CI](https://github.com/Enotrium/evaluations/actions/workflows/ci.yml/badge.svg)](https://github.com/Enotrium/evaluations/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md) [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](pyproject.toml)

Evaluations is an open, opinionated registry and framework for reproducible benchmarking of Spiking Neural Networks (SNNs) and Hyperdimensional Computing (HDC) systems. It provides a small, consistent API and CLI to run, compare, and share evaluation results.

Why this repo exists
- Single place to discover community and in-house benchmarks for SNN/HDC research.
- Reproducible runs with serialized results for easy comparison and sharing.
- Integrates with Arthedain where available, otherwise uses standalone implementations.

Quick Start
----------
Install the project for development and run a quick demo from the CLI.

```bash
pip install -e ".[dev]"

# Show environment and registered evals
oaieval status

# List available evals
oaieval list

# Run a single eval (example)
oaieval run hdc.classification --seed 42

# Run all registered evals
oaieval run-all

# Compare saved result files
oaieval compare results/
```

Core Concepts
-------------
- `Eval`: a benchmark implemented as a subclass of `BaseEval` that exposes a `run()` method and a `config`.
- `EvalRegistry`: discover/register and run evals by dotted name (e.g. `hdc.classification`).
- `EvalResult`: serializable result object saved as JSON for later comparison.

Available Evals (high level)
----------------------------
This repository ships several example evals across categories (see `evals/elsuite/`):

- SNN: `snn.bci_decoding`, `snn.ablation`, `snn.force2_benchmark`
- HDC: `hdc.classification`, `hdc.robustness`
- Hardware: `hardware.fixed_point_quantization`
- Energy: `energy.comparison`

Programmatic usage
-------------------
Run evals from Python:

```python
from evals import EvalRegistry

EvalRegistry.load_all()
res = EvalRegistry.run("hdc.classification", {"seed": 42})
print(res.metrics)
```

Adding a new eval
-----------------
Create a Python file in `evals/elsuite/<category>/` with a `BaseEval` subclass, then register it (module-level registration or via `Registry.from_yaml`). See `docs/adding_new_eval.md` for a step-by-step guide.

CLI Reference
-------------
- `oaieval status`        — Show system, PyTorch and Arthedain status
- `oaieval list`          — List registered evals
- `oaieval info <name>`   — Show eval details
- `oaieval run <name>`    — Run an eval (options: `--seed`, `--device`, `--trials`)
- `oaieval run-all`       — Run all registered evals
- `oaieval compare <dir>` — Compare saved JSON results
- `oaieval register <yaml>` — Register evals from a YAML file

Development
-----------
Recommended developer workflow:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest evals/tests/ -v

# Format & lint
black evals/ && ruff evals/
```

Project structure
-----------------
See `docs/architecture.md` for a guided tour. Top-level layout:

```
evals/                     # package
    base.py                  # BaseEval, EvalConfig, EvalResult
    api.py                   # EvalRegistry
    registry.py              # YAML/module registration helpers
    record.py                # save/load results (JSON)
    elsuite/                 # concrete eval implementations (categories)
    cli/                     # oaieval CLI entrypoints
```

Arthedain integration
---------------------
If you have Arthedain installed locally, the CLI detects it and routes to its implementations when available. Install locally and re-run `oaieval status` to see detection.

Contributing
------------
We welcome contributions. See `CONTRIBUTING.md` and `docs/adding_new_eval.md` for how to add a benchmark, write tests, and submit a clean pull request.

License
-------
MIT

----
_This README is a concise landing page; more detailed developer and contribution docs live in `docs/`._
