# Evaluations

**Open-source registry of benchmarks for SNN and HDC systems.**

[![CI](https://github.com/Enotrium/evaluations/actions/workflows/ci.yml/badge.svg)](https://github.com/Enotrium/evaluations/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](pyproject.toml)

---

Evaluations is an open framework and benchmark registry for reproducible evaluation of **Spiking Neural Networks (SNNs)** and **Hyperdimensional Computing (HDC)** systems. Inspired by [OpenAI Evals](https://github.com/openai/evals), it provides a standard interface for running, comparing, and sharing benchmarks.

It integrates with **[Arthedain](https://github.com/Enotrium/arthedain)**, Enotrium's research platform for SNN and HDC systems, and falls back to standalone benchmark implementations when Arthedain is not available.

## Why Evaluations?

Benchmarking emerging SNN and HDC systems is often fragmented, difficult to reproduce, and tightly coupled to individual research codebases.

Evaluations provides a consistent interface for running, comparing, and extending benchmarks across architectures, datasets, and hardware targets.

## Quick Start

Install:

For users:

```bash
pip install .
```

For contributors:

```bash
pip install -e ".[dev]"
```

Check environment status:

```bash
oaieval status
```

List available evaluations:

```bash
oaieval list
```

Run a single evaluation:

```bash
oaieval run hdc.classification --seed 42
```

Run all evaluations:

```bash
oaieval run-all
```

Compare previous benchmark results:

```bash
oaieval compare results/
```


## Included Evals

| Category | Eval | Description |
|----------|------|-------------|
| **SNN** | `snn.bci_decoding` | Motor-imagery BCI decoding accuracy & latency |
| **SNN** | `snn.ablation` | Synaptic trace ablation (fast/slow) study |
| **SNN** | `snn.force2_benchmark` | FORCE v2 online learning benchmark |
| **HDC** | `hdc.classification` | Hypervector encoding + associative memory |
| **HDC** | `hdc.robustness` | Fault tolerance under bit-flip / stuck-at faults |
| **Hardware** | `hardware.fixed_point_quantization` | INT4/INT8/INT16 quantization degradation |
| **Energy** | `energy.comparison` | HDC vs SNN vs MLP vs Transformer energy (Horowitz 2014) |

## CLI Usage

| Command                 | Description                             |
| ----------------------- | --------------------------------------- |
| `oaieval status`        | Show environment and integration status |
| `oaieval list`          | List registered benchmark suites        |
| `oaieval info <name>`   | Show evaluation metadata                |
| `oaieval run <name>`    | Run a specific benchmark                |
| `oaieval run-all`       | Run all registered benchmarks           |
| `oaieval compare <dir>` | Compare saved benchmark results         |

## Documentation

- [Architecture](docs/architecture.md)
- [Adding a New Eval](docs/adding_new_eval.md)
- [Contributing](CONTRIBUTING.md)
- [Basic Usage Example](examples/basic_usage.py)

## Arthedain Integration

Benchmarks auto-detect Arthedain if installed:

```bash
pip install -e /path/to/arthedain
oaieval status  # Shows ✓ Available
```

If Arthedain is unavailable, evaluations fall back to standalone benchmark implementations.

## Development

```bash
pip install -e ".[dev]"
pytest -v
black evals/
```

## License

MIT