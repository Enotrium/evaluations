# Architecture

This document gives a high-level tour of the `evaluations` repository and how its pieces fit together.

Core packages

- `evals/` — main Python package. Key modules:
  - `base.py` — `BaseEval`, `EvalConfig`, `EvalResult` (abstract eval and result model).
  - `api.py` — `EvalRegistry` for discovery, registration, and running of evals.
  - `registry.py` — helpers to register evals from YAML or modules.
  - `record.py` — saving/loading `EvalResult` objects as JSON.
  - `elsuite/` — concrete eval implementations organized by category (e.g. `snn`, `hdc`, `hardware`, `energy`).
  - `cli/` — `oaieval` CLI entrypoint.

Runtime flow

1. CLI or program calls `EvalRegistry.load_all()` to import `evals.elsuite` modules (auto-discovery).
2. `EvalRegistry` holds a mapping `name -> EvalClass`. Use `EvalRegistry.run(name)` to instantiate and execute an eval.
3. Each eval implements `run()` and returns an `EvalResult` (serializable to JSON).
4. Results are saved with `record_result(result)` so they can be compared later via `evals.metrics.Compare`.

Extending the system

- Add a new eval by creating a subclass of `BaseEval` in `evals/elsuite/<category>/`.
- Register with `EvalRegistry.register()` or via YAML using `Registry.from_yaml()`.

Notes

- The system is intentionally lightweight and framework-agnostic; where Arthedain is installed, some evals will prefer its implementations.
- Tests live under `evals/tests/` and provide runnable examples of expected behavior.
