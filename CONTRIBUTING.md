# Contributing

Thanks for wanting to contribute! This project follows a simple, focused workflow.

1. Fork the repository and create a branch for your work.
2. Ensure tests pass locally and linting/formatting is applied (`black`, `ruff`).
3. Add or update tests under `evals/tests/` for any new behavior.
4. Open a PR describing your changes and reference a related issue if available. Use the PR template in `.github/PULL_REQUEST_TEMPLATE.md`.

Guidelines
- Keep `Eval` implementations small and deterministic where possible (seed inputs, fixed data sizes).
- Provide documentation in `docs/` for non-trivial changes.
- Use `EvalResult` for outputs; prefer scalar metrics and serializable metadata.
