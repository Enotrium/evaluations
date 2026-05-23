# Adding a new evaluation

Step-by-step: a minimal example to add a new benchmark to `evaluations`.

1. Create the file `evals/elsuite/<category>/<name>.py`.

2. Implement a subclass of `BaseEval`:

```python
from evals.base import BaseEval, EvalConfig, EvalResult

class MyEval(BaseEval):
    name = "my.category"
    config = EvalConfig(description="My demo eval", seed=42)

    def run(self) -> EvalResult:
        # Put benchmark code here. Use evals.data utilities for toy datasets.
        from evals.data import make_synthetic_dataset
        X, y = make_synthetic_dataset(100, 16, 3, seed=self.config.seed)
        metrics = {"accuracy": 0.5}
        return EvalResult(name=self.name, metrics=metrics)
```

3. Register the eval:

- Option A: add module-level registration inside `evals/elsuite/<category>/__init__.py` calling `EvalRegistry.register("my.category", MyEval)`.
- Option B: create a YAML registration and run `oaieval register path/to/file.yaml`.

4. Add tests under `evals/tests/` exercising the eval run and expected metrics.

5. Run locally: `oaieval run my.category --seed 42` and `pytest evals/tests/ -v`.
