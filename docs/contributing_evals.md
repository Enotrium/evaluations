When to add a new eval
Required interface
File placement
Minimal example
Registration
Running locally
Testing expectations
PR checklist

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