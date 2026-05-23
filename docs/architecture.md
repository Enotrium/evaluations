Sections:

Overview
Core Components
Registry Flow
Eval Execution Flow
Arthedain Integration
Result Recording
Extension Points

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