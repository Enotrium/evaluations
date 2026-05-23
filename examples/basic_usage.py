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