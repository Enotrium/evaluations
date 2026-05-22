"""Completion functions — adapters for running models under evaluation.

Each completion fn wraps a model execution interface and returns
text, logprobs, or raw output tensors for evaluation.
"""

from .base import CompletionFn
from .solver_completion_fn import SolverCompletionFn
