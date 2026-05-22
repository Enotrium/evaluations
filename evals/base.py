"""BaseEval — Abstract base class for all evaluations.

Every evaluation implements:
    1. run() — execute the benchmark and return EvalResult
    2. name — unique identifier string
    3. config — EvalConfig with hyperparameters / eval settings

Results are serializable to JSON for registry comparison.
"""

from __future__ import annotations

import time
import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Configuration for a single eval run."""
    name: str = ""
    seed: int = 42
    device: str = "cpu"
    num_trials: int = 1
    description: str = ""
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """
    Structured result from a completed evaluation.

    Fields:
        name        — eval identifier (e.g. "snn.bci_decoding")
        status      — "completed", "failed", "skipped"
        metrics     — dict of metric name → scalar/array
        metadata    — run context (config, timing, versions)
        error       — error message if failed
        result_id   — unique UUID for this result
    """
    name: str
    status: str = "completed"
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    result_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "name": self.name,
            "status": self.status,
            "metrics": _serializable(self.metrics),
            "metadata": _serializable(self.metadata),
            "error": self.error,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvalResult":
        self = cls.__new__(cls)
        for k, v in d.items():
            setattr(self, k, v)
        return self

    def save(self, path: str | Path) -> Path:
        """Save result as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EvalResult":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _serializable(obj: Any) -> Any:
    """Convert numpy/torch types to native Python for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Abstract base eval
# ---------------------------------------------------------------------------

class BaseEval(ABC):
    """Abstract base for all evaluations.

    Subclasses must implement:
        - run() -> EvalResult
    Subclasses should set:
        - name (class-level string identifier)
        - config (EvalConfig instance)
    """

    name: str = "base"
    config: EvalConfig = EvalConfig()

    def __init__(self, config: Optional[EvalConfig] = None):
        if config is not None:
            self.config = config

    @abstractmethod
    def run(self) -> EvalResult:
        """Execute the evaluation and return results."""
        ...

    def __call__(self) -> EvalResult:
        return self.run()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
