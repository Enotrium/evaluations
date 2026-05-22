"""TaskState — track the state of a running evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .base import EvalConfig


@dataclass
class TaskState:
    """Mutable state for an in-progress evaluation task."""

    name: str
    config: EvalConfig
    status: str = "pending"  # pending | running | completed | failed
    progress: float = 0.0  # 0.0 → 1.0
    current_step: str = ""
    error: Optional[str] = None
    intermediate: dict[str, Any] = field(default_factory=dict)
