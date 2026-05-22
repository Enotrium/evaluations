"""Base class for completion functions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CompletionResult:
    """Result from a completion function call."""
    text: str = ""
    logprobs: Optional[list[float]] = None
    tokens: Optional[list[str]] = None
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


class CompletionFn(ABC):
    """Abstract completion function that wraps model execution."""

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> CompletionResult:
        ...
