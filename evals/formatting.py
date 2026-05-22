"""Result formatting utilities for CLI display."""

from __future__ import annotations

from typing import Any


def format_result(metrics: dict[str, Any], indent: int = 2) -> str:
    """Format a metrics dict for CLI display."""
    prefix = " " * indent
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{prefix}{key}: {value:.4f}")
        elif isinstance(value, (int, str)):
            lines.append(f"{prefix}{key}: {value}")
        elif isinstance(value, (list, tuple)) and len(value) <= 10:
            formatted = ", ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in value)
            lines.append(f"{prefix}{key}: [{formatted}]")
        elif isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            for k, v in value.items():
                lines.append(f"{prefix}  {k}: {v}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines)
