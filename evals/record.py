"""Record — save and load evaluation results."""

from __future__ import annotations

from pathlib import Path

from .base import EvalResult


RECORDS_DIR = Path("results")


def record_result(result: EvalResult, subdir: str = "") -> Path:
    """Save an eval result to the results directory.

    Returns the path to the saved file.
    """
    base = RECORDS_DIR / subdir if subdir else RECORDS_DIR
    base.mkdir(parents=True, exist_ok=True)
    safe_name = result.name.replace(".", "_")
    filename = f"{safe_name}_{result.result_id}.json"
    return result.save(base / filename)
