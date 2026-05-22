"""MetricSuite and Compare — aggregate and compare evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import EvalResult


# ---------------------------------------------------------------------------
# MetricSuite — collection of results for a single eval
# ---------------------------------------------------------------------------

class MetricSuite:
    """Aggregate multiple EvalResult objects for the same eval.

    Provides mean, std, min, max across runs.
    """

    def __init__(self, results: list[EvalResult]):
        if not results:
            raise ValueError("MetricSuite requires at least one result.")
        self.results = results
        self.name = results[0].name

    @property
    def metric_names(self) -> list[str]:
        """Return union of all metric keys across results."""
        names: set[str] = set()
        for r in self.results:
            names.update(r.metrics.keys())
        return sorted(names)

    def mean(self, metric: str) -> float:
        return float(np.mean([r.metrics.get(metric, np.nan) for r in self.results]))

    def std(self, metric: str) -> float:
        return float(np.std([r.metrics.get(metric, np.nan) for r in self.results]))

    def min(self, metric: str) -> float:
        return float(np.min([r.metrics.get(metric, np.nan) for r in self.results]))

    def max(self, metric: str) -> float:
        return float(np.max([r.metrics.get(metric, np.nan) for r in self.results]))

    def summary_dict(self) -> dict[str, str]:
        """Return a dict of 'metric: mean±std' strings."""
        d = {}
        for m in self.metric_names:
            vals = [r.metrics.get(m, None) for r in self.results]
            vals = [v for v in vals if v is not None]
            if vals:
                d[m] = f"{np.mean(vals):.4f}±{np.std(vals):.4f}"
            else:
                d[m] = "—"
        return d


# ---------------------------------------------------------------------------
# Compare — load & compare multiple MetricSuites
# ---------------------------------------------------------------------------

class Compare:
    """Load results from a directory and produce comparison tables."""

    def __init__(self, results_dir: str | Path):
        self.results_dir = Path(results_dir)
        self.results_by_eval: dict[str, list[EvalResult]] = {}
        self._load_all()

    def _load_all(self) -> None:
        for path in sorted(self.results_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                result = EvalResult.from_dict(data)
                self.results_by_eval.setdefault(result.name, []).append(result)
            except (json.JSONDecodeError, KeyError):
                continue

    @property
    def suite_names(self) -> list[str]:
        return sorted(self.results_by_eval.keys())

    def get_suite(self, name: str) -> MetricSuite:
        return MetricSuite(self.results_by_eval[name])

    def table(self) -> str:
        """Render a comparison table across all eval suites."""
        if not self.results_by_eval:
            return "No results found."

        suites = {name: MetricSuite(results) for name, results in self.results_by_eval.items()}
        all_metrics: list[str] = []
        for s in suites.values():
            all_metrics.extend(s.metric_names)
        all_metrics = sorted(set(all_metrics))

        header = f"| Eval | {' | '.join(f'{m} (mean±std)' for m in all_metrics)} |"
        sep = f"|{'|'.join('-' * max(5, len(h)) for h in header.split('|')[1:-1])}|"

        rows = []
        for name, suite in suites.items():
            summary = suite.summary_dict()
            row = f"| {name} | {' | '.join(summary.get(m, '—') for m in all_metrics)} |"
            rows.append(row)

        lines = ["Comparison Table:", "=" * len(header), header, sep] + rows
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        suites = {name: MetricSuite(results) for name, results in self.results_by_eval.items()}
        result: dict[str, Any] = {}
        for name, suite in suites.items():
            result[name] = {}
            for m in suite.metric_names:
                result[name][m] = {
                    "mean": suite.mean(m),
                    "std": suite.std(m),
                    "min": suite.min(m),
                    "max": suite.max(m),
                }
        return result

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        data = self.to_dict()
        path.write_text(json.dumps(data, indent=2))
        return path
