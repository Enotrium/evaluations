"""oaievalset — CLI for running evaluation sets (suites of evals).

Usage:
    oaievalset run <yaml>                  Run an eval set from YAML
    oaievalset compare <dir>               Compare eval set results
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

from evals.api import EvalRegistry
from evals.record import record_result


def _run_set(yaml_path: str) -> int:
    path = Path(yaml_path)
    if not path.exists():
        print(f"Config not found: {path}")
        return 1

    data = yaml.safe_load(path.read_text())
    evals_config = data.get("evals", {})
    failures = []

    for name, opts in evals_config.items():
        print(f"\n{'=' * 60}")
        print(f"Running: {name}")
        print(f"{'=' * 60}")
        t0 = time.time()
        try:
            result = EvalRegistry.run(name, opts if isinstance(opts, dict) else {})
            elapsed = time.time() - t0
            mark = "✓" if result.status == "completed" else "✗"
            print(f"  [{mark}] {result.status} ({elapsed:.2f}s)")
            if result.metrics:
                for k, v in result.metrics.items():
                    if isinstance(v, float):
                        print(f"    {k}: {v:.4f}")
                    else:
                        print(f"    {k}: {v}")
            record_result(result)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [✗] failed ({elapsed:.2f}s): {e}")
            failures.append(name)

    total = len(evals_config)
    passed = total - len(failures)
    print(f"\nSet complete: {passed}/{total} passed.")
    if failures:
        print(f"Failures: {', '.join(failures)}")
    return len(failures)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oaievalset", description="Run eval sets.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run an eval set from YAML")
    run_p.add_argument("yaml", type=str, help="Path to YAML config")

    args = parser.parse_args(argv)
    if args.command == "run":
        return _run_set(args.yaml)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
