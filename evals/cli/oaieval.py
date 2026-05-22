"""oaieval — CLI for running individual evaluations.

Usage:
    oaieval list                           List registered evals
    oaieval run <name> [--seed N] [opts]   Run an eval
    oaieval run-all                        Run all registered evals
    oaieval compare <dir>                  Compare results
    oaieval info <name>                    Show eval details
    oaieval status                         System status
    oaieval register <yaml>                Register from YAML file
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
import time
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

import numpy as np

from evals.api import EvalRegistry
from evals.base import EvalConfig, EvalResult
from evals.formatting import format_result
from evals.metrics import Compare
from evals.record import record_result
from evals.registry import Registry


def _get_version() -> str:
    try:
        return importlib.metadata.version("evaluations")
    except Exception:
        return "0.1.0"


def _status_command() -> int:
    print("=== Evaluations System Status ===")
    print()
    print(f"  Version:         {_get_version()}")
    print(f"  Python:          {sys.version.split()[0]}")
    print(f"  PyTorch:         {torch.__version__ if torch else 'N/A'}")
    print(f"  NumPy:           {np.__version__}")
    device = "cpu"
    if torch is not None and torch.cuda.is_available():
        device = "cuda"
    print(f"  Device:          {device}")
    print()

    # Arthedain discovery
    try:
        from evals.elsuite.prompt.arthedain_adapter import find_arthedain, get_arthedain_version

        found = find_arthedain()
        if found:
            print(f"  Arthedain:       ✓ Available")
            print(f"  Arthedain ver:   {get_arthedain_version()}")
            print(f"  Arthedain paths: {', '.join(found)}")
        else:
            print(f"  Arthedain:       ✗ Not found")
    except ImportError:
        print(f"  Arthedain:       ✗ Not available")

    print()
    EvalRegistry.load_all()
    names = EvalRegistry.list()
    print(f"  Registered evals: {len(names)}")
    for n in sorted(names):
        print(f"    - {n}")
    print()
    return 0


def _list_command() -> int:
    EvalRegistry.load_all()
    names = EvalRegistry.list()
    print(f"Registered evals ({len(names)}):")
    print("=" * 60)
    for n in sorted(names):
        print(f"  {n}")
    return 0


def _info_command(name: str) -> int:
    EvalRegistry.load_all()
    try:
        eval_cls = EvalRegistry.get(name)
    except KeyError:
        print(f"Eval {name!r} not found.")
        return 1

    cfg = eval_cls.config
    print(f"Eval: {name}")
    print(f"  Class:    {eval_cls.__name__}")
    print(f"  Module:   {eval_cls.__module__}")
    print(f"  Config:")
    print(f"    seed:         {cfg.seed}")
    print(f"    num_trials:   {cfg.num_trials}")
    print(f"    device:       {cfg.device}")
    if cfg.description:
        print(f"  Description: {cfg.description}")
    return 0


def _do_run(args: argparse.Namespace) -> int:
    EvalRegistry.load_all()
    name = args.name
    config_kwargs = {}
    if args.seed is not None:
        config_kwargs["seed"] = args.seed
    if args.device is not None:
        config_kwargs["device"] = args.device
    if args.trials is not None:
        config_kwargs["num_trials"] = args.trials

    print(f"Running eval: {name}")
    print("-" * 60)

    t0 = time.time()
    try:
        result = EvalRegistry.run(name, config_kwargs)
        elapsed = time.time() - t0
        status_icon = "✓" if result.status == "completed" else "✗"
        print(f"  Status: {status_icon} {result.status} ({elapsed:.2f}s)")
        if result.metrics:
            print(format_result(result.metrics))
        if result.error:
            print(f"  Error: {result.error}")

        saved = record_result(result)
        print(f"  Also saved to: {saved}")
        return 0 if result.status == "completed" else 1
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  Status: ✗ failed ({elapsed:.2f}s)")
        print(f"  Error: {e}")
        return 1


def _run_all_command() -> int:
    EvalRegistry.load_all()
    names = sorted(EvalRegistry.list())
    failures = []
    for name in names:
        print(f"\n{'=' * 60}")
        print(f"Running: {name}")
        print(f"{'=' * 60}")
        t0 = time.time()
        try:
            result = EvalRegistry.run(name)
            elapsed = time.time() - t0
            mark = "✓" if result.status == "completed" else "✗"
            print(f"  [{mark}] {result.status} ({elapsed:.2f}s)")
            record_result(result)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [✗] failed ({elapsed:.2f}s): {e}")
            failures.append(name)

    print(f"\n{'=' * 60}")
    print(f"Run-all complete. {len(names) - len(failures)}/{len(names)} passed.")
    if failures:
        print(f"Failures: {', '.join(failures)}")
    return len(failures)


def _compare_command(results_dir: str) -> int:
    comp = Compare(results_dir)
    table = comp.table()
    print(table)
    out = comp.save(Path(results_dir) / "comparison.json")
    print(f"\nFull comparison saved to: {out}")
    return 0


def _register_command(yaml_path: str) -> int:
    names = Registry.from_yaml(yaml_path)
    print(f"Registered {len(names)} evals from {yaml_path}:")
    for n in names:
        print(f"  - {n}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oaieval",
        description="Run evaluations for SNN and HDC systems.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show system status")

    sub.add_parser("list", help="List all registered evals")

    info_p = sub.add_parser("info", help="Show eval details")
    info_p.add_argument("name", type=str, help="Eval name")

    run_p = sub.add_parser("run", help="Run an eval")
    run_p.add_argument("name", type=str, help="Eval name")
    run_p.add_argument("--seed", type=int, default=None, help="Random seed")
    run_p.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    run_p.add_argument("--trials", type=int, default=None, help="Number of trials")

    sub.add_parser("run-all", help="Run all registered evals")

    cmp_p = sub.add_parser("compare", help="Compare results in a directory")
    cmp_p.add_argument("results_dir", type=str, help="Directory with result JSONs")

    reg_p = sub.add_parser("register", help="Register evals from YAML")
    reg_p.add_argument("yaml", type=str, help="Path to YAML config")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    commands = {
        "status": lambda: _status_command(),
        "list": lambda: _list_command(),
        "info": lambda: _info_command(args.name),
        "run": lambda: _do_run(args),
        "run-all": lambda: _run_all_command(),
        "compare": lambda: _compare_command(args.results_dir),
        "register": lambda: _register_command(args.yaml),
    }
    cmd = commands.get(args.command)
    if cmd is None:
        parser.print_help()
        return 1
    return cmd()


if __name__ == "__main__":
    sys.exit(main())
