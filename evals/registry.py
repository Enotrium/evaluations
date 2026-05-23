"""Registry module — YAML and file-based eval registration."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml

from .api import EvalRegistry


class Registry:
    """Handles YAML-based and file-based registration of evals."""

    @staticmethod
    def from_yaml(yaml_path: str | Path) -> list[str]:
        """Register evals from a YAML config file.

        Expected format:
            evals:
              snn.bci_decoding:
                module: evals.elsuite.snn.bci_decoding
                class: BCIDecodingEval
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"YAML config not found: {path}")

        data = yaml.safe_load(path.read_text())
        registered = []

        for eval_name, spec in data.get("evals", {}).items():
            module_path = spec["module"]
            class_name = spec["class"]
            mod = importlib.import_module(module_path)
            eval_cls = getattr(mod, class_name)
            EvalRegistry.register(eval_name, eval_cls)
            registered.append(eval_name)

        return registered

    @staticmethod
    def from_module(module_path: str) -> list[str]:
        """Discover and register all evals in a module."""
        mod = importlib.import_module(module_path)

        from .base import BaseEval

        registered = []
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseEval)
                and attr is not BaseEval
            ):
                name = getattr(attr, "name", attr_name.lower())
                EvalRegistry.register(name, attr)
                registered.append(name)
        return registered
