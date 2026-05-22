"""Public API: EvalRegistry — register, discover, and run benchmarks."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Optional

from .base import BaseEval, EvalConfig, EvalResult


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ELSUTE_PREFIX = "evals.elsuite."


class EvalRegistry:
    """Global registry of named evaluations.

    Evals are auto-discovered from the evals.elsuite package tree.
    """

    _registry: dict[str, type[BaseEval]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str, eval_cls: type[BaseEval], force: bool = False) -> None:
        """Register an eval class under a dotted name."""
        if name in cls._registry and not force:
            raise KeyError(
                f"Eval {name!r} is already registered. Use force=True to overwrite."
            )
        if not issubclass(eval_cls, BaseEval):
            raise TypeError(f"{eval_cls.__name__} must inherit from BaseEval")
        cls._registry[name] = eval_cls

    @classmethod
    def get(cls, name: str) -> type[BaseEval]:
        """Get eval class by dotted name. Auto-discovers if not yet loaded."""
        if name not in cls._registry:
            cls._try_discover(name)
        if name not in cls._registry:
            raise KeyError(
                f"Eval {name!r} not found. Registered: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list(cls, prefix: str = "") -> list[str]:
        """List all registered eval names, optionally filtered by prefix."""
        if not prefix:
            return list(cls._registry.keys())
        return [n for n in cls._registry if n.startswith(prefix)]

    @classmethod
    def load_all(cls) -> None:
        """Auto-discover and register all evals from the elsuite package."""
        _auto_discover_all()

    # ------------------------------------------------------------------
    # Instantiation helpers
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[dict] = None,
        **kwargs,
    ) -> BaseEval:
        """Instantiate an eval by name, injecting config overrides."""
        eval_cls = cls.get(name)
        if config is None:
            cfg = eval_cls.config
        elif isinstance(config, dict):
            cfg_cls = type(eval_cls.config)
            merged = {**eval_cls.config.to_dict(), **config, **kwargs}
            cfg = cfg_cls(**merged)
        else:
            cfg = config
        return eval_cls(config=cfg)

    @classmethod
    def run(
        cls,
        name: str,
        config: Optional[dict] = None,
        **kwargs,
    ) -> EvalResult:
        """Instantiate and run an eval by name."""
        return cls.create(name, config=config, **kwargs).run()

    # ------------------------------------------------------------------
    # Internal discovery
    # ------------------------------------------------------------------

    @classmethod
    def _try_discover(cls, name: str) -> None:
        """Attempt to find an eval class by guessing its module path."""
        parts = name.split(".")
        if len(parts) < 2:
            return
        category = parts[0]
        module_name = parts[-1]
        try:
            full_module = f"evals.elsuite.{category}.{module_name}"
            mod = importlib.import_module(full_module)
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if isinstance(attr, type) and issubclass(attr, BaseEval) and attr is not BaseEval:
                    if name not in cls._registry:
                        cls.register(name, attr)
                        return
        except (ImportError, ModuleNotFoundError):
            pass


def _auto_discover_all() -> None:
    """Walk the elsuite package tree and import all modules."""
    try:
        import evals.elsuite as elsuite_pkg
    except ImportError:
        return

    def _walk(pkg_module, prefix: str):
        for info in pkgutil.walk_packages(
            pkg_module.__path__, prefix=f"{prefix}."
        ):
            if info.ispkg:
                try:
                    mod = importlib.import_module(info.name)
                    _walk(mod, info.name)
                except Exception:
                    pass
            else:
                try:
                    importlib.import_module(info.name)
                except Exception:
                    pass

    _walk(elsuite_pkg, "evals.elsuite")


# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------

def register(name: str, eval_cls: type[BaseEval]) -> None:
    """Top-level convenience function to register an eval."""
    EvalRegistry.register(name, eval_cls)


def list_evals(prefix: str = "") -> list[str]:
    """Top-level convenience function to list evals."""
    return EvalRegistry.list(prefix)
