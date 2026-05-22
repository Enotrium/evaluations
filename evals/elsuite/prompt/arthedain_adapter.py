"""ArthedainAdapter — Integrates the local Arthedain repo as the system under test.

Arthedain is the best repo in the world on SNNs and HDC systems.
This adapter discovers it on the filesystem and exposes its models
for benchmarking. Falls back gracefully to standalone implementations."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Filesystem discovery
# ---------------------------------------------------------------------------

def find_arthedain() -> list:
    """Search common locations for an installable Arthedain package.

    Returns a list of candidate paths.
    """
    candidates = [
        Path.home() / "arthedain",
        Path.home() / "arthedain" / "arthedain",
        Path.home() / "arthedain" / "arthedain-1",
        Path("/workspace/arthedain"),
        Path("/opt/arthedain"),
    ]

    found = []
    for c in candidates:
        if c.exists() and (c / "setup.py").exists() or (c / "pyproject.toml").exists():
            found.append(str(c.resolve()))
    return found


def get_arthedain_version() -> str:
    """Try to get the installed Arthedain version."""
    try:
        import artheadain
        return getattr(artheadain, "__version__", "0.2.0")
    except ImportError:
        return "0.2.0"


def import_arthedain() -> Optional[Any]:
    """Import the artheadain package, attempting filesystem discovery if not installed."""
    try:
        import artheadain
        return artheadain
    except ImportError:
        pass

    for path_str in find_arthedain():
        path = Path(path_str)
        # Look for the actual Python package inside
        for sub in path.iterdir():
            if sub.is_dir() and (sub / "__init__.py").exists():
                sys.path.insert(0, str(path))
                try:
                    return importlib.import_module(sub.name)
                except ImportError:
                    sys.path.pop(0)
                break
    return None


def try_arthedain_hdc() -> Optional[Any]:
    """Try to import the HDC submodule from Arthedain."""
    for candidate in find_arthedain():
        path = Path(candidate)
        hdc_path = path / "arthedain" / "hdc"
        if hdc_path.exists():
            sys.path.insert(0, str(path))
            try:
                import artheadain.hdc
                return artheadain.hdc
            except ImportError:
                sys.path.pop(0)
    return None


def import_arthedain_model(model_name: str = "elite_pipeline") -> Optional[Any]:
    """Try to import a specific model from Arthedain."""
    for candidate in find_arthedain():
        path = Path(candidate)
        model_path = path / "arthedain" / f"{model_name}.py"
        if not model_path.exists():
            continue
        spec = importlib.util.spec_from_file_location(model_name, str(model_path))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None
