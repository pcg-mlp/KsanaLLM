# Copyright 2026 Tencent Inc.  All rights reserved.

"""Pure-Python XGrammar structural-tag API bundled with KsanaLLM.

KsanaLLM links XGrammar's C++ runtime directly, so importing the regular
``xgrammar`` package would unnecessarily require its Python native binding.
The model structural-tag builders themselves are pure Python.  During a source
build we load those modules from the pinned FetchContent checkout; ``setup.py``
copies the same upstream files into this package for installed wheels.

Do not add model templates here.  Keeping the builder modules byte-for-byte
from the pinned XGrammar source is what lets KsanaLLM inherit XGrammar's model
support instead of maintaining a second implementation.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


_UPSTREAM_MODULES = (
    "structural_tag",
    "openai_tool_call_schema",
    "builtin_structural_tag",
)


def _find_upstream_package() -> Path:
    override = os.getenv("KSANA_XGRAMMAR_PYTHON_PATH")
    candidates = []
    if override:
        candidates.append(Path(override))

    repo_root = Path(__file__).resolve().parents[3]
    candidates.extend(
        (
            repo_root / "build/third_party/xgrammar/python/xgrammar",
            repo_root / "build/_deps/xgrammar-src/python/xgrammar",
        )
    )
    for candidate in candidates:
        if all((candidate / f"{name}.py").is_file() for name in _UPSTREAM_MODULES):
            return candidate
    raise ImportError(
        "XGrammar's pure-Python structural-tag modules are unavailable. "
        "Build KsanaLLM first, or set KSANA_XGRAMMAR_PYTHON_PATH to the "
        "xgrammar/python/xgrammar directory from the pinned v0.2.5 source."
    )


def _load_upstream_module(name: str, package_dir: Path) -> ModuleType:
    qualified_name = f"{__name__}.{name}"
    existing = sys.modules.get(qualified_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        qualified_name, package_dir / f"{name}.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load XGrammar module {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


try:
    from .builtin_structural_tag import (  # type: ignore[import-not-found]
        get_model_structural_tag,
        normalize_tool_choice,
    )
    from .structural_tag import StructuralTag  # type: ignore[import-not-found]
except ModuleNotFoundError:
    _package_dir = _find_upstream_package()
    for _module_name in _UPSTREAM_MODULES:
        _load_upstream_module(_module_name, _package_dir)
    from .builtin_structural_tag import get_model_structural_tag, normalize_tool_choice
    from .structural_tag import StructuralTag


__all__ = ["StructuralTag", "get_model_structural_tag", "normalize_tool_choice"]
