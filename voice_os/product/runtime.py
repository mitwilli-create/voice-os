"""Runtime readiness report for graph-backed Voice OS commands.

The product layer is optional, so external callers need a cheap way to
verify an interpreter before they send personal prose to the draft command.
This module stays stdlib-only and returns JSON-safe data.
"""

from __future__ import annotations

from importlib import import_module
import sys
from typing import TypedDict


class DependencyReadiness(TypedDict):
    ready: bool


class DependencyStatus(DependencyReadiness, total=False):
    error: str


class RuntimeStatus(TypedDict):
    ready: bool
    python: str
    python_version: str
    python_supported: bool
    dependencies: dict[str, DependencyStatus]
    install_hint: str


def _probe(module: str, required_symbols: tuple[str, ...]) -> DependencyStatus:
    try:
        imported = import_module(module)
    except Exception as exc:
        return {
            "ready": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    missing = [name for name in required_symbols if not hasattr(imported, name)]
    if missing:
        return {
            "ready": False,
            "error": f"missing runtime symbols: {', '.join(missing)}",
        }
    return {"ready": True}


def dependency_status() -> RuntimeStatus:
    """Return the complete product-runtime capability report."""
    dependencies = {
        "langgraph": _probe(
            "langgraph.graph",
            ("END", "START", "StateGraph"),
        ),
        "langgraph-checkpoint-sqlite": _probe(
            "langgraph.checkpoint.sqlite",
            ("SqliteSaver",),
        ),
    }
    python_supported = sys.version_info >= (3, 10)
    return {
        "ready": python_supported and all(
            dependency["ready"] for dependency in dependencies.values()
        ),
        "python": sys.executable,
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "python_supported": python_supported,
        "dependencies": dependencies,
        "install_hint": (
            f"{sys.executable} -m pip install "
            "langgraph langgraph-checkpoint-sqlite"
        ),
    }
