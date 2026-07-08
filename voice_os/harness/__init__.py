"""Generation evaluation harness: held-out cases, scoring, regression gate.

Importing this package needs only the stdlib. The langgraph dependency
is touched exactly when a graph-backed function (harness_run,
harness_history, describe_harness_graph) is called, mirroring the
product and evolution layers.

    import voice_os

    voice_os.harness_run()                  # checkpointed eval run
    voice_os.harness_gate()                 # compare vs the stored baseline
    voice_os.harness_history()
    voice_os.describe_harness_graph()

Design: docs/eval-harness.md.
"""

from __future__ import annotations

import os

from .cases import build_brief, select_cases

# The comparison function lives in the gate submodule; it is exposed
# here only through harness_gate() so the package attribute "gate"
# stays the SUBMODULE (importing the bare function would shadow it).
from .gate import gate as _compare_summaries
from .gate import (
    DEFAULT_TOLERANCES,
    MIN_GATE_N,
    baseline_path,
    load_summary,
    write_baseline,
)
from .scoring import score_case, summarize

__all__ = [
    "select_cases",
    "build_brief",
    "score_case",
    "summarize",
    "harness_run",
    "harness_gate",
    "harness_history",
    "describe_harness_graph",
    "baseline_path",
    "load_summary",
    "write_baseline",
    "DEFAULT_TOLERANCES",
    "MIN_GATE_N",
]

_INSTALL_HINT = (
    "voice_os.{name}() requires the optional product-layer dependencies: "
    "pip install langgraph langgraph-checkpoint-sqlite"
)


def _graph_module(name: str):
    try:
        from . import graph
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] in ("langgraph", "langchain_core"):
            raise ImportError(_INSTALL_HINT.format(name=name)) from exc
        raise
    return graph


def harness_run(*args, **kwargs) -> dict:
    """One checkpointed evaluation run (see graph.harness_run)."""
    return _graph_module("harness_run").harness_run(*args, **kwargs)


def harness_gate(
    summary: dict | str | None = None,
    *,
    baseline: str | None = None,
    var_dir: str | None = None,
    tolerances: dict | None = None,
) -> dict:
    """Compare a summary (dict, path, or None = fresh offline run)
    against the stored baseline. Pure except for the optional fresh
    run; raises FileNotFoundError when no baseline exists yet."""
    if summary is None:
        summary = harness_run(var_dir=var_dir)["summary"]
    elif isinstance(summary, str):
        summary = load_summary(summary)
    path = baseline or baseline_path(var_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no eval baseline at {path}; establish one with "
            "python3 -m voice_os.harness gate --update-baseline"
        )
    result = _compare_summaries(summary, load_summary(path), tolerances)
    result["baseline_path"] = path
    return result


def harness_history(*args, **kwargs) -> list[dict]:
    """Prior eval runs from the harness checkpoint database."""
    return _graph_module("harness_history").harness_history(*args, **kwargs)


def describe_harness_graph() -> str:
    """Mermaid text describing the compiled eval graph."""
    return _graph_module("describe_harness_graph").describe_graph()
