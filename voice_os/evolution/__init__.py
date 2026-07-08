"""Voice evolution tracking: timelines, pattern drift, and insights.

Importing this package needs only the stdlib. The langgraph dependency
is touched exactly when a graph-backed function (drift_run,
drift_run_history, describe_drift_graph) is called, mirroring the
callable product layer.

    import voice_os

    voice_os.evolution_timeline(group_by="tier")
    voice_os.check_drift()                # pure, no persistence needed
    voice_os.drift_run()                  # checkpointed graph run
    voice_os.drift_run_history()
    voice_os.voice_insights()

Design: docs/evolution.md.
"""

from __future__ import annotations

from .baselines import (
    content_hash,
    ensure_baseline,
    latest_baseline,
    list_baselines,
)
from .insights import generate_insights
from .patterns import DEFAULT_MARKERS, diff_profiles, extract_pattern_profile
from .timeline import evolution_timeline, tier1_texts

__all__ = [
    "evolution_timeline",
    "extract_pattern_profile",
    "diff_profiles",
    "generate_insights",
    "voice_insights",
    "check_drift",
    "drift_run",
    "drift_run_history",
    "describe_drift_graph",
    "list_baselines",
    "latest_baseline",
    "ensure_baseline",
    "content_hash",
    "tier1_texts",
    "DEFAULT_MARKERS",
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


def voice_insights(chunks_dir: str | None = None, *, top_k: int = 10) -> list[dict]:
    """Ranked deterministic insights across audience, medium, goal, era."""
    return generate_insights(chunks_dir, top_k=top_k)


def check_drift(
    chunks_dir: str | None = None, var_dir: str | None = None
) -> dict:
    """Pure one-shot drift check: current Tier 1 patterns vs the latest
    stored baseline. No graph, no checkpoint, no artifact write; usable
    without langgraph installed.

    Returns {"status": "no-baseline", "profile": ...} when nothing is
    stored yet, else {"status": "ok", "baseline": manifest, "diff": ...}.
    """
    profile = extract_pattern_profile(tier1_texts(chunks_dir))
    existing = latest_baseline(var_dir)
    if existing is None:
        return {"status": "no-baseline", "profile": profile}
    manifest, body = existing
    return {
        "status": "ok",
        "baseline": manifest,
        "diff": diff_profiles(body.get("profile", {}), profile),
        "profile": profile,
    }


def drift_run(*args, **kwargs) -> dict:
    """One checkpointed drift run (see graph.drift_run)."""
    return _graph_module("drift_run").drift_run(*args, **kwargs)


def drift_run_history(*args, **kwargs) -> list[dict]:
    """Prior drift runs from the evolution checkpoint database."""
    return _graph_module("drift_run_history").drift_run_history(*args, **kwargs)


def describe_drift_graph() -> str:
    """Mermaid text describing the compiled drift graph."""
    return _graph_module("describe_drift_graph").describe_graph()
