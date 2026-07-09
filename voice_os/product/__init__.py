"""The callable product layer: voice_os.draft() and friends.

Other agents call these high-level functions without knowing LangGraph
exists. Importing this package needs only the stdlib; the langgraph
dependency is touched exactly when a graph-backed function is called,
and its absence surfaces as an ImportError with the install command.

    import voice_os

    result = voice_os.draft(
        text="...",
        channel="email",
        audience="boss",
        situation="high_stakes",
        goal="set_expectations",
    )

Design: docs/callable-layer.md.
"""

from __future__ import annotations

from ..contexts import VoiceContext
from .aliases import normalize_context
from .kb import list_kb_snapshots, load_kb, snapshot_kb
from .state import build_result, initial_state

__all__ = [
    "draft",
    "run_history",
    "describe_graph",
    "load_kb",
    "snapshot_kb",
    "list_kb_snapshots",
    "normalize_context",
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


def draft(
    text: str,
    *,
    channel: str = "email",
    audience: str = "peer",
    situation: str = "standard",
    goal: str = "unknown",
    stakes: str | None = None,
    medium: str | None = None,
    max_revisions: int = 2,
    redraft: bool = False,
    run_id: str | None = None,
    corpus_path: str | None = None,
    chunks_dir: str | None = None,
    mined_dir: str | None = None,
    banned_path: str | None = None,
    kb_dir: str | None = None,
    var_dir: str | None = None,
) -> dict:
    """Draft text in the target voice through the full graph pipeline.

    Friendly context forms are normalized (audience="boss" works), the
    canonical context is validated fail-fast, and the run is
    checkpointed under var/ for later inspection via run_history().

    redraft=True declares the input finished writing being re-voiced:
    every output sentence must be entailed by the input, and unentailed
    sentences block a pass (they force revision, then reject). With
    redraft=False (compose semantics) conservation is still measured
    and reported in the envelope, never blocking. Quote spans are
    inviolable in both modes.

    Returns a JSON-safe envelope: run_id, decision ("pass" or
    "reject"), output_text, fidelity, revisions, revision_history,
    banned_hits, conservation, mode, context, kb, trace.
    """
    if not text or not text.strip():
        raise ValueError("text must be a non-empty string")
    if max_revisions < 0:
        raise ValueError("max_revisions must be >= 0")

    ctx = normalize_context(
        channel=channel,
        audience=audience,
        situation=situation,
        goal=goal,
        stakes=stakes,
        medium=medium,
    )
    VoiceContext(**ctx).validate()  # fail fast before any graph work

    graph = _graph_module("draft")
    run = run_id or graph.new_run_id()
    initial = initial_state(
        input_text=text,
        max_revisions=max_revisions,
        redraft=redraft,
        corpus_path=corpus_path,
        chunks_dir=chunks_dir,
        mined_dir=mined_dir,
        banned_path=banned_path,
        kb_dir=kb_dir,
        var_dir=var_dir,
        **ctx,
    )
    final_state = graph.invoke(initial, run, var_dir=var_dir)
    return build_result(final_state, run)


def run_history(run_id: str, var_dir: str | None = None) -> list[dict]:
    """Checkpoint summaries for a prior draft() run, oldest first."""
    return _graph_module("run_history").run_history(run_id, var_dir=var_dir)


def describe_graph() -> str:
    """Mermaid text describing the compiled graph structure."""
    return _graph_module("describe_graph").describe_graph()
