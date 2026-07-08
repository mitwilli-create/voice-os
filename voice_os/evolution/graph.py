"""The LangGraph drift-run graph: extract -> compare -> flag -> record.

This is the ONLY module in the evolution package that imports
langgraph. The pure math lives in patterns.py / insights.py; this
graph adds checkpointed, reviewable runs so pattern drift can be
scheduled and prior runs compared.

Persistence mirrors the callable layer but in its OWN database file
(<var_dir>/evolution.sqlite) and thread namespace (drift-... run ids),
so product runs and drift runs never share a checkpoint space.

Privacy: checkpoints and the evolution_flags artifact contain pattern
distributions derived from private text. The database lives under the
gitignored var/; the artifact lives under the gitignored corpus/mined/.

No personas, no LLM calls: a drift run is offline-deterministic by
construction. Run-scoped fields: run id, baseline id, generated_at.

Design: docs/evolution.md.
"""

from __future__ import annotations

import json
import operator
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Annotated, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from . import baselines as baseline_store
from .insights import era_insights
from .patterns import DEFAULT_MARKERS, diff_profiles, extract_pattern_profile
from .timeline import REPO_ROOT, evolution_timeline, tier1_texts

DEFAULT_MINED_DIR = os.path.join(REPO_ROOT, "corpus", "mined")
FLAGS_FILENAME = "evolution_flags.json"
ARTIFACT_VERSION = "1.0"


class DriftState(TypedDict):
    # request (serializable path strings; None = repo defaults)
    chunks_dir: str | None
    var_dir: str | None
    mined_dir: str | None
    update_baseline: bool

    # extraction
    current_profile: dict
    windows: list

    # comparison
    baseline: dict          # manifest of the baseline diffed against
    baseline_created: bool  # True when this run established the baseline
    diff: dict

    # flagging
    flags: list
    insights: list
    artifact_path: str | None

    # observability
    trace_notes: Annotated[list, operator.add]


def initial_drift_state(
    *,
    chunks_dir: str | None = None,
    var_dir: str | None = None,
    mined_dir: str | None = None,
    update_baseline: bool = False,
) -> dict:
    return {
        "chunks_dir": chunks_dir,
        "var_dir": var_dir,
        "mined_dir": mined_dir,
        "update_baseline": update_baseline,
        "current_profile": {},
        "windows": [],
        "baseline": {},
        "baseline_created": False,
        "diff": {},
        "flags": [],
        "insights": [],
        "artifact_path": None,
        "trace_notes": [],
    }


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"drift-{stamp}-{uuid.uuid4().hex[:8]}"


def var_dir_for(state_or_none: str | None) -> str:
    return (
        state_or_none
        or os.environ.get("VOICE_OS_VAR_DIR")
        or os.path.join(REPO_ROOT, "var")
    )


def drift_db_path(var_dir: str | None = None) -> str:
    return os.path.join(var_dir_for(var_dir), "evolution.sqlite")


# --------------------------------------------------------------------------
# Nodes


def extract(state: DriftState) -> dict:
    """Pattern profile of the current Tier 1 voice + windowed series."""
    texts = tier1_texts(state.get("chunks_dir"))
    profile = extract_pattern_profile(texts, DEFAULT_MARKERS)
    windows = evolution_timeline(state.get("chunks_dir"), group_by="window")
    return {
        "current_profile": profile,
        "windows": windows,
        "trace_notes": [
            f"extract: {profile['n_chunks']} tier-1 chunks, "
            f"{len(windows)} dated windows"
        ],
    }


def compare(state: DriftState) -> dict:
    """Diff the current profile against the latest STORED baseline.

    The baseline anchor never moves automatically: drift accumulates
    against the stored baseline until update_baseline is passed
    explicitly (or the first run establishes one).
    """
    var_dir = state.get("var_dir")
    body = {"profile": state["current_profile"], "windows": state["windows"]}
    existing = baseline_store.latest_baseline(var_dir)
    if existing is None:
        manifest, _ = baseline_store.ensure_baseline(body, var_dir=var_dir)
        return {
            "baseline": manifest,
            "baseline_created": True,
            "diff": {},
            "trace_notes": [
                f"compare: no stored baseline; established "
                f"{manifest['baseline_id']} "
                f"(hash {manifest['content_hash'][:12]})"
            ],
        }

    manifest, baseline_body = existing
    diff = diff_profiles(
        baseline_body.get("profile", {}), state["current_profile"]
    )
    notes = [
        f"compare: diffed against baseline {manifest['baseline_id']} "
        f"({len(diff['emerging'])} emerging, {len(diff['fading'])} fading, "
        f"{len(diff['shifted'])} shifted)"
    ]
    updated = False
    if state.get("update_baseline"):
        manifest, updated = baseline_store.ensure_baseline(
            body, var_dir=var_dir
        )
        notes.append(
            "compare: baseline "
            + (
                f"updated to {manifest['baseline_id']}"
                if updated
                else f"unchanged (content matches {manifest['baseline_id']})"
            )
        )
    return {
        "baseline": manifest,
        "baseline_created": updated,
        "diff": diff,
        "trace_notes": notes,
    }


def flag(state: DriftState) -> dict:
    """Emerging/fading flags + era insights; write the mined artifact."""
    diff = state.get("diff") or {}
    flags = list(diff.get("flags", []))
    insights = era_insights(state.get("windows", []))

    mined_dir = state.get("mined_dir") or DEFAULT_MINED_DIR
    artifact_path = None
    try:
        os.makedirs(mined_dir, exist_ok=True)
        artifact_path = os.path.join(mined_dir, FLAGS_FILENAME)
        artifact = {
            "artifact": "evolution_flags",
            "version": ARTIFACT_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "miner": "evolution.graph@1.0",
            "params": {
                "baseline_id": state.get("baseline", {}).get("baseline_id"),
                "baseline_hash": state.get("baseline", {}).get("content_hash"),
            },
            "data": {
                "flags": flags,
                "emerging": diff.get("emerging", []),
                "fading": diff.get("fading", []),
                "shifted": diff.get("shifted", []),
                "sentence_shift": diff.get("sentence_shift"),
                # The current Tier 1 profile itself, so generation can
                # fuse it (docs/pattern-fusion.md). Lexicon-bucketed
                # forms and numbers only; names never enter the profile
                # (see patterns.py).
                "pattern_profile": dict(state.get("current_profile") or {}),
            },
        }
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2, sort_keys=True)
    except OSError as exc:
        artifact_path = None
        return {
            "flags": flags,
            "insights": insights,
            "artifact_path": None,
            "trace_notes": [f"flag: artifact write failed ({exc})"],
        }
    return {
        "flags": flags,
        "insights": insights,
        "artifact_path": artifact_path,
        "trace_notes": [
            f"flag: {len(flags)} flags, {len(insights)} era insights, "
            f"artifact {FLAGS_FILENAME} written"
        ],
    }


def record(state: DriftState) -> dict:
    """Final summary note; the envelope is projected from this state."""
    verdict = (
        "baseline established"
        if state.get("baseline_created") and not state.get("diff")
        else (f"{len(state.get('flags', []))} pattern flags")
    )
    return {"trace_notes": [f"record: drift run complete ({verdict})"]}


def build_graph() -> StateGraph:
    graph = StateGraph(DriftState)
    graph.add_node("extract", extract)
    graph.add_node("compare", compare)
    graph.add_node("flag", flag)
    graph.add_node("record", record)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "compare")
    graph.add_edge("compare", "flag")
    graph.add_edge("flag", "record")
    graph.add_edge("record", END)
    return graph


@contextmanager
def checkpointed_graph(var_dir: str | None = None):
    db_path = drift_db_path(var_dir)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        yield build_graph().compile(checkpointer=saver)
    finally:
        conn.close()


def build_drift_result(state: dict, run_id: str) -> dict:
    """Public JSON-safe envelope for one drift run."""
    profile = state.get("current_profile", {})
    return {
        "run_id": run_id,
        "baseline": {
            "baseline_id": state.get("baseline", {}).get("baseline_id"),
            "content_hash": state.get("baseline", {}).get("content_hash"),
            "established_this_run": state.get("baseline_created", False),
        },
        "diff": state.get("diff", {}),
        "flags": list(state.get("flags", [])),
        "insights": list(state.get("insights", [])),
        "profile_summary": {
            "n_chunks": profile.get("n_chunks", 0),
            "n_words": profile.get("n_words", 0),
        },
        "artifact_path": state.get("artifact_path"),
        "trace": list(state.get("trace_notes", [])),
    }


def drift_run(
    chunks_dir: str | None = None,
    var_dir: str | None = None,
    mined_dir: str | None = None,
    *,
    update_baseline: bool = False,
    run_id: str | None = None,
) -> dict:
    """One checkpointed drift run through the graph."""
    run = run_id or new_run_id()
    initial = initial_drift_state(
        chunks_dir=chunks_dir,
        var_dir=var_dir,
        mined_dir=mined_dir,
        update_baseline=update_baseline,
    )
    config = {"configurable": {"thread_id": run}}
    with checkpointed_graph(var_dir) as graph:
        final_state = graph.invoke(initial, config)
    return build_drift_result(final_state, run)


def drift_run_history(
    run_id: str | None = None, var_dir: str | None = None
) -> list[dict]:
    """Checkpoint summaries: one run's steps, or all runs' latest state.

    With run_id: that run's steps oldest first (like product
    run_history). Without: one latest-state summary per recorded run,
    oldest run id first, for run-over-run comparison.
    """
    db_path = drift_db_path(var_dir)
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        if run_id is not None:
            steps = []
            for snapshot in saver.list({"configurable": {"thread_id": run_id}}):
                values = snapshot.checkpoint.get("channel_values", {})
                metadata = snapshot.metadata or {}
                notes = values.get("trace_notes") or []
                node = (
                    notes[-1].split(":", 1)[0]
                    if notes
                    else metadata.get("source")
                )
                steps.append(
                    {
                        "step": metadata.get("step"),
                        "node": node,
                        "n_flags": len(values.get("flags") or []),
                        "baseline_id": (values.get("baseline") or {}).get(
                            "baseline_id"
                        ),
                    }
                )
            steps.sort(
                key=lambda record: (record["step"] is None, record["step"])
            )
            return steps

        summaries = {}
        for snapshot in saver.list(None):
            thread_id = (snapshot.config or {}).get("configurable", {}).get(
                "thread_id"
            )
            if not thread_id or thread_id in summaries:
                # saver.list yields newest first per thread, so the
                # first snapshot seen for a thread is its final state.
                continue
            values = snapshot.checkpoint.get("channel_values", {})
            summaries[thread_id] = {
                "run_id": thread_id,
                "n_flags": len(values.get("flags") or []),
                "baseline_id": (values.get("baseline") or {}).get(
                    "baseline_id"
                ),
                "baseline_established": values.get("baseline_created", False),
                "n_chunks": (values.get("current_profile") or {}).get(
                    "n_chunks", 0
                ),
            }
        return [summaries[key] for key in sorted(summaries)]
    finally:
        conn.close()


def describe_graph() -> str:
    """Mermaid text of the compiled drift graph."""
    return build_graph().compile().get_graph().draw_mermaid()
