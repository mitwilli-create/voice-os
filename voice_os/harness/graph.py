"""The LangGraph eval loop: select -> run_case (per case) -> aggregate
-> persist.

This is the ONLY module in the harness package that imports langgraph.
Each held-out case runs through the REAL product surface
(voice_os.product.draft), so the harness exercises the exact graph
callers use; scoring is the pure math in scoring.py. One checkpoint
per case: an interrupted eval resumes from the last scored case and
per-case progress is visible in harness_history().

Persistence mirrors the evolution module: this graph gets its OWN
database file (<var_dir>/harness.sqlite) and thread namespace
(eval-... run ids). Inner draft() runs receive var_dir=<var>/eval so
pipeline checkpoints and KB snapshots from eval runs stay fully
separated from product runs.

Privacy: checkpoints and full reports contain real held-out messages
and generated drafts (personal data); both live under the gitignored
var/. Summary files contain numbers and context labels only.

Run-scoped fields excluded from determinism comparisons: run id,
generated_at, report paths, inner run ids (docs/eval-harness.md).

Design: docs/eval-harness.md.
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

from .. import load_corpus
from ..product import draft as product_draft
from ..product.kb import REPO_ROOT
from . import scoring
from .cases import DEFAULT_CAP, DEFAULT_PER_CELL, select_cases

DEFAULT_CORPUS_PATH = os.path.join(REPO_ROOT, "corpus", "voice_corpus.txt")
DEFAULT_CHUNKS_DIR = os.path.join(REPO_ROOT, "corpus", "chunks")


class EvalState(TypedDict):
    # request (serializable; None = repo defaults)
    run_id: str
    corpus_path: str | None
    chunks_dir: str | None
    mined_dir: str | None
    banned_path: str | None
    kb_dir: str | None
    var_dir: str | None
    per_cell: int
    cap: int
    live: bool
    max_revisions: int

    # selection
    cases: list
    baseline_std: dict
    cursor: int

    # results
    results: Annotated[list, operator.add]
    summary: dict
    report_path: str | None
    summary_path: str | None

    # observability
    trace_notes: Annotated[list, operator.add]


def initial_eval_state(
    *,
    run_id: str,
    corpus_path: str | None = None,
    chunks_dir: str | None = None,
    mined_dir: str | None = None,
    banned_path: str | None = None,
    kb_dir: str | None = None,
    var_dir: str | None = None,
    per_cell: int = DEFAULT_PER_CELL,
    cap: int = DEFAULT_CAP,
    live: bool = False,
    max_revisions: int = 2,
) -> dict:
    return {
        "run_id": run_id,
        "corpus_path": corpus_path,
        "chunks_dir": chunks_dir,
        "mined_dir": mined_dir,
        "banned_path": banned_path,
        "kb_dir": kb_dir,
        "var_dir": var_dir,
        "per_cell": per_cell,
        "cap": cap,
        "live": live,
        "max_revisions": max_revisions,
        "cases": [],
        "baseline_std": {},
        "cursor": 0,
        "results": [],
        "summary": {},
        "report_path": None,
        "summary_path": None,
        "trace_notes": [],
    }


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"eval-{stamp}-{uuid.uuid4().hex[:8]}"


def var_dir_for(state_or_none: str | None) -> str:
    return (
        state_or_none
        or os.environ.get("VOICE_OS_VAR_DIR")
        or os.path.join(REPO_ROOT, "var")
    )


def harness_db_path(var_dir: str | None = None) -> str:
    return os.path.join(var_dir_for(var_dir), "harness.sqlite")


def reports_dir(var_dir: str | None = None) -> str:
    return os.path.join(var_dir_for(var_dir), "eval", "reports")


# --------------------------------------------------------------------------
# Nodes


def select(state: EvalState) -> dict:
    """Build the deterministic case list and load the baseline std."""
    chunks_dir = state.get("chunks_dir") or DEFAULT_CHUNKS_DIR
    corpus_path = state.get("corpus_path") or DEFAULT_CORPUS_PATH
    cases = select_cases(
        chunks_dir, per_cell=state["per_cell"], cap=state["cap"]
    )
    baseline = load_corpus(corpus_path)
    cells = sorted({f"{case['channel']}|{case['audience']}" for case in cases})
    return {
        "cases": cases,
        "baseline_std": dict(baseline.std),
        "cursor": 0,
        "trace_notes": [
            f"select: {len(cases)} cases across {len(cells)} cells "
            f"(per_cell={state['per_cell']}, cap={state['cap']})"
        ],
    }


def run_case(state: EvalState) -> dict:
    """Draft one case through the product pipeline and score it."""
    cursor = state["cursor"]
    case = state["cases"][cursor]
    eval_var = os.path.join(var_dir_for(state.get("var_dir")), "eval")
    envelope = product_draft(
        case["brief"],
        channel=case["channel"],
        audience=case["audience"],
        situation="standard",
        goal=case["goal"],
        medium=case["medium"],
        max_revisions=state["max_revisions"],
        run_id=f"{state['run_id']}-case-{case['id']}",
        corpus_path=state.get("corpus_path"),
        chunks_dir=state.get("chunks_dir"),
        mined_dir=state.get("mined_dir"),
        banned_path=state.get("banned_path"),
        kb_dir=state.get("kb_dir"),
        var_dir=eval_var,
    )
    record = scoring.score_case(
        case, envelope, state["baseline_std"], state["live"]
    )
    return {
        "results": [record],
        "cursor": cursor + 1,
        "trace_notes": [
            f"run_case: {cursor + 1}/{len(state['cases'])} "
            f"{case['channel']}|{case['audience']} "
            f"align_off={record['alignment_offline']} "
            f"decision={record['decision']}"
        ],
    }


def aggregate(state: EvalState) -> dict:
    """Fold per-case results into the numbers-only summary."""
    summary = scoring.summarize(state["results"])
    summary["params"] = {
        "per_cell": state["per_cell"],
        "cap": state["cap"],
        "max_revisions": state["max_revisions"],
    }
    overall = summary["overall"]
    return {
        "summary": summary,
        "trace_notes": [
            "aggregate: overall alignment_offline="
            f"{overall.get('alignment_offline')} "
            f"style={overall.get('style_overall')} over {summary['cases']} cases"
        ],
    }


def persist(state: EvalState) -> dict:
    """Write the full report (personal text) and the summary (numbers).

    Both land under the gitignored var/eval/reports/. The baseline is
    NEVER touched here; it moves only via the explicit gate command.
    """
    directory = reports_dir(state.get("var_dir"))
    os.makedirs(directory, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_id = state["run_id"]

    report_path = os.path.join(directory, f"{run_id}.json")
    summary_path = os.path.join(directory, f"{run_id}.summary.json")
    report = {
        "run_id": run_id,
        "generated_at": generated_at,
        "summary": state["summary"],
        "cases": state["results"],
    }
    summary = {
        "run_id": run_id,
        "generated_at": generated_at,
        **state["summary"],
    }
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, sort_keys=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1, sort_keys=True)
    except OSError as exc:
        return {
            "report_path": None,
            "summary_path": None,
            "trace_notes": [f"persist: report write failed ({exc})"],
        }
    return {
        "report_path": report_path,
        "summary_path": summary_path,
        "trace_notes": [
            f"persist: wrote {os.path.basename(report_path)} and summary"
        ],
    }


def _route(state: EvalState) -> str:
    return "next" if state["cursor"] < len(state["cases"]) else "done"


def build_graph() -> StateGraph:
    graph = StateGraph(EvalState)
    graph.add_node("select", select)
    graph.add_node("run_case", run_case)
    graph.add_node("aggregate", aggregate)
    graph.add_node("persist", persist)
    graph.add_edge(START, "select")
    graph.add_conditional_edges(
        "select", _route, {"next": "run_case", "done": "aggregate"}
    )
    graph.add_conditional_edges(
        "run_case", _route, {"next": "run_case", "done": "aggregate"}
    )
    graph.add_edge("aggregate", "persist")
    graph.add_edge("persist", END)
    return graph


@contextmanager
def checkpointed_graph(var_dir: str | None = None):
    db_path = harness_db_path(var_dir)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        yield build_graph().compile(checkpointer=saver)
    finally:
        conn.close()


def build_eval_result(state: dict, run_id: str) -> dict:
    """Public JSON-safe envelope for one eval run."""
    return {
        "run_id": run_id,
        "cases": len(state.get("results", [])),
        "summary": dict(state.get("summary", {})),
        "report_path": state.get("report_path"),
        "summary_path": state.get("summary_path"),
        "trace": list(state.get("trace_notes", [])),
    }


def harness_run(
    *,
    corpus_path: str | None = None,
    chunks_dir: str | None = None,
    mined_dir: str | None = None,
    banned_path: str | None = None,
    kb_dir: str | None = None,
    var_dir: str | None = None,
    per_cell: int = DEFAULT_PER_CELL,
    cap: int = DEFAULT_CAP,
    live: bool = False,
    max_revisions: int = 2,
    run_id: str | None = None,
) -> dict:
    """One checkpointed evaluation run through the graph.

    Offline runs (the default) force VOICE_OS_OFFLINE for the duration
    so the gate never depends on ambient credentials; live runs leave
    the environment alone and let credentials decide, exactly like
    draft().
    """
    run = run_id or new_run_id()
    initial = initial_eval_state(
        run_id=run,
        corpus_path=corpus_path,
        chunks_dir=chunks_dir,
        mined_dir=mined_dir,
        banned_path=banned_path,
        kb_dir=kb_dir,
        var_dir=var_dir,
        per_cell=per_cell,
        cap=cap,
        live=live,
        max_revisions=max_revisions,
    )
    config = {
        "configurable": {"thread_id": run},
        # One step per case plus selection/aggregation/persistence and
        # conditional-edge headroom.
        "recursion_limit": 2 * max(cap, 1) + 12,
    }
    previous_offline = os.environ.get("VOICE_OS_OFFLINE")
    if not live:
        os.environ["VOICE_OS_OFFLINE"] = "1"
    try:
        with checkpointed_graph(var_dir) as graph:
            final_state = graph.invoke(initial, config)
    finally:
        if not live:
            if previous_offline is None:
                os.environ.pop("VOICE_OS_OFFLINE", None)
            else:
                os.environ["VOICE_OS_OFFLINE"] = previous_offline
    return build_eval_result(final_state, run)


def harness_history(
    run_id: str | None = None, var_dir: str | None = None
) -> list[dict]:
    """Checkpoint summaries: one run's steps, or all runs' latest state."""
    db_path = harness_db_path(var_dir)
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
                        "cursor": values.get("cursor"),
                        "cases": len(values.get("cases") or []),
                        "scored": len(values.get("results") or []),
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
            overall = (values.get("summary") or {}).get("overall") or {}
            summaries[thread_id] = {
                "run_id": thread_id,
                "cases": len(values.get("results") or []),
                "alignment_offline": overall.get("alignment_offline"),
                "style_overall": overall.get("style_overall"),
                "summary_path": values.get("summary_path"),
            }
        return [summaries[key] for key in sorted(summaries)]
    finally:
        conn.close()


def describe_graph() -> str:
    """Mermaid text of the compiled eval graph."""
    return build_graph().compile().get_graph().draw_mermaid()
