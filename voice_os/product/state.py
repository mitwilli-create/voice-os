"""VoiceState: the serializable state the callable-layer graph runs on.

Everything here is stdlib-only and JSON/msgpack-serializable (str, int,
float, bool, list, dict, None) so SqliteSaver can checkpoint every step.
No dataclasses or model handles enter the state; the prepare node copies
the serializable parts of a QueryResult in as plain dicts and lists.

Design: docs/callable-layer.md.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict


class VoiceState(TypedDict):
    # request (set once at invoke; all post-normalization canonical values)
    input_text: str
    channel: str
    audience: str
    situation: str
    goal: str
    stakes: str          # always canonical here; None never reaches state
    medium: str | None
    max_revisions: int

    # model/KB resolution inputs (serializable path strings; None = default)
    corpus_path: str | None
    chunks_dir: str | None
    mined_dir: str | None
    banned_path: str | None
    kb_dir: str | None
    var_dir: str | None

    # calibration (seeded by the prepare node; primitives only)
    target_profile: dict
    baseline_mean: dict
    baseline_std: dict
    tone_mean: dict | None
    tone_std: dict | None
    banned: list[str]
    guidance: list[str]
    kb_meta: dict

    # working
    current_draft: str
    critique_feedback: str       # newline-joined adversarial findings
    qa_decision: Literal["pass", "revise", "reject"]
    revision_count: int
    revision_signals: list[str]  # latest gate signals, consumed by revise
    banned_hits: list[str]
    persona_modes: list[str]     # "live" / "offline" observed across nodes

    # observability (append-only reducers)
    revision_history: Annotated[list[str], operator.add]
    fidelity_scores: dict        # {"overall": float, "per_axis": {...}}
    trace_notes: Annotated[list[str], operator.add]


def initial_state(
    *,
    input_text: str,
    channel: str,
    audience: str,
    situation: str,
    goal: str,
    stakes: str,
    medium: str | None,
    max_revisions: int,
    corpus_path: str | None = None,
    chunks_dir: str | None = None,
    mined_dir: str | None = None,
    banned_path: str | None = None,
    kb_dir: str | None = None,
    var_dir: str | None = None,
) -> dict:
    """The full state a draft() run starts from. Every key present so
    checkpoints are self-describing from step zero."""
    return {
        "input_text": input_text,
        "channel": channel,
        "audience": audience,
        "situation": situation,
        "goal": goal,
        "stakes": stakes,
        "medium": medium,
        "max_revisions": max_revisions,
        "corpus_path": corpus_path,
        "chunks_dir": chunks_dir,
        "mined_dir": mined_dir,
        "banned_path": banned_path,
        "kb_dir": kb_dir,
        "var_dir": var_dir,
        "target_profile": {},
        "baseline_mean": {},
        "baseline_std": {},
        "tone_mean": None,
        "tone_std": None,
        "banned": [],
        "guidance": [],
        "kb_meta": {},
        "current_draft": "",
        "critique_feedback": "",
        "qa_decision": "revise",
        "revision_count": 0,
        "revision_signals": [],
        "banned_hits": [],
        "persona_modes": [],
        "revision_history": [],
        "fidelity_scores": {},
        "trace_notes": [],
    }


def build_result(state: dict, run_id: str) -> dict:
    """Project a finished graph state into the public JSON-safe envelope."""
    modes = state.get("persona_modes", [])
    return {
        "run_id": run_id,
        "decision": state.get("qa_decision", "reject"),
        "output_text": state.get("current_draft", ""),
        "fidelity": state.get("fidelity_scores", {}),
        "revisions": state.get("revision_count", 0),
        "revision_history": list(state.get("revision_history", [])),
        "banned_hits": list(state.get("banned_hits", [])),
        "mode": "live" if "live" in modes else "offline",
        "context": {
            "channel": state.get("channel"),
            "audience": state.get("audience"),
            "situation": state.get("situation"),
            "goal": state.get("goal"),
            "stakes": state.get("stakes"),
            "medium": state.get("medium"),
        },
        "kb": state.get("kb_meta", {}),
        "trace": list(state.get("trace_notes", [])),
    }
