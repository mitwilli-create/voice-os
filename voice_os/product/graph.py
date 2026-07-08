"""The LangGraph StateGraph for the callable layer.

This is the ONLY module in the package that imports langgraph. Node
functions are thin orchestration over the tested core: VoiceModel.query
resolves the context, GenerativePersona / AdversarialPersona do the
writing work, gate_extended makes the quality decision. The graph adds
the mission's generate -> critique -> qa_gate -> revise loop with
conditional routing, SqliteSaver checkpoints, and full observability.

Privacy: checkpoints contain draft text (personal data). The database
lives under the gitignored var/ directory (VOICE_OS_VAR_DIR override).

Design: docs/callable-layer.md.
"""

from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .. import llm
from ..axes import AxisProfile, score_text
from ..model import VoiceModel
from ..personas import AdversarialPersona, GenerativePersona
from ..qa import find_banned, gate_extended
from ..tone import ToneProfile, derive_metrics, tone_signals
from . import kb as kb_module
from .kb import REPO_ROOT
from .state import VoiceState

# Bounded LRU of loaded models keyed by the resolved load arguments, so
# repeated draft() calls in one process do not re-parse the corpus while
# a long-lived service cycling many path combinations cannot grow memory
# without limit.
_MODEL_CACHE_MAX = 4
_MODELS: OrderedDict[tuple, VoiceModel] = OrderedDict()

# Defaults resolve against the package's repo root, never the process
# CWD, so external callers can invoke draft() from anywhere (same fix
# family as the eval path hardening in PR #8).
_LOAD_DEFAULTS = {
    "corpus_path": os.path.join(REPO_ROOT, "corpus", "voice_corpus.txt"),
    "chunks_dir": os.path.join(REPO_ROOT, "corpus", "chunks"),
    "mined_dir": os.path.join(REPO_ROOT, "corpus", "mined"),
    "banned_path": os.path.join(REPO_ROOT, "data", "banned_list.txt"),
}


def _get_model(state: VoiceState) -> VoiceModel:
    key = tuple(
        state.get(name) or _LOAD_DEFAULTS[name]
        for name in ("corpus_path", "chunks_dir", "mined_dir", "banned_path")
    )
    if key in _MODELS:
        _MODELS.move_to_end(key)
        return _MODELS[key]
    corpus_path, chunks_dir, mined_dir, banned_path = key
    model = VoiceModel.load(
        corpus_path,
        chunks_dir=chunks_dir,
        mined_dir=mined_dir,
        banned_path=banned_path,
    )
    _MODELS[key] = model
    while len(_MODELS) > _MODEL_CACHE_MAX:
        _MODELS.popitem(last=False)
    return model


def _merge_modes(state: VoiceState, mode: str) -> list[str]:
    modes = set(state.get("persona_modes", []))
    modes.add(mode)
    return sorted(modes)


def _display_path(path: str) -> str:
    """Machine-neutral form of a filesystem path for the envelope.

    Absolute local paths are metadata leakage when envelopes or
    checkpoints leave the machine, so provenance records the
    repo-relative path for files under the repo and just the basename
    otherwise. sha256 + bytes remain the reproducibility identity.
    """
    absolute = os.path.abspath(path)
    root = os.path.join(REPO_ROOT, "")
    if absolute.startswith(root):
        return os.path.relpath(absolute, REPO_ROOT)
    return os.path.basename(absolute)


# Bounded memo of corpus content hashes keyed by (path, size, mtime_ns):
# the corpus is already read once per model load, so the provenance hash
# should not add a second full-file read on every draft() call. A stat
# change invalidates the entry; same sizing rationale as _MODELS.
_IDENTITY_CACHE_MAX = 4
_IDENTITY_CACHE: OrderedDict[tuple, dict] = OrderedDict()


def _corpus_identity(path: str) -> dict:
    """Content identity of the corpus file: path + sha256 + byte size.

    Content hash (not mtime) is the documented identity choice: mtime
    varies across clones and copies of byte-identical content, which
    would break reproducibility comparisons across machines
    (docs/determinism.md hardening item 2). The hash is memoized on
    the file's stat identity so repeated draft() calls do not re-read
    an unchanged corpus.
    """
    try:
        stat = os.stat(path)
    except OSError:
        # Model load already surfaced the real error, if any.
        return {"path": _display_path(path), "sha256": None, "bytes": None}
    key = (path, stat.st_size, stat.st_mtime_ns)
    if key in _IDENTITY_CACHE:
        _IDENTITY_CACHE.move_to_end(key)
        return dict(_IDENTITY_CACHE[key])

    digest = hashlib.sha256()
    total_bytes = 0
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                digest.update(block)
                total_bytes += len(block)
    except OSError:
        return {"path": _display_path(path), "sha256": None, "bytes": None}
    identity = {
        "path": _display_path(path),
        "sha256": digest.hexdigest(),
        "bytes": total_bytes,
    }
    _IDENTITY_CACHE[key] = dict(identity)
    while len(_IDENTITY_CACHE) > _IDENTITY_CACHE_MAX:
        _IDENTITY_CACHE.popitem(last=False)
    return identity


def _stamp_live_model(state: VoiceState, mode: str) -> dict:
    """Provenance update recording the engine that served a live call.

    Records the resolved model id (VOICE_OS_MODEL / DEFAULT_MODEL) the
    moment any persona reports mode "live", so checkpoints and the
    envelope are auditable to the exact engine
    (docs/determinism.md hardening item 3). Offline runs keep
    live_model None. Returns {} when there is nothing to record so
    callers can splat it into their partial state update.
    """
    if mode != "live":
        return {}
    provenance = dict(state.get("provenance", {}))
    provenance["live_model"] = llm.DEFAULT_MODEL
    return {"provenance": provenance}


# Per-exemplar word budget: chunk text is unbounded (document chunks run
# to 400 words), so state and prompt payloads are capped here rather than
# trusting every adapter's chunking policy.
_EXEMPLAR_MAX_WORDS = 120

# Calibrated per-cell gate thresholds stay inside this band: the floor
# keeps a sparse or skewed cell from disabling the gate, the ceiling is
# the hand default (docs/live-alignment.md).
_GATE_THRESHOLD_FLOOR = 0.65
_GATE_THRESHOLD_CEILING = 0.80
_GATE_MIN_CELL_N = 50


def _cell_threshold(
    gate_calibration: dict | None, channel: str, audience: str
) -> float | None:
    """The calibrated threshold for a (channel, audience) cell, or None
    when no trustworthy calibration exists (absent artifact, unknown
    cell, thin cell, or a malformed percentile)."""
    if not gate_calibration:
        return None
    cell = gate_calibration.get("cells", {}).get(f"{channel}|{audience}")
    if not cell or cell.get("n", 0) < _GATE_MIN_CELL_N:
        return None
    p40 = cell.get("p40")
    # json parses NaN/Infinity by default and NaN survives min/max, which
    # would make `fidelity >= threshold` unconditionally false; bools are
    # ints too and must not become thresholds.
    if isinstance(p40, bool) or not isinstance(p40, (int, float)):
        return None
    if not math.isfinite(p40):
        return None
    return round(
        min(max(float(p40), _GATE_THRESHOLD_FLOOR), _GATE_THRESHOLD_CEILING), 4
    )


def _bounded_exemplar(exemplar: dict) -> dict:
    """A JSON-safe copy with the text capped at _EXEMPLAR_MAX_WORDS."""
    bounded = dict(exemplar)
    words = str(bounded.get("text", "")).split()
    if len(words) > _EXEMPLAR_MAX_WORDS:
        bounded["text"] = " ".join(words[:_EXEMPLAR_MAX_WORDS])
        bounded["text_truncated"] = True
        bounded["text_words_original"] = len(words)
    return bounded


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{stamp}-{uuid.uuid4().hex[:8]}"


def var_dir_for(state_or_none: str | None) -> str:
    return state_or_none or os.environ.get("VOICE_OS_VAR_DIR") or "var"


def checkpoint_db_path(var_dir: str | None = None) -> str:
    return os.path.join(var_dir_for(var_dir), "runs.sqlite")


# --------------------------------------------------------------------------
# Nodes: each returns a partial state update of plain serializable values.


def prepare(state: VoiceState) -> dict:
    """Resolve the context to calibration data and record the KB."""
    model = _get_model(state)
    q = model.query(
        channel=state["channel"],
        audience=state["audience"],
        situation=state["situation"],
        goal=state["goal"],
        stakes=state["stakes"],
        medium=state["medium"],
    )

    bundle = kb_module.load_kb(state.get("kb_dir"))
    manifest = kb_module.ensure_snapshot(bundle, var_dir=state.get("var_dir"))
    if manifest:
        kb_meta = {
            "status": "ok",
            "snapshot_id": manifest["snapshot_id"],
            "bundle_hash": manifest["bundle_hash"],
            "files": [record["name"] for record in manifest["files"]],
        }
        kb_note = (
            f"prepare: kb snapshot {manifest['snapshot_id']} "
            f"({len(manifest['files'])} files)"
        )
    else:
        kb_meta = {"status": "absent", "kb_dir": bundle["kb_dir"]}
        kb_note = (
            f"prepare: kb not found at {bundle['kb_dir']}; drafting without KB"
        )
    if bundle.get("errors"):
        kb_meta["errors"] = bundle["errors"]

    gate_threshold = _cell_threshold(
        model.mined.gate_calibration, state["channel"], state["audience"]
    )

    notes = [
        f"prepare: context resolved (sources: {q.sources})",
        kb_note,
    ]
    if gate_threshold is not None:
        notes.append(
            f"prepare: calibrated gate threshold {gate_threshold:.4f} "
            f"for {state['channel']}|{state['audience']}"
        )
    drift_flags = q.meta.get("drift_flags") or []
    if drift_flags:
        notes.append(f"prepare: drift flags active: {drift_flags}")
    evolution_flags = q.meta.get("evolution_flags") or []
    if evolution_flags:
        notes.append(f"prepare: evolution flags active: {evolution_flags}")

    corpus_path = state.get("corpus_path") or _LOAD_DEFAULTS["corpus_path"]
    provenance = {
        "voice_os_version": q.meta.get("voice_os_version"),
        "artifacts": q.meta.get("artifacts") or {},
        "kb_bundle_hash": kb_meta.get("bundle_hash"),
        "corpus": _corpus_identity(corpus_path),
        "live_model": None,
    }

    return {
        "target_profile": dict(q.target_profile),
        "baseline_mean": dict(model.baseline.mean),
        "baseline_std": dict(model.baseline.std),
        "tone_mean": dict(q.tone.mean) if q.tone else None,
        "tone_std": dict(q.tone.std) if q.tone else None,
        "banned": list(q.banned),
        "guidance": list(q.guidance),
        # Top exemplars only: enough voice evidence for the live prompt
        # without dominating it (personal data; see state.py note).
        "exemplars": [_bounded_exemplar(e) for e in q.exemplars[:3]],
        "gate_threshold": gate_threshold,
        "kb_meta": kb_meta,
        "provenance": provenance,
        "current_draft": state["input_text"],
        "revision_count": 0,
        "trace_notes": notes,
    }


def _length_target(state: VoiceState) -> int | None:
    """The input's word count: the brief derives from the real message, so
    its length is the author's length for the situation."""
    return len(state["input_text"].split()) or None


def generate(state: VoiceState) -> dict:
    """First-pass voice transformation of the input text."""
    persona = GenerativePersona()
    result = persona.revise(
        state["input_text"],
        state["target_profile"],
        state["banned"],
        list(state["guidance"]),
        exemplars=state.get("exemplars") or None,
        length_target_words=_length_target(state),
    )
    return {
        "current_draft": result.text,
        "persona_modes": _merge_modes(state, result.mode),
        **_stamp_live_model(state, result.mode),
        "trace_notes": [
            f"generate: mode={result.mode}, notes={len(result.notes)}"
        ],
    }


def critique(state: VoiceState) -> dict:
    """Adversarial stress-test; findings feed the next revision."""
    persona = AdversarialPersona()
    result = persona.critique(
        state["current_draft"], state["target_profile"], state["banned"]
    )
    return {
        "critique_feedback": "\n".join(result.notes),
        "persona_modes": _merge_modes(state, result.mode),
        **_stamp_live_model(state, result.mode),
        "trace_notes": [
            f"critique: {len(result.notes)} findings (mode={result.mode})"
        ],
    }


def qa_gate(state: VoiceState) -> dict:
    """Score the draft and map the gate verdict to pass/revise/reject."""
    draft_text = state["current_draft"]
    scores = score_text(draft_text)
    baseline = AxisProfile(
        mean=state["baseline_mean"], std=state["baseline_std"]
    )
    tone_profile = (
        ToneProfile(mean=state["tone_mean"], std=state["tone_std"] or {})
        if state.get("tone_mean")
        else None
    )
    tone_observed = (
        derive_metrics(tone_signals(draft_text)) if tone_profile else None
    )
    gate_kwargs = {}
    if state.get("gate_threshold") is not None:
        gate_kwargs["threshold"] = state["gate_threshold"]
    result = gate_extended(
        scores,
        baseline,
        state["target_profile"],
        find_banned(draft_text, state["banned"]),
        tone_observed=tone_observed,
        tone_profile=tone_profile,
        **gate_kwargs,
    )

    if result.decision == "pass":
        decision = "pass"
    elif state["revision_count"] >= state["max_revisions"]:
        decision = "reject"
    else:
        decision = "revise"

    # Length inflation signal (node-level, advisory: it never changes the
    # gate decision). Live drafts measured 1.2x-3.1x the input length and
    # length_ratio correlated -0.60 with the judge's same_author score, so
    # overruns feed the next revision cycle as an explicit signal.
    revision_signals = list(result.revision_signals)
    input_words = len(state["input_text"].split())
    draft_words = len(draft_text.split())
    if input_words and draft_words > 1.4 * input_words:
        revision_signals.append(
            f"draft runs {draft_words} words against a {input_words}-word "
            f"input ({draft_words / input_words:.1f}x); cut it to about "
            f"{input_words} words"
        )

    return {
        "qa_decision": decision,
        "fidelity_scores": {
            "overall": result.fidelity,
            "per_axis": result.per_axis,
        },
        "banned_hits": list(result.banned_hits),
        "revision_signals": revision_signals,
        "trace_notes": [
            f"qa_gate: gate={result.decision} -> {decision} "
            f"(fidelity {result.fidelity:.3f}, revision {state['revision_count']})"
        ],
    }


def revise(state: VoiceState) -> dict:
    """Targeted revision from gate signals + carried adversarial findings."""
    findings = [
        line for line in state["critique_feedback"].splitlines() if line.strip()
    ]
    signals = (
        list(state["revision_signals"])
        + list(state["guidance"])
        + [f"adversarial finding (previous cycle): {f}" for f in findings]
    )
    persona = GenerativePersona()
    result = persona.revise(
        state["current_draft"], state["target_profile"], state["banned"],
        signals,
        exemplars=state.get("exemplars") or None,
        length_target_words=_length_target(state),
    )
    return {
        "current_draft": result.text,
        "revision_count": state["revision_count"] + 1,
        "revision_history": [state["current_draft"]],
        "persona_modes": _merge_modes(state, result.mode),
        **_stamp_live_model(state, result.mode),
        "trace_notes": [
            f"revise: revision {state['revision_count'] + 1} "
            f"(mode={result.mode}, {len(signals)} signals)"
        ],
    }


def _route(state: VoiceState) -> str:
    return state["qa_decision"]


def build_graph() -> StateGraph:
    """The uncompiled StateGraph; compile with or without a checkpointer."""
    graph = StateGraph(VoiceState)
    graph.add_node("prepare", prepare)
    graph.add_node("generate", generate)
    graph.add_node("critique", critique)
    graph.add_node("qa_gate", qa_gate)
    graph.add_node("revise", revise)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "generate")
    graph.add_edge("generate", "critique")
    graph.add_edge("critique", "qa_gate")
    graph.add_conditional_edges(
        "qa_gate", _route, {"revise": "revise", "pass": END, "reject": END}
    )
    graph.add_edge("revise", "critique")
    return graph


@contextmanager
def checkpointed_graph(var_dir: str | None = None):
    """Compiled graph with a SqliteSaver over var/runs.sqlite.

    The connection is opened per use and always closed; the database
    directory is created on demand.
    """
    db_path = checkpoint_db_path(var_dir)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        yield build_graph().compile(checkpointer=saver)
    finally:
        conn.close()


def invoke(initial: dict, run_id: str, var_dir: str | None = None) -> dict:
    """Run one draft through the checkpointed graph. Returns final state."""
    # Node executions per revision cycle is 3 (revise, critique, qa_gate);
    # size the recursion limit to the bounded loop plus setup headroom.
    limit = 3 * max(int(initial.get("max_revisions", 2)), 1) + 10
    config = {
        "configurable": {"thread_id": run_id},
        "recursion_limit": limit,
    }
    with checkpointed_graph(var_dir) as graph:
        return graph.invoke(initial, config)


def run_history(run_id: str, var_dir: str | None = None) -> list[dict]:
    """Compact checkpoint summaries for a run, oldest first."""
    db_path = checkpoint_db_path(var_dir)
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        steps = []
        for snapshot in saver.list({"configurable": {"thread_id": run_id}}):
            values = snapshot.checkpoint.get("channel_values", {})
            metadata = snapshot.metadata or {}
            # langgraph 1.x checkpoint metadata no longer names the node
            # that wrote; our own trace notes do (every node appends one
            # prefixed with its name), so derive the node from the state.
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
                    "qa_decision": values.get("qa_decision"),
                    "fidelity": (values.get("fidelity_scores") or {}).get(
                        "overall"
                    ),
                    "revision_count": values.get("revision_count"),
                }
            )
        steps.sort(key=lambda record: (record["step"] is None, record["step"]))
        return steps
    finally:
        conn.close()


def describe_graph() -> str:
    """Mermaid text of the compiled graph structure."""
    return build_graph().compile().get_graph().draw_mermaid()
