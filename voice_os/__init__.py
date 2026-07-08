"""voice_os: six-axis voice scoring, dual-persona routing, and QA gates.

Public API (the foundation for the callable voice module):

    from voice_os import load_corpus, build_baseline, score_draft, run_pipeline

Everything runs deterministically offline; Claude is layered on when
credentials resolve (see voice_os.llm).
"""

from __future__ import annotations

from .axes import AXES, AxisProfile, score_text
from .calibration import calibrate, calibrate_extended
from .contexts import GOALS, MEDIA, STAKES, VoiceContext, infer_stakes
from .corpus import CorpusEntry, build_baseline, parse_corpus
from .holdout import is_holdout
from .personas import AdversarialPersona, GenerativePersona
from .qa import (
    GateResult,
    find_banned,
    gate,
    gate_extended,
    load_banned_list,
    scrub_em_dashes,
)
from .tone import TONE_METRICS, ToneProfile, derive_metrics, tone_signals

__version__ = "0.2.0"


def load_corpus(path: str) -> AxisProfile:
    """Parse a corpus file and return its tier-weighted axis baseline."""
    return build_baseline(parse_corpus(path))


def score_draft(corpus_path: str, draft_text: str) -> dict:
    """Score a draft against a corpus baseline. Returns a JSON-safe dict."""
    baseline = load_corpus(corpus_path)
    scores = score_text(draft_text)
    fidelity, per_axis = baseline.fidelity(scores)
    return {
        "axis_scores": scores,
        "baseline": {"mean": baseline.mean, "std": baseline.std},
        "fidelity": fidelity,
        "per_axis_fidelity": per_axis,
    }


def run_cycles(
    baseline: AxisProfile,
    target: dict[str, float],
    draft_text: str,
    banned: list[str],
    max_cycles: int,
    extra_signals: list[str] | None = None,
    tone_profile: ToneProfile | None = None,
) -> tuple[list[dict], GateResult, str, set[str]]:
    """The dual-persona revision loop shared by run_pipeline and VoiceModel.

    Returns (cycle records, final gate result, final text, persona modes).
    A tone profile adds advisory deviation signals via gate_extended; with
    tone_profile=None the gate behavior is identical to gate().
    """
    extra_signals = extra_signals or []
    generative = GenerativePersona()
    adversarial = AdversarialPersona()

    # Boundary scrub: an already-in-voice draft can pass the gate at
    # cycle 0 without any persona running, so the outward em-dash ban
    # must hold on the input itself, not only on persona outputs.
    text = scrub_em_dashes(draft_text)
    cycles: list[dict] = []
    modes: set[str] = set()
    result: GateResult | None = None
    carried_findings: list[str] = []  # adversarial findings feed the next revision

    for cycle_number in range(max_cycles + 1):
        scores = score_text(text)
        # gate_extended without tone arguments returns exactly gate(); the
        # pipeline and the VoiceModel facade share this one gate path.
        tone_observed = (
            derive_metrics(tone_signals(text)) if tone_profile is not None else None
        )
        result = gate_extended(
            scores, baseline, target, find_banned(text, banned),
            tone_observed=tone_observed, tone_profile=tone_profile,
        )
        record: dict = {
            "cycle": cycle_number,
            "axis_scores": scores,
            "fidelity": result.fidelity,
            "per_axis_fidelity": result.per_axis,
            "banned_hits": result.banned_hits,
            "decision": result.decision,
            "revision_signals": result.revision_signals,
        }

        if result.decision == "pass" or cycle_number == max_cycles:
            cycles.append(record)
            break

        signals = result.revision_signals + extra_signals + [
            f"adversarial finding (previous cycle): {f}" for f in carried_findings
        ]
        revision = generative.revise(text, target, banned, signals)
        critique = adversarial.critique(revision.text, target, banned)
        modes.update({revision.mode, critique.mode})
        carried_findings = critique.notes
        record["generative_notes"] = revision.notes
        record["adversarial_findings"] = critique.notes
        cycles.append(record)
        text = revision.text

    # Return-boundary scrub: with the entry scrub and persona-output
    # scrubs this is a no-op today (scrub_em_dashes is idempotent), but
    # it makes "returned text never contains an em dash" a local
    # invariant of this function rather than a property every mutation
    # path must individually preserve.
    return cycles, result, scrub_em_dashes(text), modes


def run_pipeline(
    corpus_path: str,
    draft_text: str,
    banned_path: str | None = None,
    channel: str = "email",
    audience: str = "peer",
    situation: str = "standard",
    max_cycles: int = 2,
    *,
    goal: str = "unknown",
    stakes: str = "routine",
    medium: str | None = None,
) -> dict:
    """Full dual-persona pipeline with QA gate. Returns a JSON-safe dict.

    Flow: baseline -> register calibration -> score -> QA gate; on "cycle",
    the generative persona revises, the adversarial persona stress-tests,
    and the gate re-runs, up to max_cycles revisions.

    The keyword-only goal/stakes/medium arguments engage the extended
    calibration stack (docs/extended-model.md). With their defaults the
    output is identical to the pre-extension pipeline.
    """
    baseline = load_corpus(corpus_path)
    banned = load_banned_list(banned_path) if banned_path else []

    extended = not (goal == "unknown" and stakes == "routine" and medium is None)
    extra_signals: list[str] = []
    if extended:
        ctx = VoiceContext(
            channel=channel, audience=audience, situation=situation,
            goal=goal, stakes=stakes, medium=medium,
        )
        target, _sources = calibrate_extended(baseline, ctx)
        if goal != "unknown":
            extra_signals.append(f"communication goal: {goal}")
        if stakes != "routine":
            extra_signals.append(f"stakes level: {stakes}")
    else:
        target = calibrate(baseline, channel, audience, situation)

    cycles, result, text, modes = run_cycles(
        baseline, target, draft_text, banned, max_cycles,
        extra_signals=extra_signals,
    )

    live = "live" in modes
    if live:
        from .llm import DEFAULT_MODEL

    return {
        "meta": {
            "voice_os_version": __version__,
            "mode": "live" if live else "offline",
            "canonical_axes": list(AXES),
            # Engine stamp appears only on live runs, so the offline
            # default output stays byte-identical to the golden lock
            # (docs/determinism.md hardening item 3).
            **({"model": DEFAULT_MODEL} if live else {}),
        },
        "classification": {
            "channel": channel,
            "audience": audience,
            "situation": situation,
            # Extended keys appear only when the extended stack is engaged,
            # keeping default-argument output identical to the pre-extension
            # pipeline.
            **({"goal": goal, "stakes": stakes, "medium": medium} if extended else {}),
        },
        "baseline": {"mean": baseline.mean, "std": baseline.std},
        "target_profile": target,
        "cycles": cycles,
        "final": {
            "decision": result.decision,
            "fidelity": result.fidelity,
            "output_text": text,
        },
    }


# Imported last: model.py uses run_cycles and load_corpus defined above.
from .model import QueryResult, VoiceModel  # noqa: E402

# The callable product layer (voice_os.draft and friends) and the
# evolution module are exposed lazily via PEP 562 so `import voice_os`
# stays stdlib-only. The langgraph dependency is only touched when a
# graph-backed function is actually called (see
# voice_os/product/__init__.py and voice_os/evolution/__init__.py).
_PRODUCT_EXPORTS = frozenset(
    {
        "draft",
        "run_history",
        "describe_graph",
        "load_kb",
        "snapshot_kb",
        "list_kb_snapshots",
    }
)
_EVOLUTION_EXPORTS = frozenset(
    {
        "evolution_timeline",
        "voice_insights",
        "check_drift",
        "drift_run",
        "drift_run_history",
        "describe_drift_graph",
    }
)
_HARNESS_EXPORTS = frozenset(
    {
        "harness_run",
        "harness_gate",
        "harness_history",
        "describe_harness_graph",
    }
)


def __getattr__(name: str):
    if name in _PRODUCT_EXPORTS:
        from . import product

        return getattr(product, name)
    if name in _EVOLUTION_EXPORTS:
        from . import evolution

        return getattr(evolution, name)
    if name in _HARNESS_EXPORTS:
        from . import harness

        return getattr(harness, name)
    raise AttributeError(f"module 'voice_os' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(
        set(globals())
        | _PRODUCT_EXPORTS
        | _EVOLUTION_EXPORTS
        | _HARNESS_EXPORTS
    )
