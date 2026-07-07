"""voice_os: six-axis voice scoring, dual-persona routing, and QA gates.

Public API (the foundation for the callable voice module):

    from voice_os import load_corpus, build_baseline, score_draft, run_pipeline

Everything runs deterministically offline; Claude is layered on when
credentials resolve (see voice_os.llm).
"""

from __future__ import annotations

from .axes import AXES, AxisProfile, score_text
from .calibration import calibrate
from .corpus import CorpusEntry, build_baseline, parse_corpus
from .personas import AdversarialPersona, GenerativePersona
from .qa import GateResult, find_banned, gate, load_banned_list

__version__ = "0.1.0"


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


def run_pipeline(
    corpus_path: str,
    draft_text: str,
    banned_path: str | None = None,
    channel: str = "email",
    audience: str = "peer",
    situation: str = "standard",
    max_cycles: int = 2,
) -> dict:
    """Full dual-persona pipeline with QA gate. Returns a JSON-safe dict.

    Flow: baseline -> register calibration -> score -> QA gate; on "cycle",
    the generative persona revises, the adversarial persona stress-tests,
    and the gate re-runs, up to max_cycles revisions.
    """
    from .axes import score_text as _score

    baseline = load_corpus(corpus_path)
    banned = load_banned_list(banned_path) if banned_path else []
    target = calibrate(baseline, channel, audience, situation)

    generative = GenerativePersona()
    adversarial = AdversarialPersona()

    text = draft_text
    cycles: list[dict] = []
    modes: set[str] = set()
    result: GateResult | None = None
    carried_findings: list[str] = []  # adversarial findings feed the next revision

    for cycle_number in range(max_cycles + 1):
        scores = _score(text)
        result = gate(scores, baseline, target, find_banned(text, banned))
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

        signals = result.revision_signals + [
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

    return {
        "meta": {
            "voice_os_version": __version__,
            "mode": "live" if "live" in modes else "offline",
            "canonical_axes": list(AXES),
        },
        "classification": {
            "channel": channel,
            "audience": audience,
            "situation": situation,
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
